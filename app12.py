import os
import re
import unicodedata
import json
import hashlib
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import pandas as pd
import altair as alt
import networkx as nx
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from pyvis.network import Network

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate


# ============================================================
# 1. CONFIGURAÇÃO INICIAL
# ============================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError(
        "A variável OPENAI_API_KEY não foi encontrada. "
        "Crie um arquivo .env na mesma pasta do script com:\n"
        "OPENAI_API_KEY=sua_chave_aqui"
    )

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

DATA_DIR = Path(
    os.getenv(
        "DATA_DIR",
        r"G:\Meu Drive\Doutorado\arbovirus_rag\data_processed\data_rag_gemini"
    )
)

PERSIST_DIRECTORY = Path(
    os.getenv(
        "PERSIST_DIRECTORY",
        r"G:\Meu Drive\Doutorado\arbovirus_rag\chromadb_data_v2"
    )
)

MANIFEST_PATH = PERSIST_DIRECTORY / "_manifest.json"

ARQUIVOS_JSON = [
    'sorotipos_by_state_and_week_rag_documents.json',
    'sorotipos_by_state_rag_documents.json',
    'sorotipos_por_uf_com_codigos_e_nomes_rag_documents.json',
    'symptoms_by_state_rag_documents.json',
    'total_casos_uf_ano_rag_documents.json',
    'hospitalizacao_uf.json',
    'docs_obitos_agravo_por_uf_ano_semana.json'
]

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEBUG = os.getenv("DEBUG", "true").lower() == "true"

PERGUNTAS_SUGERIDAS = [
    'Quantos casos foram registrados por UF?',
    'Quais os sorotipos da dengue por UF?',
    'Quais sintomas predominam por estado?',
    'Quais os 10 estados com mais casos de dengue?',
    'Mostre os casos por UF em formato de gráfico',
    'Mostre um grafo combinado de sintomas e sorotipos por UF',
    'Quais sintomas predominantes e sorotipos aparecem em cada estado?',
    'Mostre um grafo semântico da UF de São Paulo',
    'Qual a hospitalização por UF?',
    'Quais estados tiveram mais internações?',
    'Qual a taxa de internação por UF?',
    'Mostre a hospitalização por UF em formato de gráfico'
]


# ============================================================
# 2. FUNÇÕES AUXILIARES
# ============================================================

def normalizar_texto(texto: Optional[str]) -> str:
    if texto is None:
        return ""
    return str(texto).strip().lower()


def extrair_primeiro_campo_existente(metadata: Dict, campos: List[str], default="") -> str:
    for campo in campos:
        if campo in metadata and metadata[campo] is not None and str(metadata[campo]).strip() != "":
            return str(metadata[campo])
    return default


def numero_seguro(valor, default=0) -> int:
    try:
        if valor is None or valor == "":
            return default
        return int(float(str(valor).replace(",", ".")))
    except Exception:
        return default


def calcular_assinatura_arquivos(data_dir: Path, arquivos: List[str]) -> Dict:
    assinatura = {}
    for nome_arquivo in arquivos:
        caminho = data_dir / nome_arquivo
        if caminho.exists():
            stat = caminho.stat()
            assinatura[nome_arquivo] = {
                "size": stat.st_size,
                "mtime": stat.st_mtime
            }
        else:
            assinatura[nome_arquivo] = {
                'size': None,
                'mtime': None
            }
    return assinatura


def salvar_manifest(assinatura: Dict) -> None:
    PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(assinatura, f, ensure_ascii=False, indent=2)


def carregar_manifest() -> Optional[Dict]:
    if not MANIFEST_PATH.exists():
        return None
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def remover_base_chroma_se_existir() -> None:
    if not PERSIST_DIRECTORY.exists():
        return

    for item in PERSIST_DIRECTORY.iterdir():
        try:
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                import shutil
                shutil.rmtree(item)
        except Exception as e:
            st.warning(f"Não foi possível remover {item}: {e}")


def inferir_categoria(metadata: Dict, texto: str) -> str:
    document_type = normalizar_texto(metadata.get("document_type", ""))
    arquivo_origem = normalizar_texto(metadata.get("arquivo_origem", ""))
    texto_norm = normalizar_texto(texto)

    if any(t in document_type or t in arquivo_origem or t in texto_norm for t in [
        'hospitalizacao',
        'hospitalização',
        'internacao',
        'internação'
    ]):
        return "hospitalizacao"

    if any(t in document_type or t in arquivo_origem or t in texto_norm for t in [
        'obito',
        'óbito',
        'mortalidade'
    ]):
        return "mortalidade"
    
    if any(t in document_type or t in arquivo_origem or t in texto_norm for t in [
        'symptom',
        'sintoma'
    ]):
        return "sintomas"    

    if any(t in document_type or t in arquivo_origem or t in texto_norm for t in [
        'sorotipo',
        'serotype'
    ]):
        return "sorotipos"

    if any(t in document_type or t in arquivo_origem or t in texto_norm for t in [
        'casos',
        'case',
        'total_casos',
        'total_cases'
    ]):
        return "casos"

    return "geral"


def inferir_granularidade(metadata: Dict, texto: str) -> str:
    document_type = normalizar_texto(metadata.get("document_type", ""))
    texto_norm = normalizar_texto(texto)

    if "week" in document_type or "semana" in document_type or "semana" in texto_norm:
        return "semanal"

    if "year" in document_type or "ano" in document_type or "ano" in texto_norm:
        return "anual"

    if "state" in document_type or "uf" in document_type or "estado" in texto_norm:
        return "estadual"

    return "indefinida"


def slug_no(texto: str) -> str:
    texto = normalizar_texto(texto)
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")
    texto = re.sub(r"[^a-z0-9_]+", "_", texto)
    texto = re.sub(r"_+", "_", texto).strip("_")
    return texto or "no"


def consolidar_itens_por_maior_valor(itens: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
    mapa = {}
    nome_original = {}

    for nome, qtd in itens:
        chave = normalizar_texto(nome)
        if chave not in mapa or qtd > mapa[chave]:
            mapa[chave] = qtd
            nome_original[chave] = nome

    return sorted(
        [(nome_original[ch], mapa[ch]) for ch in mapa],
        key=lambda x: x[1],
        reverse=True
    )


def calcular_tamanho_por_percentual(qtd: int, total_referencia: Optional[int]) -> Tuple[float, float]:
    percentual = 0.0
    if total_referencia and total_referencia > 0:
        percentual = (qtd / total_referencia) * 100.0

    size = max(14.0, min(36.0, 12.0 + percentual * 0.3))
    return size, percentual


def extrair_sintomas_do_documento(doc: Document) -> List[Tuple[str, int]]:
    sintomas_encontrados = []

    principal = str(doc.metadata.get("principal_symptom", "")).strip()
    qtd_principal = numero_seguro(doc.metadata.get("principal_symptom_count", 0), 0)

    if principal:
        sintomas_encontrados.append((principal, qtd_principal))

    chaves_ignorar = {
        'principal_symptom',
        'principal_symptom_count',
        'document_type',
        'arquivo_origem',
        'categoria',
        'granularidade',
        'uf_name',
        'uf_normalizada',
        'uf',
        'nu_ano',
        'ano_normalizado',
        'ano',
        'semana_epidemiologica_range',
        'week',
        'semana',
        'num_cases',
        'num_cases_normalizado',
        'total_cases',
        'total_casos',
        'sg_uf_not',
        'state',
        'state_name',
        'year'
    }

    termos_sintoma = [
        'febre',
        'mialgia',
        'cefaleia',
        'exantema',
        'vomito',
        'vômito',
        'nausea',
        'náusea',
        'dor',
        'artralgia',
        'conjuntivite'
    ]

    for chave, valor in doc.metadata.items():
        chave_norm = normalizar_texto(chave)

        if chave in chaves_ignorar:
            continue

        if isinstance(valor, (int, float)) and valor > 0:
            if any(t in chave_norm for t in termos_sintoma):
                nome = chave.replace("_", " ").strip().title()
                sintomas_encontrados.append((nome, numero_seguro(valor, 0)))

    sintomas_encontrados = consolidar_itens_por_maior_valor(sintomas_encontrados)
    return sintomas_encontrados


def adicionar_no_expandivel(
    G: nx.DiGraph,
    no_pai: str,
    grupo_id: str,
    grupo_label: str,
    relacao_pai_grupo: str,
    itens: List[Tuple[str, int]],
    tipo_item: str,
    prefixo_item: str,
    item_destaque: Optional[str] = None,
    rotulo_destaque: Optional[str] = None,
    total_referencia: Optional[int] = None,
    uf: Optional[str] = None,
    ano: Optional[str] = None,
    arquivo_origem: Optional[str] = None,
    base_descricao: str = "do total",
    extras_grupo: Optional[Dict] = None
):
    if not itens:
        return

    atributos_grupo = {
        "label": grupo_label,
        "tipo": "Grupo",
        "title": f"Clique para expandir/recolher {grupo_label}",
        "grupo_expandivel": True,
        "expandido": False,
        "descricao": f"Grupo semântico de {grupo_label.lower()}",
        "detalhes_html": f"""
        <h3>{grupo_label}</h3>
        <p><b>Tipo:</b> Grupo</p>
        <p><b>Ação:</b> Clique para expandir ou recolher os itens.</p>
        <p><b>UF:</b> {uf or "-"}</p>
        <p><b>Ano:</b> {ano or "-"}</p>
        """
    }

    if extras_grupo:
        atributos_grupo.update(extras_grupo)

    G.add_node(grupo_id, **atributos_grupo)
    G.add_edge(no_pai, grupo_id, relacao=relacao_pai_grupo)

    for nome, qtd in itens:
        item_id = f"{prefixo_item}_{slug_no(nome)}"
        eh_destaque = (
            item_destaque is not None and
            normalizar_texto(nome) == normalizar_texto(item_destaque)
        )

        size, percentual = calcular_tamanho_por_percentual(qtd, total_referencia)

        if eh_destaque and rotulo_destaque:
            label = f"{rotulo_destaque}: {nome} ({qtd})" if qtd > 0 else f"{rotulo_destaque}: {nome}"
            tipo_no = f"{tipo_item}Destaque"
        else:
            label = f"{nome} ({qtd})" if qtd > 0 else nome
            tipo_no = tipo_item

        titulo = label
        if total_referencia and total_referencia > 0:
            titulo = f"{label} — {percentual:.1f}% {base_descricao} ({qtd}/{total_referencia})"

        detalhes_html = f"""
        <h3>{nome}</h3>
        <p><b>Tipo:</b> {tipo_item}</p>
        <p><b>Quantidade:</b> {qtd}</p>
        <p><b>Percentual:</b> {percentual:.1f}%</p>
        <p><b>UF:</b> {uf or "-"}</p>
        <p><b>Ano:</b> {ano or "-"}</p>
        <p><b>Arquivo de origem:</b> {arquivo_origem or "-"}</p>
        """

        G.add_node(
            item_id,
            label=label,
            tipo=tipo_no,
            title=titulo,
            hidden=True,
            parent_group=grupo_id,
            size=size,
            percentual=round(percentual, 2),
            qtd=qtd,
            uf=uf,
            ano=ano,
            arquivo_origem=arquivo_origem,
            descricao=f"{tipo_item} associado à UF {uf}" if uf else tipo_item,
            detalhes_html=detalhes_html
        )

        G.add_edge(
            grupo_id,
            item_id,
            relacao="contém",
            hidden=True,
            parent_group=grupo_id
        )

# ============================================================
# 3. LEITURA DOS DOCUMENTOS JSON
# ============================================================

@st.cache_data
def carregar_documentos_json() -> List[Document]:
    documentos = []

    for nome_arquivo in ARQUIVOS_JSON:
        caminho = DATA_DIR / nome_arquivo

        if not caminho.exists():
            st.warning(f"Arquivo não encontrado: {caminho}")
            continue

        with open(caminho, "r", encoding="utf-8") as f:
            conteudo = json.load(f)

        if not isinstance(conteudo, list):
            st.warning(f"O arquivo {nome_arquivo} não contém uma lista de documentos.")
            continue

        for item in conteudo:
            if not isinstance(item, dict):
                continue

            metadata = item.get("metadata", {}) or {}
            texto = item.get("text_content", "") or ""

            # =========================
            # Normalização específica dos campos
            # =========================
            if "tipo_documento" in metadata and "document_type" not in metadata:
                metadata["document_type"] = metadata["tipo_documento"]

            if "nome_uf" in metadata and "uf_name" not in metadata:
                metadata["uf_name"] = metadata["nome_uf"]
            
            # compatibilidade com novo arquivo de hospitalização
            if "uf_sigla" in metadata and "uf" not in metadata:
                metadata["uf"] = metadata["uf_sigla"]

            if "uf_sigla" in metadata and "sg_uf_not" not in metadata:
                metadata["sg_uf_not"] = metadata["uf_sigla"]

            if "ano" in metadata and "nu_ano" not in metadata:
                metadata["nu_ano"] = metadata["ano"]
            
            if "internacoes" in metadata and "sim" not in metadata:
                metadata["sim"] = metadata["internacoes"]

            if "nao_hospitalizados" in metadata and "nao" not in metadata:
                metadata["nao"] = metadata["nao_hospitalizados"]

            if "taxa_internacao_bruta" in metadata and "taxa_internacao" not in metadata:
                metadata["taxa_internacao"] = metadata["taxa_internacao_bruta"]

            # Para documentos por período, usa a semana final como semana normalizada
            if "semana_final" in metadata and "semana" not in metadata:
                metadata["semana"] = str(metadata["semana_final"]).zfill(2)

            if "semana_epidemiologica" in metadata and "semana" not in metadata:
                metadata["semana"] = str(metadata["semana_epidemiologica"]).zfill(2)

            # Normalização para óbitos
            if "obitos_agravo" in metadata and "obitos" not in metadata:
                metadata["obitos"] = metadata["obitos_agravo"]

            metadata["arquivo_origem"] = nome_arquivo
            metadata["document_type"] = str(metadata.get("document_type", "desconhecido"))
            metadata["categoria"] = inferir_categoria(metadata, texto)
            metadata["granularidade"] = inferir_granularidade(metadata, texto)

            metadata["uf_normalizada"] = extrair_primeiro_campo_existente(
                metadata,
                [
                    "uf",
                    "uf_name",
                    "nome_uf",
                    "state",
                    "state_name",
                    "sg_uf_not"
                ]
            )

            metadata["ano_normalizado"] = extrair_primeiro_campo_existente(
                metadata,
                [
                    "year",
                    "ano",
                    "ano_notificacao",
                    "nu_ano"
                ]
            )

            metadata["semana_normalizada"] = extrair_primeiro_campo_existente(
                metadata,
                [
                    "week",
                    "semana",
                    "semana_epidemiologica",
                    "nu_semana_notificacao",
                    "semana_epidemiologica_range"
                ]
            )

            metadata["sorotipo_normalizado"] = extrair_primeiro_campo_existente(
                metadata,
                [
                    "sorotipo",
                    "serotype"
                ]
            )

            metadata["num_cases_normalizado"] = extrair_primeiro_campo_existente(
                metadata,
                [
                    "num_cases",
                    "total_cases",
                    "casos",
                    "total_casos"
                ]
            )

            documentos.append(
                Document(
                    page_content=texto,
                    metadata=metadata
                )
            )

    return documentos

# ============================================================
# 4. VECTORSTORE
# ============================================================

@st.cache_resource
def criar_ou_carregar_vectorstore(_documentos: List[Document], assinatura_atual_hash: str):
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)

    manifest_salvo = carregar_manifest()
    assinatura_atual = calcular_assinatura_arquivos(DATA_DIR, ARQUIVOS_JSON)

    precisa_recriar = False

    if manifest_salvo is None:
        precisa_recriar = True
    elif manifest_salvo != assinatura_atual:
        precisa_recriar = True

    if precisa_recriar:
        st.info("Mudança detectada nos arquivos JSON. Recriando a base vetorial...")
        remover_base_chroma_se_existir()

        vectorstore = Chroma.from_documents(
            documents=_documentos,
            embedding=embeddings,
            persist_directory=str(PERSIST_DIRECTORY),
            collection_name="arboviroses_docs"
        )

        salvar_manifest(assinatura_atual)
        return vectorstore

    try:
        vectorstore = Chroma(
            persist_directory=str(PERSIST_DIRECTORY),
            embedding_function=embeddings,
            collection_name="arboviroses_docs"
        )
        return vectorstore
    except Exception as e:
        st.warning(f"Falha ao carregar a base vetorial existente. Recriando. Detalhe: {e}")

        remover_base_chroma_se_existir()

        vectorstore = Chroma.from_documents(
            documents=_documentos,
            embedding=embeddings,
            persist_directory=str(PERSIST_DIRECTORY),
            collection_name="arboviroses_docs"
        )

        salvar_manifest(assinatura_atual)
        return vectorstore


# ============================================================
# 5. ENTENDIMENTO DA PERGUNTA
# ============================================================

SINTOMAS_CHAVE = [
    'febre',
    'mialgia',
    'cefaleia',
    'exantema',
    'vomito',
    'vômito',
    'nausea',
    'náusea',
    'dor',
    'artralgia',
    'conjuntivite',
    'sintoma',
    'sintomas'
]

SOROTIPOS_CHAVE = [
    'sorotipo',
    'sorotipos',
    'den 1',
    'den1',
    'den 2',
    'den2',
    'den 3',
    'den3',
    'den 4',
    'den4'
]

CASOS_CHAVE = [
    'caso',
    'casos',
    'quantos',
    'total',
    'incidência',
    'incidencia',
    'notificados',
    'número',
    'numero',
    'registrados',
    'uf',
    'estado',
    'estados'
]

TEMPORAL_CHAVE = [
    'semana',
    'semanal',
    'ano',
    'anos',
    '2023',
    '2024',
    '2025',
    '2026',
    'evolução',
    'evolucao',
    'temporal'
]

HOSPITALIZACAO_CHAVE = [
    'hospitalização',
    'hospitalizacao',
    'internação',
    'internacao',
    'internações',
    'internacoes',
    'hospitalizado',
    'hospitalizados',
    'hospitalizada',
    'hospitalizadas',
    'internado',
    'internados',
    'taxa de internação',
    'taxa de internacao',
    'taxa de hospitalização',
    'taxa de hospitalizacao'
]

UF_SIGLAS = [
    'AC',
    'AL',
    'AP',
    'AM',
    'BA',
    'CE',
    'DF',
    'ES',
    'GO',
    'MA',
    'MT',
    'MS',
    'MG',
    'PA',
    'PB',
    'PR',
    'PE',
    'PI',
    'RJ',
    'RN',
    'RS',
    'RO',
    'RR',
    'SC',
    'SP',
    'SE',
    'TO'
]

UF_NOMES = [
    'acre',
    'alagoas',
    'amapá',
    'amapa',
    'amazonas',
    'bahia',
    'ceará',
    'ceara',
    'distrito federal',
    'espírito santo',
    'espirito santo',
    'goiás',
    'goias',
    'maranhão',
    'maranhao',
    'mato grosso',
    'mato grosso do sul',
    'minas gerais',
    'pará',
    'para',
    'paraíba',
    'paraiba',
    'paraná',
    'parana',
    'pernambuco',
    'piauí',
    'piaui',
    'rio de janeiro',
    'rio grande do norte',
    'rio grande do sul',
    'rondônia',
    'rondonia',
    'roraima',
    'santa catarina',
    'são paulo',
    'sao paulo',
    'sergipe',
    'tocantins'
]


def classificar_pergunta(pergunta: str) -> Dict:
    p = normalizar_texto(pergunta)

    intencoes = {
        "sintomas": any(k in p for k in SINTOMAS_CHAVE),
        "sorotipos": any(k in p for k in SOROTIPOS_CHAVE),
        "casos": any(k in p for k in CASOS_CHAVE),
        "temporal": any(k in p for k in TEMPORAL_CHAVE),
        "hospitalizacao": any(k in p for k in HOSPITALIZACAO_CHAVE),
    }

    anos = re.findall(r"\b(20\d{2})\b", p)
    semanas = re.findall(r"\bsemana\s*(\d{1,2})\b", p)

    ufs_sigla_encontradas = []
    palavras = re.findall(r"\b[A-Za-zÀ-ÿ0-9]+\b", pergunta)
    for token in palavras:
        token_upper = token.upper()
        if token_upper in UF_SIGLAS:
            ufs_sigla_encontradas.append(token_upper)

    ufs_nome_encontradas = [uf for uf in UF_NOMES if uf in p]

    return {
        "intencoes": intencoes,
        "anos": list(set(anos)),
        "semanas": list(set(semanas)),
        "ufs_sigla": list(set(ufs_sigla_encontradas)),
        "ufs_nome": list(set(ufs_nome_encontradas)),
    }


def eh_pergunta_total_casos_por_uf(pergunta: str, query_info: Dict) -> bool:
    pergunta_norm = normalizar_texto(pergunta)
    menciona_uf_ou_estado = (
        "uf" in pergunta_norm or
        "estado" in pergunta_norm or
        "estados" in pergunta_norm or
        "por uf" in pergunta_norm or
        "por estado" in pergunta_norm
    )
    return (
        query_info["intencoes"]["casos"]
        and menciona_uf_ou_estado
        and not query_info["intencoes"]["sorotipos"]
        and not query_info["intencoes"]["sintomas"]
    )


def eh_pergunta_sorotipos_por_uf(pergunta: str, query_info: Dict) -> bool:
    pergunta_norm = normalizar_texto(pergunta)
    menciona_uf_ou_estado = (
        "uf" in pergunta_norm or
        "estado" in pergunta_norm or
        "estados" in pergunta_norm or
        "por uf" in pergunta_norm or
        "por estado" in pergunta_norm
    )
    return (
        query_info["intencoes"]["sorotipos"]
        and menciona_uf_ou_estado
        and not query_info["intencoes"]["sintomas"]
    )


def eh_pergunta_sintomas_por_estado(pergunta: str, query_info: Dict) -> bool:
    pergunta_norm = normalizar_texto(pergunta)
    menciona_estado = (
        "estado" in pergunta_norm or
        "estados" in pergunta_norm or
        "uf" in pergunta_norm or
        "por estado" in pergunta_norm or
        "por uf" in pergunta_norm
    )
    return (
        query_info["intencoes"]["sintomas"]
        and menciona_estado
        and not query_info["intencoes"]["sorotipos"]
    )
def eh_pergunta_hospitalizacao_por_uf(pergunta: str, query_info: Dict) -> bool:
    pergunta_norm = normalizar_texto(pergunta)
    menciona_uf_ou_estado = (
        "uf" in pergunta_norm or
        "estado" in pergunta_norm or
        "estados" in pergunta_norm or
        "por uf" in pergunta_norm or
        "por estado" in pergunta_norm
    )
    return (
        query_info["intencoes"]["hospitalizacao"]
        and menciona_uf_ou_estado
        and not query_info["intencoes"]["sorotipos"]
        and not query_info["intencoes"]["sintomas"]
    )

# ============================================================
# 6. PESOS POR CATEGORIA E DOCUMENT TYPE
# ============================================================

def definir_pesos(query_info: Dict) -> Dict:
    intencoes = query_info["intencoes"]

    pesos_categoria = {
        'sintomas': 1.0,
        'sorotipos': 1.0,
        'casos': 1.0,
        'hospitalizacao': 1.0,
        'geral': 0.6
    }

    pesos_document_type = {
        'symptoms_by_state': 1.0,
        'sorotipos_by_state_and_week': 1.0,
        'sorotipos_by_state': 1.0,
        'sorotipos_por_uf_com_codigos_e_nomes': 1.0,
        'total_casos_uf_ano': 1.0,
        'hospitalizacao_uf': 1.0,
        'hospitalizacao_por_uf_semana': 1.0,
        'hospitalizacao_por_uf_periodo': 1.0,
        'desconhecido': 0.7
    }   

    if intencoes["sintomas"]:
        pesos_categoria["sintomas"] = 2.5
        pesos_categoria["sorotipos"] = 0.8
        pesos_categoria["casos"] = 1.0
        pesos_categoria["hospitalizacao"] = 0.8
        pesos_document_type["symptoms_by_state"] = 3.0

    if intencoes["sorotipos"]:
        pesos_categoria["sorotipos"] = 2.5
        pesos_categoria["sintomas"] = 0.8
        pesos_categoria["casos"] = 1.0
        pesos_categoria["hospitalizacao"] = 0.8
        pesos_document_type["sorotipos_by_state_and_week"] = 3.0
        pesos_document_type["sorotipos_by_state"] = 2.5
        pesos_document_type["sorotipos_por_uf_com_codigos_e_nomes"] = 2.8

    if intencoes["casos"]:
        pesos_categoria["casos"] = 2.5
        pesos_categoria["sintomas"] = 0.9
        pesos_categoria["sorotipos"] = 1.0
        pesos_categoria["hospitalizacao"] = 1.2
        pesos_document_type["total_casos_uf_ano"] = 3.0

    if intencoes["hospitalizacao"]:
        pesos_categoria["hospitalizacao"] = 2.8
        pesos_categoria["casos"] = 1.2
        pesos_categoria["sintomas"] = 0.8
        pesos_categoria["sorotipos"] = 0.8
        pesos_document_type["hospitalizacao_uf"] = 3.2
        pesos_document_type["hospitalizacao_por_uf_semana"] = 3.2
        pesos_document_type["hospitalizacao_por_uf_periodo"] = 3.2

    if intencoes["temporal"]:
        pesos_document_type["sorotipos_by_state_and_week"] += 0.8

    return {
        "pesos_categoria": pesos_categoria,
        "pesos_document_type": pesos_document_type
    }


# ============================================================
# 7. SCORE HEURÍSTICO
# ============================================================

def score_heuristico(pergunta: str, doc: Document, query_info: Dict, pesos: Dict) -> Tuple[float, List[str]]:
    score = 0.0
    justificativas = []

    categoria = normalizar_texto(doc.metadata.get("categoria", "geral"))
    document_type = normalizar_texto(doc.metadata.get("document_type", "desconhecido"))
    texto = normalizar_texto(doc.page_content)
    metadata_text = json.dumps(doc.metadata, ensure_ascii=False).lower()

    peso_cat = pesos["pesos_categoria"].get(categoria, 0.5)
    score += peso_cat
    justificativas.append(f"peso_categoria={peso_cat}")

    peso_doc_type = pesos["pesos_document_type"].get(document_type, 0.7)
    score += peso_doc_type
    justificativas.append(f"peso_document_type={peso_doc_type}")

    for ano in query_info["anos"]:
        if ano in texto or ano in metadata_text:
            score += 1.2
            justificativas.append(f"match_ano={ano}")

    for semana in query_info["semanas"]:
        if semana in texto or semana in metadata_text:
            score += 1.3
            justificativas.append(f"match_semana={semana}")

    for uf in query_info["ufs_sigla"]:
        if uf.lower() in texto or uf.lower() in metadata_text:
            score += 1.4
            justificativas.append(f"match_uf_sigla={uf}")

    for uf_nome in query_info["ufs_nome"]:
        if uf_nome in texto or uf_nome in metadata_text:
            score += 1.4
            justificativas.append(f"match_uf_nome={uf_nome}")

    if query_info["intencoes"]["sintomas"]:
        matches = sum(1 for termo in SINTOMAS_CHAVE if termo in texto or termo in metadata_text)
        bonus = min(matches * 0.25, 1.5)
        score += bonus
        if bonus > 0:
            justificativas.append(f"bonus_sintomas={bonus:.2f}")

    if query_info["intencoes"]["sorotipos"]:
        matches = sum(1 for termo in SOROTIPOS_CHAVE if termo in texto or termo in metadata_text)
        bonus = min(matches * 0.25, 1.5)
        score += bonus
        if bonus > 0:
            justificativas.append(f"bonus_sorotipos={bonus:.2f}")

    if query_info["intencoes"]["casos"]:
        matches = sum(1 for termo in CASOS_CHAVE if termo in texto or termo in metadata_text)
        bonus = min(matches * 0.25, 1.5)
        score += bonus
        if bonus > 0:
            justificativas.append(f"bonus_casos={bonus:.2f}")

    return score, justificativas


# ============================================================
# 8. FILTRAGEM POR DOCUMENT TYPE
# ============================================================

def filtrar_por_document_type(documentos: List[Document], query_info: Dict, pergunta: str = "") -> List[Document]:
    if eh_pergunta_hospitalizacao_por_uf(pergunta, query_info):
        docs_filtrados = [
            doc for doc in documentos
            if normalizar_texto(doc.metadata.get("document_type", "")) in [
                "hospitalizacao_uf",
                "hospitalizacao_por_uf_semana",
                "hospitalizacao_por_uf_periodo"
            ]  
        ]    
        return docs_filtrados if docs_filtrados else documentos
    
    if eh_pergunta_total_casos_por_uf(pergunta, query_info):
        docs_filtrados = [
            doc for doc in documentos
            if normalizar_texto(doc.metadata.get("document_type", "")) == "total_casos_uf_ano"
        ]
        return docs_filtrados if docs_filtrados else documentos

    if eh_pergunta_sorotipos_por_uf(pergunta, query_info):
        docs_filtrados = [
            doc for doc in documentos
            if normalizar_texto(doc.metadata.get("document_type", "")) in [
                'sorotipos_por_uf_com_codigos_e_nomes',
                'sorotipos_by_state'
            ]
        ]
        return docs_filtrados if docs_filtrados else documentos

    intencoes = query_info["intencoes"]
    termos_prioritarios = []

    if intencoes["sintomas"]:
        termos_prioritarios.extend([
            'symptoms_by_state',
            'symptom',
            'sintoma'
        ])

    if intencoes["sorotipos"]:
        termos_prioritarios.extend([
            'sorotipos_by_state_and_week',
            'sorotipos_by_state',
            'sorotipos_por_uf_com_codigos_e_nomes',
            'sorotipo',
            'serotype'
        ])

    if intencoes["hospitalizacao"]:
        termos_prioritarios.extend([
            'hospitalizacao_uf',
            'hospitalizacao',
            'internacao',
            'internação'
        ])

    if intencoes["casos"]:
        termos_prioritarios.extend([
            'total_casos_uf_ano',
            'total_casos',
            'total_cases'
        ])

    if not termos_prioritarios:
        return documentos

    docs_filtrados = []
    for doc in documentos:
        doc_type = normalizar_texto(doc.metadata.get("document_type", ""))
        arquivo = normalizar_texto(doc.metadata.get("arquivo_origem", ""))

        if any(t in doc_type or t in arquivo for t in termos_prioritarios):
            docs_filtrados.append(doc)

    return docs_filtrados if docs_filtrados else documentos


# ============================================================
# 9. RERANKING COM LLM
# ============================================================

def rerank_com_llm(pergunta: str, documentos: List[Document], llm) -> List[Document]:
    prompt_rerank = PromptTemplate(
        input_variables=[
            'pergunta',
            'texto',
            'metadata'
        ],
        template="""
Você é um especialista em vigilância de arboviroses.

Pergunta do usuário:
{pergunta}

Trecho do documento:
{texto}

Metadados:
{metadata}

Avalie quão relevante esse trecho é para responder a pergunta.

Critérios:
1. correspondência com o foco da pergunta (sintomas, sorotipos ou casos);
2. aderência temporal (semana/ano) quando aplicável;
3. aderência espacial (UF/estado) quando aplicável;
4. utilidade objetiva para uma resposta correta.

Responda apenas com um número de 0 a 10.
"""
    )

    docs_com_score = []

    for doc in documentos:
        try:
            resposta = llm.invoke(
                prompt_rerank.format(
                    pergunta=pergunta,
                    texto=doc.page_content,
                    metadata=json.dumps(doc.metadata, ensure_ascii=False)
                )
            ).content.strip()

            resposta_limpa = resposta.replace(",", ".").strip()
            match = re.search(r"(\d+(\.\d+)?)", resposta_limpa)
            score = float(match.group(1)) if match else 0.0
        except Exception:
            score = 0.0

        doc.metadata["score_llm"] = score
        docs_com_score.append((score, doc))

    docs_ordenados = sorted(docs_com_score, key=lambda x: x[0], reverse=True)
    return [doc for _, doc in docs_ordenados]


# ============================================================
# 10. FLUXOS DETERMINÍSTICOS
# ============================================================

def filtrar_docs_total_casos_por_uf(documentos: List[Document], query_info: Dict) -> List[Document]:
    docs_casos = [
        doc for doc in documentos
        if normalizar_texto(doc.metadata.get("document_type", "")) == "total_casos_uf_ano"
    ]

    if query_info["anos"]:
        anos_desejados = set(query_info["anos"])
        docs_casos = [
            doc for doc in docs_casos
            if str(doc.metadata.get("nu_ano", "")) in anos_desejados
            or str(doc.metadata.get("ano_normalizado", "")) in anos_desejados
        ]

    if query_info["ufs_nome"]:
        ufs_nome_desejadas = set(query_info["ufs_nome"])
        docs_casos = [
            doc for doc in docs_casos
            if normalizar_texto(doc.metadata.get("uf_name", "")) in ufs_nome_desejadas
            or normalizar_texto(doc.metadata.get("uf_normalizada", "")) in ufs_nome_desejadas
        ]

    if query_info["ufs_sigla"]:
        siglas_desejadas = set(query_info["ufs_sigla"])
        docs_casos = [
            doc for doc in docs_casos
            if str(doc.metadata.get("sg_uf_not", "")).upper() in siglas_desejadas
            or str(doc.metadata.get("uf_normalizada", "")).upper() in siglas_desejadas
        ]

    return sorted(
        docs_casos,
        key=lambda d: numero_seguro(d.metadata.get("total_casos", d.metadata.get("num_cases_normalizado", 0)), 0),
        reverse=True
    )

def filtrar_docs_hospitalizacao_por_uf(documentos: List[Document], query_info: Dict) -> List[Document]:
    docs = [
        doc for doc in documentos
        if normalizar_texto(doc.metadata.get("document_type", "")) in [
            "hospitalizacao_uf",
            "hospitalizacao_por_uf_semana",
            "hospitalizacao_por_uf_periodo"
        ]
    ]

    if query_info["anos"]:
        anos_desejados = set(query_info["anos"])
        docs = [
            doc for doc in docs
            if str(doc.metadata.get("nu_ano", doc.metadata.get("ano_normalizado", doc.metadata.get("ano", "")))) in anos_desejados
        ]

    if query_info["ufs_nome"]:
        ufs_nome_desejadas = set(query_info["ufs_nome"])
        docs = [
            doc for doc in docs
            if normalizar_texto(doc.metadata.get("uf_name", "")) in ufs_nome_desejadas
            or normalizar_texto(doc.metadata.get("uf_normalizada", "")) in ufs_nome_desejadas
            or normalizar_texto(doc.metadata.get("uf", "")) in ufs_nome_desejadas
        ]

    if query_info["ufs_sigla"]:
        siglas_desejadas = set(query_info["ufs_sigla"])
        docs = [
            doc for doc in docs
            if str(doc.metadata.get("sg_uf_not", "")).upper() in siglas_desejadas
            or str(doc.metadata.get("uf", "")).upper() in siglas_desejadas
            or str(doc.metadata.get("uf_sigla", "")).upper() in siglas_desejadas
            or str(doc.metadata.get("uf_normalizada", "")).upper() in siglas_desejadas
        ]

    def chave_ordenacao(doc):
        md = doc.metadata
        taxa = md.get(
            "taxa_internacao",
            md.get("taxa_internacao_bruta", md.get("taxa_hospitalizacao", 0))
        )
        try:
            taxa = float(str(taxa).replace(",", "."))
        except Exception:
            taxa = 0.0

        internacoes = numero_seguro(
            md.get("internacoes", md.get("sim", md.get("hospitalizacoes", md.get("total_internacoes", 0)))),
            0
        )

        return (taxa, internacoes)

    return sorted(docs, key=chave_ordenacao, reverse=True)

def responder_hospitalizacao_por_uf(documentos: List[Document], pergunta: str) -> Tuple[str, List[Document], str]:
    query_info = classificar_pergunta(pergunta)
    docs_ordenados = filtrar_docs_hospitalizacao_por_uf(documentos, query_info)

    if not docs_ordenados:
        return "Não encontrei documentos de hospitalização por UF compatíveis com a pergunta.", [], "deterministico_hospitalizacao"

    anos_encontrados = sorted({
        str(doc.metadata.get("nu_ano", doc.metadata.get("ano_normalizado", doc.metadata.get("ano", ""))))
        for doc in docs_ordenados
        if str(doc.metadata.get("nu_ano", doc.metadata.get("ano_normalizado", doc.metadata.get("ano", "")))).strip()
    })

    faixas_encontradas = sorted({
        str(doc.metadata.get("semana_epidemiologica_range", "")).strip()
        for doc in docs_ordenados
        if str(doc.metadata.get("semana_epidemiologica_range", "")).strip()
    })

    linhas = []
    for doc in docs_ordenados:
        md = doc.metadata
        uf = md.get("uf_name", md.get("uf_normalizada", md.get("uf", "UF não informada")))

        internacoes = numero_seguro(
            md.get("internacoes", md.get("sim", md.get("hospitalizacoes", md.get("total_internacoes", 0)))),
            0
        )

        total_casos = numero_seguro(
            md.get("total_casos", md.get("total_cases", md.get("num_cases_normalizado", 0))),
            0
        )

        taxa = md.get(
            "taxa_internacao",
            md.get("taxa_internacao_bruta", md.get("taxa_hospitalizacao", 0))
        )
        try:
            taxa = float(str(taxa).replace(",", "."))
        except Exception:
            taxa = 0.0

        ano = md.get("nu_ano", md.get("ano_normalizado", md.get("ano", "ano não informado")))
        faixa = str(md.get("semana_epidemiologica_range", "")).strip()

        if faixa:
            linhas.append(
                f"- {uf}: {internacoes} internações em {total_casos} casos "
                f"(taxa de internação: {taxa:.2f}%, ano {ano}, faixa epidemiológica {faixa})"
            )
        else:
            linhas.append(
                f"- {uf}: {internacoes} internações em {total_casos} casos "
                f"(taxa de internação: {taxa:.2f}%, ano {ano})"
            )

    cabecalho = "Hospitalização por UF"
    if anos_encontrados:
        cabecalho += f" para o(s) ano(s) {', '.join(anos_encontrados)}"
    if len(faixas_encontradas) == 1:
        cabecalho += f", considerando a faixa epidemiológica {faixas_encontradas[0]}"
    cabecalho += f". Foram encontradas {len(docs_ordenados)} UFs com registros de hospitalização."

    resposta = cabecalho + "\n\n" + "\n".join(linhas)
    return resposta, docs_ordenados, "deterministico_hospitalizacao"

def responder_total_casos_por_uf(documentos: List[Document], pergunta: str) -> Tuple[str, List[Document], str]:
    query_info = classificar_pergunta(pergunta)
    docs_ordenados = filtrar_docs_total_casos_por_uf(documentos, query_info)

    if not docs_ordenados:
        return "Não encontrei documentos do tipo total_casos_uf_ano compatíveis com a pergunta.", [], "deterministico_casos"

    total_ufs = len(docs_ordenados)

    anos_encontrados = sorted({
        str(doc.metadata.get("nu_ano", doc.metadata.get("ano_normalizado", "")))
        for doc in docs_ordenados
        if str(doc.metadata.get("nu_ano", doc.metadata.get("ano_normalizado", ""))).strip() != ""
    })

    semanas_encontradas = sorted({
        str(doc.metadata.get("semana_epidemiologica_range", ""))
        for doc in docs_ordenados
        if str(doc.metadata.get("semana_epidemiologica_range", "")).strip() != ""
    })

    linhas = []
    for doc in docs_ordenados:
        uf = doc.metadata.get("uf_name", "UF não informada")
        total = numero_seguro(doc.metadata.get("total_casos", doc.metadata.get("num_cases_normalizado", 0)), 0)
        ano = doc.metadata.get("nu_ano", doc.metadata.get("ano_normalizado", "ano não informado"))
        faixa = doc.metadata.get("semana_epidemiologica_range", "")

        if faixa:
            linhas.append(f"- {uf}: {total} casos (ano {ano}, semanas {faixa})")
        else:
            linhas.append(f"- {uf}: {total} casos (ano {ano})")

    cabecalho = "Total de casos registrados por UF"
    if anos_encontrados:
        cabecalho += f" para o(s) ano(s) {', '.join(anos_encontrados)}"
    if semanas_encontradas and len(semanas_encontradas) == 1:
        cabecalho += f", considerando a faixa epidemiológica {semanas_encontradas[0]}"
    cabecalho += f". Foram encontradas {total_ufs} UFs no conjunto recuperado."

    resposta = cabecalho + "\n\n" + "\n".join(linhas)
    return resposta, docs_ordenados, "deterministico_casos"


def filtrar_docs_sorotipos_por_uf(documentos: List[Document], query_info: Dict) -> List[Document]:
    docs_sorotipos = [
        doc for doc in documentos
        if normalizar_texto(doc.metadata.get("document_type", "")) in [
            'sorotipos_por_uf_com_codigos_e_nomes',
            'sorotipos_by_state'
        ]
    ]

    if query_info["anos"]:
        anos_desejados = set(query_info["anos"])
        docs_sorotipos = [
            doc for doc in docs_sorotipos
            if str(doc.metadata.get("nu_ano", "")) in anos_desejados
            or str(doc.metadata.get("ano_normalizado", "")) in anos_desejados
        ]

    if query_info["ufs_nome"]:
        ufs_nome_desejadas = set(query_info["ufs_nome"])
        docs_sorotipos = [
            doc for doc in docs_sorotipos
            if normalizar_texto(doc.metadata.get("uf_name", "")) in ufs_nome_desejadas
            or normalizar_texto(doc.metadata.get("uf_normalizada", "")) in ufs_nome_desejadas
        ]

    if query_info["ufs_sigla"]:
        siglas_desejadas = set(query_info["ufs_sigla"])
        docs_sorotipos = [
            doc for doc in docs_sorotipos
            if str(doc.metadata.get("sg_uf_not", "")).upper() in siglas_desejadas
            or str(doc.metadata.get("uf_normalizada", "")).upper() in siglas_desejadas
        ]

    return sorted(
        docs_sorotipos,
        key=lambda d: (
            str(d.metadata.get("uf_name", "")),
            -numero_seguro(d.metadata.get("total_casos", d.metadata.get("num_cases_normalizado", 0)), 0)
        )
    )


def responder_sorotipos_por_uf(documentos: List[Document], pergunta: str) -> Tuple[str, List[Document], str]:
    query_info = classificar_pergunta(pergunta)
    docs_ordenados = filtrar_docs_sorotipos_por_uf(documentos, query_info)

    if not docs_ordenados:
        return "Não encontrei documentos de sorotipos por UF compatíveis com a pergunta.", [], "deterministico_sorotipos"

    agrupado = {}
    anos_encontrados = set()
    semanas_encontradas = set()

    for doc in docs_ordenados:
        uf = doc.metadata.get("uf_name", "UF não informada")
        sorotipo = doc.metadata.get("sorotipo", doc.metadata.get("sorotipo_normalizado", "sorotipo não informado"))
        total = numero_seguro(doc.metadata.get("total_casos", doc.metadata.get("num_cases_normalizado", 0)), 0)

        agrupado.setdefault(uf, [])
        agrupado[uf].append((sorotipo, total))

        if doc.metadata.get("nu_ano"):
            anos_encontrados.add(str(doc.metadata.get("nu_ano")))
        if doc.metadata.get("semana_epidemiologica_range"):
            semanas_encontradas.add(str(doc.metadata.get("semana_epidemiologica_range")))

    linhas = []
    for uf in sorted(agrupado.keys()):
        linhas.append(f"**{uf}:**")
        sorotipos_ordenados = sorted(agrupado[uf], key=lambda x: x[1], reverse=True)
        for sorotipo, total in sorotipos_ordenados:
            linhas.append(f"- {sorotipo}: {total} casos")
        linhas.append("")

    cabecalho = "Sorotipos da dengue por UF"
    if anos_encontrados:
        cabecalho += f" para o(s) ano(s) {', '.join(sorted(anos_encontrados))}"
    if len(semanas_encontradas) == 1:
        cabecalho += f", considerando a faixa epidemiológica {list(semanas_encontradas)[0]}"
    cabecalho += f". Foram encontradas {len(agrupado)} UFs com registros de sorotipo."

    resposta = cabecalho + "\n\n" + "\n".join(linhas)
    return resposta, docs_ordenados, "deterministico_sorotipos"


def filtrar_docs_sintomas_por_estado(documentos: List[Document], query_info: Dict) -> List[Document]:
    docs = [
        doc for doc in documentos
        if normalizar_texto(doc.metadata.get("document_type", "")) == "symptoms_by_state"
    ]

    if query_info["anos"]:
        anos_desejados = set(query_info["anos"])
        docs = [
            doc for doc in docs
            if str(doc.metadata.get("nu_ano", "")) in anos_desejados
            or str(doc.metadata.get("ano_normalizado", "")) in anos_desejados
        ]

    if query_info["ufs_nome"]:
        ufs_nome_desejadas = set(query_info["ufs_nome"])
        docs = [
            doc for doc in docs
            if normalizar_texto(doc.metadata.get("uf_name", "")) in ufs_nome_desejadas
            or normalizar_texto(doc.metadata.get("uf_normalizada", "")) in ufs_nome_desejadas
        ]

    return sorted(
        docs,
        key=lambda d: numero_seguro(d.metadata.get("total_cases", d.metadata.get("num_cases_normalizado", 0)), 0),
        reverse=True
    )


def responder_sintomas_por_estado(documentos: List[Document], pergunta: str) -> Tuple[str, List[Document], str]:
    query_info = classificar_pergunta(pergunta)
    docs = filtrar_docs_sintomas_por_estado(documentos, query_info)

    if not docs:
        return "Não encontrei documentos de sintomas por estado compatíveis com a pergunta.", [], "deterministico_sintomas"

    anos = sorted({
        str(doc.metadata.get("nu_ano", doc.metadata.get("ano_normalizado", "")))
        for doc in docs if str(doc.metadata.get("nu_ano", doc.metadata.get("ano_normalizado", ""))).strip()
    })

    semanas = sorted({
        str(doc.metadata.get("semana_epidemiologica_range", ""))
        for doc in docs if str(doc.metadata.get("semana_epidemiologica_range", "")).strip()
    })

    linhas = []
    for doc in docs:
        uf = doc.metadata.get("uf_name", "UF não informada")
        principal = doc.metadata.get("principal_symptom", "não informado")
        total_principal = numero_seguro(doc.metadata.get("principal_symptom_count", 0))
        total_cases = numero_seguro(doc.metadata.get("total_cases", doc.metadata.get("num_cases_normalizado", 0)))
        linhas.append(
            f"- {uf}: sintoma predominante = {principal} ({total_principal} ocorrências em {total_cases} casos)"
        )

    cabecalho = "Sintomas predominantes por estado"
    if anos:
        cabecalho += f" para o(s) ano(s) {', '.join(anos)}"
    if len(semanas) == 1:
        cabecalho += f", considerando a faixa epidemiológica {semanas[0]}"
    cabecalho += f". Foram encontradas {len(docs)} UFs com registros."

    resposta = cabecalho + "\n\n" + "\n".join(linhas)
    return resposta, docs, "deterministico_sintomas"


def buscar_obitos_por_uf_ano_semana(
    documentos: List[Document],
    uf_norm: str,
    ano_filtro: Optional[str] = None,
    semana_filtro: Optional[str] = None
) -> Optional[Document]:
    candidatos = []

    for doc in documentos:
        md = doc.metadata
        if normalizar_texto(md.get("document_type", "")) != "obitos_agravo_por_uf":
            continue

        uf_doc = str(md.get("uf_name", "")).strip().lower()
        uf_doc_alt = str(md.get("uf_normalizada", "")).strip().lower()

        if uf_doc != uf_norm and uf_doc_alt != uf_norm:
            continue

        ano_doc = str(md.get("nu_ano", md.get("ano_normalizado", md.get("ano", "")))).strip()
        semana_doc = str(md.get("semana", md.get("semana_normalizada", ""))).strip().zfill(2)

        if ano_filtro is not None and str(ano_filtro).strip() != ano_doc:
            continue

        if semana_filtro is not None and str(semana_filtro).strip().zfill(2) != semana_doc:
            continue

        candidatos.append(doc)

    if not candidatos:
        return None

    return candidatos[0]

# ============================================================
# 11. FONTES E DATAFRAMES
# ============================================================

def resumir_fontes_deterministicas(fontes: List[Document]) -> Dict:
    if not fontes:
        return {}

    document_types = sorted(set(str(doc.metadata.get("document_type", "")) for doc in fontes))
    anos = sorted(set(str(doc.metadata.get("nu_ano", doc.metadata.get("ano_normalizado", ""))) for doc in fontes if str(doc.metadata.get("nu_ano", doc.metadata.get("ano_normalizado", ""))).strip()))
    semanas = sorted(set(str(doc.metadata.get("semana_epidemiologica_range", "")) for doc in fontes if str(doc.metadata.get("semana_epidemiologica_range", "")).strip()))
    arquivos = sorted(set(str(doc.metadata.get("arquivo_origem", "")) for doc in fontes))
    categorias = sorted(set(str(doc.metadata.get("categoria", "")) for doc in fontes))
    granularidades = sorted(set(str(doc.metadata.get("granularidade", "")) for doc in fontes))

    ufs_unicas = len(set(str(doc.metadata.get("uf_name", "")) for doc in fontes if str(doc.metadata.get("uf_name", "")).strip()))

    return {
        "document_type": document_types[0] if len(document_types) == 1 else document_types,
        "nu_ano": anos[0] if len(anos) == 1 else anos,
        "semana_epidemiologica_range": semanas[0] if len(semanas) == 1 else semanas,
        "arquivo_origem": arquivos[0] if len(arquivos) == 1 else arquivos,
        "categoria": categorias[0] if len(categorias) == 1 else categorias,
        "granularidade": granularidades[0] if len(granularidades) == 1 else granularidades,
        "quantidade_registros_utilizados": len(fontes),
        "ufs_consideradas": ufs_unicas
    }


def fontes_para_dataframe_total_casos(fontes: List[Document]) -> pd.DataFrame:
    registros = []
    for doc in fontes:
        registros.append({
            "UF": doc.metadata.get("uf_name"),
            "Total de Casos": numero_seguro(doc.metadata.get("total_casos")),
            "Ano": doc.metadata.get("nu_ano", doc.metadata.get("ano_normalizado")),
            "Faixa Epidemiológica": doc.metadata.get("semana_epidemiologica_range"),
        })

    df = pd.DataFrame(registros)
    if not df.empty:
        df = df.sort_values("Total de Casos", ascending=False).reset_index(drop=True)
    return df

def fontes_para_dataframe_hospitalizacao(fontes: List[Document]) -> pd.DataFrame:
    registros = []
    for doc in fontes:
        md = doc.metadata

        internacoes = numero_seguro(
            md.get("internacoes", md.get("sim", md.get("hospitalizacoes", md.get("total_internacoes", 0)))),
            0
        )

        total_casos = numero_seguro(
            md.get("total_casos", md.get("num_cases_normalizado", md.get("total_cases", 0))),
            0
        )

        taxa = md.get(
            "taxa_internacao",
            md.get("taxa_internacao_bruta", md.get("taxa_hospitalizacao", 0))
        )
        try:
            taxa = float(str(taxa).replace(",", "."))
        except Exception:
            taxa = 0.0

        registros.append({
            "UF": md.get("uf_name", md.get("uf_normalizada", md.get("uf"))),
            "Internações": internacoes,
            "Total de Casos": total_casos,
            "Taxa de Internação (%)": taxa,
            "Ano": md.get("nu_ano", md.get("ano_normalizado", md.get("ano"))),
            "Faixa Epidemiológica": md.get("semana_epidemiologica_range", ""),
            "Semana Final": md.get("semana_final", md.get("semana", md.get("semana_normalizada", "")))
        })

    df = pd.DataFrame(registros)
    if not df.empty:
        df = df.sort_values(
            ['Taxa de Internação (%)', 'Internações'],
            ascending=[False, False]
        ).reset_index(drop=True)
    return df

def fontes_para_dataframe_sorotipos(fontes: List[Document]) -> pd.DataFrame:
    registros = []
    for doc in fontes:
        registros.append({
            "UF": doc.metadata.get("uf_name"),
            "Sorotipo": doc.metadata.get("sorotipo", doc.metadata.get("sorotipo_normalizado")),
            "Total de Casos": numero_seguro(doc.metadata.get("total_casos", doc.metadata.get("num_cases_normalizado", 0))),
            "Ano": doc.metadata.get("nu_ano", doc.metadata.get("ano_normalizado")),
            "Faixa Epidemiológica": doc.metadata.get("semana_epidemiologica_range"),
        })

    df = pd.DataFrame(registros)
    if not df.empty:
        df = df.sort_values([
            'UF',
            'Total de Casos'
        ], ascending=[
            True,
            False
        ]).reset_index(drop=True)
    return df


def fontes_para_dataframe_sintomas(fontes: List[Document]) -> pd.DataFrame:
    registros = []
    for doc in fontes:
        registros.append({
            "UF": doc.metadata.get("uf_name"),
            "Total de Casos": numero_seguro(doc.metadata.get("total_cases", doc.metadata.get("num_cases_normalizado", 0))),
            "Sintoma Predominante": doc.metadata.get("principal_symptom"),
            "Qtd Sintoma Predominante": numero_seguro(doc.metadata.get("principal_symptom_count", 0)),
            "Ano": doc.metadata.get("nu_ano", doc.metadata.get("ano_normalizado")),
            "Faixa Epidemiológica": doc.metadata.get("semana_epidemiologica_range"),
        })

    df = pd.DataFrame(registros)
    if not df.empty:
        df = df.sort_values("Total de Casos", ascending=False).reset_index(drop=True)
    return df


# ============================================================
# 12. RECUPERAÇÃO INTELIGENTE (FLUXO GERAL)
# ============================================================

def recuperar_documentos_inteligentes(pergunta: str, vectorstore, documentos_base: List[Document]) -> List[Document]:
    query_info = classificar_pergunta(pergunta)
    pesos = definir_pesos(query_info)

    documentos_recuperados = vectorstore.similarity_search(pergunta, k=24)

    if DEBUG:
        st.write("Quantidade recuperada no similarity_search:", len(documentos_recuperados))
        for i, doc in enumerate(documentos_recuperados, 1):
            st.write(
                f"RAW {i}. arquivo={doc.metadata.get('arquivo_origem')} | "
                f"document_type={doc.metadata.get('document_type')} | "
                f"categoria={doc.metadata.get('categoria')}"
            )

    documentos_filtrados = filtrar_por_document_type(documentos_recuperados, query_info, pergunta)

    if len(documentos_filtrados) < 5:
        documentos_filtrados = documentos_recuperados

    docs_com_score = []
    for doc in documentos_filtrados:
        score, justificativas = score_heuristico(pergunta, doc, query_info, pesos)
        doc.metadata["score_heuristico"] = round(score, 4)
        doc.metadata["justificativa_heuristica"] = justificativas
        docs_com_score.append((score, doc))

    docs_ordenados = sorted(docs_com_score, key=lambda x: x[0], reverse=True)
    return [doc for _, doc in docs_ordenados[:10]]


# ============================================================
# 13. RESPOSTA FINAL
# ============================================================

def responder_pergunta(pergunta: str, vectorstore, documentos):
    query_info = classificar_pergunta(pergunta)

    if eh_pergunta_hospitalizacao_por_uf(pergunta, query_info):
        if DEBUG:
            st.subheader("Depuração da recuperação")
            st.write("Classificação da pergunta:", query_info)
            st.write("Fluxo determinístico ativado para hospitalização por UF.")
        return responder_hospitalizacao_por_uf(documentos, pergunta)

    if eh_pergunta_total_casos_por_uf(pergunta, query_info):
        if DEBUG:
            st.subheader("Depuração da recuperação")
            st.write("Classificação da pergunta:", query_info)
            st.write("Fluxo determinístico ativado para total de casos por UF.")
        return responder_total_casos_por_uf(documentos, pergunta)

    if eh_pergunta_sorotipos_por_uf(pergunta, query_info):
        if DEBUG:
            st.subheader("Depuração da recuperação")
            st.write("Classificação da pergunta:", query_info)
            st.write("Fluxo determinístico ativado para sorotipos por UF.")
        return responder_sorotipos_por_uf(documentos, pergunta)

    if eh_pergunta_sintomas_por_estado(pergunta, query_info):
        if DEBUG:
            st.subheader("Depuração da recuperação")
            st.write("Classificação da pergunta:", query_info)
            st.write("Fluxo determinístico ativado para sintomas por estado.")
        return responder_sintomas_por_estado(documentos, pergunta)

    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0
    )

    candidatos = recuperar_documentos_inteligentes(pergunta, vectorstore, documentos)

    if DEBUG:
        st.subheader("Depuração da recuperação")
        st.write("Classificação da pergunta:", query_info)

        for i, doc in enumerate(candidatos, 1):
            st.write(
                f"{i}. arquivo={doc.metadata.get('arquivo_origem')} | "
                f"document_type={doc.metadata.get('document_type')} | "
                f"categoria={doc.metadata.get('categoria')} | "
                f"score_heuristico={doc.metadata.get('score_heuristico')}"
            )

    rerankeados = rerank_com_llm(pergunta, candidatos, llm)
    contexto_final = rerankeados[:4]

    contexto_texto = "\n\n".join([
        f"Texto: {doc.page_content}\nMetadados: {json.dumps(doc.metadata, ensure_ascii=False)}"
        for doc in contexto_final
    ])

    prompt_final = f"""
Você é um agente especialista em vigilância de arboviroses.

Responda apenas com base no contexto abaixo.
Se a pergunta não puder ser respondida com segurança a partir do contexto, diga isso explicitamente.

Contexto:
{contexto_texto}

Pergunta:
{pergunta}

Instruções:
- Responda em português do Brasil.
- Seja claro, objetivo e tecnicamente correto.
- Quando houver evidência no contexto, mencione UF, semana, ano, sorotipo, sintomas ou total de casos.
- Não invente dados fora do contexto.
- Se houver ambiguidade, explique.
- Se a pergunta pedir comparação, síntese ou interpretação, organize a resposta de forma analítica.
"""

    resposta = llm.invoke(prompt_final)
    return resposta.content, contexto_final, "llm"


# ============================================================
# 14. GRAFOS
# ============================================================

def listar_ufs_unicas(fontes: List[Document]) -> List[str]:
    return sorted({
        str(doc.metadata.get("uf_name", "")).strip()
        for doc in fontes
        if str(doc.metadata.get("uf_name", "")).strip()
    })


def construir_grafo_egocentrico(fontes: List[Document], modo_resposta: str, uf_focal: str) -> nx.DiGraph:
    G = nx.DiGraph()

    if not uf_focal:
        return G

    docs_uf = [
        doc for doc in fontes
        if str(doc.metadata.get("uf_name", "")).strip().lower() == uf_focal.strip().lower()
    ]

    if not docs_uf:
        return G

    uf_label = uf_focal
    G.add_node(uf_label, tipo="UF", title=uf_label)

    if modo_resposta == "deterministico_casos":
        doc = docs_uf[0]
        total = numero_seguro(doc.metadata.get("total_casos", 0))
        ano = str(doc.metadata.get("nu_ano", doc.metadata.get("ano_normalizado", "")))
        faixa = str(doc.metadata.get("semana_epidemiologica_range", ""))

        nodo_total = f"{total} casos"
        G.add_node(nodo_total, tipo="Indicador", title=f"Total de casos: {total}")
        G.add_edge(uf_label, nodo_total, relacao="temTotalCasos")

        if ano:
            nodo_ano = f"Ano {ano}"
            G.add_node(nodo_ano, tipo="Tempo", title=nodo_ano)
            G.add_edge(uf_label, nodo_ano, relacao="referenteAoAno")

        if faixa:
            nodo_faixa = f"SE {faixa}"
            G.add_node(nodo_faixa, tipo="Tempo", title=nodo_faixa)
            G.add_edge(uf_label, nodo_faixa, relacao="ocorreNaSemanaEpidemiologica")

        return G

    if modo_resposta == "deterministico_sorotipos":
        for doc in docs_uf:
            sorotipo = str(doc.metadata.get("sorotipo", doc.metadata.get("sorotipo_normalizado", "Sorotipo")))
            total = numero_seguro(doc.metadata.get("total_casos", doc.metadata.get("num_cases_normalizado", 0)))
            G.add_node(sorotipo, tipo="Sorotipo", title=sorotipo)
            G.add_edge(uf_label, sorotipo, relacao=f"temRegistroDeSorotipo ({total})")

        ano = str(docs_uf[0].metadata.get("nu_ano", docs_uf[0].metadata.get("ano_normalizado", "")))
        if ano:
            nodo_ano = f"Ano {ano}"
            G.add_node(nodo_ano, tipo="Tempo", title=nodo_ano)
            G.add_edge(uf_label, nodo_ano, relacao="referenteAoAno")

        return G

    if modo_resposta == "deterministico_sintomas":
        doc = docs_uf[0]

        principal = str(doc.metadata.get("principal_symptom", "Sintoma"))
        qtd_principal = numero_seguro(doc.metadata.get("principal_symptom_count", 0))
        total_cases = numero_seguro(doc.metadata.get("total_cases", doc.metadata.get("num_cases_normalizado", 0)))
        ano = str(doc.metadata.get("nu_ano", doc.metadata.get("ano_normalizado", "")))

        G.add_node(principal, tipo="Sintoma", title=principal)
        G.add_edge(uf_label, principal, relacao=f"temSintomaPredominante ({qtd_principal})")

        nodo_total = f"{total_cases} casos"
        G.add_node(nodo_total, tipo="Indicador", title=nodo_total)
        G.add_edge(uf_label, nodo_total, relacao="temTotalCasos")

        if ano:
            nodo_ano = f"Ano {ano}"
            G.add_node(nodo_ano, tipo="Tempo", title=nodo_ano)
            G.add_edge(uf_label, nodo_ano, relacao="referenteAoAno")

        return G

    if modo_resposta == "deterministico_hospitalizacao":
        doc = docs_uf[0]
        md = doc.metadata

        internacoes = numero_seguro(
            md.get("internacoes", md.get("sim", md.get("hospitalizacoes", md.get("total_internacoes", 0)))),
            0
        )
        total_casos = numero_seguro(
            md.get("total_casos", md.get("total_cases", md.get("num_cases_normalizado", 0))),
            0
        )

        taxa = md.get(
            "taxa_internacao",
            md.get("taxa_internacao_bruta", md.get("taxa_hospitalizacao", 0))
        )
        try:
            taxa = float(str(taxa).replace(",", "."))
        except Exception:
            taxa = 0.0

        ano = str(md.get("nu_ano", md.get("ano_normalizado", md.get("ano", ""))))
        faixa = str(md.get("semana_epidemiologica_range", "")).strip()

        nodo_hospitalizacao = f"hospitalizacao_{slug_no(uf_label)}"
        G.add_node(
            nodo_hospitalizacao,
            tipo="Categoria",
            title="Hospitalização",
            label="Hospitalização"
        )
        G.add_edge(uf_label, nodo_hospitalizacao, relacao="temHospitalizacao")

        nodo_internacoes = f"{internacoes} internações"
        G.add_node(nodo_internacoes, tipo="Indicador", title=f"Internações: {internacoes}")
        G.add_edge(nodo_hospitalizacao, nodo_internacoes, relacao="temInternacoes")

        nodo_taxa = f"{taxa:.2f}% taxa"
        G.add_node(nodo_taxa, tipo="Indicador", title=f"Taxa de internação: {taxa:.2f}%")
        G.add_edge(nodo_hospitalizacao, nodo_taxa, relacao="temTaxaDeInternacao")

        nodo_total = f"{total_casos} casos"
        G.add_node(nodo_total, tipo="Indicador", title=f"Total de casos: {total_casos}")
        G.add_edge(nodo_hospitalizacao, nodo_total, relacao="temTotalCasos")

        if ano:
            nodo_ano = f"Ano {ano}"
            G.add_node(nodo_ano, tipo="Tempo", title=nodo_ano)
            G.add_edge(nodo_hospitalizacao, nodo_ano, relacao="referenteAoAno")

        if faixa:
            nodo_faixa = f"SE {faixa}"
            G.add_node(nodo_faixa, tipo="Tempo", title=f"Faixa epidemiológica {faixa}")
            G.add_edge(nodo_hospitalizacao, nodo_faixa, relacao="temFaixaEpidemiologica")

        return G

    return G


def construir_grafo_combinado_por_uf(
    documentos: List[Document],
    uf_focal: str,
    ano_filtro: Optional[str] = None,
    incluir_casos: bool = True,
    incluir_sintomas: bool = True,
    incluir_sorotipos: bool = True,
    incluir_hospitalizacao: bool = True
) -> nx.DiGraph:
    G = nx.DiGraph()

    if not uf_focal:
        return G

    uf_norm = uf_focal.strip().lower()
    uf_slug = slug_no(uf_focal)

    G.add_node(uf_focal, label=uf_focal, tipo="UF", title=uf_focal)

    total_casos_uf = 0

    if incluir_casos:
        docs_casos = [
            doc for doc in documentos
            if str(doc.metadata.get("uf_name", "")).strip().lower() == uf_norm
            and normalizar_texto(doc.metadata.get("document_type", "")) == "total_casos_uf_ano"
        ]
        if ano_filtro:
            docs_casos = [
                doc for doc in docs_casos
                if str(doc.metadata.get("nu_ano", doc.metadata.get("ano_normalizado", ""))) == str(ano_filtro)
            ]

        if docs_casos:
            doc = docs_casos[0]
            total_casos_uf = numero_seguro(doc.metadata.get("total_casos", 0))
            ano = str(doc.metadata.get("nu_ano", doc.metadata.get("ano_normalizado", "")))
            faixa = str(doc.metadata.get("semana_epidemiologica_range", ""))

            nodo_total = f"indicador_total_{uf_slug}"
            G.add_node(
                nodo_total,
                label=f"{total_casos_uf} casos",
                tipo="Indicador",
                title=f"Total de casos: {total_casos_uf}"
            )
            G.add_edge(uf_focal, nodo_total, relacao="temTotalCasos")

            if ano:
                nodo_ano = f"ano_{uf_slug}_{ano}"
                G.add_node(nodo_ano, label=f"Ano {ano}", tipo="Tempo", title=f"Ano {ano}")
                G.add_edge(uf_focal, nodo_ano, relacao="referenteAoAno")

            if faixa:
                nodo_faixa = f"faixa_{uf_slug}_{slug_no(faixa)}"
                G.add_node(
                    nodo_faixa,
                    label=f"SE {faixa}",
                    tipo="Tempo",
                    title=f"Semana epidemiológica {faixa}"
                )
                G.add_edge(uf_focal, nodo_faixa, relacao="temFaixaEpidemiologica")

    if incluir_sintomas:
        docs_sintomas = [
            doc for doc in documentos
            if str(doc.metadata.get("uf_name", "")).strip().lower() == uf_norm
            and normalizar_texto(doc.metadata.get("document_type", "")) == "symptoms_by_state"
        ]
        if ano_filtro:
            docs_sintomas = [
                doc for doc in docs_sintomas
                if str(doc.metadata.get("nu_ano", doc.metadata.get("ano_normalizado", ""))) == str(ano_filtro)
            ]

        if docs_sintomas:
            doc = docs_sintomas[0]
            sintomas = extrair_sintomas_do_documento(doc)

            if sintomas:
                sintoma_pred, _ = sintomas[0]

                adicionar_no_expandivel(
                    G=G,
                    no_pai=uf_focal,
                    grupo_id=f"grupo_sintomas_{uf_slug}",
                    grupo_label="Sintomas",
                    relacao_pai_grupo="temSintoma",
                    itens=sintomas,
                    tipo_item="Sintoma",
                    prefixo_item=f"sintoma_{uf_slug}",
                    item_destaque=sintoma_pred,
                    rotulo_destaque="Predominante",
                    total_referencia=total_casos_uf if total_casos_uf > 0 else None,
                    base_descricao="dos casos"
                )

    if incluir_sorotipos:
        docs_sorotipos = [
            doc for doc in documentos
            if str(doc.metadata.get("uf_name", "")).strip().lower() == uf_norm
            and normalizar_texto(doc.metadata.get("document_type", "")) in [
                'sorotipos_por_uf_com_codigos_e_nomes',
                'sorotipos_by_state'
            ]
        ]
        if ano_filtro:
            docs_sorotipos = [
                doc for doc in docs_sorotipos
                if str(doc.metadata.get("nu_ano", doc.metadata.get("ano_normalizado", ""))) == str(ano_filtro)
            ]

        sorotipos = []
        for doc in docs_sorotipos:
            sorotipo = str(doc.metadata.get("sorotipo", doc.metadata.get("sorotipo_normalizado", "Sorotipo"))).strip()
            total = numero_seguro(doc.metadata.get("total_casos", doc.metadata.get("num_cases_normalizado", 0)))
            if sorotipo:
                sorotipos.append((sorotipo, total))

        sorotipos_unicos = consolidar_itens_por_maior_valor(sorotipos)
        total_sorotipos_informados = sum(qtd for _, qtd in sorotipos_unicos)
        sem_sorotipo = 0
        perc_incompletude = None
        perc_cobertura = None

        if total_casos_uf > 0:
            sem_sorotipo = max(total_casos_uf - total_sorotipos_informados, 0)
            perc_incompletude = (sem_sorotipo / total_casos_uf) * 100.0
            perc_cobertura = (total_sorotipos_informados / total_casos_uf) * 100.0

        if sorotipos_unicos:
            sorotipo_pred, _ = sorotipos_unicos[0]

            adicionar_no_expandivel(
                G=G,
                no_pai=uf_focal,
                grupo_id=f"grupo_sorotipos_{uf_slug}",
                grupo_label="Sorotipos",
                relacao_pai_grupo="temRegistroDeSorotipo",
                itens=sorotipos_unicos,
                tipo_item="Sorotipo",
                prefixo_item=f"sorotipo_{uf_slug}",
                item_destaque=sorotipo_pred,
                rotulo_destaque="Predominante",
                total_referencia=total_sorotipos_informados if total_sorotipos_informados > 0 else None,
                uf=uf_focal,
                ano=ano_filtro,
                arquivo_origem="sorotipos_por_uf_com_codigos_e_nomes / sorotipos_by_state",
                base_descricao="dos casos com sorotipo informado",
                extras_grupo={
                    "qtd": total_sorotipos_informados,
                    "percentual": round(perc_cobertura, 2) if perc_cobertura is not None else None,
                    "total_casos_uf": total_casos_uf,
                    "sem_sorotipo": sem_sorotipo,
                    "perc_incompletude": round(perc_incompletude, 2) if perc_incompletude is not None else None,
                    "descricao_base": "Percentuais dos sorotipos calculados sobre os casos com sorotipo informado."
                }
            )
        

            if incluir_hospitalizacao:
                docs_hospitalizacao = [
                    doc for doc in documentos
                    if str(doc.metadata.get("uf_name", "")).strip().lower() == uf_norm
                    and normalizar_texto(doc.metadata.get("document_type", "")) in [
                        "hospitalizacao_uf",
                        "hospitalizacao_por_uf_semana",
                        "hospitalizacao_por_uf_periodo"
                    ]
                ]

                if ano_filtro:
                    docs_hospitalizacao = [
                        doc for doc in docs_hospitalizacao
                        if str(doc.metadata.get("nu_ano", doc.metadata.get("ano_normalizado", doc.metadata.get("ano", "")))) == str(ano_filtro)
                    ]

                if docs_hospitalizacao:
                    doc_hosp = docs_hospitalizacao[0]
                    md = doc_hosp.metadata

                    internacoes = numero_seguro(
                        md.get("internacoes", md.get("sim", 0)),
                        0
                    )
                    nao_hospitalizados = numero_seguro(
                        md.get("nao_hospitalizados", md.get("nao", 0)),
                        0
                    )
                    ignorado = numero_seguro(md.get("ignorado", 0), 0)
                    nao_informado = numero_seguro(md.get("nao_informado", 0), 0)
                    total_casos_hosp = numero_seguro(
                        md.get("total_casos", md.get("total_cases", md.get("num_cases_normalizado", 0))),
                        0
                    )

                    taxa_bruta = md.get("taxa_internacao_bruta", md.get("taxa_internacao", 0))
                    taxa_valida = md.get("taxa_internacao_valida", 0)

                    try:
                        taxa_bruta = float(str(taxa_bruta).replace(",", "."))
                    except Exception:
                        taxa_bruta = 0.0

                    try:
                        taxa_valida = float(str(taxa_valida).replace(",", "."))
                    except Exception:
                        taxa_valida = 0.0

                    ano_hosp = str(md.get("nu_ano", md.get("ano_normalizado", md.get("ano", "")))).strip()
                    faixa_hosp = str(md.get("semana_epidemiologica_range", "")).strip()
                    semana_final_hosp = str(md.get("semana_final", md.get("semana", md.get("semana_normalizada", "")))).strip().zfill(2)

                    # Busca opcional de óbito pela semana final do período
                    doc_obito = buscar_obitos_por_uf_ano_semana(
                        documentos=documentos,
                        uf_norm=uf_norm,
                        ano_filtro=ano_hosp if ano_hosp else None,
                        semana_filtro=semana_final_hosp if semana_final_hosp else None
                    )

                    obitos_agravo = 0
                    if doc_obito is not None:
                        obitos_agravo = numero_seguro(
                            doc_obito.metadata.get("obitos_agravo", doc_obito.metadata.get("obitos", 0)),
                            0
                        )

                    grupo_hospitalizacao_id = f"grupo_hospitalizacao_{uf_slug}"

                    detalhes_grupo = f"""
                    <h3>Hospitalização</h3>
                    <p><b>UF:</b> {uf_focal}</p>
                    <p><b>Ano:</b> {ano_hosp or "-"}</p>
                    <p><b>Faixa epidemiológica:</b> {faixa_hosp or "-"}</p>
                    <p><b>Semana final:</b> {semana_final_hosp or "-"}</p>
                    <p><b>Total de casos:</b> {total_casos_hosp}</p>
                    <p><b>Taxa bruta de internação:</b> {taxa_bruta:.2f}%</p>
                    <p><b>Taxa válida de internação:</b> {taxa_valida:.2f}%</p>
                    <p><b>Óbitos por agravo:</b> {obitos_agravo}</p>
                    """

                    G.add_node(
                        grupo_hospitalizacao_id,
                        label="Hospitalização",
                        tipo="Grupo",
                        title="Clique para expandir/recolher hospitalização",
                        grupo_expandivel=True,
                        expandido=False,
                        descricao="Grupo semântico de hospitalização",
                        uf=uf_focal,
                        ano=ano_hosp,
                        faixa_epidemiologica=faixa_hosp,
                        semana_final=semana_final_hosp,
                        detalhes_html=detalhes_grupo
                    )
                    G.add_edge(uf_focal, grupo_hospitalizacao_id, relacao="temHospitalizacao")

                    itens_hospitalizacao = [
                        ("Internados", internacoes),
                        ("Não hospitalizados", nao_hospitalizados),
                        ("Ignorados", ignorado),
                        ("Não informados", nao_informado),
                        ("Óbitos por agravo", obitos_agravo)
                    ]

                    for nome_item, valor_item in itens_hospitalizacao:
                        item_id = f"{grupo_hospitalizacao_id}_{slug_no(nome_item)}"
                        size, percentual = calcular_tamanho_por_percentual(
                            valor_item,
                            total_casos_hosp if total_casos_hosp > 0 else None
                        )

                        titulo = f"{nome_item}: {valor_item}"
                        if total_casos_hosp > 0:
                            titulo += f" — {percentual:.1f}% do total ({valor_item}/{total_casos_hosp})"

                        detalhes_item = f"""
                        <h3>{nome_item}</h3>
                        <p><b>Quantidade:</b> {valor_item}</p>
                        <p><b>UF:</b> {uf_focal}</p>
                        <p><b>Ano:</b> {ano_hosp or "-"}</p>
                        <p><b>Faixa epidemiológica:</b> {faixa_hosp or "-"}</p>
                        <p><b>Total de casos:</b> {total_casos_hosp}</p>
                        """

                        G.add_node(
                            item_id,
                            label=f"{nome_item} ({valor_item})",
                            tipo="IndicadorHospitalizacao",
                            title=titulo,
                            hidden=True,
                            parent_group=grupo_hospitalizacao_id,
                            size=size,
                            percentual=round(percentual, 2),
                            qtd=valor_item,
                            uf=uf_focal,
                            ano=ano_hosp,
                            faixa_epidemiologica=faixa_hosp,
                            descricao=f"{nome_item} associado à hospitalização em {uf_focal}",
                            detalhes_html=detalhes_item
                        )

                        G.add_edge(
                            grupo_hospitalizacao_id,
                            item_id,
                            relacao="contém",
                            hidden=True,
                            parent_group=grupo_hospitalizacao_id
                        )

                    taxa_bruta_id = f"{grupo_hospitalizacao_id}_taxa_bruta"
                    G.add_node(
                        taxa_bruta_id,
                        label=f"Taxa bruta ({taxa_bruta:.2f}%)",
                        tipo="IndicadorHospitalizacao",
                        title=f"Taxa bruta de internação: {taxa_bruta:.2f}%",
                        hidden=True,
                        parent_group=grupo_hospitalizacao_id,
                        size=18,
                        qtd=taxa_bruta,
                        uf=uf_focal,
                        ano=ano_hosp,
                        faixa_epidemiologica=faixa_hosp,
                        descricao="Taxa bruta de internação",
                        detalhes_html=f"""
                        <h3>Taxa bruta de internação</h3>
                        <p><b>Valor:</b> {taxa_bruta:.2f}%</p>
                        <p><b>UF:</b> {uf_focal}</p>
                        <p><b>Ano:</b> {ano_hosp or "-"}</p>
                        <p><b>Faixa epidemiológica:</b> {faixa_hosp or "-"}</p>
                        """
                    )
                    G.add_edge(
                        grupo_hospitalizacao_id,
                        taxa_bruta_id,
                        relacao="contém",
                        hidden=True,
                        parent_group=grupo_hospitalizacao_id
                    )

                    taxa_valida_id = f"{grupo_hospitalizacao_id}_taxa_valida"
                    G.add_node(
                        taxa_valida_id,
                        label=f"Taxa válida ({taxa_valida:.2f}%)",
                        tipo="IndicadorHospitalizacao",
                        title=f"Taxa válida de internação: {taxa_valida:.2f}%",
                        hidden=True,
                        parent_group=grupo_hospitalizacao_id,
                        size=18,
                        qtd=taxa_valida,
                        uf=uf_focal,
                        ano=ano_hosp,
                        faixa_epidemiologica=faixa_hosp,
                        descricao="Taxa válida de internação",
                        detalhes_html=f"""
                        <h3>Taxa válida de internação</h3>
                        <p><b>Valor:</b> {taxa_valida:.2f}%</p>
                        <p><b>UF:</b> {uf_focal}</p>
                        <p><b>Ano:</b> {ano_hosp or "-"}</p>
                        <p><b>Faixa epidemiológica:</b> {faixa_hosp or "-"}</p>
                        """
                    )
                    G.add_edge(
                        grupo_hospitalizacao_id,
                        taxa_valida_id,
                        relacao="contém",
                        hidden=True,
                        parent_group=grupo_hospitalizacao_id
                    )

    return G


def renderizar_grafo_pyvis(G: nx.DiGraph, titulo: str):
    if G.number_of_nodes() == 0:
        st.warning("Não há dados suficientes para gerar o grafo.")
        return

    net = Network(
        height="650px",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="black"
    )
    net.from_nx(G)

    # =========================
    # Estilo dos nós
    # =========================
    for node in net.nodes:
        tipo = node.get("tipo", "")
        tamanho_original = node.get("size", None)

        if tipo == "UF":
            node["color"] = "#4F81BD"
            node["shape"] = "dot"
            node["size"] = 28

        elif tipo == "Grupo":
            node["color"] = "#D9EAD3"
            node["shape"] = "box"
            node["size"] = 22

        elif tipo == "Sorotipo":
            node["color"] = {
                'background': '#9BBB59',
                'border': '#6AA84F'
            }
            node["shape"] = "dot"
            node["size"] = tamanho_original if tamanho_original is not None else 18
            node["borderWidth"] = 1

        elif tipo == "SorotipoDestaque":
            node["color"] = {
                'background': '#9BBB59',
                'border': '#274E13'
            }
            node["shape"] = "dot"
            node["size"] = max((tamanho_original if tamanho_original is not None else 18), 24)
            node["borderWidth"] = 4

        elif tipo == "Sintoma":
            node["color"] = {
                'background': '#C0504D',
                'border': '#A61C00'
            }
            node["shape"] = "dot"
            node["size"] = tamanho_original if tamanho_original is not None else 18
            node["borderWidth"] = 1

        elif tipo == "SintomaDestaque":
            node["color"] = {
                'background': '#C0504D',
                'border': '#660000'
            }
            node["shape"] = "dot"
            node["size"] = max((tamanho_original if tamanho_original is not None else 18), 24)
            node["borderWidth"] = 4

        elif tipo == "Indicador":
            node["color"] = "#8064A2"
            node["shape"] = "box"
            node["size"] = 18

        elif tipo == "Tempo":
            node["color"] = "#F79646"
            node["shape"] = "ellipse"
            node["size"] = 16

        node["label"] = node.get("label", node.get("id"))

    # =========================
    # Estilo das arestas
    # =========================
    for edge in net.edges:
        relacao = edge.get("relacao", "")
        edge["label"] = relacao
        edge["title"] = relacao
        edge["arrows"] = "to"

    net.repulsion(
        node_distance=180,
        central_gravity=0.12,
        spring_length=170,
        spring_strength=0.05
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        net.save_graph(tmp_file.name)
        html = Path(tmp_file.name).read_text(encoding="utf-8")

    # =========================
    # Script: expansão/recolhimento + painel lateral
    # =========================
    script_interativo = """
    <script type="text/javascript">
    (function() {
        function escapeHtml(valor) {
            if (valor === null || valor === undefined) return "-";
            return String(valor)
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#039;");
        }

        function montarPainel(nodeData) {
            const panel = document.getElementById("node-info-panel");
            if (!panel) return;

            if (!nodeData) {
                panel.innerHTML = `
                    <h3 style="margin-top:0;">Detalhes do nó</h3>
                    <p>Clique em um nó para visualizar suas informações.</p>
                `;
                return;
            }

            const id = nodeData.id ?? "-";
            const label = nodeData.label ?? "-";
            const tipo = nodeData.tipo ?? "-";
            const title = nodeData.title ?? "-";
            const qtd = nodeData.qtd ?? "-";
            const percentual = nodeData.percentual ?? "-";
            const parentGroup = nodeData.parent_group ?? "-";
            const expandivel = nodeData.grupo_expandivel ? "Sim" : "Não";
            const expandido = nodeData.expandido ? "Sim" : "Não";
            const hidden = nodeData.hidden ? "Sim" : "Não";

            const totalCasosUf = nodeData.total_casos_uf ?? "-";
            const semSorotipo = nodeData.sem_sorotipo ?? "-";
            const percIncompletude = nodeData.perc_incompletude ?? "-";
            const descricaoBase = nodeData.descricao_base ?? "-";
            const taxaBruta = nodeData.taxa_bruta ?? "-";
            const taxaValida = nodeData.taxa_valida ?? "-";

            panel.innerHTML = `
                <h3 style="margin-top:0; color:#1f2937;">${escapeHtml(label)}</h3>
                <div style="display:grid; grid-template-columns: 160px 1fr; gap:8px 10px; font-size:14px;">
                    <div><b>ID</b></div><div>${escapeHtml(id)}</div>
                    <div><b>Tipo</b></div><div>${escapeHtml(tipo)}</div>
                    <div><b>Descrição</b></div><div>${escapeHtml(title)}</div>
                    <div><b>Quantidade</b></div><div>${escapeHtml(qtd)}</div>
                    <div><b>Percentual</b></div><div>${escapeHtml(percentual)}</div>
                    <div><b>Grupo pai</b></div><div>${escapeHtml(parentGroup)}</div>
                    <div><b>Expandível</b></div><div>${escapeHtml(expandivel)}</div>
                    <div><b>Expandido</b></div><div>${escapeHtml(expandido)}</div>
                    <div><b>Oculto</b></div><div>${escapeHtml(hidden)}</div>
                    <div><b>Total de casos UF</b></div><div>${escapeHtml(totalCasosUf)}</div>
                    <div><b>Sem sorotipo</b></div><div>${escapeHtml(semSorotipo)}</div>
                    <div><b>Incompletude (%)</b></div><div>${escapeHtml(percIncompletude)}</div>
                    <div><b>Base semântica</b></div><div>${escapeHtml(descricaoBase)}</div>
                    <div><b>Taxa bruta (%)</b></div><div>${escapeHtml(taxaBruta)}</div>
                    <div><b>Taxa válida (%)</b></div><div>${escapeHtml(taxaValida)}</div>
                </div>
            `;
        }

        function alternarGrupo(nodeId) {
            const nodeData = nodes.get(nodeId);
            if (!nodeData || !nodeData.grupo_expandivel) return;

            const novoEstado = !nodeData.expandido;

            nodes.update({
                id: nodeId,
                expandido: novoEstado
            });

            const todosNos = nodes.get();
            const todasArestas = edges.get();

            const nosFilhos = todosNos
                .filter(n => n.parent_group === nodeId)
                .map(n => ({
                    id: n.id,
                    hidden: !novoEstado
                }));

            const arestasFilhas = todasArestas
                .filter(e => e.parent_group === nodeId)
                .map(e => ({
                    id: e.id,
                    hidden: !novoEstado
                }));

            if (nosFilhos.length > 0){
                nodes.update(nosFilhos);
            }
            if (arestasFilhas.length > 0){
                edges.update(arestasFilhas);
            }
        }

        function iniciarInteratividade() {
            if (
                typeof network === "undefined" ||
                typeof nodes === "undefined" ||
                typeof edges === "undefined"
            ) {
                setTimeout(iniciarInteratividade, 300);
                return;
            }

            montarPainel(null);

            network.on("click", function(params) {
                if (!params.nodes || params.nodes.length === 0) {
                    montarPainel(null);
                    return;
                }

                const nodeId = params.nodes[0];
                const nodeData = nodes.get(nodeId);

                if (nodeData && nodeData.grupo_expandivel) {
                    alternarGrupo(nodeId);
                }

                montarPainel(nodes.get(nodeId));
            });
        }

        setTimeout(iniciarInteratividade, 300);
    })();
    </script>
    """

    # =========================
    # CSS do layout
    # =========================
    css_layout = """
    <style>
        .grafo-wrapper {
            display: flex;
            gap: 16px;
            width: 100%;
            height: 700px;
            box-sizing: border-box;
            font-family: Arial, sans-serif;
        }

        .grafo-area {
            flex: 2.2;
            min-width: 0;
            border: 1px solid #d1d5db;
            border-radius: 10px;
            overflow: hidden;
            background: #fff;
        }

        .painel-area {
            flex: 1;
            min-width: 280px;
            border: 1px solid #d1d5db;
            border-radius: 10px;
            padding: 16px;
            overflow-y: auto;
            background: #f9fafb;
            color: #111827;
            box-sizing: border-box;
        }

        .titulo-grafo-custom {
            margin: 0 0 10px 0;
            font-family: Arial, sans-serif;
            color: #111827;
        }
    </style>
    """

    # =========================
    # Reorganiza o HTML do pyvis
    # =========================
    html = html.replace(
        '<body>',
        f'''
        <body>
        {css_layout}
        <h3 class="titulo-grafo-custom">{titulo}</h3>
        <div class="grafo-wrapper">
            <div class="grafo-area">
        '''
    )

    html = html.replace(
        '</body>',
        f'''
            </div>
            <div id="node-info-panel" class="painel-area">
                <h3 style="margin-top:0;">Detalhes do nó</h3>
                <p>Clique em um nó para visualizar suas informações.</p>
            </div>
        </div>
        {script_interativo}
        </body>
        '''
    )

    components.html(html, height=760, scrolling=True)

# ============================================================
# 15. VISUALIZAÇÃO
# ============================================================

def inferir_visualizacao(tipo_visualizacao: str, instrucao_usuario: str) -> str:
    texto = normalizar_texto(instrucao_usuario)

    if tipo_visualizacao != "Automática":
        return tipo_visualizacao

    if "top 10" in texto or "top10" in texto:
        return "Top 10"

    if "tabela" in texto:
        return "Tabela"

    if "barra" in texto or "barras" in texto or "grafico" in texto or "gráfico" in texto:
        return "Barras"

    return "Barras"


def renderizar_barras_df(df: pd.DataFrame, coluna_categoria: str, coluna_valor: str, top_n: Optional[int] = None):
    df_plot = df.copy()
    if top_n is not None:
        df_plot = df_plot.head(top_n).copy()

    grafico = alt.Chart(df_plot).mark_bar().encode(
        y=alt.Y(f"{coluna_categoria}:N", sort="-x", title=coluna_categoria),
        x=alt.X(f"{coluna_valor}:Q", title=coluna_valor),
        tooltip=list(df_plot.columns)
    ).properties(
        width=900,
        height=max(300, min(900, 30 * len(df_plot)))
    )

    st.altair_chart(grafico, use_container_width=True)
    st.dataframe(df_plot, use_container_width=True)


def renderizar_visualizacao(fontes: List[Document], modo_resposta: str, tipo_visualizacao: str, instrucao_usuario: str):
    visualizacao_escolhida = inferir_visualizacao(tipo_visualizacao, instrucao_usuario)

    if modo_resposta == "deterministico_casos":
        df = fontes_para_dataframe_total_casos(fontes)
        if df.empty:
            st.warning("Não há dados suficientes para gerar a visualização.")
            return
        df = df.sort_values("Total de Casos", ascending=False).reset_index(drop=True)

        if visualizacao_escolhida == "Tabela":
            st.dataframe(df, use_container_width=True)
            return
        if visualizacao_escolhida == "Top 10":
            renderizar_barras_df(df, "UF", "Total de Casos", top_n=10)
            return
        renderizar_barras_df(df, "UF", "Total de Casos")
        return

    if modo_resposta == "deterministico_sorotipos":
        df = fontes_para_dataframe_sorotipos(fontes)
        if df.empty:
            st.warning("Não há dados suficientes para gerar a visualização.")
            return

        if visualizacao_escolhida == "Tabela":
            st.dataframe(df, use_container_width=True)
            return

        df_plot = df.groupby("Sorotipo", as_index=False)["Total de Casos"].sum().sort_values("Total de Casos", ascending=False)
        renderizar_barras_df(df_plot, "Sorotipo", "Total de Casos")
        return

    if modo_resposta == "deterministico_hospitalizacao":
        df = fontes_para_dataframe_hospitalizacao(fontes)
        if df.empty:
            st.warning("Não há dados suficientes para gerar a visualização.")
            return

        if visualizacao_escolhida == "Tabela":
            st.dataframe(df, use_container_width=True)
            return

        texto = normalizar_texto(instrucao_usuario)
        if "taxa" in texto or "internação" in texto or "internacao" in texto or "hospitalização" in texto or "hospitalizacao" in texto:
            renderizar_barras_df(df, "UF", "Taxa de Internação (%)", top_n=10 if visualizacao_escolhida == "Top 10" else None)
        else:
            renderizar_barras_df(df, "UF", "Internações", top_n=10 if visualizacao_escolhida == "Top 10" else None)
        return

    if modo_resposta == "deterministico_hospitalizacao":
        st.dataframe(fontes_para_dataframe_hospitalizacao(fontes), use_container_width=True)

    if modo_resposta == "deterministico_sintomas":
        df = fontes_para_dataframe_sintomas(fontes)
        if df.empty:
            st.warning("Não há dados suficientes para gerar a visualização.")
            return

        if visualizacao_escolhida == "Tabela":
            st.dataframe(df, use_container_width=True)
            return

        df_plot = df[[
            'UF',
            'Qtd Sintoma Predominante'
        ]].sort_values("Qtd Sintoma Predominante", ascending=False)
        renderizar_barras_df(df_plot, "UF", "Qtd Sintoma Predominante")
        return

    st.info("A visualização em gráfico está disponível apenas para os fluxos determinísticos nesta versão.")


# ============================================================
# 16. ESTADO DA SESSÃO
# ============================================================

if "pergunta_atual" not in st.session_state:
    st.session_state.pergunta_atual = ""

if "executar_consulta" not in st.session_state:
    st.session_state.executar_consulta = False

if "ultimo_resultado" not in st.session_state:
    st.session_state.ultimo_resultado = None

if "campo_pergunta" not in st.session_state:
    st.session_state.campo_pergunta = ""

if "select_pergunta_sugerida" not in st.session_state:
    st.session_state.select_pergunta_sugerida = ""


# ============================================================
# 17. INTERFACE STREAMLIT
# ============================================================

st.set_page_config(page_title="Agente Inteligente de Arboviroses", layout="wide")

st.title("🦟 Agente Inteligente de Arboviroses")
st.markdown(
    "Consulta semântica com filtros por `document_type`, fluxos determinísticos para consultas agregadas, "
    "respostas com apoio de LLM para consultas abertas, gráficos e grafos."
)

assinatura_atual = calcular_assinatura_arquivos(DATA_DIR, ARQUIVOS_JSON)
assinatura_atual_hash = hashlib.md5(
    json.dumps(assinatura_atual, sort_keys=True).encode("utf-8")
).hexdigest()

st.markdown("### Perguntas sugeridas")
pergunta_sugerida = st.selectbox(
    "Escolha uma pergunta sugerida",
    [
        ''
    ] + PERGUNTAS_SUGERIDAS,
    index=0,
    key="select_pergunta_sugerida"
)

col_a, col_b, col_c = st.columns([
    1,
    1,
    1
])

with col_a:
    if st.button("Usar pergunta sugerida"):
        texto_sugerido = st.session_state.get("select_pergunta_sugerida", "")
        st.session_state.pergunta_atual = texto_sugerido
        st.session_state.campo_pergunta = texto_sugerido
        st.session_state.executar_consulta = False
        st.rerun()

with col_b:
    if st.button("Consultar"):
        texto_digitado = st.session_state.get("campo_pergunta", "").strip()
        st.session_state.pergunta_atual = texto_digitado
        st.session_state.executar_consulta = True
        st.rerun()

with col_c:
    if st.button("Limpar pergunta"):
        st.session_state.pergunta_atual = ""
        st.session_state.campo_pergunta = ""
        st.session_state.executar_consulta = False
        st.session_state.ultimo_resultado = None
        st.session_state.select_pergunta_sugerida = ""
        st.rerun()

st.text_input(
    "Digite sua pergunta:",
    key="campo_pergunta"
)

pergunta = st.session_state.pergunta_atual.strip()

if st.session_state.executar_consulta and pergunta:
    with st.spinner("Consultando base de arboviroses..."):
        documentos = carregar_documentos_json()

        if DEBUG:
            tipos = sorted(set(str(doc.metadata.get("document_type", "")) for doc in documentos))
            st.write("Document types encontrados:", tipos)

        vectorstore = criar_ou_carregar_vectorstore(documentos, assinatura_atual_hash)
        resposta, fontes, modo_resposta = responder_pergunta(pergunta, vectorstore, documentos)

        st.session_state.ultimo_resultado = {
            "pergunta": pergunta,
            "resposta": resposta,
            "fontes": fontes,
            "modo_resposta": modo_resposta,
            "documentos": documentos
        }

if st.session_state.ultimo_resultado:
    resultado = st.session_state.ultimo_resultado
    pergunta = resultado["pergunta"]
    resposta = resultado["resposta"]
    fontes = resultado["fontes"]
    modo_resposta = resultado["modo_resposta"]
    documentos = resultado["documentos"]

    st.subheader("Pergunta atual")
    st.write(pergunta)

    st.subheader("Resposta")
    st.write(resposta)

    st.subheader("Fontes utilizadas")

    if modo_resposta.startswith("deterministico"):
        resumo_fontes = resumir_fontes_deterministicas(fontes)
        st.json(resumo_fontes)

        with st.expander("Ver registros considerados"):
            if modo_resposta == "deterministico_casos":
                st.dataframe(fontes_para_dataframe_total_casos(fontes), use_container_width=True)
            elif modo_resposta == "deterministico_sorotipos":
                st.dataframe(fontes_para_dataframe_sorotipos(fontes), use_container_width=True)
            elif modo_resposta == "deterministico_sintomas":
                st.dataframe(fontes_para_dataframe_sintomas(fontes), use_container_width=True)

        st.subheader("Visualização dos resultados")
        tipo_visualizacao = st.selectbox(
            "Tipo de visualização",
            [
                'Automática',
                'Barras',
                'Top 10',
                'Tabela'
            ],
            index=0
        )

        instrucao_visualizacao = st.text_input(
            "Descreva como deseja visualizar (opcional)",
            placeholder="Ex.: gráfico de barras por UF, top 10 estados, tabela"
        )

        if st.button("Gerar visualização"):
            renderizar_visualizacao(
                fontes=fontes,
                modo_resposta=modo_resposta,
                tipo_visualizacao=tipo_visualizacao,
                instrucao_usuario=instrucao_visualizacao
            )

        st.subheader("Grafo da resposta atual")
        ufs_disponiveis = listar_ufs_unicas(fontes)

        if ufs_disponiveis:
            uf_focal = st.selectbox(
                "Escolha a UF para visualizar o grafo da resposta atual",
                ufs_disponiveis,
                index=0,
                key=f"uf_focal_resposta_{modo_resposta}"
            )

            if st.button("Gerar grafo da resposta atual"):
                G = construir_grafo_egocentrico(fontes, modo_resposta, uf_focal)
                renderizar_grafo_pyvis(G, f"Grafo egocêntrico de {uf_focal}")
        else:
            st.info("Nenhuma UF disponível para o grafo da resposta atual.")

        st.subheader("Grafo combinado por UF")
        todas_ufs = sorted({
            str(doc.metadata.get("uf_name", "")).strip()
            for doc in documentos
            if str(doc.metadata.get("uf_name", "")).strip()
        })

        if todas_ufs:
            col1, col2 = st.columns([
                2,
                1
            ])

            with col1:
                uf_grafo_combinado = st.selectbox(
                    "Escolha a UF para combinar casos, sintomas e sorotipos",
                    todas_ufs,
                    index=0,
                    key="uf_grafo_combinado"
                )

            anos_disponiveis = sorted({
                str(doc.metadata.get("nu_ano", doc.metadata.get("ano_normalizado", ""))).strip()
                for doc in documentos
                if str(doc.metadata.get("nu_ano", doc.metadata.get("ano_normalizado", ""))).strip()
            })

            with col2:
                ano_grafo = st.selectbox(
                    "Ano (opcional)",
                    [
                        'Todos'
                    ] + anos_disponiveis,
                    index=0,
                    key="ano_grafo_combinado"
                )

            combinacoes_grafo = st.multiselect(
                "Escolha os elementos para combinar no grafo",
                [
                    'Casos',
                    'Sintomas',
                    'Sorotipos',
                    'Hospitalização'
                ],
                default=[
                    'Casos',
                    'Sintomas',
                    'Sorotipos',
                    'Hospitalização'
                ],
                key="combinacoes_grafo"
            )

            if st.button("Gerar grafo combinado"):
                ano_filtro = None if ano_grafo == "Todos" else ano_grafo

                G = construir_grafo_combinado_por_uf(
                    documentos=documentos,
                    uf_focal=uf_grafo_combinado,
                    ano_filtro=ano_filtro,
                    incluir_casos="Casos" in combinacoes_grafo,
                    incluir_sintomas="Sintomas" in combinacoes_grafo,
                    incluir_sorotipos="Sorotipos" in combinacoes_grafo,
                    incluir_hospitalizacao="Hospitalização" in combinacoes_grafo
                )

                titulo_partes = " + ".join(combinacoes_grafo) if combinacoes_grafo else "Sem elementos"
                renderizar_grafo_pyvis(G, f"Grafo combinado de {uf_grafo_combinado} ({titulo_partes})")
        else:
            st.info("Nenhuma UF disponível para o grafo combinado.")

    else:
        for i, doc in enumerate(fontes, start=1):
            st.markdown(f"### Trecho {i}")
            st.write(f"**Arquivo origem:** {doc.metadata.get('arquivo_origem')}")
            st.write(f"**Document type:** {doc.metadata.get('document_type')}")
            st.write(f"**Categoria:** {doc.metadata.get('categoria')}")
            st.write(f"**Granularidade:** {doc.metadata.get('granularidade')}")
            st.write(f"**Score heurístico:** {doc.metadata.get('score_heuristico')}")
            st.write(f"**Score LLM:** {doc.metadata.get('score_llm')}")
            st.write("**Justificativas heurísticas:**")
            st.write(doc.metadata.get("justificativa_heuristica"))
            st.write("**Metadados completos:**")
            st.json(doc.metadata)
            st.write("**Texto:**")
            st.write(doc.page_content)
            st.divider()