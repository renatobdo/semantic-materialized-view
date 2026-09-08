# ============================================================
# LEGACY VERSION
# ============================================================
#
# Snapshot da versão 13 do protótipo.
# Mantido apenas para rastreabilidade histórica do desenvolvimento.
#
# A versão atual da aplicação encontra-se em:
#     app.py
#
# ============================================================

import os
import re
import unicodedata
import json
import tiktoken
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
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
)
from typing import List, Dict


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

# ============================================================
# CONFIGURAÇÃO DOS DIRETÓRIOS
# ============================================================

ANO = int(
    os.getenv(
        "ANO",
        "2026"
    )
)

PASTA_BASE = Path(
    os.getenv(
        "PASTA_BASE",
        r"G:\Meu Drive\testes\arbovirus_rag"
    )
)

# ------------------------------------------------------------
# Documentos semânticos produzidos pelos Notebooks 06 e 07
# ------------------------------------------------------------

PASTA_DOCS = (
    PASTA_BASE
    / "data_docs"
    / "SINAN"
    / str(ANO)
)

# ------------------------------------------------------------
# Inventário dos documentos semânticos
# ------------------------------------------------------------

PASTA_INVENTARIO = (
    PASTA_DOCS
    / "_inventario"
)

# ------------------------------------------------------------
# Panorama geral produzido pelo Notebook 07
# ------------------------------------------------------------

PASTA_PANORAMA_GERAL = (
    PASTA_DOCS
    / "panorama_geral"
)

# ------------------------------------------------------------
# Banco vetorial
# ------------------------------------------------------------

PASTA_BASE_VECTORSTORE = Path(
    os.getenv(
        "PASTA_BASE_VECTORSTORE",
        r"C:\testes\arbovirus_rag_app\data"
    )
)

PERSIST_DIRECTORY = (
    PASTA_BASE_VECTORSTORE
    / "vectorstore"
    / "SINAN"
    / str(ANO)
    / "chromadb"
)

MANIFEST_PATH = (
    PERSIST_DIRECTORY
    / "_manifest.json"
)


# G:\Meu Drive\testes\arbovirus_rag\
# │
# ├── data_analytics\
# │   └── SINAN\
# │       └── 2026\
# │
# ├── data_docs\
# │   └── SINAN\
# │       └── 2026\
# │           ├── clinico\
# │           ├── desfechos\
# │           ├── geografico\
# │           ├── temporal\
# │           ├── virologico\
# │           ├── panorama_geral\
# │           └── _inventario\
# │
# └── vectorstore\
#     └── SINAN\
#         └── 2026\
#             └── chromadb\


# ============================================================
# VALIDAÇÃO DOS DIRETÓRIOS 
# ============================================================

def validar_diretorios():
    caminhos = {
        "Pasta base": PASTA_BASE,
        "Documentos semânticos": PASTA_DOCS,
        "Inventário": PASTA_INVENTARIO,
        "Panorama geral": PASTA_PANORAMA_GERAL,
    }

    for nome, caminho in caminhos.items():

        if caminho.exists():
            print(
                f"[OK] {nome}: "
                f"{caminho}"
            )
        else:
            print(
                f"[AUSENTE] {nome}: "
                f"{caminho}"
            )


validar_diretorios()


# Alteração 1
# #
# DATA_DIR = Path(
#     os.getenv(
#         "DATA_DIR",
#         r"G:\Meu Drive\Doutorado\arbovirus_rag\data_processed\data_rag_gemini"
#     )
# )

# PERSIST_DIRECTORY = Path(
#     os.getenv(
#         "PERSIST_DIRECTORY",
#         r"G:\Meu Drive\Doutorado\arbovirus_rag\chromadb_data_v2"
#     )
# )

# MANIFEST_PATH = PERSIST_DIRECTORY / "_manifest.json"

# ARQUIVOS_JSON = [
#     'sorotipos_by_state_and_week_rag_documents.json',
#     'sorotipos_by_state_rag_documents.json',
#     'sorotipos_por_uf_com_codigos_e_nomes_rag_documents.json',
#     'symptoms_by_state_rag_documents.json',
#     'total_casos_uf_ano_rag_documents.json',
#     'hospitalizacao_uf.json',
#     'docs_obitos_agravo_por_uf_ano_semana.json'
# ]

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEBUG = os.getenv(
    "DEBUG",
    "false"
).lower() == "true"

PERGUNTAS_SUGERIDAS = [
    'Quantos casos foram registrados por UF?',
    'Quais os sorotipos da dengue por UF?',
    'Qual o sintoma predominante no Brasil?',
    #'Quais sintomas predominam por estado?',
    'Quais os 10 estados com mais casos de dengue?',
    'Mostre os casos por UF em formato de gráfico',
    'Mostre um grafo combinado de sintomas, sorotipos, hospitalizações e óbitos por UF',
    'Quais sintomas predominantes e sorotipos aparecem em cada estado?',
    'Mostre um grafo semântico da UF de São Paulo',
    'Qual a hospitalização por UF?',
    'Quais estados tiveram mais internações?',
    'Qual a taxa de internação por UF?',
    'Mostre a hospitalização por UF em formato de gráfico'
]

# ============================================================
# DIMENSOES_PERGUNTAS
# ===========================================================
DIMENSOES_PERGUNTAS = {
    "Clínica": {
        "dominios": ["clinico"],
        "perguntas": [
            "Quais são os sintomas predominantes por UF?",
            "Quais sinais clínicos aparecem com maior frequência?",
            "Como os sintomas variam entre os estados?"
        ]
    },

    "Virológica": {
        "dominios": ["virologico"],
        "perguntas": [
            "Quais os sorotipos da dengue por UF?",
            "Qual o sorotipo predominante em cada estado?",
            "Quais UFs apresentam maior diversidade de sorotipos?"
        ]
    },

    "Desfechos": {
        "dominios": ["desfechos"],
        "perguntas": [
            "Quais UFs apresentam mais hospitalizações?",
            "Como os óbitos se distribuem por estado?",
            "Quais estados apresentam maiores proporções de hospitalização?"
        ]
    },

    "Temporal": {
        "dominios": ["temporal"],
        "perguntas": [
            "Como os casos evoluíram ao longo das semanas epidemiológicas?",
            "Em quais semanas ocorreram os maiores picos?",
            "Quais UFs apresentaram aumento recente de casos?"
        ]
    },

    "Geográfica": {
        "dominios": ["geografico"],
        "perguntas": [
            "Quais UFs apresentam maior número de casos?",
            "Quais municípios apresentam maior incidência?",
            "Como os casos se distribuem geograficamente?"
        ]
    },

    "Combinada": {
        "dominios": ["clinico", "virologico", "desfechos", "temporal", "geografico"],
        "perguntas": [
            "Quais sintomas predominantes e sorotipos aparecem em cada estado?",
            "Quais UFs apresentam aumento de casos e maior hospitalização?",
            "Quais sorotipos predominam nas UFs com maior número de óbitos?"
        ]
    }
}


# ============================================================
# 2. FUNÇÕES AUXILIARES
# ============================================================


def normalizar_texto(texto: Optional[str]) -> str:
    """
    Normaliza texto para comparação lexical:
    - converte para string;
    - transforma em minúsculas;
    - remove acentos;
    - remove espaços excedentes.
    """

    if texto is None:
        return ""

    texto = str(texto).strip().lower()

    texto = unicodedata.normalize(
        "NFKD",
        texto
    )

    texto = "".join(
        caractere
        for caractere in texto
        if not unicodedata.combining(caractere)
    )

    texto = re.sub(
        r"\s+",
        " ",
        texto
    )

    return texto.strip()

# ============================================================
# detectar_dominios_pergunta
# ===========================================================

def detectar_dominios_pergunta(pergunta):

    pergunta_lower = normalizar_texto(pergunta)

    dominios = []

    termos_clinicos = [
        "sintoma",
        "sintomas",
        "sinais clinicos",
        "febre",
        "mialgia",
        "cefaleia"
    ]

    termos_virologicos = [
        "sorotipo",
        "sorotipos",
        "denv"
    ]

    termos_desfechos = [
        "hospitalizacao",
        "hospitalizacoes",
        "internacao",
        "internacoes",
        "obito",
        "obitos",
        "evolucao"
    ]

    termos_temporais = [
        "semana",
        "semanas",
        "temporal",
        "evolucao dos casos",
        "pico",
        "aumento recente",
        "reducao recente"
    ]

    # Aqui ficam somente conceitos realmente geográficos.
    # UF/estado/município isoladamente representam granularidade.
    termos_geograficos = [
        "incidencia",
        "distribuicao geografica",
        "distribuicao espacial",
        "localizacao geografica"
    ]

    if any(
        termo in pergunta_lower
        for termo in termos_clinicos
    ):
        dominios.append("clinico")

    if any(
        termo in pergunta_lower
        for termo in termos_virologicos
    ):
        dominios.append("virologico")

    if any(
        termo in pergunta_lower
        for termo in termos_desfechos
    ):
        dominios.append("desfechos")

    if any(
        termo in pergunta_lower
        for termo in termos_temporais
    ):
        dominios.append("temporal")

    if any(
        termo in pergunta_lower
        for termo in termos_geograficos
    ):
        dominios.append("geografico")

    return dominios

# ============================================================
# Aqui existe um detalhe importante: “por UF” não deveria necessariamente fazer o sistema recuperar documentos geográficos.
# UF nesse caso pode ser apenas a granularidade.
# Então eu refinaria a função e separaria:
# domínio ≠ granularidade
# Por exemplo:
# Quais sintomas predominantes e sorotipos aparecem em cada estado?
# tem:
# dominios = [
#     "clinico",
#     "virologico"
# ]
# granularidade = "uf"
# e não:
# [
#     "clinico",
#     "virologico",
#     "geografico"
# ]
# Portanto, faremos uma segunda função:
# Detectar granularidade
# ============================================================

def detectar_granularidade(pergunta):
    pergunta_lower = normalizar_texto(
        pergunta
    )

    termos_uf = [
        "por uf",
        "cada uf",
        "por estado",
        "cada estado",
        "entre os estados"
    ]

    termos_municipio = [
        "por município",
        "por municipio",
        "cada município",
        "cada municipio"
    ]

    if any(t in pergunta_lower for t in termos_uf):
        return "uf"

    if any(t in pergunta_lower for t in termos_municipio):
        return "municipio"

    return None

# ===========================================================
#E retirar uf, estado, município da detecção do domínio geográfico.
#A função final ficará assim:
# ============================================================

def analisar_pergunta(pergunta):

    dominios = detectar_dominios_pergunta(
        pergunta
    )

    granularidade = detectar_granularidade(
        pergunta
    )

    return {
        "pergunta": pergunta,
        "dominios": dominios,
        "granularidade": granularidade,
        "consulta_combinada": len(dominios) > 1
    }

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

    # ============================================================
    # CRIAÇÃO DOS ITENS DO GRUPO
    # ============================================================

    for item in itens:

        # --------------------------------------------------------
        # Compatibilidade:
        #
        # formato antigo:
        # (nome, quantidade)
        #
        # formato com percentual explícito:
        # (nome, quantidade, percentual)
        # --------------------------------------------------------

        if len(item) == 3:

            nome, qtd, percentual_explicito = item

        else:

            nome, qtd = item
            percentual_explicito = None

        item_id = (
            f"{prefixo_item}_"
            f"{slug_no(nome)}"
        )

        eh_destaque = (
            item_destaque is not None
            and
            normalizar_texto(nome)
            ==
            normalizar_texto(item_destaque)
        )

        # --------------------------------------------------------
        # PERCENTUAL
        # --------------------------------------------------------

        if percentual_explicito is not None:

            percentual = float(
                percentual_explicito
            )

            size = max(
                14.0,
                min(
                    36.0,
                    12.0
                    + percentual * 0.3
                )
            )

        else:

            size, percentual = (
                calcular_tamanho_por_percentual(
                    qtd,
                    total_referencia
                )
            )

        if eh_destaque and rotulo_destaque:
            label = f"{rotulo_destaque}: {nome} ({qtd})" if qtd > 0 else f"{rotulo_destaque}: {nome}"
            tipo_no = f"{tipo_item}Destaque"
        else:
            label = f"{nome} ({qtd})" if qtd > 0 else nome
            tipo_no = tipo_item

        titulo = label

        if percentual_explicito is not None:

            titulo = (
                f"{label} — "
                f"{percentual:.1f}% "
                f"{base_descricao}"
            )

        elif (
            total_referencia
            and total_referencia > 0
        ):

            titulo = (
                f"{label} — "
                f"{percentual:.1f}% "
                f"{base_descricao} "
                f"({qtd}/{total_referencia})"
            )

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

# @st.cache_data
# def carregar_documentos_json() -> List[Document]:
#     documentos = []

#     for nome_arquivo in ARQUIVOS_JSON:
#         caminho = DATA_DIR / nome_arquivo

#         if not caminho.exists():
#             st.warning(f"Arquivo não encontrado: {caminho}")
#             continue

#         with open(caminho, "r", encoding="utf-8") as f:
#             conteudo = json.load(f)

#         if not isinstance(conteudo, list):
#             st.warning(f"O arquivo {nome_arquivo} não contém uma lista de documentos.")
#             continue

#         for item in conteudo:
#             if not isinstance(item, dict):
#                 continue

#             metadata = item.get("metadata", {}) or {}
#             texto = item.get("text_content", "") or ""

#             # =========================
#             # Normalização específica dos campos
#             # =========================
#             if "tipo_documento" in metadata and "document_type" not in metadata:
#                 metadata["document_type"] = metadata["tipo_documento"]

#             if "nome_uf" in metadata and "uf_name" not in metadata:
#                 metadata["uf_name"] = metadata["nome_uf"]
            
#             # compatibilidade com novo arquivo de hospitalização
#             if "uf_sigla" in metadata and "uf" not in metadata:
#                 metadata["uf"] = metadata["uf_sigla"]

#             if "uf_sigla" in metadata and "sg_uf_not" not in metadata:
#                 metadata["sg_uf_not"] = metadata["uf_sigla"]

#             if "ano" in metadata and "nu_ano" not in metadata:
#                 metadata["nu_ano"] = metadata["ano"]
            
#             if "internacoes" in metadata and "sim" not in metadata:
#                 metadata["sim"] = metadata["internacoes"]

#             if "nao_hospitalizados" in metadata and "nao" not in metadata:
#                 metadata["nao"] = metadata["nao_hospitalizados"]

#             if "taxa_internacao_bruta" in metadata and "taxa_internacao" not in metadata:
#                 metadata["taxa_internacao"] = metadata["taxa_internacao_bruta"]

#             # Para documentos por período, usa a semana final como semana normalizada
#             if "semana_final" in metadata and "semana" not in metadata:
#                 metadata["semana"] = str(metadata["semana_final"]).zfill(2)

#             if "semana_epidemiologica" in metadata and "semana" not in metadata:
#                 metadata["semana"] = str(metadata["semana_epidemiologica"]).zfill(2)

#             # Normalização para óbitos
#             if "obitos_agravo" in metadata and "obitos" not in metadata:
#                 metadata["obitos"] = metadata["obitos_agravo"]

#             metadata["arquivo_origem"] = nome_arquivo
#             metadata["document_type"] = str(metadata.get("document_type", "desconhecido"))
#             metadata["categoria"] = inferir_categoria(metadata, texto)
#             metadata["granularidade"] = inferir_granularidade(metadata, texto)

#             metadata["uf_normalizada"] = extrair_primeiro_campo_existente(
#                 metadata,
#                 [
#                     "uf",
#                     "uf_name",
#                     "nome_uf",
#                     "state",
#                     "state_name",
#                     "sg_uf_not"
#                 ]
#             )

#             metadata["ano_normalizado"] = extrair_primeiro_campo_existente(
#                 metadata,
#                 [
#                     "year",
#                     "ano",
#                     "ano_notificacao",
#                     "nu_ano"
#                 ]
#             )

#             metadata["semana_normalizada"] = extrair_primeiro_campo_existente(
#                 metadata,
#                 [
#                     "week",
#                     "semana",
#                     "semana_epidemiologica",
#                     "nu_semana_notificacao",
#                     "semana_epidemiologica_range"
#                 ]
#             )

#             metadata["sorotipo_normalizado"] = extrair_primeiro_campo_existente(
#                 metadata,
#                 [
#                     "sorotipo",
#                     "serotype"
#                 ]
#             )

#             metadata["num_cases_normalizado"] = extrair_primeiro_campo_existente(
#                 metadata,
#                 [
#                     "num_cases",
#                     "total_cases",
#                     "casos",
#                     "total_casos"
#                 ]
#             )

#             documentos.append(
#                 Document(
#                     page_content=texto,
#                     metadata=metadata
#                 )
#             )

#     return documentos

def eh_pergunta_grafo_combinado(
    pergunta: str,
    query_info: Dict
) -> bool:

    p = normalizar_texto(pergunta)

    menciona_grafo = (
        "grafo" in p
        or "grafo semantico" in p
    )

    menciona_combinacao = (
        "combinado" in p
        or "combinar" in p
        or (
            query_info.get(
                "intencoes",
                {}
            ).get("clinico", False)
            and
            query_info.get(
                "intencoes",
                {}
            ).get("sorotipos", False)
        )
    )

    return (
        menciona_grafo
        and menciona_combinacao
    )



# ============================================================
# 3. LEITURA DAS VISÕES SEMÂNTICAS MATERIALIZADAS
# ============================================================

def caminho_relativo_seguro(caminho: Path, base: Path) -> str:
    """
    Retorna o caminho relativo à pasta-base.
    Caso não seja possível, retorna o caminho completo.
    """
    try:
        return str(caminho.relative_to(base))
    except ValueError:
        return str(caminho)


def obter_valor_escopo(
    escopo: Dict,
    chaves: List[str],
    default=None
):
    """
    Procura o primeiro campo existente dentro de escopo.
    """
    if not isinstance(escopo, dict):
        return default

    for chave in chaves:
        valor = escopo.get(chave)

        if valor is not None and str(valor).strip() != "":
            return valor

    return default


def converter_metadata_chroma(metadata: Dict) -> Dict:
    """
    Garante que os metadados enviados ao LangChain/Chroma
    contenham apenas tipos simples.

    Chroma aceita principalmente:
    str, int, float e bool.
    """
    metadata_limpa = {}

    for chave, valor in metadata.items():

        if valor is None:
            continue

        if isinstance(valor, (str, int, float, bool)):
            metadata_limpa[chave] = valor

        else:
            # Estruturas complexas não entram diretamente
            # como metadata do Chroma.
            metadata_limpa[chave] = json.dumps(
                valor,
                ensure_ascii=False
            )

    return metadata_limpa


def extrair_metadata_vsm(
    documento_json: Dict,
    caminho_json: Path,
    caminho_markdown: Path
) -> Dict:
    """
    Extrai os principais metadados estruturados da VSM
    em formato adequado para recuperação e filtragem.
    """

    fonte = documento_json.get("fonte", {}) or {}
    escopo = documento_json.get("escopo", {}) or {}

    document_id = documento_json.get(
        "document_id",
        caminho_json.stem
    )

    titulo = documento_json.get(
        "titulo",
        document_id
    )

    tipo_documento = documento_json.get(
        "tipo_documento",
        "desconhecido"
    )

    dominio = documento_json.get(
        "dominio",
        "desconhecido"
    )

    # --------------------------------------------------------
    # Fonte
    # --------------------------------------------------------

    if isinstance(fonte, dict):
        fonte_nome = (
            fonte.get("nome")
            or fonte.get("sistema")
            or fonte.get("fonte")
            or "SINAN"
        )
    else:
        fonte_nome = str(fonte)

    # --------------------------------------------------------
    # Ano
    # --------------------------------------------------------

    ano = obter_valor_escopo(
        escopo,
        [
            "ano",
            "year"
        ],
        ANO
    )

    # --------------------------------------------------------
    # Localização
    # --------------------------------------------------------

    codigo_uf = obter_valor_escopo(
        escopo,
        [
            "codigo_uf",
            "uf_codigo",
            "codigo_ibge_uf",
            "sg_uf"
        ],
        ""
    )

    uf_nome = obter_valor_escopo(
        escopo,
        [
            "uf_nome",
            "nome_uf",
            "uf"
        ],
        ""
    )

    municipio = obter_valor_escopo(
        escopo,
        [
            "municipio",
            "municipio_nome",
            "nome_municipio"
        ],
        ""
    )

    localizacao = obter_valor_escopo(
        escopo,
        [
            "localizacao",
            "localização"
        ],
        ""
    )

    # --------------------------------------------------------
    # Período epidemiológico
    # --------------------------------------------------------

    semana_inicial = obter_valor_escopo(
        escopo,
        [
            "semana_inicial",
            "semana_epidemiologica_inicial"
        ],
        None
    )

    semana_final = obter_valor_escopo(
        escopo,
        [
            "semana_final",
            "semana_epidemiologica_final"
        ],
        None
    )

    # --------------------------------------------------------
    # Conceitos e palavras-chave
    #
    # Aqui são serializados como texto simples para que
    # posteriormente possam também participar do reranking.
    # --------------------------------------------------------

    conceitos = documento_json.get(
        "conceitos_semanticos",
        []
    ) or []

    palavras_chave = documento_json.get(
        "palavras_chave",
        []
    ) or []

    if isinstance(conceitos, list):
        conceitos_texto = ", ".join(
            str(x) for x in conceitos
        )
    else:
        conceitos_texto = str(conceitos)

    if isinstance(palavras_chave, list):
        palavras_chave_texto = ", ".join(
            str(x) for x in palavras_chave
        )
    else:
        palavras_chave_texto = str(palavras_chave)

    metadata = {
        "document_id": str(document_id),
        "titulo": str(titulo),

        "tipo_documento": str(tipo_documento),

        # Alias mantido temporariamente porque várias partes
        # do app12.py ainda utilizam document_type.
        "document_type": str(tipo_documento),

        "dominio": str(dominio),

        # Alias temporário para o código antigo
        "categoria": str(dominio),

        "fonte": str(fonte_nome),

        "ano": int(ano) if str(ano).isdigit() else str(ano),

        "codigo_uf": (
            str(codigo_uf)
            if codigo_uf is not None
            else ""
        ),

        "uf_nome": (
            str(uf_nome)
            if uf_nome is not None
            else ""
        ),

        # Alias temporário para facilitar filtros antigos
        "uf": (
            str(uf_nome)
            if uf_nome is not None
            else ""
        ),
        "municipio": (
            str(municipio)
            if municipio is not None
            else ""
        ),
        "localizacao": (
            str(localizacao)
            if localizacao is not None
            else ""
        ),

        "conceitos_semanticos": conceitos_texto,
        "palavras_chave": palavras_chave_texto,

        "arquivo_json": caminho_relativo_seguro(
            caminho_json,
            PASTA_DOCS
        ),

        "arquivo_markdown": caminho_relativo_seguro(
            caminho_markdown,
            PASTA_DOCS
        ),

        # Mantido por compatibilidade temporária com app12.py
        "arquivo_origem": caminho_json.name,

        "origem_documento": "VSM"
    }

    if semana_inicial is not None:
        metadata["semana_inicial"] = int(
            semana_inicial
        )

    if semana_final is not None:
        metadata["semana_final"] = int(
            semana_final
        )

    if (
        semana_inicial is not None
        and semana_final is not None
    ):
        metadata["periodo_epidemiologico"] = (
            f"{semana_inicial}-{semana_final}"
        )

    return converter_metadata_chroma(metadata)


@st.cache_data
def carregar_documentos_vsm() -> List[Document]:
    """
    Carrega as Visões Semânticas Materializadas produzidas
    pelos Notebooks 06 e 07.

    JSON:
        estrutura computável / metadados / evidências

    Markdown:
        representação textual utilizada como page_content
        do LangChain Document.
    """

    documentos = []

    arquivos_json = sorted(
        PASTA_DOCS.rglob("*.json")
    )

    total_json = 0
    total_carregados = 0
    total_sem_markdown = 0
    total_invalidos = 0

    
    for caminho_json in arquivos_json:

        # ----------------------------------------------------
        # Ignorar artefatos administrativos
        # ----------------------------------------------------

        if "_inventario" in caminho_json.parts:
            continue

        total_json += 1

        # ----------------------------------------------------
        # Ler JSON
        # ----------------------------------------------------

        try:

            with open(
                caminho_json,
                "r",
                encoding="utf-8"
            ) as arquivo:

                documento_json = json.load(
                    arquivo
                )

        except Exception as erro:

            total_invalidos += 1

            st.warning(
                f"Erro ao ler JSON: "
                f"{caminho_json}\n\n"
                f"{erro}"
            )

            continue

        # ----------------------------------------------------
        # Validar estrutura mínima da VSM
        # ----------------------------------------------------

        if not isinstance(documento_json, dict):

            total_invalidos += 1

            st.warning(
                f"JSON ignorado por não representar "
                f"um objeto VSM: {caminho_json}"
            )

            continue

        if "document_id" not in documento_json:

            total_invalidos += 1

            st.warning(
                f"JSON ignorado por não possuir "
                f"document_id: {caminho_json}"
            )

            continue

        # ----------------------------------------------------
        # Encontrar Markdown correspondente
        # ----------------------------------------------------

        caminho_markdown = (
            caminho_json.with_suffix(".md")
        )

        if not caminho_markdown.exists():

            total_sem_markdown += 1

            st.warning(
                f"Markdown correspondente não encontrado: "
                f"{caminho_markdown}"
            )

            continue

        # ----------------------------------------------------
        # Ler Markdown
        # ----------------------------------------------------

        try:

            texto_markdown = (
                caminho_markdown
                .read_text(
                    encoding="utf-8"
                )
                .strip()
            )

        except Exception as erro:

            total_invalidos += 1

            st.warning(
                f"Erro ao ler Markdown: "
                f"{caminho_markdown}\n\n"
                f"{erro}"
            )

            continue

        if not texto_markdown:

            total_invalidos += 1

            st.warning(
                f"Markdown vazio: "
                f"{caminho_markdown}"
            )

            continue

        # ----------------------------------------------------
        # Extrair metadados estruturados
        # ----------------------------------------------------

        metadata = extrair_metadata_vsm(
            documento_json=documento_json,
            caminho_json=caminho_json,
            caminho_markdown=caminho_markdown
        )

        # ----------------------------------------------------
        # Criar documento LangChain
        # ----------------------------------------------------

        documento_langchain = Document(
            page_content=texto_markdown,
            metadata=metadata
        )

        documentos.append(
            documento_langchain
        )

        total_carregados += 1

    # --------------------------------------------------------
    # Resumo de carregamento
    # --------------------------------------------------------

    if DEBUG:

        st.subheader(
            "Carregamento das Visões Semânticas"
        )

        st.write(
            "Arquivos JSON encontrados:",
            total_json
        )

        st.write(
            "VSMs carregadas:",
            total_carregados
        )

        st.write(
            "Sem Markdown correspondente:",
            total_sem_markdown
        )

        st.write(
            "Arquivos inválidos:",
            total_invalidos
        )
        

    return documentos

# # ============================================================
# # TESTE: VERIFICAR COMO A UF ESTÁ SALVA NO JSON
# # ============================================================

# for caminho_json in sorted(PASTA_DOCS.rglob("*.json")):

#     # Ignora inventário
#     if "_inventario" in caminho_json.parts:
#         continue

#     try:
#         with open(
#             caminho_json,
#             "r",
#             encoding="utf-8"
#         ) as arquivo:
#             doc_json = json.load(arquivo)

#         # Procurar um documento estadual clínico
#         if (
#             isinstance(doc_json, dict)
#             and doc_json.get("tipo_documento") == "perfil_clinico_uf"
#         ):

#             print("=" * 80)
#             print("ARQUIVO:")
#             print(caminho_json)

#             print("\nDOCUMENT_ID:")
#             print(
#                 doc_json.get("document_id")
#             )

#             print("\nCHAVES DO DOCUMENTO:")
#             print(
#                 list(doc_json.keys())
#             )

#             print("\nESCOPO:")
#             print(
#                 json.dumps(
#                     doc_json.get("escopo", {}),
#                     ensure_ascii=False,
#                     indent=2
#                 )
#             )

#             print("=" * 80)

#             break

#     except Exception as erro:
#         print(
#             f"Erro ao ler {caminho_json}: {erro}"
#         )

# ============================================================
# TESTE DO CARREGAMENTO DAS VSMs
# ============================================================

# if DEBUG:

#     documentos_teste = (
#         carregar_documentos_vsm()
#     )

#     print(
#         "Total de documentos carregados:",
#         len(documentos_teste)
#     )

#     if documentos_teste:

#         primeiro = documentos_teste[0]

#         print("\nMETADATA:")
#         print(
#             json.dumps(
#                 primeiro.metadata,
#                 ensure_ascii=False,
#                 indent=2
#             )
#         )

#         print("\nINÍCIO DO MARKDOWN:")
#         print(
#             primeiro.page_content[:1000]
#         )

#     registros = []

#     for doc in documentos_teste:

#         registros.append({
#             "document_id":
#                 doc.metadata.get("document_id"),

#             "tipo_documento":
#                 doc.metadata.get("tipo_documento"),

#             "dominio":
#                 doc.metadata.get("dominio"),

#             "ano":
#                 doc.metadata.get("ano"),

#             "uf":
#                 doc.metadata.get("uf"),

#             "arquivo_json":
#                 doc.metadata.get("arquivo_json"),

#             "arquivo_markdown":
#                 doc.metadata.get(
#                     "arquivo_markdown"
#                 )
#         })
        
#     df_documentos_vsm = pd.DataFrame(
#         registros
#     )

#     st.subheader(
#         "Inventário das VSMs carregadas"
#     )

#     st.dataframe(
#         df_documentos_vsm,
#         use_container_width=True
#     )

#     if not df_documentos_vsm.empty:

#         st.subheader(
#             "Documentos por domínio"
#         )

#         st.write(
#             df_documentos_vsm[
#                 "dominio"
#             ].value_counts()
#         )

#         st.subheader(
#             "Documentos por tipo"
#         )

#         st.write(
#             df_documentos_vsm[
#                 "tipo_documento"
#             ].value_counts()
#         )

# # ============================================================
# # TESTE: VERIFICAR SE AS UFs FORAM EXTRAÍDAS CORRETAMENTE
# # ============================================================

# if DEBUG:

#     documentos_teste = carregar_documentos_vsm()

#     registros = []

#     for doc in documentos_teste:

#         # Mostrar somente documentos estaduais
#         if doc.metadata.get("codigo_uf"):

#             registros.append({
#                 "document_id":
#                     doc.metadata.get("document_id"),

#                 "tipo_documento":
#                     doc.metadata.get("tipo_documento"),

#                 "dominio":
#                     doc.metadata.get("dominio"),

#                 "codigo_uf":
#                     doc.metadata.get("codigo_uf"),

#                 "uf":
#                     doc.metadata.get("uf")
#             })

#     df_teste_ufs = pd.DataFrame(registros)

#     st.subheader(
#         "Teste dos metadados de UF"
#     )

#     st.dataframe(
#         df_teste_ufs,
#         use_container_width=True
#     )

# ============================================================
# PASSO 4.1 — PREPARAÇÃO DAS VSMs PARA INDEXAÇÃO
# ============================================================

SECOES_INDEXAVEIS = {
    "Síntese epidemiológica": "sintese",
    "Indicadores": "indicadores",
    "Evidências": "evidencias",
    "Interpretação dos resultados": "interpretacao",
    "Observações sobre os dados": "observacoes_dados",
    "Conceitos semânticos": "conceitos_semanticos",
}

# ============================================================
# 4.1.2 Função para separar o Markdown pelas seções
# ============================================================
# Observe que essa função não corta por número de caracteres ou tokens.
# Isso é proposital.
# Estamos usando a estrutura que você definiu na própria VSM como fronteira semântica.

def extrair_secoes_markdown(texto_markdown: str) -> Dict[str, str]:
    """
    Extrai seções de nível 2 (##) de uma VSM em Markdown.

    Retorna:
        {
            "Síntese epidemiológica": "...",
            "Indicadores": "...",
            ...
        }
    """

    if not texto_markdown:
        return {}

    secoes = {}

    padrao = re.compile(
        r"^##\s+(.+?)\s*$",
        flags=re.MULTILINE
    )

    matches = list(
        padrao.finditer(texto_markdown)
    )

    for i, match in enumerate(matches):

        titulo_secao = match.group(1).strip()

        inicio = match.end()

        if i + 1 < len(matches):
            fim = matches[i + 1].start()
        else:
            fim = len(texto_markdown)

        conteudo = (
            texto_markdown[inicio:fim]
            .strip()
        )

        if conteudo:
            secoes[titulo_secao] = conteudo

    return secoes

# ============================================================
# 4.1.3 Função para localizar as seções indexáveis
# ============================================================
#Agora precisamos evitar problemas com acentos, maiúsculas ou pequenas variações nos títulos.
def selecionar_secoes_indexaveis(
    secoes: Dict[str, str]
) -> Dict[str, Dict[str, str]]:
    """
    Seleciona apenas as seções semanticamente relevantes
    para recuperação vetorial.
    """

    resultado = {}

    secoes_normalizadas = {
        normalizar_texto(titulo): (
            titulo,
            conteudo
        )
        for titulo, conteudo in secoes.items()
    }

    for titulo_esperado, secao_id in SECOES_INDEXAVEIS.items():

        chave = normalizar_texto(
            titulo_esperado
        )

        if chave not in secoes_normalizadas:
            continue

        titulo_original, conteudo = (
            secoes_normalizadas[chave]
        )

        resultado[secao_id] = {
            "titulo": titulo_original,
            "conteudo": conteudo,
        }

    return resultado

# ============================================================
# 4.1.4 — Metadados herdados da VSM
# ===========================================================
#Vamos criar uma função para herdar os metadados do documento-pai:
# Um chunk de:

# Evidências

# não pode virar um texto isolado sem sabermos que pertence, por exemplo, a:

# SINAN_DENGUE_2026_VIROLOGICO_35
# São Paulo
# domínio virológico
# 2026    
# chunk
#  ↓
# parent_document_id
#  ↓
# VSM original
#  ↓
# JSON + Markdown

# Isso será útil posteriormente para explicar de onde veio uma resposta.


def criar_metadata_chunk(
    documento: Document,
    secao: str,
    chunk_type: str
) -> Dict:
    """
    Cria os metadados de um chunk preservando
    a rastreabilidade até a VSM de origem.
    """

    metadata_pai = documento.metadata.copy()

    document_id = metadata_pai.get(
        "document_id",
        ""
    )

    metadata_chunk = metadata_pai.copy()

    metadata_chunk.update({
        "parent_document_id": document_id,
        "secao": secao,
        "chunk_type": chunk_type,
        "origem_documento": "VSM",
    })

    return metadata_chunk

# ============================================================
# 4.1.5 — Criar o chunk do documento completo
# ===========================================================

# Criar o chunk do documento completo
# Além das seções, recomendo manter uma representação da VSM inteira.
# Isso ajuda em perguntas amplas, como:
# “Qual é o panorama epidemiológico da dengue em 2026?”
# ou:
# “Faça um resumo da situação da dengue em Goiás.”
# Função:

def criar_chunk_documento_completo(
    documento: Document
) -> Document:

    metadata = criar_metadata_chunk(
        documento=documento,
        secao="documento_completo",
        chunk_type="full_document"
    )

    return Document(
        page_content=documento.page_content,
        metadata=metadata
    )

# ============================================================
#4.1.6 — Criar os chunks por seção
# ===========================================================

# Criar os chunks por seção

# Agora:
def criar_chunks_secoes(
    documento: Document
) -> List[Document]:

    secoes = extrair_secoes_markdown(
        documento.page_content
    )

    secoes_indexaveis = (
        selecionar_secoes_indexaveis(
            secoes
        )
    )

    chunks = []

    for secao_id, dados_secao in (
        secoes_indexaveis.items()
    ):

        titulo = dados_secao["titulo"]
        conteudo = dados_secao["conteudo"]

        metadata = criar_metadata_chunk(
            documento=documento,
            secao=secao_id,
            chunk_type="semantic_section"
        )

        # Mantemos o título da VSM para dar contexto
        # ao conteúdo recuperado.

        titulo_documento = (
            documento.metadata.get(
                "titulo",
                ""
            )
        )

        texto_chunk = (
            f"# {titulo_documento}\n\n"
            f"## {titulo}\n\n"
            f"{conteudo}"
        )

        chunk = Document(
            page_content=texto_chunk,
            metadata=metadata
        )

        chunks.append(chunk)

    return chunks

# Repare neste detalhe:
# texto_chunk = (
#     f"# {titulo_documento}\n\n"
#     f"## {titulo}\n\n"
#     f"{conteudo}"
# )

# Não estamos colocando apenas:
# DENV-2: 48,3%
# no embedding.
# Estamos colocando algo como:
# # Perfil virológico da dengue em São Paulo
# ## Evidências
# ...
# Isso fornece contexto semântico ao embedding.

# ============================================================
#4.1.7 — Construir todas as unidades de indexação
# ==========================================================
#Construir todas as unidades de indexação
# Agora juntamos:
# documento completo
# +
# seções
#para cada VSM.
@st.cache_data
def preparar_documentos_indexacao(
    documentos_vsm: List[Document]
) -> List[Document]:
    """
    Constrói as unidades que futuramente serão
    enviadas ao modelo de embeddings.
    """

    documentos_indexacao = []

    for documento in documentos_vsm:

        # --------------------------------------------
        # 1. VSM completa
        # --------------------------------------------

        documento_completo = (
            criar_chunk_documento_completo(
                documento
            )
        )

        documentos_indexacao.append(
            documento_completo
        )

        # --------------------------------------------
        # 2. Seções semânticas
        # --------------------------------------------

        chunks_secoes = (
            criar_chunks_secoes(
                documento
            )
        )

        documentos_indexacao.extend(
            chunks_secoes
        )

    return documentos_indexacao

# ============================================================
# 4.1.8 — Executar em memória
# ==========================================================
# Executar em memória
# Como você já possui:
# documentos_vsm
# com os 168 documentos carregados, execute:

# ============================================================
# PASSO 4.1.8 — CARREGAR VSMs E PREPARAR PARA INDEXAÇÃO
# ============================================================


documentos_vsm = carregar_documentos_vsm()

# ============================================================
# DIAGNÓSTICO — VSMs DO DOMÍNIO VIROLÓGICO
# ============================================================

dados = []

for doc in documentos_vsm:

    if normalizar_texto(
        doc.metadata.get("dominio", "")
    ) == "virologico":

        dados.append({
            "document_id":
                doc.metadata.get("document_id"),

            "tipo_documento":
                doc.metadata.get("tipo_documento"),

            "dominio":
                doc.metadata.get("dominio"),

            "codigo_uf":
                doc.metadata.get("codigo_uf"),

            "uf_nome":
                doc.metadata.get("uf_nome"),

            "arquivo_json":
                doc.metadata.get("arquivo_json"),
        })


if DEBUG:

    df_virologico = pd.DataFrame(
        dados
    )

    st.dataframe(
        df_virologico,
        use_container_width=True,
        hide_index=True
    )

    st.write(
        "Total de VSMs virológicas:",
        len(df_virologico)
    )

    st.write(
        "Tipos encontrados:",
        df_virologico[
            "tipo_documento"
        ].value_counts()
    )

# ============================================================
# PREPARAR UNIDADES DE INDEXAÇÃO
# ============================================================

documentos_indexacao = (
    preparar_documentos_indexacao(
        documentos_vsm
    )
)


if DEBUG:

    st.subheader(
        "Preparação das VSMs para indexação"
    )

    st.write(
        "VSMs originais:",
        len(documentos_vsm)
    )

    st.write(
        "Unidades preparadas para indexação:",
        len(documentos_indexacao)
    )

# Não espere necessariamente 168 × 7 = 1176.

# Algumas VSMs podem não possuir todas as seis seções indexáveis ou alguma seção pode estar vazia. Isso é justamente algo que vamos auditar no Passo 4.2.

# ============================================================
# PASSO 4.1.9 — INSPEÇÃO DAS UNIDADES DE INDEXAÇÃO
# ============================================================

if DEBUG and documentos_indexacao:

    st.subheader(
        "Exemplo de unidades de indexação"
    )

    for i, doc in enumerate(
        documentos_indexacao[:5]
    ):

        with st.expander(
            f"{i} — "
            f"{doc.metadata.get('chunk_type')} — "
            f"{doc.metadata.get('secao')}"
        ):

            st.write(
                "Document ID:",
                doc.metadata.get(
                    "document_id"
                )
            )

            st.write(
                "Parent Document ID:",
                doc.metadata.get(
                    "parent_document_id"
                )
            )

            st.write(
                "Domínio:",
                doc.metadata.get(
                    "dominio"
                )
            )

            st.write(
                "Tipo:",
                doc.metadata.get(
                    "tipo_documento"
                )
            )

            st.write(
                "UF:",
                doc.metadata.get(
                    "uf_nome"
                )
            )

            st.write(
                "Chunk type:",
                doc.metadata.get(
                    "chunk_type"
                )
            )

            st.write(
                "Seção:",
                doc.metadata.get(
                    "secao"
                )
            )

            st.markdown(
                doc.page_content[:1500]
            )

# O que precisamos verificar antes do 4.2

# Um documento estadual deverá produzir algo conceitualmente parecido com:

# VSM original
# SINAN_DENGUE_2026_VIROLOGICO_35
# │
# ├── full_document
# │
# ├── sintese
# ├── indicadores
# ├── evidencias
# ├── interpretacao
# ├── observacoes_dados
# └── conceitos_semanticos

# E todos devem continuar associados a:

# parent_document_id
# dominio
# tipo_documento
# ano
# codigo_uf
# uf_nome
# ...
# ============================================================
# PASSO 4.2 — AUDITORIA DAS UNIDADES DE INDEXAÇÃO
# ============================================================

def auditar_documentos_indexacao(
    documentos_indexacao: List[Document]
) -> pd.DataFrame:

    registros = []

    for doc in documentos_indexacao:

        texto = doc.page_content or ""

        registros.append({
            "document_id":
                doc.metadata.get("document_id"),

            "parent_document_id":
                doc.metadata.get("parent_document_id"),

            "dominio":
                doc.metadata.get("dominio"),

            "tipo_documento":
                doc.metadata.get("tipo_documento"),

            "codigo_uf":
                doc.metadata.get("codigo_uf"),

            "uf_nome":
                doc.metadata.get("uf_nome"),

            "chunk_type":
                doc.metadata.get("chunk_type"),

            "secao":
                doc.metadata.get("secao"),

            "tamanho_caracteres":
                len(texto),

            "tamanho_palavras":
                len(texto.split()),
        })

    return pd.DataFrame(registros)

#Agora:


#E vamos mostrar um resumo:
if DEBUG:
    df_auditoria_chunks = (
        auditar_documentos_indexacao(
            documentos_indexacao
        )
    )
    st.subheader(
        "Auditoria das unidades de indexação"
    )

    st.write(
        "Total de unidades:",
        len(df_auditoria_chunks)
    )

    st.write(
        "Documentos-pai distintos:",
        df_auditoria_chunks[
            "parent_document_id"
        ].nunique()
    )

    st.write(
        "Chunks full_document:",
        (
            df_auditoria_chunks[
                "chunk_type"
            ]
            == "full_document"
        ).sum()
    )

    st.write(
        "Chunks semantic_section:",
        (
            df_auditoria_chunks[
                "chunk_type"
            ]
            == "semantic_section"
        ).sum()
    )

# Distribuição por seção

# Acrescente:
if DEBUG:

    st.subheader(
        "Distribuição dos chunks por seção"
    )

    distribuicao_secoes = (
        df_auditoria_chunks[
            "secao"
        ]
        .value_counts()
        .rename_axis("secao")
        .reset_index(name="quantidade")
    )

    st.dataframe(
        distribuicao_secoes,
        use_container_width=True,
        hide_index=True
    )
#Verificar VSMs com menos de seis seções

#Esse teste é ainda mais importante:

if DEBUG:

    secoes_por_documento = (
        df_auditoria_chunks[
            df_auditoria_chunks[
                "chunk_type"
            ] == "semantic_section"
        ]
        .groupby(
            "parent_document_id"
        )
        .size()
        .reset_index(
            name="quantidade_secoes"
        )
    )

    documentos_secoes_incompletas = (
        secoes_por_documento[
            secoes_por_documento[
                "quantidade_secoes"
            ] < len(SECOES_INDEXAVEIS)
        ]
        .sort_values(
            "quantidade_secoes"
        )
    )

    st.subheader(
        "VSMs com menos seções indexáveis"
    )

    if documentos_secoes_incompletas.empty:

        st.success(
            "Todas as VSMs possuem todas "
            "as seções indexáveis."
        )

    else:

        st.dataframe(
            documentos_secoes_incompletas,
            use_container_width=True,
            hide_index=True
        )

#Provavelmente aparecerá uma ou duas VSMs. Não vamos corrigir automaticamente. Primeiro verificamos quais são.

#Tamanho dos chunks

#Agora uma análise essencial antes dos embeddings:

if DEBUG:

    st.subheader(
        "Tamanho das unidades de indexação"
    )

    resumo_tamanho = (
        df_auditoria_chunks
        .groupby(
            ["chunk_type", "secao"]
        )
        .agg(
            quantidade=(
                "document_id",
                "count"
            ),
            caracteres_min=(
                "tamanho_caracteres",
                "min"
            ),
            caracteres_mediana=(
                "tamanho_caracteres",
                "median"
            ),
            caracteres_media=(
                "tamanho_caracteres",
                "mean"
            ),
            caracteres_max=(
                "tamanho_caracteres",
                "max"
            ),
            palavras_media=(
                "tamanho_palavras",
                "mean"
            ),
            palavras_max=(
                "tamanho_palavras",
                "max"
            ),
        )
        .reset_index()
    )

    st.dataframe(
        resumo_tamanho,
        use_container_width=True,
        hide_index=True
    )

# Isso vai responder uma questão que eu considero central:

# Nossas fronteiras semânticas naturais estão produzindo chunks de tamanho adequado ou algumas seções estão excessivamente grandes?
# Só depois dessa resposta decidimos se evidencias, por exemplo, precisa de subdivisão.

# Ver os maiores chunks

# Por fim:
if DEBUG:

    st.subheader(
        "Maiores unidades de indexação"
    )

    maiores_chunks = (
        df_auditoria_chunks
        .sort_values(
            "tamanho_caracteres",
            ascending=False
        )
        .head(15)
    )

    st.dataframe(
        maiores_chunks,
        use_container_width=True,
        hide_index=True
    )
# ============================================================
# PASSO 4.3 — REFINAMENTO DAS UNIDADES DE INDEXAÇÃO
# ============================================================

# Limite preventivo por caracteres.
# A validação real por tokens será feita posteriormente.
MAX_CARACTERES_CHUNK = 6000

# Evita criar fragmentos muito pequenos quando for possível
# agregá-los ao bloco seguinte.
MIN_CARACTERES_CHUNK = 500

# ============================================================
#4.3.2 — separar subtítulos ###
# ===========================================================

#Adicione:

def extrair_subsecoes_markdown(
    conteudo: str
) -> List[Dict[str, str]]:
    """
    Divide uma seção Markdown pelos subtítulos de nível 3 (###).

    Se não houver subtítulos, retorna o conteúdo inteiro
    como uma única subseção.
    """

    if not conteudo:
        return []

    padrao = re.compile(
        r"^###\s+(.+?)\s*$",
        flags=re.MULTILINE
    )

    matches = list(
        padrao.finditer(conteudo)
    )

    if not matches:
        return [{
            "titulo": "",
            "conteudo": conteudo.strip()
        }]

    subsecoes = []

    # Conteúdo existente antes do primeiro ###
    prefixo = conteudo[:matches[0].start()].strip()

    if prefixo:
        subsecoes.append({
            "titulo": "",
            "conteudo": prefixo
        })

    for i, match in enumerate(matches):

        titulo = match.group(1).strip()
        inicio = match.end()

        if i + 1 < len(matches):
            fim = matches[i + 1].start()
        else:
            fim = len(conteudo)

        texto = conteudo[inicio:fim].strip()

        if texto:
            subsecoes.append({
                "titulo": titulo,
                "conteudo": texto
            })

    return subsecoes
# ============================================================
#4.3.3 — dividir apenas blocos ainda muito grandes
# ===========================================================

#Agora:
def dividir_texto_em_blocos(
    texto: str,
    max_caracteres: int = MAX_CARACTERES_CHUNK
) -> List[str]:
    """
    Divide textos grandes preservando, quando possível,
    parágrafos e linhas Markdown completas.

    Não divide textos que já estejam abaixo do limite.
    """

    texto = (texto or "").strip()

    if not texto:
        return []

    if len(texto) <= max_caracteres:
        return [texto]

    # Primeiro tenta preservar parágrafos
    blocos_origem = re.split(
        r"\n\s*\n",
        texto
    )

    blocos_finais = []
    atual = ""

    for bloco in blocos_origem:

        bloco = bloco.strip()

        if not bloco:
            continue

        # ----------------------------------------------------
        # Caso um único bloco já seja maior que o limite,
        # tenta quebrá-lo por linhas.
        # ----------------------------------------------------

        if len(bloco) > max_caracteres:

            if atual:
                blocos_finais.append(
                    atual.strip()
                )
                atual = ""

            linhas = bloco.splitlines()
            atual_linhas = ""

            for linha in linhas:

                linha = linha.strip()

                if not linha:
                    continue

                candidato = (
                    atual_linhas
                    + ("\n" if atual_linhas else "")
                    + linha
                )

                if (
                    len(candidato)
                    <= max_caracteres
                ):
                    atual_linhas = candidato

                else:

                    if atual_linhas:
                        blocos_finais.append(
                            atual_linhas.strip()
                        )

                    # Situação extrema:
                    # uma única linha maior que o limite.
                    if len(linha) > max_caracteres:

                        for inicio in range(
                            0,
                            len(linha),
                            max_caracteres
                        ):
                            blocos_finais.append(
                                linha[
                                    inicio:
                                    inicio + max_caracteres
                                ].strip()
                            )

                        atual_linhas = ""

                    else:
                        atual_linhas = linha

            if atual_linhas:
                blocos_finais.append(
                    atual_linhas.strip()
                )

            continue

        # ----------------------------------------------------
        # Bloco normal
        # ----------------------------------------------------

        candidato = (
            atual
            + ("\n\n" if atual else "")
            + bloco
        )

        if len(candidato) <= max_caracteres:

            atual = candidato

        else:

            if atual:
                blocos_finais.append(
                    atual.strip()
                )

            atual = bloco

    if atual:
        blocos_finais.append(
            atual.strip()
        )

    return [
        bloco
        for bloco in blocos_finais
        if bloco
    ]

# ========================================================
#4.3.4 — criar identificador do chunk
# ========================================================
#É importante termos rastreabilidade até cada fragmento:
def criar_chunk_id(
    document_id: str,
    secao: str,
    subseccao: str,
    indice: int
) -> str:

    partes = [
        str(document_id),
        str(secao),
    ]

    if subseccao:
        partes.append(
            normalizar_texto(subseccao)
            .replace(" ", "_")
        )

    partes.append(
        f"{indice:03d}"
    )

    return "__".join(partes)
# ============================================================
#4.3.5 — nova função para criar os chunks

#Agora vamos substituir a atual criar_chunks_secoes() por uma versão refinada.

#Sugiro renomear, em vez de apagar a anterior, enquanto estamos testando:
# ============================================================


def criar_chunks_secoes_refinados(
    documento: Document
) -> List[Document]:

    secoes = extrair_secoes_markdown(
        documento.page_content
    )

    secoes_indexaveis = (
        selecionar_secoes_indexaveis(
            secoes
        )
    )

    chunks = []

    document_id = documento.metadata.get(
        "document_id",
        ""
    )

    titulo_documento = documento.metadata.get(
        "titulo",
        ""
    )

    for secao_id, dados_secao in (
        secoes_indexaveis.items()
    ):

        titulo_secao = (
            dados_secao["titulo"]
        )

        conteudo = (
            dados_secao["conteudo"]
        )

        # ====================================================
        # SEÇÕES NORMAIS
        #
        # Mantemos integralmente enquanto estiverem
        # abaixo do limite.
        # ====================================================

        if len(conteudo) <= MAX_CARACTERES_CHUNK:

            metadata = criar_metadata_chunk(
                documento=documento,
                secao=secao_id,
                chunk_type="semantic_section"
            )

            chunk_id = criar_chunk_id(
                document_id=document_id,
                secao=secao_id,
                subseccao="",
                indice=1
            )

            metadata.update({
                "chunk_id": chunk_id,
                "subsecao": "",
                "chunk_indice": 1,
                "chunk_total_secao": 1
            })

            texto_chunk = (
                f"# {titulo_documento}\n\n"
                f"## {titulo_secao}\n\n"
                f"{conteudo}"
            )

            chunks.append(
                Document(
                    page_content=texto_chunk,
                    metadata=metadata
                )
            )

            continue

        # ====================================================
        # SEÇÃO GRANDE
        #
        # Primeiro usa a estrutura ###.
        # ====================================================

        subsecoes = (
            extrair_subsecoes_markdown(
                conteudo
            )
        )

        fragmentos = []

        for subsecao in subsecoes:

            titulo_subsecao = (
                subsecao["titulo"]
            )

            conteudo_subsecao = (
                subsecao["conteudo"]
            )

            partes = dividir_texto_em_blocos(
                conteudo_subsecao,
                max_caracteres=MAX_CARACTERES_CHUNK
            )

            for parte in partes:

                fragmentos.append({
                    "titulo_subsecao":
                        titulo_subsecao,

                    "conteudo":
                        parte
                })

        total_fragmentos = len(fragmentos)

        for indice, fragmento in enumerate(
            fragmentos,
            start=1
        ):

            titulo_subsecao = (
                fragmento[
                    "titulo_subsecao"
                ]
            )

            conteudo_fragmento = (
                fragmento[
                    "conteudo"
                ]
            )

            metadata = criar_metadata_chunk(
                documento=documento,
                secao=secao_id,
                chunk_type="semantic_subsection"
            )

            chunk_id = criar_chunk_id(
                document_id=document_id,
                secao=secao_id,
                subseccao=titulo_subsecao,
                indice=indice
            )

            metadata.update({
                "chunk_id":
                    chunk_id,

                "subsecao":
                    titulo_subsecao,

                "chunk_indice":
                    indice,

                "chunk_total_secao":
                    total_fragmentos
            })

            texto_chunk = (
                f"# {titulo_documento}\n\n"
                f"## {titulo_secao}\n\n"
            )

            if titulo_subsecao:

                texto_chunk += (
                    f"### {titulo_subsecao}\n\n"
                )

            texto_chunk += (
                conteudo_fragmento
            )

            chunks.append(
                Document(
                    page_content=texto_chunk,
                    metadata=metadata
                )
            )

    return chunks

# ============================================================
#4.3.6 — nova preparação para indexação

#Agora não usaremos mais full_document na coleção que futuramente irá para embeddings:
# ============================================================
def preparar_documentos_indexacao_refinados(
    documentos_vsm: List[Document]
) -> List[Document]:
    """
    Gera exclusivamente as unidades destinadas
    à recuperação vetorial.

    As VSMs completas continuam preservadas em
    documentos_vsm, mas não são duplicadas no índice.
    """

    documentos_indexacao = []

    for documento in documentos_vsm:

        chunks = (
            criar_chunks_secoes_refinados(
                documento
            )
        )

        documentos_indexacao.extend(
            chunks
        )

    return documentos_indexacao

# E execute:

documentos_indexacao_refinados = (
    preparar_documentos_indexacao_refinados(
        documentos_vsm
    )
)

# ============================================================
# 4.3.7 — primeira comparação
#
#Mostre na interface:
if DEBUG:

    st.subheader(
        "Refinamento das unidades de indexação"
    )

    st.write(
        "VSMs originais:",
        len(documentos_vsm)
    )

    st.write(
        "Unidades antes do refinamento:",
        len(documentos_indexacao)
    )

    st.write(
        "Unidades após o refinamento:",
        len(
            documentos_indexacao_refinados
        )
    )

# É normal o número mudar.

# Não tente obter exatamente 1.006, 1.174 ou qualquer outro valor predefinido. O número correto será consequência da fragmentação efetivamente necessária.
# ============================================================
#4.3.8 — auditar novamente
# ===========================================================

#Podemos reutilizar sua função do 4.2:

df_auditoria_chunks_refinados = (
    auditar_documentos_indexacao(
        documentos_indexacao_refinados
    )
)

#E acrescente:
if DEBUG:

    st.subheader(
        "Auditoria após refinamento"
    )

    st.write(
        "Total de unidades:",
        len(
            df_auditoria_chunks_refinados
        )
    )

    st.write(
        "Documentos-pai distintos:",
        df_auditoria_chunks_refinados[
            "parent_document_id"
        ].nunique()
    )

    st.write(
        "Maior chunk em caracteres:",
        df_auditoria_chunks_refinados[
            "tamanho_caracteres"
        ].max()
    )

    st.write(
        "Mediana em caracteres:",
        df_auditoria_chunks_refinados[
            "tamanho_caracteres"
        ].median()
    )

# ============================================================
#4.3.9 — distribuição dos novos tipos
# ===========================================================
#Também quero ver:

if DEBUG:

    distribuicao_tipos_refinados = (
        df_auditoria_chunks_refinados[
            "chunk_type"
        ]
        .value_counts()
        .rename_axis(
            "chunk_type"
        )
        .reset_index(
            name="quantidade"
        )
    )

    st.subheader(
        "Tipos de chunks após refinamento"
    )

    st.dataframe(
        distribuicao_tipos_refinados,
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# PASSO 4.4 — CONTAGEM DE TOKENS E ESTIMATIVA DE CUSTO
# ============================================================

def obter_tokenizador_embeddings():
    """
    Retorna o tokenizador adequado para estimar
    os tokens enviados ao modelo de embeddings.
    """

    try:
        return tiktoken.encoding_for_model(
            EMBEDDING_MODEL
        )

    except KeyError:
        return tiktoken.get_encoding(
            "cl100k_base"
        )

def contar_tokens_texto(
    texto: str,
    encoder
) -> int:

    if not texto:
        return 0

    return len(
        encoder.encode(texto)
    )

def auditar_tokens_embeddings(
    documentos: List[Document]
) -> pd.DataFrame:

    encoder = (
        obter_tokenizador_embeddings()
    )

    registros = []

    for doc in documentos:

        texto = (
            doc.page_content
            or ""
        )

        total_tokens = (
            contar_tokens_texto(
                texto,
                encoder
            )
        )

        registros.append({
            "document_id":
                doc.metadata.get(
                    "document_id"
                ),

            "parent_document_id":
                doc.metadata.get(
                    "parent_document_id"
                ),

            "dominio":
                doc.metadata.get(
                    "dominio"
                ),

            "tipo_documento":
                doc.metadata.get(
                    "tipo_documento"
                ),

            "chunk_type":
                doc.metadata.get(
                    "chunk_type"
                ),

            "secao":
                doc.metadata.get(
                    "secao"
                ),

            "subsecao":
                doc.metadata.get(
                    "subsecao"
                ),

            "chunk_id":
                doc.metadata.get(
                    "chunk_id"
                ),

            "caracteres":
                len(texto),

            "tokens":
                total_tokens
        })

    return pd.DataFrame(
        registros
    )

df_tokens_embeddings = (
    auditar_tokens_embeddings(
        documentos_indexacao_refinados
    )
)

total_tokens_embeddings = int(
    df_tokens_embeddings[
        "tokens"
    ].sum()
)

media_tokens_chunk = (
    df_tokens_embeddings[
        "tokens"
    ].mean()
)

mediana_tokens_chunk = (
    df_tokens_embeddings[
        "tokens"
    ].median()
)

max_tokens_chunk = int(
    df_tokens_embeddings[
        "tokens"
    ].max()
)

PRECO_EMBEDDING_USD_1M = 0.02

custo_estimado_embeddings = (
    total_tokens_embeddings
    / 1_000_000
    * PRECO_EMBEDDING_USD_1M
)

if DEBUG:

    st.subheader(
        "Estimativa de tokens e custo dos embeddings"
    )

    st.write(
        "Modelo:",
        EMBEDDING_MODEL
    )

    st.write(
        "Unidades para embedding:",
        len(
            documentos_indexacao_refinados
        )
    )

    st.write(
        "Tokens totais:",
        f"{total_tokens_embeddings:,}"
        .replace(",", ".")
    )

    st.write(
        "Média de tokens por unidade:",
        f"{media_tokens_chunk:,.1f}"
        .replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )

    st.write(
        "Mediana de tokens por unidade:",
        f"{mediana_tokens_chunk:,.1f}"
        .replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )

    st.write(
        "Maior unidade em tokens:",
        f"{max_tokens_chunk:,}"
        .replace(",", ".")
    )

    st.write(
        "Preço considerado:",
        "US$ 0,02 por 1 milhão de tokens"
    )

    st.write(
        "Custo estimado para criação dos embeddings:",
        f"US$ {custo_estimado_embeddings:.6f}"
    )

if DEBUG:

    st.subheader(
        "Maiores chunks em tokens"
    )

    maiores_tokens = (
        df_tokens_embeddings
        .sort_values(
            "tokens",
            ascending=False
        )
        .head(15)
    )

    st.dataframe(
        maiores_tokens,
        use_container_width=True,
        hide_index=True
    )    

# ============================================================
# PASSO 4.5.1 — CONFIRMAR DESTINO DA NOVA BASE VETORIAL
# ============================================================

if DEBUG:

    st.subheader(
        "Destino da nova base vetorial"
    )

    st.write(
        "Diretório:",
        str(PERSIST_DIRECTORY)
    )

    st.write(
        "Caminho absoluto:",
        str(
            Path(
                PERSIST_DIRECTORY
            ).resolve()
        )
    )

    st.write(
        "Existe atualmente:",
        Path(
            PERSIST_DIRECTORY
        ).exists()
    )

if Path(PERSIST_DIRECTORY).exists():

    itens_existentes = list(
        Path(
            PERSIST_DIRECTORY
        ).iterdir()
    )

    if DEBUG:

        st.write(
            "Quantidade de itens existentes:",
            len(itens_existentes)
        )

        st.write(
            "Primeiros itens:",
            [
                item.name
                for item in itens_existentes[:20]
            ]
        )

pasta_chroma = Path(
    PERSIST_DIRECTORY
)

if DEBUG:

    st.write(
        "Base existe AGORA:",
        pasta_chroma.exists()
    )

    if pasta_chroma.exists():

        arquivos_chroma = list(
            pasta_chroma.rglob("*")
        )

        st.write(
            "Itens encontrados na pasta:",
            len(arquivos_chroma)
        )

        st.write(
            "Primeiros itens:",
            [
                str(item.relative_to(pasta_chroma))
                for item in arquivos_chroma[:20]
            ]
        )


#DEBUG = False
# ============================================================
# PASSO 4.5.2 — CRIAR OU CARREGAR BASE VETORIAL VSM
# ============================================================

from pathlib import Path

from langchain_openai import OpenAIEmbeddings



NOME_COLECAO_VSM = "sinan_vsm_2026"
# ============================================================
# CARREGAMENTO LAZY DO VECTORSTORE VSM
# ============================================================

@st.cache_resource
def carregar_vectorstore_vsm():

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )

    vectorstore = Chroma(
        persist_directory=str(
            PERSIST_DIRECTORY
        ),
        embedding_function=embeddings,
        collection_name=NOME_COLECAO_VSM
    )

    return vectorstore


def recuperar_vsm_hibrido(
    pergunta,
    vectorstore,
    k_vetorial=24,
    k_final=5
):

    # --------------------------------------------------------
    # 1. Classificação da pergunta
    # --------------------------------------------------------

    classificacao = classificar_pergunta(
        pergunta
    )

    # --------------------------------------------------------
    # 2. Verificação de escopo
    # --------------------------------------------------------

    status_escopo = classificacao.get(
        "status_escopo",
        ESCOPO_VALIDO
    )

    if status_escopo != ESCOPO_VALIDO:

        return {
            "classificacao": classificacao,
            "resultados_vetoriais": [],
            "resultados_filtrados": [],
            "resultados_finais": []
        }

    # --------------------------------------------------------
    # 3. Busca vetorial ampla
    # --------------------------------------------------------

    resultados_vetoriais = (
        vectorstore.similarity_search(
            pergunta,
            k=k_vetorial
        )
    )

    # --------------------------------------------------------
    # 4. Filtragem semântica
    # --------------------------------------------------------

    resultados_filtrados = (
        filtrar_por_document_type(
            resultados_vetoriais,
            classificacao
        )
    )

    # --------------------------------------------------------
    # 5. Score heurístico
    # --------------------------------------------------------

    resultados_com_score = []

    for doc in resultados_filtrados:

        score = calcular_score_documento(
            doc,
            classificacao
        )

        resultados_com_score.append(
            {
                "documento": doc,
                "score": score
            }
        )

    # --------------------------------------------------------
    # 6. Ordenação
    # --------------------------------------------------------

    resultados_com_score = sorted(
        resultados_com_score,
        key=lambda x: x["score"],
        reverse=True
    )

    # --------------------------------------------------------
    # 7. Top-k final
    # --------------------------------------------------------

    resultados_finais = (
        resultados_com_score[
            :k_final
        ]
    )

    return {
        "classificacao": classificacao,
        "resultados_vetoriais": resultados_vetoriais,
        "resultados_filtrados": resultados_filtrados,
        "resultados_finais": resultados_finais
    }


# ============================================================
# PASSO 4.6.2
# TESTE CONTROLADO DA RECUPERAÇÃO HÍBRIDA
# ============================================================

# if DEBUG:

#     st.subheader(
#         "Teste da recuperação híbrida"
#     )

#     pergunta_teste = (
#         "Quais são os sorotipos de dengue "
#         "registrados em São Paulo?"
#     )

#     resultado_teste = (
#         recuperar_vsm_hibrido(
#             pergunta=pergunta_teste,
#             vectorstore=vectorstore_vsm,
#             k_vetorial=24,
#             k_final=5
#         )
#     )

#     st.write(
#         "Pergunta:",
#         pergunta_teste
#     )

#     st.write(
#         "Classificação:",
#         resultado_teste[
#             "classificacao"
#         ]
#     )

#     st.write(
#         "Resultados vetoriais:",
#         len(
#             resultado_teste[
#                 "resultados_vetoriais"
#             ]
#         )
#     )

#     st.write(
#         "Resultados após filtro:",
#         len(
#             resultado_teste[
#                 "resultados_filtrados"
#             ]
#         )
#     )

# #Agora vamos mostrar o ranking final:

#     st.write(
#         "### Ranking final"
#     )

#     for posicao, item in enumerate(
#         resultado_teste[
#             "resultados_finais"
#         ],
#         start=1
#     ):

#         doc = item["documento"]
#         score = item["score"]

#         st.markdown(
#             f"#### {posicao}. "
#             f"{doc.metadata.get('document_id')}"
#         )

#         st.write(
#             "Score:",
#             score
#         )

#         st.write(
#             "Domínio:",
#             doc.metadata.get("dominio")
#         )

#         st.write(
#             "Tipo:",
#             doc.metadata.get(
#                 "tipo_documento"
#             )
#         )

#         st.write(
#             "UF:",
#             doc.metadata.get("uf")
#         )

#         st.write(
#             "Seção:",
#             doc.metadata.get("secao")
#         )

#         st.write(
#             "Chunk:",
#             doc.metadata.get(
#                 "chunk_id"
#             )
#         )

#         st.write(
#             doc.page_content[:800]
#         )

#         st.divider()

# ============================================================
# PASSO 4.6.A
# BASELINE — RECUPERAÇÃO VETORIAL PURA

# Você terá dois blocos diferentes

# A — Baseline:

# vectorstore_vsm.similarity_search(
#     pergunta,
#     k=5
# )

# Fluxo:

# Pergunta
#    ↓
# embedding
#    ↓
# Chroma
#    ↓
# Top-5

# B — Método híbrido:

# recuperar_documentos_inteligentes(
#     pergunta=pergunta,
#     vectorstore=vectorstore_vsm,
#     documentos_base=documentos_vsm
# )

# Fluxo:

# Pergunta
#    ↓
# classificar_pergunta()
#    ↓
# similarity_search(k=24)
#    ↓
# filtrar_por_document_type()
#    ↓
# score_heuristico()
#    ↓
# Top-10

# Seu recuperar_documentos_inteligentes() realmente implementa essa segunda sequência: classifica, busca 24 resultados, filtra e aplica score_heuristico.

# Então, não apague a bateria que gerou aqueles cinco resultados. Renomeie-a para BASELINE — RECUPERAÇÃO VETORIAL PURA.

# Ela pode ser importante mais tarde para compararmos, com as mesmas perguntas, o baseline vetorial contra a recuperação híbrida.
# ============================================================

if DEBUG:

    st.subheader(
        "Baseline — recuperação vetorial pura"
    )

    perguntas_teste = [

        (
            "Sorotipos — São Paulo",
            "Quais são os sorotipos de dengue "
            "registrados em São Paulo?"
        ),

        (
            "Clínico — Minas Gerais",
            "Quais são os principais sintomas "
            "registrados em Minas Gerais?"
        ),

        (
            "Incidência — Goiás",
            "Qual a incidência de dengue em Goiás?"
        ),

        (
            "Temporal — Brasil",
            "Qual foi a evolução das notificações "
            "ao longo das semanas?"
        ),

        (
            "Panorama — Brasil",
            "Qual é o panorama geral da dengue "
            "em 2026?"
        ),
    ]

    # for titulo_teste, pergunta_teste in perguntas_teste:

    #     st.markdown(
    #         f"### {titulo_teste}"
    #     )

    #     st.write(
    #         "Pergunta:",
    #         pergunta_teste
    #     )

    #     resultados = (
    #         vectorstore_vsm.similarity_search(
    #             pergunta_teste,
    #             k=5
    #         )
    #     )

    #     for i, doc in enumerate(
    #         resultados,
    #         start=1
    #     ):

    #         st.write(
    #             f"{i} — "
    #             f"{doc.metadata.get('parent_document_id')} "
    #             f"— {doc.metadata.get('secao')}"
    #         )

    #     st.divider()

# ============================================================
# 5. ENTENDIMENTO DA PERGUNTA
# ============================================================

NOTIFICACOES_CHAVE = [
    "notificacao",
    "notificacoes",
    "notificado",
    "notificados",
    "total de notificacoes",
    "quantidade de notificacoes",
    "numero de notificacoes",
    # Linguagem natural comum do usuário
    "caso",
    "casos",
    "total de casos",
    "quantidade de casos",
    "numero de casos",
]

TEMPORAL_CHAVE = [
    "semana",
    "semanas",
    "semanal",
    "semana epidemiologica",
    "periodo",
    "periodo epidemiologico",
    "temporal",
    "serie temporal",
    "evolucao temporal",
    "tendencia",
    "pico",
    "media semanal",
    "mediana semanal",
    "aumento",
    "reducao",
    "redução",
]

GEOGRAFICO_CHAVE = [
    "uf",
    "ufs",
    "estado",
    "estados",
    "municipio",
    "municipios",
    "municipal",
    "geografico",
    "geografica",
    "distribuicao geografica",
    "distribuição geográfica",
    "residencia",
    "residência",
    "notificacao por uf",
    "notificação por uf",
]

INCIDENCIA_CHAVE = [
    "incidencia",
    "incidência",
    "taxa de incidencia",
    "taxa de incidência",
    "por 100 mil",
    "100 mil habitantes",
    "100.000 habitantes",
    "criticidade",
]

AUTOCTONIA_CHAVE = [
    "autoctone",
    "autóctone",
    "autoctones",
    "autóctones",
    "autoctonia",
    "autoctonia",
    "local provavel de infeccao",
    "local provável de infecção",
    "provavel infeccao",
    "provável infecção",
    "uf provavel de infeccao",
    "uf provável de infecção",
    "municipio provavel de infeccao",
    "município provável de infecção",
    "residencia e infeccao",
    "residência e infecção",
    "concordancia",
    "concordância",
]

SOROTIPOS_CHAVE = [
    "sorotipo",
    "sorotipos",
    "denv-1",
    "denv 1",
    "den1",
    "den 1",
    "denv-2",
    "denv 2",
    "den2",
    "den 2",
    "denv-3",
    "denv 3",
    "den3",
    "den 3",
    "denv-4",
    "denv 4",
    "den4",
    "den 4",
    "virologico",
    "virologica",
    "virológico",
    "virológica",
]

CLINICO_CHAVE = [
    "clinico",
    "clinica",
    "clínico",
    "clínica",
    "sintoma",
    "sintomas",
    "sinal clinico",
    "sinais clinicos",
    "sinal clínico",
    "sinais clínicos",
    "febre",
    "mialgia",
    "cefaleia",
    "cefaleia",
    "exantema",
    "vomito",
    "vômito",
    "nausea",
    "náusea",
    "dor nas costas",
    "conjuntivite",
    "artrite",
    "artralgia",
    "petequia",
    "petéquia",
    "leucopenia",
    "prova do laco",
    "prova do laço",
    "dor retro-orbital",
]

DOENCAS_PREEXISTENTES_CHAVE = [
    "doenca pre-existente",
    "doenca preexistente",
    "doencas pre-existentes",
    "doencas preexistentes",
    "comorbidade",
    "comorbidades",
    "diabetes",
    "hipertensao",
    "doenca hematologica",
    "doencas hematologicas",
    "hematologica",
    "hepatopatia",
    "doenca renal",
    "doencas renais",
    "renal",
    "doenca acido-peptica",
    "autoimune",
    "auto-imune",
]

HOSPITALIZACAO_CHAVE = [
    "hospitalizacao",
    "hospitalizacoes",
    "hospitalizado",
    "hospitalizados",
    "hospitalizada",
    "hospitalizadas",
    "internacao",
    "internacoes",
    "internado",
    "internados",
    "internada",
    "internadas",
    "taxa de hospitalizacao",
    "taxa de internacao",
]

EVOLUCAO_CHAVE = [
    "evolucao",
    "evolução",
    "desfecho",
    "desfechos",
    "cura",
    "curado",
    "curados",
    "resultado do caso",
]

OBITOS_CHAVE = [
    "obito",
    "óbito",
    "obitos",
    "óbitos",
    "morte",
    "mortes",
    "mortalidade",
    "obito pelo agravo",
    "óbito pelo agravo",
    "obito por outras causas",
    "óbito por outras causas",
    "obito em investigacao",
    "óbito em investigação",
]

PANORAMA_CHAVE = [
    "panorama",
    "panorama geral",
    "visao geral",
    "visão geral",
    "resumo geral",
    "situacao epidemiologica",
    "situação epidemiológica",
    "cenario epidemiologico",
    "cenário epidemiológico",
    "perfil epidemiologico",
    "perfil epidemiológico",
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
    "acre",
    "alagoas",
    "amapá",
    "amazonas",
    "bahia",
    "ceará",
    "distrito federal",
    "espírito santo",
    "goiás",
    "maranhão",
    "mato grosso",
    "mato grosso do sul",
    "minas gerais",
    "pará",
    "paraíba",
    "paraná",
    "pernambuco",
    "piauí",
    "rio de janeiro",
    "rio grande do norte",
    "rio grande do sul",
    "rondônia",
    "roraima",
    "santa catarina",
    "são paulo",
    "sergipe",
    "tocantins",
]

def classificar_pergunta(pergunta: str) -> Dict:
    """
    Classifica a pergunta segundo as dimensões temáticas
    presentes nas Visões Semânticas Materializadas (VSM).

    Uma pergunta pode possuir várias intenções simultaneamente.
    Exemplo:
        "Qual a incidência de dengue em Goiás?"
    pode ativar:
        geografico=True
        incidencia=True
    """

    p = normalizar_texto(pergunta)

    # --------------------------------------------------------
    # INTENÇÕES TEMÁTICAS
    # --------------------------------------------------------

    intencoes = {
        "notificacoes":
            any(k in p for k in NOTIFICACOES_CHAVE),

        "temporal":
            any(k in p for k in TEMPORAL_CHAVE),

        "geografico":
            any(k in p for k in GEOGRAFICO_CHAVE),

        "incidencia":
            any(k in p for k in INCIDENCIA_CHAVE),

        "autoctonia":
            any(k in p for k in AUTOCTONIA_CHAVE),

        "sorotipos":
            any(k in p for k in SOROTIPOS_CHAVE),

        "clinico":
            any(k in p for k in CLINICO_CHAVE),

        "doencas_preexistentes":
            any(
                k in p
                for k in DOENCAS_PREEXISTENTES_CHAVE
            ),

        "hospitalizacao":
            any(k in p for k in HOSPITALIZACAO_CHAVE),

        "evolucao":
            any(k in p for k in EVOLUCAO_CHAVE),

        "obitos":
            any(k in p for k in OBITOS_CHAVE),

        "panorama":
            any(k in p for k in PANORAMA_CHAVE),
    }
    # ============================================================
    # DESAMBIGUAÇÃO:
    # EVOLUÇÃO TEMPORAL DAS NOTIFICAÇÕES
    # ============================================================

    padrao_evolucao_notificacoes = (
        (
            "evoluir" in p
            or "evoluiu" in p
            or "evoluiram" in p
            or "evolucao das notificacoes" in p
            or "evolucao de notificacoes" in p
        )
        and intencoes["notificacoes"]
    )

    if padrao_evolucao_notificacoes:
        intencoes["temporal"] = True
        intencoes["evolucao"] = False


    # ============================================================
    # DESAMBIGUAÇÃO:
    # "EVOLUÇÃO" TEMPORAL × CAMPO DE DESFECHO EVOLUCAO
    # ============================================================

    padroes_evolucao_temporal = [
        "ao longo das semanas",
        "ao longo da semana",
        "por semana",
        "entre as semanas",
        "evolucao temporal",
        "serie temporal",
        "tendencia",
        "comportamento temporal",
        "variacao semanal",
    ]

    indicadores_desfecho = [
        "cura",
        "curado",
        "curados",
        "obito",
        "obitos",
        "morte",
        "mortes",
        "desfecho",
        "desfechos",
        "resultado do caso",
    ]

    contexto_temporal_evolucao = (
        intencoes["temporal"]
        and any(
            padrao in p
            for padrao in padroes_evolucao_temporal
        )
    )

    contexto_desfecho_evolucao = any(
        termo in p
        for termo in indicadores_desfecho
    )

    # Quando "evolução" aparece claramente no sentido temporal,
    # não tratamos a palavra isolada como intenção de desfecho.
    if (
        contexto_temporal_evolucao
        and not contexto_desfecho_evolucao
    ):
        intencoes["evolucao"] = False

    # --------------------------------------------------------
    # ANO
    # --------------------------------------------------------

    anos = re.findall(
        r"\b(20\d{2})\b",
        p
    )

    # --------------------------------------------------------
    # SEMANA EPIDEMIOLÓGICA
    # --------------------------------------------------------

    semanas = re.findall(
        r"\b(?:semana|se)\s*(?:epidemiologica\s*)?(\d{1,2})\b",
        p
    )

    # --------------------------------------------------------
    # SIGLAS DE UF
    # --------------------------------------------------------

    ufs_sigla_encontradas = []

    palavras = re.findall(
        r"\b[A-Za-zÀ-ÿ0-9]+\b",
        pergunta
    )

    for token in palavras:

        token_upper = token.upper()

        if token_upper in UF_SIGLAS:
            ufs_sigla_encontradas.append(
                token_upper
            )

    # --------------------------------------------------------
    # NOMES DE UF
    # --------------------------------------------------------

    # --------------------------------------------------------
    # IDENTIFICAÇÃO DE UFs POR NOME
    # --------------------------------------------------------

    # Mantemos uma versão do texto com acentos para evitar
    # ambiguidades como "Pará" × preposição "para".
    p_original = str(pergunta).strip().lower()

    ufs_nome_encontradas = []

    for uf in UF_NOMES:

        uf_original = str(uf).strip().lower()

        # Procura o nome completo da UF preservando acentos.
        padrao = rf"(?<!\w){re.escape(uf_original)}(?!\w)"

        if re.search(
            padrao,
            p_original,
            flags=re.IGNORECASE
        ):
            ufs_nome_encontradas.append(
                uf
            )
    # --------------------------------------------------------
    # TRATAMENTO ESPECIAL PARA "PARÁ" SEM ACENTO
    # --------------------------------------------------------

    if "pará" not in ufs_nome_encontradas:

        padroes_para_estado = [
            r"\bno para\b",
            r"\bdo para\b",
            r"\bestado do para\b",
            r"\bestado de para\b",
        ]

        if any(
            re.search(
                padrao,
                p
            )
            for padrao in padroes_para_estado
        ):
            ufs_nome_encontradas.append(
                "Pará"
            )
    
    # Uma UF explícita torna a pergunta geográfica
    if (
        ufs_sigla_encontradas
        or ufs_nome_encontradas
    ):
        intencoes["geografico"] = True




    # --------------------------------------------------------
    # DETERMINAR DOMÍNIOS SEMÂNTICOS PRIORITÁRIOS
    # --------------------------------------------------------

    dominios = []

    # --------------------------------------------------------
    # LOCALIZAÇÃO EXPLÍCITA TAMBÉM IMPLICA DIMENSÃO GEOGRÁFICA
    # --------------------------------------------------------

    if (
        ufs_sigla_encontradas
        or ufs_nome_encontradas
    ):
        intencoes["geografico"] = True

    if (
        intencoes["clinico"]
        or intencoes["doencas_preexistentes"]
    ):
        dominios.append("clinico")

    if (
        intencoes["hospitalizacao"]
        or intencoes["evolucao"]
        or intencoes["obitos"]
    ):
        dominios.append("desfechos")

    if (
        intencoes["geografico"]
        or intencoes["incidencia"]
        or intencoes["autoctonia"]
    ):
        dominios.append("geografico")

    if intencoes["temporal"]:
        dominios.append("temporal")

    if intencoes["sorotipos"]:
        dominios.append("virologico")

    if intencoes["panorama"]:
        dominios.append("epidemiologico")

    # Remover duplicações preservando a ordem
    dominios = list(dict.fromkeys(dominios))




    # --------------------------------------------------------
    # CLASSIFICAÇÃO DO ESCOPO GEOGRÁFICO
    # --------------------------------------------------------

    tem_uf = bool(
        ufs_sigla_encontradas
        or ufs_nome_encontradas
    )

    escopo_geografico = (
        "uf"
        if tem_uf
        else "nacional"
    )

    return {
        "intencoes": intencoes,

        "dominios": dominios,

        "anos": sorted(
            set(anos)
        ),

        "semanas": sorted(
            set(semanas),
            key=int
        ),

        "ufs_sigla": sorted(
            set(ufs_sigla_encontradas)
        ),

        "ufs_nome": sorted(
            set(
                normalizar_texto(uf)
                for uf in ufs_nome_encontradas
            )
        ),

        "escopo_geografico":
            escopo_geografico,
    }




# ============================================================
# PASSO 4.11
# PROTEÇÃO DE ESCOPO
# ============================================================

TERMOS_DENGUE = [
    "dengue",
    "sinan"
]
TERMOS_FORA_DOMINIO = [
    "gripe",
    "influenza",
    "covid",
    "covid-19",
    "coronavirus",
    "sarampo",
    "tuberculose",
    "hiv",
    "aids",
    "malaria",
    "hepatite",
    "cancer",
]

TERMOS_SEM_COBERTURA = [
    "tratamento",
    "tratar",
    "medicamento",
    "medicamentos",
    "remedio",
    "remedios",
    "vacina",
    "vacinacao",
    "prevencao",
    "prevenir",
    "repelente",
    "transmissao",
    "mosquito",
    "aedes",
    "aedes aegypti"
]


def validar_escopo_pergunta(
    pergunta: str,
    classificacao: Dict
) -> Dict:
    """
    Classifica a pergunta quanto ao escopo da aplicação.

    Estados possíveis:
    - valido
    - sem_cobertura
    - fora_dominio
    """

    p = normalizar_texto(pergunta)

    intencoes = (
        classificacao.get(
            "intencoes",
            {}
        )
        or {}
    )

    # --------------------------------------------------------
    # 1. Verificar se existe alguma intenção coberta
    # --------------------------------------------------------

    intencoes_cobertas = [
        "notificacoes",
        "temporal",
        "geografico",
        "incidencia",
        "autoctonia",
        "sorotipos",
        "clinico",
        "doencas_preexistentes",
        "hospitalizacao",
        "evolucao",
        "obitos",
        "panorama",
    ]

    possui_intencao_coberta = any(
        intencoes.get(
            intencao,
            False
        )
        for intencao
        in intencoes_cobertas
    )

    # --------------------------------------------------------
    # 2. Verificar menção explícita à dengue/SINAN
    # --------------------------------------------------------

    menciona_dengue = any(
        termo in p
        for termo
        in TERMOS_DENGUE
    )

    # --------------------------------------------------------
    # 3. Verificar menção explícita a outro domínio
    # --------------------------------------------------------

    menciona_outro_dominio = any(
        termo in p
        for termo
        in TERMOS_FORA_DOMINIO
    )

    # --------------------------------------------------------
    # 4. Verificar pedido conhecido sem cobertura
    # --------------------------------------------------------

    pede_conteudo_sem_cobertura = any(
        termo in p
        for termo
        in TERMOS_SEM_COBERTURA
    )

    # --------------------------------------------------------
    # 5. Outro domínio explícito
    #
    # Exemplo:
    # "Quantos casos de gripe por UF?"
    # --------------------------------------------------------

    if menciona_outro_dominio:

        return {
            "status": "fora_dominio",
            "motivo": (
                "A pergunta menciona explicitamente uma doença "
                "ou condição fora do escopo das VSMs disponíveis."
            )
        }

    # --------------------------------------------------------
    # 6. Dengue + conteúdo conhecido sem cobertura
    #
    # Exemplo:
    # "Qual o tratamento da dengue?"
    # --------------------------------------------------------

    if (
        menciona_dengue
        and pede_conteudo_sem_cobertura
    ):

        return {
            "status": "sem_cobertura",
            "motivo": (
                "A pergunta é relacionada à dengue, "
                "mas solicita informação que não está "
                "representada nas VSMs disponíveis."
            )
        }

    # --------------------------------------------------------
    # 7. Intenção epidemiológica coberta
    #
    # A aplicação assume dengue/SINAN como contexto padrão
    # quando nenhuma outra doença é explicitamente informada.
    #
    # Exemplos válidos:
    # "Quantos casos foram registrados por UF?"
    # "Qual a incidência em Goiás?"
    # "Quais os sorotipos em São Paulo?"
    # --------------------------------------------------------

    if possui_intencao_coberta:

        return {
            "status": "valido",
            "motivo": (
                "A pergunta possui intenção coberta pelas VSMs "
                "e está dentro do contexto epidemiológico "
                "da aplicação."
            )
        }

    # --------------------------------------------------------
    # 8. Dengue mencionada, mas sem evidência/intenção coberta
    #
    # Exemplo:
    # "Quem descobriu a dengue?"
    # --------------------------------------------------------

    if menciona_dengue:

        return {
            "status": "sem_cobertura",
            "motivo": (
                "A pergunta está relacionada à dengue, "
                "mas o conteúdo solicitado não está "
                "representado nas VSMs disponíveis."
            )
        }

    # --------------------------------------------------------
    # 9. Caso geral fora do domínio
    # --------------------------------------------------------

    return {
        "status": "fora_dominio",
        "motivo": (
            "A pergunta não apresenta intenção epidemiológica "
            "coberta pelas VSMs disponíveis."
        )
    }


# 4.11.2 — Criar respostas determinísticas de bloqueio

# Logo depois, adicione:

def gerar_resposta_escopo(
    resultado_escopo: Dict
) -> str:

    status = resultado_escopo.get(
        "status",
        "fora_dominio"
    )

    if status == "sem_cobertura":

        return (
            "A pergunta está relacionada à dengue, "
            "mas os documentos semânticos disponíveis "
            "não contêm evidências suficientes sobre esse tema. "
            "A aplicação está limitada às informações "
            "representadas nas VSMs derivadas do SINAN."
        )

    if status == "fora_dominio":

        return (
            "A pergunta está fora do escopo desta aplicação. "
            "O sistema responde consultas relacionadas "
            "à vigilância epidemiológica da dengue "
            "com base nas evidências disponíveis nas VSMs "
            "derivadas do SINAN."
        )

    return ""



# # TESTES
# if DEBUG:

#     st.subheader(
#         "Teste da desambiguação de evolução"
#     )

#     perguntas_desambiguacao = [
#         "Qual foi a evolução das notificações ao longo das semanas?",
#         "Qual foi a evolução dos casos quanto à cura e óbito?",
#         "Como evoluíram as notificações semanalmente?",
#         "Quais foram os desfechos dos casos?",
#     ]

#     for pergunta_teste in perguntas_desambiguacao:

#         resultado = classificar_pergunta(
#             pergunta_teste
#         )

#         st.write(
#             pergunta_teste,
#             resultado["intencoes"],
#             resultado["dominios"]
#         )

# 1. valido
#    → pergunta compatível com as VSMs

# 2. sem_cobertura
#    → pergunta relacionada à dengue,
#      mas não representada nas VSMs

# 3. fora_dominio
#    → pergunta sem relação com dengue/
#      vigilância epidemiológica

ESCOPO_VALIDO = "valido"
ESCOPO_SEM_COBERTURA = "sem_cobertura"
ESCOPO_FORA_DOMINIO = "fora_dominio"
TERMOS_SEM_COBERTURA = [
    "tratamento",
    "tratar",
    "tratado",
    "terapia",
    "terapeutico",
    "medicamento",
    "medicamentos",
    "remedio",
    "remedios",
    "vacina",
    "vacinacao",
    "vacinar",
    "prevencao",
    "prevenir",
    "profilaxia",
    "diagnostico",
    "diagnosticar",
]

def classificar_escopo_pergunta(
    pergunta: str,
    query_info: Dict
) -> str:
    """
    Classifica a pergunta quanto ao escopo da aplicação.

    Estados possíveis:
        - valido
        - sem_cobertura
        - fora_dominio
    """

    p = normalizar_texto(pergunta)

    intencoes = query_info.get(
        "intencoes",
        {}
    )

    # ========================================================
    # 1. A PERGUNTA ESTÁ RELACIONADA À DENGUE?
    # ========================================================

    termos_dominio = [
        "dengue",
        "sinan",
        "arbovirose",
        "arboviroses",
        "epidemiologia",
        "epidemiologico",
        "vigilancia epidemiologica",
    ]

    relacionada_dominio = any(
        termo in p
        for termo in termos_dominio
    )

    # ========================================================
    # 2. TEMA RELACIONADO À DENGUE, MAS NÃO COBERTO
    #    PELAS VSMs
    # ========================================================

    solicitacao_sem_cobertura = any(
        termo in p
        for termo in TERMOS_SEM_COBERTURA
    )

    if (
        relacionada_dominio
        and solicitacao_sem_cobertura
    ):
        return ESCOPO_SEM_COBERTURA

    # ========================================================
    # 3. INTENÇÕES REALMENTE COBERTAS PELAS VSMs
    # ========================================================

    intencoes_cobertas = [
        "notificacoes",
        "temporal",
        "geografico",
        "incidencia",
        "autoctonia",
        "sorotipos",
        "clinico",
        "doencas_preexistentes",
        "hospitalizacao",
        "evolucao",
        "obitos",
        "panorama",
    ]

    possui_intencao_coberta = any(
        intencoes.get(
            intencao,
            False
        )
        for intencao in intencoes_cobertas
    )

    if possui_intencao_coberta:
        return ESCOPO_VALIDO

    # ========================================================
    # 4. É SOBRE DENGUE, MAS NÃO HÁ EVIDÊNCIA REPRESENTADA
    # ========================================================

    if relacionada_dominio:
        return ESCOPO_SEM_COBERTURA

    # ========================================================
    # 5. FORA DO DOMÍNIO
    # ========================================================

    return ESCOPO_FORA_DOMINIO

# ========================================================
# função para gerar a mensagem adequada:
# ========================================================

def resposta_fora_escopo(
    classificacao_escopo: str
) -> str:

    if classificacao_escopo == ESCOPO_SEM_COBERTURA:

        return (
            "A pergunta está relacionada à dengue ou à vigilância "
            "epidemiológica, mas a informação solicitada não está "
            "representada nas fontes atualmente indexadas pela aplicação."
        )

    if classificacao_escopo == ESCOPO_FORA_DOMINIO:

        return (
            "Esta pergunta está fora do escopo da aplicação. "
            "O sistema é voltado à análise epidemiológica da dengue "
            "com base nas Visões Semânticas Materializadas construídas "
            "a partir dos dados do SINAN."
        )

    return ""


# # ============================================================
# # TESTE DO NOVO CLASSIFICADOR DE PERGUNTAS
# # ============================================================

# if DEBUG:

#     st.subheader(
#         "Teste do novo classificador"
#     )

#     perguntas_teste = [
#         "Qual o panorama geral da dengue em 2026?",

#         "Quantas notificações ocorreram em Goiás?",

#         "Qual foi a incidência em Goiás?",

#         "Como evoluíram as notificações ao longo das semanas?",

#         "Quais sorotipos foram identificados em São Paulo?",

#         "Quais foram os sintomas mais frequentes em Minas Gerais?",

#         "Quais doenças preexistentes foram mais frequentes?",

#         "Qual a proporção de hospitalização por UF?",

#         "Qual foi a evolução dos registros?",

#         "Qual foi o perfil dos óbitos?",

#         "Qual a proporção de casos autóctones?",

#         "Qual a concordância entre residência e local provável de infecção?",
#     ]

#     resultados_teste = []

#     for pergunta_teste in perguntas_teste:

#         classificacao = classificar_pergunta(
#             pergunta_teste
#         )

#         resultados_teste.append({
#             "pergunta":
#                 pergunta_teste,

#             "dominios":
#                 ", ".join(
#                     classificacao["dominios"]
#                 ),

#             "intencoes":
#                 ", ".join(
#                     nome
#                     for nome, ativo
#                     in classificacao[
#                         "intencoes"
#                     ].items()
#                     if ativo
#                 ),

#             "ufs":
#                 ", ".join(
#                     classificacao[
#                         "ufs_nome"
#                     ]
#                     + classificacao[
#                         "ufs_sigla"
#                     ]
#                 ),

#             "anos":
#                 ", ".join(
#                     classificacao["anos"]
#                 ),

#             "semanas":
#                 ", ".join(
#                     classificacao["semanas"]
#                 ),

#             "escopo":
#                 classificacao[
#                     "escopo_geografico"
#                 ]
#         })

#     df_teste_classificador = pd.DataFrame(
#         resultados_teste
#     )

#     st.dataframe(
#         df_teste_classificador,
#         use_container_width=True
#     )


# # ============================================================
# # TESTE DA CLASSIFICAÇÃO DE ESCOPO
# # ============================================================

# if DEBUG:

#     st.subheader("Teste da classificação de escopo")

#     perguntas_teste_escopo = [

#         # Cobertas pelas VSMs
#         "Qual foi a incidência em Goiás?",
#         "Quais sorotipos foram identificados em São Paulo?",
#         "Qual o perfil dos óbitos?",
#         "Qual a incidência no Pará?",
#         "Qual a incidência no Para?",

#         # Dengue, mas sem cobertura
#         "Qual é o tratamento recomendado para dengue?",
#         "Existe vacina contra dengue?",
#         "Como prevenir a dengue?",

#         # Fora do domínio
#         "Quem ganhou a Copa do Mundo de 2022?",
#         "Qual é a capital da França?",
#         "Quais são os sintomas da gripe?",
#     ]

#     resultados_escopo = []

#     for pergunta_teste in perguntas_teste_escopo:

#         query_info = classificar_pergunta(
#             pergunta_teste
#         )

#         classificacao = classificar_escopo_pergunta(
#             pergunta_teste,
#             query_info
#         )

#         intencoes_ativas = [
#             nome
#             for nome, ativa
#             in query_info.get(
#                 "intencoes",
#                 {}
#             ).items()
#             if ativa
#         ]

#         dominios = query_info.get(
#             "dominios",
#             []
#         )

#         resultados_escopo.append({
#             "pergunta": pergunta_teste,

#             "classificacao_escopo":
#                 classificacao,

#             "dominios":
#                 ", ".join(dominios)
#                 if dominios
#                 else "-",

#             "intencoes":
#                 ", ".join(intencoes_ativas)
#                 if intencoes_ativas
#                 else "-",
#         })

#     df_teste_escopo = pd.DataFrame(
#         resultados_escopo
#     )

#     st.dataframe(
#         df_teste_escopo,
#         use_container_width=True
#     )



# ============================================================

perguntas_teste_escopo = [

    # Cobertas pelas VSMs
    "Qual foi a incidência em Goiás?",
    "Quais sorotipos foram identificados em São Paulo?",
    "Qual o perfil dos óbitos?",
    "Qual a incidência no Pará?",
    "Qual a incidência no Para?",

    # Dengue, mas sem cobertura
    "Qual é o tratamento recomendado para dengue?",
    "Existe vacina contra dengue?",
    "Como prevenir a dengue?",

    # Fora do domínio
    "Quem ganhou a Copa do Mundo de 2022?",
    "Qual é a capital da França?",
    "Quais são os sintomas da gripe?",
]


# ============================================================
# IDENTIFICAR CONSULTA DE TOTAL DE CASOS POR UF
# ============================================================

def eh_pergunta_total_casos_por_uf(
    pergunta: str,
    query_info: Dict
) -> bool:

    pergunta_norm = normalizar_texto(
        pergunta
    )

    intencoes = query_info.get(
        "intencoes",
        {}
    )

    # --------------------------------------------------------
    # Consulta explicitamente por UF / estado
    # --------------------------------------------------------

    menciona_uf_ou_estado = (
        "uf" in pergunta_norm
        or "ufs" in pergunta_norm
        or "estado" in pergunta_norm
        or "estados" in pergunta_norm
        or "por uf" in pergunta_norm
        or "por estado" in pergunta_norm
    )

    # --------------------------------------------------------
    # Consulta nacional de distribuição geográfica
    #
    # Exemplos:
    # "Como os casos se distribuem geograficamente?"
    # "Qual a distribuição geográfica dos casos?"
    # --------------------------------------------------------

    menciona_distribuicao_geografica = (
        "distribuicao geografica"
        in pergunta_norm

        or "distribuem geograficamente"
        in pergunta_norm

        or "distribuidos geograficamente"
        in pergunta_norm

        or "distribuicao espacial"
        in pergunta_norm
    )

    return (
        intencoes.get(
            "notificacoes",
            False
        )

        and (
            menciona_uf_ou_estado
            or menciona_distribuicao_geografica
        )

        and not intencoes.get(
            "incidencia",
            False
        )

        and not intencoes.get(
            "sorotipos",
            False
        )

        and not intencoes.get(
            "clinico",
            False
        )

        and not intencoes.get(
            "hospitalizacao",
            False
        )

        and not intencoes.get(
            "obitos",
            False
        )
    )



# ============================================================
# IDENTIFICAR CONSULTA COMBINADA:
# SINTOMAS + SOROTIPOS POR UF
# ============================================================

def eh_pergunta_sintomas_sorotipos_por_uf(
    pergunta: str,
    query_info: Dict
) -> bool:

    pergunta_norm = normalizar_texto(
        pergunta
    )

    intencoes = query_info.get(
        "intencoes",
        {}
    )

    menciona_uf_ou_estado = (
        "uf" in pergunta_norm
        or "estado" in pergunta_norm
        or "estados" in pergunta_norm
        or "por uf" in pergunta_norm
        or "por estado" in pergunta_norm
        or "cada estado" in pergunta_norm
        or "cada uf" in pergunta_norm
    )

    return (
        intencoes.get(
            "clinico",
            False
        )
        and intencoes.get(
            "sorotipos",
            False
        )
        and menciona_uf_ou_estado
    )

# ============================================================
# IDENTIFICAR CONSULTA DE SOROTIPOS POR UF
# ============================================================

def eh_pergunta_sorotipos_por_uf(
    pergunta: str,
    query_info: Dict
) -> bool:

    pergunta_norm = normalizar_texto(
        pergunta
    )

    intencoes = query_info.get(
        "intencoes",
        {}
    )

    menciona_uf_ou_estado = (
        "uf" in pergunta_norm
        or "estado" in pergunta_norm
        or "estados" in pergunta_norm
        or "por uf" in pergunta_norm
        or "por estado" in pergunta_norm
    )

    return (
        intencoes.get(
            "sorotipos",
            False
        )
        and menciona_uf_ou_estado
        and not intencoes.get(
            "clinico",
            False
        )
    )


# ============================================================
# IDENTIFICAR CONSULTA CLÍNICA POR UF
# ============================================================

def eh_pergunta_sintomas_por_estado(
    pergunta: str,
    query_info: Dict
) -> bool:

    pergunta_norm = normalizar_texto(
        pergunta
    )

    intencoes = query_info.get(
        "intencoes",
        {}
    )

    menciona_estado = (
        "estado" in pergunta_norm
        or "estados" in pergunta_norm
        or "uf" in pergunta_norm
        or "por estado" in pergunta_norm
        or "por uf" in pergunta_norm
    )

    return (
        intencoes.get(
            "clinico",
            False
        )
        and menciona_estado
        and not intencoes.get(
            "sorotipos",
            False
        )
    )

# ============================================================
# IDENTIFICAR CONSULTA DE HOSPITALIZAÇÃO POR UF
# ============================================================

def eh_pergunta_hospitalizacao_por_uf(
    pergunta: str,
    query_info: Dict
) -> bool:

    pergunta_norm = normalizar_texto(
        pergunta
    )

    intencoes = query_info.get(
        "intencoes",
        {}
    )

    menciona_uf_ou_estado = (
        "uf" in pergunta_norm
        or "estado" in pergunta_norm
        or "estados" in pergunta_norm
        or "por uf" in pergunta_norm
        or "por estado" in pergunta_norm
    )

    return (
        intencoes.get(
            "hospitalizacao",
            False
        )
        and menciona_uf_ou_estado
        and not intencoes.get(
            "sorotipos",
            False
        )
        and not intencoes.get(
            "clinico",
            False
        )
    )

def eh_pergunta_obitos_por_uf(
    pergunta: str,
    query_info: Dict
) -> bool:

    pergunta_norm = normalizar_texto(
        pergunta
    )

    intencoes = query_info.get(
        "intencoes",
        {}
    )

    menciona_uf_ou_estado = (
        "uf" in pergunta_norm
        or "ufs" in pergunta_norm
        or "estado" in pergunta_norm
        or "estados" in pergunta_norm
        or "por uf" in pergunta_norm
        or "por estado" in pergunta_norm
    )

    return (
        intencoes.get(
            "obitos",
            False
        )
        and menciona_uf_ou_estado
        and not intencoes.get(
            "sorotipos",
            False
        )
        and not intencoes.get(
            "clinico",
            False
        )
    )

# ============================================================
# 6. MAPEAMENTO ENTRE INTENÇÕES E TIPOS DE VSM
# ============================================================

TIPOS_VSM_POR_INTENCAO = {

    "temporal": [
        "perfil_temporal_uf",
        "panorama_temporal_nacional",
    ],

    "geografico": [
        "distribuicao_geografica_uf",
        "panorama_geografico_nacional",
    ],

    "incidencia": [
        "distribuicao_geografica_uf",
        "panorama_geografico_nacional",
    ],

    "autoctonia": [
        "distribuicao_geografica_uf",
        "panorama_geografico_nacional",
    ],

    "sorotipos": [
        "perfil_virologico_uf",
        "panorama_virologico_nacional",
    ],

    "clinico": [
        "perfil_clinico_uf",
        "panorama_clinico_nacional",
    ],

    "doencas_preexistentes": [
        "perfil_clinico_uf",
        "panorama_clinico_nacional",
    ],

    "hospitalizacao": [
        "desfechos_uf",
        "panorama_desfechos_nacional",
    ],

    "evolucao": [
        "desfechos_uf",
        "panorama_desfechos_nacional",
    ],

    "obitos": [
        "perfil_obitos_uf",
        "panorama_obitos_nacional",
    ],

    "panorama": [
        "panorama_epidemiologico_integrado",
    ],
}

# Observe que notificacoes não aparece aqui propositalmente. É uma intenção transversal. Uma pergunta sobre notificações
#  pode ser temporal, geográfica ou fazer parte de um panorama.

# ============================================================
# 6. PESOS POR CATEGORIA E DOCUMENT TYPE
# ============================================================

def definir_pesos(query_info: Dict) -> Dict:
    """
    Define prioridades para recuperação das VSMs.

    A priorização distingue:
    - intenção temática principal;
    - dimensão geográfica usada como filtro;
    - domínio semântico;
    - tipo de documento;
    - escopo nacional ou estadual.
    """

    intencoes = query_info.get(
        "intencoes",
        {}
    )

    dominios_prioritarios = set(
        query_info.get(
            "dominios",
            []
        )
    )

    tipos_prioritarios = set()

    # --------------------------------------------------------
    # IDENTIFICAR SE EXISTE UMA INTENÇÃO TEMÁTICA ESPECÍFICA
    # --------------------------------------------------------

    intencoes_tematicas = [
        "temporal",
        "incidencia",
        "autoctonia",
        "sorotipos",
        "clinico",
        "doencas_preexistentes",
        "hospitalizacao",
        "evolucao",
        "obitos",
        "panorama",
    ]

    existe_intencao_tematica = any(
        intencoes.get(
            intencao,
            False
        )
        for intencao
        in intencoes_tematicas
    )

    # --------------------------------------------------------
    # TIPOS PRIORITÁRIOS
    # --------------------------------------------------------

    for intencao, ativa in intencoes.items():

        if not ativa:
            continue

        # "geografico" pode ser apenas um recorte de UF.
        # Se existe outra intenção temática específica,
        # não usamos automaticamente os documentos
        # geográficos como documentos principais.
        if (
            intencao == "geografico"
            and existe_intencao_tematica
        ):
            continue

        for tipo in TIPOS_VSM_POR_INTENCAO.get(
            intencao,
            []
        ):
            tipos_prioritarios.add(
                normalizar_texto(
                    tipo
                )
            )

    return {

        "peso_base": 1.0,

        "peso_dominio_prioritario": 3.0,

        "peso_tipo_prioritario": 4.0,

        "peso_ano": 1.2,

        "peso_semana": 1.4,

        "peso_uf": 2.5,

        "peso_escopo_uf": 2.0,

        "peso_escopo_nacional": 1.0,

        "dominios_prioritarios":
            dominios_prioritarios,

        "tipos_prioritarios":
            tipos_prioritarios,
    }

# ============================================================
# 7. SCORE HEURÍSTICO
# ============================================================

def score_heuristico(
    pergunta: str,
    doc: Document,
    query_info: Dict,
    pesos: Dict
) -> Tuple[float, List[str]]:
    """
    Calcula um score heurístico para uma VSM.

    O score não substitui a similaridade vetorial.
    Ele serve como etapa adicional de priorização
    baseada na estrutura semântica das VSMs.
    """

    score = 0.0
    justificativas = []

    # --------------------------------------------------------
    # DADOS DO DOCUMENTO
    # --------------------------------------------------------

    dominio = normalizar_texto(
        doc.metadata.get(
            "dominio",
            doc.metadata.get(
                "categoria",
                ""
            )
        )
    )

    tipo_documento = normalizar_texto(
        doc.metadata.get(
            "tipo_documento",
            doc.metadata.get(
                "document_type",
                ""
            )
        )
    )

    texto = normalizar_texto(
        doc.page_content
    )

    metadata_text = normalizar_texto(
        json.dumps(
            doc.metadata,
            ensure_ascii=False
        )
    )

    texto_total = (
        texto
        + " "
        + metadata_text
    )

    # --------------------------------------------------------
    # 1. PESO BASE
    # --------------------------------------------------------

    peso_base = pesos[
        "peso_base"
    ]

    score += peso_base

    justificativas.append(
        f"base={peso_base}"
    )

    # --------------------------------------------------------
    # 2. DOMÍNIO PRIORITÁRIO
    # --------------------------------------------------------

    dominios_prioritarios = {
        normalizar_texto(d)
        for d in pesos[
            "dominios_prioritarios"
        ]
    }

    if (
        dominio
        and dominio in dominios_prioritarios
    ):

        bonus = pesos[
            "peso_dominio_prioritario"
        ]

        score += bonus

        justificativas.append(
            f"dominio={dominio}:+{bonus}"
        )

    # --------------------------------------------------------
    # 3. TIPO DE DOCUMENTO PRIORITÁRIO
    # --------------------------------------------------------

    tipos_prioritarios = pesos[
        "tipos_prioritarios"
    ]

    if (
        tipo_documento
        and tipo_documento in tipos_prioritarios
    ):

        bonus = pesos[
            "peso_tipo_prioritario"
        ]

        score += bonus

        justificativas.append(
            f"tipo={tipo_documento}:+{bonus}"
        )

    # --------------------------------------------------------
    # 4. ANO
    # --------------------------------------------------------

    for ano in query_info.get(
        "anos",
        []
    ):

        if str(ano) in texto_total:

            bonus = pesos[
                "peso_ano"
            ]

            score += bonus

            justificativas.append(
                f"ano={ano}:+{bonus}"
            )

    # --------------------------------------------------------
    # 5. SEMANA EPIDEMIOLÓGICA
    # --------------------------------------------------------

    for semana in query_info.get(
        "semanas",
        []
    ):

        semana_doc_ini = str(
            doc.metadata.get(
                "semana_inicial",
                ""
            )
        )

        semana_doc_fim = str(
            doc.metadata.get(
                "semana_final",
                ""
            )
        )

        encontrou_semana = False

        try:

            semana_int = int(
                semana
            )

            if (
                semana_doc_ini
                and semana_doc_fim
            ):

                inicio = int(
                    semana_doc_ini
                )

                fim = int(
                    semana_doc_fim
                )

                encontrou_semana = (
                    inicio
                    <= semana_int
                    <= fim
                )

        except (
            ValueError,
            TypeError
        ):
            pass

        if (
            encontrou_semana
            or str(semana) in texto
        ):

            bonus = pesos[
                "peso_semana"
            ]

            score += bonus

            justificativas.append(
                f"semana={semana}:+{bonus}"
            )

    # --------------------------------------------------------
    # 6. UF — NOME
    # --------------------------------------------------------

    uf_doc = normalizar_texto(
        doc.metadata.get(
            "uf_nome",
            doc.metadata.get(
                "uf",
                ""
            )
        )
    )

    for uf_nome in query_info.get(
        "ufs_nome",
        []
    ):

        uf_nome_norm = (
            normalizar_texto(
                uf_nome
            )
        )

        if (
            uf_nome_norm
            and (
                uf_nome_norm == uf_doc
                or uf_nome_norm in texto_total
            )
        ):

            bonus = pesos[
                "peso_uf"
            ]

            score += bonus

            justificativas.append(
                f"uf={uf_nome_norm}:+{bonus}"
            )

            break

    # --------------------------------------------------------
    # 7. UF — SIGLA
    # --------------------------------------------------------

    for uf_sigla in query_info.get(
        "ufs_sigla",
        []
    ):

        uf_sigla_norm = (
            normalizar_texto(
                uf_sigla
            )
        )

        if (
            uf_sigla_norm
            and uf_sigla_norm in texto_total
        ):

            bonus = pesos[
                "peso_uf"
            ]

            score += bonus

            justificativas.append(
                f"uf_sigla={uf_sigla}:+{bonus}"
            )

            break

    # --------------------------------------------------------
    # 8. ADEQUAÇÃO DO ESCOPO
    # --------------------------------------------------------

    pergunta_tem_uf = bool(
        query_info.get(
            "ufs_nome"
        )
        or query_info.get(
            "ufs_sigla"
        )
    )

    documento_tem_uf = bool(
        str(
            doc.metadata.get(
                "codigo_uf",
                ""
            )
        ).strip()
    )

    if (
        pergunta_tem_uf
        and documento_tem_uf
    ):

        bonus = pesos[
            "peso_escopo_uf"
        ]

        score += bonus

        justificativas.append(
            f"escopo_uf:+{bonus}"
        )

    elif (
        not pergunta_tem_uf
        and not documento_tem_uf
    ):

        bonus = pesos[
            "peso_escopo_nacional"
        ]

        score += bonus

        justificativas.append(
            f"escopo_nacional:+{bonus}"
        )

    return (
        score,
        justificativas
    )

# Essa função passa a usar os metadados que acabamos de validar:
# dominio
# tipo_documento
# codigo_uf
# uf_nome
# ano
# semana_inicial
# semana_final

# E mantém fallback temporário para:
# categoria
# document_type
# uf


# ============================================================
# 8. FILTRAGEM POR DOCUMENT TYPE
# ============================================================

def filtrar_por_document_type(
    documentos: List[Document],
    query_info: Dict,
    pergunta: str = ""
) -> List[Document]:
    """
    Filtra documentos recuperados de acordo com
    os domínios e tipos de VSM identificados na pergunta.

    O nome da função é mantido temporariamente
    por compatibilidade com o restante do aplicativo.
    """

    if not documentos:
        return []

    intencoes = query_info.get(
        "intencoes",
        {}
    )

    dominios_prioritarios = {
        normalizar_texto(d)
        for d in query_info.get(
            "dominios",
            []
        )
    }

    # --------------------------------------------------------
    # TIPOS PRIORITÁRIOS
    # --------------------------------------------------------

    tipos_prioritarios = set()

    intencoes_tematicas = [
        "temporal",
        "incidencia",
        "autoctonia",
        "sorotipos",
        "clinico",
        "doencas_preexistentes",
        "hospitalizacao",
        "evolucao",
        "obitos",
        "panorama",
    ]

    existe_intencao_tematica = any(
        intencoes.get(
            intencao,
            False
        )
        for intencao
        in intencoes_tematicas
    )

    for intencao, ativa in intencoes.items():

        if not ativa:
            continue

        if (
            intencao == "geografico"
            and existe_intencao_tematica
        ):
            continue

        for tipo in TIPOS_VSM_POR_INTENCAO.get(
            intencao,
            []
        ):

            tipos_prioritarios.add(
                normalizar_texto(
                    tipo
                )
            )

    # --------------------------------------------------------
    # PANORAMA GERAL
    # --------------------------------------------------------

    if intencoes.get(
        "panorama",
        False
    ):

        docs_panorama = [
            doc
            for doc in documentos
            if normalizar_texto(
                doc.metadata.get(
                    "tipo_documento",
                    ""
                )
            )
            == "panorama_epidemiologico_integrado"
        ]

        if docs_panorama:
            return docs_panorama

    # --------------------------------------------------------
    # FILTRO POR DOMÍNIO
    # --------------------------------------------------------

    docs_dominio = []

    if dominios_prioritarios:

        for doc in documentos:

            dominio = normalizar_texto(
                doc.metadata.get(
                    "dominio",
                    doc.metadata.get(
                        "categoria",
                        ""
                    )
                )
            )

            if dominio in dominios_prioritarios:

                docs_dominio.append(
                    doc
                )

    else:

        docs_dominio = documentos

    # --------------------------------------------------------
    # FILTRO POR TIPO
    # --------------------------------------------------------

    docs_tipo = []

    if tipos_prioritarios:

        for doc in docs_dominio:

            tipo = normalizar_texto(
                doc.metadata.get(
                    "tipo_documento",
                    doc.metadata.get(
                        "document_type",
                        ""
                    )
                )
            )

            if tipo in tipos_prioritarios:

                docs_tipo.append(
                    doc
                )

    # Se encontrou tipos específicos, usa-os.
    if docs_tipo:
        candidatos = docs_tipo

    elif docs_dominio:
        candidatos = docs_dominio

    else:
        candidatos = documentos

    # --------------------------------------------------------
    # FILTRO POR UF EXPLÍCITA
    # --------------------------------------------------------

    ufs_nome = {
        normalizar_texto(uf)
        for uf in query_info.get(
            "ufs_nome",
            []
        )
    }

    if ufs_nome:

        docs_uf = [
            doc
            for doc in candidatos
            if normalizar_texto(
                doc.metadata.get(
                    "uf_nome",
                    doc.metadata.get(
                        "uf",
                        ""
                    )
                )
            )
            in ufs_nome
        ]

        if docs_uf:
            candidatos = docs_uf

    return candidatos

# # ============================================================
# # TESTE DA PRIORIZAÇÃO DAS VSMs — SEM CHROMA
# # ============================================================

# if DEBUG:

#     st.subheader(
#         "Teste da priorização semântica das VSMs"
#     )

#     perguntas_teste_priorizacao = [

#         "Quais sorotipos foram identificados em São Paulo?",

#         "Quais foram os sintomas mais frequentes em Minas Gerais?",

#         "Qual foi a incidência em Goiás?",

#         "Qual a proporção de hospitalização por UF?",

#         "Qual foi o perfil dos óbitos?",

#         "Qual a proporção de casos autóctones?",

#         "Como evoluíram as notificações ao longo das semanas?",

#         "Qual o panorama geral da dengue em 2026?",
#     ]

#     documentos_teste = (
#         carregar_documentos_vsm()
#     )

#     resultados = []

#     for pergunta_teste in perguntas_teste_priorizacao:

#         query_info = classificar_pergunta(
#             pergunta_teste
#         )

#         pesos = definir_pesos(
#             query_info
#         )

#         documentos_filtrados = (
#             filtrar_por_document_type(
#                 documentos_teste,
#                 query_info,
#                 pergunta_teste
#             )
#         )

#         docs_com_score = []

#         for doc in documentos_filtrados:

#             score, justificativas = (
#                 score_heuristico(
#                     pergunta_teste,
#                     doc,
#                     query_info,
#                     pesos
#                 )
#             )

#             docs_com_score.append(
#                 (
#                     score,
#                     doc,
#                     justificativas
#                 )
#             )

#         docs_ordenados = sorted(
#             docs_com_score,
#             key=lambda x: x[0],
#             reverse=True
#         )

#         # Mostrar somente os 5 primeiros
#         for posicao, (
#             score,
#             doc,
#             justificativas
#         ) in enumerate(
#             docs_ordenados[:5],
#             start=1
#         ):

#             resultados.append({

#                 "pergunta":
#                     pergunta_teste,

#                 "posicao":
#                     posicao,

#                 "score":
#                     round(
#                         score,
#                         2
#                     ),

#                 "document_id":
#                     doc.metadata.get(
#                         "document_id"
#                     ),

#                 "dominio":
#                     doc.metadata.get(
#                         "dominio"
#                     ),

#                 "tipo_documento":
#                     doc.metadata.get(
#                         "tipo_documento"
#                     ),

#                 "uf":
#                     doc.metadata.get(
#                         "uf"
#                     ),

#                 "justificativa":
#                     " | ".join(
#                         justificativas
#                     ),
#             })

#     df_teste_priorizacao = pd.DataFrame(
#         resultados
#     )

#     st.dataframe(
#         df_teste_priorizacao,
#         use_container_width=True
#     )

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

# def filtrar_docs_total_casos_por_uf(documentos: List[Document], query_info: Dict) -> List[Document]:
#     docs_casos = [
#         doc for doc in documentos
#         if normalizar_texto(doc.metadata.get("document_type", "")) == "total_casos_uf_ano"
#     ]

#     if query_info["anos"]:
#         anos_desejados = set(query_info["anos"])
#         docs_casos = [
#             doc for doc in docs_casos
#             if str(doc.metadata.get("nu_ano", "")) in anos_desejados
#             or str(doc.metadata.get("ano_normalizado", "")) in anos_desejados
#         ]

#     if query_info["ufs_nome"]:
#         ufs_nome_desejadas = set(query_info["ufs_nome"])
#         docs_casos = [
#             doc for doc in docs_casos
#             if normalizar_texto(doc.metadata.get("uf_name", "")) in ufs_nome_desejadas
#             or normalizar_texto(doc.metadata.get("uf_normalizada", "")) in ufs_nome_desejadas
#         ]

#     if query_info["ufs_sigla"]:
#         siglas_desejadas = set(query_info["ufs_sigla"])
#         docs_casos = [
#             doc for doc in docs_casos
#             if str(doc.metadata.get("sg_uf_not", "")).upper() in siglas_desejadas
#             or str(doc.metadata.get("uf_normalizada", "")).upper() in siglas_desejadas
#         ]

#     return sorted(
#         docs_casos,
#         key=lambda d: numero_seguro(d.metadata.get("total_casos", d.metadata.get("num_cases_normalizado", 0)), 0),
#         reverse=True
#     )
def filtrar_docs_total_casos_por_uf(
    documentos: List[Document],
    query_info: Dict
) -> List[Document]:

    # --------------------------------------------------------
    # 1. Selecionar VSMs geográficas estaduais
    # --------------------------------------------------------

    docs_casos = [
        doc
        for doc in documentos
        if (
            normalizar_texto(
                doc.metadata.get(
                    "dominio",
                    ""
                )
            ) == "geografico"
            and
            normalizar_texto(
                doc.metadata.get(
                    "tipo_documento",
                    ""
                )
            ) == "distribuicao_geografica_uf"
        )
    ]

    # --------------------------------------------------------
    # 2. Filtrar ano, se informado
    # --------------------------------------------------------

    anos_desejados = {
        str(ano)
        for ano in query_info.get(
            "anos",
            []
        )
    }

    if anos_desejados:

        docs_casos = [
            doc
            for doc in docs_casos
            if str(
                doc.metadata.get(
                    "ano",
                    ""
                )
            ) in anos_desejados
        ]

    # --------------------------------------------------------
    # 3. Filtrar UF, se informada
    # --------------------------------------------------------

    ufs_desejadas = {
        normalizar_texto(uf)
        for uf in query_info.get(
            "ufs_nome",
            []
        )
    }

    if ufs_desejadas:

        docs_casos = [
            doc
            for doc in docs_casos
            if normalizar_texto(
                doc.metadata.get(
                    "uf_nome",
                    ""
                )
            ) in ufs_desejadas
        ]

    return sorted(
        docs_casos,
        key=lambda doc:
            normalizar_texto(
                doc.metadata.get(
                    "uf_nome",
                    ""
                )
            )
    )
# ============================================================

def filtrar_docs_obitos_por_uf(
    documentos: List[Document],
    query_info: Dict,
    pergunta: str = ""
) -> List[Document]:

    docs = [
        doc
        for doc in documentos
        if (
            normalizar_texto(
                doc.metadata.get(
                    "dominio",
                    ""
                )
            ) == "desfechos"

            and

            normalizar_texto(
                doc.metadata.get(
                    "tipo_documento",
                    doc.metadata.get(
                        "document_type",
                        ""
                    )
                )
            ) == "perfil_obitos_uf"
        )
    ]

    # --------------------------------------------------------
    # FILTRO POR ANO
    # --------------------------------------------------------

    if query_info.get("anos"):

        anos_desejados = set(
            query_info["anos"]
        )

        docs = [
            doc
            for doc in docs
            if str(
                doc.metadata.get(
                    "ano",
                    ""
                )
            ) in anos_desejados
        ]

    # --------------------------------------------------------
    # FILTRO POR UF ESPECÍFICA
    # --------------------------------------------------------

    if query_info.get("ufs_nome"):

        ufs_desejadas = {
            normalizar_texto(uf)
            for uf in query_info[
                "ufs_nome"
            ]
        }

        docs = [
            doc
            for doc in docs
            if normalizar_texto(
                doc.metadata.get(
                    "uf_nome",
                    doc.metadata.get(
                        "uf",
                        ""
                    )
                )
            ) in ufs_desejadas
        ]

    # --------------------------------------------------------
    # ORDENAÇÃO PELO TOTAL DE ÓBITOS
    # --------------------------------------------------------

    def total_obitos(doc):

        itens = extrair_obitos_vsm(
            doc
        )

        return sum(
            numero_seguro(valor, 0)
            for _, valor in itens
        )

    return sorted(
        docs,
        key=total_obitos,
        reverse=True
    )
# ============================================================
def responder_obitos_por_uf(
    documentos: List[Document],
    pergunta: str
) -> Tuple[
    str,
    List[Document],
    str
]:

    query_info = classificar_pergunta(
        pergunta
    )

    docs_ordenados = (
        filtrar_docs_obitos_por_uf(
            documentos=documentos,
            query_info=query_info,
            pergunta=pergunta
        )
    )

    if not docs_ordenados:

        return (
            "Não encontrei documentos de "
            "óbitos por UF compatíveis "
            "com a pergunta.",
            [],
            "deterministico_obitos"
        )

    linhas = []

    for posicao, doc in enumerate(
        docs_ordenados,
        start=1
    ):

        uf = str(
            doc.metadata.get(
                "uf_nome",
                doc.metadata.get(
                    "uf",
                    ""
                )
            )
        )

        ano = str(
            doc.metadata.get(
                "ano",
                ""
            )
        )

        itens = extrair_obitos_vsm(
            doc
        )

        mapa = {
            normalizar_texto(nome):
                numero_seguro(valor, 0)
            for nome, valor in itens
        }

        obito_agravo = mapa.get(
            "obito pelo agravo",
            0
        )

        obito_outras = mapa.get(
            "obito por outras causas",
            0
        )

        obito_investigacao = mapa.get(
            "obito em investigacao",
            0
        )

        total_obitos = (
            obito_agravo
            + obito_outras
            + obito_investigacao
        )

        linhas.append(
            (
                f"{posicao}. {uf}: "
                f"{total_obitos:,} óbitos "
                f"(pelo agravo: "
                f"{obito_agravo:,}; "
                f"outras causas: "
                f"{obito_outras:,}; "
                f"em investigação: "
                f"{obito_investigacao:,}; "
                f"ano {ano})"
            )
            .replace(",", ".")
        )

    cabecalho = (
        "Distribuição de óbitos por UF. "
        f"Foram encontradas "
        f"{len(docs_ordenados)} UFs "
        "com VSMs de óbitos."
    )

    resposta = (
        cabecalho
        + "\n\n"
        + "\n".join(linhas)
    )

    return (
        resposta,
        docs_ordenados,
        "deterministico_obitos"
    )
# ===========================================================

def filtrar_docs_hospitalizacao_por_uf(
    documentos: List[Document],
    query_info: Dict,
    pergunta: str = ""
) -> List[Document]:

    docs = [
        doc
        for doc in documentos
        if (
            normalizar_texto(
                doc.metadata.get(
                    "dominio",
                    ""
                )
            ) == "desfechos"

            and

            normalizar_texto(
                doc.metadata.get(
                    "tipo_documento",
                    doc.metadata.get(
                        "document_type",
                        ""
                    )
                )
            ) == "desfechos_uf"
        )
    ]
    # --------------------------------------------------------
    # FILTRO POR ANO
    # --------------------------------------------------------

    if query_info.get("anos"):

        anos_desejados = set(
            query_info["anos"]
        )

        docs = [
            doc
            for doc in docs
            if str(
                doc.metadata.get(
                    "nu_ano",
                    doc.metadata.get(
                        "ano_normalizado",
                        doc.metadata.get(
                            "ano",
                            ""
                        )
                    )
                )
            )
            in anos_desejados
        ]

    # --------------------------------------------------------
    # FILTRO POR NOME DA UF
    # --------------------------------------------------------

    if query_info.get("ufs_nome"):

        ufs_nome_desejadas = {
            normalizar_texto(uf)
            for uf in query_info["ufs_nome"]
        }

        docs = [
            doc
            for doc in docs
            if (
                normalizar_texto(
                    doc.metadata.get(
                        "uf_name",
                        ""
                    )
                )
                in ufs_nome_desejadas

                or

                normalizar_texto(
                    doc.metadata.get(
                        "uf_normalizada",
                        ""
                    )
                )
                in ufs_nome_desejadas

                or

                normalizar_texto(
                    doc.metadata.get(
                        "uf",
                        ""
                    )
                )
                in ufs_nome_desejadas
            )
        ]

    # --------------------------------------------------------
    # FILTRO POR SIGLA DA UF
    # --------------------------------------------------------

    if query_info.get("ufs_sigla"):

        siglas_desejadas = {
            str(uf).upper()
            for uf in query_info["ufs_sigla"]
        }

        docs = [
            doc
            for doc in docs
            if (
                str(
                    doc.metadata.get(
                        "sg_uf_not",
                        ""
                    )
                ).upper()
                in siglas_desejadas

                or

                str(
                    doc.metadata.get(
                        "uf",
                        ""
                    )
                ).upper()
                in siglas_desejadas

                or

                str(
                    doc.metadata.get(
                        "uf_sigla",
                        ""
                    )
                ).upper()
                in siglas_desejadas

                or

                str(
                    doc.metadata.get(
                        "uf_normalizada",
                        ""
                    )
                ).upper()
                in siglas_desejadas
            )
        ]

    # --------------------------------------------------------
    # DEFINIR CRITÉRIO DE ORDENAÇÃO
    # --------------------------------------------------------

    pergunta_norm = normalizar_texto(
        pergunta
    )

    ordenar_por_taxa = any(
        termo in pergunta_norm
        for termo in [
            "taxa",
            "proporcao",
            "proporcoes",
            "percentual",
            "percentuais",
        ]
    )

    # --------------------------------------------------------
    # EXTRAIR MÉTRICAS
    # --------------------------------------------------------

    def extrair_metricas(doc):

        hospitalizacao = (
            extrair_hospitalizacao_vsm(
                doc
            )
        )

        mapa = {
            normalizar_texto(nome): valor
            for nome, valor
            in hospitalizacao
        }

        internacoes = numero_seguro(
            mapa.get(
                "hospitalizados",
                0
            ),
            0
        )

        nao_hospitalizados = numero_seguro(
            mapa.get(
                "nao hospitalizados",
                0
            ),
            0
        )

        avaliaveis = (
            internacoes
            + nao_hospitalizados
        )

        taxa = (
            (
                internacoes
                / avaliaveis
            )
            * 100
            if avaliaveis > 0
            else 0.0
        )

        return (
            internacoes,
            taxa
        )

    # --------------------------------------------------------
    # CHAVE DE ORDENAÇÃO
    # --------------------------------------------------------

    def chave_ordenacao(doc):

        internacoes, taxa = (
            extrair_metricas(doc)
        )

        if ordenar_por_taxa:

            return (
                taxa,
                internacoes
            )

        return (
            internacoes,
            taxa
        )

    # --------------------------------------------------------
    # RESULTADO
    # --------------------------------------------------------

    return sorted(
        docs,
        key=chave_ordenacao,
        reverse=True
    )

def responder_hospitalizacao_por_uf(documentos: List[Document], pergunta: str) -> Tuple[str, List[Document], str]:
    query_info = classificar_pergunta(
        pergunta
    )

    docs_ordenados = (
        filtrar_docs_hospitalizacao_por_uf(
            documentos=documentos,
            query_info=query_info,
            pergunta=pergunta
        )
    )

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

    for posicao, doc in enumerate(
        docs_ordenados,
        start=1
    ):

        md = doc.metadata

        uf = md.get(
            "uf_nome",
            md.get(
                "uf",
                "UF não informada"
            )
        )

        metricas = (
            extrair_metricas_hospitalizacao_vsm(
                doc
            )
        )

        hospitalizados = metricas[
            "hospitalizados"
        ]

        avaliaveis = metricas[
            "avaliaveis"
        ]

        taxa = metricas[
            "taxa_hospitalizacao"
        ]

        ano = md.get(
            "ano",
            "ano não informado"
        )

        linhas.append(
            f"{posicao}. {uf}: "
            f"{hospitalizados:,} hospitalizações "
            f"em {avaliaveis:,} registros avaliáveis "
            f"(taxa de hospitalização: "
            f"{taxa:.2f}%, ano {ano})"
            .replace(",", ".")
        )

    cabecalho = "Hospitalização por UF"
    if anos_encontrados:
        cabecalho += f" para o(s) ano(s) {', '.join(anos_encontrados)}"
    if len(faixas_encontradas) == 1:
        cabecalho += f", considerando a faixa epidemiológica {faixas_encontradas[0]}"
    cabecalho += f". Foram encontradas {len(docs_ordenados)} UFs com registros de hospitalização."

    resposta = cabecalho + "\n\n" + "\n".join(linhas)
    return resposta, docs_ordenados, "deterministico_hospitalizacao"

def responder_total_casos_por_uf(
    documentos: List[Document],
    pergunta: str
) -> Tuple[str, List[Document], str]:

    query_info = classificar_pergunta(
        pergunta
    )

    docs_ordenados = (
        filtrar_docs_total_casos_por_uf(
            documentos=documentos,
            query_info=query_info
        )
    )

    if not docs_ordenados:

        return (
            "Não encontrei VSMs geográficas por UF "
            "compatíveis com a pergunta.",
            [],
            "deterministico_casos"
        )

    resultados = []
    anos_encontrados = set()

    for doc in docs_ordenados:

        uf = str(
            doc.metadata.get(
                "uf_nome",
                "UF não informada"
            )
        ).strip()

        ano = str(
            doc.metadata.get(
                "ano",
                ""
            )
        ).strip()

        if ano:
            anos_encontrados.add(
                ano
            )

        total = (
            extrair_total_registros_vsm(
                doc
            )
        )

        if total is None:
            total = 0

        resultados.append({
            "uf": uf,
            "ano": ano,
            "total": total,
            "documento": doc
        })

    # --------------------------------------------------------
    # Ordenar alfabeticamente por UF
    # --------------------------------------------------------

    resultados = sorted(
        resultados,
        key=lambda item:
            normalizar_texto(
                item["uf"]
            )
    )

    linhas = []

    for resultado in resultados:

        uf = resultado["uf"]
        total = resultado["total"]

        total_formatado = (
            f"{total:,}"
            .replace(",", ".")
        )

        linhas.append(
            f"- **{uf}:** "
            f"{total_formatado} casos"
        )

    cabecalho = (
        "Total de casos registrados por UF"
    )

    if anos_encontrados:

        cabecalho += (
            " em "
            + ", ".join(
                sorted(
                    anos_encontrados
                )
            )
        )

    cabecalho += (
        f". Foram analisadas "
        f"{len(docs_ordenados)} "
        f"VSMs geográficas estaduais."
    )

    resposta = (
        cabecalho
        + "\n\n"
        + "\n".join(
            linhas
        )
    )

    return (
        resposta,
        docs_ordenados,
        "deterministico_casos"
    )

# ============================================================
# EXTRAIR VALOR NUMÉRICO POR RÓTULO DE UMA VSM
# ============================================================

def extrair_valor_rotulado_vsm(
    texto: str,
    rotulos: List[str]
) -> Optional[int]:

    texto_norm = normalizar_texto(
        texto or ""
    )

    for rotulo in rotulos:

        rotulo_norm = normalizar_texto(
            rotulo
        )

        padrao = re.compile(
            rf"{re.escape(rotulo_norm)}"
            rf"[^\d\n]{{0,80}}"
            rf"([\d\.\,]+)",
            flags=re.IGNORECASE
        )

        match = padrao.search(
            texto_norm
        )

        if not match:
            continue

        valor = (
            match.group(1)
            .replace(".", "")
            .replace(",", "")
            .strip()
        )

        try:
            return int(valor)
        except ValueError:
            continue

    return None


# ============================================================
# EXTRAIR HOSPITALIZAÇÃO DA VSM DE DESFECHOS
# ============================================================

def extrair_hospitalizacao_vsm(
    doc: Document
) -> List[Tuple[str, int]]:

    texto = doc.page_content or ""

    indicadores = {
        "Hospitalizados": [
            "hospitalizados",
            "hospitalizado",
            "internados",
            "internado",
        ],

        "Não hospitalizados": [
            "nao hospitalizados",
            "não hospitalizados",
            "nao internados",
            "não internados",
        ],

        "Ignorados": [
            "hospitalizacao ignorada",
            "hospitalização ignorada",
            "ignorado",
        ],

        "Não informados": [
            "hospitalizacao nao informada",
            "hospitalização não informada",
            "nao informado",
            "não informado",
        ],
    }

    resultados = []

    for nome, rotulos in indicadores.items():

        valor = extrair_valor_rotulado_vsm(
            texto,
            rotulos
        )

        if valor is not None:
            resultados.append(
                (
                    nome,
                    valor
                )
            )

    return resultados

def extrair_metricas_hospitalizacao_vsm(
    doc: Document
) -> Dict:

    itens = extrair_hospitalizacao_vsm(
        doc
    )

    mapa = {
        normalizar_texto(nome): numero_seguro(valor, 0)
        for nome, valor in itens
    }

    hospitalizados = mapa.get(
        "hospitalizados",
        0
    )

    nao_hospitalizados = mapa.get(
        "nao hospitalizados",
        0
    )

    ignorados = mapa.get(
        "ignorados",
        0
    )

    nao_informados = mapa.get(
        "nao informados",
        0
    )

    avaliaveis = (
        hospitalizados
        + nao_hospitalizados
    )

    total_registros = (
        hospitalizados
        + nao_hospitalizados
        + ignorados
        + nao_informados
    )

    taxa_hospitalizacao = (
        hospitalizados
        / avaliaveis
        * 100.0
        if avaliaveis > 0
        else 0.0
    )

    return {
        "hospitalizados": hospitalizados,
        "nao_hospitalizados": nao_hospitalizados,
        "ignorados": ignorados,
        "nao_informados": nao_informados,
        "avaliaveis": avaliaveis,
        "total_registros": total_registros,
        "taxa_hospitalizacao": taxa_hospitalizacao,
    }

# ============================================================
# EXTRAIR ÓBITOS DA VSM DE ÓBITOS
# ============================================================

def extrair_obitos_vsm(
    doc: Document
) -> List[Tuple[str, int]]:

    texto = doc.page_content or ""

    indicadores = {
        "Óbito pelo agravo": [
            "obito pelo agravo",
            "óbito pelo agravo",
            "obitos pelo agravo",
            "óbitos pelo agravo",
        ],

        "Óbito por outras causas": [
            "obito por outras causas",
            "óbito por outras causas",
            "obitos por outras causas",
            "óbitos por outras causas",
        ],

        "Óbito em investigação": [
            "obito em investigacao",
            "óbito em investigação",
            "obitos em investigacao",
            "óbitos em investigação",
        ],
    }

    resultados = []

    for nome, rotulos in indicadores.items():

        valor = extrair_valor_rotulado_vsm(
            texto,
            rotulos
        )

        if valor is not None:
            resultados.append(
                (
                    nome,
                    valor
                )
            )

    return resultados

# ============================================================
# EXTRAIR SINTOMAS DE UMA VSM CLÍNICA
# ============================================================
def extrair_sintomas_vsm(
    doc: Document
) -> List[Tuple[str, int]]:

    texto = doc.page_content or ""

    # --------------------------------------------------------
    # Extrai especificamente as linhas da seção de
    # sinais clínicos da VSM.
    #
    # Exemplo esperado:
    #
    # **Sinal clinico:** Febre |
    # **Total registros:** 65.195 |
    # **Avaliaveis:** 64.694 |
    # **Sim:** 54.137 |
    # **Nao:** 10.557 |
    # ...
    #
    # O valor utilizado no grafo é "Sim", pois representa
    # os registros em que o sinal clínico esteve presente.
    # --------------------------------------------------------

    padrao = re.compile(
        r"\*\*Sinal clinico:\*\*\s*"
        r"([^|\n]+?)"
        r"\s*\|\s*"
        r"\*\*Total registros:\*\*\s*"
        r"[\d\.\,]+"
        r"\s*\|\s*"
        r"\*\*Avaliaveis:\*\*\s*"
        r"[\d\.\,]+"
        r"\s*\|\s*"
        r"\*\*Sim:\*\*\s*"
        r"([\d\.\,]+)",
        flags=re.IGNORECASE
    )

    resultados = []

    for sintoma, total_sim_texto in padrao.findall(
        texto
    ):

        sintoma = sintoma.strip()

        valor_limpo = (
            total_sim_texto
            .replace(".", "")
            .replace(",", "")
            .strip()
        )

        try:
            total_sim = int(
                valor_limpo
            )
        except ValueError:
            continue

        resultados.append(
            (
                sintoma,
                total_sim
            )
        )

    return sorted(
        resultados,
        key=lambda item: item[1],
        reverse=True
    )
# ============================================================
# EXTRAIR SOROTIPOS DE UMA VSM VIROLÓGICA
# ============================================================

def extrair_sorotipos_vsm(
    doc: Document
) -> List[tuple]:

    texto = doc.page_content or ""

    padrao = re.compile(
        r"\*\*Sorotipo:\*\*\s*"
        r"(DENV-\d+)"
        r"\s*\|\s*"
        r"\*\*Total registros:\*\*\s*"
        r"([\d\.\,]+)",
        flags=re.IGNORECASE
    )

    resultados = []

    for sorotipo, total_texto in padrao.findall(texto):

        total_limpo = (
            total_texto
            .replace(".", "")
            .replace(",", "")
            .strip()
        )

        try:
            total = int(total_limpo)
        except ValueError:
            total = 0

        resultados.append(
            (
                sorotipo.upper(),
                total
            )
        )

    return resultados

# ============================================================
# EXTRAIR SINTOMAS COM PERCENTUAL PARA O GRAFO
# ============================================================

def extrair_sintomas_vsm_grafo(
    doc: Document
) -> List[Tuple[str, int, float]]:

    texto = doc.page_content or ""

    padrao = re.compile(
        r"\*\*Sinal clinico:\*\*\s*"
        r"([^|\n]+?)"
        r"\s*\|\s*"
        r"\*\*Total registros:\*\*\s*"
        r"[\d\.\,]+"
        r"\s*\|\s*"
        r"\*\*Avaliaveis:\*\*\s*"
        r"([\d\.\,]+)"
        r"\s*\|\s*"
        r"\*\*Sim:\*\*\s*"
        r"([\d\.\,]+)",
        flags=re.IGNORECASE
    )

    resultados = []

    for (
        sintoma,
        avaliaveis_texto,
        total_sim_texto
    ) in padrao.findall(texto):

        sintoma = sintoma.strip()

        avaliaveis_limpo = (
            avaliaveis_texto
            .replace(".", "")
            .replace(",", "")
            .strip()
        )

        total_sim_limpo = (
            total_sim_texto
            .replace(".", "")
            .replace(",", "")
            .strip()
        )

        try:
            avaliaveis = int(
                avaliaveis_limpo
            )

            total_sim = int(
                total_sim_limpo
            )

        except ValueError:
            continue

        percentual = (
            (total_sim / avaliaveis) * 100
            if avaliaveis > 0
            else 0.0
        )

        resultados.append(
            (
                sintoma,
                total_sim,
                percentual
            )
        )

    return sorted(
        resultados,
        key=lambda item: item[1],
        reverse=True
    )

# ============================================================
# FILTRAR VSMs VIROLÓGICAS POR UF
# ============================================================

def filtrar_docs_sorotipos_por_uf(
    documentos: List[Document],
    query_info: Dict
) -> List[Document]:

    # --------------------------------------------------------
    # 1. Selecionar somente as VSMs estaduais virológicas
    # --------------------------------------------------------

    docs_sorotipos = [
        doc
        for doc in documentos
        if (
            normalizar_texto(
                doc.metadata.get(
                    "dominio",
                    ""
                )
            ) == "virologico"
            and
            normalizar_texto(
                doc.metadata.get(
                    "tipo_documento",
                    ""
                )
            ) == "perfil_virologico_uf"
        )
    ]

    # --------------------------------------------------------
    # 2. Filtrar por ano, caso explicitado na pergunta
    # --------------------------------------------------------

    anos_desejados = {
        str(ano)
        for ano in query_info.get(
            "anos",
            []
        )
    }

    if anos_desejados:

        docs_sorotipos = [
            doc
            for doc in docs_sorotipos
            if str(
                doc.metadata.get(
                    "ano",
                    ""
                )
            ) in anos_desejados
        ]

    # --------------------------------------------------------
    # 3. Filtrar por UF, caso explicitada na pergunta
    # --------------------------------------------------------

    ufs_desejadas = {
        normalizar_texto(uf)
        for uf in query_info.get(
            "ufs_nome",
            []
        )
    }

    if ufs_desejadas:

        docs_sorotipos = [
            doc
            for doc in docs_sorotipos
            if normalizar_texto(
                doc.metadata.get(
                    "uf_nome",
                    ""
                )
            ) in ufs_desejadas
        ]

    # --------------------------------------------------------
    # 4. Ordenar alfabeticamente por UF
    # --------------------------------------------------------

    return sorted(
        docs_sorotipos,
        key=lambda doc:
            normalizar_texto(
                doc.metadata.get(
                    "uf_nome",
                    ""
                )
            )
    )


def extrair_total_registros_vsm(
    doc: Document
) -> Optional[int]:

    texto = doc.page_content or ""

    padrao = re.compile(
        r"\*\*Total registros:\*\*\s*"
        r"([\d\.\,]+)",
        flags=re.IGNORECASE
    )

    match = padrao.search(texto)

    if not match:
        return None

    valor = (
        match.group(1)
        .replace(".", "")
        .replace(",", "")
        .strip()
    )

    try:
        return int(valor)
    except ValueError:
        return None


# ============================================================
# RESPONDER SOROTIPOS POR UF
# ============================================================

def responder_sorotipos_por_uf(
    documentos: List[Document],
    pergunta: str
) -> Tuple[str, List[Document], str]:

    # --------------------------------------------------------
    # 1. Classificar pergunta
    # --------------------------------------------------------

    query_info = classificar_pergunta(
        pergunta
    )

    # --------------------------------------------------------
    # 2. Recuperar VSMs estaduais virológicas
    # --------------------------------------------------------

    docs_ordenados = (
        filtrar_docs_sorotipos_por_uf(
            documentos=documentos,
            query_info=query_info
        )
    )

    if not docs_ordenados:

        return (
            "Não encontrei VSMs virológicas por UF "
            "compatíveis com a pergunta.",
            [],
            "deterministico_sorotipos"
        )

    # --------------------------------------------------------
    # 3. Extrair sorotipos das VSMs
    # --------------------------------------------------------

    resultados = []

    anos_encontrados = set()

    for doc in docs_ordenados:

        uf = str(
            doc.metadata.get(
                "uf_nome",
                "UF não informada"
            )
        ).strip()

        ano = str(
            doc.metadata.get(
                "ano",
                ""
            )
        ).strip()

        if ano:
            anos_encontrados.add(
                ano
            )

        # A função já existente no app13.py
        sorotipos = (
            extrair_sorotipos_vsm(
                doc
            )
        )
        # ------------------------------------------------
        # Identificar sorotipo predominante
        # ------------------------------------------------

        sorotipo_predominante = None
        total_predominante = 0

        if sorotipos:

            sorotipo_predominante, total_predominante = max(
                sorotipos,
                key=lambda item: item[1]
            )

        resultados.append({
            "uf": uf,
            "ano": ano,
            "sorotipos": sorotipos,
            "documento": doc
        })

    # --------------------------------------------------------
    # 4. Construir resposta
    # --------------------------------------------------------

    linhas = []

    for resultado in resultados:

        uf = resultado[
            "uf"
        ]

        sorotipos = resultado[
            "sorotipos"
        ]

        linhas.append(
            f"**{uf}:**"
        )

        if sorotipos:

            for sorotipo, total in sorted(
                sorotipos,
                key=lambda item: item[0]
            ):

                linhas.append(
                    f"- {sorotipo}: "
                    f"{total:,}".replace(",", ".")
                    + " registros"
                )

        else:

            linhas.append(
                "- Nenhum sorotipo informado "
                "na VSM."
            )

        linhas.append("")

    # --------------------------------------------------------
    # 5. Cabeçalho
    # --------------------------------------------------------

    cabecalho = (
        "Sorotipos da dengue por UF"
    )

    if anos_encontrados:

        cabecalho += (
            " para "
            + ", ".join(
                sorted(
                    anos_encontrados
                )
            )
        )

    cabecalho += (
        f". Foram analisadas "
        f"{len(docs_ordenados)} "
        f"VSMs estaduais virológicas."
    )

    # --------------------------------------------------------
    # 6. Resposta final
    # --------------------------------------------------------

    resposta = (
        cabecalho
        + "\n\n"
        + "\n".join(linhas)
    )

    return (
        resposta,
        docs_ordenados,
        "deterministico_sorotipos"
    )

# ============================================================
# TESTE DO FLUXO DETERMINÍSTICO — SOROTIPOS POR UF
# ============================================================

pergunta_teste = (
    "Quais os sorotipos da dengue por UF?"
)

resposta_teste = (
    responder_sorotipos_por_uf(
        documentos_vsm,
        pergunta_teste
    )
)

resposta, docs_recuperados, modo = (
    resposta_teste
)

print(
    "Modo:",
    modo
)

print(
    "VSMs recuperadas:",
    len(docs_recuperados)
)

print(
    "\nUFs recuperadas:"
)

for doc in docs_recuperados:

    print(
        doc.metadata.get(
            "codigo_uf"
        ),
        "-",
        doc.metadata.get(
            "uf_nome"
        )
    )

print(
    "\nRESPOSTA:\n"
)

print(
    resposta
)

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


def fontes_para_dataframe_total_casos(
    fontes: List[Document]
) -> pd.DataFrame:

    registros = []

    for doc in fontes:

        uf = str(
            doc.metadata.get(
                "uf_nome",
                doc.metadata.get(
                    "uf",
                    ""
                )
            )
        ).strip()

        if not uf:
            continue

        total = (
            extrair_total_registros_vsm(
                doc
            )
        )

        if total is None:
            continue

        ano = str(
            doc.metadata.get(
                "ano",
                ""
            )
        ).strip()

        registros.append({
            "UF": uf,
            "Total de Casos": total,
            "Ano": ano
        })

    if not registros:

        return pd.DataFrame(
            columns=[
                "UF",
                "Total de Casos",
                "Ano"
            ]
        )

    return (
        pd.DataFrame(
            registros
        )
        .drop_duplicates(
            subset=["UF"]
        )
        .sort_values(
            "Total de Casos",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

def fontes_para_dataframe_hospitalizacao(
    fontes: List[Document]
) -> pd.DataFrame:

    registros = []

    for doc in fontes:

        md = doc.metadata

        metricas = (
            extrair_metricas_hospitalizacao_vsm(
                doc
            )
        )

        registros.append({

            "UF":
                md.get(
                    "uf_nome",
                    md.get(
                        "uf",
                        ""
                    )
                ),

            "Hospitalizações":
                metricas[
                    "hospitalizados"
                ],

            "Não hospitalizados":
                metricas[
                    "nao_hospitalizados"
                ],

            "Registros avaliáveis":
                metricas[
                    "avaliaveis"
                ],

            "Ignorados":
                metricas[
                    "ignorados"
                ],

            "Não informados":
                metricas[
                    "nao_informados"
                ],

            "Taxa de Hospitalização (%)":
                metricas[
                    "taxa_hospitalizacao"
                ],

            "Ano":
                md.get(
                    "ano",
                    ""
                ),
        })

    df = pd.DataFrame(
        registros
    )

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


def pergunta_pede_comparacao_por_uf(
    pergunta: str
) -> bool:

    p = normalizar_texto(pergunta)

    padroes = [
        "por uf",
        "por estado",
        "por estados",
        "em cada uf",
        "em cada estado",
        "entre as ufs",
        "entre os estados",
        "por unidade federativa",
        "em cada unidade federativa",
    ]

    return any(
        padrao in p
        for padrao in padroes
    )
# ============================================================
# PASSO 4.6
# FILTRO DE METADADOS PARA A BUSCA VETORIAL
# ============================================================

def construir_filtro_vetorial(
    query_info: Dict,
    pergunta: str
):
    """
    Constrói um filtro de metadados para restringir
    semanticamente o espaço de busca no Chroma.

    Retorna:
        dict compatível com o filtro do Chroma
        ou None quando a busca deve permanecer global.
    """

    intencoes = query_info.get(
        "intencoes",
        {}
    )

    p = normalizar_texto(pergunta)

    tem_uf = bool(
        query_info.get("ufs_nome")
        or query_info.get("ufs_sigla")
    )
    comparacao_por_uf = (
        pergunta_pede_comparacao_por_uf(
            pergunta
        )
    )

    # --------------------------------------------------------
    # PANORAMA GERAL
    # --------------------------------------------------------

    if intencoes.get("panorama"):

        return {
            "tipo_documento":
                "panorama_epidemiologico_integrado"
        }

    # --------------------------------------------------------
    # DETERMINAR INTENÇÃO TEMÁTICA PRINCIPAL
    # --------------------------------------------------------

    ordem_intencoes = [
        "sorotipos",
        "clinico",
        "doencas_preexistentes",
        "hospitalizacao",
        "obitos",
        "evolucao",
        "incidencia",
        "autoctonia",
        "temporal",
    ]

    

    # --------------------------------------------------------
    # DEFINIR INTENÇÃO PRINCIPAL
    # --------------------------------------------------------

    intencao_principal = None

    # Caso especial:
    # evolução geral incluindo cura e óbito.
    # Nesse contexto, queremos o panorama de desfechos,
    # não o documento específico de óbitos.
    if (
        intencoes.get("evolucao", False)
        and intencoes.get("obitos", False)
        and "cura" in p
    ):
        intencao_principal = "evolucao"

    # Caso geral
    if intencao_principal is None:

        for intencao in ordem_intencoes:

            if intencoes.get(
                intencao,
                False
            ):
                intencao_principal = intencao
                break
            
    if intencao_principal is None:
        return None

    tipos = TIPOS_VSM_POR_INTENCAO.get(
        intencao_principal,
        []
    )

    if not tipos:
        return None

    # --------------------------------------------------------
    # DOCUMENTOS POR UF
    #
    # Usa VSM estadual quando:
    # - existe uma UF específica; ou
    # - a pergunta pede comparação/distribuição por UF.
    # --------------------------------------------------------

    if tem_uf or comparacao_por_uf:    

        tipos_uf = [
            tipo
            for tipo in tipos
            if tipo.endswith("_uf")
        ]

        if tipos_uf:

            if len(tipos_uf) == 1:

                return {
                    "tipo_documento":
                        tipos_uf[0]
                }

            return {
                "tipo_documento": {
                    "$in":
                        tipos_uf
                }
            }

    # --------------------------------------------------------
    # SEM UF EXPLÍCITA
    # prioriza panorama nacional
    # --------------------------------------------------------

    tipos_nacionais = [
        tipo
        for tipo in tipos
        if (
            "nacional" in tipo
            or "panorama_" in tipo
        )
    ]

    if tipos_nacionais:

        if len(tipos_nacionais) == 1:

            return {
                "tipo_documento":
                    tipos_nacionais[0]
            }

        return {
            "tipo_documento": {
                "$in":
                    tipos_nacionais
            }
        }

    return None


# ============================================================
# 12. RECUPERAÇÃO INTELIGENTE (FLUXO GERAL)
# ============================================================

def recuperar_documentos_inteligentes(
    pergunta: str,
    vectorstore,
    documentos_base: List[Document]
) -> List[Document]:

    query_info = classificar_pergunta(
        pergunta
    )

    pesos = definir_pesos(
        query_info
    )

    # ========================================================
    # 1. CONSTRUIR FILTRO SEMÂNTICO PARA O CHROMA
    # ========================================================

    filtro_vetorial = (
        construir_filtro_vetorial(
            query_info,
            pergunta
        )
    )

    # ========================================================
    # 2. BUSCA VETORIAL
    # ========================================================

    if filtro_vetorial:

        documentos_recuperados = (
            vectorstore.similarity_search(
                pergunta,
                k=24,
                filter=filtro_vetorial
            )
        )

    else:

        documentos_recuperados = (
            vectorstore.similarity_search(
                pergunta,
                k=24
            )
        )

    # ========================================================
    # DEBUG
    # ========================================================

    if DEBUG:

        st.write(
            "Classificação:",
            query_info
        )

        st.write(
            "Filtro vetorial:",
            filtro_vetorial
        )

        st.write(
            "Quantidade recuperada:",
            len(documentos_recuperados)
        )

        for i, doc in enumerate(
            documentos_recuperados,
            start=1
        ):

            st.write(
                f"RAW {i}. "
                f"parent={doc.metadata.get('parent_document_id')} | "
                f"tipo={doc.metadata.get('tipo_documento')} | "
                f"dominio={doc.metadata.get('dominio')} | "
                f"uf={doc.metadata.get('uf_nome')} | "
                f"secao={doc.metadata.get('secao')}"
            )

    # ========================================================
    # 3. FILTRAGEM SEMÂNTICA COMPLEMENTAR
    # ========================================================

    documentos_filtrados = (
        filtrar_por_document_type(
            documentos_recuperados,
            query_info,
            pergunta
        )
    )

    # Fallback apenas se nenhum documento sobreviver
    if len(documentos_filtrados) == 0:
        documentos_filtrados = documentos_recuperados

    # ========================================================
    # 4. SCORE HEURÍSTICO
    # ========================================================

    docs_com_score = []

    for doc in documentos_filtrados:

        score, justificativas = (
            score_heuristico(
                pergunta,
                doc,
                query_info,
                pesos
            )
        )

        doc.metadata[
            "score_heuristico"
        ] = round(
            score,
            4
        )

        doc.metadata[
            "justificativa_heuristica"
        ] = justificativas

        docs_com_score.append(
            (
                score,
                doc
            )
        )

    # ========================================================
    # 5. ORDENAÇÃO FINAL
    # ========================================================

    docs_ordenados = sorted(
        docs_com_score,
        key=lambda x: x[0],
        reverse=True
    )

    return [
        doc
        for _, doc
        in docs_ordenados[:10]
    ]

# ============================================================
# PASSO 4.6.B
# MÉTODO PROPOSTO — RECUPERAÇÃO HÍBRIDA
# ============================================================

# if DEBUG:

#     st.subheader(
#         "Testes da recuperação híbrida"
#     )

#     perguntas_teste_hibrido = [

#         (
#             "Sorotipos — São Paulo",
#             "Quais são os sorotipos de dengue "
#             "registrados em São Paulo?"
#         ),

#         (
#             "Clínico — Minas Gerais",
#             "Quais são os principais sintomas "
#             "registrados em Minas Gerais?"
#         ),

#         (
#             "Incidência — Goiás",
#             "Qual a incidência de dengue em Goiás?"
#         ),

#         (
#             "Temporal — Brasil",
#             "Qual foi a evolução das notificações "
#             "ao longo das semanas?"
#         ),
#         (
#             "Evolução — desfecho",
#             "Qual foi a evolução dos casos "
#             "quanto à cura e óbito?"
#         ),
#         (
#             "Panorama — Brasil",
#             "Qual é o panorama geral da dengue "
#             "em 2026?"
#         ),
#     ]

#     for titulo, pergunta in perguntas_teste_hibrido:

#         st.markdown(
#             f"### {titulo}"
#         )

#         st.write(
#             "Pergunta:",
#             pergunta
#         )

#         documentos = (
#             recuperar_documentos_inteligentes(
#                 pergunta=pergunta,
#                 vectorstore=vectorstore_vsm,
#                 documentos_base=documentos_vsm
#             )
#         )

#         for posicao, doc in enumerate(
#             documentos,
#             start=1
#         ):

#             st.write(
#                 f"{posicao} — "
#                 f"{doc.metadata.get('parent_document_id')} — "
#                 f"{doc.metadata.get('secao')} — "
#                 f"score="
#                 f"{doc.metadata.get('score_heuristico')}"
#             )

#         st.divider()



# ============================================================
# PASSO 4.7
# MONTAGEM DO CONTEXTO PARA O LLM
# ============================================================


# A lógica é deliberada:

# Indicadores
#     ↓
# Evidências
#     ↓
# Síntese
#     ↓
# Interpretação
#     ↓
# Observações sobre os dados
#     ↓
# Conceitos semânticos

# Para perguntas epidemiológicas factuais, queremos colocar primeiro a evidência quantitativa. observacoes_dados continua disponível porque pode 
# conter limitações importantes, mas não deve ocupar o começo do contexto.
# ------------------------------------------------------------
# 4.7.1 — PRIORIDADE DAS SEÇÕES SEMÂNTICAS
# ------------------------------------------------------------

PRIORIDADE_SECOES_CONTEXTO = {
    "indicadores": 1,
    "evidencias": 2,
    "sintese": 3,
    "interpretacao": 4,
    "observacoes_dados": 5,
    "conceitos_semanticos": 6,
}

# ------------------------------------------------------------
# LIMITE DE CHUNKS POR SEÇÃO
# ------------------------------------------------------------

MAX_CHUNKS_POR_SECAO = {
    "indicadores": 1,
    "evidencias": 2,
    "sintese": 1,
    "interpretacao": 1,
    "observacoes_dados": 1,
    "conceitos_semanticos": 1,
}

# ------------------------------------------------------------
# 4.7.2 — CONTAGEM DE TOKENS
# ------------------------------------------------------------

def contar_tokens_texto(
    texto: str,
    modelo: str = LLM_MODEL
) -> int:

    if not texto:
        return 0

    try:

        encoding = tiktoken.encoding_for_model(
            modelo
        )

    except Exception:

        encoding = tiktoken.get_encoding(
            "o200k_base"
        )

    return len(
        encoding.encode(texto)
    )

# ------------------------------------------------------------
# 4.7.3 — FORMATAR UM CHUNK PARA O CONTEXTO
# ------------------------------------------------------------

def formatar_chunk_contexto(
    doc: Document,
    posicao: int
) -> str:

    md = doc.metadata

    parent_id = md.get(
        "parent_document_id",
        md.get(
            "document_id",
            ""
        )
    )

    titulo = md.get(
        "titulo",
        ""
    )

    tipo_documento = md.get(
        "tipo_documento",
        md.get(
            "document_type",
            ""
        )
    )

    dominio = md.get(
        "dominio",
        md.get(
            "categoria",
            ""
        )
    )

    ano = md.get(
        "ano",
        ""
    )

    uf = md.get(
        "uf_nome",
        md.get(
            "uf",
            ""
        )
    )

    secao = md.get(
        "secao",
        ""
    )

    semana_inicial = md.get(
        "semana_inicial",
        ""
    )

    semana_final = md.get(
        "semana_final",
        ""
    )

    cabecalho = [
        f"[EVIDÊNCIA {posicao}]",
        f"document_id: {parent_id}",
        f"titulo: {titulo}",
        f"tipo_documento: {tipo_documento}",
        f"dominio: {dominio}",
        f"ano: {ano}",
    ]

    if uf:
        cabecalho.append(
            f"uf: {uf}"
        )

    if semana_inicial != "":
        cabecalho.append(
            f"semana_inicial: {semana_inicial}"
        )

    if semana_final != "":
        cabecalho.append(
            f"semana_final: {semana_final}"
        )

    if secao:
        cabecalho.append(
            f"secao: {secao}"
        )

    cabecalho.append(
        ""
    )

    cabecalho.append(
        doc.page_content.strip()
    )

    return "\n".join(cabecalho)

# ------------------------------------------------------------
# 4.7.4 — ORDENAR CHUNKS PARA O CONTEXTO
# ------------------------------------------------------------

def ordenar_chunks_contexto(
    documentos: List[Document]
) -> List[Document]:

    return sorted(
        documentos,
        key=lambda doc: (
            PRIORIDADE_SECOES_CONTEXTO.get(
                normalizar_texto(
                    doc.metadata.get(
                        "secao",
                        ""
                    )
                ),
                99
            ),
            -float(
                doc.metadata.get(
                    "score_heuristico",
                    0
                )
            )
        )
    )


# ------------------------------------------------------------
# 4.7.5 — REMOVER DUPLICATAS
# ------------------------------------------------------------

def remover_chunks_duplicados(
    documentos: List[Document]
) -> List[Document]:

    documentos_unicos = []

    assinaturas = set()

    for doc in documentos:

        texto = (
            doc.page_content
            or ""
        ).strip()

        assinatura = hashlib.sha256(
            texto.encode(
                "utf-8"
            )
        ).hexdigest()

        if assinatura in assinaturas:
            continue

        assinaturas.add(
            assinatura
        )

        documentos_unicos.append(
            doc
        )

    return documentos_unicos

# ------------------------------------------------------------
# 4.7.6 — MONTAR CONTEXTO FINAL
# ------------------------------------------------------------

def montar_contexto_llm(
    documentos: List[Document],
    limite_tokens: int = 6000,
    max_chunks: int = 7
) -> Tuple[str, List[Document], Dict]:

    if not documentos:

        return (
            "",
            [],
            {
                "chunks_recebidos": 0,
                "chunks_unicos": 0,
                "chunks_selecionados": 0,
                "tokens_contexto": 0,
                "limite_tokens": limite_tokens,
            }
        )

    # --------------------------------------------------------
    # 1. Remover duplicatas
    # --------------------------------------------------------

    documentos_unicos = (
        remover_chunks_duplicados(
            documentos
        )
    )

    # --------------------------------------------------------
    # 2. Ordenar por utilidade semântica
    # --------------------------------------------------------

    documentos_ordenados = (
        ordenar_chunks_contexto(
            documentos_unicos
        )
    )

    # --------------------------------------------------------
    # 3. Selecionar respeitando:
    #    - orçamento de tokens
    #    - limite total de chunks
    #    - limite de chunks por seção
    # --------------------------------------------------------

    documentos_selecionados = []

    blocos_contexto = []

    total_tokens = 0

    # Controle da quantidade de chunks
    # selecionados por seção semântica
    contagem_por_secao = {}

    for doc in documentos_ordenados:

        # ----------------------------------------------------
        # 3.1 Identificar seção
        # ----------------------------------------------------

        secao = normalizar_texto(
            doc.metadata.get(
                "secao",
                ""
            )
        )

        # ----------------------------------------------------
        # 3.2 Verificar limite permitido para a seção
        # ----------------------------------------------------

        limite_secao = (
            MAX_CHUNKS_POR_SECAO.get(
                secao,
                1
            )
        )

        quantidade_secao = (
            contagem_por_secao.get(
                secao,
                0
            )
        )

        # Evita concentração excessiva de
        # chunks pertencentes à mesma seção
        if quantidade_secao >= limite_secao:
            continue

        # ----------------------------------------------------
        # 3.3 Verificar limite total de chunks
        # ----------------------------------------------------

        if (
            len(documentos_selecionados)
            >= max_chunks
        ):
            break

        # ----------------------------------------------------
        # 3.4 Definir posição da evidência
        # ----------------------------------------------------

        posicao = (
            len(documentos_selecionados)
            + 1
        )

        # ----------------------------------------------------
        # 3.5 Formatar o chunk
        # ----------------------------------------------------

        bloco = formatar_chunk_contexto(
            doc,
            posicao
        )

        # ----------------------------------------------------
        # 3.6 Contar tokens
        # ----------------------------------------------------

        tokens_bloco = contar_tokens_texto(
            bloco
        )

        # ----------------------------------------------------
        # 3.7 Verificar orçamento de tokens
        # ----------------------------------------------------

        if (
            total_tokens + tokens_bloco
            > limite_tokens
        ):
            continue

        # ----------------------------------------------------
        # 3.8 Adicionar chunk ao contexto
        # ----------------------------------------------------

        documentos_selecionados.append(
            doc
        )

        blocos_contexto.append(
            bloco
        )

        total_tokens += tokens_bloco

        # ----------------------------------------------------
        # 3.9 Atualizar controle por seção
        # ----------------------------------------------------

        contagem_por_secao[secao] = (
            quantidade_secao + 1
        )

    # --------------------------------------------------------
    # 4. Montar texto final do contexto
    # --------------------------------------------------------

    contexto_texto = (
        "\n\n"
        + "=" * 70
        + "\n\n"
    ).join(
        blocos_contexto
    )

    # --------------------------------------------------------
    # 5. Estatísticas
    # --------------------------------------------------------

    estatisticas = {
        "chunks_recebidos":
            len(documentos),

        "chunks_unicos":
            len(documentos_unicos),

        "chunks_selecionados":
            len(documentos_selecionados),

        "tokens_contexto":
            total_tokens,

        "limite_tokens":
            limite_tokens,

        "chunks_por_secao":
            contagem_por_secao,
    }

    # --------------------------------------------------------
    # 6. Retorno
    # --------------------------------------------------------

    return (
        contexto_texto,
        documentos_selecionados,
        estatisticas
    )


# Por que começar com 6000 tokens?

# Não é um limite do modelo. É um orçamento operacional nosso.

# Queremos evitar o comportamento:

# tem 10 chunks
# → manda os 10
# → contexto enorme
# → informação repetida
# → geração menos controlada

# Começamos conservadoramente com 6.000. Depois podemos avaliar experimentalmente se 4.000, 6.000 ou 8.000 funciona melhor.

# 4.7.7 — Testar sem chamar o LLM

# Isso é importante: ainda não vamos gastar chamada de geração.

# Logo abaixo das funções, coloque:

# ============================================================
# PASSO 4.7.7
# TESTE DA MONTAGEM DO CONTEXTO
# ============================================================

# if DEBUG:

#     st.subheader(
#         "Teste da montagem do contexto para o LLM"
#     )

#     perguntas_teste_contexto = [

#         (
#             "Temporal — Brasil",
#             "Qual foi a evolução das notificações "
#             "ao longo das semanas?"
#         ),

#         (
#             "Clínico — Minas Gerais",
#             "Quais são os principais sintomas "
#             "registrados em Minas Gerais?"
#         ),  
#         (
#             "Incidência — Goiás",
#             "Qual a incidência de dengue em Goiás?"
#         ),

#         (
#             "Evolução — desfechos",
#             "Qual foi a evolução dos casos "
#             "quanto à cura e óbito?"
#         ),

#         (
#             "Panorama — Brasil",
#             "Qual é o panorama geral da dengue "
#             "em 2026?"
#         ),
#     ]

#     for titulo, pergunta in (
#         perguntas_teste_contexto
#     ):

#         st.markdown(
#             f"### {titulo}"
#         )

#         documentos_recuperados = (
#             recuperar_documentos_inteligentes(
#                 pergunta=pergunta,
#                 vectorstore=vectorstore_vsm,
#                 documentos_base=documentos_vsm
#             )
#         )

#         (
#             contexto,
#             docs_contexto,
#             estatisticas
#         ) = montar_contexto_llm(
#             documentos_recuperados
#         )

#         st.write(
#             "Estatísticas:",
#             estatisticas
#         )

#         st.write(
#             "Chunks selecionados:"
#         )

#         for i, doc in enumerate(
#             docs_contexto,
#             start=1
#         ):

#             st.write(
#                 f"{i}. "
#                 f"{doc.metadata.get('parent_document_id')} | "
#                 f"{doc.metadata.get('secao')} | "
#                 f"score="
#                 f"{doc.metadata.get('score_heuristico')}"
#             )

#         with st.expander(
#             "Visualizar contexto final"
#         ):

#             st.text(
#                 contexto
#             )

#         st.divider()


# ============================================================
# 13. RESPOSTA FINAL
# ============================================================
# Pergunta
#    ↓
# classificar_pergunta()
#    ↓
# classificar_escopo_pergunta()
#    ↓
#  ┌──────────────────────────────┐
#  │ valido                       │
#  │ → continua para recuperação  │
#  └──────────────────────────────┘

#  ┌──────────────────────────────┐
#  │ sem_cobertura                │
#  │ → responde sem usar RAG      │
#  └──────────────────────────────┘

#  ┌──────────────────────────────┐
#  │ fora_dominio                 │
#  │ → responde sem usar RAG      │
#  └──────────────────────────────┘

# ============================================================
# PASSO 4.8
# GERAÇÃO DE RESPOSTA FUNDAMENTADA NO CONTEXTO
# ============================================================

PROMPT_SISTEMA_RAG = """
Você é um assistente especializado em vigilância epidemiológica
da dengue com base em documentos semânticos derivados do SINAN.

Responda exclusivamente com base nas evidências fornecidas no contexto.

Regras obrigatórias:

1. Não utilize conhecimento externo ao contexto.
2. Não invente valores, relações ou explicações causais.
3. Preserve os termos epidemiológicos presentes nas evidências.
4. Diferencie notificações, registros, incidência, residência,
   local de notificação e provável local de infecção.
5. Quando houver percentuais, deixe claro o denominador ou subconjunto
   ao qual eles se referem, sempre que essa informação estiver disponível.
6. Considere explicitamente as limitações e observações sobre os dados.
7. Não transforme associação descritiva em relação causal.
8. Se o contexto não contiver evidência suficiente para responder,
   informe claramente que os documentos recuperados não fornecem
   informação suficiente.
9. Não complete a resposta com conhecimento geral sobre dengue.
10. Responda em português do Brasil.
11. Seja objetivo, mas forneça contexto suficiente para que os números
    não sejam interpretados de forma incorreta.
"""

# 4.8.2 — Função que gera a resposta

# Agora adicione:
def gerar_resposta_fundamentada(
    pergunta: str,
    documentos_recuperados: List[Document],
    llm,
    limite_tokens_contexto: int = 6000,
    max_chunks_contexto: int = 7
) -> Dict:

    # --------------------------------------------------------
    # 1. Verificar existência de evidências
    # --------------------------------------------------------

    if not documentos_recuperados:

        return {
            "resposta": (
                "Não foram encontradas evidências suficientes "
                "nos documentos disponíveis para responder "
                "a esta pergunta."
            ),
            "contexto": "",
            "documentos_contexto": [],
            "estatisticas_contexto": {
                "chunks_recebidos": 0,
                "chunks_unicos": 0,
                "chunks_selecionados": 0,
                "tokens_contexto": 0,
                "limite_tokens": limite_tokens_contexto,
            },
        }

    # --------------------------------------------------------
    # 2. Montar contexto controlado
    # --------------------------------------------------------

    (
        contexto_texto,
        documentos_contexto,
        estatisticas_contexto
    ) = montar_contexto_llm(
        documentos=documentos_recuperados,
        limite_tokens=limite_tokens_contexto,
        max_chunks=max_chunks_contexto
    )

    # --------------------------------------------------------
    # 3. Verificar se algum contexto foi selecionado
    # --------------------------------------------------------

    if not contexto_texto.strip():

        return {
            "resposta": (
                "Os documentos recuperados não forneceram "
                "contexto suficiente para responder "
                "à pergunta com segurança."
            ),
            "contexto": "",
            "documentos_contexto": [],
            "estatisticas_contexto":
                estatisticas_contexto,
        }

    # --------------------------------------------------------
    # 4. Construir mensagem do usuário
    # --------------------------------------------------------

    prompt_usuario = f"""
PERGUNTA:
{pergunta}

CONTEXTO EPIDEMIOLÓGICO RECUPERADO:
{contexto_texto}

TAREFA:
Responda à pergunta utilizando apenas o contexto acima.

Se houver limitações relevantes nos dados, mencione-as de forma
proporcional à pergunta.

Se a resposta exigir informação não presente no contexto, diga
explicitamente que a informação não está disponível nas evidências
recuperadas.
"""

    # --------------------------------------------------------
    # 5. Chamar o LLM
    # --------------------------------------------------------

    mensagens = [
        SystemMessage(
            content=PROMPT_SISTEMA_RAG
        ),
        HumanMessage(
            content=prompt_usuario
        ),
    ]

    resposta_llm = llm.invoke(
        mensagens
    )

    # --------------------------------------------------------
    # 6. Extrair texto da resposta
    # --------------------------------------------------------

    resposta_texto = (
        resposta_llm.content
        if hasattr(
            resposta_llm,
            "content"
        )
        else str(resposta_llm)
    )

    # --------------------------------------------------------
    # 7. Retornar resposta + rastreabilidade
    # --------------------------------------------------------

    return {
        "resposta":
            resposta_texto,

        "contexto":
            contexto_texto,

        "documentos_contexto":
            documentos_contexto,

        "estatisticas_contexto":
            estatisticas_contexto,
    }

# 4.8.3 — Importante: confira os imports




def responder_pergunta(pergunta: str, vectorstore, documentos):
    query_info = classificar_pergunta(pergunta)

    classificacao_escopo = classificar_escopo_pergunta(
        pergunta,
        query_info
    )

    if classificacao_escopo != ESCOPO_VALIDO:

        return resposta_fora_escopo(
            classificacao_escopo
        )

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


# 4.8.4 — Teste isolado

# Ainda não altere responder_pergunta().

# Logo após essa função, crie um teste sob DEBUG:


# if DEBUG:

#     st.subheader(
#         "Teste da geração fundamentada — Passo 4.8"
#     )

#     pergunta_teste_48 = (
#         "Qual a incidência de dengue em Goiás?"
#     )

#     # Recuperação híbrida já construída no Passo 4.6
#     documentos_teste_48 = (
#         recuperar_documentos_inteligentes(
#             pergunta_teste_48,
#             vectorstore_vsm,
#             documentos_vsm
#         )
#     )

#     llm_teste_48 = ChatOpenAI(
#         model=LLM_MODEL,
#         temperature=0
#     )

#     resultado_48 = (
#         gerar_resposta_fundamentada(
#             pergunta=pergunta_teste_48,
#             documentos_recuperados=
#                 documentos_teste_48,
#             llm=llm_teste_48
#         )
#     )

#     st.markdown(
#         "#### Pergunta"
#     )

#     st.write(
#         pergunta_teste_48
#     )

#     st.markdown(
#         "#### Resposta do LLM"
#     )

#     st.write(
#         resultado_48[
#             "resposta"
#         ]
#     )

#     st.markdown(
#         "#### Estatísticas do contexto"
#     )

#     st.json(
#         resultado_48[
#             "estatisticas_contexto"
#         ]
#     )

# ============================================================
# RECUPERAÇÃO COMBINADA POR UF
# ============================================================

def recuperar_combinada_por_uf(
    pergunta: str,
    dominios: List[str],
    documentos_base: List[Document]
) -> List[Document]:
    """
    Recupera VSMs estaduais dos domínios explicitamente
    identificados em uma consulta multidimensional.

    Exemplo:
        "Quais sintomas predominantes e sorotipos aparecem
        em cada estado?"

    dominios:
        ["clinico", "virologico"]

    granularidade:
        UF

    Nesse caso recupera somente:
        - perfil_clinico_uf
        - perfil_virologico_uf

    Não inclui automaticamente documentos geográficos,
    hospitalizações, óbitos ou outros domínios.
    """

    # --------------------------------------------------------
    # 1. Normalizar domínios solicitados
    # --------------------------------------------------------

    dominios_desejados = {
        normalizar_texto(dominio)
        for dominio in dominios
        if dominio
    }

    # --------------------------------------------------------
    # 2. Mapear domínio -> VSM estadual
    # --------------------------------------------------------

    tipos_por_dominio = {
        "clinico": {
            "perfil_clinico_uf"
        },

        "virologico": {
            "perfil_virologico_uf"
        },

        "desfechos": {
            "desfechos_uf"
        },

        "temporal": {
            "serie_temporal_uf"
        },

        "geografico": {
            "distribuicao_geografica_uf"
        },
    }

    # --------------------------------------------------------
    # 3. Determinar tipos permitidos exclusivamente
    #    pelos domínios presentes na pergunta
    # --------------------------------------------------------

    tipos_permitidos = set()

    for dominio in dominios_desejados:

        tipos_permitidos.update(
            tipos_por_dominio.get(
                dominio,
                set()
            )
        )

    if not tipos_permitidos:
        return []

    # --------------------------------------------------------
    # 4. Selecionar somente VSMs estaduais correspondentes
    # --------------------------------------------------------

    documentos_selecionados = []

    for doc in documentos_base:

        metadata = doc.metadata or {}

        dominio_doc = normalizar_texto(
            metadata.get(
                "dominio",
                ""
            )
        )

        tipo_doc = normalizar_texto(
            metadata.get(
                "tipo_documento",
                metadata.get(
                    "document_type",
                    ""
                )
            )
        )

        uf = str(
            metadata.get(
                "uf_nome",
                metadata.get(
                    "uf",
                    ""
                )
            )
        ).strip()

        # Deve pertencer a um dos domínios pedidos
        if dominio_doc not in dominios_desejados:
            continue

        # Deve ser uma VSM estadual apropriada
        if tipo_doc not in tipos_permitidos:
            continue

        # Consulta "por UF" exige identificação da UF
        if not uf:
            continue

        documentos_selecionados.append(
            doc
        )

    # --------------------------------------------------------
    # 5. Ordenação determinística:
    #    UF -> domínio -> tipo
    # --------------------------------------------------------

    documentos_selecionados = sorted(
        documentos_selecionados,
        key=lambda doc: (
            normalizar_texto(
                doc.metadata.get(
                    "uf_nome",
                    doc.metadata.get(
                        "uf",
                        ""
                    )
                )
            ),
            normalizar_texto(
                doc.metadata.get(
                    "dominio",
                    ""
                )
            ),
            normalizar_texto(
                doc.metadata.get(
                    "tipo_documento",
                    ""
                )
            )
        )
    )

    # --------------------------------------------------------
    # 6. Debug
    # --------------------------------------------------------

    if DEBUG:

        st.subheader(
            "Recuperação combinada por UF"
        )

        st.write(
            "Domínios solicitados:",
            sorted(dominios_desejados)
        )

        st.write(
            "Tipos permitidos:",
            sorted(tipos_permitidos)
        )

        st.write(
            "Documentos recuperados:",
            len(documentos_selecionados)
        )

        st.write(
            "Distribuição por domínio:",
            {
                dominio: sum(
                    1
                    for doc in documentos_selecionados
                    if normalizar_texto(
                        doc.metadata.get(
                            "dominio",
                            ""
                        )
                    ) == dominio
                )
                for dominio
                in sorted(dominios_desejados)
            }
        )

    return documentos_selecionados

def recuperar_evidencias(
    pergunta,
    vectorstore,
    documentos_base,
    analise_semantica
):

    dominios = (
        analise_semantica.get(
            "dominios",
            []
        )
    )

    granularidade = (
        analise_semantica.get(
            "granularidade"
        )
    )

    consulta_combinada = (
        analise_semantica.get(
            "consulta_combinada",
            False
        )
    )

    # ---------------------------------------------
    # Consulta multidimensional por UF
    # ---------------------------------------------

    if (
        consulta_combinada
        and granularidade == "uf"
    ):

        return recuperar_combinada_por_uf(
            pergunta=pergunta,
            dominios=dominios,
            documentos_base=documentos_base
        )

    # ---------------------------------------------
    # Consulta normal
    # ---------------------------------------------

    return recuperar_documentos_inteligentes(
        pergunta=pergunta,
        vectorstore=vectorstore,
        documentos_base=documentos_base
    )


# ============================================================
# RESPOSTA DETERMINÍSTICA — SINTOMAS PREDOMINANTES POR UF
# ============================================================

def responder_sintomas_por_uf_vsm(
    documentos: List[Document],
    pergunta: str
) -> Tuple[str, List[Document], str]:

    # --------------------------------------------------------
    # 1. Selecionar somente VSMs clínicas estaduais
    # --------------------------------------------------------

    docs_clinicos = [
        doc
        for doc in documentos
        if (
            normalizar_texto(
                doc.metadata.get(
                    "dominio",
                    ""
                )
            ) == "clinico"

            and

            normalizar_texto(
                doc.metadata.get(
                    "tipo_documento",
                    ""
                )
            ) == "perfil_clinico_uf"

            and

            str(
                doc.metadata.get(
                    "uf_nome",
                    ""
                )
            ).strip()
        )
    ]

    if not docs_clinicos:

        return (
            "Não foram encontradas VSMs clínicas por UF "
            "compatíveis com a consulta.",
            [],
            "deterministico_sintomas_uf"
        )

    # --------------------------------------------------------
    # 2. Ordenar por UF
    # --------------------------------------------------------

    docs_clinicos = sorted(
        docs_clinicos,
        key=lambda doc: normalizar_texto(
            doc.metadata.get(
                "uf_nome",
                ""
            )
        )
    )

    # --------------------------------------------------------
    # 3. Extrair sintoma predominante de cada UF
    # --------------------------------------------------------

    linhas = []
    documentos_utilizados = []

    for doc in docs_clinicos:

        uf = (
            doc.metadata.get(
                "uf_nome",
                "UF não informada"
            )
        )

        sintomas = extrair_sintomas_vsm(
            doc
        )

        if sintomas:

            sintoma_predominante, quantidade = (
                sintomas[0]
            )

            linhas.append(
                f"- **{uf}**: "
                f"{sintoma_predominante}"
                + (
                    f" ({quantidade} registros)"
                    if quantidade
                    else ""
                )
            )

        else:

            linhas.append(
                f"- **{uf}**: "
                "não foi possível identificar "
                "o sintoma predominante na VSM."
            )

        documentos_utilizados.append(
            doc
        )

    # --------------------------------------------------------
    # 4. Montar resposta
    # --------------------------------------------------------

    resposta = (
        "Sintomas predominantes por UF:\n\n"
        + "\n".join(
            linhas
        )
    )

    return (
        resposta,
        documentos_utilizados,
        "deterministico_sintomas_uf"
    )

# ============================================================
# RESPOSTA DETERMINÍSTICA
# SINTOMAS PREDOMINANTES + SOROTIPOS POR UF
# ============================================================

def responder_sintomas_sorotipos_por_uf_vsm(
    documentos: List[Document],
    pergunta: str
) -> Tuple[str, List[Document], str]:

    # --------------------------------------------------------
    # 1. Classificar pergunta
    # --------------------------------------------------------

    query_info = classificar_pergunta(
        pergunta
    )

    anos_desejados = {
        str(ano)
        for ano in query_info.get(
            "anos",
            []
        )
    }

    # --------------------------------------------------------
    # 2. Selecionar VSMs clínicas estaduais
    # --------------------------------------------------------

    docs_clinicos = [
        doc
        for doc in documentos
        if (
            normalizar_texto(
                doc.metadata.get(
                    "dominio",
                    ""
                )
            ) == "clinico"

            and

            normalizar_texto(
                doc.metadata.get(
                    "tipo_documento",
                    doc.metadata.get(
                        "document_type",
                        ""
                    )
                )
            ) == "perfil_clinico_uf"

            and

            str(
                doc.metadata.get(
                    "uf_nome",
                    doc.metadata.get(
                        "uf",
                        ""
                    )
                )
            ).strip()
        )
    ]

    # --------------------------------------------------------
    # 3. Selecionar VSMs virológicas estaduais
    # --------------------------------------------------------

    docs_virologicos = [
        doc
        for doc in documentos
        if (
            normalizar_texto(
                doc.metadata.get(
                    "dominio",
                    ""
                )
            ) == "virologico"

            and

            normalizar_texto(
                doc.metadata.get(
                    "tipo_documento",
                    doc.metadata.get(
                        "document_type",
                        ""
                    )
                )
            ) == "perfil_virologico_uf"

            and

            str(
                doc.metadata.get(
                    "uf_nome",
                    doc.metadata.get(
                        "uf",
                        ""
                    )
                )
            ).strip()
        )
    ]

    # --------------------------------------------------------
    # 4. Filtrar por ano, caso a pergunta informe um ano
    # --------------------------------------------------------

    if anos_desejados:

        docs_clinicos = [
            doc
            for doc in docs_clinicos
            if str(
                doc.metadata.get(
                    "ano",
                    ""
                )
            ) in anos_desejados
        ]

        docs_virologicos = [
            doc
            for doc in docs_virologicos
            if str(
                doc.metadata.get(
                    "ano",
                    ""
                )
            ) in anos_desejados
        ]

    # --------------------------------------------------------
    # 5. Verificar existência das VSMs
    # --------------------------------------------------------

    if (
        not docs_clinicos
        and not docs_virologicos
    ):

        return (
            "Não foram encontradas VSMs clínicas ou "
            "virológicas estaduais compatíveis com a consulta.",
            [],
            "deterministico_sintomas_sorotipos_uf"
        )

    # --------------------------------------------------------
    # 6. Indexar VSMs pela UF
    # --------------------------------------------------------

    clinicos_por_uf = {}

    for doc in docs_clinicos:

        uf = str(
            doc.metadata.get(
                "uf_nome",
                doc.metadata.get(
                    "uf",
                    ""
                )
            )
        ).strip()

        if uf:
            clinicos_por_uf[
                normalizar_texto(uf)
            ] = doc

    virologicos_por_uf = {}

    for doc in docs_virologicos:

        uf = str(
            doc.metadata.get(
                "uf_nome",
                doc.metadata.get(
                    "uf",
                    ""
                )
            )
        ).strip()

        if uf:
            virologicos_por_uf[
                normalizar_texto(uf)
            ] = doc

    # --------------------------------------------------------
    # 7. União das UFs disponíveis nos dois domínios
    # --------------------------------------------------------

    chaves_ufs = (
        set(clinicos_por_uf.keys())
        |
        set(virologicos_por_uf.keys())
    )

    # --------------------------------------------------------
    # 8. Descobrir nome original de cada UF
    # --------------------------------------------------------

    nomes_ufs = {}

    for chave in chaves_ufs:

        doc = (
            clinicos_por_uf.get(chave)
            or
            virologicos_por_uf.get(chave)
        )

        nomes_ufs[chave] = str(
            doc.metadata.get(
                "uf_nome",
                doc.metadata.get(
                    "uf",
                    chave
                )
            )
        ).strip()

    # --------------------------------------------------------
    # 9. Ordenar alfabeticamente por UF
    # --------------------------------------------------------

    chaves_ufs = sorted(
        chaves_ufs,
        key=lambda chave:
            normalizar_texto(
                nomes_ufs.get(
                    chave,
                    chave
                )
            )
    )

    # --------------------------------------------------------
    # 10. Construir resultado por UF
    # --------------------------------------------------------

    linhas = []

    documentos_utilizados = []

    anos_encontrados = set()

    for chave_uf in chaves_ufs:

        uf = nomes_ufs[
            chave_uf
        ]

        doc_clinico = (
            clinicos_por_uf.get(
                chave_uf
            )
        )

        doc_virologico = (
            virologicos_por_uf.get(
                chave_uf
            )
        )

        # ====================================================
        # SINTOMA PREDOMINANTE
        # ====================================================

        sintoma_texto = (
            "não disponível"
        )

        if doc_clinico:

            sintomas = (
                extrair_sintomas_vsm(
                    doc_clinico
                )
            )

            if sintomas:

                (
                    sintoma_predominante,
                    qtd_sintoma
                ) = sintomas[0]

                sintoma_texto = (
                    f"{sintoma_predominante} "
                    f"({qtd_sintoma:,} registros)"
                    .replace(",", ".")
                )

            documentos_utilizados.append(
                doc_clinico
            )

            ano = str(
                doc_clinico.metadata.get(
                    "ano",
                    ""
                )
            ).strip()

            if ano:
                anos_encontrados.add(
                    ano
                )

        # ====================================================
        # SOROTIPOS
        # ====================================================

        sorotipos_texto = (
            "nenhum sorotipo informado"
        )

        if doc_virologico:

            sorotipos = (
                extrair_sorotipos_vsm(
                    doc_virologico
                )
            )

            if sorotipos:

                # Ordena por DENV-1, DENV-2...
                sorotipos = sorted(
                    sorotipos,
                    key=lambda item:
                        item[0]
                )

                sorotipos_formatados = []

                for (
                    sorotipo,
                    quantidade
                ) in sorotipos:

                    sorotipos_formatados.append(
                        (
                            f"{sorotipo} "
                            f"({quantidade:,})"
                        ).replace(
                            ",",
                            "."
                        )
                    )

                sorotipos_texto = (
                    ", ".join(
                        sorotipos_formatados
                    )
                )

            documentos_utilizados.append(
                doc_virologico
            )

            ano = str(
                doc_virologico.metadata.get(
                    "ano",
                    ""
                )
            ).strip()

            if ano:
                anos_encontrados.add(
                    ano
                )

        # ====================================================
        # LINHA DA UF
        # ====================================================

        linhas.append(
            f"- **{uf}**: "
            f"sintoma predominante — "
            f"{sintoma_texto}; "
            f"sorotipos — "
            f"{sorotipos_texto}."
        )

    # --------------------------------------------------------
    # 11. Remover eventual duplicação de documentos
    # --------------------------------------------------------

    docs_unicos = []

    ids_vistos = set()

    for doc in documentos_utilizados:

        identificador = (
            doc.metadata.get(
                "document_id"
            )
            or
            doc.metadata.get(
                "arquivo_json"
            )
            or
            id(doc)
        )

        if identificador in ids_vistos:
            continue

        ids_vistos.add(
            identificador
        )

        docs_unicos.append(
            doc
        )

    # --------------------------------------------------------
    # 12. Cabeçalho
    # --------------------------------------------------------

    cabecalho = (
        "Sintomas predominantes e sorotipos "
        "da dengue por UF"
    )

    if anos_encontrados:

        cabecalho += (
            " para "
            + ", ".join(
                sorted(
                    anos_encontrados
                )
            )
        )

    cabecalho += (
        f". Foram analisadas "
        f"{len(chaves_ufs)} UFs, "
        f"com {len(docs_clinicos)} VSMs clínicas "
        f"e {len(docs_virologicos)} VSMs virológicas."
    )

    # --------------------------------------------------------
    # 13. Resposta final
    # --------------------------------------------------------

    resposta = (
        cabecalho
        + "\n\n"
        + "\n".join(
            linhas
        )
    )

    return (
        resposta,
        docs_unicos,
        "deterministico_sintomas_sorotipos_uf"
    )
# ============================================================
# PASSO 4.9
# INTEGRAÇÃO DO PIPELINE VSM
# ============================================================

def responder_pergunta_vsm(
    pergunta: str,
    documentos_base: List[Document],
    vectorstore=None,
    llm=None,
    analise_semantica=None
) -> Dict:

   
    # --------------------------------------------------------
    # 1. Validar pergunta
    # --------------------------------------------------------

    pergunta = (
        pergunta.strip()
        if pergunta
        else ""
    )

    if not pergunta:

        return {
            "pergunta": "",
            "resposta": (
                "Informe uma pergunta para realizar a consulta."
            ),
            "documentos_recuperados": [],
            "documentos_contexto": [],
            "estatisticas_contexto": {},
        }

    # --------------------------------------------------------
    # 1.1 Análise semântica da pergunta
    # --------------------------------------------------------

    if analise_semantica is None:

        analise_semantica = (
            analisar_pergunta(
                pergunta
            )
        )

    # --------------------------------------------------------
    # 2. Classificação e proteção de escopo
    # --------------------------------------------------------

    classificacao = (
        classificar_pergunta(
            pergunta
        )
    )

    resultado_escopo = (
        validar_escopo_pergunta(
            pergunta=pergunta,
            classificacao=classificacao
        )
    )

    if DEBUG:

        st.write(
            "Validação de escopo:",
            resultado_escopo
        )

    if (
        resultado_escopo[
            "status"
        ]
        != "valido"
    ):

        return {
            "pergunta":
                pergunta,

            "resposta":
                gerar_resposta_escopo(
                    resultado_escopo
                ),

            "classificacao":
                classificacao,

            "escopo":
                resultado_escopo,

            "documentos_recuperados":
                [],

            "documentos_contexto":
                [],

            "estatisticas_contexto":
                {
                    "chunks_recebidos": 0,
                    "chunks_unicos": 0,
                    "chunks_selecionados": 0,
                    "tokens_contexto": 0,
                    "limite_tokens": 6000,
                },

            "contexto":
                "",
        }

    if eh_pergunta_grafo_combinado(
        pergunta,
        classificacao
    ):

        return {
            "pergunta": pergunta,

            "resposta": (
                "Selecione uma UF abaixo para gerar o "
                "grafo combinado de sintomas e sorotipos."
            ),

            "classificacao": classificacao,
            "escopo": resultado_escopo,

            "documentos_recuperados": [],
            "documentos_contexto": [],

            "estatisticas_contexto": {
                "modo": "deterministico_grafo_combinado",
                "documentos_recuperados": 0,
                "chunks_recebidos": 0,
                "chunks_unicos": 0,
                "chunks_selecionados": 0,
                "tokens_contexto": 0,
                "limite_tokens": 0,
            },

            "estatisticas_contexto": {
                "modo": "deterministico_grafo_combinado",
                "documentos_recuperados":
                    len(documentos_base),
                "chunks_recebidos":
                    len(documentos_base),
                "chunks_unicos":
                    len(documentos_base),
                "chunks_selecionados":
                    len(documentos_base),
                "tokens_contexto": 0,
                "limite_tokens": 0,
            },

            "contexto": "",

            "modo":
                "deterministico_grafo_combinado",
        }


    # ========================================================
    # RESPOSTAS DETERMINÍSTICAS PARA CONSULTAS EXAUSTIVAS
    # ========================================================

    # --------------------------------------------------------
    # Sintomas predominantes + sorotipos por UF
    # --------------------------------------------------------

    if eh_pergunta_sintomas_sorotipos_por_uf(
        pergunta,
        classificacao
    ):

        (
            resposta_deterministica,
            documentos_deterministicos,
            modo_deterministico
        ) = responder_sintomas_sorotipos_por_uf_vsm(
            documentos=documentos_base,
            pergunta=pergunta
        )

        if DEBUG:

            st.subheader(
                "Fluxo determinístico — "
                "sintomas + sorotipos por UF"
            )

            st.write(
                "Modo:",
                modo_deterministico
            )

            st.write(
                "VSMs recuperadas:",
                len(
                    documentos_deterministicos
                )
            )

            st.write(
                "Domínios recuperados:",
                sorted(
                    {
                        doc.metadata.get(
                            "dominio",
                            ""
                        )
                        for doc
                        in documentos_deterministicos
                    }
                )
            )

            st.write(
                "UFs recuperadas:",
                sorted(
                    {
                        doc.metadata.get(
                            "uf_nome",
                            doc.metadata.get(
                                "uf",
                                ""
                            )
                        )
                        for doc
                        in documentos_deterministicos
                    }
                )
            )

        return {
            "pergunta":
                pergunta,

            "resposta":
                resposta_deterministica,

            "classificacao":
                classificacao,

            "escopo":
                resultado_escopo,

            "documentos_recuperados":
                documentos_deterministicos,

            "documentos_contexto":
                documentos_deterministicos,

            "estatisticas_contexto": {

                "modo":
                    modo_deterministico,

                "documentos_recuperados":
                    len(
                        documentos_deterministicos
                    ),

                "chunks_recebidos":
                    len(
                        documentos_deterministicos
                    ),

                "chunks_unicos":
                    len(
                        documentos_deterministicos
                    ),

                "chunks_selecionados":
                    len(
                        documentos_deterministicos
                    ),

                "tokens_contexto":
                    0,

                "limite_tokens":
                    0,
            },

            "contexto":
                "",

            "modo":
                modo_deterministico,
        }



    # --------------------------------------------------------
    # Total de casos por UF
    # --------------------------------------------------------

    if eh_pergunta_total_casos_por_uf(
        pergunta,
        classificacao
    ):

        (
            resposta_deterministica,
            documentos_deterministicos,
            modo_deterministico
        ) = responder_total_casos_por_uf(
            documentos=documentos_base,
            pergunta=pergunta
        )

        if DEBUG:

            st.subheader(
                "Fluxo determinístico"
            )

            st.write(
                "Modo:",
                modo_deterministico
            )

            st.write(
                "VSMs recuperadas:",
                len(documentos_deterministicos)
            )

        return {
            "pergunta":
                pergunta,

            "resposta":
                resposta_deterministica,

            "classificacao":
                classificacao,

            "escopo":
                resultado_escopo,

            "documentos_recuperados":
                documentos_deterministicos,

            "documentos_contexto":
                documentos_deterministicos,

            "estatisticas_contexto": {
                "modo":
                    modo_deterministico,

                "documentos_recuperados":
                    len(documentos_deterministicos),

                "chunks_recebidos":
                    len(documentos_deterministicos),

                "chunks_unicos":
                    len(documentos_deterministicos),

                "chunks_selecionados":
                    len(documentos_deterministicos),

                "tokens_contexto":
                    0,

                "limite_tokens":
                    6000,
            },

            "contexto":
                "",

            "modo":
                modo_deterministico,
        }
    # --------------------------------------------------------
    # Sorotipos por UF
    # --------------------------------------------------------

    if eh_pergunta_sorotipos_por_uf(
        pergunta,
        classificacao
    ):

        (
            resposta_deterministica,
            documentos_deterministicos,
            modo_deterministico
        ) = responder_sorotipos_por_uf(
            documentos=documentos_base,
            pergunta=pergunta
        )

        if DEBUG:

            st.subheader(
                "Fluxo determinístico"
            )

            st.write(
                "Modo:",
                modo_deterministico
            )

            st.write(
                "VSMs recuperadas:",
                len(
                    documentos_deterministicos
                )
            )

            st.write(
                "UFs recuperadas:",
                [
                    doc.metadata.get(
                        "uf_nome"
                    )
                    for doc
                    in documentos_deterministicos
                ]
            )

        return {
            "pergunta":
                pergunta,

            "resposta":
                resposta_deterministica,

            "classificacao":
                classificacao,

            "escopo":
                resultado_escopo,

            "documentos_recuperados":
                documentos_deterministicos,

            "documentos_contexto":
                documentos_deterministicos,

            "estatisticas_contexto": {
                "modo":
                    modo_deterministico,

                "documentos_recuperados":
                    len(
                        documentos_deterministicos
                    ),

                "chunks_recebidos":
                    len(
                        documentos_deterministicos
                    ),

                "chunks_unicos":
                    len(
                        documentos_deterministicos
                    ),

                "chunks_selecionados":
                    len(
                        documentos_deterministicos
                    ),

                "tokens_contexto":
                    0,

                "limite_tokens":
                    6000,
            },

            "contexto":
                "",

            "modo":
                modo_deterministico,
    }

    # --------------------------------------------------------
    # Hospitalizações por UF
    # --------------------------------------------------------

    if eh_pergunta_hospitalizacao_por_uf(
        pergunta,
        classificacao
    ):

        (
            resposta_deterministica,
            documentos_deterministicos,
            modo_deterministico
        ) = responder_hospitalizacao_por_uf(
            documentos=documentos_base,
            pergunta=pergunta
        )

        if DEBUG:

            st.subheader(
                "Fluxo determinístico — hospitalizações por UF"
            )

            st.write(
                "Modo:",
                modo_deterministico
            )

            st.write(
                "VSMs de hospitalização recuperadas:",
                len(documentos_deterministicos)
            )

            st.write(
                "UFs recuperadas:",
                [
                    doc.metadata.get(
                        "uf_nome",
                        doc.metadata.get(
                            "uf",
                            ""
                        )
                    )
                    for doc
                    in documentos_deterministicos
                ]
            )

        return {
            "pergunta":
                pergunta,

            "resposta":
                resposta_deterministica,

            "classificacao":
                classificacao,

            "escopo":
                resultado_escopo,

            "documentos_recuperados":
                documentos_deterministicos,

            "documentos_contexto":
                documentos_deterministicos,

            "estatisticas_contexto": {
                "modo":
                    modo_deterministico,

                "documentos_recuperados":
                    len(documentos_deterministicos),

                "chunks_recebidos":
                    len(documentos_deterministicos),

                "chunks_unicos":
                    len(documentos_deterministicos),

                "chunks_selecionados":
                    len(documentos_deterministicos),

                "tokens_contexto":
                    0,

                "limite_tokens":
                    0,
            },

            "contexto":
                "",

            "modo":
                modo_deterministico,
        }

    # --------------------------------------------------------
    # Óbitos por UF
    # --------------------------------------------------------

    if eh_pergunta_obitos_por_uf(
        pergunta,
        classificacao
    ):

        (
            resposta_deterministica,
            documentos_deterministicos,
            modo_deterministico
        ) = responder_obitos_por_uf(
            documentos=documentos_base,
            pergunta=pergunta
        )

        if DEBUG:

            st.subheader(
                "Fluxo determinístico — óbitos por UF"
            )

            st.write(
                "Modo:",
                modo_deterministico
            )

            st.write(
                "VSMs de óbitos recuperadas:",
                len(
                    documentos_deterministicos
                )
            )

            st.write(
                "UFs recuperadas:",
                [
                    doc.metadata.get(
                        "uf_nome",
                        doc.metadata.get(
                            "uf",
                            ""
                        )
                    )
                    for doc
                    in documentos_deterministicos
                ]
            )

        return {
            "pergunta":
                pergunta,

            "resposta":
                resposta_deterministica,

            "classificacao":
                classificacao,

            "escopo":
                resultado_escopo,

            "documentos_recuperados":
                documentos_deterministicos,

            "documentos_contexto":
                documentos_deterministicos,

            "estatisticas_contexto": {
                "modo":
                    modo_deterministico,

                "documentos_recuperados":
                    len(
                        documentos_deterministicos
                    ),

                "chunks_recebidos":
                    len(
                        documentos_deterministicos
                    ),

                "chunks_unicos":
                    len(
                        documentos_deterministicos
                    ),

                "chunks_selecionados":
                    len(
                        documentos_deterministicos
                    ),

                "tokens_contexto":
                    0,

                "limite_tokens":
                    0,
            },

            "contexto":
                "",

            "modo":
                modo_deterministico,
        }
    # --------------------------------------------------------
    # Sintomas predominantes por UF
    # --------------------------------------------------------

    if eh_pergunta_sintomas_por_estado(
        pergunta,
        classificacao
    ):

        (
            resposta_deterministica,
            documentos_deterministicos,
            modo_deterministico
        ) = responder_sintomas_por_uf_vsm(
            documentos=documentos_base,
            pergunta=pergunta
        )

        if DEBUG:

            st.subheader(
                "Fluxo determinístico — sintomas por UF"
            )

            st.write(
                "Modo:",
                modo_deterministico
            )

            st.write(
                "VSMs clínicas recuperadas:",
                len(
                    documentos_deterministicos
                )
            )

            st.write(
                "UFs recuperadas:",
                [
                    doc.metadata.get(
                        "uf_nome"
                    )
                    for doc
                    in documentos_deterministicos
                ]
            )

        return {
            "pergunta":
                pergunta,

            "resposta":
                resposta_deterministica,

            "classificacao":
                classificacao,

            "escopo":
                resultado_escopo,

            "documentos_recuperados":
                documentos_deterministicos,

            "documentos_contexto":
                documentos_deterministicos,

            "estatisticas_contexto": {
                "modo":
                    modo_deterministico,

                "documentos_recuperados":
                    len(
                        documentos_deterministicos
                    ),

                "chunks_recebidos":
                    len(
                        documentos_deterministicos
                    ),

                "chunks_unicos":
                    len(
                        documentos_deterministicos
                    ),

                "chunks_selecionados":
                    len(
                        documentos_deterministicos
                    ),

                "tokens_contexto":
                    0,

                "limite_tokens":
                    0,
            },

            "contexto":
                "",

            "modo":
                modo_deterministico,
        }
    # --------------------------------------------------------
    # A partir daqui a pergunta precisa de RAG
    # --------------------------------------------------------

    if vectorstore is None:

        vectorstore = (
            carregar_vectorstore_vsm()
        )

    if llm is None:

        llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0
        )
    # --------------------------------------------------------
    # 3. Recuperação híbrida
    # --------------------------------------------------------

    documentos_recuperados = (
        recuperar_evidencias(
            pergunta=pergunta,
            vectorstore=vectorstore,
            documentos_base=documentos_base,
            analise_semantica=analise_semantica
        )
    )

    # --------------------------------------------------------
    # 4. Geração fundamentada
    # --------------------------------------------------------

    resultado_geracao = (
        gerar_resposta_fundamentada(
            pergunta=pergunta,
            documentos_recuperados=
                documentos_recuperados,
            llm=llm,
            limite_tokens_contexto=6000,
            max_chunks_contexto=7
        )
    )

    # --------------------------------------------------------
    # 5. Resultado consolidado
    # --------------------------------------------------------

    return {
        "pergunta":
            pergunta,

        "resposta":
            resultado_geracao[
                "resposta"
            ],
        "classificacao":
            classificacao,

        "escopo":
            resultado_escopo,

        "documentos_recuperados":
            documentos_recuperados,

        "documentos_contexto":
            resultado_geracao[
                "documentos_contexto"
            ],

        "estatisticas_contexto":
            resultado_geracao[
                "estatisticas_contexto"
            ],

        "contexto":
            resultado_geracao[
                "contexto"
            ],
    }

# ============================================================
# PASSO 4.11.5
# TESTES NEGATIVOS DE ESCOPO
# ============================================================

# if DEBUG:

#     st.subheader(
#         "Testes de proteção de escopo — Passo 4.11"
#     )

#     perguntas_teste_411 = [
#         "Qual é o tratamento da dengue?",
#         "Quais são os sintomas da gripe?",
#         "Quem ganhou a Copa do Mundo de 2022?",
#         "Quais sorotipos de dengue foram identificados em São Paulo?",
#     ]

#     for indice, pergunta_teste in enumerate(
#         perguntas_teste_411,
#         start=1
#     ):

#         st.markdown(
#             f"## Teste de escopo {indice}"
#         )

#         st.write(
#             "Pergunta:",
#             pergunta_teste
#         )

#         classificacao_teste = (
#             classificar_pergunta(
#                 pergunta_teste
#             )
#         )

#         resultado_escopo_teste = (
#             validar_escopo_pergunta(
#                 pergunta=pergunta_teste,
#                 classificacao=
#                     classificacao_teste
#             )
#         )

#         st.write(
#             "Status:",
#             resultado_escopo_teste[
#                 "status"
#             ]
#         )

#         st.write(
#             "Motivo:",
#             resultado_escopo_teste[
#                 "motivo"
#             ]
#         )

#         st.write(
#             "Classificação:",
#             classificacao_teste
#         )

#         st.divider()


# ============================================================
# PASSO 4.11.6
# TESTE INTEGRADO DA PROTEÇÃO
# ============================================================

# if DEBUG:

#     st.subheader(
#         "Teste integrado da proteção de escopo"
#     )

#     perguntas_integradas_411 = [
#         "Qual é o tratamento da dengue?",
#         "Quais são os sintomas da gripe?",
#         "Quais sorotipos de dengue foram identificados em São Paulo?",
#     ]

#     for pergunta_teste in perguntas_integradas_411:

#         resultado = (
#             responder_pergunta_vsm(
#                 pergunta=pergunta_teste,
#                 vectorstore=vectorstore_vsm,
#                 documentos_base=documentos_vsm
#             )
#         )

#         st.markdown(
#             f"### {pergunta_teste}"
#         )

#         st.write(
#             "Status:",
#             resultado[
#                 "escopo"
#             ][
#                 "status"
#             ]
#         )

#         st.write(
#             "Resposta:",
#             resultado[
#                 "resposta"
#             ]
#         )

#         st.write(
#             "Chunks recuperados:",
#             len(
#                 resultado[
#                     "documentos_recuperados"
#                 ]
#             )
#         )

#         st.divider()



# ============================================================
# PASSO 4.9.3
# TESTE DO PIPELINE INTEGRADO
# ============================================================

# if DEBUG:

#     st.subheader(
#         "Teste do pipeline integrado — Passo 4.9"
#     )

#     pergunta_teste_49 = (
#         "Qual a incidência de dengue em Goiás?"
#     )

#     llm_teste_49 = ChatOpenAI(
#         model=LLM_MODEL,
#         temperature=0
#     )

#     resultado_49 = (
#         responder_pergunta_vsm(
#             pergunta=pergunta_teste_49,
#             vectorstore=vectorstore_vsm,
#             documentos_base=documentos_vsm,
#             llm=llm_teste_49
#         )
#     )

#     st.markdown(
#         "#### Pergunta"
#     )

#     st.write(
#         resultado_49[
#             "pergunta"
#         ]
#     )

#     st.markdown(
#         "#### Resposta"
#     )

#     st.write(
#         resultado_49[
#             "resposta"
#         ]
#     )

#     st.markdown(
#         "#### Quantidade de chunks recuperados"
#     )

#     st.write(
#         len(
#             resultado_49[
#                 "documentos_recuperados"
#             ]
#         )
#     )

#     st.markdown(
#         "#### Estatísticas do contexto"
#     )

#     st.json(
#         resultado_49[
#             "estatisticas_contexto"
#         ]
#     )


# ============================================================
# PASSO 4.10
# BATERIA DE TESTES MULTIDOMÍNIO
# ============================================================

# if DEBUG:

#     st.subheader(
#         "Bateria de testes multidomínio — Passo 4.10"
#     )

#     perguntas_teste_410 = [
#         "Como evoluíram as notificações de dengue no Brasil em 2026?",
#         "Quais são os principais sintomas de dengue em Minas Gerais?",
#         "Quais sorotipos foram identificados em São Paulo?",
#         "Qual foi a evolução dos casos em relação a cura e óbito no Brasil?",
#         "Qual é o panorama geral da dengue no Brasil em 2026?"
#     ]

#     llm_teste_410 = ChatOpenAI(
#         model=LLM_MODEL,
#         temperature=0
#     )

#     for indice, pergunta_teste in enumerate(
#         perguntas_teste_410,
#         start=1
#     ):

#         st.markdown(
#             f"## Teste {indice}"
#         )

#         st.markdown(
#             "### Pergunta"
#         )

#         st.write(
#             pergunta_teste
#         )

#         resultado_teste = (
#             responder_pergunta_vsm(
#                 pergunta=pergunta_teste,
#                 vectorstore=vectorstore_vsm,
#                 documentos_base=documentos_vsm,
#                 llm=llm_teste_410
#             )
#         )

#         # ----------------------------------------------------
#         # Classificação da pergunta
#         # ----------------------------------------------------

#         classificacao = (
#             classificar_pergunta(
#                 pergunta_teste
#             )
#         )

#         st.markdown(
#             "### Classificação"
#         )

#         st.json(
#             classificacao
#         )

#         # ----------------------------------------------------
#         # Chunks recuperados
#         # ----------------------------------------------------

#         documentos_recuperados = (
#             resultado_teste[
#                 "documentos_recuperados"
#             ]
#         )

#         st.markdown(
#             "### Quantidade de chunks recuperados"
#         )

#         st.write(
#             len(documentos_recuperados)
#         )

#         # ----------------------------------------------------
#         # Documentos-pai recuperados
#         # ----------------------------------------------------

#         pais_recuperados = sorted({
#             doc.metadata.get(
#                 "parent_document_id"
#             )
#             for doc in documentos_recuperados
#             if doc.metadata.get(
#                 "parent_document_id"
#             )
#         })

#         st.markdown(
#             "### Documentos-pai recuperados"
#         )

#         st.write(
#             pais_recuperados
#         )

#         # ----------------------------------------------------
#         # Tipos de documento recuperados
#         # ----------------------------------------------------

#         tipos_recuperados = sorted({
#             doc.metadata.get(
#                 "tipo_documento"
#             )
#             for doc in documentos_recuperados
#             if doc.metadata.get(
#                 "tipo_documento"
#             )
#         })

#         st.markdown(
#             "### Tipos de documentos recuperados"
#         )

#         st.write(
#             tipos_recuperados
#         )

#         # ----------------------------------------------------
#         # UFs recuperadas
#         # ----------------------------------------------------

#         ufs_recuperadas = sorted({
#             doc.metadata.get(
#                 "uf_nome"
#             )
#             for doc in documentos_recuperados
#             if doc.metadata.get(
#                 "uf_nome"
#             )
#         })

#         st.markdown(
#             "### UFs recuperadas"
#         )

#         st.write(
#             ufs_recuperadas
#         )

#         # ----------------------------------------------------
#         # Chunks selecionados para contexto
#         # ----------------------------------------------------

#         documentos_contexto = (
#             resultado_teste[
#                 "documentos_contexto"
#             ]
#         )

#         secoes_contexto = [
#             {
#                 "parent_document_id":
#                     doc.metadata.get(
#                         "parent_document_id"
#                     ),

#                 "secao":
#                     doc.metadata.get(
#                         "secao"
#                     ),

#                 "tipo_documento":
#                     doc.metadata.get(
#                         "tipo_documento"
#                     ),

#                 "uf":
#                     doc.metadata.get(
#                         "uf_nome"
#                     ),
#             }
#             for doc in documentos_contexto
#         ]

#         st.markdown(
#             "### Chunks selecionados para o contexto"
#         )

#         st.json(
#             secoes_contexto
#         )

#         # ----------------------------------------------------
#         # Resposta
#         # ----------------------------------------------------

#         st.markdown(
#             "### Resposta"
#         )

#         st.write(
#             resultado_teste[
#                 "resposta"
#             ]
#         )

#         # ----------------------------------------------------
#         # Estatísticas do contexto
#         # ----------------------------------------------------

#         st.markdown(
#             "### Estatísticas do contexto"
#         )

#         st.json(
#             resultado_teste[
#                 "estatisticas_contexto"
#             ]
#         )

#         st.divider()






# ============================================================
# APOIO À VISUALIZAÇÃO DAS VSMs
# ============================================================

@st.cache_data
def carregar_json_vsm_por_document_id(
    document_id: str
) -> Optional[Dict]:
    """
    Recupera o JSON completo de uma VSM a partir do document_id.

    Esta função é destinada à visualização e não interfere
    no processo de recuperação vetorial ou geração da resposta.
    """

    if not document_id:
        return None

    for doc in documentos_vsm:

        if (
            str(
                doc.metadata.get(
                    "document_id",
                    ""
                )
            ).strip()
            == str(document_id).strip()
        ):

            caminho_relativo = (
                doc.metadata.get(
                    "arquivo_json",
                    ""
                )
            )

            if not caminho_relativo:
                return None

            caminho_json = (
                PASTA_DOCS
                / Path(caminho_relativo)
            )

            if not caminho_json.exists():
                return None

            try:

                with open(
                    caminho_json,
                    "r",
                    encoding="utf-8"
                ) as arquivo:

                    return json.load(
                        arquivo
                    )

            except Exception:
                return None

    return None


# ===========================================================

def extrair_serie_temporal_vsm(
    doc: Document
) -> List[Tuple[int, int]]:

    texto = doc.page_content or ""

    # --------------------------------------------------------
    # Normalizar o Markdown
    # --------------------------------------------------------

    texto_norm = normalizar_texto(
        texto
    )

    # Remover marcações de negrito do Markdown
    texto_norm = texto_norm.replace(
        "**",
        ""
    )

    # --------------------------------------------------------
    # Formato real da VSM:
    #
    # Semana epidemiologica: 1 |
    # Total registros: 4.219 |
    # Percentual total: 0,95%
    # --------------------------------------------------------

    padrao = re.compile(
        r"semana epidemiologica:\s*"
        r"(\d{1,2})"
        r"\s*\|\s*"
        r"total registros:\s*"
        r"([\d\.,]+)",
        flags=re.IGNORECASE
    )

    resultados = {}

    for semana_txt, valor_txt in (
        padrao.findall(
            texto_norm
        )
    ):

        try:

            semana = int(
                semana_txt
            )

            valor = int(
                valor_txt
                .replace(".", "")
                .replace(",", "")
                .strip()
            )

        except ValueError:

            continue

        if 1 <= semana <= 53:

            resultados[
                semana
            ] = valor

    return sorted(
        resultados.items(),
        key=lambda x: x[0]
    )

# ============================================================
def dataframe_temporal_vsm(
    documentos: List[Document],
    ano_filtro: Optional[str] = None
) -> pd.DataFrame:

    registros = []

    # --------------------------------------------------------
    # Priorizar panorama temporal nacional
    # --------------------------------------------------------

    docs_temporais = [
        doc
        for doc in documentos
        if (
            normalizar_texto(
                doc.metadata.get(
                    "dominio",
                    ""
                )
            ) == "temporal"

            and

            normalizar_texto(
                doc.metadata.get(
                    "tipo_documento",
                    ""
                )
            ) == "panorama_temporal_nacional"
        )
    ]

    if ano_filtro:

        docs_temporais = [
            doc
            for doc in docs_temporais
            if str(
                doc.metadata.get(
                    "ano",
                    ""
                )
            ) == str(
                ano_filtro
            )
        ]

    if not docs_temporais:
        return pd.DataFrame(
            columns=[
                "Semana Epidemiológica",
                "Casos"
            ]
        )

    doc = docs_temporais[0]
    if DEBUG:

        st.write(
            "DEBUG — VSM temporal:",
            doc.metadata
        )

        st.write(
            "DEBUG — conteúdo temporal:"
        )

        st.text(
            doc.page_content
        )

    serie = extrair_serie_temporal_vsm(
        doc
    )

    if DEBUG:

        st.write(
            "DEBUG — série temporal extraída:"
        )

        st.write(
            serie
        )

        st.write(
            "Quantidade de semanas:",
            len(serie)
        )

    for semana, casos in serie:

        registros.append({
            "Semana Epidemiológica":
                semana,
            "Casos":
                casos
        })

    df = pd.DataFrame(
        registros
    )

    if not df.empty:

        df = (
            df
            .sort_values(
                "Semana Epidemiológica"
            )
            .reset_index(drop=True)
        )

    return df

# ============================================================


# 2. Crie uma função específica para extrair os sinais clínicos da VSM

# Logo abaixo, coloque:

def extrair_sinais_clinicos_vsm(
    documento_json: Dict
) -> List[Dict]:
    """
    Extrai os sinais clínicos do bloco estruturado
    evidencias.sinais_clinicos da VSM.
    """

    if not isinstance(
        documento_json,
        dict
    ):
        return []

    evidencias = (
        documento_json.get(
            "evidencias",
            {}
        )
        or {}
    )

    sinais = (
        evidencias.get(
            "sinais_clinicos",
            []
        )
        or []
    )

    resultado = []

    for item in sinais:

        if not isinstance(
            item,
            dict
        ):
            continue

        nome = str(
            item.get(
                "sinal_clinico",
                ""
            )
        ).strip()

        if not nome:
            continue

        percentual = item.get(
            "percentual_sim",
            0
        )

        quantidade = item.get(
            "sim",
            0
        )

        avaliaveis = item.get(
            "avaliaveis",
            0
        )

        ausentes = item.get(
            "ausentes",
            0
        )

        try:
            percentual = float(
                percentual
            )
        except (TypeError, ValueError):
            percentual = 0.0

        resultado.append({
            "sinal_clinico":
                nome,

            "percentual_sim":
                percentual,

            "sim":
                numero_seguro(
                    quantidade,
                    0
                ),

            "avaliaveis":
                numero_seguro(
                    avaliaveis,
                    0
                ),

            "ausentes":
                numero_seguro(
                    ausentes,
                    0
                ),
        })

    return sorted(
        resultado,
        key=lambda x:
            x["percentual_sim"],
        reverse=True
    )

# 3. Agora criamos o grafo clínico nacional

# Ainda antes de renderizar_grafo_pyvis(), acrescente:


def construir_grafo_clinico_nacional_vsm(
    documento_json: Dict
) -> nx.DiGraph:

    G = nx.DiGraph()

    sinais = (
        extrair_sinais_clinicos_vsm(
            documento_json
        )
    )

    if not sinais:
        return G

    # --------------------------------------------------------
    # Nó Brasil
    # --------------------------------------------------------

    no_brasil = "BRASIL_CLINICO"

    G.add_node(
        no_brasil,
        label="Brasil",
        tipo="UF",
        title="Brasil"
    )

    # --------------------------------------------------------
    # Grupo de sinais clínicos
    # --------------------------------------------------------

    grupo_id = (
        "GRUPO_SINAIS_CLINICOS_BRASIL"
    )

    G.add_node(
        grupo_id,
        label="Sinais clínicos",
        tipo="Grupo",
        title=(
            "Sinais clínicos registrados "
            "nos registros de dengue"
        ),
        grupo_expandivel=True,
        expandido=False
    )

    G.add_edge(
        no_brasil,
        grupo_id,
        relacao="apresentaPerfilClinico"
    )

    # --------------------------------------------------------
    # Identificar predominante
    # --------------------------------------------------------

    sintoma_predominante = (
        sinais[0]["sinal_clinico"]
    )

    # --------------------------------------------------------
    # Todos os sintomas
    # --------------------------------------------------------

    for item in sinais:

        nome = (
            item["sinal_clinico"]
        )

        percentual = (
            item["percentual_sim"]
        )

        quantidade = (
            item["sim"]
        )

        avaliaveis = (
            item["avaliaveis"]
        )

        ausentes = (
            item["ausentes"]
        )

        node_id = (
            "sintoma_brasil_"
            + slug_no(nome)
        )

        destaque = (
            nome
            == sintoma_predominante
        )

        G.add_node(
            node_id,

            label=(
                f"{nome}\n"
                f"{percentual:.2f}%"
            ),

            tipo=(
                "SintomaDestaque"
                if destaque
                else "Sintoma"
            ),

            title=(
                f"{nome}: "
                f"{percentual:.2f}%"
            ),

            hidden=True,

            parent_group=
                grupo_id,

            percentual=
                round(
                    percentual,
                    2
                ),

            qtd=
                quantidade,

            avaliaveis=
                avaliaveis,

            ausentes=
                ausentes,

            descricao=(
                "Percentual de respostas "
                "positivas entre os registros "
                "avaliáveis para este sinal "
                "clínico."
            )
        )

        G.add_edge(
            grupo_id,
            node_id,
            relacao=(
                "predominante"
                if destaque
                else "contém"
            ),
            hidden=True,
            parent_group=
                grupo_id
        )

    return G

# ============================================================
# 14. GRAFOS
# ============================================================
def listar_ufs_unicas(
    fontes: List[Document]
) -> List[str]:

    return sorted({
        str(
            doc.metadata.get(
                "uf_nome",
                doc.metadata.get(
                    "uf_name",
                    doc.metadata.get(
                        "uf",
                        ""
                    )
                )
            )
        ).strip()

        for doc in fontes

        if str(
            doc.metadata.get(
                "uf_nome",
                doc.metadata.get(
                    "uf_name",
                    doc.metadata.get(
                        "uf",
                        ""
                    )
                )
            )
        ).strip()
    })


import re

# ===========================================================

def extrair_municipios_incidencia_vsm(
    doc: Document
) -> List[Dict]:

    texto = doc.page_content or ""

    # --------------------------------------------------------
    # Formato real da VSM:
    #
    # Codigo municipio: 1100262 |
    # Municipio: Rio Crespo |
    # Populacao 2022: 3.471 |
    # Notificações: 47 |
    # Incidencia 100mil: 1.354,08 |
    # Grau criticidade: Alta
    # --------------------------------------------------------

    padrao = re.compile(
        r"Codigo municipio:\*{0,2}\s*"
        r"(\d+)"
        r"\s*\|\s*\*{0,2}Municipio:\*{0,2}\s*"
        r"([^|]+?)"
        r"\s*\|\s*\*{0,2}Populacao 2022:\*{0,2}\s*"
        r"([\d\.,]+)"
        r"\s*\|\s*\*{0,2}Notifica(?:ções|coes):\*{0,2}\s*"
        r"([\d\.,]+)"
        r"\s*\|\s*\*{0,2}Incidencia 100mil:\*{0,2}\s*"
        r"([\d\.,]+)"
        r"\s*\|\s*\*{0,2}Grau criticidade:\*{0,2}\s*"
        r"([^\-\n]+)",
        flags=re.IGNORECASE
    )

    resultados = []

    for match in padrao.finditer(
        texto
    ):

        codigo = match.group(1).strip()
        municipio = match.group(2).strip()
        populacao_txt = match.group(3).strip()
        notificacoes_txt = match.group(4).strip()
        incidencia_txt = match.group(5).strip()
        criticidade = match.group(6).strip()

        try:

            populacao = int(
                populacao_txt
                .replace(".", "")
                .replace(",", "")
            )

            notificacoes = int(
                notificacoes_txt
                .replace(".", "")
                .replace(",", "")
            )

            # Formato brasileiro:
            # 1.354,08 -> 1354.08
            incidencia = float(
                incidencia_txt
                .replace(".", "")
                .replace(",", ".")
            )

        except ValueError:
            continue

        resultados.append({
            "Código Município": codigo,
            "Município": municipio,
            "UF": str(
                doc.metadata.get(
                    "uf_nome",
                    ""
                )
            ).strip(),
            "População": populacao,
            "Notificações": notificacoes,
            "Incidência por 100 mil":
                incidencia,
            "Criticidade": criticidade
        })

    return resultados

# ==========================================================
#     
def dataframe_incidencia_municipal_vsm(
    documentos: List[Document],
    ano_filtro: Optional[str] = None
) -> pd.DataFrame:

    registros = []

    for doc in documentos:

        if (
            normalizar_texto(
                doc.metadata.get(
                    "dominio",
                    ""
                )
            ) != "geografico"
        ):
            continue

        if (
            normalizar_texto(
                doc.metadata.get(
                    "tipo_documento",
                    ""
                )
            ) != "distribuicao_geografica_uf"
        ):
            continue

        # ----------------------------------------------------
        # Ano
        # ----------------------------------------------------

        if (
            ano_filtro
            and str(
                doc.metadata.get(
                    "ano",
                    ""
                )
            ) != str(
                ano_filtro
            )
        ):
            continue

        registros.extend(
            extrair_municipios_incidencia_vsm(
                doc
            )
        )

    if not registros:

        return pd.DataFrame(
            columns=[
                "Código Município",
                "Município",
                "UF",
                "População",
                "Notificações",
                "Incidência por 100 mil",
                "Criticidade"
            ]
        )

    df = pd.DataFrame(
        registros
    )

    # Evitar eventual duplicidade
    df = (
        df
        .drop_duplicates(
            subset=[
                "Código Município"
            ]
        )
        .sort_values(
            "Incidência por 100 mil",
            ascending=False
        )
        .reset_index(drop=True)
    )
    df["Município / UF"] = (
        df["Município"]
        + " — "
        + df["UF"]
    )
    return df
# ===========================================================

def dataframe_notificacoes_por_uf_vsm(
    documentos: List[Document]
) -> pd.DataFrame:

    registros = []

    for doc in documentos:

        if (
            str(
                doc.metadata.get(
                    "tipo_documento",
                    ""
                )
            ).strip()
            != "distribuicao_geografica_uf"
        ):
            continue

        uf = str(
            doc.metadata.get(
                "uf_nome",
                ""
            )
        ).strip()

        if not uf:
            continue

        total = extrair_total_registros_vsm(
            doc
        )

        if total is None:
            continue

        registros.append({
            "UF": uf,
            "Notificações": total
        })

    if not registros:
        return pd.DataFrame(
            columns=[
                "UF",
                "Notificações"
            ]
        )

    return (
        pd.DataFrame(registros)
        .drop_duplicates(
            subset=["UF"]
        )
        .sort_values(
            "Notificações",
            ascending=False
        )
        .reset_index(drop=True)
    )


def construir_grafo_egocentrico(fontes: List[Document], modo_resposta: str, uf_focal: str) -> nx.DiGraph:
    G = nx.DiGraph()

    if not uf_focal:
        return G

    docs_uf = [
        doc
        for doc in fontes
        if str(
            doc.metadata.get(
                "uf_nome",
                doc.metadata.get(
                    "uf_name",
                    doc.metadata.get(
                        "uf",
                        ""
                    )
                )
            )
        ).strip().lower()
        == uf_focal.strip().lower()
    ]

    if not docs_uf:
        return G

    uf_label = uf_focal
    G.add_node(uf_label, tipo="UF", title=uf_label)

    # ========================================================
    # GRAFO PARA VISÕES SEMÂNTICAS MATERIALIZADAS
    # ========================================================

    if modo_resposta == "vsm":

        # ----------------------------------------------------
        # PERFIL VIROLÓGICO POR UF
        # ----------------------------------------------------

        docs_virologicos = [
            doc
            for doc in docs_uf
            if str(
                doc.metadata.get(
                    "tipo_documento",
                    ""
                )
            ).strip()
            == "perfil_virologico_uf"
        ]

        if docs_virologicos:

            doc = docs_virologicos[0]

            G.nodes[uf_label].update(
                tipo="UF",
                title=uf_label,

                ano=str(
                    doc.metadata.get(
                        "ano",
                        ""
                    )
                ),

                periodo_epidemiologico=str(
                    doc.metadata.get(
                        "periodo_epidemiologico",
                        ""
                    )
                ),

                dominio=str(
                    doc.metadata.get(
                        "dominio",
                        ""
                    )
                ),

                document_id=str(
                    doc.metadata.get(
                        "document_id",
                        ""
                    )
                ),

                fonte=str(
                    doc.metadata.get(
                        "fonte",
                        ""
                    )
                )
            )

            sorotipos = (
                extrair_sorotipos_vsm(
                    doc
                )
            )
            # ------------------------------------------------
            # Identificar sorotipo predominante
            # ------------------------------------------------

            sorotipo_predominante = None
            total_predominante = 0

            if sorotipos:

                (
                    sorotipo_predominante,
                    total_predominante
                ) = max(
                    sorotipos,
                    key=lambda item: item[1]
                )
            # ------------------------------------------------
            # Nó intermediário: Sorotipos
            # ------------------------------------------------

            no_grupo = (
                f"sorotipos_{slug_no(uf_label)}"
            )

            G.add_node(
                no_grupo,
                label="Sorotipos",
                tipo="Categoria",
                title="Sorotipos identificados"
            )

            G.add_edge(
                uf_label,
                no_grupo,
                relacao="possuiSorotiposInformados"
            )

            # ------------------------------------------------
            # Nós DENV-1, DENV-2, DENV-3, DENV-4
            # ------------------------------------------------

            for sorotipo, total in sorotipos:

                no_sorotipo = (
                    f"{slug_no(uf_label)}_"
                    f"{slug_no(sorotipo)}"
                )

                eh_predominante = (
                    sorotipo
                    == sorotipo_predominante
                )

                # ------------------------------------------------
                # Tamanho visual
                # ------------------------------------------------

                tamanho_no = (
                    36
                    if eh_predominante
                    else 22
                )

                # ------------------------------------------------
                # Tipo semântico
                # ------------------------------------------------

                tipo_sorotipo = (
                    "SorotipoPredominante"
                    if eh_predominante
                    else "Sorotipo"
                )

                # ------------------------------------------------
                # Tooltip
                # ------------------------------------------------

                if eh_predominante:

                    titulo = (
                        f"{sorotipo}: "
                        f"{total} registros"
                        f"\nSorotipo predominante em {uf_label}"
                    )

                else:

                    titulo = (
                        f"{sorotipo}: "
                        f"{total} registros"
                    )

                # ------------------------------------------------
                # Criar nó
                # ------------------------------------------------

                G.add_node(
                    no_sorotipo,

                    label=(
                        f"{sorotipo}\n"
                        f"{total} registros"
                    ),

                    tipo=tipo_sorotipo,

                    title=titulo,

                    quantidade=total,

                    unidade="registros",

                    predominante=eh_predominante,

                    size=tamanho_no,

                    uf=uf_label,

                    ano=str(
                        doc.metadata.get(
                            "ano",
                            ""
                        )
                    ),

                    document_id=str(
                        doc.metadata.get(
                            "document_id",
                            ""
                        )
                    ),

                    dominio="virologico"
                )

                # ------------------------------------------------
                # Relação semântica
                # ------------------------------------------------

                relacao_sorotipo = (
                    "temPredominancia"
                    if eh_predominante
                    else "temRegistros"
                )

                G.add_edge(
                    no_grupo,
                    no_sorotipo,
                    relacao=relacao_sorotipo
                )

            # ------------------------------------------------
            # Ano
            # ------------------------------------------------

            ano = str(
                doc.metadata.get(
                    "ano",
                    ""
                )
            ).strip()

            if ano:

                no_ano = (
                    f"ano_{slug_no(uf_label)}_{ano}"
                )

                G.add_node(
                    no_ano,
                    label=f"Ano {ano}",
                    tipo="Tempo",
                    title=f"Ano {ano}"
                )

                G.add_edge(
                    uf_label,
                    no_ano,
                    relacao="referenteAoAno"
                )

            # ------------------------------------------------
            # Período epidemiológico
            # ------------------------------------------------

            periodo = str(
                doc.metadata.get(
                    "periodo_epidemiologico",
                    ""
                )
            ).strip()

            if periodo:

                no_periodo = (
                    f"periodo_{slug_no(uf_label)}"
                )

                G.add_node(
                    no_periodo,
                    label=f"SE {periodo}",
                    tipo="Tempo",
                    title=(
                        "Semanas epidemiológicas "
                        f"{periodo}"
                    )
                )

                G.add_edge(
                    uf_label,
                    no_periodo,
                    relacao="referenteAoPeriodo"
                )

            return G    



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

    print("\n" + "=" * 60)
    print("DEBUG GRAFO EGOCÊNTRICO")
    print("=" * 60)

    print(
        "UF selecionada:",
        uf_focal
    )

    print(
        "Modo da resposta:",
        modo_resposta
    )

    print(
        "Quantidade de documentos da UF:",
        len(docs_uf)
    )

    for i, doc in enumerate(
        docs_uf[:5],
        start=1
    ):

        print(
            f"\nDocumento {i}:"
        )

        print(
            "UF:",
            doc.metadata.get(
                "uf_nome",
                doc.metadata.get(
                    "uf",
                    ""
                )
            )
        )

        print(
            "Domínio:",
            doc.metadata.get(
                "dominio"
            )
        )

        print(
            "Tipo:",
            doc.metadata.get(
                "tipo_documento"
            )
        )

        print(
            "Hospitalização extraída:",
            extrair_hospitalizacao_vsm(
                doc
            )
        )
    if modo_resposta == "deterministico_hospitalizacao":

        doc = docs_uf[0]
        md = doc.metadata

        # ========================================================
        # MÉTRICAS EXTRAÍDAS DIRETAMENTE DA VSM
        # ========================================================

        metricas = (
            extrair_metricas_hospitalizacao_vsm(
                doc
            )
        )

        hospitalizados = metricas[
            "hospitalizados"
        ]

        nao_hospitalizados = metricas[
            "nao_hospitalizados"
        ]

        avaliaveis = metricas[
            "avaliaveis"
        ]

        ignorados = metricas[
            "ignorados"
        ]

        nao_informados = metricas[
            "nao_informados"
        ]

        taxa = metricas[
            "taxa_hospitalizacao"
        ]

        ano = str(
            md.get(
                "ano",
                ""
            )
        )

        # ========================================================
        # NÓ CENTRAL DE HOSPITALIZAÇÃO
        # ========================================================

        nodo_hospitalizacao = (
            f"hospitalizacao_"
            f"{slug_no(uf_label)}"
        )

        G.add_node(
            nodo_hospitalizacao,
            tipo="Categoria",
            title=(
                f"Hospitalização em "
                f"{uf_label}"
            ),
            label="Hospitalização"
        )

        G.add_edge(
            uf_label,
            nodo_hospitalizacao,
            relacao="temHospitalizacao"
        )

        # ========================================================
        # HOSPITALIZADOS
        # ========================================================

        nodo_hospitalizados = (
            f"{nodo_hospitalizacao}_"
            f"hospitalizados"
        )

        G.add_node(
            nodo_hospitalizados,
            tipo="Indicador",
            label=(
                "Hospitalizados\n"
                f"{hospitalizados:,}"
                .replace(",", ".")
            ),
            title=(
                f"Hospitalizados: "
                f"{hospitalizados:,}"
                .replace(",", ".")
            )
        )

        G.add_edge(
            nodo_hospitalizacao,
            nodo_hospitalizados,
            relacao="temHospitalizados"
        )

        # ========================================================
        # NÃO HOSPITALIZADOS
        # ========================================================

        nodo_nao_hospitalizados = (
            f"{nodo_hospitalizacao}_"
            f"nao_hospitalizados"
        )

        G.add_node(
            nodo_nao_hospitalizados,
            tipo="Indicador",
            label=(
                "Não hospitalizados\n"
                f"{nao_hospitalizados:,}"
                .replace(",", ".")
            ),
            title=(
                f"Não hospitalizados: "
                f"{nao_hospitalizados:,}"
                .replace(",", ".")
            )
        )

        G.add_edge(
            nodo_hospitalizacao,
            nodo_nao_hospitalizados,
            relacao="temNaoHospitalizados"
        )

        # ========================================================
        # REGISTROS AVALIÁVEIS
        # ========================================================

        nodo_avaliaveis = (
            f"{nodo_hospitalizacao}_"
            f"avaliaveis"
        )

        G.add_node(
            nodo_avaliaveis,
            tipo="Indicador",
            label=(
                "Registros avaliáveis\n"
                f"{avaliaveis:,}"
                .replace(",", ".")
            ),
            title=(
                f"Registros avaliáveis: "
                f"{avaliaveis:,}"
                .replace(",", ".")
            )
        )

        G.add_edge(
            nodo_hospitalizacao,
            nodo_avaliaveis,
            relacao="temRegistrosAvaliaveis"
        )

        # ========================================================
        # TAXA DE HOSPITALIZAÇÃO
        # ========================================================

        nodo_taxa = (
            f"{nodo_hospitalizacao}_"
            f"taxa"
        )

        G.add_node(
            nodo_taxa,
            tipo="Indicador",
            label=(
                "Taxa de hospitalização\n"
                f"{taxa:.2f}%"
            ),
            title=(
                f"Taxa de hospitalização: "
                f"{taxa:.2f}%"
            )
        )

        G.add_edge(
            nodo_hospitalizacao,
            nodo_taxa,
            relacao="temTaxaHospitalizacao"
        )

        # ========================================================
        # IGNORADOS
        # ========================================================

        if ignorados > 0:

            nodo_ignorados = (
                f"{nodo_hospitalizacao}_"
                f"ignorados"
            )

            G.add_node(
                nodo_ignorados,
                tipo="Indicador",
                label=(
                    "Ignorados\n"
                    f"{ignorados:,}"
                    .replace(",", ".")
                ),
                title=(
                    f"Ignorados: "
                    f"{ignorados:,}"
                    .replace(",", ".")
                )
            )

            G.add_edge(
                nodo_hospitalizacao,
                nodo_ignorados,
                relacao="temIgnorados"
            )

        # ========================================================
        # NÃO INFORMADOS
        # ========================================================

        if nao_informados > 0:

            nodo_nao_informados = (
                f"{nodo_hospitalizacao}_"
                f"nao_informados"
            )

            G.add_node(
                nodo_nao_informados,
                tipo="Indicador",
                label=(
                    "Não informados\n"
                    f"{nao_informados:,}"
                    .replace(",", ".")
                ),
                title=(
                    f"Não informados: "
                    f"{nao_informados:,}"
                    .replace(",", ".")
                )
            )

            G.add_edge(
                nodo_hospitalizacao,
                nodo_nao_informados,
                relacao="temNaoInformados"
            )

        # ========================================================
        # ANO
        # ========================================================

        if ano:

            nodo_ano = (
                f"{nodo_hospitalizacao}_"
                f"ano_{ano}"
            )

            G.add_node(
                nodo_ano,
                tipo="Tempo",
                label=f"Ano {ano}",
                title=f"Ano {ano}"
            )

            G.add_edge(
                nodo_hospitalizacao,
                nodo_ano,
                relacao="referenteAoAno"
            )

        return G

    return G

# ============================================================
# LOCALIZAR VSM NACIONAL
# ============================================================

def buscar_vsm_nacional(
    documentos: List[Document],
    tipo_documento: str,
    ano_filtro: Optional[str] = None
) -> Optional[Document]:

    candidatos = [
        doc
        for doc in documentos
        if normalizar_texto(
            doc.metadata.get(
                "tipo_documento",
                ""
            )
        ) == normalizar_texto(
            tipo_documento
        )
    ]

    if ano_filtro:

        candidatos = [
            doc
            for doc in candidatos
            if str(
                doc.metadata.get(
                    "ano",
                    ""
                )
            ) == str(
                ano_filtro
            )
        ]

    if not candidatos:
        return None

    return candidatos[0]

# ============================================================
# AGREGAR ITENS DAS VSMs ESTADUAIS
# ============================================================

def agregar_itens_vsm_ufs(
    documentos: List[Document],
    dominio: str,
    tipo_documento: str,
    extrator,
    ano_filtro: Optional[str] = None
) -> List[Tuple[str, int]]:

    acumulado = {}

    docs = [
        doc
        for doc in documentos
        if (
            normalizar_texto(
                doc.metadata.get(
                    "dominio",
                    ""
                )
            ) == normalizar_texto(
                dominio
            )

            and

            normalizar_texto(
                doc.metadata.get(
                    "tipo_documento",
                    ""
                )
            ) == normalizar_texto(
                tipo_documento
            )
        )
    ]

    if ano_filtro:

        docs = [
            doc
            for doc in docs
            if str(
                doc.metadata.get(
                    "ano",
                    ""
                )
            ) == str(
                ano_filtro
            )
        ]

    for doc in docs:

        itens = extrator(
            doc
        )

        for nome, quantidade in itens:

            chave = normalizar_texto(
                nome
            )

            if chave not in acumulado:

                acumulado[chave] = {
                    "nome": nome,
                    "quantidade": 0
                }

            acumulado[chave][
                "quantidade"
            ] += quantidade

    resultado = [
        (
            dados["nome"],
            dados["quantidade"]
        )
        for dados
        in acumulado.values()
    ]

    return sorted(
        resultado,
        key=lambda item: item[1],
        reverse=True
    )

# ============================================================
# GRAFO COMBINADO — NÍVEL NACIONAL
# ============================================================

def construir_grafo_combinado_nacional(
    documentos: List[Document],
    ano_filtro: Optional[str] = None,
    incluir_sintomas: bool = False,
    incluir_sorotipos: bool = False,
    incluir_hospitalizacao: bool = False,
    incluir_obitos: bool = False
) -> nx.DiGraph:

    G = nx.DiGraph()

    # --------------------------------------------------------
    # NÓ CENTRAL
    # --------------------------------------------------------

    no_brasil = "BRASIL"

    G.add_node(
        no_brasil,
        label="Brasil",
        tipo="Pais",
        title=(
            "Panorama nacional da dengue"
            + (
                f" — {ano_filtro}"
                if ano_filtro
                else ""
            )
        )
    )

    # ========================================================
    # SINTOMAS
    # ========================================================

    if incluir_sintomas:

        doc_nacional = buscar_vsm_nacional(
            documentos=documentos,
            tipo_documento=
                "panorama_clinico_nacional",
            ano_filtro=ano_filtro
        )

        sintomas = []

        if doc_nacional:

            sintomas = (
                extrair_sintomas_vsm_grafo(
                    doc_nacional
                )
            )

        # Fallback: agregar UFs
        if not sintomas:

            sintomas = agregar_itens_vsm_ufs(
                documentos=documentos,
                dominio="clinico",
                tipo_documento=
                    "perfil_clinico_uf",
                extrator=
                    extrair_sintomas_vsm,
                ano_filtro=ano_filtro
            )

        if sintomas:

            predominante = (
                sintomas[0][0]
            )

            adicionar_no_expandivel(
                G=G,
                no_pai=no_brasil,
                grupo_id=
                    "grupo_sintomas_brasil",
                grupo_label="Sintomas",
                relacao_pai_grupo=
                    "apresentaPerfilClinico",
                itens=sintomas,
                tipo_item="Sintoma",
                prefixo_item=
                    "sintoma_brasil",
                item_destaque=
                    predominante,
                rotulo_destaque=
                    "Predominante",
                total_referencia=None,
                uf="Brasil",
                ano=ano_filtro,
                arquivo_origem=(
                    doc_nacional.metadata.get(
                        "arquivo_json",
                        ""
                    )
                    if doc_nacional
                    else ""
                ),
                base_descricao=
                    "dos registros clínicos"
            )

    # ========================================================
    # SOROTIPOS
    # ========================================================

    if incluir_sorotipos:

        doc_nacional = buscar_vsm_nacional(
            documentos=documentos,
            tipo_documento=
                "panorama_virologico_nacional",
            ano_filtro=ano_filtro
        )

        sorotipos = []

        if doc_nacional:

            sorotipos = (
                extrair_sorotipos_vsm(
                    doc_nacional
                )
            )

        if not sorotipos:

            sorotipos = agregar_itens_vsm_ufs(
                documentos=documentos,
                dominio="virologico",
                tipo_documento=
                    "perfil_virologico_uf",
                extrator=
                    extrair_sorotipos_vsm,
                ano_filtro=ano_filtro
            )

        if sorotipos:

            sorotipos = (
                consolidar_itens_por_maior_valor(
                    sorotipos
                )
            )

            total_sorotipos = sum(
                qtd
                for _, qtd
                in sorotipos
            )

            predominante = (
                sorotipos[0][0]
            )

            adicionar_no_expandivel(
                G=G,
                no_pai=no_brasil,
                grupo_id=
                    "grupo_sorotipos_brasil",
                grupo_label="Sorotipos",
                relacao_pai_grupo=
                    "temRegistroDeSorotipo",
                itens=sorotipos,
                tipo_item="Sorotipo",
                prefixo_item=
                    "sorotipo_brasil",
                item_destaque=
                    predominante,
                rotulo_destaque=
                    "Predominante",
                total_referencia=(
                    total_sorotipos
                    if total_sorotipos > 0
                    else None
                ),
                uf="Brasil",
                ano=ano_filtro,
                arquivo_origem=(
                    doc_nacional.metadata.get(
                        "arquivo_json",
                        ""
                    )
                    if doc_nacional
                    else ""
                ),
                base_descricao=(
                    "dos registros com "
                    "sorotipo informado"
                )
            )

    # ========================================================
    # HOSPITALIZAÇÕES
    # ========================================================

    if incluir_hospitalizacao:

        doc_nacional = buscar_vsm_nacional(
            documentos=documentos,
            tipo_documento=
                "panorama_desfechos_nacional",
            ano_filtro=ano_filtro
        )

        hospitalizacao = []

        if doc_nacional:

            hospitalizacao = (
                extrair_hospitalizacao_vsm(
                    doc_nacional
                )
            )

        if not hospitalizacao:

            hospitalizacao = (
                agregar_itens_vsm_ufs(
                    documentos=documentos,
                    dominio="desfechos",
                    tipo_documento=
                        "desfechos_uf",
                    extrator=
                        extrair_hospitalizacao_vsm,
                    ano_filtro=ano_filtro
                )
            )

        if hospitalizacao:

            total = sum(
                qtd
                for _, qtd
                in hospitalizacao
            )

            adicionar_no_expandivel(
                G=G,
                no_pai=no_brasil,
                grupo_id=
                    "grupo_hospitalizacao_brasil",
                grupo_label="Hospitalizações",
                relacao_pai_grupo=
                    "temHospitalizacao",
                itens=hospitalizacao,
                tipo_item=
                    "Hospitalizacao",
                prefixo_item=
                    "hospitalizacao_brasil",
                total_referencia=(
                    total
                    if total > 0
                    else None
                ),
                uf="Brasil",
                ano=ano_filtro,
                arquivo_origem=(
                    doc_nacional.metadata.get(
                        "arquivo_json",
                        ""
                    )
                    if doc_nacional
                    else ""
                )
            )

    # ========================================================
    # ÓBITOS
    # ========================================================

    if incluir_obitos:

        doc_nacional = buscar_vsm_nacional(
            documentos=documentos,
            tipo_documento=
                "panorama_obitos_nacional",
            ano_filtro=ano_filtro
        )

        obitos = []

        if doc_nacional:

            obitos = (
                extrair_obitos_vsm(
                    doc_nacional
                )
            )

        if not obitos:

            obitos = (
                agregar_itens_vsm_ufs(
                    documentos=documentos,
                    dominio="desfechos",
                    tipo_documento=
                        "perfil_obitos_uf",
                    extrator=
                        extrair_obitos_vsm,
                    ano_filtro=ano_filtro
                )
            )

        if obitos:

            total = sum(
                qtd
                for _, qtd
                in obitos
            )

            adicionar_no_expandivel(
                G=G,
                no_pai=no_brasil,
                grupo_id=
                    "grupo_obitos_brasil",
                grupo_label="Óbitos",
                relacao_pai_grupo=
                    "temObito",
                itens=obitos,
                tipo_item="Obito",
                prefixo_item=
                    "obito_brasil",
                total_referencia=(
                    total
                    if total > 0
                    else None
                ),
                uf="Brasil",
                ano=ano_filtro,
                arquivo_origem=(
                    doc_nacional.metadata.get(
                        "arquivo_json",
                        ""
                    )
                    if doc_nacional
                    else ""
                )
            )

    return G

def construir_grafo_combinado_por_uf(
    documentos: List[Document],
    uf_focal: str,
    ano_filtro: Optional[str] = None,
    incluir_casos: bool = False,
    incluir_sintomas: bool = False,
    incluir_sorotipos: bool = False,
    incluir_hospitalizacao: bool = False,
    incluir_obitos: bool = False
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
            doc
            for doc in documentos
            if (
                normalizar_texto(
                    doc.metadata.get(
                        "uf_nome",
                        ""
                    )
                )
                == normalizar_texto(
                    uf_focal
                )

                and

                normalizar_texto(
                    doc.metadata.get(
                        "dominio",
                        ""
                    )
                ) == "clinico"

                and

                normalizar_texto(
                    doc.metadata.get(
                        "tipo_documento",
                        ""
                    )
                ) == "perfil_clinico_uf"
            )
        ]
        if ano_filtro:
            docs_sintomas = [
                doc
                for doc in docs_sintomas
                if str(
                    doc.metadata.get(
                        "ano",
                        ""
                    )
                ) == str(ano_filtro)
            ]

        if docs_sintomas:

            doc = docs_sintomas[0]

            if DEBUG:

                st.subheader(
                    "DEBUG — VSM clínica selecionada"
                )

                st.write(
                    "Document ID:",
                    doc.metadata.get(
                        "document_id"
                    )
                )

                st.write(
                    "UF:",
                    doc.metadata.get(
                        "uf_nome"
                    )
                )

                st.write(
                    "Tipo:",
                    doc.metadata.get(
                        "tipo_documento"
                    )
                )

                st.markdown(
                    doc.page_content
                )

            sintomas = extrair_sintomas_vsm_grafo(
                doc
            )

            if DEBUG:

                st.write(
                    "Sintomas extraídos:",
                    sintomas
                )

            if sintomas:
                sintoma_pred = sintomas[0][0]

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
                    total_referencia=None,
                    uf=uf_focal,
                    ano=str(
                        doc.metadata.get(
                            "ano",
                            ""
                        )
                    ),
                    arquivo_origem=(
                        doc.metadata.get(
                            "arquivo_json",
                            ""
                        )
                    ),
                    base_descricao="dos registros clínicos"
                )

    if incluir_sorotipos:
        docs_sorotipos = [
            doc
            for doc in documentos
            if (
                normalizar_texto(
                    doc.metadata.get(
                        "uf_nome",
                        ""
                    )
                )
                == normalizar_texto(
                    uf_focal
                )

                and

                normalizar_texto(
                    doc.metadata.get(
                        "dominio",
                        ""
                    )
                ) == "virologico"

                and

                normalizar_texto(
                    doc.metadata.get(
                        "tipo_documento",
                        ""
                    )
                ) == "perfil_virologico_uf"
            )
        ]
        if ano_filtro:
            docs_sorotipos = [
                doc
                for doc in docs_sorotipos
                if str(
                    doc.metadata.get(
                        "ano",
                        ""
                    )
                ) == str(ano_filtro)
            ]

        sorotipos = []

        for doc in docs_sorotipos:

            sorotipos.extend(
                extrair_sorotipos_vsm(
                    doc
                )
            )            

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
                arquivo_origem=(
                    docs_sorotipos[0]
                    .metadata.get(
                        "arquivo_json",
                        ""
                    )
                    if docs_sorotipos
                    else ""
                ),
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
        
    # ============================================================
    # HOSPITALIZAÇÃO — DOMÍNIO DESFECHOS
    # ============================================================

    if incluir_hospitalizacao:

        docs_hospitalizacao = [
            doc
            for doc in documentos
            if (
                normalizar_texto(
                    doc.metadata.get(
                        "uf_nome",
                        ""
                    )
                )
                == normalizar_texto(
                    uf_focal
                )

                and

                normalizar_texto(
                    doc.metadata.get(
                        "dominio",
                        ""
                    )
                ) == "desfechos"

                and

                normalizar_texto(
                    doc.metadata.get(
                        "tipo_documento",
                        ""
                    )
                ) == "desfechos_uf"
            )
        ]

        if ano_filtro:

            docs_hospitalizacao = [
                doc
                for doc in docs_hospitalizacao
                if str(
                    doc.metadata.get(
                        "ano",
                        ""
                    )
                ) == str(
                    ano_filtro
                )
            ]

        if docs_hospitalizacao:

            doc_hosp = docs_hospitalizacao[0]

            hospitalizacao = (
                extrair_hospitalizacao_vsm(
                    doc_hosp
                )
            )

            if hospitalizacao:

                grupo_hosp_id = (
                    f"grupo_hospitalizacao_"
                    f"{uf_slug}"
                )

                G.add_node(
                    grupo_hosp_id,
                    label="Hospitalizações",
                    tipo="Grupo",
                    title=(
                        "Indicadores de hospitalização "
                        f"em {uf_focal}"
                    ),
                    grupo_expandivel=True,
                    expandido=False,
                    uf=uf_focal,
                    ano=str(
                        doc_hosp.metadata.get(
                            "ano",
                            ""
                        )
                    ),
                    descricao=(
                        "Desfechos relacionados "
                        "à hospitalização"
                    )
                )

                G.add_edge(
                    uf_focal,
                    grupo_hosp_id,
                    relacao="temHospitalizacao"
                )

                total_hosp = sum(
                    valor
                    for _, valor
                    in hospitalizacao
                )

                for nome, valor in hospitalizacao:

                    item_id = (
                        f"{grupo_hosp_id}_"
                        f"{slug_no(nome)}"
                    )

                    size, percentual = (
                        calcular_tamanho_por_percentual(
                            valor,
                            (
                                total_hosp
                                if total_hosp > 0
                                else None
                            )
                        )
                    )

                    G.add_node(
                        item_id,
                        label=(
                            f"{nome}\n"
                            f"{valor:,}"
                            .replace(",", ".")
                        ),
                        tipo="Hospitalizacao",
                        title=(
                            f"{nome}: "
                            f"{valor:,}"
                            .replace(",", ".")
                        ),
                        hidden=True,
                        parent_group=grupo_hosp_id,
                        size=size,
                        qtd=valor,
                        percentual=round(
                            percentual,
                            2
                        ),
                        uf=uf_focal,
                        ano=str(
                            doc_hosp.metadata.get(
                                "ano",
                                ""
                            )
                        )
                    )

                    G.add_edge(
                        grupo_hosp_id,
                        item_id,
                        relacao="contém",
                        hidden=True,
                        parent_group=grupo_hosp_id
                    )
    
    # ============================================================
    # ÓBITOS — DOMÍNIO DESFECHOS
    # ============================================================

    if incluir_obitos:

        docs_obitos = [
            doc
            for doc in documentos
            if (
                normalizar_texto(
                    doc.metadata.get(
                        "uf_nome",
                        ""
                    )
                )
                == normalizar_texto(
                    uf_focal
                )

                and

                normalizar_texto(
                    doc.metadata.get(
                        "dominio",
                        ""
                    )
                ) == "desfechos"

                and

                normalizar_texto(
                    doc.metadata.get(
                        "tipo_documento",
                        ""
                    )
                ) == "perfil_obitos_uf"
            )
        ]

        if ano_filtro:

            docs_obitos = [
                doc
                for doc in docs_obitos
                if str(
                    doc.metadata.get(
                        "ano",
                        ""
                    )
                ) == str(
                    ano_filtro
                )
            ]

        if docs_obitos:

            doc_obito = docs_obitos[0]

            obitos = (
                extrair_obitos_vsm(
                    doc_obito
                )
            )

            if obitos:

                grupo_obitos_id = (
                    f"grupo_obitos_"
                    f"{uf_slug}"
                )

                G.add_node(
                    grupo_obitos_id,
                    label="Óbitos",
                    tipo="Grupo",
                    title=(
                        "Indicadores de óbitos "
                        f"em {uf_focal}"
                    ),
                    grupo_expandivel=True,
                    expandido=False,
                    uf=uf_focal,
                    ano=str(
                        doc_obito.metadata.get(
                            "ano",
                            ""
                        )
                    ),
                    descricao=(
                        "Desfechos relacionados "
                        "a óbitos"
                    )
                )

                G.add_edge(
                    uf_focal,
                    grupo_obitos_id,
                    relacao="temObito"
                )

                total_obitos = sum(
                    valor
                    for _, valor
                    in obitos
                )

                for nome, valor in obitos:

                    item_id = (
                        f"{grupo_obitos_id}_"
                        f"{slug_no(nome)}"
                    )

                    size, percentual = (
                        calcular_tamanho_por_percentual(
                            valor,
                            (
                                total_obitos
                                if total_obitos > 0
                                else None
                            )
                        )
                    )

                    G.add_node(
                        item_id,
                        label=(
                            f"{nome}\n"
                            f"{valor:,}"
                            .replace(",", ".")
                        ),
                        tipo="Obito",
                        title=(
                            f"{nome}: "
                            f"{valor:,}"
                            .replace(",", ".")
                        ),
                        hidden=True,
                        parent_group=grupo_obitos_id,
                        size=size,
                        qtd=valor,
                        percentual=round(
                            percentual,
                            2
                        ),
                        uf=uf_focal,
                        ano=str(
                            doc_obito.metadata.get(
                                "ano",
                                ""
                            )
                        )
                    )

                    G.add_edge(
                        grupo_obitos_id,
                        item_id,
                        relacao="contém",
                        hidden=True,
                        parent_group=grupo_obitos_id
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
        elif tipo == "Hospitalizacao":

            node["shape"] = "dot"
            node["size"] = (
                tamanho_original
                if tamanho_original is not None
                else 18
            )
            node["borderWidth"] = 2


        elif tipo == "Obito":

            node["shape"] = "dot"
            node["size"] = (
                tamanho_original
                if tamanho_original is not None
                else 18
            )
            node["borderWidth"] = 2
            
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

            let linhas = "";

            function adicionarCampo(rotulo, valor) {
                if (
                    valor !== undefined &&
                    valor !== null &&
                    String(valor).trim() !== "" &&
                    String(valor).trim() !== "-"
                ) {
                    linhas += `
                        <div><b>${escapeHtml(rotulo)}</b></div>
                        <div>${escapeHtml(valor)}</div>
                    `;
                }
            }

            adicionarCampo(
                "ID",
                nodeData.id
            );

            adicionarCampo(
                "Tipo",
                nodeData.tipo
            );

            adicionarCampo(
                "Descrição",
                nodeData.title
            );

            adicionarCampo(
                "Quantidade",
                nodeData.quantidade
            );

            adicionarCampo(
                "Unidade",
                nodeData.unidade
            );

            adicionarCampo(
                "UF",
                nodeData.uf
            );

            adicionarCampo(
                "Ano",
                nodeData.ano
            );

            adicionarCampo(
                "Período epidemiológico",
                nodeData.periodo_epidemiologico
            );

            adicionarCampo(
                "Domínio",
                nodeData.dominio
            );

            adicionarCampo(
                "Fonte",
                nodeData.fonte
            );

            adicionarCampo(
                "Documento de origem",
                nodeData.document_id
            );

            adicionarCampo(
                "Percentual",
                nodeData.percentual
            );

            adicionarCampo(
                "Grupo pai",
                nodeData.parent_group
            );
            // ========================================================
            // ITENS DO GRUPO
            // ========================================================

            let itensGrupoHtml = "";

            if (nodeData.grupo_expandivel) {

                const itensGrupo = nodes
                    .get()
                    .filter(
                        n => n.parent_group === nodeData.id
                    )
                    .sort(
                        (a, b) =>
                            (Number(b.qtd) || 0)
                            -
                            (Number(a.qtd) || 0)
                    );

                if (itensGrupo.length > 0) {

                    itensGrupoHtml = `
                        <div
                            style="
                                margin-top:18px;
                                border-top:1px solid #d1d5db;
                                padding-top:12px;
                            "
                        >
                            <h4
                                style="
                                    margin:0 0 10px 0;
                                    color:#374151;
                                "
                            >
                                Informações do grupo
                            </h4>

                            <div
                                style="
                                    display:flex;
                                    flex-direction:column;
                                    gap:6px;
                                "
                            >
                    `;

                    itensGrupo.forEach(
                        item => {

                            const nome =
                                item.label
                                ?? item.id
                                ?? "-";

                            const quantidade =
                                item.qtd
                                ?? item.quantidade
                                ?? "-";

                            const percentual =
                                item.percentual;

                            let complemento =
                                `${escapeHtml(quantidade)} registros`;

                            if (
                                percentual !== undefined
                                &&
                                percentual !== null
                                &&
                                percentual !== "-"
                            ) {
                                complemento +=
                                    ` — ${escapeHtml(percentual)}%`;
                            }

                            itensGrupoHtml += `
                                <div
                                    style="
                                        padding:8px 10px;
                                        background:#ffffff;
                                        border:1px solid #e5e7eb;
                                        border-radius:6px;
                                        font-size:14px;
                                    "
                                >
                                    <b>${escapeHtml(nome)}</b>
                                    <br>
                                    <span
                                        style="
                                            color:#4b5563;
                                        "
                                    >
                                        ${complemento}
                                    </span>
                                </div>
                            `;
                        }
                    );

                    itensGrupoHtml += `
                            </div>
                        </div>
                    `;
                }
            }

            panel.innerHTML = `
                <h3
                    style="
                        margin-top:0;
                        color:#1f2937;
                    "
                >
                    ${escapeHtml(label)}
                </h3>

                <div
                    style="
                        display:grid;
                        grid-template-columns:160px 1fr;
                        gap:8px 10px;
                        font-size:14px;
                    "
                >
                    ${linhas}
                </div>

                ${itensGrupoHtml}
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

def renderizar_linha_temporal_df(
    df: pd.DataFrame,
    coluna_tempo: str,
    coluna_valor: str
):

    if df.empty:

        st.warning(
            "Não há dados suficientes "
            "para gerar a série temporal."
        )

        return

    grafico = (
        alt.Chart(df)
        .mark_line(
            point=True
        )
        .encode(
            x=alt.X(
                f"{coluna_tempo}:O",
                title=coluna_tempo,
                sort=None
            ),

            y=alt.Y(
                f"{coluna_valor}:Q",
                title=coluna_valor
            ),

            tooltip=[
                alt.Tooltip(
                    f"{coluna_tempo}:O",
                    title="Semana"
                ),
                alt.Tooltip(
                    f"{coluna_valor}:Q",
                    title="Casos",
                    format=","
                )
            ]
        )
        .properties(
            width=900,
            height=450
        )
    )

    st.altair_chart(
        grafico,
        use_container_width=True
    )

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

        texto = normalizar_texto(
            instrucao_usuario
        )

        usar_taxa = any(
            termo in texto
            for termo in [
                "taxa",
                "proporcao",
                "proporcoes",
                "percentual",
                "percentuais",
            ]
        )

        if usar_taxa:

            renderizar_barras_df(
                df,
                "UF",
                "Taxa de Hospitalização (%)",
                top_n=(
                    10
                    if visualizacao_escolhida == "Top 10"
                    else None
                )
            )

        else:

            renderizar_barras_df(
                df,
                "UF",
                "Hospitalizações",
                top_n=(
                    10
                    if visualizacao_escolhida == "Top 10"
                    else None
                )
            )

        return

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
# CALLBACK — PERGUNTA SUGERIDA
# ============================================================

def atualizar_pergunta_sugerida():

    texto_sugerido = (
        st.session_state.get(
            "select_pergunta_sugerida",
            ""
        )
    )

    st.session_state.campo_pergunta = (
        texto_sugerido
    )

    st.session_state.pergunta_atual = (
        texto_sugerido
    )

    st.session_state.executar_consulta = False


# ============================================================
# 17. INTERFACE STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Agente Inteligente de Arboviroses",
    layout="wide"
)

st.title(
    "🦟 Agente Inteligente de Arboviroses"
)

st.markdown(
    "Consulta semântica baseada em Visões Semânticas "
    "Materializadas (VSMs), com recuperação híbrida "
    "e geração de respostas fundamentadas em evidências."
)

# ============================================================
# CONSULTA SEMÂNTICA
# ============================================================

st.subheader("Consulta semântica")

dimensao_selecionada = st.selectbox(
    "Dimensão da consulta",
    options=list(DIMENSOES_PERGUNTAS.keys()),
    index=0
)

perguntas_disponiveis = (
    DIMENSOES_PERGUNTAS[
        dimensao_selecionada
    ]["perguntas"]
)

pergunta_sugerida = st.selectbox(
    "Pergunta sugerida",
    options=perguntas_disponiveis
)

st.caption(
    "Você pode selecionar uma pergunta sugerida "
    "ou escrever sua própria pergunta."
)

pergunta_livre = st.text_input(
    "Digite sua própria pergunta",
    placeholder=(
        "Ex.: Quais sintomas predominantes "
        "e sorotipos aparecem em cada estado?"
    )
)


# ============================================================
# DEFINIR A PERGUNTA ATUAL
# ============================================================

if pergunta_livre.strip():
    pergunta = pergunta_livre.strip()
else:
    pergunta = pergunta_sugerida


# ============================================================
# ANALISAR DIMENSÕES E GRANULARIDADE
# ============================================================

analise_interface = analisar_pergunta(
    pergunta
)

dominios_identificados = (
    analise_interface["dominios"]
)

granularidade_identificada = (
    analise_interface["granularidade"]
)

if dominios_identificados:

    st.info(
        "Dimensões identificadas: "
        + ", ".join(
            dominios_identificados
        )
        + (
            f" | Granularidade: "
            f"{granularidade_identificada}"
            if granularidade_identificada
            else ""
        )
    )


# ============================================================
# BOTÕES
# ============================================================

col_a, col_b = st.columns([
    1,
    1
])

with col_a:

    if st.button("Consultar"):

        st.session_state.pergunta_atual = (
            pergunta
        )

        st.session_state.executar_consulta = True


with col_b:

    if st.button("Limpar pergunta"):

        st.session_state.pergunta_atual = ""
        st.session_state.executar_consulta = False
        st.session_state.ultimo_resultado = None

        st.rerun()



resultado_vsm = responder_pergunta_vsm(
    pergunta=pergunta,
    documentos_base=documentos_vsm,
    analise_semantica=analise_interface
)

# ============================================================
# PASSO 4.12
# INTEGRAÇÃO DO PIPELINE VSM COM A INTERFACE
# ============================================================

pergunta = (
    st.session_state
    .pergunta_atual
    .strip()
)

if (
    st.session_state.executar_consulta
    and pergunta
):

    with st.spinner(
        "Consultando evidências epidemiológicas..."
    ):

        # ----------------------------------------------------
        # IMPORTANTE
        #
        # Não recriar:
        # - documentos_vsm
        # - embeddings
        # - vectorstore
        #
        # Ambos já foram carregados e validados
        # anteriormente na inicialização da aplicação.
        # ----------------------------------------------------

        resultado_vsm = responder_pergunta_vsm(
            pergunta=pergunta,
            documentos_base=documentos_vsm,
            analise_semantica=analise_interface
        )

        # ----------------------------------------------------
        # Identificar modo da resposta
        # ----------------------------------------------------

        escopo_resultado = (
            resultado_vsm.get(
                "escopo",
                {}
            )
            or {}
        )

        status_escopo = (
            escopo_resultado.get(
                "status",
                "valido"
            )
        )

        if status_escopo == "valido":

            modo_resposta = (
                resultado_vsm.get(
                    "modo",
                    "vsm"
                )
            )

        elif status_escopo == "sem_cobertura":

            modo_resposta = (
                "escopo_sem_cobertura"
            )

        else:

            modo_resposta = (
                "escopo_fora_dominio"
            )

        # ----------------------------------------------------
        # Usar os chunks efetivamente enviados ao LLM
        # como fontes da resposta.
        #
        # Para respostas bloqueadas pelo gate,
        # documentos_contexto será vazio.
        # ----------------------------------------------------

        fontes = (
            resultado_vsm.get(
                "documentos_contexto",
                []
            )
            or []
        )

        # ----------------------------------------------------
        # Armazenar resultado na sessão
        # ----------------------------------------------------

        st.session_state.ultimo_resultado = {

            "pergunta":
                pergunta,

            "resposta":
                resultado_vsm[
                    "resposta"
                ],

            "fontes":
                fontes,

            "modo_resposta":
                modo_resposta,

            "documentos":
                documentos_vsm,

            "classificacao":
                resultado_vsm.get(
                    "classificacao",
                    {}
                ),

            "escopo":
                escopo_resultado,

            "estatisticas_contexto":
                resultado_vsm.get(
                    "estatisticas_contexto",
                    {}
                ),

            "documentos_recuperados":
                resultado_vsm.get(
                    "documentos_recuperados",
                    []
                ),
        }

    # --------------------------------------------------------
    # Evita executar novamente o RAG a cada rerun do Streamlit
    # --------------------------------------------------------

    st.session_state.executar_consulta = False

if st.session_state.ultimo_resultado:

    resultado = st.session_state.ultimo_resultado

    pergunta = resultado["pergunta"]
    resposta = resultado["resposta"]
    fontes = resultado.get("fontes", [])
    modo_resposta = resultado.get("modo_resposta", "")
    documentos = resultado.get("documentos", [])

    classificacao = resultado.get(
        "classificacao",
        {}
    )

    escopo = resultado.get(
        "escopo",
        {}
    )

    estatisticas_contexto = resultado.get(
        "estatisticas_contexto",
        {}
    )

    # ========================================================
    # PERGUNTA E RESPOSTA
    # ========================================================

    st.subheader("Pergunta atual")
    st.write(pergunta)

    st.subheader("Resposta")
    st.write(resposta)

    st.subheader("Evidências utilizadas")

    # ========================================================
    # 1. RESPOSTAS BLOQUEADAS PELO GATE DE ESCOPO
    # ========================================================

    if modo_resposta in [
        "escopo_sem_cobertura",
        "escopo_fora_dominio",
    ]:

        status = escopo.get(
            "status",
            ""
        )

        if status == "sem_cobertura":

            st.info(
                "A consulta pertence ao domínio da dengue, "
                "mas não possui cobertura nas VSMs disponíveis."
            )

        elif status == "fora_dominio":

            st.info(
                "A consulta está fora do domínio coberto "
                "pela aplicação."
            )

        st.caption(
            "Nenhuma evidência foi recuperada para esta consulta."
        )

        if DEBUG:

            with st.expander(
                "Detalhes da proteção de escopo"
            ):

                st.markdown(
                    "#### Classificação semântica"
                )

                st.json(
                    classificacao
                )

                st.markdown(
                    "#### Validação de escopo"
                )

                st.json(
                    escopo
                )

                st.write(
                    "Chunks recuperados: 0"
                )

    elif modo_resposta == "deterministico_grafo_combinado":

        st.info(
            "Selecione uma UF para gerar o grafo "
            "combinado de sintomas, sorotipos, "
            "hospitalizações e óbitos."
        )

        todas_ufs = sorted({
            str(
                doc.metadata.get(
                    "uf_nome",
                    ""
                )
            ).strip()
            for doc in documentos
            if str(
                doc.metadata.get(
                    "uf_nome",
                    ""
                )
            ).strip()
        })

        if todas_ufs:

            uf_grafo_combinado = st.selectbox(
                "Escolha a UF",
                todas_ufs,
                key="uf_grafo_combinado_pergunta"
            )

            if st.button(
                "Gerar grafo combinado",
                key="btn_grafo_combinado_pergunta"
            ):

                G = construir_grafo_combinado_por_uf(
                    documentos=documentos,
                    uf_focal=uf_grafo_combinado,
                    ano_filtro=str(ANO),
                    incluir_casos=False,
                    incluir_sintomas=True,
                    incluir_sorotipos=True,
                    incluir_hospitalizacao=True,
                    incluir_obitos=True
                )

                renderizar_grafo_pyvis(
                    G,
                    (
                        "Grafo semântico integrado de "
                        f"{uf_grafo_combinado} "
                        "(Sintomas + Sorotipos + "
                        "Hospitalizações + Óbitos)"
                    )
                )
    # ========================================================
    # 2. FLUXOS DETERMINÍSTICOS ANTIGOS
    # ========================================================

    elif modo_resposta.startswith(
        "deterministico"
    ):

        resumo_fontes = (
            resumir_fontes_deterministicas(
                fontes
            )
        )

        st.json(
            resumo_fontes
        )

        with st.expander(
            "Ver registros considerados"
        ):

            if (
                modo_resposta
                == "deterministico_casos"
            ):

                st.dataframe(
                    fontes_para_dataframe_total_casos(
                        fontes
                    ),
                    use_container_width=True
                )

            elif (
                modo_resposta
                == "deterministico_sorotipos"
            ):

                st.dataframe(
                    fontes_para_dataframe_sorotipos(
                        fontes
                    ),
                    use_container_width=True
                )

            elif (
                modo_resposta
                == "deterministico_sintomas"
            ):

                st.dataframe(
                    fontes_para_dataframe_sintomas(
                        fontes
                    ),
                    use_container_width=True
                )

            elif (
                modo_resposta
                == "deterministico_hospitalizacao"
            ):

                st.dataframe(
                    fontes_para_dataframe_hospitalizacao(
                        fontes
                    ),
                    use_container_width=True
                )


        # ====================================================
        # VISUALIZAÇÃO AUTOMÁTICA — CASOS POR UF
        # ====================================================

        if (
            modo_resposta
            == "deterministico_casos"
        ):

            df_casos_uf = (
                fontes_para_dataframe_total_casos(
                    fontes
                )
            )

            if not df_casos_uf.empty:

                st.subheader(
                    "Distribuição dos casos "
                    "de dengue por UF"
                )

                renderizar_barras_df(
                    df=df_casos_uf,
                    coluna_categoria="UF",
                    coluna_valor="Total de Casos"
                )

            else:

                st.warning(
                    "Não foram encontrados dados "
                    "suficientes para gerar "
                    "o gráfico por UF."
                )

        
        intencoes_consulta = (
            classificacao.get(
                "intencoes",
                {}
            )
            or {}
        )
        pergunta_norm = normalizar_texto(
            pergunta
        )

        eh_consulta_quantitativa_geografica = (
            "distribuicao geografica"
            in pergunta_norm

            or "distribuem geograficamente"
            in pergunta_norm

            or "distribuidos geograficamente"
            in pergunta_norm

            or (
                intencoes_consulta.get(
                    "notificacoes",
                    False
                )
                and intencoes_consulta.get(
                    "geografico",
                    False
                )
                and classificacao.get(
                    "escopo_geografico"
                ) == "nacional"
            )
        )
        # ====================================================
        # GRAFO COMBINADO
        # ====================================================
        if not eh_consulta_quantitativa_geografica:
            st.subheader(
                "Grafo combinado por UF"
            )

            todas_ufs = sorted({
                str(
                    doc.metadata.get(
                        "uf_name",
                        doc.metadata.get(
                            "uf_nome",
                            ""
                        )
                    )
                ).strip()
                for doc in documentos
                if str(
                    doc.metadata.get(
                        "uf_name",
                        doc.metadata.get(
                            "uf_nome",
                            ""
                        )
                    )
                ).strip()
            })

            if todas_ufs:

                col1, col2 = st.columns(
                    [
                        2,
                        1
                    ]
                )

                # ----------------------------------------------------
                # NÍVEL GEOGRÁFICO DO GRAFO COMBINADO
                # ----------------------------------------------------

                opcoes_nivel_grafo = (
                    ["Nacional"]
                    + todas_ufs
                )

                with col1:

                    nivel_grafo_combinado = (
                        st.selectbox(
                            "Nível geográfico do grafo combinado",
                            opcoes_nivel_grafo,
                            index=0,
                            key="nivel_grafo_combinado"
                        )
                    )

                anos_disponiveis = sorted({
                    str(
                        doc.metadata.get(
                            "ano",
                            doc.metadata.get(
                                "nu_ano",
                                doc.metadata.get(
                                    "ano_normalizado",
                                    ""
                                )
                            )
                        )
                    ).strip()
                    for doc in documentos
                    if str(
                        doc.metadata.get(
                            "ano",
                            doc.metadata.get(
                                "nu_ano",
                                doc.metadata.get(
                                    "ano_normalizado",
                                    ""
                                )
                            )
                        )
                    ).strip()
                })

                with col2:

                    ano_grafo = (
                        st.selectbox(
                            "Ano (opcional)",
                            [
                                "Todos"
                            ]
                            + anos_disponiveis,
                            index=0,
                            key="ano_grafo_combinado"
                        )
                    )

                # ----------------------------------------------------
                # ELEMENTOS PADRÃO DO GRAFO CONFORME A CONSULTA
                # ----------------------------------------------------

                if modo_resposta == "deterministico_sorotipos":

                    default_grafo = [
                        "Sorotipos"
                    ]

                elif modo_resposta == "deterministico_sintomas":

                    default_grafo = [
                        "Sintomas"
                    ]

                elif modo_resposta == "deterministico_hospitalizacao":

                    default_grafo = [
                        "Hospitalizações"
                    ]

                elif modo_resposta == "deterministico_casos":

                    default_grafo = [
                        "Casos"
                    ]

                else:

                    default_grafo = [
                        "Sintomas",
                        "Sorotipos",
                        "Hospitalizações",
                        "Óbitos",
                    ]

                # ----------------------------------------------------
                # ELEMENTOS DISPONÍVEIS CONFORME O NÍVEL
                # ----------------------------------------------------

                if nivel_grafo_combinado == "Nacional":

                    opcoes_elementos = [
                        "Sintomas",
                        "Sorotipos",
                        "Hospitalizações",
                        "Óbitos",
                    ]

                else:

                    opcoes_elementos = [
                        "Casos",
                        "Sintomas",
                        "Sorotipos",
                        "Hospitalizações",
                        "Óbitos",
                    ]

                # ----------------------------------------------------
                # Garantir que o default exista nas opções
                # ----------------------------------------------------

                default_grafo_valido = [
                    elemento
                    for elemento in default_grafo
                    if elemento in opcoes_elementos
                ]

                # ----------------------------------------------------
                # MULTISELECT
                # ----------------------------------------------------

                combinacoes_grafo = (
                    st.multiselect(
                        (
                            "Escolha os elementos "
                            "para combinar no grafo"
                        ),
                        opcoes_elementos,
                        default=default_grafo_valido,
                        key="combinacoes_grafo"
                    )
                )

                if st.button(
                    "Gerar grafo combinado"
                ):

                    ano_filtro = (
                        None
                        if ano_grafo == "Todos"
                        else ano_grafo
                    )

                    # ========================================================
                    # NÍVEL NACIONAL
                    # ========================================================

                    if nivel_grafo_combinado == "Nacional":

                        G = (
                            construir_grafo_combinado_nacional(
                                documentos=documentos,
                                ano_filtro=ano_filtro,

                                incluir_sintomas=(
                                    "Sintomas"
                                    in combinacoes_grafo
                                ),

                                incluir_sorotipos=(
                                    "Sorotipos"
                                    in combinacoes_grafo
                                ),

                                incluir_hospitalizacao=(
                                    "Hospitalizações"
                                    in combinacoes_grafo
                                ),

                                incluir_obitos=(
                                    "Óbitos"
                                    in combinacoes_grafo
                                )
                            )
                        )

                    # ========================================================
                    # NÍVEL UF
                    # ========================================================

                    else:

                        G = (
                            construir_grafo_combinado_por_uf(
                                documentos=documentos,

                                uf_focal=
                                    nivel_grafo_combinado,

                                ano_filtro=
                                    ano_filtro,

                                incluir_casos=(
                                    "Casos"
                                    in combinacoes_grafo
                                ),

                                incluir_sintomas=(
                                    "Sintomas"
                                    in combinacoes_grafo
                                ),

                                incluir_sorotipos=(
                                    "Sorotipos"
                                    in combinacoes_grafo
                                ),

                                incluir_hospitalizacao=(
                                    "Hospitalizações"
                                    in combinacoes_grafo
                                ),

                                incluir_obitos=(
                                    "Óbitos"
                                    in combinacoes_grafo
                                )
                            )
                        )

                    # ========================================================
                    # TÍTULO
                    # ========================================================

                    titulo_partes = (
                        " + ".join(
                            combinacoes_grafo
                        )
                        if combinacoes_grafo
                        else "Sem elementos"
                    )

                    renderizar_grafo_pyvis(
                        G,
                        (
                            "Grafo combinado — "
                            f"{nivel_grafo_combinado} "
                            f"({titulo_partes})"
                        )
                    )

            else:

                st.info(
                    "Nenhuma UF disponível "
                    "para o grafo combinado."
                )

    # ========================================================
    # 3. NOVO PIPELINE VSM
    # ========================================================

    elif modo_resposta == "vsm":

        if fontes:

            st.caption(
                f"{len(fontes)} evidência(s) "
                "utilizada(s) na construção "
                "da resposta."
            )

            for i, doc in enumerate(
                fontes,
                start=1
            ):

                parent_id = (
                    doc.metadata.get(
                        "parent_document_id",
                        ""
                    )
                )

                secao = (
                    doc.metadata.get(
                        "secao",
                        ""
                    )
                )

                titulo_expander = (
                    f"Evidência {i}"
                )

                if secao:

                    titulo_expander += (
                        f" — {secao}"
                    )

                with st.expander(
                    titulo_expander
                ):

                    st.write(
                        "**Documento VSM:**",
                        parent_id
                    )

                    st.write(
                        "**Tipo de documento:**",
                        doc.metadata.get(
                            "tipo_documento",
                            ""
                        )
                    )

                    st.write(
                        "**Domínio:**",
                        doc.metadata.get(
                            "dominio",
                            ""
                        )
                    )

                    uf_nome = (
                        doc.metadata.get(
                            "uf_nome",
                            ""
                        )
                    )

                    if uf_nome:

                        st.write(
                            "**UF:**",
                            uf_nome
                        )

                    ano_doc = (
                        doc.metadata.get(
                            "ano",
                            ""
                        )
                    )

                    if ano_doc:

                        st.write(
                            "**Ano:**",
                            ano_doc
                        )

                    if secao:

                        st.write(
                            "**Seção:**",
                            secao
                        )

                    score_heuristico = (
                        doc.metadata.get(
                            "score_heuristico",
                            None
                        )
                    )

                    if (
                        score_heuristico
                        is not None
                    ):

                        st.write(
                            "**Score heurístico:**",
                            score_heuristico
                        )

                    if DEBUG:

                        justificativas = (
                            doc.metadata.get(
                                "justificativa_heuristica",
                                []
                            )
                        )

                        if justificativas:

                            st.write(
                                "**Justificativas "
                                "heurísticas:**"
                            )

                            st.write(
                                justificativas
                            )

                        st.write(
                            "**Metadados completos:**"
                        )

                        st.json(
                            doc.metadata
                        )

                    st.markdown(
                        "**Trecho utilizado:**"
                    )

                    st.write(
                        doc.page_content
                    )

        else:

            st.info(
                "Nenhuma evidência foi utilizada "
                "para esta resposta."
            )
        # ====================================================
        # CONTROLE DO GRAFO GENÉRICO DA RESPOSTA
        # ====================================================

        mostrar_grafo_vsm = True

        # ====================================================
        # VISUALIZAÇÃO CLÍNICA DA VSM PAI
        # ====================================================

        intencoes_consulta = (
            classificacao.get(
                "intencoes",
                {}
            )
            or {}
        )

        escopo_consulta = (
            classificacao.get(
                "escopo_geografico",
                ""
            )
        )

        if (
            intencoes_consulta.get(
                "clinico",
                False
            )
            and escopo_consulta
                == "nacional"
        ):

            ids_pais = []

            for doc in fontes:

                parent_id = (
                    doc.metadata.get(
                        "parent_document_id",
                        ""
                    )
                )

                if (
                    parent_id
                    and parent_id
                    not in ids_pais
                ):
                    ids_pais.append(
                        parent_id
                    )

            for parent_id in ids_pais:

                documento_json = (
                    carregar_json_vsm_por_document_id(
                        parent_id
                    )
                )

                if not documento_json:
                    continue

                if (
                    documento_json.get(
                        "tipo_documento"
                    )
                    !=
                    "panorama_clinico_nacional"
                ):
                    continue

                st.subheader(
                    "Grafo do perfil clínico nacional"
                )

                G = (
                    construir_grafo_clinico_nacional_vsm(
                        documento_json
                    )
                )

                if (
                    G.number_of_nodes()
                    > 0
                ):

                    renderizar_grafo_pyvis(
                        G,
                        (
                            "Perfil clínico "
                            "da dengue no Brasil"
                        )
                    )

                    # Já existe um grafo especializado para esta consulta.
                    # Portanto, não exibir o grafo genérico da resposta atual.
                    mostrar_grafo_vsm = False

                break


        # ====================================================
        # RASTREABILIDADE DO NOVO PIPELINE
        # ====================================================

        if DEBUG:

            with st.expander(
                "Rastreabilidade da consulta"
            ):

                st.markdown(
                    "#### Classificação semântica"
                )

                st.json(
                    classificacao
                )

                st.markdown(
                    "#### Validação de escopo"
                )

                st.json(
                    escopo
                )

                st.markdown(
                    "#### Estatísticas do contexto"
                )

                st.json(
                    estatisticas_contexto
                )

        # ========================================================
        # VISUALIZAÇÃO VSM
        # ========================================================

        #mostrar_grafo_vsm = True

        intencoes_consulta = (
            classificacao.get(
                "intencoes",
                {}
            )
            or {}
        )
        # ========================================================
        # VISUALIZAÇÃO — INCIDÊNCIA MUNICIPAL
        # ========================================================

        if (
            intencoes_consulta.get(
                "incidencia",
                False
            )
        ):

            df_incidencia = (
                dataframe_incidencia_municipal_vsm(
                    documentos=documentos,
                    ano_filtro=str(ANO)
                )
            )

            if not df_incidencia.empty:

                st.subheader(
                    "Municípios com maior incidência "
                    "de dengue"
                )

                renderizar_barras_df(
                    df=df_incidencia,
                    coluna_categoria="Município / UF",
                    coluna_valor="Incidência por 100 mil",
                    top_n=10
                )

                mostrar_grafo_vsm = False

        # ========================================================
        # VISUALIZAÇÃO TEMPORAL
        # ========================================================

        if (
            intencoes_consulta.get(
                "temporal",
                False
            )
            and not intencoes_consulta.get(
                "hospitalizacao",
                False
            )
            and not intencoes_consulta.get(
                "obitos",
                False
            )
            and not intencoes_consulta.get(
                "clinico",
                False
            )
            and not intencoes_consulta.get(
                "sorotipos",
                False
            )
        ):

            df_temporal = (
                dataframe_temporal_vsm(
                    documentos=documentos,
                    ano_filtro=str(ANO)
                )
            )

            if not df_temporal.empty:

                st.subheader(
                    "Evolução dos casos por "
                    "semana epidemiológica"
                )

                renderizar_linha_temporal_df(
                    df=df_temporal,
                    coluna_tempo=
                        "Semana Epidemiológica",
                    coluna_valor="Casos"
                )

                mostrar_grafo_vsm = False
        # ========================================================
        # VISUALIZAÇÃO — DISTRIBUIÇÃO GEOGRÁFICA DOS CASOS
        # ========================================================

        pergunta_norm = normalizar_texto(
            pergunta
        )

        eh_distribuicao_geografica = (
            "distribuicao geografica"
            in pergunta_norm

            or "distribuem geograficamente"
            in pergunta_norm

            or "distribuidos geograficamente"
            in pergunta_norm

            or (
                intencoes_consulta.get(
                    "notificacoes",
                    False
                )
                and intencoes_consulta.get(
                    "geografico",
                    False
                )
                and classificacao.get(
                    "escopo_geografico"
                ) == "nacional"
            )
        )

        if eh_distribuicao_geografica:

            # ----------------------------------------------------
            # Esta consulta é quantitativa/geográfica.
            # O grafo semântico genérico não é apropriado.
            # ----------------------------------------------------

            mostrar_grafo_vsm = False

            df_notificacoes_uf = (
                dataframe_notificacoes_por_uf_vsm(
                    documentos
                )
            )

            if not df_notificacoes_uf.empty:

                st.subheader(
                    "Distribuição dos casos de dengue por UF"
                )

                renderizar_barras_df(
                    df=df_notificacoes_uf,
                    coluna_categoria="UF",
                    coluna_valor="Notificações"
                )

            else:

                st.warning(
                    "Não foram encontrados dados suficientes "
                    "para gerar a distribuição geográfica por UF."
                )
        # ========================================================
        # GRAFO DA RESPOSTA VSM
        # ========================================================
        if mostrar_grafo_vsm:
            st.subheader(
                "Grafo da resposta atual"
            )

            # --------------------------------------------------------
            # Documentos disponíveis para construção do grafo
            # --------------------------------------------------------

            documentos_grafo = fontes

            # Consultas nacionais, como "sorotipos por UF",
            # podem recuperar apenas a VSM nacional.
            # Nesse caso, usamos as VSMs estaduais do mesmo domínio
            # exclusivamente para a visualização.
            if not listar_ufs_unicas(documentos_grafo):

                dominios_fontes = {
                    str(
                        doc.metadata.get(
                            "dominio",
                            ""
                        )
                    ).strip()
                    for doc in fontes
                    if str(
                        doc.metadata.get(
                            "dominio",
                            ""
                        )
                    ).strip()
                }

                documentos_grafo = [
                    doc
                    for doc in documentos
                    if (
                        str(
                            doc.metadata.get(
                                "dominio",
                                ""
                            )
                        ).strip()
                        in dominios_fontes
                        and str(
                            doc.metadata.get(
                                "uf_nome",
                                ""
                            )
                        ).strip()
                    )
                ]

            ufs_disponiveis = (
                listar_ufs_unicas(
                    documentos_grafo
                )
            )

            if ufs_disponiveis:

                uf_focal = st.selectbox(
                    "Escolha a UF para visualizar "
                    "o grafo da resposta atual",
                    ufs_disponiveis,
                    index=0,
                    key="uf_focal_resposta_vsm"
                )
                # # ====================================================
                # # DEBUG TEMPORÁRIO — DOCUMENTO VSM DA UF
                # # ====================================================

                # st.write(
                #     "### DEBUG — documento VSM da UF selecionada"
                # )

                # for doc in documentos_grafo:

                #     uf_doc = str(
                #         doc.metadata.get(
                #             "uf_nome",
                #             ""
                #         )
                #     ).strip()

                #     if uf_doc == uf_focal:

                #         st.write(
                #             "Tipo:",
                #             doc.metadata.get(
                #                 "tipo_documento"
                #             )
                #         )

                #         st.write("Metadados:")
                #         st.json(doc.metadata)

                #         st.write("Conteúdo:")
                #         st.write(doc.page_content)

                #         break
                if st.button(
                    "Gerar grafo da resposta atual",
                    key="gerar_grafo_resposta_vsm"
                ):

                    G = construir_grafo_egocentrico(
                        documentos_grafo,
                        modo_resposta,
                        uf_focal
                    )

                    renderizar_grafo_pyvis(
                        G,
                        (
                            "Grafo egocêntrico de "
                            f"{uf_focal}"
                        )
                    )

            else:

                st.info(
                    "Nenhuma UF disponível para "
                    "o grafo da resposta atual."
                )
    # ========================================================
    # 4. FALLBACK
    # ========================================================

    else:

        st.warning(
            "O modo de resposta retornado não foi "
            "reconhecido pela interface."
        )

        if DEBUG:

            st.write(
                "modo_resposta:",
                modo_resposta
            )

            st.write(
                "Quantidade de fontes:",
                len(fontes)
            )