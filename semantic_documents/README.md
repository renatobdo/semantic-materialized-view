# Documentos Semânticos

Este diretório contém as **Visões Semânticas Materializadas (VSMs)** produzidas pelo pipeline de processamento e análise dos dados epidemiológicos do SINAN.

As VSMs transformam resultados analíticos previamente calculados em documentos estruturados e textuais que podem ser utilizados pelos mecanismos de recuperação semântica e pelo agente inteligente.

## Organização

Os documentos estão organizados por fonte de dados, ano e dimensão semântica:

```text
semantic_documents/
└── SINAN/
    └── 2026/
        ├── _inventario/
        ├── clinicas/
        │   ├── panorama_clinico/
        │   └── perfil_clinico_por_uf/
        ├── desfechos/
        │   ├── desfechos_por_uf/
        │   ├── panorama_desfechos/
        │   ├── panorama_obitos/
        │   └── perfil_obitos_por_uf/
        ├── geograficos/
        │   ├── distribuicao_por_uf/
        │   └── panorama_geografico/
        ├── panorama_geral/
        ├── temporais/
        │   ├── comportamento_por_uf/
        │   └── panorama_temporal/
        └── virologicas/
            ├── panorama_sorotipos/
            └── sorotipos_por_uf/
```

## Dimensões semânticas

As VSMs atualmente materializadas representam cinco dimensões principais da vigilância epidemiológica:

* **Geográfica:** distribuição espacial dos registros e indicadores por unidade federativa.
* **Virológica:** distribuição e identificação dos sorotipos da dengue.
* **Clínica:** sinais, sintomas e características clínicas dos casos.
* **Desfechos:** hospitalizações, evolução dos casos e óbitos.
* **Temporal:** comportamento dos registros ao longo das semanas epidemiológicas.

Além dessas dimensões, o diretório `panorama_geral/` contém uma visão consolidada dos principais resultados epidemiológicos.

## Representações dos documentos

O conjunto atual contém **168 documentos semânticos**, armazenados em duas representações correspondentes:

* **JSON:** representação estruturada da VSM, incluindo seus dados e metadados.
* **Markdown:** representação textual da mesma VSM, adequada à inspeção humana e ao processamento baseado em linguagem.

Assim, cada documento semântico possui uma representação estruturada e uma representação textual correspondente.

```text
Visão Semântica Materializada
            |
       +----+----+
       |         |
      JSON    Markdown
```

No conjunto referente ao SINAN 2026 estão disponíveis:

* 168 arquivos JSON;
* 168 arquivos Markdown;
* 7 arquivos CSV auxiliares;
* 343 arquivos no total.

## Inventário

O diretório `_inventario/` contém arquivos auxiliares utilizados para descrever e verificar o conjunto de documentos semânticos gerados.

Esses arquivos permitem inspecionar aspectos como:

* quantidade de documentos;
* distribuição por domínio;
* distribuição por tipo de documento;
* identificação dos documentos produzidos pelo pipeline.

## Geração das VSMs

Os documentos deste diretório são produzidos principalmente pelo notebook:

`notebooks/06_documentos_semanticos_SINAN_FINAL.ipynb`

Esse notebook utiliza os resultados consolidados das etapas analíticas anteriores e os transforma em documentos semânticos padronizados.

A sequência completa do pipeline está documentada em:

`notebooks/README.md`

## Uso no sistema

As VSMs funcionam como uma camada intermediária entre os dados epidemiológicos processados e os mecanismos baseados em linguagem.

De forma simplificada:

```text
Dados epidemiológicos
        |
        v
Processamento e análise
        |
        v
Visões Semânticas Materializadas
        |
        v
Indexação / Recuperação semântica
        |
        v
Agente baseado em LLM
        |
        v
Resposta fundamentada em evidências
```

Essa abordagem permite que resultados epidemiológicos previamente calculados sejam recuperados como unidades de evidência durante a interpretação de consultas em linguagem natural.

## Dados de origem

Os documentos deste diretório são **artefatos derivados** do processamento dos dados epidemiológicos utilizados pelo projeto.

Os microdados brutos do SINAN não são distribuídos neste diretório.

Os documentos aqui disponibilizados representam resultados agregados e materializados pelo pipeline para fins de pesquisa, recuperação semântica, experimentação e reprodutibilidade.
