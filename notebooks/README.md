# Notebooks

Este diretório contém os notebooks utilizados no pipeline de processamento, análise e geração de documentos semânticos a partir de dados de vigilância epidemiológica do SINAN, com enriquecimento por dados de referência do IBGE.

A organização segue uma sequência lógica de execução, desde a preparação das referências e dos dados brutos até a geração das Visões Semânticas Materializadas e do panorama consolidado.

## Ordem de execução

### 00 — Preparação da referência IBGE

**Arquivo:** `00_preparacao_referencia_IBGE.ipynb`

Prepara os dados de referência do IBGE utilizados nas etapas posteriores do pipeline.

Principais atividades:

* tratamento dos dados de municípios e unidades federativas;
* padronização de códigos geográficos;
* preparação de atributos como população, região, latitude e longitude;
* geração dos arquivos de referência utilizados no enriquecimento dos dados epidemiológicos.

---

### 01 — Configuração e obtenção dos dados SINAN

**Arquivo:** `01_configuracao_e_download_de_dados_SINAN.ipynb`

Responsável pela preparação do ambiente e obtenção dos dados epidemiológicos do SINAN.

Principais atividades:

* definição dos diretórios do projeto;
* configuração do ano de análise;
* obtenção dos arquivos de dados;
* conversão e preparação inicial dos dados para processamento.

---

### 02 — Pré-processamento dos dados SINAN

**Arquivo:** `02_pre_processamento_sinan.ipynb`

Executa a limpeza, normalização e transformação inicial dos registros do SINAN.

Principais atividades:

* seleção das variáveis relevantes;
* padronização dos tipos de dados;
* decodificação de variáveis categóricas;
* tratamento de valores ausentes;
* criação de atributos auxiliares;
* geração do conjunto de dados pré-processado.

---

### 03 — Validação dos dados SINAN

**Arquivo:** `03_validacao_sinan.ipynb`

Avalia a qualidade dos dados utilizados pelo pipeline.

Entre as verificações realizadas estão:

* completude;
* validade de domínio;
* consistência entre atributos relacionados;
* identificação de valores ausentes ou inválidos;
* geração de relatórios de qualidade dos dados.

Essa etapa fornece evidências sobre a qualidade dos dados utilizados nas análises subsequentes.

---

### 04 — Enriquecimento dos dados SINAN

**Arquivo:** `04_enriquecimento_SINAN.ipynb`

Integra os registros epidemiológicos do SINAN com os dados geográficos e demográficos preparados a partir do IBGE.

Entre as informações adicionadas estão:

* município;
* unidade federativa;
* região;
* população;
* latitude;
* longitude;
* atributos auxiliares utilizados nas análises epidemiológicas.

Essa etapa também verifica a consistência entre códigos geográficos presentes no SINAN e nas referências externas.

---

## 05 — Análises epidemiológicas

A etapa 05 é dividida em diferentes dimensões analíticas.

### 05 — Dimensão geográfica

**Arquivo:** `05_analises_geograficas_SINAN.ipynb`

Produz análises relacionadas à distribuição espacial dos casos.

Exemplos:

* casos por unidade federativa;
* casos por município;
* distribuição regional;
* indicadores populacionais;
* incidência epidemiológica.

---

### 05.1 — Dimensão virológica

**Arquivo:** `05_1_analises_virologicas_SINAN.ipynb`

Analisa informações relacionadas aos sorotipos da dengue.

Exemplos:

* distribuição dos sorotipos;
* sorotipos por unidade federativa;
* proporção de registros com sorotipo identificado;
* comparação entre regiões geográficas.

---

### 05.2 — Dimensão clínica

**Arquivo:** `05_2_analises_clinicas_SINAN.ipynb`

Analisa características clínicas dos casos registrados.

Entre as variáveis consideradas estão sinais, sintomas e condições clínicas associadas aos registros epidemiológicos.

---

### 05.3 — Dimensão de desfechos

**Arquivo:** `05_3_analises_desfechos_SINAN.ipynb`

Analisa os principais desfechos associados aos casos.

Entre eles:

* hospitalização;
* evolução do caso;
* óbitos;
* outros indicadores relacionados à gravidade e evolução clínica.

---

### 05.4 — Autoctonia e provável fonte de infecção

**Arquivo:** `05_4_autoctonia_fonte_infeccao_SINAN.ipynb`

Analisa informações relacionadas à origem provável da infecção.

Inclui:

* classificação de casos autóctones;
* unidade federativa provável de infecção;
* município provável de infecção;
* distribuição espacial da provável fonte de infecção.

---

### 05.5 — Dimensão temporal

**Arquivo:** `05_5_analises_temporais_SINAN.ipynb`

Analisa a evolução temporal dos registros epidemiológicos.

Entre os indicadores produzidos estão:

* distribuição por semana epidemiológica;
* séries temporais;
* médias semanais;
* identificação de picos;
* variações recentes;
* tendências temporais.

---

### 05.6 — Consolidação analítica

**Arquivo:** `05_6_consolidacao_analitica_SINAN.ipynb`

Consolida os resultados produzidos pelas diferentes dimensões analíticas.

O objetivo é preparar estruturas padronizadas que possam ser utilizadas na geração dos documentos semânticos.

As principais dimensões consideradas são:

* geográfica;
* virológica;
* clínica;
* desfechos;
* temporal;
* autoctonia e provável fonte de infecção.

---

## 06 — Geração dos documentos semânticos

**Arquivo:** `06_documentos_semanticos_SINAN_FINAL.ipynb`

Transforma os resultados analíticos em documentos semânticos estruturados.

Esses documentos representam as **Visões Semânticas Materializadas (VSMs)** utilizadas pelo sistema de recuperação e geração de respostas em linguagem natural.

Os documentos incluem, entre outros elementos:

* identificador;
* tipo de documento;
* domínio semântico;
* fonte dos dados;
* período de referência;
* localização geográfica;
* indicadores epidemiológicos;
* evidências;
* texto descritivo;
* metadados utilizados na recuperação semântica.

Os documentos são gerados em formatos adequados para armazenamento, inspeção e indexação em bases vetoriais.

---

## 07 — Panorama geral

**Arquivo:** `07_panorama_geral_SINAN.ipynb`

Produz uma visão consolidada dos principais resultados epidemiológicos obtidos pelo pipeline.

Essa etapa integra informações provenientes das diferentes dimensões analíticas para fornecer uma visão geral do conjunto de dados analisado.

---

## Fluxo resumido

```text
Dados de referência IBGE
        |
        v
00 - Preparação IBGE
        |
        v
01 - Obtenção dos dados SINAN
        |
        v
02 - Pré-processamento
        |
        v
03 - Validação
        |
        v
04 - Enriquecimento
        |
        v
05 - Análises epidemiológicas
        |
        +-- Geográfica
        +-- Virológica
        +-- Clínica
        +-- Desfechos
        +-- Autoctonia / Fonte de infecção
        +-- Temporal
        |
        v
05.6 - Consolidação analítica
        |
        v
06 - Documentos semânticos / VSMs
        |
        v
07 - Panorama geral
```

## Diretório `legacy`

O diretório `legacy/` contém versões anteriores ou experimentais dos notebooks e scripts mantidos para fins de rastreabilidade e histórico do desenvolvimento.

Esses arquivos não representam necessariamente a versão atual do pipeline.

Atualmente estão armazenados nesse diretório:

* `01-generate-semantic-materialized-view.ipynb`
* `02-simple-example-of-semantic-views.py`
* `06_documentos_semanticos_SINAN_revisado.ipynb`

Para novas execuções e experimentos, devem ser priorizados os notebooks localizados diretamente no diretório `notebooks/`.

## Observação sobre os dados

Os notebooks foram desenvolvidos para processamento de dados epidemiológicos do SINAN e dados auxiliares de referência.

Os dados brutos utilizados nas análises não são necessariamente distribuídos neste repositório. Os caminhos de entrada e saída devem ser configurados conforme o ambiente de execução.

O projeto foi desenvolvido principalmente em ambiente Python/Jupyter/Google Colab.
