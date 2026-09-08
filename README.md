# Semantic Materialized Views for Epidemiological Surveillance

Este repositório apresenta uma abordagem baseada em **Visões Semânticas Materializadas (VSMs)** (*Semantic Materialized Views*) para apoiar consultas em linguagem natural sobre dados de vigilância epidemiológica.

O projeto foi desenvolvido inicialmente no contexto da trilha de **Demos e Aplicações do SBBD 2026** e evoluiu para incorporar mecanismos de recuperação semântica, geração de respostas fundamentadas em evidências e visualização de informações epidemiológicas por meio de grafos.

A implementação atual concentra-se principalmente em dados de **dengue provenientes do SINAN**, organizados em diferentes dimensões semânticas.

## Visão geral

A abordagem transforma resultados epidemiológicos previamente processados e agregados em documentos semanticamente estruturados, denominados **Visões Semânticas Materializadas (VSMs)**.

Esses documentos funcionam como uma camada intermediária entre os dados epidemiológicos tabulares e os mecanismos de recuperação e geração de respostas em linguagem natural.

A figura abaixo apresenta um exemplo da estrutura utilizada para representar um documento semântico.

<img width="1076" height="681" alt="DocSemanticoCompleto" src="https://github.com/user-attachments/assets/2df558e1-3063-4984-bcc1-0ad0ab6ca8a1" />

## Objetivo

O objetivo principal deste projeto é investigar como **representações semânticas intermediárias** podem facilitar a interação entre dados epidemiológicos estruturados e sistemas baseados em linguagem natural.

Especificamente, busca-se:

* estruturar resultados epidemiológicos na forma de Visões Semânticas Materializadas;
* preservar contexto, domínio, granularidade e proveniência das informações;
* permitir consultas epidemiológicas em linguagem natural;
* recuperar evidências semanticamente relacionadas às perguntas;
* combinar diferentes dimensões epidemiológicas;
* produzir respostas fundamentadas nas evidências recuperadas;
* preservar a rastreabilidade entre respostas e documentos de origem;
* oferecer representações gráficas das relações entre informações epidemiológicas.

## Motivação

Sistemas de vigilância epidemiológica, como o SINAN, disponibilizam dados predominantemente em formato tabular, contendo códigos, variáveis especializadas e diferentes níveis de granularidade.

Embora esse formato seja adequado para processamento estatístico e análise epidemiológica, sua utilização direta por sistemas baseados em linguagem pode dificultar a interpretação do significado das variáveis, do contexto epidemiológico e das relações entre diferentes dimensões dos dados.

A abordagem proposta introduz uma camada semântica intermediária:

```text
Dados epidemiológicos
        ↓
Processamento e agregação
        ↓
Visões Semânticas Materializadas
        ↓
Recuperação de evidências
        ↓
Sistema baseado em linguagem
        ↓
Resposta fundamentada
```

As VSMs representam resultados epidemiológicos previamente calculados em uma estrutura textual acompanhada de metadados semânticos.

## Dimensões semânticas

A versão atual organiza as informações epidemiológicas em cinco dimensões principais:

| Dimensão       | Exemplos de informações                      |
| -------------- | -------------------------------------------- |
| **Clínica**    | sintomas e sinais clínicos                   |
| **Virológica** | sorotipos da dengue                          |
| **Geográfica** | distribuição por Unidade Federativa          |
| **Temporal**   | evolução dos casos por semana epidemiológica |
| **Desfechos**  | hospitalizações e óbitos                     |

Essas dimensões podem ser consultadas individualmente ou combinadas em perguntas multidimensionais.

Por exemplo:

```text
Quais sintomas predominantes e sorotipos aparecem em cada estado?
```

Nesse caso, a consulta envolve simultaneamente as dimensões **clínica** e **virológica**, utilizando a Unidade Federativa como granularidade da análise.

## Visões Semânticas Materializadas

As VSMs são documentos derivados de análises epidemiológicas previamente executadas.

Cada documento combina conteúdo textual e metadados estruturados que podem representar informações como:

* domínio semântico;
* tipo de documento;
* doença;
* fonte dos dados;
* ano;
* período;
* Unidade Federativa;
* conceitos epidemiológicos;
* palavras-chave;
* perguntas de competência;
* identificadores de proveniência.

Os documentos podem ser representados em formatos como **JSON** e **Markdown**.

A materialização dessas informações permite que resultados epidemiológicos sejam calculados previamente e posteriormente recuperados pelo agente sem a necessidade de executar novamente todas as operações sobre os dados tabulares originais.

## Agente Inteligente de Arboviroses

O repositório também contém uma aplicação desenvolvida em **Streamlit** para consulta das VSMs em linguagem natural.

A aplicação permite realizar perguntas epidemiológicas como:

```text
Quais os sorotipos da dengue por UF?

Quais os sintomas predominantes por estado?

Quais estados apresentam mais hospitalizações?

Quais sintomas predominantes e sorotipos aparecem em cada estado?

Como os casos evoluíram ao longo das semanas epidemiológicas?
```

O agente analisa a pergunta para identificar aspectos como intenção, domínio semântico e granularidade da consulta.

## Estratégias de recuperação

Dependendo da natureza da pergunta, diferentes estratégias podem ser utilizadas.

### Recuperação híbrida

Para consultas interpretativas, o sistema combina mecanismos como:

* busca por similaridade vetorial;
* filtragem por metadados;
* identificação do domínio semântico;
* classificação da intenção da pergunta;
* identificação da granularidade;
* pontuação heurística;
* seleção das evidências mais relevantes.

### Consultas determinísticas

Perguntas que exigem cobertura exaustiva, como consultas envolvendo todas as Unidades Federativas, podem utilizar fluxos determinísticos.

Isso evita que informações sejam omitidas devido à limitação do número de documentos recuperados pela busca vetorial.

## Respostas fundamentadas em evidências

Quando um modelo de linguagem é utilizado para geração da resposta, o contexto é construído a partir das VSMs recuperadas.

O objetivo é reduzir respostas não fundamentadas e permitir que o usuário identifique as evidências utilizadas pelo sistema.

A aplicação preserva informações como:

* documento de origem;
* tipo de VSM;
* domínio;
* Unidade Federativa;
* ano;
* seção recuperada;
* metadados;
* trecho utilizado como evidência.

Essa estratégia contribui para a **rastreabilidade e explicabilidade** das respostas.

## Grafos semânticos

Além das respostas textuais, a aplicação permite construir visualizações gráficas das informações recuperadas.

Os grafos podem representar e combinar elementos como:

* casos;
* sintomas;
* sorotipos;
* hospitalizações;
* óbitos.

As visualizações permitem explorar relações entre diferentes dimensões epidemiológicas tanto em nível nacional quanto por Unidade Federativa.

## Arquitetura simplificada

```text
                  Dados epidemiológicos
                          │
                          ▼
              Pré-processamento e validação
                          │
                          ▼
                     Enriquecimento
                          │
                          ▼
                Análises agregadas
                          │
                          ▼
          Visões Semânticas Materializadas
                    │             │
                    ▼             ▼
                  JSON         Markdown
                    │             │
                    └──────┬──────┘
                           ▼
                  Chunking semântico
                           │
                           ▼
                    Banco vetorial
                           │
                           ▼
                Análise da pergunta
                           │
                ┌──────────┴──────────┐
                ▼                     ▼
       Fluxo determinístico    Recuperação híbrida
                │                     │
                └──────────┬──────────┘
                           ▼
                 Evidências recuperadas
                           │
                           ▼
                  Modelo de linguagem
                           │
                           ▼
                 Resposta fundamentada
                           │
                 ┌─────────┴─────────┐
                 ▼                   ▼
          Rastreabilidade      Grafo semântico
```

## Tecnologias

Entre as principais tecnologias utilizadas estão:

* **Python**
* **Streamlit**
* **LangChain**
* **OpenAI**
* **ChromaDB**
* **Pandas**
* **Altair**
* **NetworkX**
* **PyVis**

## Estrutura do repositório

A estrutura do projeto está sendo progressivamente organizada da seguinte forma:

```text
semantic-materialized-view/
│
├── app.py
│
├── legacy/
│   ├── app12.py
│   └── app_v13.py
│
├── notebooks/
├── README.md
├── requirements.txt
├── .env.example
└── .gitignore
```

A pasta `notebooks` contém os notebooks utilizados nas etapas de preparação, análise dos dados e geração dos documentos semanticamente estruturados.

### Versões anteriores

Versões anteriores do protótipo são mantidas no diretório `legacy/`
apenas para fins de rastreabilidade do desenvolvimento.

A versão atual da aplicação é `app.py`.

## Instalação

Clone o repositório:

```bash
git clone https://github.com/renatobdo/semantic-materialized-view.git
```

Entre no diretório:

```bash
cd semantic-materialized-view
```

Crie um ambiente virtual:

```bash
python -m venv .venv
```

### Windows

Ative o ambiente:

```bash
.venv\Scripts\activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

## Configuração

As informações sensíveis, como chaves de API, não devem ser armazenadas diretamente no código-fonte.

Crie um arquivo `.env` local a partir do arquivo `.env.example`.

Por exemplo:

```text
OPENAI_API_KEY=sua_chave_aqui
```

O arquivo `.env` deve permanecer fora do controle de versão.

## Execução

Com o ambiente virtual ativado:

```bash
streamlit run app.py
```

O Streamlit disponibilizará a aplicação localmente no navegador.

## Contexto de pesquisa

Este repositório faz parte de uma pesquisa sobre o uso de **representações semânticas, recuperação de informação e modelos de linguagem para vigilância epidemiológica**.

O trabalho investiga particularmente o papel das Visões Semânticas Materializadas como uma camada de representação entre fontes epidemiológicas estruturadas e sistemas baseados em linguagem.

A arquitetura de pesquisa mais ampla considera ainda a integração progressiva com ontologias, mecanismos de validação semântica, proveniência, explicabilidade e outras fontes de dados epidemiológicos.

## Limitações

Este software constitui um **protótipo de pesquisa**.

A implementação atual está concentrada principalmente em dados de dengue e não deve ser interpretada como ferramenta clínica, sistema diagnóstico ou plataforma oficial de vigilância epidemiológica.

Os resultados produzidos dependem diretamente da qualidade, completude, período e atualização das fontes utilizadas para gerar as VSMs.

## Status do projeto

🚧 **Em desenvolvimento**

O projeto está sendo continuamente ampliado com novos mecanismos de recuperação, integração semântica, visualização, rastreabilidade e avaliação experimental.

## Publicações

O conceito de Visões Semânticas Materializadas utilizado neste repositório foi desenvolvido no contexto de trabalhos de pesquisa relacionados à integração semântica e consulta de dados epidemiológicos.

As referências bibliográficas e informações das publicações relacionadas serão adicionadas progressivamente ao repositório.

## Licença

A licença de distribuição do projeto será definida antes da disponibilização de uma versão estável.
