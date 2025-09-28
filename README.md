# Assistente de regras e narrativas para RPG de mesa

Implementação de um assistente de regras e narrativas para RPG de mesa, utilizando por IA Generativa e RAG.

---

### Índice
- [Sobre](#sobre)
- [Funcionalidades](#funcionalidades)
- [Arquitetura e Tecnologias](#arquitetura-e-tecnologias)
- [Como Usar a Aplicação](#como-usar-a-aplicação)
- [Começando: Guia de Instalação Local](#começando-guia-de-instalação-local)
- [Autores](#autores)

---

## Sobre
Este projeto implementa um assistente de IA que utiliza a arquitetura **RAG (Retrieval-Augmented Generation)**. Ao carregar um livro de regras em PDF, o sistema se torna um "especialista" naquele conteúdo, fornecendo respostas rápidas, precisas e contextualizadas, sem o risco de "alucinações" ou regras inventadas.

> **Exemplo Prático:** Em vez de parar o jogo para pesquisar, um mestre pode simplesmente perguntar:
> *"Um ladino pode usar o ataque furtivo com um machado de batalha que encontrou?"*
> E obter uma resposta instantânea e fundamentada nas regras oficiais.

## Funcionalidades

*   **Upload de Documentos:** Permite carregar um ou mais PDFs com os livros de regras para criar uma base de conhecimento personalizada.
*   **Consulta em Linguagem Natural:** Permite usar o chat para interagir com libguagem natural.
*   **Modos de Operação:**
    *   **Consultor de Regras:** Fornece respostas diretas, literais e baseadas estritamente no texto do livro. Ideal para tirar dúvidas rápidas.
    *   **Mestre Guia:** Usa as regras como inspiração para gerar ideias criativas e descritivas, como sugestões de encontros, descrições de monstros ou ganchos de aventura.
*   **Respostas Rápidas:** Mantém a dinâmica do jogo com respostas geradas em segundos, sempre baseadas no material fornecido.
*   **Privacidade:** Os documentos são processados localmente e não são compartilhados ou usados para treinar outros modelos.

## Arquitetura e Tecnologias

O projeto utiliza um pipeline RAG:

*   **Interface:** `Streamlit`
*   **Processamento de PDF:** `PyPDF2`
*   **Orquestração de IA:** `LangChain`
*   **Modelo de Linguagem (LLM):** `Google Gemini 2.5 Flash`
*   **Modelo de Embeddings:** `HuggingFace instructor-xl` (executado localmente)
*   **Banco de Dados Vetorial:** `ChromaDB` (persistente localmente)

# Fluxo
Os PDFs são processados, divididos em blocos de texto ("chunks"), e transformados em vetores (embeddings) que são armazenados no ChromaDB. Quando uma pergunta é feita, o sistema busca os chunks mais relevantes e os envia ao Gemini junto com a pergunta original para gerar uma resposta contextualizada.

## Como Usar a Aplicação

1.  **Carregue os Documentos/Livros:** Na barra lateral, faça o upload de um ou mais arquivos PDF(Arquivo de no máximo 200mb) contendo as regras do seu sistema de RPG.
2.  **Processe os Arquivos:** Clique no botão "Processar". O sistema irá ler os documentos e criar o banco de dados vetorial. Este passo só precisa ser feito uma vez (a menos que você queira adicionar novos livros).
3.  **Escolha o Modo:** Selecione "Consultor de Regras" para perguntas diretas ou "Mestre Guia" para ideias criativas.
4.  **Faça sua Pergunta:** Digite sua dúvida na caixa de texto e pressione Enter.

---

## Guia de Instalação Local

Passos para clonar e executar o projeto em uma máquina local.

### Pré-requisitos

*   Python 3.8 ou superior
*   Git

### 1. Clone o Repositório

```bash
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio
```

### 2. Crie um Ambiente Virtual(Venv) e Instale as Dependências

Para isolar as dependências do projeto.

```bash
# Crie o ambiente virtual
python -m venv venv

# Ative o ambiente (Windows)
.\venv\Scripts\activate

# Ative o ambiente (macOS/Linux)
source venv/bin/activate
```

Instale todas as dependências:
```bash
pip install -r requirements.txt
```
**Nota:** A primeira execução irá baixar o modelo de embeddings `instructor-xl` (cerca de 5GB), o que pode levar algum tempo.

### 3. Configurar Chave de API

O projeto utiliza a API do Google Gemini.

1.  Crie um arquivo chamado `.env` na raiz do projeto.
2.  Adicione sua chave de API ao arquivo da seguinte forma:

```env
GOOGLE_API_KEY="SUA_CHAVE_API_AQUI"
```

### 4. Execute a Aplicação

Com tudo configurado, inicie o servidor do Streamlit:

```bash
streamlit run app.py
```

Seu navegador deve abrir automaticamente com a aplicação em execução.

## Autores

*   **Cassiel Stavros**
*   **Juan Paes**
*   **Victor Albarado**