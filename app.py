import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from htmlTemplates import bot_template, user_template, css
import os
import logging

# Impressão de logs no terminal
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Diretório do banco de dados vetorial
CHROMA_DB_DIRECTORY = "chroma_db_rpg"


def load_llm():
    """Carrega o modelo Gemini via API."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.5,
    )
    return llm


def get_pdf_text(pdf_files):
    logging.info(f"Iniciando extração de texto de {len(pdf_files)} arquivos PDF.")
    text = ""
    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    logging.info(f"Extração de texto concluída. Total de caracteres: {len(text)}")
    return text


def get_text_chunks(text):
    logging.info("Iniciando divisão do texto em chunks.")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text)
    logging.info(f"Divisão concluída. Número de chunks: {len(chunks)}")
    return chunks


def get_vector_store(text_chunks, embeddings):
    # Verifica se o diretório existe e se não está vazio
    if os.path.exists(CHROMA_DB_DIRECTORY) and os.listdir(CHROMA_DB_DIRECTORY):
        logging.info("Carregando banco de dados vetorial existente do disco.")
        vector_store = Chroma(
            persist_directory=CHROMA_DB_DIRECTORY, embedding_function=embeddings
        )
    else:
        logging.info("Criando um novo banco de dados vetorial.")
        vector_store = Chroma.from_texts(
            texts=text_chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DB_DIRECTORY,
        )
    logging.info("Banco de dados vetorial pronto.")
    return vector_store


def get_conversation_chain(vector_store, llm):
    logging.info("Criando a cadeia de conversação.")
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(
            search_kwargs={"k": 5}
        ),  # Aumenta o número de docs recuperados
        memory=memory,
        return_source_documents=True,
    )
    logging.info("Cadeia de conversação criada com sucesso.")
    return conversation_chain


def handle_user_input(question, mode):
    # Verifica se a conversa foi iniciada
    if st.session_state.conversation is None:
        st.warning("Por favor, processe um PDF antes de fazer perguntas.")
        return

    # Ajusta a temperatura do modelo com base no modo
    if mode == "Consultor de Regras":
        st.session_state.llm.temperature = 0.2  # Mais determinístico
        final_question = f"Baseado estritamente nas regras fornecidas no contexto, responda a seguinte pergunta: {question}. Cite a regra específica se possível."
    elif mode == "Mestre Guia":
        st.session_state.llm.temperature = 0.8  # Mais criativo
        final_question = f"Use as regras fornecidas no contexto como inspiração para gerar uma ideia criativa e interessante para um mestre de RPG. Seja descritivo. Pergunta original do usuário: {question}"

    logging.info(f"Processando pergunta no modo '{mode}': {question}")
    response = st.session_state.conversation.invoke({"question": final_question})
    st.session_state.chat_history = response["chat_history"]

    # Exibição do chat
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(
                user_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True,
            )
        else:
            st.write(
                bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True
            )

    # Exibição dos documentos fonte
    with st.expander("Fontes Consultadas no Livro de Regras"):
        for doc in response["source_documents"]:
            st.info(f"Fonte (Página aproximada): {doc.metadata.get('page', 'N/A')}")
            st.text(f"...{doc.page_content}...")
    logging.info("Resposta e fontes exibidas para o usuário.")


def main():
    load_dotenv()
    st.set_page_config(page_title="Mestre de RPG com IA", page_icon=":dragon_face:")
    st.write(css, unsafe_allow_html=True)

    # Session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    # Carrega modelos apenas uma vez
    if "embeddings" not in st.session_state:
        with st.spinner("Carregando modelo de embeddings..."):
            st.session_state.embeddings = HuggingFaceEmbeddings(
                model_name="hkunlp/instructor-large",
                model_kwargs={"device": "cpu"},
            )
    if "llm" not in st.session_state:
        with st.spinner("Conectando ao Mestre... (Google Gemini)"):
            st.session_state.llm = load_llm()

    st.header("Mestre de RPG com IA :dragon_face:")

    # Formulario do Chat
    with st.form("chat_form", clear_on_submit=True):
        mode = st.radio(
            "Escolha o modo do assistente:",
            ("Consultor de Regras", "Mestre Guia"),
            key="mode_selection",
        )
        question = st.text_input(
            "Pergunte sobre as regras ou peça uma ideia ao Mestre:", key="user_input"
        )

        submitted = st.form_submit_button("Enviar")

        if submitted and question:
            handle_user_input(question, mode)

    with st.sidebar:
        st.subheader("Seus Livros de Regras")
        pdf_files = st.file_uploader(
            "Carregue seus PDFs e clique em 'Processar'", accept_multiple_files=True
        )

        if st.button("Processar"):
            if pdf_files:
                with st.spinner("Analisando os tomos antigos..."):
                    raw_text = get_pdf_text(pdf_files)
                    text_chunks = get_text_chunks(raw_text)
                    st.session_state.vector_store = get_vector_store(
                        text_chunks, st.session_state.embeddings
                    )
                    st.session_state.conversation = get_conversation_chain(
                        st.session_state.vector_store, st.session_state.llm
                    )
                    st.success("O conhecimento foi absorvido.")
            else:
                st.warning("Você precisa carregar pelo menos um PDF.")


if __name__ == "__main__":
    main()
