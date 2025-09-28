import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from htmlTemplates import bot_template, user_template, css
import os

# Diretório para o banco de dados vetorial persistente
CHROMA_DB_DIRECTORY = "chroma_db_rpg"


def load_llm():
    """Carrega o modelo Gemini 2.5 Flash via API."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
    )
    return llm


def get_pdf_text(pdf_files):
    text = ""
    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    )
    return text_splitter.split_text(text)


def get_vector_store(text_chunks, embeddings):
    if os.path.exists(CHROMA_DB_DIRECTORY) and os.listdir(CHROMA_DB_DIRECTORY):
        st.info("Carregando banco de dados de regras existente...")
        vector_store = Chroma(
            persist_directory=CHROMA_DB_DIRECTORY, embedding_function=embeddings
        )
    else:
        st.info("Criando um novo banco de dados de regras...")
        vector_store = Chroma.from_texts(
            texts=text_chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DB_DIRECTORY,
        )

    return vector_store


def get_conversation_chain(vector_store, llm):
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 8}),
        memory=memory,
        return_source_documents=True,
    )


def handle_user_input(question, mode):
    if st.session_state.conversation is None:
        st.warning("Por favor, processe um PDF antes de fazer perguntas.")
        return

    final_question = question
    if mode == "Consultor de Regras":
        final_question = f"Baseado estritamente nas regras fornecidas no contexto, responda a seguinte pergunta: {question}. Cite a regra específica se possível."
    elif mode == "Mestre Guia":
        final_question = f"Use as regras fornecidas no contexto como inspiração para gerar uma ideia criativa e interessante para um mestre de RPG. Seja descritivo. Pergunta original do usuário: {question}"

    response = st.session_state.conversation.invoke({"question": final_question})
    st.session_state.chat_history = response["chat_history"]

    with st.expander("Fontes Consultadas"):
        st.write(response["source_documents"])

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


def main():
    load_dotenv()  # Carrega as variáveis do .env (incluindo a GOOGLE_API_KEY)
    st.set_page_config(page_title="Mestre de RPG com IA", page_icon=":dragon_face:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # Carrega o modelo de embedding localmente (só uma vez)
    if "embeddings" not in st.session_state:
        with st.spinner("Carregando modelo de embeddings (pode levar um tempo)..."):
            st.session_state.embeddings = HuggingFaceEmbeddings(
                model_name="hkunlp/instructor-xl",
                model_kwargs={"device": "cpu"},
            )

    # Carrega o LLM via API (só uma vez)
    if "llm" not in st.session_state:
        with st.spinner("Conectando ao Mestre... (Google Gemini)"):
            st.session_state.llm = load_llm()

    st.header("Mestre de RPG com IA :dragon_face:")

    # Cria a variável 'mode' a partir da seleção do usuário
    mode = st.radio(
        "Escolha o modo do assistente:",
        ("Consultor de Regras", "Mestre Guia"),
        key="mode_selection",
    )

    question = st.text_input("Pergunte sobre as regras ou peça uma ideia ao Mestre:")
    if question:
        # Passa 'question' E 'mode' para a função
        handle_user_input(question, mode)

    with st.sidebar:
        st.subheader("Seus Livros de Regras")
        pdf_files = st.file_uploader(
            "Carregue seus PDFs e clique em 'Processar'", accept_multiple_files=True
        )

        if st.button("Processar"):
            if pdf_files:
                with st.spinner("Analisando os tomos antigos... (Processando PDFs)"):
                    raw_text = get_pdf_text(pdf_files)
                    text_chunks = get_text_chunks(raw_text)
                    vector_store = get_vector_store(
                        text_chunks, st.session_state.embeddings
                    )
                    st.session_state.conversation = get_conversation_chain(
                        vector_store, st.session_state.llm
                    )
                    st.success("Pronto! O conhecimento foi absorvido. Pode perguntar.")
            else:
                st.warning("Você precisa carregar pelo menos um PDF.")


if __name__ == "__main__":
    main()
