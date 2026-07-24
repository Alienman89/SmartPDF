import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(page_title="Local Notes AI Assistant", page_icon="📚")
st.title("📚 Local Notes AI Assistant (Ollama + ChromaDB)")
st.caption("Aplikacja RAG w 100% darmowa i lokalna")


@st.cache_resource
def get_ollama_models():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    llm = ChatOllama(model="llama3.2", temperature=0.2)
    return embeddings, llm


embeddings, llm = get_ollama_models()


@st.cache_resource
def get_vector_store():
    return Chroma(
        collection_name="notes_documents",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )


vector_store = get_vector_store()

with st.sidebar:
    st.header("1. Wgraj dokumenty")
    uploaded_file = st.file_uploader("Wybierz plik PDF z notatkami", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Przetwórz i Zapisz w ChromaDB"):
            with st.spinner("Czytam PDF i zapisuję wektory w ChromaDB..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                loader = PDFPlumberLoader(tmp_path)
                docs = loader.load()

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                splits = text_splitter.split_documents(docs)

                vector_store.add_documents(splits)
                os.remove(tmp_path)

                st.success(f"Sukces! Dodano {len(splits)} fragmentów do bazy ChromaDB!")


st.header("2. Zadaj pytanie do swoich notatek")

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])



def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)



if prompt := st.chat_input("O co chcesz zapytać?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Szukam odpowiedzi w ChromaDB..."):
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})


            prompt_template = ChatPromptTemplate.from_template(
                "Jesteś pomocnym asystentem akademickim. Odpowiadaj na pytania "
                "wyłącznie na podstawie poniższego kontekstu. Jeśli w kontekście nie ma odpowiedzi, "
                "powiedz wprost, że nie znalazłeś informacji w notatkach.\n\n"
                "Kontekst:\n{context}\n\n"
                "Pytanie: {question}"
            )


            retrieved_docs = retriever.invoke(prompt)


            rag_chain = (
                    {"context": lambda x: format_docs(retrieved_docs), "question": RunnablePassthrough()}
                    | prompt_template
                    | llm
                    | StrOutputParser()
            )


            answer = rag_chain.invoke(prompt)
            st.markdown(answer)


            with st.expander("Zobacz fragmenty z pliku PDF"):
                for doc in retrieved_docs:
                    st.write(f"- **Strona {doc.metadata.get('page', 'N/A')}**: {doc.page_content[:150]}...")

    st.session_state.messages.append({"role": "assistant", "content": answer})