import streamlit as st
import os
import time
import tempfile
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API Key from .env
groq_api_key = os.getenv('GROQ_API_KEY')

# Page configuration
st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("PDF Chatbot")

# Initialize session state variables
if 'processed_pdfs' not in st.session_state:
    st.session_state.processed_pdfs = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None

# Cache embeddings model
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

# Function to process PDFs and create vector store
def process_pdfs(pdf_files):
    with st.spinner("Processing your PDFs... This may take a minute depending on file size."):
        try:
            embeddings = get_embeddings()
            all_docs = []

            for pdf_file in pdf_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                    temp_file.write(pdf_file.read())
                    temp_path = temp_file.name

                loader = PyPDFLoader(temp_path)
                docs = loader.load()

                for doc in docs:
                    doc.metadata['source'] = pdf_file.name

                all_docs.extend(docs)
                os.unlink(temp_path)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            splits = text_splitter.split_documents(all_docs)

            vector_store = FAISS.from_documents(splits, embeddings)
            retriever = vector_store.as_retriever(search_kwargs={"k": 5})

            return vector_store, retriever

        except Exception as e:
            st.error(f"Error processing PDFs: {str(e)}")
            return None, None

# Initialize LLM
@st.cache_resource
def get_llm():
    return ChatGroq(
        api_key=groq_api_key,
        model_name="llama-3.1-8b-instant"
    )

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def create_rag_chain(retriever):
    llm = get_llm()

    prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant that answers questions based on the provided PDF documents.

    Context: {context}

    Question: {question}

    Answer ONLY based on the provided context. 
    If the answer is not available, reply: 
    "I don't have enough information to answer this question based on the provided documents."
    """)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever

with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=['pdf'])

    if uploaded_files and st.button("Process Documents"):
        vector_store, retriever = process_pdfs(uploaded_files)
        if vector_store:
            st.session_state.vector_store = vector_store
            st.session_state.retriever = retriever
            st.session_state.processed_pdfs = True
            st.success(f"✅ Successfully processed {len(uploaded_files)} PDF files")
        else:
            st.error("Failed to process documents")

    if st.session_state.processed_pdfs:
        st.sidebar.success("Documents processed and ready ✅")
    else:
        st.sidebar.info("Please upload and process documents first ❗")

    if st.session_state.chat_history and st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

st.markdown("### Ask questions about your documents")

for message in st.session_state.chat_history:
    st.chat_message(message["role"]).write(message["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not st.session_state.processed_pdfs:
        response = "Please upload and process documents first."
        st.chat_message("assistant").write(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
    else:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            try:
                rag_chain, retriever = create_rag_chain(st.session_state.retriever)
                start_time = time.time()
                response_text = rag_chain.invoke(prompt)
                elapsed_time = time.time() - start_time

                message_placeholder.write(response_text)
                st.caption(f"Response time: {elapsed_time:.2f} seconds")

                st.session_state.chat_history.append({"role": "assistant", "content": response_text})

                relevant_docs = retriever.invoke(prompt)
                with st.expander("Sources"):
                    for i, doc in enumerate(relevant_docs):
                        st.markdown(f"**Source {i+1}: {doc.metadata.get('source', 'Unknown')}**")
                        st.markdown(doc.page_content[:300] + "...")
                        st.markdown("---")

            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                message_placeholder.error(error_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

if not st.session_state.processed_pdfs and not st.session_state.chat_history:
    st.info("👈 Upload PDFs from the sidebar to get started")
