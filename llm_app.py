import streamlit as st
from llama_index.core import VectorStoreIndex, Settings, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI as OpenAILLM
from llama_index.embeddings.openai import OpenAIEmbedding, OpenAIEmbeddingModelType
from tempfile import TemporaryDirectory
import openai
import os

st.set_page_config(page_title="📄 RAG PDF Q&A", layout="centered")
st.title("📄 RAG PDF Q&A App")

# Ask user for OpenAI API key
api_key = st.text_input("🔑 Enter your OpenAI API key", type="password")
if not api_key:
    st.warning("Please enter your OpenAI API key to proceed.")
    st.stop()

# Set key globally
openai.api_key = api_key

# Initialize OpenAI LLM and embeddings
@st.cache_resource
def init_openai_models(api_key):
    embed_model = OpenAIEmbedding(model=OpenAIEmbeddingModelType.TEXT_EMBED_ADA_002)
    llm = OpenAILLM(model="gpt-3.5-turbo")
    Settings.embed_model = embed_model
    Settings.llm = llm
    Settings.chunk_size = 256
    Settings.chunk_overlap = 0
    return embed_model, llm

embed_model, llm = init_openai_models(api_key)

# Upload PDFs
uploaded_files = st.file_uploader("Upload one or more PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    with TemporaryDirectory() as tmpdir:
        for file in uploaded_files:
            file_path = os.path.join(tmpdir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.read())

        st.success("✅ PDFs uploaded! Building the index...")

        # Read and index documents
        documents = SimpleDirectoryReader(tmpdir).load_data()
        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine()

        st.success("✅ Index is ready! Ask questions below.")

        # Query interface
        query = st.text_input("💬 Ask a question about the PDFs:")
        if query:
            with st.spinner("Thinking..."):
                response = query_engine.query(query)
            st.markdown("### 📘 Answer")
            st.write(response.response if hasattr(response, "response") else str(response))

