import streamlit as st
import openai
import os
from tempfile import TemporaryDirectory
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.openai import OpenAI as OpenAILLM
from llama_index.embeddings.openai import OpenAIEmbedding, OpenAIEmbeddingModelType
from llama_index.agent.openai import OpenAIAgent
from llama_index.agent.tools.query_engine import QueryEngineTool

st.set_page_config(page_title="🤖 ReAct + RAG PDF Agent", layout="centered")
st.title("🤖 ReAct + RAG Agent for PDFs")

api_key = st.text_input("🔑 Enter your OpenAI API key", type="password")
if not api_key:
    st.warning("Please enter your OpenAI API key to proceed.")
    st.stop()
openai.api_key = api_key

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

uploaded_files = st.file_uploader("📎 Upload one or more PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    with TemporaryDirectory() as tmpdir:
        for file in uploaded_files:
            file_path = os.path.join(tmpdir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.read())

        st.success("✅ PDFs uploaded! Building index...")
        documents = SimpleDirectoryReader(tmpdir).load_data()
        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine()

        # RAG tool for the agent
        rag_tool = QueryEngineTool.from_defaults(
            query_engine=query_engine,
            name="pdf_rag_tool",
            description="Retrieves information from the uploaded PDFs"
        )

        # Agent setup with tool
        agent = OpenAIAgent.from_tools(
            tools=[rag_tool],
            llm=llm,
            verbose=True,
            system_prompt="You are an intelligent agent. Use 'pdf_rag_tool' to answer user questions using the uploaded PDFs."
        )

        st.success("✅ Agent is ready! Ask your question below.")
        query = st.text_input("💬 Ask your question (the agent may use tools to reason and answer):")
        if query:
            with st.spinner("🤔 Thinking, reasoning, and using tools..."):
                response = agent.chat(query)
            st.markdown("### 📘 Agent Answer")
            st.write(response.response if hasattr(response, "response") else str(response))
