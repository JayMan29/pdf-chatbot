import streamlit as st
from openai import OpenAI
import PyPDF2
import os
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import pickle

# Setup Groq API via OpenAI wrapper (for chat completions)
client = OpenAI(
    api_key=st.secrets["groq"]["api_key"], 
    base_url="https://api.groq.com/openai/v1"
)

st.title("📚 PDF Chatbot with Step-by-Step Agent (Free HuggingFace Embeddings + Groq RAG)")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

Path("pdfs").mkdir(exist_ok=True)
Path("indexes").mkdir(exist_ok=True)

vectorstore = None

if uploaded_file:
    filename = uploaded_file.name
    file_path = f"pdfs/{filename}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"✅ Saved PDF to {file_path}")

    index_path = f"indexes/{filename}.faiss"
    if os.path.exists(index_path):
        st.info("🔁 Reusing saved vector index...")
        with open(index_path, "rb") as f:
            vectorstore = pickle.load(f)
    else:
        reader = PyPDF2.PdfReader(file_path)
        full_text = "".join(page.extract_text() or "" for page in reader.pages)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.create_documents([full_text])
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)
        with open(index_path, "wb") as f:
            pickle.dump(vectorstore, f)
        st.success("✅ Vector index created and saved locally!")

if uploaded_file and vectorstore:
    question = st.text_input("Ask a question about the PDF:")
    if question:
        with st.spinner("Thinking..."):
            docs = vectorstore.similarity_search(question, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])
            agent_prompt = f"""
You're a reasoning assistant. Given a document and a question, think through the answer step by step.

Context from PDF:
{context}

Question: {question}

Your task:
1. Break down the problem into steps
2. Think through each step clearly
3. Provide a final answer at the end

Respond in this format:
Step 1: ...
Step 2: ...
...
✅ Final Answer: ...
"""
            response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": agent_prompt}]
            )
            st.markdown("**Answer:**")
            st.write(response.choices[0].message.content)
