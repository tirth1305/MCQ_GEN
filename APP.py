import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Initialize Azure OpenAI components
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("EMBEDDING_AZURE_OPENAI_DEPLOYMENT"),
    openai_api_version=os.getenv("EMBEDDING_AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("EMBEDDING_AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("EMBEDDING_AZURE_OPENAI_API_KEY")
)

llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0.7,
    max_tokens=500
)

def process_pdf(pdf_file):
    """Process PDF file and create FAISS vector store"""
    # Save uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(pdf_file.getvalue())
    
    # Load PDF
    loader = PyPDFLoader("temp.pdf")
    pages = loader.load()
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(pages)
    
    # Create FAISS vector store
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Clean up temporary file
    os.remove("temp.pdf")
    
    return vector_store

def generate_mcqs(vector_store):
    """Generate MCQs using the vector store"""
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True
    )
    
    mcq_prompt = """Generate 5 multiple choice questions based on the following context. 
    For each question, provide 4 options (A, B, C, D) and indicate the correct answer.
    Format each question as:
    Q1: [Question]
    A) [Option A]
    B) [Option B]
    C) [Option C]
    D) [Option D]
    Answer: [Correct option letter]
    
    Context: {context}"""
    
    prompt = PromptTemplate(
        template=mcq_prompt,
        input_variables=["context"]
    )
    
    response = qa_chain({"query": "Generate MCQs"})
    return response["result"]

def generate_short_answers(vector_store):
    """Generate short answer questions using the vector store"""
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True
    )
    
    sa_prompt = """Generate 5 short answer questions based on the following context.
    For each question, provide a brief answer.
    Format each question as:
    Q1: [Question]
    Answer: [Brief answer]
    
    Context: {context}"""
    
    prompt = PromptTemplate(
        template=sa_prompt,
        input_variables=["context"]
    )
    
    response = qa_chain({"query": "Generate short answer questions"})
    return response["result"]

# Streamlit UI
st.title("RAG-based MCQ Generator")
st.write("Upload a GenAI-related PDF to generate questions")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        vector_store = process_pdf(uploaded_file)
    
    # Generate MCQs
    with st.spinner("Generating MCQs..."):
        mcqs = generate_mcqs(vector_store)
        st.subheader("Multiple Choice Questions")
        st.write(mcqs)
    
    # Generate Short Answer Questions
    with st.spinner("Generating Short Answer Questions..."):
        short_answers = generate_short_answers(vector_store)
        st.subheader("Short Answer Questions")
        st.write(short_answers)