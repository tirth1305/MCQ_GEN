# RAG-based MCQ Generator

This application uses RAG (Retrieval-Augmented Generation) to generate multiple-choice questions and short answer questions from PDF documents. It leverages LangChain, FAISS, Azure OpenAI, and Streamlit to create an interactive question generation system.

## Features

- Upload PDF documents
- Extract and process content using LangChain
- Store document chunks in FAISS vector store
- Generate 5 MCQs with options and answers
- Generate 5 Short Answer Questions
- Interactive Streamlit interface

## Setup

1. Create a `.env` file in the project root with your Azure OpenAI credentials:
```
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_API_VERSION=your_api_version
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit application:
```bash
streamlit run APP.py
```

## Usage

1. Open the application in your web browser
2. Upload a PDF document using the file uploader
3. Wait for the processing to complete
4. View the generated MCQs and Short Answer Questions

## Requirements

- Python 3.8+
- Azure OpenAI API access
- Internet connection for API calls 