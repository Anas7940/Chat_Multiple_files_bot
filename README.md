# Chat with Documents

A Streamlit app that allows users to upload PDF and PPTX files, extract text from them, and then ask questions based on the content of the uploaded documents. The app uses Google Generative AI and Langchain to generate responses.

## Features

- **Document Upload**: Upload PDF and PPTX files.
- **Text Extraction**: Extract text content from PDF and PPTX files.
- **Document Indexing**: Automatically splits the text into chunks and indexes it using FAISS for efficient retrieval.
- **Question Answering**: Ask questions about the content of the uploaded documents, and receive detailed answers based on the indexed text using Google Generative AI.

## Requirements

Before running the app, make sure you have the following dependencies:

- Python 3.x
- Streamlit
- langchain
- langchain-google-genai
- google-generativeai
- PyPDF2
- python-pptx
- FAISS
- dotenv

You can install the dependencies using pip:

```bash
pip install streamlit langchain langchain-google-genai google-generativeai PyPDF2 python-pptx faiss-cpu dotenv
