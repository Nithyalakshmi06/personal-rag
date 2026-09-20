# 🤖 Personal RAG

A Personal Retrieval-Augmented Generation (RAG) system that
answers questions based on my resume using FAISS, Sentence
Transformers, and Gemini.

## 🚀 Features

- Resume document ingestion
- Text chunking
- Semantic embeddings
- FAISS vector search
- Top-K document retrieval
- Gemini-powered answer generation
- Grounded responses using retrieved context

## 🏗️ Architecture

User Question
↓
Retriever
↓
FAISS Vector Store
↓
Relevant Resume Chunks
↓
Gemini
↓
Generated Answer

## 🛠️ Technologies

- Python
- Sentence Transformers
- FAISS
- Gemini API
- Streamlit
- Pandas

## 📸 Demo

![Uploading image.png…]()


## 📂 Project Structure

```text
personal-rag/
├── embeddings/
├── generation/
├── ingestion/
├── retrieval/
├── vector_store/
├── app.py
├── requirements.txt
└── README.md
