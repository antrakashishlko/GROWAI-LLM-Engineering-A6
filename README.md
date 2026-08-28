# GROWAI LLM Engineering – Assignment 6

##  Document Q&A System with Hybrid Search

This project demonstrates how a Retrieval-Augmented Generation (RAG) system can retrieve relevant information from a document and use it to generate context-grounded answers with a local LLM.

## Purpose

The purpose of this assignment is to understand how a complete RAG pipeline works, including document chunking, vector embeddings, keyword search, hybrid retrieval, result fusion, reranking and LLM-based answer generation.

## Features

- Loads and processes an Artificial Intelligence document
- Splits the document using RecursiveCharacterTextSplitter
- Uses a chunk size of 300 with 50-character overlap
- Generates embeddings using `all-MiniLM-L6-v2`
- Stores embeddings and document chunks in ChromaDB
- Implements keyword-based retrieval using BM25Okapi
- Combines vector and BM25 search using Reciprocal Rank Fusion (RRF)
- Reranks the top 10 results using a cross-encoder
- Selects the top 3 relevant chunks as context
- Uses the local Qwen3 0.6B model through Ollama
- Generates answers using a context-restricted LangChain prompt
- Tests the system with five different questions

## Requirements

- Python 3.x
- LangChain
- Sentence Transformers
- ChromaDB
- rank-bm25
- LangChain Ollama
- Ollama
- Qwen3 0.6B

Install the required dependencies using:
```text
pip install -r requirements.txt
```

Make sure the Qwen3 0.6B model is available locally:
```text
ollama pull qwen3:0.6b
```

## Installation

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install the dependencies using `requirements.txt`.
4. Make sure Ollama is installed and running.
5. Make sure the `qwen3:0.6b` model is available locally.
6. Keep `document.txt` in the same directory as the Python file.

## How to Run

Run the Python script:
```text
python rag_engine.py
```

The program loads and chunks the document, creates embeddings, builds the BM25 index, performs hybrid retrieval, reranks the retrieved results and generates answers using the local LLM.

## Project Files

- `rag_engine.py` – Main Python program containing the complete RAG pipeline
- `document.txt` – Artificial Intelligence document used as the knowledge source
- `requirements.txt` – Required Python dependencies
- `.gitignore` – Files and folders excluded from Git tracking
- `chroma_db/` – Local ChromaDB storage generated automatically during execution and excluded from Git tracking

## RAG Pipeline

The complete workflow is:

### Document → Chunking → Embeddings → ChromaDB + BM25 → Hybrid Search → RRF → Top 10 → Cross-Encoder → Top 3 → LLM → Answer

## Testing

The system is tested using five different questions.

The test cases include a keyword-focused question about `Moravec's Paradox` to demonstrate BM25 retrieval, a semantic question to test vector retrieval and additional questions related to AI history, ethics and regulation.

## Real-World Relevance

This type of RAG system can be used for document-based AI assistants.

For example, organizations can use similar systems to answer questions from technical documentation, research papers, manuals, policies or internal knowledge bases.

## Edge Case

If a question asks for information that is not present in the document, the retrieved context may not contain the required answer. To handle this, the system prompt instructs the LLM to respond with `Not in context`.

Another possible failure point is when Ollama is not running or the required model is unavailable, which prevents the final answer-generation stage.

## Assignment

GROWAI LLM Engineering & Generative AI – Assignment 6
