# HR Policy RAG Assistant

This repository contains a Retrieval-Augmented Generation (RAG) system designed to act as an intelligent HR assistant. It can answer questions about a company's HR policies by referencing a provided policy document. The system is built using LangChain, with Groq for fast LLM inference and Jina for embeddings.

## Features

- **RAG Pipeline**: Implements a full RAG pipeline to provide context-aware answers from a source document (`hr_policy.txt`).
- **Interactive UI**: A user-friendly web interface built with Streamlit allows for easy interaction with the HR assistant.
- **Command-Line Interface**: A simple CLI script (`main.py`) is included to demonstrate the agent's capabilities.
- **Efficient Vector Storage**: Uses FAISS for in-memory vector search and caches the created index locally to speed up subsequent runs.
- **Modular Architecture**: The core logic is organized into a clean, reusable Python package (`hr_assistant`).
- **Fast Inference**: Leverages the Groq API for near-instantaneous LLM responses.

## How It Works

The application follows a standard RAG architecture:

1.  **Document Loading**: The `hr_policy.txt` file is loaded into memory.
2.  **Text Splitting**: The document is divided into smaller, overlapping chunks to ensure semantic context is maintained.
3.  **Embedding**: Each chunk is converted into a numerical vector representation using Jina embeddings.
4.  **Vector Storage**: The embeddings are stored in a FAISS vector store. The store is saved to disk (`data/faiss_index`) on its first creation to avoid reprocessing.
5.  **Retrieval**: When a user asks a question, the system embeds the query and performs a similarity search in the FAISS index to find the most relevant document chunks.
6.  **Generation**: The retrieved chunks are passed as context, along with the original question, to the Groq LLM, which generates a coherent and accurate answer.

## Setup and Installation

Follow these steps to set up and run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/shashavali-8524/HR_Policy_RAG.git
cd HR_Policy_RAG
```

### 2. Create a Virtual Environment

It is recommended to use a virtual environment to manage dependencies.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies

Install the required Python packages from `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

You will need API keys from Groq and Jina AI.

1.  Create a file named `.env` in the root directory of the project.
2.  Add your API keys to the file as follows:

```
GROQ_API_KEY="YOUR_GROQ_API_KEY"
JINA_API_KEY="YOUR_JINA_API_KEY"
```

## Usage

You can interact with the HR assistant through either the Streamlit web app or the command-line interface.

### Running the Streamlit Web App

This is the recommended way to use the assistant.

```bash
streamlit run app.py
```

Navigate to the local URL provided by Streamlit (usually `http://localhost:8501`) in your web browser to start chatting with the HR assistant.



### Running the Command-Line Demo

To run a quick demonstration in your terminal that asks a few pre-defined questions:

```bash
python main.py
```

## Project Structure

```
├── hr_assistant/         # Core Python package for the RAG pipeline
│   ├── agent.py          # Creates the LangChain agent
│   ├── config.py         # Configuration settings (models, paths, prompts)
│   ├── document_loader.py# Loads the source text document
│   ├── embedding.py      # Initializes the Jina embedding model
│   ├── LLM.py            # Initializes the Groq LLM
│   ├── pipeline.py       # Orchestrates the RAG pipeline setup and queries
│   ├── splitter.py       # Splits documents into chunks
│   ├── tools.py          # Defines the custom search tool for the agent
│   └── vector_store.py   # Manages the FAISS vector store
├── app.py                # Streamlit web application entry point
├── main.py               # Command-line demo script
├── hr_policy.txt         # The source document containing HR policies
├── Rag.ipynb             # Jupyter Notebook for experimentation and development
├── requirements.txt      # Project dependencies
└── README.md             # This file
