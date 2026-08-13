import os
from langchain_community.vectorstores import FAISS 

from hr_assistant import config
from hr_assistant.embedding import get_embeddings_model

def build_vector_store(chunks):
    """Embed every chunk and build a searchable FAISS index In memory."""
    embeddings_model = get_embeddings_model()
    return FAISS.from_documents(chunks, embeddings_model)

## store vector store

def save_vector_store(vector_store, file_path: str = config.VECTOR_STORE_PATH)-> None:
    """Save the FAISS index to disk."""
    vector_store.save_local(file_path)

#load vector store

def load_vector_store(file_path: str = config.VECTOR_STORE_PATH):
    """Load the FAISS index from disk."""
    embeddings_model = get_embeddings_model()
    return FAISS.load_local(file_path, embeddings_model, allow_dangerous_deserialization=True)


def vector_store_exists(file_path: str = config.VECTOR_STORE_PATH) -> bool:
    """Check if the FAISS index file exists on disk."""
    return os.path.exists(os.path.join(file_path, "index.faiss"))

def get_retriever(vector_store, k: int = config.TOP_K_RESULTS):
    """Return a retriever object for the FAISS index."""
    return vector_store.as_retriever(search_kwargs={"k": k})                 