
from hr_assistant import config
from hr_assistant.agent import create_hr_agent
from hr_assistant.document_loader import load_document
from hr_assistant.LLM import get_llm
from hr_assistant.splitter import split_into_chunks
from hr_assistant.tools import create_search_tool

from hr_assistant.vector_store import (
    build_vector_store,
    get_retriever,
    load_vector_store,
    save_vector_store,
    vector_store_exists,
)

def build_vector_store_for_document(file_path: str = config.DATA_FILE_PATH):
    """Load the HR policy document, split it into chunks, and build a FAISS vector store."""
    if vector_store_exists():
        print("Found a saved vector store. Loading it from disk...")
        return load_vector_store()

    print("No saved vector store found. Building a new one...")
    document = load_document(file_path)
    chunks = split_into_chunks(document)
    print(f"Loaded '{file_path}' and split it into {len(chunks)} chunks.")

    vector_store = build_vector_store(chunks)
    save_vector_store(vector_store)
    print(f" Vector store built and saved to '{config.VECTOR_STORE_PATH}'.")
    return vector_store

def build_hr_assistant(file_path: str = config.DATA_FILE_PATH):
    """Build the full RAG agent, ready to answer questions."""
    config.check_api_keys()

    vector_store = build_vector_store_for_document(file_path)
    retriever = get_retriever(vector_store)
    search_tool = create_search_tool(retriever)

    llm = get_llm()
    agent = create_hr_agent(llm, [search_tool])
    
    return agent

def ask(agent, question: str) -> str:
    """Ask the HR agent a question and return the answer."""
    response = agent.invoke({"messages": [{"role": "user", "content": question}]})

    if isinstance(response, dict) and "messages" in response:
        last_message = response["messages"][-1]
        if hasattr(last_message, "content"):
            return last_message.content
        return last_message["content"]

    return getattr(response, "content", str(response))