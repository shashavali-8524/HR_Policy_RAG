from langchain_community.embeddings import JinaEmbeddings

from hr_assistant import config

def get_embeddings_model():
    """Return a JinaEmbeddings object initialized with the API key and model name."""
    return JinaEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)

