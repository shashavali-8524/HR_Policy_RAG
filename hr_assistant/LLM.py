
from langchain_groq import ChatGroq

from hr_assistant import config


def get_llm():
    """Return a ChatGroq object initialized with the API key and model name."""
    return ChatGroq(model_name=config.LLM_MODEL_NAME, temperature=0)
