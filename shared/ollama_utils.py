"""Ollama LLM and Embedding utilities."""

from langchain_ollama import ChatOllama, OllamaEmbeddings
from shared.config import settings


def get_llm(temperature: float = 0.0, **kwargs) -> ChatOllama:
    """Get a ChatOllama LLM instance."""
    return ChatOllama(
        model=settings.LLM_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=temperature,
        keep_alive="30m",
        num_ctx=4096,
        **kwargs,
    )


def get_embeddings() -> OllamaEmbeddings:
    """Get OllamaEmbeddings instance for nomic-embed-text."""
    return OllamaEmbeddings(
        model=settings.EMBEDDING_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
    )


def check_ollama_health() -> bool:
    """Check if Ollama is running and models are available."""
    try:
        import ollama as ollama_client

        client = ollama_client.Client(host=settings.OLLAMA_BASE_URL)
        models = client.list()
        model_names = [m.model for m in models.models] if models.models else []
        llm_ok = any(settings.LLM_MODEL in name for name in model_names)
        embed_ok = any(settings.EMBEDDING_MODEL.split(":")[0] in name for name in model_names)
        return llm_ok and embed_ok
    except Exception:
        return False
