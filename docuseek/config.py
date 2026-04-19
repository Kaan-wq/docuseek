from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Qdrant
    qdrant_host: str
    qdrant_port: int = 6333
    qdrant_collection_name: str
    qdrant_api_key: str | None = None
    qdrant_cluster_endpoint: str | None = None

    # Hugging Face
    hf_token: str

    # GitHub
    github_token: str

    # Mistral
    mistral_api_key: str
    mistral_model: str = "mistral-small-latest"

    # Langfuse
    langfuse_public_key: str
    langfuse_secret_key: str
    langfuse_host: str = "https://cloud.langfuse.com"

    # Dense Embedding
    dense_embd_model_name: str = "microsoft/harrier-oss-v1-0.6b"
    dense_embd_dim: int = 1024

    # Late Interaction Embedding
    late_interaction_embd_model_name: str = "jinaai/jina-colbert-v2"
    late_interaction_embd_dim: int = 64

    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Retrieval
    retrieval_top_k: int = 15


settings = Settings()
