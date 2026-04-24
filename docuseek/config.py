from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "docuseek"
    qdrant_api_key: str | None = None
    qdrant_cluster_endpoint: str | None = None

    # Hugging Face
    hf_token: str

    # GitHub
    github_token: str

    # Mistral
    mistral_api_key: str
    mistral_model: str = "mistral-small-latest"

    # Gemini
    gemini_api_key: str
    gemini_model: str = "gemini-2.5-flash"

    # Langfuse
    langfuse_public_key: str
    langfuse_secret_key: str
    langfuse_host: str = "https://cloud.langfuse.com"

    # Dense Embedding
    dense_embd_model_name: str = "microsoft/harrier-oss-v1-270m"
    dense_embd_dim: int = 640

    # Late Interaction Embedding
    late_interaction_embd_model_name: str = "jinaai/jina-colbert-v2"
    late_interaction_embd_dim: int = 64

    # Cross-Encoder Reranker
    cross_encoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"

    # ColBERT Reranker
    colbert_model_name: str = "jinaai/jina-colbert-v2"

    # GLiNER
    gliner_model_name: str = "urchade/gliner_medium-v2.1"

    # HyDE + MultiQuery
    query_model_name: str = "microsoft/Phi-4-mini-instruct"

    # Retrieval
    retrieval_top_k: int = 15


settings = Settings()
