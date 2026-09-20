from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration and hyperparameter settings."""
    app_env: str = "development"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    streamlit_port: int = 8501

    # Base directory paths
    base_dir: Path = Path(__file__).resolve().parent.parent
    policy_pdf_path: Path = (
        Path(__file__).resolve().parent.parent / "data" / "policy" / "USGIC-CSCIndividualHealthInsurance_2017-2018.pdf"
        if (Path(__file__).resolve().parent.parent / "data" / "policy" / "USGIC-CSCIndividualHealthInsurance_2017-2018.pdf").exists()
        else Path(__file__).resolve().parent.parent.parent / "policy" / "USGIC-CSCIndividualHealthInsurance_2017-2018.pdf"
    )
    policy_chunks_path: Path = Path(__file__).resolve().parent / "rag" / "policy_chunks.json"

    # Retrieval configuration
    retrieval_top_k: int = 5
    rerank_top_k: int = 3
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # Multi-Agent and LLM parameters
    llm_provider: str = "rule_grounded"
    llm_model: Optional[str] = None
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
