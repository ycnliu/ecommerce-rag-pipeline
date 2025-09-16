"""
Configuration management for the e-commerce RAG pipeline.
"""
import os
from typing import Optional, List
from pydantic import BaseSettings, Field
from pathlib import Path


class Config(BaseSettings):
    """Application configuration using Pydantic settings."""

    # Application settings
    app_name: str = Field(default="E-commerce RAG Pipeline", env="APP_NAME")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # API settings
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=1, env="API_WORKERS")

    # CLIP Embedding settings
    clip_model_name: str = Field(default="openai/clip-vit-base-patch32", env="CLIP_MODEL_NAME")
    embedding_dimension: int = Field(default=512, env="EMBEDDING_DIMENSION")
    device: Optional[str] = Field(default=None, env="DEVICE")  # auto-detect if None
    model_cache_dir: Optional[str] = Field(default=None, env="MODEL_CACHE_DIR")

    # FAISS Vector Database settings
    faiss_index_type: str = Field(default="flat", env="FAISS_INDEX_TYPE")
    faiss_metric: str = Field(default="l2", env="FAISS_METRIC")
    faiss_nlist: int = Field(default=100, env="FAISS_NLIST")
    faiss_nprobe: int = Field(default=10, env="FAISS_NPROBE")
    faiss_index_path: Optional[str] = Field(default=None, env="FAISS_INDEX_PATH")
    faiss_metadata_path: Optional[str] = Field(default=None, env="FAISS_METADATA_PATH")

    # LLM settings
    llm_provider: str = Field(default="huggingface", env="LLM_PROVIDER")
    llm_model_name: str = Field(default="mistralai/Mixtral-8x7B-Instruct-v0.1", env="LLM_MODEL_NAME")
    llm_api_token: Optional[str] = Field(default=None, env="LLM_API_TOKEN")
    llm_base_url: Optional[str] = Field(default=None, env="LLM_BASE_URL")
    llm_max_tokens: int = Field(default=300, env="LLM_MAX_TOKENS")
    llm_temperature: float = Field(default=0.1, env="LLM_TEMPERATURE")

    # Data settings
    data_csv_path: Optional[str] = Field(default=None, env="DATA_CSV_PATH")
    embeddings_cache_path: Optional[str] = Field(default=None, env="EMBEDDINGS_CACHE_PATH")

    # Processing settings
    batch_size: int = Field(default=32, env="BATCH_SIZE")
    max_prompt_chars: int = Field(default=6000, env="MAX_PROMPT_CHARS")
    search_k_default: int = Field(default=5, env="SEARCH_K_DEFAULT")
    search_k_max: int = Field(default=50, env="SEARCH_K_MAX")

    # Security settings
    allowed_origins: List[str] = Field(default=["*"], env="ALLOWED_ORIGINS")
    max_file_size_mb: int = Field(default=10, env="MAX_FILE_SIZE_MB")

    # Logging settings
    log_file_path: Optional[str] = Field(default=None, env="LOG_FILE_PATH")
    log_rotation: str = Field(default="10 MB", env="LOG_ROTATION")
    log_retention: str = Field(default="1 week", env="LOG_RETENTION")

    class Config:
        """Pydantic config."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def __init__(self, **kwargs):
        """Initialize configuration with validation."""
        super().__init__(**kwargs)
        self._validate_paths()
        self._set_defaults()

    def _validate_paths(self) -> None:
        """Validate and create necessary paths."""
        # Create model cache directory if specified
        if self.model_cache_dir:
            Path(self.model_cache_dir).mkdir(parents=True, exist_ok=True)

        # Validate data file exists if specified
        if self.data_csv_path and not os.path.exists(self.data_csv_path):
            raise ValueError(f"Data CSV file not found: {self.data_csv_path}")

        # Create directories for index files if specified
        if self.faiss_index_path:
            Path(self.faiss_index_path).parent.mkdir(parents=True, exist_ok=True)

        if self.faiss_metadata_path:
            Path(self.faiss_metadata_path).parent.mkdir(parents=True, exist_ok=True)

        # Create log directory if specified
        if self.log_file_path:
            Path(self.log_file_path).parent.mkdir(parents=True, exist_ok=True)

    def _set_defaults(self) -> None:
        """Set intelligent defaults based on other settings."""
        # Set default paths relative to project root
        project_root = Path(__file__).parent.parent.parent

        if not self.model_cache_dir:
            self.model_cache_dir = str(project_root / "models" / "cache")
            Path(self.model_cache_dir).mkdir(parents=True, exist_ok=True)

        if not self.faiss_index_path:
            self.faiss_index_path = str(project_root / "models" / "product_index.faiss")

        if not self.faiss_metadata_path:
            self.faiss_metadata_path = str(project_root / "models" / "product_metadata.pkl")

        if not self.embeddings_cache_path:
            self.embeddings_cache_path = str(project_root / "models" / "embeddings_cache.npy")

        if not self.log_file_path:
            log_dir = project_root / "logs"
            log_dir.mkdir(exist_ok=True)
            self.log_file_path = str(log_dir / "app.log")

        # Set embedding dimension based on model
        if "clip-vit-base-patch32" in self.clip_model_name.lower():
            self.embedding_dimension = 512
        elif "clip-vit-large" in self.clip_model_name.lower():
            self.embedding_dimension = 768

    def get_model_paths(self) -> dict:
        """Get all model-related paths."""
        return {
            "faiss_index": self.faiss_index_path,
            "faiss_metadata": self.faiss_metadata_path,
            "embeddings_cache": self.embeddings_cache_path,
            "model_cache": self.model_cache_dir
        }

    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.debug or self.log_level.upper() == "DEBUG"

    def get_cors_config(self) -> dict:
        """Get CORS configuration."""
        return {
            "allow_origins": self.allowed_origins,
            "allow_credentials": True,
            "allow_methods": ["*"],
            "allow_headers": ["*"]
        }

    def get_uvicorn_config(self) -> dict:
        """Get Uvicorn server configuration."""
        return {
            "host": self.api_host,
            "port": self.api_port,
            "workers": self.api_workers,
            "log_level": self.log_level.lower(),
            "reload": self.is_development()
        }

    def validate_llm_config(self) -> bool:
        """Validate LLM configuration."""
        if not self.llm_api_token:
            return False

        if self.llm_provider.lower() not in ["huggingface", "openai"]:
            return False

        return True

    def __str__(self) -> str:
        """String representation (safe for logging)."""
        safe_config = self.dict()

        # Mask sensitive information
        if safe_config.get("llm_api_token"):
            safe_config["llm_api_token"] = "***masked***"

        return f"Config({safe_config})"