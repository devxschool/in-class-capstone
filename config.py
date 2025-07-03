import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings and configuration."""
    
    # OpenAI Configuration
    openai_api_key: str
    
    # Pinecone Configuration
    pinecone_api_key: str
    pinecone_environment: str
    pinecone_index_name: str = "in-class-capstone"
    
    # Embedding Model Configuration
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # LLM Configuration
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.7
    
    # Document Processing
    max_file_size_mb: int = 50
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Create global settings instance
settings = Settings() 