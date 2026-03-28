from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    project_name: str = "Traffic Law RAG"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6334
    embedding_model_name: str = "KeepItReal/vietnamese-sbert"
    
    class Config:
        env_file = ".env"
