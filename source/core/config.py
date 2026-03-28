from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    project_name: str = "Traffic Law RAG"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6334
    embedding_model_name: str = "KeepItReal/vietnamese-sbert"
    api_key: str = ""
    
    class Config:
        env_file = ".env"
        extra = "allow"
