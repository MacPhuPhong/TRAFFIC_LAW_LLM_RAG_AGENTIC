from fastapi import FastAPI
from pydantic import BaseModel
from traffic_rag.source.generation.rag_pipeline import TrafficRAGPipeline
from source.core.config import Settings

app = FastAPI(title="Vietnam Road Traffic Law RAG")
settings = Settings()
pipeline = TrafficRAGPipeline(settings)

class QueryRequest(BaseModel):
    query: str

@app.post("/api/chat")
async def chat(request: QueryRequest):
    result = pipeline.run(request.query)
    return result

@app.get("/health")
async def health():
    return {"status": "ok"}
