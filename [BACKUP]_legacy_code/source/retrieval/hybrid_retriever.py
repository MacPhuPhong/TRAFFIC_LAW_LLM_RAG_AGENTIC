import json
import numpy as np
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from pyvi import ViTokenizer
from source.core.config import Settings
from source.model.embedding_model import Sentences_Transformer_Embedding

class HybridRetriever:
    def __init__(self, settings: Settings, collection_name: str = "Traffic_Law_Hybrid"):
        self.settings = settings
        self.collection_name = collection_name
        self.client = QdrantClient(host=self.settings.qdrant_host, port=self.settings.qdrant_port)
        self.embedding_model = Sentences_Transformer_Embedding(settings)
        self.bm25 = None
        self.corpus_chunks = []
        
    def _tokenize(self, text: str):
        return ViTokenizer.tokenize(text).lower().split()

    def index_data(self, chunks_path: str):
        with open(chunks_path, 'r', encoding='utf-8') as f:
            self.corpus_chunks = json.load(f)
            
        # Prepare for BM25
        tokenized_corpus = [self._tokenize(c['content']) for c in self.corpus_chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # Prepare for Qdrant (Dense)
        # Check if collection exists, if not create
        collections = self.client.get_collections().collections
        if not any(c.name == self.collection_name for c in collections):
            dummy_vec = self.embedding_model.embeddings_bkai.embed_query("test")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=len(dummy_vec), distance=Distance.COSINE),
            )
            
        points = []
        for i, chunk in enumerate(self.corpus_chunks):
            vector = self.embedding_model.embeddings_bkai.embed_query(chunk['content'])
            points.append(PointStruct(
                id=i,
                vector=vector,
                payload=chunk
            ))
        
        batch_size = 500
        for b_idx in range(0, len(points), batch_size):
            batch = points[b_idx:b_idx+batch_size]
            self.client.upsert(collection_name=self.collection_name, points=batch)
            print(f"Indexed batch {b_idx} -> {b_idx+len(batch)} points into {self.collection_name}")

    def search(self, query: str, top_k: int = 5, alpha: float = 0.6):
        # 1. Semantic Search (Dense)
        query_vector = self.embedding_model.embeddings_bkai.embed_query(query)
        dense_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k * 2
        )
        
        # 2. BM25 Search (Sparse)
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # 3. Hybrid Fusion
        # Normalized scores
        # Dense scores are already 0-1 (cosine)
        # BM25 scores need normalization
        if np.max(bm25_scores) > 0:
            bm25_scores = bm25_scores / np.max(bm25_scores)
            
        # Combine
        combined_results = []
        for i, chunk in enumerate(self.corpus_chunks):
            # Dense score from Qdrant if present
            dense_score = 0
            for dr in dense_results:
                if dr.id == i:
                    dense_score = dr.score
                    break
            
            final_score = alpha * dense_score + (1 - alpha) * bm25_scores[i]
            if final_score > 0:
                combined_results.append({
                    "chunk": chunk,
                    "score": final_score
                })
                
        # Sort and return top_k
        combined_results = sorted(combined_results, key=lambda x: x['score'], reverse=True)
        return combined_results[:top_k]
