import time
import json
import hashlib
import os
from src.emotion_analysis.persistence.vector_store import vector_store

CACHE_COLLECTION = "semantic_cache"
CACHE_TTL = float(os.environ.get("CACHE_TTL_SECONDS", 86400.0))

class SemanticRouter:
    def __init__(self):
        self.collection = vector_store.client.get_or_create_collection(
            name=CACHE_COLLECTION,
            embedding_function=vector_store.ef,
            metadata={"hnsw:space": "cosine"}
        )
        
    def check_cache(self, query: str, threshold: float = 0.05) -> str | None:
        results = self.collection.query(
            query_texts=[query],
            n_results=1
        )
        if not results['documents'] or not results['documents'][0]:
            return None
            
        distance = results['distances'][0][0]
        meta = results['metadatas'][0][0]
        
        timestamp = float(meta.get("timestamp", 0))
        if time.time() - timestamp > CACHE_TTL:
            return None # Expired (Lazy deletion happens on access effectively, though physically remains until cleanup)
            
        if distance < threshold:
            return meta.get("response")
        return None

    def save_to_cache(self, query: str, response: dict):
        doc_id = hashlib.md5(query.encode('utf-8')).hexdigest()
        self.collection.upsert(
            documents=[query],
            metadatas=[{"response": json.dumps(response), "timestamp": time.time()}],
            ids=[doc_id]
        )

    def cleanup_expired(self):
        """Background cleanup: aggressively prune dead entries."""
        try:
            self.collection.delete(where={"timestamp": {"$lt": time.time() - CACHE_TTL}})
        except Exception:
            pass

semantic_router = SemanticRouter()
