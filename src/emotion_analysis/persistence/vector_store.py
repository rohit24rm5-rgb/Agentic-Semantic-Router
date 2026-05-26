import os
import chromadb
from chromadb.utils import embedding_functions

# Connect to existing ChromaDB dynamically
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
DB_PATH = os.path.join(BASE_DIR, "data", "chroma_db")
client = chromadb.PersistentClient(path=DB_PATH)
ef = embedding_functions.DefaultEmbeddingFunction()

class VectorStore:
    def __init__(self):
        self.client = client
        self.ef = ef
        
    def similarity_search(self, query: str, k: int = 4, filter: dict = None) -> list[dict]:
        all_results = []
        target_collections = []
        
        if filter and "source" in filter:
            target_collections.append(self.client.get_or_create_collection(
                name=filter["source"], 
                embedding_function=self.ef,
                metadata={"hnsw:space": "cosine"}
            ))
            search_filter = {k: v for k, v in filter.items() if k != "source"}
            if not search_filter:
                search_filter = None
        else:
            # Fan-out: explicitly query known knowledge collections
            target_collections = [
                self.client.get_or_create_collection("sarcasm_context", embedding_function=self.ef, metadata={"hnsw:space": "cosine"}),
                self.client.get_or_create_collection("sentiment140_context", embedding_function=self.ef, metadata={"hnsw:space": "cosine"})
            ]
            search_filter = None

        for col in target_collections:
            kwargs = {"query_texts": [query], "n_results": k}
            if search_filter:
                kwargs["where"] = search_filter
                
            res = col.query(**kwargs)
            
            if res["documents"] and res["documents"][0]:
                distances = res["distances"][0]
                
                for doc, dist, meta in zip(res["documents"][0], distances, res["metadatas"][0]):
                    if not meta: meta = {}
                    meta["source_collection"] = col.name
                    all_results.append({
                        "text": doc,
                        "metadata": meta,
                        "distance": dist,
                        "normalized_distance": dist
                    })
                    
        # Sort globally across all collections by raw distance
        all_results.sort(key=lambda x: x["distance"])
        return all_results[:k]

vector_store = VectorStore()
