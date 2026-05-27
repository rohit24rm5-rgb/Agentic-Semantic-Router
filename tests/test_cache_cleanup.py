import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.emotion_analysis.core.router import semantic_router, CACHE_TTL

def test_cache_cleanup():
    print("Running Cleanup Integration Test...")
    
    col = semantic_router.collection
    
    expired_id = "test_expired_1"
    fresh_id = "test_fresh_1"
    
    # Delete if they exist
    try:
        col.delete(ids=[expired_id, fresh_id])
    except Exception:
        pass
        
    now = time.time()
    expired_time = now - CACHE_TTL - 3600 # 1 hour past expiration
    fresh_time = now - 3600 # 1 hour old (still fresh)
    
    # Insert entries
    col.upsert(
        documents=["expired doc", "fresh doc"],
        metadatas=[
            {"response": "expired", "timestamp": expired_time},
            {"response": "fresh", "timestamp": fresh_time}
        ],
        ids=[expired_id, fresh_id]
    )
    
    print("Entries inserted. Running cleanup...")
    semantic_router.cleanup_expired()
    
    # Assertions
    res = col.get(ids=[expired_id, fresh_id])
    remaining_ids = res["ids"]
    
    print(f"Remaining IDs after cleanup: {remaining_ids}")
    
    assert fresh_id in remaining_ids, "Fresh entry was incorrectly deleted!"
    assert expired_id not in remaining_ids, "Expired entry was not deleted!"
    
    print("SUCCESS: ChromaDB server-side $lt cleanup is fully verified.")
