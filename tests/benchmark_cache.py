import time
import json
import statistics
import os
import sys
import csv
from dotenv import load_dotenv

# Load environment before anything else
load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.emotion_analysis.core.router import semantic_router

DATASET_B = [
    "I feel sad",
    "I'm feeling down",
    "I feel depressed",
    "I am extremely unhappy",
    "Sorrow fills my heart"
] * 200 # 1000 items

THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.25]

def run_csv_benchmark():
    results = []
    
    print("Running Semantic Cache CSV Benchmark...")
    for thresh in THRESHOLDS:
        hits = 0
        latencies = []
        
        for query in DATASET_B:
            start = time.time()
            res = semantic_router.check_cache(query, threshold=thresh)
            latencies.append((time.time() - start) * 1000)
            if res:
                hits += 1
                
        hit_rate = hits / len(DATASET_B)
        avg_lat = statistics.mean(latencies)
        p95 = statistics.quantiles(latencies, n=100)[94] if len(latencies) > 1 else latencies[0]
        
        # Calculate simulated token reduction (assuming misses cost tokens, hits don't)
        token_reduction = hit_rate * 0.77 # Expected 77% token correlation ratio
        
        results.append({
            "threshold": thresh,
            "cache_hit_rate": round(hit_rate, 4),
            "avg_latency_ms": round(avg_lat, 2),
            "p95_latency_ms": round(p95, 2),
            "token_reduction_est": round(token_reduction, 4)
        })
        
    csv_file = os.path.join(os.path.dirname(__file__), "benchmark_results.csv")
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["threshold", "cache_hit_rate", "avg_latency_ms", "p95_latency_ms", "token_reduction_est"])
        writer.writeheader()
        writer.writerows(results)
        
    print(f"Results written to {csv_file}")

if __name__ == "__main__":
    run_csv_benchmark()
