import time
import requests
import statistics

# Make sure FastAPI is running on port 8000
API_URL = "http://localhost:8000/analyze"

TEST_QUERIES = [
    "I am absolutely thrilled with the results today!",
    "This is the worst experience of my entire life. So disappointed.",
    "Oh great, another perfectly sunny day when I have to stay inside and work.",
    "I guess it's fine. Nothing special.",
    "Wow, I didn't expect to actually win. This is crazy!"
]

def run_latency_test(iterations: int = 5):
    print(f"Starting Latency Benchmarks ({iterations} iterations per query)...")
    print("-" * 50)
    
    all_latencies = []
    
    for query in TEST_QUERIES:
        print(f"\nQuery: '{query}'")
        query_latencies = []
        
        for i in range(iterations):
            start = time.time()
            try:
                # We use the FastAPI endpoint which triggers the LangGraph swarm
                res = requests.post(API_URL, json={"text": query})
                if res.status_code == 200:
                    data = res.json()
                    backend_latency = data["metrics"]["latency_ms"]
                    query_latencies.append(backend_latency)
                    all_latencies.append(backend_latency)
                else:
                    print(f"  [Iter {i+1}] Error: {res.status_code}")
            except Exception as e:
                print(f"  [Iter {i+1}] Request failed: {e}")
                
        if query_latencies:
            avg_lat = statistics.mean(query_latencies)
            p95_lat = statistics.quantiles(query_latencies, n=100)[94] if len(query_latencies) >= 2 else query_latencies[0]
            print(f"  -> Avg Latency: {avg_lat:.2f}ms | P95 Latency: {p95_lat:.2f}ms")
            
    if all_latencies:
        print("\n" + "=" * 50)
        print("OVERALL BENCHMARK SUMMARY")
        print(f"Total Requests: {len(all_latencies)}")
        print(f"Global Avg Latency: {statistics.mean(all_latencies):.2f}ms")
        print(f"Global P95 Latency: {statistics.quantiles(all_latencies, n=100)[94] if len(all_latencies) >= 2 else all_latencies[0]:.2f}ms")
        print("=" * 50)

if __name__ == "__main__":
    print("Ensure Ollama/Groq FastAPI server is running before executing.")
    # run_latency_test()
