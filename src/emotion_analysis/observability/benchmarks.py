import time
import logging
import threading
import collections
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Benchmarks")

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, reset_timeout: float = 60.0):
        self.state = "CLOSED"
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.lock = threading.Lock()
        
        # Metrics
        self.open_count = 0
        self.half_open_count = 0
        self.fallback_requests = 0
        self.recovery_times = []

    def allow_request(self) -> bool:
        """Returns True if primary system should be attempted."""
        with self.lock:
            if self.state == "CLOSED":
                return True
                
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.reset_timeout:
                    self.state = "HALF-OPEN"
                    self.half_open_count += 1
                    return True # Allow exactly 1 test request
                return False
                
            if self.state == "HALF-OPEN":
                return False # A test request is actively running. Block concurrents.
        return False

    def record_success(self):
        with self.lock:
            if self.state == "HALF-OPEN":
                recovery_time = time.time() - self.last_failure_time
                self.recovery_times.append(recovery_time)
            self.state = "CLOSED"
            self.failure_count = 0

    def record_failure(self):
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.state == "HALF-OPEN" or self.failure_count >= self.failure_threshold:
                if self.state != "OPEN":
                    self.open_count += 1
                self.state = "OPEN"
                
    def record_fallback(self):
        with self.lock:
            self.fallback_requests += 1

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            avg_recovery = sum(self.recovery_times) / len(self.recovery_times) if self.recovery_times else 0.0
            return {
                "circuit_current_state": self.state,
                "failure_count": self.failure_count,
                "circuit_open_count": self.open_count,
                "circuit_half_open_count": self.half_open_count,
                "fallback_request_count": self.fallback_requests,
                "average_recovery_time": avg_recovery
            }

class BenchmarkTracker:
    def __init__(self):
        self.metrics = collections.deque(maxlen=1000)
        self.lock = threading.Lock()
        self.cb = CircuitBreaker()

    def log_request(self, query: str, final_output: dict, latency_ms: float, is_fallback: bool, cache_hit: bool, active_provider: str):
        metric = {
            "query_length": len(query),
            "latency_ms": latency_ms,
            "is_fallback": is_fallback,
            "cache_hit": cache_hit,
            "emotion": final_output.get("final_emotion", ""),
            "sarcastic": final_output.get("is_sarcastic", False)
        }
        with self.lock:
            self.metrics.append(metric)
        
        logger.info(f"BENCHMARK: Latency: {latency_ms:.2f}ms | Cache Hit: {cache_hit} | Provider: {active_provider}")

    def get_summary(self) -> Dict[str, Any]:
        with self.lock:
            total = len(self.metrics)
            if total == 0:
                return {"total_requests": 0}
            
            avg_latency = sum(m["latency_ms"] for m in self.metrics) / total
            cache_hits = sum(1 for m in self.metrics if m["cache_hit"])
            fallbacks = sum(1 for m in self.metrics if m["is_fallback"])
            
            groq_success = total - fallbacks
            
        cb_snap = self.cb.snapshot()

        return {
            **cb_snap,
            "Groq_success_rate": round(groq_success / total, 2),
            "Gemini_fallback_rate": round(fallbacks / total, 2),
            "total_requests": total,
            "average_latency_ms": round(avg_latency, 2),
            "cache_hit_rate": round(cache_hits / total, 2),
        }

benchmark_tracker = BenchmarkTracker()
