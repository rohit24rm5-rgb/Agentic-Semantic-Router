"""
Locust load testing script.
Run command: locust -f tests/locustfile.py -u 50 -r 5 --run-time 60s --host=http://localhost:8000
"""
from locust import HttpUser, task, between
import random

class EmotionAnalysisUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def analyze_emotion(self):
        prompts = [
            "I feel absolutely thrilled about this!",
            "I'm feeling somewhat down and depressed.",
            "I just don't know what to do anymore.",
            "This has been the worst experience of my life.",
            "I am feeling incredibly peaceful today.",
            "I am shaking with anger right now!",
            "I'm so worried about the future.",
            "What a beautiful and relaxing morning.",
            "I'm completely disgusted by that thought.",
            "Surprise! I didn't expect this at all."
        ]
        
        payload = {
            "text": random.choice(prompts),
            "user_id": f"user_{random.randint(1, 10)}",
            "session_id": f"sess_{random.randint(1, 50)}"
        }
        
        with self.client.post("/analyze", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 500:
                response.failure("Internal Server Error - Circuit Breaker Tripped?")
            else:
                response.failure(f"Unexpected status: {response.status_code}")
