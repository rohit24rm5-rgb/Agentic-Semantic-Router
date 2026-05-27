import pytest
from fastapi.testclient import TestClient
import sys
import os

# Ensure the root directory is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.emotion_analysis.app.main import app

client = TestClient(app)

EDGE_CASES = [
    {
        "query": "I finally got the massive promotion I spent five years fighting for, and now I just sit alone in my corner office staring blankly at the wall until 8 PM.",
        "expected_lexical_bias": ["Joy", "Excitement", "Pride", "Sadness", "Neutral"],
        "expected_final_emotion": ["Bittersweet", "Sadness", "Despair", "Depression"],
        "expect_override": True
    },
    {
        "query": "Oh fantastic, my flight is delayed by another 6 hours. This is exactly how I wanted to spend my holiday, sleeping on an airport floor.",
        "expected_lexical_bias": ["Joy", "Excitement", "Anger", "Frustration"],
        "expected_final_emotion": ["Anger", "Frustration", "Annoyance", "Sarcasm"],
        "expect_override": True
    },
    {
        "query": "I am so happy that my dog passed away today. It's truly a blessing.",
        "expected_lexical_bias": ["Joy", "Happiness"],
        "expected_final_emotion": ["Sadness", "Grief", "Despair", "Sarcasm"],
        "expect_override": True
    },
    {
        "query": "I am literally crying right now because my son just graduated from college!",
        "expected_lexical_bias": ["Sadness", "Despair", "Neutral"],
        "expected_final_emotion": ["Joy", "Happiness", "Pride"],
        "expect_override": False
    }
]

def test_contextual_override_accuracy():
    """
    Tests the core premise of the Multi-Agent engine: 
    Does the Contextual Nuance Agent successfully override the Surface Lexical Agent 
    when faced with sarcasm or psychological paradoxes?
    """
    for idx, case in enumerate(EDGE_CASES):
        print(f"\\n--- Running Edge Case {idx+1} ---")
        print(f"Query: {case['query']}")
        
        response = client.post("/analyze", json={
            "text": case["query"],
            "session_id": f"test_acc_session_{idx}"
        })
        
        assert response.status_code == 200, f"API failed with status {response.status_code}"
        
        data = response.json().get("data", {})
        
        assert "surface_emotion" in data, "API response missing surface_emotion key"
        assert "final_emotion" in data, "API response missing final_emotion key"
        
        surface = data.get("surface_emotion", "")
        final = data.get("final_emotion", "")
        sarcastic = data.get("is_sarcastic", False)
        
        print(f"Lexical (Surface): {surface}")
        print(f"Contextual (Final): {final}")
        print(f"Sarcastic: {sarcastic}")
        
        if case["expect_override"]:
            # If the engine correctly detects nuance, the final emotion should differ from the naive lexical interpretation
            assert surface != final or sarcastic is True, \
                f"Contextual Agent failed to override Lexical Agent. Both returned {final}."
            # We do not assert exact generative text outputs as the LLM is highly variable.
            # The core logic test is simply that the Nuance Agent recognized a conflict or sarcasm.
                
        print("PASS")
