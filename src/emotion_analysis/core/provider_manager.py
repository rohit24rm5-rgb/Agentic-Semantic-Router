import os
import logging

logger = logging.getLogger("ProviderManager")

class ProviderManager:
    def __init__(self):
        # Load Groq Keys
        self.groq_keys = [
            os.environ.get("GROQ_API_KEY"),
            os.environ.get("GROQ_API_KEY_2")
        ]
        self.groq_keys = [k for k in self.groq_keys if k]
        if not self.groq_keys:
            raise ValueError("CRITICAL: No GROQ_API_KEY found in environment.")
        self.active_groq_idx = 0
        
        # Load Gemini Keys
        self.gemini_keys = [
            os.environ.get("GOOGLE_API_KEY"),
            os.environ.get("GOOGLE_API_KEY_2")
        ]
        self.gemini_keys = [k for k in self.gemini_keys if k]
        self.active_gemini_idx = 0

    def get_active_groq_key(self) -> str:
        return self.groq_keys[self.active_groq_idx]

    def get_active_gemini_key(self) -> str:
        if not self.gemini_keys:
            return ""
        return self.gemini_keys[self.active_gemini_idx]

    def rotate_groq_key(self) -> bool:
        """Rotates to the next available Groq key. Returns True if rotation was successful."""
        if len(self.groq_keys) > 1:
            old_idx = self.active_groq_idx
            self.active_groq_idx = (self.active_groq_idx + 1) % len(self.groq_keys)
            if self.active_groq_idx != old_idx:
                logger.warning(f"Rotated Groq Key: Index {old_idx} -> {self.active_groq_idx}")
                return True
        return False

    def rotate_gemini_key(self) -> bool:
        """Rotates to the next available Gemini key. Returns True if rotation was successful."""
        if len(self.gemini_keys) > 1:
            old_idx = self.active_gemini_idx
            self.active_gemini_idx = (self.active_gemini_idx + 1) % len(self.gemini_keys)
            if self.active_gemini_idx != old_idx:
                logger.warning(f"Rotated Gemini Key: Index {old_idx} -> {self.active_gemini_idx}")
                return True
        return False

    def get_active_provider_name(self, is_fallback: bool = False) -> str:
        """Returns a string representation of the active provider and key index for benchmarking."""
        if is_fallback:
            return f"Gemini (Key {self.active_gemini_idx + 1})"
        return f"Groq (Key {self.active_groq_idx + 1})"

provider_manager = ProviderManager()
