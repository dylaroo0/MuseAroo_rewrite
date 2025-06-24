from typing import Dict, Any
from .base_analyzer import BaseAnalyzer

class HarmonyAnalyzer(BaseAnalyzer):
    """Analyzes the harmony of an audio file."""
    name = "harmony"

    def analyze(self, file_path: str) -> Dict[str, Any]:
        """Returns mock harmony analysis data."""
        print(f"  - Running mock harmony analysis on {file_path}...")
        return {
            "key": "C Major",
            "chord_progression": ["C", "G", "Am", "F"]
        }

