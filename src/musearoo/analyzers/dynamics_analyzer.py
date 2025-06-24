from typing import Dict, Any
import librosa
import numpy as np
from .base_analyzer import BaseAnalyzer

class DynamicsAnalyzer(BaseAnalyzer):
    """
    Analyzes the dynamics of an audio file, including loudness and dynamic range.
    """
    name = "dynamics"

    def analyze(self, file_path: str) -> Dict[str, Any]:
        """
        Calculates the root-mean-square (RMS) energy and dynamic range.

        Args:
            file_path: The path to the audio file.

        Returns:
            A dictionary containing dynamics-related features.
        """
        print(f"  - Running dynamics analysis on {file_path}...")
        try:
            # Load audio file
            y, sr = librosa.load(file_path)

            if y.size == 0:
                print(f"    - Warning: Empty audio file at {file_path}")
                return {
                    "overall_loudness": 0,
                    "dynamic_range": 0
                }

            # Calculate RMS energy
            rms = librosa.feature.rms(y=y)[0]

            # Calculate overall loudness (average RMS) and dynamic range (std of RMS)
            loudness = np.mean(rms)
            dynamic_range = np.std(rms)

            # Normalize values to a 0-1 range (this is a simple heuristic)
            # A more robust implementation would use a reference or learned min/max
            loudness_normalized = loudness / 0.4 
            dynamic_range_normalized = dynamic_range / 0.2

            return {
                "overall_loudness": float(np.clip(loudness_normalized, 0, 1)),
                "dynamic_range": float(np.clip(dynamic_range_normalized, 0, 1))
            }
        except Exception as e:
            print(f"    - Error in DynamicsAnalyzer: {e}")
            return {
                "overall_loudness": 0,
                "dynamic_range": 0
            }
