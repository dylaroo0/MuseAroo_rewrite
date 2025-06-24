from typing import Dict, Any

class BaseAnalyzer:
    """Base class for all analyzer plugins."""
    
    # The name of the analyzer, used as a key in the results dictionary.
    name: str = "base_analyzer"

    def analyze(self, file_path: str) -> Dict[str, Any]:
        """
        Analyzes the given audio file and returns a dictionary of features.
        This method should be overridden by all subclasses.

        Args:
            file_path: The path to the audio file to analyze.

        Returns:
            A dictionary containing the analysis results.
        """
        raise NotImplementedError("Each analyzer must implement the 'analyze' method.")
