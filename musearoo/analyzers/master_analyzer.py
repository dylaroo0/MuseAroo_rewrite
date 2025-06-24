import os
import importlib
from typing import Dict, Any, List
from .base_analyzer import BaseAnalyzer

class MasterAnalyzer:
    """Discovers, runs, and aggregates results from all analyzer plugins."""

    def __init__(self, analyzer_directory: str = None):
        """
        Initializes the MasterAnalyzer and discovers all available analyzer plugins.

        Args:
            analyzer_directory: The directory to search for analyzer plugins.
                                Defaults to the 'analyzers' directory.
        """
        if analyzer_directory is None:
            analyzer_directory = os.path.dirname(__file__)
        
        self.analyzers: List[BaseAnalyzer] = self._discover_analyzers(analyzer_directory)

    def _discover_analyzers(self, directory: str) -> List[BaseAnalyzer]:
        """Dynamically discovers and loads all analyzer plugins."""
        discovered_analyzers = []
        for filename in os.listdir(directory):
            if filename.endswith('_analyzer.py') and filename != 'master_analyzer.py' and filename != 'base_analyzer.py':
                module_name = f".{filename[:-3]}"
                try:
                    module = importlib.import_module(module_name, package='musearoo.analyzers')
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if isinstance(attr, type) and issubclass(attr, BaseAnalyzer) and attr is not BaseAnalyzer:
                            discovered_analyzers.append(attr())
                            print(f"Discovered analyzer: {attr.name}")
                except ImportError as e:
                    print(f"Error importing analyzer {module_name}: {e}")
        return discovered_analyzers

    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """
        Runs all discovered analyzers on the given file and aggregates the results.

        Args:
            file_path: The path to the audio file to analyze.

        Returns:
            A dictionary containing the aggregated analysis results.
        """
        aggregated_results: Dict[str, Any] = {}
        print(f"\nAnalyzing file: {file_path} with {len(self.analyzers)} analyzers...")
        for analyzer in self.analyzers:
            try:
                result = analyzer.analyze(file_path)
                aggregated_results[analyzer.name] = result
                print(f"  -> {analyzer.name} analysis complete.")
            except Exception as e:
                print(f"Error running analyzer {analyzer.name}: {e}")
        
        print("\nMaster analysis complete.")
        return aggregated_results

