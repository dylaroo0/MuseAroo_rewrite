import click
import json
from musearoo.analyzers.master_analyzer import MasterAnalyzer

@click.group()
def cli():
    """MuseAroo Command-Line Interface"""
    pass

@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
def analyze(file_path: str):
    """
    Run the full analysis pipeline on an audio file.
    """
    click.echo(click.style("Initializing MuseAroo Analysis Pipeline...", fg='cyan'))
    
    # Initialize the master analyzer, which discovers all plugins
    master_analyzer = MasterAnalyzer()
    
    # Run the analysis
    analysis_results = master_analyzer.analyze_file(file_path)
    
    # Print the results
    click.echo(click.style("\nFinal Aggregated Analysis:", fg='green', bold=True))
    click.echo(json.dumps(analysis_results, indent=2))

if __name__ == '__main__':
    cli()

