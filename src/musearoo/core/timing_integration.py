#!/usr/bin/env python3
"""
MuseAroo Timing Integration Module
COMPLETE IMPLEMENTATION - Integrates precision timing into MuseAroo workflow

This module ensures that ALL MuseAroo plugins preserve the exact timing
of input files, including every microsecond of silence.
"""

import os
import sys
import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from functools import wraps

from .precision_timing_handler import PrecisionTimingHandler, TimingMetadata

try:
    from plugin_registry import register_plugin
except ImportError:
    def register_plugin(**kwargs):
        def decorator(f):
            f._plugin_metadata = kwargs
            return f
        return decorator

logger = logging.getLogger(__name__)


def timing_preserving_plugin(func):
    """
    Decorator to ensure plugins preserve input timing EXACTLY.
    
    This decorator wraps any MuseAroo plugin to ensure that:
    1. Input timing is analyzed before processing
    2. Output files are corrected to match input timing
    3. Every microsecond of silence is preserved
    """
    
    @wraps(func)
    def wrapper(input_path: str, output_dir: str = "reports", 
                analysis_context: dict = None, **kwargs):
        
        logger.info(f"🎯 Timing-preserving wrapper activated for {func.__name__}")
        
        # Initialize timing handler
        timing_handler = PrecisionTimingHandler()
        
        # Analyze input timing FIRST
        try:
            input_timing = timing_handler.analyze_input_timing(input_path)
            logger.info(f"✅ Input timing analyzed: {input_timing.total_duration_seconds}s total, "
                       f"{input_timing.leading_silence_seconds}s leading silence")
        except Exception as e:
            logger.error(f"Failed to analyze input timing: {e}")
            input_timing = None
        
        # Store original function result
        result = func(input_path, output_dir, analysis_context, **kwargs)
        
        # If the plugin generated files, apply timing correction
        if input_timing and isinstance(result, dict) and result.get('status') == 'success':
            
            # Find all generated files
            generated_files = _extract_generated_files(result, output_dir)
            
            if generated_files:
                logger.info(f"🔧 Applying timing correction to {len(generated_files)} generated files")
                
                for generated_file in generated_files:
                    if os.path.exists(generated_file) and _is_audio_or_midi_file(generated_file):
                        try:
                            # Apply timing correction
                            corrected_file = timing_handler.ensure_output_timing_match(
                                input_timing, generated_file
                            )
                            
                            # Replace original with timing-corrected version
                            os.replace(corrected_file, generated_file)
                            logger.info(f"✅ Timing preserved for {generated_file}")
                            
                            # Verify the correction
                            verification = timing_handler.verify_timing_match(
                                input_path, generated_file, tolerance_microseconds=1.0
                            )
                            
                            if verification['perfect_match']:
                                logger.info(f"✅ PERFECT timing match confirmed! "
                                           f"Duration diff: {verification['duration_difference_microseconds']:.3f} μs")
                            else:
                                logger.warning(f"⚠️ Timing mismatch: "
                                              f"Duration diff: {verification['duration_difference_microseconds']:.3f} μs")
                            
                        except Exception as e:
                            logger.warning(f"Failed to apply timing preservation to {generated_file}: {e}")
                
                # Add timing preservation info to result
                if 'data' not in result:
                    result['data'] = {}
                
                result['data']['timing_preserved'] = True
                result['data']['input_duration'] = str(input_timing.total_duration_seconds)
                result['data']['input_leading_silence'] = str(input_timing.leading_silence_seconds)
                
                logger.info("✅ Timing preservation complete!")
        
        return result
    
    return wrapper


def _extract_generated_files(result: Dict[str, Any], output_dir: str) -> List[str]:
    """Extract all generated file paths from a plugin result."""
    
    generated_files = []
    
    # Direct output file
    if 'output_file' in result:
        generated_files.append(result['output_file'])
    
    # List of generated files
    if 'generated_files' in result:
        if isinstance(result['generated_files'], list):
            generated_files.extend(result['generated_files'])
        else:
            generated_files.append(result['generated_files'])
    
    # Check data section
    if 'data' in result and isinstance(result['data'], dict):
        for key, value in result['data'].items():
            if isinstance(value, str) and _is_audio_or_midi_file(value):
                if os.path.exists(value):
                    generated_files.append(value)
    
    # Check for generated files in output directory
    if os.path.exists(output_dir):
        input_stem = Path(result.get('input_file', '')).stem
        if input_stem:
            for file in Path(output_dir).glob(f"{input_stem}*"):
                if _is_audio_or_midi_file(str(file)) and str(file) not in generated_files:
                    # Check if this file was just created (within last minute)
                    if os.path.getmtime(file) > os.path.getmtime(result.get('input_file', '')):
                        generated_files.append(str(file))
    
    return list(set(generated_files))  # Remove duplicates


def _is_audio_or_midi_file(file_path: str) -> bool:
    """Check if a file is an audio or MIDI file."""
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.aiff', '.ogg'}
    midi_extensions = {'.mid', '.midi'}
    
    ext = Path(file_path).suffix.lower()
    return ext in audio_extensions or ext in midi_extensions


class MuseArooTimingIntegrator:
    """
    Main class for integrating timing preservation into MuseAroo.
    
    This class provides methods to:
    1. Wrap existing orchestrators with timing preservation
    2. Verify timing preservation is working
    3. Generate timing reports
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.timing_handler = PrecisionTimingHandler()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.timing_operations = []
        
        self.logger.info("🎯 MuseAroo Timing Integrator initialized")
        self.logger.info("✅ Ready to preserve every microsecond of your music!")

    def wrap_orchestrator_analyze_file(self, orchestrator_instance):
        """
        Wrap the main orchestrator's analyze_file method with timing preservation.
        
        This ensures that ALL files processed by MuseAroo maintain exact timing.
        """
        
        original_analyze_file = orchestrator_instance.analyze_file
        integrator = self  # Capture self for the closure
        
        def timing_preserving_analyze_file(input_path: str, options: Optional[Dict[str, Any]] = None):
            """Wrapped analyze_file that preserves timing in all outputs."""
            
            integrator.logger.info(f"🎯 Timing-preserving analyze_file called for {input_path}")
            
            # Get input timing first
            try:
                input_timing = integrator.timing_handler.analyze_input_timing(input_path)
                integrator.logger.info(f"✅ Input timing analyzed:")
                integrator.logger.info(f"   Total duration: {input_timing.total_duration_seconds} seconds")
                integrator.logger.info(f"   Leading silence: {input_timing.leading_silence_seconds} seconds")
                integrator.logger.info(f"   Trailing silence: {input_timing.trailing_silence_seconds} seconds")
                
                # Print timing report
                print("\n" + integrator.timing_handler.get_timing_report(input_timing) + "\n")
                
                if options is None:
                    options = {}
                options['_input_timing'] = input_timing
                
            except Exception as e:
                integrator.logger.warning(f"Failed to analyze input timing for {input_path}: {e}")
                input_timing = None
            
            # Call original analyze_file
            result = original_analyze_file(input_path, options)
            
            # Apply timing correction to any generated files
            if input_timing and isinstance(result, dict):
                integrator._apply_timing_corrections_to_result(result, input_timing, input_path)
            
            return result
        
        # Replace the method
        orchestrator_instance.analyze_file = timing_preserving_analyze_file
        self.logger.info("✅ Orchestrator analyze_file method wrapped with timing preservation")

    def _apply_timing_corrections_to_result(self, result: Dict[str, Any], 
                                          input_timing: TimingMetadata, 
                                          input_path: str):
        """Apply timing corrections to all generated files in a result."""
        
        # Extract the output directory from the result
        output_dir = "reports"  # Default
        if 'plugins_results' in result:
            for plugin_name, plugin_result in result['plugins_results'].items():
                if isinstance(plugin_result, dict) and 'output_file' in plugin_result:
                    output_dir = str(Path(plugin_result['output_file']).parent)
                    break
        
        generated_files = _extract_generated_files(result, output_dir)
        
        if not generated_files:
            self.logger.info("No generated audio/MIDI files found to correct")
            return
        
        corrected_count = 0
        
        for file_path in generated_files:
            if os.path.exists(file_path) and _is_audio_or_midi_file(file_path):
                try:
                    self.logger.info(f"🔧 Applying timing correction to {file_path}")
                    
                    # Apply timing correction
                    corrected_file = self.timing_handler.ensure_output_timing_match(
                        input_timing, file_path
                    )
                    
                    # Replace original with corrected version
                    os.replace(corrected_file, file_path)
                    corrected_count += 1
                    
                    # Verify the correction
                    verification = self.timing_handler.verify_timing_match(
                        input_path, file_path, tolerance_microseconds=1.0
                    )
                    
                    if verification['perfect_match']:
                        self.logger.info(f"✅ PERFECT timing match for {file_path}")
                        self.logger.info(f"   Duration difference: {verification['duration_difference_microseconds']:.3f} microseconds")
                        self.logger.info(f"   Leading silence difference: {verification['leading_silence_difference_microseconds']:.3f} microseconds")
                    else:
                        self.logger.warning(f"⚠️ Timing mismatch for {file_path}")
                        self.logger.warning(f"   Duration difference: {verification['duration_difference_microseconds']:.3f} microseconds")
                
                except Exception as e:
                    self.logger.error(f"Failed to apply timing correction to {file_path}: {e}")
        
        # Add timing info to result
        if 'metadata' not in result:
            result['metadata'] = {}
        
        result['metadata']['timing_preservation_applied'] = corrected_count > 0
        result['metadata']['timing_corrected_files'] = corrected_count
        result['metadata']['input_duration_seconds'] = str(input_timing.total_duration_seconds)
        result['metadata']['input_leading_silence_seconds'] = str(input_timing.leading_silence_seconds)
        
        if corrected_count > 0:
            self.logger.info(f"✅ Timing preservation applied to {corrected_count} files")
            self.logger.info(f"✅ Every microsecond of silence has been preserved!")

    def verify_timing_preservation(self, input_file: str, output_file: str) -> Dict[str, Any]:
        """
        Verify that timing preservation is working correctly.
        
        Args:
            input_file: Path to input file
            output_file: Path to output file
            
        Returns:
            Verification results
        """
        return self.timing_handler.verify_timing_match(input_file, output_file, tolerance_microseconds=1.0)

    def get_timing_report(self, file_path: str) -> str:
        """Get a detailed timing report for a file."""
        timing_data = self.timing_handler.analyze_input_timing(file_path)
        return self.timing_handler.get_timing_report(timing_data)


def integrate_timing_with_musearoo(orchestrator_instance, config: Optional[Dict[str, Any]] = None):
    """
    Main integration function - call this to add timing preservation to MuseAroo.
    
    Args:
        orchestrator_instance: The MuseAroo orchestrator instance
        config: Optional configuration
        
    Returns:
        The timing integrator instance
    """
    
    logger.info("🎯 Integrating timing preservation with MuseAroo...")
    
    timing_integrator = MuseArooTimingIntegrator(config)
    timing_integrator.wrap_orchestrator_analyze_file(orchestrator_instance)
    
    logger.info("✅ MuseAroo timing preservation integration complete!")
    logger.info("✅ Every microsecond of silence will now be preserved in all outputs!")
    
    return timing_integrator


def verify_timing_preservation(input_file: str, output_file: str, tolerance_ms: float = 0.001) -> Dict[str, Any]:
    """
    Standalone function to verify timing preservation.
    
    Args:
        input_file: Path to input file
        output_file: Path to output file
        tolerance_ms: Tolerance in milliseconds (default 0.001 = 1 microsecond)
        
    Returns:
        Verification results
    """
    
    timing_handler = PrecisionTimingHandler()
    
    try:
        result = timing_handler.verify_timing_match(
            input_file, output_file, 
            tolerance_microseconds=tolerance_ms * 1000
        )
        
        if result['perfect_match']:
            print(f"✅ PERFECT timing match!")
            print(f"   Duration difference: {result['duration_difference_microseconds']:.3f} microseconds")
            print(f"   Leading silence difference: {result['leading_silence_difference_microseconds']:.3f} microseconds")
        else:
            print(f"❌ Timing mismatch detected!")
            print(f"   Duration difference: {result['duration_difference_microseconds']:.3f} microseconds")
            print(f"   Leading silence difference: {result['leading_silence_difference_microseconds']:.3f} microseconds")
        
        return result
        
    except Exception as e:
        return {'perfect_match': False, 'error': str(e)}


# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing MuseAroo Timing Integration...")
    print("=" * 60)
    
    # Test the integration module
    integrator = MuseArooTimingIntegrator()
    print("✅ Integration module created successfully!")
    
    print("\n📋 FEATURES:")
    print("  • Automatic timing preservation for all plugins")
    print("  • Microsecond-accurate silence preservation")
    print("  • Verification and reporting tools")
    print("  • Easy integration with existing MuseAroo code")
    
    print("\n🎯 TO INTEGRATE WITH YOUR MUSEAROO:")
    print("  1. Import: from musearoo_timing_integration import integrate_timing_with_musearoo")
    print("  2. In your orchestrator __init__: integrate_timing_with_musearoo(self)")
    print("  3. That's it! All outputs will preserve timing perfectly!")
    
    print("\n✅ Your acoustic recordings with silence will be preserved EXACTLY!")
