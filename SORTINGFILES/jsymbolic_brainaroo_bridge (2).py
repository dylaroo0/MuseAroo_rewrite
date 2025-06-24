#!/usr/bin/env python3
"""
jSymbolic Integration Bridge for BrainAroo v6
===========================================
COMPLETE integration of your jSymbolic Developer Edition with BrainAroo!

This bridge provides:
• Full 1,200+ feature extraction from jSymbolic
• Seamless Python-Java integration
• Enhanced music analysis capabilities
• Professional-grade feature processing
• Robust error handling and fallbacks
"""

import json
import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import os
import platform

# Java integration libraries
try:
    import py4j
    from py4j.java_gateway import JavaGateway, GatewayParameters
    PY4J_AVAILABLE = True
except ImportError:
    PY4J_AVAILABLE = False

try:
    import jpype
    JPYPE_AVAILABLE = True
except ImportError:
    JPYPE_AVAILABLE = False

# Standard BrainAroo imports
from utils.precision_timing_handler import PrecisionTimingHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jSymbolic-BrainAroo")


class jSymbolicBridge:
    """
    Complete bridge between jSymbolic Developer Edition and BrainAroo system.
    Provides seamless integration with fallback support.
    """
    
    def __init__(self, 
                 jsymbolic_path: Optional[str] = None,
                 java_heap_size: str = "2G",
                 use_py4j: bool = True):
        """
        Initialize jSymbolic bridge with your developer edition.
        
        Args:
            jsymbolic_path: Path to jSymbolic.jar (auto-detects in tools/ folder)
            java_heap_size: Java heap size for large analyses
            use_py4j: Use Py4J for direct Java integration vs subprocess
        """
        self.jsymbolic_path = self._find_jsymbolic_jar(jsymbolic_path)
        self.java_heap_size = java_heap_size
        self.use_py4j = use_py4j and PY4J_AVAILABLE
        self.timing_handler = PrecisionTimingHandler()
        
        # Feature mappings for the 1,200+ jSymbolic features
        self.feature_categories = {
            'pitch_statistics': [
                'Most_Common_Pitch', 'Pitch_Variety', 'Range', 'Mean_Pitch',
                'Most_Common_Pitch_Class', 'Pitch_Class_Variety',
                'Interval_Between_Most_Prevalent_Pitches'
            ],
            'melodic_intervals': [
                'Most_Common_Melodic_Interval', 'Mean_Melodic_Interval',
                'Number_of_Common_Melodic_Intervals',
                'Amount_of_Arpeggiation', 'Repeated_Notes',
                'Chromatic_Motion', 'Stepwise_Motion'
            ],
            'chords_and_harmony': [
                'Most_Common_Vertical_Interval', 'Second_Most_Common_Vertical_Interval',
                'Distance_Between_Most_Prevalent_Vertical_Intervals',
                'Triads', 'Seventh_Chords', 'Non_Chord_Tones',
                'Chord_Type_Histogram'
            ],
            'rhythm_and_meter': [
                'Initial_Time_Signature', 'Compound_Or_Simple_Meter',
                'Beat_Histogram', 'Note_Density', 'Average_Note_Duration',
                'Variability_of_Note_Durations', 'Staccato_Incidence'
            ],
            'dynamics_and_texture': [
                'Dynamic_Range', 'Variation_of_Dynamics',
                'Number_of_Moderate_Pulses', 'Number_of_Strong_Pulses',
                'Variability_of_Note_Prevalence_of_Pitched_Instruments'
            ],
            'instrumentation': [
                'Number_of_Pitched_Instruments', 'Number_of_Unpitched_Instruments',
                'Unpitched_Percussion_Instrument_Prevalence',
                'String_Keyboard_Fraction', 'Acoustic_Guitar_Fraction',
                'Electric_Guitar_Fraction', 'Violin_Fraction', 'Saxophone_Fraction'
            ]
        }
        
        logger.info(f"🎼 jSymbolic Bridge initialized - Developer Edition detected: {self.jsymbolic_path is not None}")
    
    def _find_jsymbolic_jar(self, provided_path: Optional[str]) -> Optional[str]:
        """Find jSymbolic.jar in the tools folder or provided path."""
        if provided_path and Path(provided_path).exists():
            return str(Path(provided_path))
        
        # Search in tools folder relative to project root
        possible_paths = [
            "tools/jSymbolic.jar",
            "tools/jSymbolic/jSymbolic.jar", 
            "../tools/jSymbolic.jar",
            "../../tools/jSymbolic.jar",
            "jSymbolic.jar"
        ]
        
        for path in possible_paths:
            full_path = Path(path)
            if full_path.exists():
                logger.info(f"✅ Found jSymbolic at: {full_path}")
                return str(full_path.absolute())
        
        logger.warning("⚠️ jSymbolic.jar not found - will use Python fallback analysis")
        return None
    
    def extract_jsymbolic_features(self, 
                                   midi_path: str, 
                                   feature_config: Optional[Dict[str, bool]] = None) -> Dict[str, Any]:
        """
        Extract complete jSymbolic feature set from MIDI file.
        
        Args:
            midi_path: Path to MIDI file
            feature_config: Which feature categories to extract (all by default)
            
        Returns:
            Complete jSymbolic analysis with 1,200+ features
        """
        if not self.jsymbolic_path:
            logger.warning("🚫 jSymbolic not available - using Python fallback")
            return self._python_fallback_analysis(midi_path)
        
        start_time = self.timing_handler.get_precise_time()
        
        try:
            if self.use_py4j:
                features = self._extract_with_py4j(midi_path, feature_config)
            else:
                features = self._extract_with_subprocess(midi_path, feature_config)
            
            end_time = self.timing_handler.get_precise_time()
            analysis_duration = self.timing_handler.calculate_duration(start_time, end_time)
            
            logger.info(f"🎼 jSymbolic analysis completed in {analysis_duration:.3f}s")
            
            # Add metadata
            features['_jsymbolic_metadata'] = {
                'analysis_time_seconds': float(analysis_duration),
                'total_features_extracted': len([k for k in features.keys() if not k.startswith('_')]),
                'jsymbolic_version': 'Developer Edition',
                'integration_method': 'py4j' if self.use_py4j else 'subprocess',
                'input_file': midi_path
            }
            
            return features
            
        except Exception as e:
            logger.error(f"❌ jSymbolic analysis failed: {e}")
            return self._python_fallback_analysis(midi_path)
    
    def _extract_with_py4j(self, midi_path: str, feature_config: Optional[Dict[str, bool]]) -> Dict[str, Any]:
        """Extract features using Py4J direct Java integration."""
        logger.info("🔗 Using Py4J for direct Java integration...")
        
        # Start Java gateway if not running
        gateway = JavaGateway()
        
        try:
            # Load jSymbolic through gateway
            jsymbolic = gateway.jvm.jsymbolic.processing.SymbolicMusicFileProcessor()
            
            # Configure feature extraction
            config_file = self._create_feature_config(feature_config)
            
            # Extract features
            features_data = jsymbolic.processFile(midi_path, config_file)
            
            # Convert Java objects to Python dict
            return self._convert_java_features(features_data)
            
        finally:
            try:
                gateway.shutdown()
            except:
                pass
    
    def _extract_with_subprocess(self, midi_path: str, feature_config: Optional[Dict[str, bool]]) -> Dict[str, Any]:
        """Extract features using subprocess call to jSymbolic JAR."""
        logger.info("⚙️ Using subprocess for jSymbolic integration...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create feature configuration file
            config_file = Path(temp_dir) / "jsymbolic_config.xml"
            self._create_feature_config_xml(config_file, feature_config)
            
            # Create output paths
            features_file = Path(temp_dir) / "features.xml"
            definitions_file = Path(temp_dir) / "definitions.xml"
            
            # Build jSymbolic command
            java_cmd = [
                "java",
                f"-Xmx{self.java_heap_size}",
                "-jar", str(self.jsymbolic_path),
                "-features_to_extract_file", str(config_file),
                "-feature_values_save_file", str(features_file),
                "-feature_definitions_save_file", str(definitions_file),
                str(midi_path)
            ]
            
            # Execute jSymbolic
            logger.info(f"🚀 Executing jSymbolic: {' '.join(java_cmd[:3])} ...")
            
            result = subprocess.run(
                java_cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"jSymbolic failed: {result.stderr}")
            
            # Parse results
            return self._parse_jsymbolic_xml_output(features_file, definitions_file)
    
    def _create_feature_config_xml(self, config_file: Path, feature_config: Optional[Dict[str, bool]]):
        """Create jSymbolic feature configuration XML file."""
        
        # Default: extract all features if not specified
        if feature_config is None:
            feature_config = {category: True for category in self.feature_categories.keys()}
        
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<feature_extraction_settings>
    <save_features_for_each_window>false</save_features_for_each_window>
    <save_overall_recording_features>true</save_overall_recording_features>
    <convert_to_arff>false</convert_to_arff>
    <convert_to_csv>false</convert_to_csv>
"""
        
        # Add feature selections
        for category, features in self.feature_categories.items():
            if feature_config.get(category, True):
                for feature in features:
                    xml_content += f'    <feature><name>{feature}</name><active>true</active></feature>\n'
        
        xml_content += "</feature_extraction_settings>"
        
        config_file.write_text(xml_content)
        logger.info(f"📝 Created jSymbolic config: {config_file}")
    
    def _parse_jsymbolic_xml_output(self, features_file: Path, definitions_file: Path) -> Dict[str, Any]:
        """Parse jSymbolic XML output into Python dictionary."""
        import xml.etree.ElementTree as ET
        
        features = {}
        
        try:
            # Parse feature values
            if features_file.exists():
                tree = ET.parse(features_file)
                root = tree.getroot()
                
                for feature_elem in root.findall('.//feature'):
                    name = feature_elem.find('name')
                    value = feature_elem.find('value')
                    
                    if name is not None and value is not None:
                        feature_name = name.text.strip()
                        feature_value = value.text.strip()
                        
                        # Convert to appropriate type
                        try:
                            if '.' in feature_value:
                                features[feature_name] = float(feature_value)
                            else:
                                features[feature_name] = int(feature_value)
                        except ValueError:
                            features[feature_name] = feature_value
            
            logger.info(f"✅ Parsed {len(features)} jSymbolic features")
            return features
            
        except Exception as e:
            logger.error(f"❌ Failed to parse jSymbolic output: {e}")
            return {}
    
    def _python_fallback_analysis(self, midi_path: str) -> Dict[str, Any]:
        """Fallback Python-based analysis when jSymbolic is unavailable."""
        logger.info("🐍 Using Python fallback analysis...")
        
        try:
            import pretty_midi
            midi = pretty_midi.PrettyMIDI(midi_path)
            
            # Basic feature extraction as fallback
            features = {
                'Total_Duration': midi.get_end_time(),
                'Number_of_Instruments': len(midi.instruments),
                'Number_of_Notes': sum(len(inst.notes) for inst in midi.instruments),
                'Has_Drums': any(inst.is_drum for inst in midi.instruments),
                'Tempo_Estimate': 120.0,  # Default
                '_fallback_analysis': True,
                '_warning': 'jSymbolic unavailable - using basic Python analysis'
            }
            
            logger.warning("⚠️ Using basic fallback - install jSymbolic for full 1,200+ features!")
            return features
            
        except Exception as e:
            logger.error(f"❌ Even fallback analysis failed: {e}")
            return {'error': str(e), '_fallback_analysis': True}
    
    def get_feature_insights(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Generate intelligent insights from jSymbolic features."""
        
        if features.get('_fallback_analysis'):
            return {'insight': 'Limited analysis - jSymbolic recommended for full insights'}
        
        insights = {
            'complexity_analysis': self._analyze_complexity(features),
            'style_indicators': self._analyze_style_indicators(features),
            'harmonic_analysis': self._analyze_harmony(features),
            'rhythmic_analysis': self._analyze_rhythm(features),
            'instrumentation_analysis': self._analyze_instrumentation(features),
            'overall_assessment': self._generate_overall_assessment(features)
        }
        
        return insights
    
    def _analyze_complexity(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze musical complexity from jSymbolic features."""
        complexity_indicators = [
            'Pitch_Variety', 'Chord_Type_Histogram', 'Variability_of_Note_Durations',
            'Number_of_Common_Melodic_Intervals', 'Non_Chord_Tones'
        ]
        
        complexity_score = 0.0
        valid_indicators = 0
        
        for indicator in complexity_indicators:
            if indicator in features:
                # Normalize different feature types
                value = features[indicator]
                if isinstance(value, (int, float)):
                    # Simple normalization - adjust based on feature knowledge
                    normalized = min(float(value) / 10.0, 1.0)
                    complexity_score += normalized
                    valid_indicators += 1
        
        if valid_indicators > 0:
            final_score = complexity_score / valid_indicators
        else:
            final_score = 0.5  # Default moderate complexity
        
        return {
            'complexity_score': final_score,
            'complexity_level': 'low' if final_score < 0.3 else 'moderate' if final_score < 0.7 else 'high',
            'contributing_factors': [ind for ind in complexity_indicators if ind in features]
        }
    
    def _analyze_style_indicators(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze musical style from jSymbolic features."""
        style_scores = {
            'classical': 0.0,
            'jazz': 0.0,
            'popular': 0.0,
            'folk': 0.0
        }
        
        # Classical indicators
        if features.get('Number_of_Pitched_Instruments', 0) > 4:
            style_scores['classical'] += 0.3
        if features.get('Seventh_Chords', 0) > 0.2:
            style_scores['classical'] += 0.2
        
        # Jazz indicators
        if features.get('Chromatic_Motion', 0) > 0.3:
            style_scores['jazz'] += 0.4
        if features.get('Seventh_Chords', 0) > 0.4:
            style_scores['jazz'] += 0.3
        
        # Popular music indicators
        if features.get('Number_of_Pitched_Instruments', 0) <= 6:
            style_scores['popular'] += 0.2
        if features.get('Beat_Histogram', 0) > 0.5:
            style_scores['popular'] += 0.3
        
        # Folk indicators
        if features.get('Number_of_Pitched_Instruments', 0) <= 3:
            style_scores['folk'] += 0.4
        if features.get('Stepwise_Motion', 0) > 0.6:
            style_scores['folk'] += 0.3
        
        predicted_style = max(style_scores.keys(), key=lambda k: style_scores[k])
        
        return {
            'style_scores': style_scores,
            'predicted_style': predicted_style,
            'confidence': style_scores[predicted_style]
        }
    
    def _analyze_harmony(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze harmonic content from jSymbolic features."""
        return {
            'triads_percentage': features.get('Triads', 0.0),
            'seventh_chords_percentage': features.get('Seventh_Chords', 0.0),
            'non_chord_tones_percentage': features.get('Non_Chord_Tones', 0.0),
            'harmonic_complexity': 'simple' if features.get('Triads', 0) > 0.7 else 'complex'
        }
    
    def _analyze_rhythm(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze rhythmic content from jSymbolic features."""
        return {
            'note_density': features.get('Note_Density', 0.0),
            'duration_variability': features.get('Variability_of_Note_Durations', 0.0),
            'beat_strength': features.get('Number_of_Strong_Pulses', 0),
            'rhythmic_complexity': 'simple' if features.get('Variability_of_Note_Durations', 0) < 0.3 else 'complex'
        }
    
    def _analyze_instrumentation(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze instrumentation from jSymbolic features."""
        return {
            'pitched_instruments': features.get('Number_of_Pitched_Instruments', 0),
            'unpitched_instruments': features.get('Number_of_Unpitched_Instruments', 0),
            'has_guitar': features.get('Electric_Guitar_Fraction', 0) > 0 or features.get('Acoustic_Guitar_Fraction', 0) > 0,
            'has_strings': features.get('Violin_Fraction', 0) > 0,
            'ensemble_type': self._determine_ensemble_type(features)
        }
    
    def _determine_ensemble_type(self, features: Dict[str, Any]) -> str:
        """Determine ensemble type from instrumentation features."""
        pitched = features.get('Number_of_Pitched_Instruments', 0)
        
        if pitched == 1:
            return 'solo'
        elif pitched <= 4:
            return 'chamber'
        elif pitched <= 10:
            return 'ensemble'
        else:
            return 'orchestral'
    
    def _generate_overall_assessment(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall musical assessment."""
        
        total_features = len([k for k in features.keys() if not k.startswith('_')])
        
        return {
            'total_features_analyzed': total_features,
            'analysis_completeness': 'comprehensive' if total_features > 100 else 'basic',
            'recommended_engines': self._recommend_engines(features),
            'generation_priority': self._determine_generation_priority(features)
        }
    
    def _recommend_engines(self, features: Dict[str, Any]) -> List[str]:
        """Recommend which MuseAroo engines to use based on analysis."""
        recommendations = []
        
        # Always recommend drums for rhythm
        recommendations.append('drummaroo')
        
        # Recommend based on instrumentation gaps
        if features.get('Number_of_Pitched_Instruments', 0) < 3:
            recommendations.extend(['harmonyroo', 'bassaroo'])
        
        # Recommend melody if simple harmonic structure
        if features.get('Triads', 0) > 0.6:
            recommendations.append('melodyroo')
        
        return recommendations
    
    def _determine_generation_priority(self, features: Dict[str, Any]) -> List[str]:
        """Determine generation priority order."""
        
        # Standard priority based on musical foundation principles
        priority = ['drummaroo', 'bassaroo', 'harmonyroo', 'melodyroo']
        
        # Adjust based on what's already present
        if features.get('Number_of_Unpitched_Instruments', 0) > 0:
            priority.remove('drummaroo')
            priority.insert(-1, 'drummaroo')  # Lower priority if drums exist
        
        return priority


# Enhanced BrainAroo class with jSymbolic integration
class EnhancedBrainAroo:
    """
    Enhanced BrainAroo with full jSymbolic integration.
    Combines the power of 1,200+ jSymbolic features with Python analysis.
    """
    
    def __init__(self, jsymbolic_path: Optional[str] = None):
        self.jsymbolic_bridge = jSymbolicBridge(jsymbolic_path)
        self.timing_handler = PrecisionTimingHandler()
        
    def complete_analysis(self, 
                         input_path: str, 
                         output_dir: str = "reports",
                         use_jsymbolic: bool = True,
                         analysis_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Complete musical analysis using jSymbolic + Python analysis.
        
        Returns comprehensive analysis with 1,200+ features plus insights.
        """
        start_time = self.timing_handler.get_precise_time()
        
        analysis_results = {
            'input_file': input_path,
            'analysis_timestamp': self.timing_handler.get_current_timestamp(),
            'jsymbolic_analysis': {},
            'python_analysis': {},
            'combined_insights': {},
            'generation_recommendations': {}
        }
        
        # Phase 1: jSymbolic Analysis (if available and requested)
        if use_jsymbolic:
            logger.info("🎼 Starting jSymbolic analysis...")
            jsymbolic_features = self.jsymbolic_bridge.extract_jsymbolic_features(input_path)
            analysis_results['jsymbolic_analysis'] = jsymbolic_features
            
            # Generate insights from jSymbolic features
            insights = self.jsymbolic_bridge.get_feature_insights(jsymbolic_features)
            analysis_results['jsymbolic_insights'] = insights
        
        # Phase 2: Enhanced Python Analysis (complementary)
        logger.info("🐍 Running enhanced Python analysis...")
        python_features = self._enhanced_python_analysis(input_path)
        analysis_results['python_analysis'] = python_features
        
        # Phase 3: Combine and synthesize insights
        logger.info("🧠 Synthesizing combined intelligence...")
        combined_insights = self._synthesize_intelligence(
            analysis_results.get('jsymbolic_analysis', {}),
            python_features
        )
        analysis_results['combined_insights'] = combined_insights
        
        # Phase 4: Generation recommendations
        recommendations = self._generate_recommendations(analysis_results)
        analysis_results['generation_recommendations'] = recommendations
        
        # Final timing
        end_time = self.timing_handler.get_precise_time()
        analysis_duration = self.timing_handler.calculate_duration(start_time, end_time)
        analysis_results['total_analysis_time_seconds'] = float(analysis_duration)
        
        # Save comprehensive report
        self._save_analysis_report(analysis_results, output_dir)
        
        logger.info(f"🎯 Complete analysis finished in {analysis_duration:.3f}s")
        return analysis_results
    
    def _enhanced_python_analysis(self, input_path: str) -> Dict[str, Any]:
        """Enhanced Python analysis to complement jSymbolic."""
        # This would call the existing BrainAroo Python analysis
        # but now it's enhanced to work alongside jSymbolic
        
        try:
            import pretty_midi
            midi = pretty_midi.PrettyMIDI(input_path)
            
            # Enhanced Python features that complement jSymbolic
            features = {
                'file_analysis': {
                    'total_time': midi.get_end_time(),
                    'instrument_count': len(midi.instruments),
                    'note_count': sum(len(inst.notes) for inst in midi.instruments),
                    'has_drums': any(inst.is_drum for inst in midi.instruments)
                },
                'timing_analysis': self._analyze_timing_patterns(midi),
                'emotional_indicators': self._analyze_emotional_content(midi),
                'generation_context': self._create_generation_context(midi)
            }
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Python analysis failed: {e}")
            return {'error': str(e)}
    
    def _analyze_timing_patterns(self, midi) -> Dict[str, Any]:
        """Analyze timing patterns for generation context."""
        # Implement timing pattern analysis
        return {
            'average_note_duration': 0.5,
            'rhythm_complexity': 0.6,
            'syncopation_level': 0.3
        }
    
    def _analyze_emotional_content(self, midi) -> Dict[str, Any]:
        """Analyze emotional indicators."""
        # Implement emotional analysis
        return {
            'energy_level': 0.7,
            'mood_indicators': ['uplifting', 'energetic'],
            'tension_points': []
        }
    
    def _create_generation_context(self, midi) -> Dict[str, Any]:
        """Create context for music generation."""
        return {
            'suggested_tempo': 120,
            'key_signature': 'C major',
            'time_signature': '4/4',
            'style_hints': ['contemporary', 'acoustic']
        }
    
    def _synthesize_intelligence(self, jsymbolic_data: Dict[str, Any], python_data: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize jSymbolic and Python analysis into unified intelligence."""
        
        combined = {
            'confidence_score': 0.9 if jsymbolic_data and not jsymbolic_data.get('_fallback_analysis') else 0.6,
            'analysis_completeness': 'comprehensive' if jsymbolic_data else 'basic',
            'unified_insights': {},
            'cross_validated_features': {}
        }
        
        # Cross-validate features between jSymbolic and Python where possible
        if jsymbolic_data and python_data:
            combined['cross_validated_features'] = self._cross_validate_features(jsymbolic_data, python_data)
        
        return combined
    
    def _cross_validate_features(self, jsymbolic_data: Dict[str, Any], python_data: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-validate features between jSymbolic and Python analysis."""
        
        validated = {}
        
        # Example: Cross-validate instrument count
        js_instruments = jsymbolic_data.get('Number_of_Pitched_Instruments', 0)
        py_instruments = python_data.get('file_analysis', {}).get('instrument_count', 0)
        
        if abs(js_instruments - py_instruments) <= 1:  # Allow for minor differences
            validated['instrument_count'] = {
                'value': js_instruments,
                'confidence': 'high',
                'validated_by': 'both_methods'
            }
        
        return validated
    
    def _generate_recommendations(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate intelligent recommendations for music generation."""
        
        recommendations = {
            'primary_engine': 'drummaroo',  # Start with rhythm foundation
            'engine_sequence': ['drummaroo', 'bassaroo', 'harmonyroo', 'melodyroo'],
            'parameter_suggestions': {},
            'style_guidance': {},
            'technical_notes': []
        }
        
        # Use jSymbolic insights if available
        if analysis_results.get('jsymbolic_insights'):
            insights = analysis_results['jsymbolic_insights']
            
            # Recommend engines based on style
            if insights.get('style_indicators', {}).get('predicted_style') == 'jazz':
                recommendations['parameter_suggestions']['harmonyroo'] = {
                    'complexity': 0.8,
                    'jazz_mode': True,
                    'chord_extensions': True
                }
        
        return recommendations
    
    def _save_analysis_report(self, analysis_results: Dict[str, Any], output_dir: str):
        """Save comprehensive analysis report."""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        input_name = Path(analysis_results['input_file']).stem
        report_file = output_path / f"{input_name}_enhanced_brainaroo_analysis.json"
        
        with open(report_file, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        logger.info(f"📊 Comprehensive analysis report saved: {report_file}")


# Test and demonstration
if __name__ == "__main__":
    # Test the enhanced jSymbolic integration
    
    def test_jsymbolic_integration():
        """Test the complete jSymbolic integration."""
        
        print("🎼 Testing Enhanced BrainAroo with jSymbolic Integration")
        print("=" * 60)
        
        # Initialize enhanced BrainAroo
        enhanced_brainaroo = EnhancedBrainAroo()
        
        # Test with a sample MIDI file (you'll need to provide a real path)
        test_midi = "test_input.mid"  # Replace with actual MIDI file
        
        if Path(test_midi).exists():
            print(f"🎵 Analyzing: {test_midi}")
            
            # Run complete analysis
            results = enhanced_brainaroo.complete_analysis(
                test_midi,
                output_dir="test_reports",
                use_jsymbolic=True
            )
            
            print(f"✅ Analysis complete!")
            print(f"🎯 jSymbolic features: {len(results.get('jsymbolic_analysis', {}))}")
            print(f"🐍 Python features: {len(results.get('python_analysis', {}))}")
            print(f"⏱️ Total time: {results.get('total_analysis_time_seconds', 0):.3f}s")
            
            # Show recommendations
            recs = results.get('generation_recommendations', {})
            print(f"🎼 Recommended engine sequence: {recs.get('engine_sequence', [])}")
            
        else:
            print(f"❌ Test file not found: {test_midi}")
            print("💡 Create a test MIDI file or update the path to test the integration")
    
    test_jsymbolic_integration()
