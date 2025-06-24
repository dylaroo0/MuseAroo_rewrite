#!/usr/bin/env python3
"""
DrumRoo Generation Manager
===============================
Manages drum pattern generations with version control, A/B comparison, and Max4Live integration
for the Algorithmic Universal Drummaroo Engine.
"""

import json
import numpy as np
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import mido
import logging
import asyncio

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the DrumRoo engine and UI controls
try:
    from src.engines.drummaroo import AlgorithmicDrummaroo, DrumAlgorithm
    from src.ui.controls.drummaroo_controls import DrummarooUIControls
except ImportError as e:
    logger.error(f"Could not import DrumRoo modules: {e}")
    raise

# Max4Live Device Parameters - Full 51 Parameters
M4L_PARAMETERS = {
    # Groove Parameters (10)
    'groove_intensity': {'name': 'Groove Intensity', 'range': (0, 1), 'default': 0.5},
    'pattern_complexity': {'name': 'Pattern Complexity', 'range': (0, 1), 'default': 0.5},
    'swing_amount': {'name': 'Swing Amount', 'range': (0, 1), 'default': 0.0},
    'shuffle_feel': {'name': 'Shuffle Feel', 'range': (0, 1), 'default': 0.0},
    'syncopation_level': {'name': 'Syncopation', 'range': (0, 1), 'default': 0.3},
    'polyrhythm_amount': {'name': 'Polyrhythm', 'range': (0, 1), 'default': 0.0},
    'humanization': {'name': 'Humanization', 'range': (0, 1), 'default': 0.7},
    'timing_tightness': {'name': 'Timing Tightness', 'range': (0, 1), 'default': 0.8},
    'cross_rhythm_amount': {'name': 'Cross Rhythm', 'range': (0, 1), 'default': 0.0},
    'rhythmic_displacement': {'name': 'Rhythmic Displacement', 'range': (0, 1), 'default': 0.0},
    
    # Instrument Density (5)
    'kick_density': {'name': 'Kick Density', 'range': (0, 1), 'default': 0.5},
    'snare_density': {'name': 'Snare Density', 'range': (0, 1), 'default': 0.5},
    'hihat_density': {'name': 'HiHat Density', 'range': (0, 1), 'default': 0.6},
    'cymbal_density': {'name': 'Cymbal Density', 'range': (0, 1), 'default': 0.3},
    'percussion_density': {'name': 'Perc Density', 'range': (0, 1), 'default': 0.2},
    
    # Dynamics & Articulation (6)
    'dynamic_range': {'name': 'Dynamic Range', 'range': (0, 1), 'default': 0.6},
    'ghost_note_density': {'name': 'Ghost Notes', 'range': (0, 1), 'default': 0.3},
    'accent_strength': {'name': 'Accent Strength', 'range': (0, 1), 'default': 0.6},
    'velocity_variation': {'name': 'Velocity Var', 'range': (0, 1), 'default': 0.4},
    'micro_timing_amount': {'name': 'Micro Timing', 'range': (0, 1), 'default': 0.0},
    'hihat_openness': {'name': 'HiHat Openness', 'range': (0, 1), 'default': 0.3},
    
    # Fills & Transitions (4)
    'fill_frequency': {'name': 'Fill Frequency', 'range': (0, 1), 'default': 0.3},
    'fill_complexity': {'name': 'Fill Complexity', 'range': (0, 1), 'default': 0.5},
    'fill_length': {'name': 'Fill Length', 'range': (0, 1), 'default': 0.5},
    'transition_smoothness': {'name': 'Transition Smooth', 'range': (0, 1), 'default': 0.7},
    
    # Style Influences (8)
    'rock_influence': {'name': 'Rock Influence', 'range': (0, 1), 'default': 0.0},
    'jazz_influence': {'name': 'Jazz Influence', 'range': (0, 1), 'default': 0.0},
    'funk_influence': {'name': 'Funk Influence', 'range': (0, 1), 'default': 0.0},
    'latin_influence': {'name': 'Latin Influence', 'range': (0, 1), 'default': 0.0},
    'electronic_influence': {'name': 'Electronic Influence', 'range': (0, 1), 'default': 0.0},
    'metal_influence': {'name': 'Metal Influence', 'range': (0, 1), 'default': 0.0},
    'hiphop_influence': {'name': 'HipHop Influence', 'range': (0, 1), 'default': 0.0},
    'world_influence': {'name': 'World Influence', 'range': (0, 1), 'default': 0.0},
    
    # Kit Settings (5)
    'kick_variation': {'name': 'Kick Variation', 'range': (0, 1), 'default': 0.5},
    'snare_variation': {'name': 'Snare Variation', 'range': (0, 1), 'default': 0.5},
    'ride_vs_hihat': {'name': 'Ride vs HiHat', 'range': (0, 1), 'default': 0.2},
    'tom_usage': {'name': 'Tom Usage', 'range': (0, 1), 'default': 0.4},
    'percussion_variety': {'name': 'Perc Variety', 'range': (0, 1), 'default': 0.3},
    
    # Advanced Controls (7)
    'random_seed': {'name': 'Random Seed', 'range': (0, 10000), 'default': 42, 'type': 'int'},
    'generation_iterations': {'name': 'Generation Iter', 'range': (1, 10), 'default': 3, 'type': 'int'},
    'complexity_variance': {'name': 'Complexity Var', 'range': (0, 1), 'default': 0.2},
    'density_fluctuation': {'name': 'Density Fluct', 'range': (0, 1), 'default': 0.3},
    'style_blend_weight': {'name': 'Style Blend', 'range': (0, 1), 'default': 0.5},
    'odd_time_tendency': {'name': 'Odd Time Tendency', 'range': (0, 1), 'default': 0.0},
    'metric_modulation': {'name': 'Metric Modulation', 'range': (0, 1), 'default': 0.0},
    
    # Performance & Mix (6)
    'limb_independence': {'name': 'Limb Independence', 'range': (0, 1), 'default': 0.7},
    'stamina_simulation': {'name': 'Stamina Simulation', 'range': (0, 1), 'default': 0.0},
    'technique_precision': {'name': 'Technique Precision', 'range': (0, 1), 'default': 0.8},
    'room_presence': {'name': 'Room Presence', 'range': (0, 1), 'default': 0.5},
    'compression_amount': {'name': 'Compression', 'range': (0, 1), 'default': 0.3},
    'eq_brightness': {'name': 'EQ Brightness', 'range': (0, 1), 'default': 0.5},
    'stereo_width': {'name': 'Stereo Width', 'range': (0, 1), 'default': 0.7},
    # Optional: Algorithm weights (uncomment to enable M4L control)
    # 'groove_architect_weight': {'name': 'Groove Architect', 'range': (0, 1), 'default': 0.5},
    # 'polyrhythmic_engine_weight': {'name': 'Polyrhythmic Engine', 'range': (0, 1), 'default': 0.5},
    # 'fill_generator_weight': {'name': 'Fill Generator', 'range': (0, 1), 'default': 0.5},
}

@dataclass
class DrumGenerationVersion:
    """Single drum generation version with metadata"""
    version_id: str
    timestamp: str
    control_snapshot: Dict[str, float]
    midi_data: List[Dict]
    audio_preview: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    rating: Optional[int] = None
    is_favorite: bool = False
    notes: str = ""
    groove_feel: str = ""
    pattern_complexity: float = 0.0
    fill_density: float = 0.0
    kit_pieces_used: Set[str] = field(default_factory=set)
    style_blend: Dict[str, float] = field(default_factory=dict)
    polyrhythmic_elements: List[str] = field(default_factory=list)
    m4l_parameters: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "version_id": self.version_id,
            "timestamp": self.timestamp,
            "control_snapshot": self.control_snapshot,
            "midi_data": self.midi_data,
            "metadata": self.metadata,
            "tags": self.tags,
            "rating": self.rating,
            "is_favorite": self.is_favorite,
            "notes": self.notes,
            "groove_feel": self.groove_feel,
            "pattern_complexity": self.pattern_complexity,
            "fill_density": self.fill_density,
            "kit_pieces_used": list(self.kit_pieces_used),
            "style_blend": self.style_blend,
            "polyrhythmic_elements": self.polyrhythmic_elements,
            "m4l_parameters": self.m4l_parameters
        }

class DrumRooGenerationManager:
    """Manages drum pattern generations with version control and A/B comparison"""
    
    def __init__(self, drum_roo: AlgorithmicDrummaroo):
        self.drum_roo = drum_roo
        self.sessions: Dict[str, Dict] = {}
        self.current_session_id = None
        self.version_history: List[DrumGenerationVersion] = []
        self.comparison_state: Optional[Tuple[str, str]] = None
        try:
            self.midi_out = mido.open_output('DrumRoo Output', virtual=True)
        except Exception as e:
            logger.error(f"Failed to open MIDI output: {e}")
            self.midi_out = None
        
        # Initialize with default parameters
        self.current_controls = self._create_default_controls()
        self.current_m4l_params = {k: v['default'] for k, v in M4L_PARAMETERS.items()}
        
    def _create_default_controls(self) -> DrummarooUIControls:
        """Create default DrummarooUIControls instance"""
        return DrummarooUIControls(
            groove_intensity=0.5,
            pattern_complexity=0.5,
            swing_amount=0.0,
            shuffle_feel=0.0,
            syncopation_level=0.3,
            polyrhythm_amount=0.0,
            humanization=0.7,
            timing_tightness=0.8,
            cross_rhythm_amount=0.0,
            rhythmic_displacement=0.0,
            kick_density=0.5,
            snare_density=0.5,
            hihat_density=0.6,
            cymbal_density=0.3,
            percussion_density=0.2,
            dynamic_range=0.6,
            ghost_note_density=0.3,
            accent_strength=0.6,
            velocity_variation=0.4,
            micro_timing_amount=0.0,
            hihat_openness=0.3,
            fill_frequency=0.3,
            fill_complexity=0.5,
            fill_length=0.5,
            transition_smoothness=0.7,
            rock_influence=0.0,
            jazz_influence=0.0,
            funk_influence=0.0,
            latin_influence=0.0,
            electronic_influence=0.0,
            metal_influence=0.0,
            hiphop_influence=0.0,
            world_influence=0.0,
            kick_variation=0.5,
            snare_variation=0.5,
            ride_vs_hihat=0.2,
            tom_usage=0.4,
            percussion_variety=0.3,
            random_seed=42,
            generation_iterations=3,
            complexity_variance=0.2,
            density_fluctuation=0.3,
            style_blend_weight=0.5,
            odd_time_tendency=0.0,
            metric_modulation=0.0,
            limb_independence=0.7,
            stamina_simulation=0.0,
            technique_precision=0.8,
            room_presence=0.5,
            compression_amount=0.3,
            eq_brightness=0.5,
            stereo_width=0.7,
            algorithm_weights={algo: 0.5 for algo in DrumAlgorithm}
        )
    
    def _generate_version_id(self) -> str:
        """Generate unique version ID"""
        timestamp = datetime.now().isoformat()
        return hashlib.sha256(timestamp.encode()).hexdigest()[:8]
    
    def start_new_session(self, name: str) -> str:
        """Start a new generation session"""
        session_id = hashlib.md5(name.encode()).hexdigest()[:6]
        self.sessions[session_id] = {
            'name': name,
            'versions': [],
            'created_at': datetime.now().isoformat(),
            'last_modified': datetime.now().isoformat()
        }
        self.current_session_id = session_id
        logger.info(f"Started new session: {name} ({session_id})")
        return session_id
    
    async def generate_new_version(self, section: str, bars: int = 8) -> DrumGenerationVersion:
        """Generate a new drum pattern version"""
        if not self.current_session_id:
            raise RuntimeError("No active session. Create a session first.")
        
        # Apply Max4Live parameters to controls
        self._apply_m4l_params()
        
        # Generate drum pattern
        try:
            midi_events = await self.drum_roo.generate_drums(
                section=section,
                length_microseconds=int(60_000_000 / self.drum_roo.rhythm_intel.tempo * bars * self.drum_roo.rhythm_intel.time_signature[0]),
                ui_params=self.current_controls  # Fixed: pass ui_params
            )
        except Exception as e:
            logger.error(f"Failed to generate drums: {e}")
            raise
        
        # Create version object
        version_id = self._generate_version_id()
        new_version = DrumGenerationVersion(
            version_id=version_id,
            timestamp=datetime.now().isoformat(),
            control_snapshot=self.current_controls.to_dict(),
            midi_data=midi_events,
            m4l_parameters=self.current_m4l_params.copy(),
            groove_feel=self._detect_groove_feel(),
            pattern_complexity=self._calculate_pattern_complexity(midi_events),
            fill_density=self._calculate_fill_density(midi_events),
            kit_pieces_used=self._detect_kit_pieces(midi_events),
            style_blend=self._calculate_style_blend()
        )
        
        # Add to history and session
        self.version_history.append(new_version)
        self.sessions[self.current_session_id]['versions'].append(version_id)
        self.sessions[self.current_session_id]['last_modified'] = datetime.now().isoformat()
        logger.info(f"Generated version {version_id} for section {section}")
        
        return new_version
    
    def _apply_m4l_params(self):
        """Apply Max4Live parameters to controls"""
        for param, value in self.current_m4l_params.items():
            if hasattr(self.current_controls, param):
                setattr(self.current_controls, param, value)
        # Optional: Handle algorithm_weights (uncomment to enable)
        # if 'groove_architect_weight' in self.current_m4l_params:
        #     self.current_controls.algorithm_weights[DrumAlgorithm.GROOVE_ARCHITECT] = self.current_m4l_params['groove_architect_weight']
        # if 'polyrhythmic_engine_weight' in self.current_m4l_params:
        #     self.current_controls.algorithm_weights[DrumAlgorithm.POLYRHYTHMIC_ENGINE] = self.current_m4l_params['polyrhythmic_engine_weight']
        # if 'fill_generator_weight' in self.current_m4l_params:
        #     self.current_controls.algorithm_weights[DrumAlgorithm.FILL_GENERATOR] = self.current_m4l_params['fill_generator_weight']
    
    def _detect_groove_feel(self) -> str:
        """Analyze controls to determine groove feel"""
        if self.current_controls.swing_amount > 0.6:
            return "Swing"
        if self.current_controls.shuffle_feel > 0.5:
            return "Shuffle"
        if self.current_controls.syncopation_level > 0.7:
            return "Syncopated"
        return "Straight"
    
    def _calculate_pattern_complexity(self, midi_events) -> float:
        """Calculate pattern complexity from MIDI events"""
        event_count = len(midi_events)
        unique_pitches = len({e['pitch'] for e in midi_events})
        return min(1.0, (event_count * unique_pitches) / 500.0)
    
    def _calculate_fill_density(self, midi_events) -> float:
        """Calculate fill density from MIDI events"""
        total_events = len(midi_events)
        if total_events == 0:
            return 0.0
        fill_events = sum(1 for e in midi_events if e.get('is_fill', False))
        return fill_events / total_events
    
    def _detect_kit_pieces(self, midi_events) -> Set[str]:
        """Detect used kit pieces from MIDI pitches"""
        kit_map = {
            range(35, 37): "Kick",
            range(38, 41): "Snare",
            range(42, 45): "HiHat",
            range(46, 47): "Ride",
            range(41, 42): "Tom",
            range(45, 46): "Tom",
            range(48, 54): "Tom",
            range(54, 80): "Percussion"
        }
        
        pieces = set()
        for event in midi_events:
            for pitch_range, piece in kit_map.items():
                if event['pitch'] in pitch_range:
                    pieces.add(piece)
                    break
        return pieces
    
    def _calculate_style_blend(self) -> Dict[str, float]:
        """Calculate style blend from influence parameters"""
        total = sum([
            self.current_controls.rock_influence,
            self.current_controls.jazz_influence,
            self.current_controls.funk_influence,
            self.current_controls.latin_influence,
            self.current_controls.electronic_influence,
            self.current_controls.metal_influence,
            self.current_controls.hiphop_influence,
            self.current_controls.world_influence
        ]) or 1.0
        
        return {
            'rock': self.current_controls.rock_influence / total,
            'jazz': self.current_controls.jazz_influence / total,
            'funk': self.current_controls.funk_influence / total,
            'latin': self.current_controls.latin_influence / total,
            'electronic': self.current_controls.electronic_influence / total,
            'metal': self.current_controls.metal_influence / total,
            'hiphop': self.current_controls.hiphop_influence / total,
            'world': self.current_controls.world_influence / total
        }
    
    def play_version(self, version_id: str):
        """Play a version through MIDI output"""
        if not self.midi_out:
            logger.error("MIDI output not available")
            return
        version = self.get_version(version_id)
        for event in version.midi_data:
            msg = mido.Message(
                'note_on',
                note=event['pitch'],
                velocity=int(event['velocity'] * 127),  # Scale 0.0-1.0 to 0-127
                time=event['start_time'] / 1_000_000
            )
            self.midi_out.send(msg)
    
    def get_version(self, version_id: str) -> DrumGenerationVersion:
        """Get version by ID"""
        for version in self.version_history:
            if version.version_id == version_id:
                return version
        raise ValueError(f"Version not found: {version_id}")
    
    def compare_versions(self, version_id_a: str, version_id_b: str):
        """Set versions for A/B comparison"""
        self.comparison_state = (version_id_a, version_id_b)
    
    def get_comparison_data(self) -> Tuple[DrumGenerationVersion, DrumGenerationVersion]:
        """Get comparison versions"""
        if not self.comparison_state:
            raise RuntimeError("No versions set for comparison")
        return (
            self.get_version(self.comparison_state[0]),
            self.get_version(self.comparison_state[1])
        )
    
    def export_session(self, session_id: str, filename: str):
        """Export session to JSON file"""
        session = self.sessions[session_id]
        session_data = {
            'metadata': session,
            'versions': [v.to_dict() for v in self.version_history if v.version_id in session['versions']]
        }
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2)
        logger.info(f"Exported session {session_id} to {filename}")
    
    def import_session(self, filename: str) -> str:
        """Import session from JSON file"""
        try:
            with open(filename, 'r') as f:
                session_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to import session from {filename}: {e}")
            raise
        
        session_id = session_data['metadata']['created_at']
        self.sessions[session_id] = session_data['metadata']
        
        for v_data in session_data['versions']:
            version = DrumGenerationVersion(
                version_id=v_data['version_id'],
                timestamp=v_data['timestamp'],
                control_snapshot=v_data['control_snapshot'],
                midi_data=v_data.get('midi_data', []),
                metadata=v_data['metadata'],
                tags=v_data['tags'],
                rating=v_data['rating'],
                is_favorite=v_data['is_favorite'],
                notes=v_data['notes'],
                groove_feel=v_data['groove_feel'],
                pattern_complexity=v_data['pattern_complexity'],
                fill_density=v_data['fill_density'],
                kit_pieces_used=set(v_data['kit_pieces_used']),
                style_blend=v_data['style_blend'],
                polyrhythmic_elements=v_data['polyrhythmic_elements'],
                m4l_parameters=v_data['m4l_parameters']
            )
            self.version_history.append(version)
        
        logger.info(f"Imported session {session_id} from {filename}")
        return session_id
    
    def update_m4l_parameter(self, param_name: str, value: float):
        """Update a Max4Live parameter value"""
        if param_name not in M4L_PARAMETERS:
            raise ValueError(f"Invalid parameter: {param_name}")
        self.current_m4l_params[param_name] = value
        logger.debug(f"Updated M4L parameter {param_name} to {value}")
    
    def get_m4l_parameters(self) -> Dict[str, float]:
        """Get current Max4Live parameters"""
        return self.current_m4l_params.copy()
    
    def save_m4l_preset(self, preset_name: str):
        """Save current parameters as a preset"""
        self.sessions.setdefault('presets', {})[preset_name] = self.current_m4l_params.copy()
        logger.info(f"Saved M4L preset: {preset_name}")
    
    def load_m4l_preset(self, preset_name: str):
        """Load a saved parameter preset"""
        if preset_name in self.sessions.get('presets', {}):
            self.current_m4l_params = self.sessions['presets'][preset_name].copy()
            logger.info(f"Loaded M4L preset: {preset_name}")

# Example usage
if __name__ == "__main__":
    async def main():
        analyzer_data = {"tempo": 120, "time_signature": "4/4"}
        drum_engine = AlgorithmicDrummaroo(analyzer_data)
        manager = DrumRooGenerationManager(drum_engine)
        
        session_id = manager.start_new_session("Demo Session")
        manager.update_m4l_parameter('rock_influence', 0.8)
        manager.update_m4l_parameter('kick_density', 0.7)
        
        verse_version = await manager.generate_new_version("verse", 8)
        chorus_version = await manager.generate_new_version("chorus", 8)
        
        manager.compare_versions(verse_version.version_id, chorus_version.version_id)
        v1, v2 = manager.get_comparison_data()
        
        print(f"Verse complexity: {v1.pattern_complexity:.2f}")
        print(f"Chorus complexity: {v2.pattern_complexity:.2f}")
        print(f"Verse kit pieces: {', '.join(v1.kit_pieces_used)}")
        
        manager.export_session(session_id, "demo_session.json")
    
    asyncio.run(main())