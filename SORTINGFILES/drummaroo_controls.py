#!/usr/bin/env python3
"""
Drummaroo Control Parameters v1.0
=================================
Defines all UI control parameters for the Algorithmic Drummaroo Engine.
Contains 50+ parameters across 8 categories with full type annotations.
"""

from dataclasses import dataclass, field
from typing import Dict, List
from src.engines.drummaroo import DrumAlgorithm  # Import for algorithm_weights


@dataclass
class DrummarooUIControls:
    """
    Comprehensive control parameters for drum pattern generation.
    Organized into 8 categories with 50+ parameters.
    """

    # --------------------------
    # 1. Groove Parameters (10)
    # --------------------------
    groove_intensity: float = 0.5
    pattern_complexity: float = 0.5
    swing_amount: float = 0.0
    shuffle_feel: float = 0.0
    syncopation_level: float = 0.3
    polyrhythm_amount: float = 0.0
    humanization: float = 0.7
    timing_tightness: float = 0.8
    cross_rhythm_amount: float = 0.0  # Added from drummaroo.py
    rhythmic_displacement: float = 0.0  # Added from drummaroo.py

    # ---------------------------
    # 2. Instrument Density (5)
    # ---------------------------
    kick_density: float = 0.5
    snare_density: float = 0.5
    hihat_density: float = 0.6
    cymbal_density: float = 0.3
    percussion_density: float = 0.2

    # ---------------------------------
    # 3. Dynamics & Articulation (6)
    # ---------------------------------
    dynamic_range: float = 0.6
    ghost_note_density: float = 0.3
    accent_strength: float = 0.6
    velocity_variation: float = 0.4
    micro_timing_amount: float = 0.0
    hihat_openness: float = 0.3

    # ---------------------------
    # 4. Fills & Transitions (4)
    # ---------------------------
    fill_frequency: float = 0.3
    fill_complexity: float = 0.5
    fill_length: float = 0.5
    transition_smoothness: float = 0.7

    # --------------------------
    # 5. Style Influences (8)
    # --------------------------
    rock_influence: float = 0.0
    jazz_influence: float = 0.0
    funk_influence: float = 0.0
    latin_influence: float = 0.0
    electronic_influence: float = 0.0
    metal_influence: float = 0.0
    hiphop_influence: float = 0.0
    world_influence: float = 0.0

    # ----------------------
    # 6. Kit Settings (5)
    # ----------------------
    kick_variation: float = 0.5
    snare_variation: float = 0.5
    ride_vs_hihat: float = 0.2
    tom_usage: float = 0.4
    percussion_variety: float = 0.3

    # -------------------------
    # 7. Advanced Controls (7)
    # -------------------------
    random_seed: int = 42
    generation_iterations: int = 3
    complexity_variance: float = 0.2
    density_fluctuation: float = 0.3
    style_blend_weight: float = 0.5
    odd_time_tendency: float = 0.0  # Added from drummaroo.py
    metric_modulation: float = 0.0  # Added from drummaroo.py

    # -------------------------
    # 8. Performance & Mix (6)
    # -------------------------
    limb_independence: float = 0.7  # Added from drummaroo.py
    stamina_simulation: float = 0.0  # Added from drummaroo.py
    technique_precision: float = 0.8  # Added from drummaroo.py
    room_presence: float = 0.5  # Added from drummaroo.py
    compression_amount: float = 0.3  # Added from drummaroo.py
    eq_brightness: float = 0.5  # Added from drummaroo.py
    stereo_width: float = 0.7  # Added from drummaroo.py

    # -------------------------
    # Algorithm Weights
    # -------------------------
    algorithm_weights: Dict[DrumAlgorithm, float] = field(default_factory=lambda: {algo: 0.5 for algo in DrumAlgorithm})

    def to_dict(self) -> Dict[str, float]:
        """Convert parameters to dictionary for serialization"""
        return {
            # Groove Parameters
            "groove_intensity": self.groove_intensity,
            "pattern_complexity": self.pattern_complexity,
            "swing_amount": self.swing_amount,
            "shuffle_feel": self.shuffle_feel,
            "syncopation_level": self.syncopation_level,
            "polyrhythm_amount": self.polyrhythm_amount,
            "humanization": self.humanization,
            "timing_tightness": self.timing_tightness,
            "cross_rhythm_amount": self.cross_rhythm_amount,
            "rhythmic_displacement": self.rhythmic_displacement,
            # Instrument Density
            "kick_density": self.kick_density,
            "snare_density": self.snare_density,
            "hihat_density": self.hihat_density,
            "cymbal_density": self.cymbal_density,  # Fixed typo
            "percussion_density": self.percussion_density,
            # Dynamics & Articulation
            "dynamic_range": self.dynamic_range,
            "ghost_note_density": self.ghost_note_density,
            "accent_strength": self.accent_strength,
            "velocity_variation": self.velocity_variation,
            "micro_timing_amount": self.micro_timing_amount,
            "hihat_openness": self.hihat_openness,
            # Fills & Transitions
            "fill_frequency": self.fill_frequency,
            "fill_complexity": self.fill_complexity,
            "fill_length": self.fill_length,
            "transition_smoothness": self.transition_smoothness,
            # Style Influences
            "rock_influence": self.rock_influence,
            "jazz_influence": self.jazz_influence,
            "funk_influence": self.funk_influence,
            "latin_influence": self.latin_influence,
            "electronic_influence": self.electronic_influence,
            "metal_influence": self.metal_influence,
            "hiphop_influence": self.hiphop_influence,
            "world_influence": self.world_influence,
            # Kit Settings
            "kick_variation": self.kick_variation,
            "snare_variation": self.snare_variation,
            "ride_vs_hihat": self.ride_vs_hihat,
            "tom_usage": self.tom_usage,
            "percussion_variety": self.percussion_variety,
            # Advanced Controls
            "random_seed": self.random_seed,
            "generation_iterations": self.generation_iterations,
            "complexity_variance": self.complexity_variance,
            "density_fluctuation": self.density_fluctuation,
            "style_blend_weight": self.style_blend_weight,
            "odd_time_tendency": self.odd_time_tendency,
            "metric_modulation": self.metric_modulation,
            # Performance & Mix
            "limb_independence": self.limb_independence,
            "stamina_simulation": self.stamina_simulation,
            "technique_precision": self.technique_precision,
            "room_presence": self.room_presence,
            "compression_amount": self.compression_amount,
            "eq_brightness": self.eq_brightness,
            "stereo_width": self.stereo_width,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]):
        """Create controls from dictionary"""
        return cls(**data)

    def get_parameter_metadata(self) -> List[Dict]:
        """Get metadata for all parameters"""
        return [
            # Groove Parameters
            {
                "name": "Groove Intensity",
                "key": "groove_intensity",
                "min": 0.0,
                "max": 1.0,
                "type": "float",
                "default": 0.5,
            },
            {
                "name": "Pattern Complexity",
                "key": "pattern_complexity",
                "min": 0.0,
                "max": 1.0,
                "type": "float",
                "default": 0.5,
            },
            {
                "name": "Swing Amount",
                "key": "swing_amount",
                "min": 0.0,
                "max": 1.0,
                "type": "float",
                "default": 0.0,
            },
            {
                "name": "Shuffle Feel",
                "key": "shuffle_feel",
                "min": 0.0,
                "max": 1.0,
                "type": "float",
                "default": 0.0,
            },
            {
                "name": "Syncopation",
                "key": "syncopation_level",
                "min": 0.0,
                "max": 1.0,
                "type": "float",
                "default": 0.3,
            },
            {
                "name": "Polyrhythm",
                "key": "polyrhythm_amount",
                "min": 0.0,
                "max": 1.0,
                "type": "float",
                "default": 0.0,
            },
            {
                "name": "Humanization",
                "key": "humanization",
                "min": 0.0,
                "max": 1.0,
                "type": "float",
                "default": 0.7,
            },
            {
                "name": "Timing Tightness",
                "key": "timing_tightness",
                "min": 0.0,
                "max": 1.0,
                "type": "float",
                "default": 0.8,
            },
            {
                "name": "Cross Rhythm Amount",
                "key": "cross_rhythm_amount",
                "min": 0.0,
                "max": 1.0,
                "type": "float",
                "default": 0.0,
            },
            {
                "name": "Rhythmic Displacement",
                "key": "rhythmic_displacement",
                "min": 0.0,
                "max": 1.0,
                "type": "float",
                "default": 0.0,
            },
            # Instrument Density
            {
                "name": "Kick Density",
                "key": "kick_density",
                "min": 0.0,
                "max": 1.0,
                "type": "float",
                "default": 0.5,
            },
            {
                "name": "Snare Density",
                "key": "snare_density",
                "min": 0.0,
                "max": 1.0,
                "type": "float",
                "default": 0.5,
            },
            {
                "name": "HiHat Density",
                "key": "hihat_density",
                "min": 0.0,
                "max": 1.0,
                "type": "float",
                "default": 0.6,
            },
            {
                "name": "Cymbal Density",
                "key": "cymbal_density",
                "min": 0.0,
                "max": 1.0,
                "type": "float",
                "default": 0.3,
            },
            {
                "name": "Perc Density",
                "key": "percussion_density",
                "min": 0.0,
                "max": 1.0,
                "type": "float",
                "default": 0.2,
            },
            # Dynamics & Articulation
            {
                "name": "Dynamic Range",
                "key": "dynamic_range",
                "min": 0.0,
                "max": 1.0,
                "type": "float",
                "default": 0.6,
            },
            {
                "name": "Ghost Notes",
                "key": "ghost_note_density",
                "min": 0.0,
                "max": 1.0,
                "type": "float",
                "default": 0.3,
            },
            {
                "name": "Accent Strength",
                "key": "accent_strength",
                "min": 0.0,
                "max": 1.0,
                "type": "float",
                "default": 0.6,
            },
            {
                "name": "Velocity Var",
                "key": "velocity_variation",
                "min": 0.0,
                "max": 1.0,
                "type": "float",
                "default": 0.4,
            },
            {
                "name": "Micro Timing",
                "key": "micro_timing_amount",
                "min": 0.0,
                "max": 1.0,
                "type": "float",
                "default": 0.0,
            },
            {
                "name": "HiHat Openness",
                "key": "hihat_openness",
                "min": 0.0,
                "max": 1.0,
                "type": "float",
                "default": 0.3,
            },
            # Fills & Transitions
            {
                "name": "Fill Frequency",
                "key": "fill_frequency",
                "min": 0.0,
                "max": 1.0,
                "type": "float",
                "default": 0.3,
            },
            {
                "name": "Fill Complexity",
                "key": "fill_complexity",
                "min": 0.0,
                "max": 1.0,
                "type": "float",
                "default": 0.5,
            },
            {
                "name": "Fill Length",
                "key": "fill_length",
                "min": 0.0,
                "max": 1.0,
                "type": "float",
                "default": 0.5,
            },
            {
                "name": "Transition Smooth",
                "key": "transition_smoothness",
                "min": 0.0,
                "max": 1.0,
                "type": "float",
                "default": 0.7,
            },
            # Style Influences
            {
                "name": "Rock Influence",
                "key": "rock_influence",
                "min": 0.0,
                "max": 1.0,
                "type": "float",
                "default": 0.0,
            },
            {
                "name": "Jazz Influence",
                "key": "jazz_influence",
                "min": 0.0,
                "max": 1.0,
                "type": "float",
                "default": 0.0,
            },
            {
                "name": "Funk Influence",
                "key": "funk_influence",
                "min": 0.0,
                "max": 1.0,
                "type": "float",
                "default": 0.0,
            },
            {
                "name": "Latin Influence",
                "key": "latin_influence",
                "min": 0.0,
                "max": 1.0,
                "type": "float",
                "default": 0.0,
            },
            {
                "name": "Electronic Influence",
                "key": "electronic_influence",
                "min": 0.0,
                "max": 1.0,
                "type": "float",
                "default": 0.0,
            },
            {
                "name": "Metal Influence",
                "key": "metal_influence",
                "min": 0.0,
                "max": 1.0,
                "type": "float",
                "default": 0.0,
            },
            {
                "name": "HipHop Influence",
                "key": "hiphop_influence",
                "min": 0.0,
                "max": 1.0,
                "type": "float",
                "default": 0.0,
            },
            {
                "name": "World Influence",
                "key": "world_influence",
                "min": 0.0,
                "max": 1.0,
                "type": "float",
                "default": 0.0,
            },
            # Kit Settings
            {
                "name": "Kick Variation",
                "key": "kick_variation",
                "min": 0.0,
                "max": 1.0,
                "type": "float",
                "default": 0.5,
            },
            {
                "name": "Snare Variation",
                "key": "snare_variation",
                "min": 0.0,
                "max": 1.0,
                "type": "float",
                "default": 0.5,
            },
            {
                "name": "Ride vs HiHat",
                "key": "ride_vs_hihat",
                "min": 0.0,
                "max": 1.0,
                "type": "float",
                "default": 0.2,
            },
            {
                "name": "Tom Usage",
                "key": "tom_usage",
                "min": 0.0,
                "max": 1.0,
                "type": "float",
                "default": 0.4,
            },  # Fixed syntax error
            {
                "name": "Perc Variety",
                "key": "percussion_variety",
                "min": 0.0,
                "max": 1.0,
                "type": "float",
                "default": 0.3,
            },
            # Advanced Controls
            {
                "name": "Random Seed",
                "key": "random_seed",
                "min": 0,
                "max": 10000,
                "type": "int",
                "default": 42,
            },
            {
                "name": "Generation Iter",
                "key": "generation_iterations",
                "min": 1,
                "max": 10,
                "type": "int",
                "default": 3,
            },
            {
                "name": "Complexity Var",
                "key": "complexity_variance",
                "min": 0.0,
                "max": 1.0,
                "type": "float",
                "default": 0.2,
            },
            {
                "name": "Density Fluct",
                "key": "density_fluctuation",
                "min": 0.0,
                "max": 1.0,
                "type": "float",
                "default": 0.3,
            },
            {
                "name": "Style Blend",
                "key": "style_blend_weight",
                "min": 0.0,
                "max": 1.0,
                "type": "float",
                "default": 0.5,
            },
            {
                "name": "Odd Time Tendency",
                "key": "odd_time_tendency",
                "min": 0.0,
                "max": 1.0,
                "type": "float",
                "default": 0.0,
            },
            {
                "name": "Metric Modulation",
                "key": "metric_modulation",
                "min": 0.0,
                "max": 1.0,
                "type": "float",
                "default": 0.0,
            },
            # Performance & Mix
            {
                "name": "Limb Independence",
                "key": "limb_independence",
                "min": 0.0,
                "max": 1.0,
                "type": "float",
                "default": 0.7,
            },
            {
                "name": "Stamina Simulation",
                "key": "stamina_simulation",
                "min": 0.0,
                "max": 1.0,
                "type": "float",
                "default": 0.0,
            },
            {
                "name": "Technique Precision",
                "key": "technique_precision",
                "min": 0.0,
                "max": 1.0,
                "type": "float",
                "default": 0.8,
            },
            {
                "name": "Room Presence",
                "key": "room_presence",
                "min": 0.0,
                "max": 1.0,
                "type": "float",
                "default": 0.5,
            },
            {
                "name": "Compression Amount",
                "key": "compression_amount",
                "min": 0.0,
                "max": 1.0,
                "type": "float",
                "default": 0.3,
            },
            {
                "name": "EQ Brightness",
                "key": "eq_brightness",
                "min": 0.0,
                "max": 1.0,
                "type": "float",
                "default": 0.5,
            },
            {
                "name": "Stereo Width",
                "key": "stereo_width",
                "min": 0.0,
                "max": 1.0,
                "type": "float",
                "default": 0.7,
            },
        ]

    def __post_init__(self):
        """Ensure algorithm_weights is initialized"""
        if not self.algorithm_weights:
            self.algorithm_weights = {algo: 0.5 for algo in DrumAlgorithm}


# Example usage
if __name__ == "__main__":
    # Create default controls
    controls = DrummarooUIControls()

    # Access parameters
    print(f"Default groove intensity: {controls.groove_intensity}")

    # Modify parameters
    controls.rock_influence = 0.8
    controls.kick_density = 0.7

    # Serialize to dictionary
    params_dict = controls.to_dict()
    print("Serialized parameters:", params_dict)

    # Create from dictionary
    new_controls = DrummarooUIControls.from_dict(params_dict)
    print(f"Rock influence: {new_controls.rock_influence}")

    # Get parameter metadata for UI
    metadata = controls.get_parameter_metadata()
    print(f"Found {len(metadata)} parameters")