#!/usr/bin/env python3
import mido
import asyncio
from src.engines.drummaroo import AlgorithmicDrummaroo
from src.ui.controls.drummaroo_controls import DrummarooUIControls
from src.core.precision_timing_handler import TimingMetadata

class AbletonDrummarooAdapter:
    def __init__(self, ableton_transport):
        self.ableton = ableton_transport
        self.midi_out = mido.open_output('Drummaroo Output', virtual=True)
        self.engine = None
        self.ui_params = DrummarooUIControls()
        
    def update_params(self, params: dict):
        """Map Ableton parameters to Drummaroo UI controls"""
        mapping = {
            'groove_intensity': 'GrooveIntensity',
            'pattern_complexity': 'Complexity',
            'swing_amount': 'Swing',
            'shuffle_feel': 'ShuffleFeel',
            'syncopation_level': 'Syncopation',
            'polyrhythm_amount': 'Polyrhythm',
            'humanization': 'Humanization',
            'timing_tightness': 'TimingTightness',
            'cross_rhythm_amount': 'CrossRhythm',
            'rhythmic_displacement': 'RhythmicDisplacement',
            'kick_density': 'KickDensity',
            'snare_density': 'SnareDensity',
            'hihat_density': 'HiHatDensity',
            'cymbal_density': 'CymbalDensity',
            'percussion_density': 'PercDensity',
            'dynamic_range': 'DynamicRange',
            'ghost_note_density': 'GhostNotes',
            'accent_strength': 'AccentStrength',
            'velocity_variation': 'VelocityVar',
            'micro_timing_amount': 'MicroTiming',
            'hihat_openness': 'HiHatOpenness',
            'fill_frequency': 'FillFrequency',
            'fill_complexity': 'FillComplexity',
            'fill_length': 'FillLength',
            'transition_smoothness': 'TransitionSmooth',
            'rock_influence': 'RockInfluence',
            'jazz_influence': 'JazzInfluence',
            'funk_influence': 'FunkInfluence',
            'latin_influence': 'LatinInfluence',
            'electronic_influence': 'ElectronicInfluence',
            'metal_influence': 'MetalInfluence',
            'hiphop_influence': 'HipHopInfluence',
            'world_influence': 'WorldInfluence',
            'kick_variation': 'KickVariation',
            'snare_variation': 'SnareVariation',
            'ride_vs_hihat': 'RideVsHiHat',
            'tom_usage': 'TomUsage',
            'percussion_variety': 'PercVariety',
            'random_seed': 'RandomSeed',
            'generation_iterations': 'GenerationIter',
            'complexity_variance': 'ComplexityVar',
            'density_fluctuation': 'DensityFluct',
            'style_blend_weight': 'StyleBlend',
            'odd_time_tendency': 'OddTimeTendency',
            'metric_modulation': 'MetricModulation',
            'limb_independence': 'LimbIndependence',
            'stamina_simulation': 'StaminaSimulation',
            'technique_precision': 'TechniquePrecision',
            'room_presence': 'RoomPresence',
            'compression_amount': 'Compression',
            'eq_brightness': 'EQBrightness',
            'stereo_width': 'StereoWidth',
        }
        
        for py_param, ableton_param in mapping.items():
            if ableton_param in params:
                setattr(self.ui_params, py_param, params[ableton_param])
                
        if self.engine:
            self.engine.update_ui_parameters(self.ui_params)
    
    def get_timing_metadata(self) -> TimingMetadata:
        """Convert Ableton transport to precision timing"""
        return TimingMetadata(
            tempo=self.ableton.tempo,
            time_signature=(self.ableton.time_signature_numerator, 
                           self.ableton.time_signature_denominator),
            total_duration_seconds=self.ableton.song_length,
            leading_silence_seconds=0.5
        )
    
    async def generate_for_section(self, section: str, bars: int):
        """Generate drums for a section"""
        if not self.engine:
            self.engine = AlgorithmicDrummaroo(
                analyzer_data={
                    'tempo': self.ableton.tempo,
                    'time_signature': f"{self.ableton.time_signature_numerator}/{self.ableton.time_signature_denominator}"
                },
                ui_params=self.ui_params
            )
        
        beats_per_bar = self.ableton.time_signature_numerator
        microseconds_per_bar = 60_000_000 / self.ableton.tempo * beats_per_bar
        length_microseconds = int(bars * microseconds_per_bar)
        
        drum_events = await self.engine.generate_drums(section, length_microseconds)
        self._send_to_ableton(drum_events)
    
    def _send_to_ableton(self, events):
        for event in events:
            msg = mido.Message(
                'note_on',
                note=event['pitch'],
                velocity=event['velocity'],
                time=event['start_time'] / 1_000_000
            )
            self.midi_out.send(msg)