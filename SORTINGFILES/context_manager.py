#!/usr/bin/env python3
"""
MuseAroo Analysis Context Manager
Manages analysis context and inter-plugin data sharing throughout the pipeline
"""

import os
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
import weakref


class AnalysisPhase(Enum):
    """Pipeline phases"""
    INITIALIZED = "initialized"
    ANALYZING = "analyzing"
    GENERATING = "generating"
    ARRANGING = "arranging"
    EXPORTING = "exporting"
    COMPLETE = "complete"
    ERROR = "error"


class DataQuality(Enum):
    """Data quality levels"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


@dataclass
class PluginResult:
    """Individual plugin execution result"""
    plugin_name: str
    status: str
    execution_time: float
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    quality: DataQuality = DataQuality.UNKNOWN
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['quality'] = self.quality.value
        return result


@dataclass
class TimingContext:
    """Timing-specific context data"""
    total_duration_seconds: float = 0.0
    leading_silence_seconds: float = 0.0
    trailing_silence_seconds: float = 0.0
    sample_rate: int = 44100
    tempo: float = 120.0
    time_signature: tuple = (4, 4)
    beat_positions: List[float] = field(default_factory=list)
    bar_positions: List[float] = field(default_factory=list)
    timing_precision: str = "microsecond"
    timing_source: str = "PrecisionTimingHandler"
    
    def get_total_bars(self) -> int:
        """Calculate total number of bars"""
        if self.tempo <= 0:
            return 1
        
        beats_per_bar = self.time_signature[0]
        seconds_per_beat = 60.0 / self.tempo
        seconds_per_bar = seconds_per_beat * beats_per_bar
        
        active_duration = self.total_duration_seconds - self.leading_silence_seconds
        return max(1, int(active_duration / seconds_per_bar))


@dataclass
class MusicalContext:
    """Musical analysis context"""
    key: str = "C major"
    tempo: float = 120.0
    time_signature: tuple = (4, 4)
    style: str = "unknown"
    style_confidence: float = 0.0
    energy_level: float = 0.5
    complexity_level: float = 0.5
    emotional_valence: float = 0.0
    emotional_arousal: float = 0.5
    harmonic_complexity: float = 0.5
    rhythmic_complexity: float = 0.5
    form_complexity: float = 0.5
    instrumentation: List[str] = field(default_factory=list)
    missing_instruments: List[str] = field(default_factory=list)
    detected_sections: List[Dict[str, Any]] = field(default_factory=list)
    chord_progression: List[str] = field(default_factory=list)


@dataclass
class FileContext:
    """File-specific context"""
    input_path: str = ""
    file_type: str = "unknown"
    file_size_bytes: int = 0
    file_format: str = ""
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    bit_depth: Optional[int] = None
    duration_seconds: float = 0.0
    is_stereo: bool = False
    has_metadata: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    creation_time: Optional[datetime] = None
    last_modified: Optional[datetime] = None


class AnalysisContext:
    """
    Central context manager for analysis pipeline.
    Manages data sharing between plugins and tracks pipeline state.
    """
    
    def __init__(self, 
                 input_path: str,
                 output_dir: str = "reports",
                 session_id: Optional[str] = None,
                 real_time: bool = False):
        
        self.session_id = session_id or f"session_{int(time.time())}"
        self.input_path = input_path
        self.output_dir = output_dir
        self.real_time = real_time
        self.created_at = datetime.now()
        
        # Pipeline state
        self.phase = AnalysisPhase.INITIALIZED
        self.progress = 0.0
        self.error_message: Optional[str] = None
        
        # Context data
        self.timing_context = TimingContext()
        self.musical_context = MusicalContext()
        self.file_context = FileContext(input_path=input_path)
        
        # Plugin results storage
        self.plugin_results: Dict[str, PluginResult] = {}
        self.execution_order: List[str] = []
        
        # Shared data between plugins
        self.shared_data: Dict[str, Any] = {}
        self.feature_cache: Dict[str, Any] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Listeners for real-time updates
        self._listeners: Set[weakref.ref] = set()
        
        # Logger
        self.logger = logging.getLogger(f"AnalysisContext.{self.session_id}")
        
        self.logger.info(f"Created analysis context for {input_path}")
    
    def set_phase(self, phase: AnalysisPhase, progress: Optional[float] = None):
        """Update pipeline phase"""
        with self._lock:
            self.phase = phase
            if progress is not None:
                self.progress = max(0.0, min(1.0, progress))
            
            self.logger.info(f"Phase changed to {phase.value} (progress: {self.progress:.1%})")
            self._notify_listeners('phase_change', {
                'phase': phase.value,
                'progress': self.progress
            })
    
    def add_plugin_result(self, plugin_name: str, result: PluginResult):
        """Add result from plugin execution"""
        with self._lock:
            self.plugin_results[plugin_name] = result
            if plugin_name not in self.execution_order:
                self.execution_order.append(plugin_name)
            
            self.logger.info(f"Added result from plugin: {plugin_name} ({result.status})")
            
            # Update shared data if plugin provides it
            if 'shared_data' in result.data:
                self.shared_data.update(result.data['shared_data'])
            
            self._notify_listeners('plugin_complete', {
                'plugin_name': plugin_name,
                'result': result.to_dict()
            })
    
    def update_from_plugin_result(self, plugin_name: str, result: Dict[str, Any]):
        """Update context from plugin result (backward compatibility)"""
        
        plugin_result = PluginResult(
            plugin_name=plugin_name,
            status=result.get('status', 'unknown'),
            execution_time=result.get('execution_time', 0.0),
            data=result.get('data', {}),
            error=result.get('error'),
            warnings=result.get('warnings', [])
        )
        
        self.add_plugin_result(plugin_name, plugin_result)
        
        # Update context fields based on plugin data
        data = result.get('data', {})
        
        # Update timing context
        if 'tempo' in data:
            self.timing_context.tempo = data['tempo']
            self.musical_context.tempo = data['tempo']
        
        if 'time_signature' in data:
            self.timing_context.time_signature = data['time_signature']
            self.musical_context.time_signature = data['time_signature']
        
        if 'duration' in data:
            self.timing_context.total_duration_seconds = data['duration']
        
        # Update musical context
        if 'key' in data:
            self.musical_context.key = data['key']
        
        if 'style' in data:
            self.musical_context.style = data['style']
        
        if 'energy' in data:
            self.musical_context.energy_level = data['energy']
        
        if 'complexity' in data:
            self.musical_context.complexity_level = data['complexity']
        
        if 'missing_instruments' in data:
            self.musical_context.missing_instruments = data['missing_instruments']
        
        if 'instrumentation' in data:
            self.musical_context.instrumentation = data['instrumentation']
    
    def get_plugin_context(self) -> Dict[str, Any]:
        """Get context data for plugin execution"""
        with self._lock:
            return {
                'session_id': self.session_id,
                'input_path': self.input_path,
                'output_dir': self.output_dir,
                'timing': asdict(self.timing_context),
                'musical': asdict(self.musical_context),
                'file': asdict(self.file_context),
                'shared_data': self.shared_data.copy(),
                'previous_results': {
                    name: result.to_dict() 
                    for name, result in self.plugin_results.items()
                },
                'execution_order': self.execution_order.copy(),
                'phase': self.phase.value,
                'progress': self.progress
            }
    
    def get_feature(self, feature_name: str, default: Any = None) -> Any:
        """Get cached feature value"""
        return self.feature_cache.get(feature_name, default)
    
    def set_feature(self, feature_name: str, value: Any):
        """Cache feature value"""
        with self._lock:
            self.feature_cache[feature_name] = value
    
    def get_plugin_result(self, plugin_name: str) -> Optional[PluginResult]:
        """Get result from specific plugin"""
        return self.plugin_results.get(plugin_name)
    
    def has_plugin_run(self, plugin_name: str) -> bool:
        """Check if plugin has been executed"""
        return plugin_name in self.plugin_results
    
    def get_successful_plugins(self) -> List[str]:
        """Get list of successfully executed plugins"""
        return [
            name for name, result in self.plugin_results.items()
            if result.status == 'success'
        ]
    
    def get_failed_plugins(self) -> List[str]:
        """Get list of failed plugins"""
        return [
            name for name, result in self.plugin_results.items()
            if result.status == 'error'
        ]
    
    def get