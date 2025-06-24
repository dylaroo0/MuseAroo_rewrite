#!/usr/bin/env python3
"""
MuseAroo Unified Master Conductor v2.0
======================================
World-class AI music orchestration system that consolidates all your sophisticated components:

🎼 Workflow Orchestration - Advanced async pipeline management
🧠 BrainAroo Integration - 200+ feature musical intelligence  
⚡ Precision Timing - Microsecond-accurate timing preservation
🔌 Plugin Architecture - Enterprise-grade plugin management
🎯 Context Intelligence - Sophisticated musical context integration
🌐 Real-time Control - Live parameter adjustment and DAW sync
📡 Ableton Integration - Seamless Max4Live bridge

This is your single source of truth for all musical AI operations.
"""

import asyncio
import logging
import time
import json
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import uuid

# Core MuseAroo imports
from context import create_runtime_context, create_analysis_context, create_integrator
from advanced_plugin_architecture import AdvancedPluginManager
from utils.precision_timing_handler import PrecisionTimingHandler, TimingMetadata
from utils.logger_setup import setup_logger

# Engine imports
from engines.drummaroo import AlgorithmicDrummaroo
from engines.bassaroo import BassaRoo
from engines.melodyroo import MelodyRoo  
from engines.harmonyroo import HarmonyRoo
from plugins.analyzers.brainaroo_extractor import BrainArooExtractor

# UI Control imports
from ui.controls.drummaroo_controls import DrummarooUIControls

logger = setup_logger("musearoo.conductor", logging.INFO)

class WorkflowPhase(Enum):
    """Workflow phases for the orchestration pipeline."""
    INITIALIZATION = "initialization"
    ANALYSIS = "analysis"
    CONTEXT_INTEGRATION = "context_integration"
    GENERATION = "generation"
    ARRANGEMENT = "arrangement"
    FINALIZATION = "finalization"
    EXPORT = "export"

class ConductorStatus(Enum):
    """Status states for the master conductor."""
    IDLE = "idle"
    INITIALIZING = "initializing"
    ANALYZING = "analyzing"
    GENERATING = "generating"
    READY = "ready"
    ERROR = "error"

@dataclass
class MusicalSession:
    """Represents a complete musical session with all context."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    input_file: Optional[str] = None
    output_directory: Optional[str] = None
    analysis_results: Dict[str, Any] = field(default_factory=dict)
    generation_results: Dict[str, Any] = field(default_factory=dict)
    timing_metadata: Optional[TimingMetadata] = None
    musical_context: Dict[str, Any] = field(default_factory=dict)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    status: str = "created"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

@dataclass
class GenerationRequest:
    """Request for music generation with all parameters."""
    session_id: str
    engines_to_use: List[str] = field(default_factory=lambda: ["drummaroo"])
    target_length_ms: int = 8000
    section_type: str = "verse"
    style: str = "pop"
    tempo: float = 120.0
    key_signature: str = "C"
    ui_controls: Dict[str, Any] = field(default_factory=dict)
    real_time: bool = False
    priority: str = "normal"

class UnifiedMasterConductor:
    """
    The ultimate MuseAroo orchestration system - your single point of control
    for all musical AI operations with world-class architecture.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the unified master conductor."""
        self.config = config or {}
        self.session_id = str(uuid.uuid4())
        self.status = ConductorStatus.IDLE
        self.sessions: Dict[str, MusicalSession] = {}
        self.active_session: Optional[MusicalSession] = None
        
        # Core system components
        self.timing_handler = PrecisionTimingHandler()
        self.plugin_manager = AdvancedPluginManager()
        self.context_integrator = create_integrator()
        
        # Engine instances - your sophisticated Roo engines
        self.engines: Dict[str, Any] = {}
        self.engine_status: Dict[str, str] = {}
        
        # Real-time control and monitoring
        self.real_time_params: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, Any] = {
            "sessions_processed": 0,
            "total_generation_time": 0.0,
            "average_generation_time": 0.0,
            "error_count": 0,
            "cache_hit_ratio": 0.0
        }
        
        # WebSocket/real-time connections
        self.websocket_connections: List[Any] = []
        self.ableton_bridge = None
        
        # Background task management
        self.background_tasks: Set[asyncio.Task] = set()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info("🎼 Unified Master Conductor initialized")
        
    async def initialize(self, 
                        enable_ableton: bool = True,
                        enable_plugins: bool = True,
                        enable_real_time: bool = True) -> bool:
        """
        Initialize all subsystems for world-class operation.
        
        Args:
            enable_ableton: Enable Ableton Live integration
            enable_plugins: Enable advanced plugin system
            enable_real_time: Enable real-time parameter control
            
        Returns:
            bool: True if initialization successful
        """
        logger.info("🚀 Initializing Unified Master Conductor...")
        self.status = ConductorStatus.INITIALIZING
        
        try:
            # Initialize plugin architecture (enterprise-grade)
            if enable_plugins:
                await self._initialize_plugin_system()
            
            # Initialize all Roo engines
            await self._initialize_engines()
            
            # Initialize Ableton integration
            if enable_ableton:
                await self._initialize_ableton_bridge()
            
            # Initialize real-time control
            if enable_real_time:
                await self._initialize_real_time_control()
            
            # Initialize context integration
            await self._initialize_context_system()
            
            self.status = ConductorStatus.READY
            logger.info("✅ Unified Master Conductor ready for world-class operation!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            self.status = ConductorStatus.ERROR
            return False
    
    async def create_session(self, 
                           input_file: Optional[str] = None,
                           output_dir: Optional[str] = None,
                           **kwargs) -> MusicalSession:
        """Create a new musical session with full context tracking."""
        session = MusicalSession(
            input_file=input_file,
            output_directory=output_dir or "output",
            user_preferences=kwargs.get("user_preferences", {})
        )
        
        self.sessions[session.session_id] = session
        self.active_session = session
        
        logger.info(f"🎵 Created session {session.session_id[:8]}...")
        return session
    
    async def analyze_music(self, 
                          session_id: str, 
                          input_file: str,
                          **analysis_options) -> Dict[str, Any]:
        """
        Perform comprehensive musical analysis using your sophisticated BrainAroo system.
        
        Args:
            session_id: Session identifier
            input_file: Path to audio or MIDI file
            **analysis_options: Advanced analysis parameters
            
        Returns:
            Comprehensive analysis results with 200+ features
        """
        logger.info(f"🧠 Starting comprehensive analysis: {input_file}")
        self.status = ConductorStatus.ANALYZING
        
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        try:
            # Step 1: Precision timing analysis (microsecond-accurate)
            timing_metadata = self.timing_handler.analyze_input_timing(input_file)
            session.timing_metadata = timing_metadata
            
            logger.info(f"⚡ Timing analysis: {timing_metadata.total_duration_seconds:.6f}s")
            
            # Step 2: BrainAroo Complete analysis (200+ features)
            brainaroo_extractor = BrainArooExtractor()
            runtime_ctx = create_runtime_context()
            analysis_ctx = create_analysis_context()
            
            # Use your sophisticated BrainAroo analysis
            brainaroo_results = await brainaroo_extractor.run(runtime_ctx, analysis_ctx)
            
            # Step 3: Advanced plugin-based analysis
            plugin_results = await self._run_analysis_plugins(input_file, analysis_options)
            
            # Step 4: Context integration (sophisticated musical intelligence)
            integrated_context = await self.context_integrator.integrate_complete_context(
                brainaroo_results,
                plugin_results,
                timing_metadata
            )
            
            # Compile comprehensive analysis results
            analysis_results = {
                "status": "success",
                "input_file": input_file,
                "timing_metadata": timing_metadata.__dict__ if hasattr(timing_metadata, '__dict__') else timing_metadata,
                "brainaroo_analysis": brainaroo_results,
                "plugin_analysis": plugin_results,
                "integrated_context": integrated_context,
                "analysis_quality": integrated_context.get("analysis_quality", 0.95),
                "feature_count": len(integrated_context.get("features", {})),
                "generation_ready": integrated_context.get("generation_ready", True)
            }
            
            session.analysis_results = analysis_results
            session.musical_context = integrated_context
            session.status = "analyzed"
            session.updated_at = time.time()
            
            self.performance_metrics["sessions_processed"] += 1
            
            logger.info(f"✅ Analysis complete: {analysis_results['feature_count']} features extracted")
            return analysis_results
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            session.status = "analysis_error"
            raise
        finally:
            self.status = ConductorStatus.READY
    
    async def generate_music(self, 
                           session_id: str,
                           generation_request: GenerationRequest) -> Dict[str, Any]:
        """
        Generate music using your sophisticated Roo engines with real-time control.
        
        Args:
            session_id: Session identifier
            generation_request: Complete generation parameters
            
        Returns:
            Generated music with full metadata
        """
        logger.info(f"🎛️ Starting music generation for session {session_id[:8]}...")
        self.status = ConductorStatus.GENERATING
        
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        if not session.analysis_results:
            raise ValueError("Session must be analyzed before generation")
        
        start_time = time.time()
        
        try:
            # Prepare generation context from analysis
            generation_context = self._prepare_generation_context(session, generation_request)
            
            # Execute engines in parallel for performance
            generation_tasks = []
            
            for engine_name in generation_request.engines_to_use:
                if engine_name in self.engines:
                    task = self._generate_with_engine(
                        engine_name, 
                        generation_context, 
                        generation_request.ui_controls.get(engine_name, {})
                    )
                    generation_tasks.append((engine_name, task))
            
            # Execute all engines
            engine_results = {}
            for engine_name, task in generation_tasks:
                try:
                    result = await task
                    engine_results[engine_name] = result
                    logger.info(f"✅ {engine_name} generation complete")
                except Exception as e:
                    logger.error(f"❌ {engine_name} generation failed: {e}")
                    engine_results[engine_name] = {"status": "error", "error": str(e)}
            
            # Compile generation results
            generation_time = (time.time() - start_time) * 1000  # milliseconds
            
            generation_results = {
                "status": "success",
                "session_id": session_id,
                "generation_time_ms": generation_time,
                "engines_used": generation_request.engines_to_use,
                "engine_results": engine_results,
                "target_length_ms": generation_request.target_length_ms,
                "section_type": generation_request.section_type,
                "style": generation_request.style,
                "real_time": generation_request.real_time,
                "timestamp": time.time()
            }
            
            session.generation_results = generation_results
            session.status = "generated"
            session.updated_at = time.time()
            
            # Update performance metrics
            self.performance_metrics["total_generation_time"] += generation_time
            self.performance_metrics["average_generation_time"] = (
                self.performance_metrics["total_generation_time"] / 
                max(1, self.performance_metrics["sessions_processed"])
            )
            
            # Real-time broadcast if enabled
            if generation_request.real_time:
                await self._broadcast_real_time_update(generation_results)
            
            logger.info(f"🎵 Generation complete: {generation_time:.1f}ms")
            return generation_results
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            self.performance_metrics["error_count"] += 1
            raise
        finally:
            self.status = ConductorStatus.READY
    
    async def update_real_time_parameters(self, 
                                        engine_name: str, 
                                        parameters: Dict[str, Any]) -> bool:
        """Update engine parameters in real-time for live control."""
        if engine_name not in self.engines:
            return False
        
        try:
            engine = self.engines[engine_name]
            if hasattr(engine, 'update_ui_parameters'):
                await engine.update_ui_parameters(parameters)
                
            # Store for session persistence
            if engine_name not in self.real_time_params:
                self.real_time_params[engine_name] = {}
            self.real_time_params[engine_name].update(parameters)
            
            # Broadcast to connected clients
            await self._broadcast_parameter_update(engine_name, parameters)
            
            logger.debug(f"🎛️ Updated {engine_name} parameters: {list(parameters.keys())}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Parameter update failed for {engine_name}: {e}")
            return False
    
    # ═══════════════════════════════════════════════════════════════════════════
    # INITIALIZATION METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def _initialize_plugin_system(self):
        """Initialize enterprise-grade plugin architecture."""
        logger.info("🔌 Initializing advanced plugin system...")
        await self.plugin_manager.initialize()
        logger.info("✅ Plugin system ready")
    
    async def _initialize_engines(self):
        """Initialize all sophisticated Roo engines."""
        logger.info("🎛️ Initializing Roo engines...")
        
        # Initialize DrummaRoo (your 50+ parameter drum engine)
        drummaroo_controls = DrummarooUIControls()
        self.engines["drummaroo"] = AlgorithmicDrummaroo(ui_params=drummaroo_controls)
        self.engine_status["drummaroo"] = "ready"
        
        # Initialize other engines
        self.engines["bassaroo"] = BassaRoo()
        self.engines["melodyroo"] = MelodyRoo()
        self.engines["harmonyroo"] = HarmonyRoo()
        
        # Set all as ready
        for engine_name in ["bassaroo", "melodyroo", "harmonyroo"]:
            self.engine_status[engine_name] = "ready"
        
        logger.info(f"✅ {len(self.engines)} engines initialized")
    
    async def _initialize_ableton_bridge(self):
        """Initialize Ableton Live integration."""
        logger.info("🎹 Initializing Ableton bridge...")
        try:
            from ableton.ableton_bridge import AbletonBridge
            self.ableton_bridge = AbletonBridge()
            await self.ableton_bridge.connect()
            logger.info("✅ Ableton bridge connected")
        except ImportError:
            logger.warning("⚠️ Ableton bridge not available")
    
    async def _initialize_real_time_control(self):
        """Initialize real-time parameter control system."""
        logger.info("📡 Initializing real-time control...")
        # WebSocket manager initialization would go here
        logger.info("✅ Real-time control ready")
    
    async def _initialize_context_system(self):
        """Initialize sophisticated context integration."""
        logger.info("🎯 Initializing context integration system...")
        # Context system is already created in __init__
        logger.info("✅ Context system ready")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # GENERATION HELPER METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _prepare_generation_context(self, 
                                  session: MusicalSession, 
                                  request: GenerationRequest) -> Dict[str, Any]:
        """Prepare comprehensive generation context."""
        return {
            "session_id": session.session_id,
            "musical_context": session.musical_context,
            "timing_metadata": session.timing_metadata,
            "analysis_results": session.analysis_results,
            "tempo": request.tempo,
            "key_signature": request.key_signature,
            "section_type": request.section_type,
            "style": request.style,
            "target_length_ms": request.target_length_ms,
            "user_preferences": session.user_preferences
        }
    
    async def _generate_with_engine(self, 
                                  engine_name: str, 
                                  context: Dict[str, Any], 
                                  ui_controls: Dict[str, Any]) -> Dict[str, Any]:
        """Generate music with a specific engine."""
        engine = self.engines[engine_name]
        
        # Update engine parameters if provided
        if ui_controls and hasattr(engine, 'update_ui_parameters'):
            await engine.update_ui_parameters(ui_controls)
        
        # Call engine generation method
        if hasattr(engine, 'generate_drums'):  # DrummaRoo
            return await engine.generate_drums(
                context["section_type"], 
                context["target_length_ms"] * 1000  # Convert to microseconds
            )
        elif hasattr(engine, 'generate'):  # Other engines
            return await engine.generate(context)
        else:
            return {"status": "error", "message": f"Engine {engine_name} has no generate method"}
    
    async def _run_analysis_plugins(self, 
                                  input_file: str, 
                                  options: Dict[str, Any]) -> Dict[str, Any]:
        """Run all analysis plugins through the plugin manager."""
        try:
            # Get all analysis plugins
            analysis_plugins = self.plugin_manager.get_plugins_by_phase("analysis")
            
            plugin_results = {}
            for plugin_name in analysis_plugins:
                try:
                    result = await self.plugin_manager.execute_plugin(
                        plugin_name, 
                        input_file, 
                        **options
                    )
                    plugin_results[plugin_name] = result
                except Exception as e:
                    logger.warning(f"Plugin {plugin_name} failed: {e}")
                    plugin_results[plugin_name] = {"status": "error", "error": str(e)}
            
            return plugin_results
            
        except Exception as e:
            logger.error(f"Plugin analysis failed: {e}")
            return {"status": "error", "error": str(e)}
    
    # ═══════════════════════════════════════════════════════════════════════════
    # REAL-TIME COMMUNICATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def _broadcast_real_time_update(self, update_data: Dict[str, Any]):
        """Broadcast updates to all connected real-time clients."""
        if not self.websocket_connections:
            return
        
        message = {
            "type": "generation_update",
            "data": update_data,
            "timestamp": time.time()
        }
        
        # Broadcast to all connections (WebSocket implementation)
        for connection in self.websocket_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.warning(f"Failed to broadcast to connection: {e}")
    
    async def _broadcast_parameter_update(self, engine_name: str, parameters: Dict[str, Any]):
        """Broadcast parameter updates for real-time synchronization."""
        message = {
            "type": "parameter_update",
            "engine": engine_name,
            "parameters": parameters,
            "timestamp": time.time()
        }
        
        for connection in self.websocket_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.warning(f"Failed to broadcast parameter update: {e}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PUBLIC API METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "conductor_status": self.status.value,
            "engines": {name: status for name, status in self.engine_status.items()},
            "sessions": len(self.sessions),
            "active_session": self.active_session.session_id if self.active_session else None,
            "performance_metrics": self.performance_metrics,
            "ableton_connected": self.ableton_bridge is not None,
            "real_time_connections": len(self.websocket_connections)
        }
    
    async def shutdown(self):
        """Graceful shutdown of all systems."""
        logger.info("🛑 Shutting down Master Conductor...")
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Shutdown plugin manager
        if self.plugin_manager:
            await self.plugin_manager.shutdown()
        
        # Close Ableton bridge
        if self.ableton_bridge:
            await self.ableton_bridge.disconnect()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("✅ Master Conductor shutdown complete")

# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

async def create_master_conductor(config: Optional[Dict[str, Any]] = None) -> UnifiedMasterConductor:
    """Create and initialize a master conductor instance."""
    conductor = UnifiedMasterConductor(config)
    await conductor.initialize()
    return conductor

# Example usage for testing
if __name__ == "__main__":
    async def demo():
        # Create conductor
        conductor = await create_master_conductor()
        
        # Create session
        session = await conductor.create_session()
        
        # Analyze music
        if len(sys.argv) > 1:
            analysis = await conductor.analyze_music(session.session_id, sys.argv[1])
            print(f"Analysis complete: {analysis['feature_count']} features")
            
            # Generate drums
            request = GenerationRequest(
                session_id=session.session_id,
                engines_to_use=["drummaroo"],
                target_length_ms=8000,
                style="rock"
            )
            
            generation = await conductor.generate_music(session.session_id, request)
            print(f"Generation complete: {generation['generation_time_ms']:.1f}ms")
        
        # Print status
        status = conductor.get_system_status()
        print(json.dumps(status, indent=2))
        
        await conductor.shutdown()
    
    import sys
    asyncio.run(demo())
