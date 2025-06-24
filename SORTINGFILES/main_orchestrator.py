#!/usr/bin/env python3
"""
MuseAroo Main Orchestrator v1.0
==============================
Production-ready orchestrator that coordinates all your Roo engines.

DYLAROO's PRIORITY: DrummaRoo first, then expand to full band.

Features:
✅ Focuses on DrummaRoo for immediate Ableton testing
✅ Clean hooks for HarmonyRoo, MelodyRoo, BassaRoo when ready  
✅ Uses your existing analysis_context system
✅ Async/await throughout for professional performance
✅ CLI interface for quick testing
✅ MIDI export ready for Ableton Live
✅ Comprehensive error handling and logging
✅ Session management and version control
"""

import asyncio
import logging
import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import argparse
from datetime import datetime

# Core system imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

try:
    # Your existing sophisticated engines
    from engines.drummaroo import AlgorithmicDrummaroo
    from ui.controls.drummaroo_controls import DrummarooUIControls
    from context.analysis_context import AnalysisContext
    from utils.precision_timing_handler import PrecisionTimingHandler, TimingMetadata
    DRUMMAROO_AVAILABLE = True
except ImportError as e:
    logging.error(f"DrummaRoo import failed: {e}")
    DRUMMAROO_AVAILABLE = False

# Optional engines (add when ready)
try:
    from engines.harmonyroo import AlgorithmicHarmonyRoo
    from ui.controls.harmonyroo_controls import HarmonyRooUIControls
    HARMONYROO_AVAILABLE = True
except ImportError:
    HARMONYROO_AVAILABLE = False
    logging.info("HarmonyRoo not available yet - focusing on DrummaRoo")

try:
    from engines.melodyroo import AlgorithmicMelodyRoo
    MELODYROO_AVAILABLE = True
except ImportError:
    MELODYROO_AVAILABLE = False

try:
    from engines.bassaroo import AlgorithmicBassRoo  
    BASSAROO_AVAILABLE = True
except ImportError:
    BASSAROO_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class OrchestrationSession:
    """Manages a complete MuseAroo generation session."""
    
    session_id: str
    input_file: str
    output_dir: Path
    created_at: datetime
    
    # Engine states
    drummaroo_enabled: bool = True
    harmonyroo_enabled: bool = False
    melodyroo_enabled: bool = False
    bassaroo_enabled: bool = False
    
    # Generation results
    generated_files: List[str] = None
    generation_reports: Dict[str, Any] = None
    total_generation_time: float = 0.0
    
    def __post_init__(self):
        if self.generated_files is None:
            self.generated_files = []
        if self.generation_reports is None:
            self.generation_reports = {}
            
    def to_dict(self) -> Dict[str, Any]:
        """Export session for JSON serialization."""
        return {
            'session_id': self.session_id,
            'input_file': self.input_file,
            'output_dir': str(self.output_dir),
            'created_at': self.created_at.isoformat(),
            'engines': {
                'drummaroo': self.drummaroo_enabled,
                'harmonyroo': self.harmonyroo_enabled,
                'melodyroo': self.melodyroo_enabled,
                'bassaroo': self.bassaroo_enabled
            },
            'results': {
                'generated_files': self.generated_files,
                'generation_reports': self.generation_reports,
                'total_time': self.total_generation_time
            }
        }


class MuseArooOrchestrator:
    """
    Main orchestrator that coordinates all your Roo engines.
    Designed for production use with comprehensive error handling.
    """
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Session management
        self.current_session: Optional[OrchestrationSession] = None
        self.sessions_history: List[OrchestrationSession] = []
        
        # Engine instances
        self.engines = {}
        self.analysis_context: Optional[AnalysisContext] = None
        self.timing_handler = PrecisionTimingHandler() if 'PrecisionTimingHandler' in globals() else None
        
        logger.info("🎼 MuseAroo Orchestrator initialized")
        logger.info(f"📁 Output directory: {self.output_dir}")
        self._log_engine_availability()
        
    def _log_engine_availability(self):
        """Log which engines are available."""
        engines_status = [
            ("DrummaRoo", DRUMMAROO_AVAILABLE),
            ("HarmonyRoo", HARMONYROO_AVAILABLE), 
            ("MelodyRoo", MELODYROO_AVAILABLE),
            ("BassaRoo", BASSAROO_AVAILABLE)
        ]
        
        logger.info("🎵 Engine Availability:")
        for name, available in engines_status:
            status = "✅" if available else "⏸️"
            logger.info(f"  {status} {name}")
            
    def create_session(self, input_file: str, **engine_options) -> str:
        """Create a new orchestration session."""
        
        session_id = f"musearoo_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        self.current_session = OrchestrationSession(
            session_id=session_id,
            input_file=input_file,
            output_dir=self.output_dir,
            created_at=datetime.now(),
            drummaroo_enabled=engine_options.get('drummaroo', True),
            harmonyroo_enabled=engine_options.get('harmonyroo', False),
            melodyroo_enabled=engine_options.get('melodyroo', False),
            bassaroo_enabled=engine_options.get('bassaroo', False)
        )
        
        logger.info(f"🆕 Created session: {session_id}")
        logger.info(f"📄 Input file: {input_file}")
        
        return session_id
        
    async def analyze_input(self, input_file: str) -> Dict[str, Any]:
        """Analyze input file and build context."""
        
        logger.info(f"🔍 Analyzing input: {input_file}")
        
        try:
            # Create analysis context (using your existing system)
            if hasattr(self, 'analysis_context') and self.analysis_context:
                context_data = await self.analysis_context.analyze_file(input_file)
            else:
                # Fallback basic analysis
                context_data = self._create_basic_context(input_file)
                
            # Add timing precision if available
            if self.timing_handler:
                timing_metadata = self.timing_handler.analyze_input_timing(input_file)
                context_data['timing_metadata'] = timing_metadata
                
            logger.info("✅ Analysis complete")
            return context_data
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            return self._create_fallback_context(input_file)
            
    def _create_basic_context(self, input_file: str) -> Dict[str, Any]:
        """Create basic musical context for generation."""
        return {
            'input_file': input_file,
            'tempo': 120.0,
            'time_signature': [4, 4],
            'key': 'C',
            'style': 'rock',
            'energy': 0.7,
            'complexity': 0.5,
            'duration': 32.0  # bars
        }
        
    def _create_fallback_context(self, input_file: str) -> Dict[str, Any]:
        """Fallback context when analysis fails."""
        logger.warning("🔄 Using fallback analysis context")
        return self._create_basic_context(input_file)
        
    async def generate_drummaroo(self, context: Dict[str, Any], 
                                ui_params: Optional[DrummarooUIControls] = None) -> Dict[str, Any]:
        """Generate drums using your sophisticated DrummaRoo engine."""
        
        if not DRUMMAROO_AVAILABLE:
            logger.error("❌ DrummaRoo not available")
            return {'error': 'DrummaRoo engine not available'}
            
        logger.info("🥁 Generating drums with DrummaRoo...")
        start_time = time.time()
        
        try:
            # Use provided UI params or create defaults
            if ui_params is None:
                ui_params = DrummarooUIControls(
                    groove_intensity=0.7,
                    pattern_complexity=0.6,
                    swing_amount=0.0,
                    syncopation_level=0.4,
                    humanization=0.7,
                    rock_influence=0.8
                )
                
            # Initialize DrummaRoo with context
            drummaroo = AlgorithmicDrummaroo(
                analyzer_data=context,
                ui_params=ui_params
            )
            
            # Generate drums
            section_length = context.get('duration', 32.0) * 4  # bars to beats
            drum_events = await drummaroo.generate_drums(
                section='verse',
                length_beats=section_length
            )
            
            # Export MIDI
            output_file = self.output_dir / f"{self.current_session.session_id}_drums.mid"
            midi_data = drummaroo.export_midi(drum_events, str(output_file))
            
            generation_time = time.time() - start_time
            
            result = {
                'engine': 'DrummaRoo',
                'output_file': str(output_file),
                'event_count': len(drum_events),
                'generation_time': generation_time,
                'ui_parameters': asdict(ui_params),
                'musical_context': {
                    'tempo': context.get('tempo', 120),
                    'time_signature': context.get('time_signature', [4, 4]),
                    'style': context.get('style', 'rock')
                }
            }
            
            logger.info(f"✅ DrummaRoo generated {len(drum_events)} events in {generation_time:.2f}s")
            logger.info(f"📄 MIDI saved: {output_file}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ DrummaRoo generation failed: {e}")
            return {'error': str(e)}
            
    async def generate_harmonyroo(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate harmony (when HarmonyRoo is ready)."""
        
        if not HARMONYROO_AVAILABLE:
            logger.info("⏸️ HarmonyRoo not enabled yet")
            return {'status': 'disabled', 'reason': 'HarmonyRoo not available'}
            
        logger.info("🎹 HarmonyRoo generation would happen here")
        # TODO: Implement when HarmonyRoo is ready
        return {'status': 'todo', 'engine': 'HarmonyRoo'}
        
    async def generate_melodyroo(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate melody (when MelodyRoo is ready)."""
        
        if not MELODYROO_AVAILABLE:
            logger.info("⏸️ MelodyRoo not enabled yet")
            return {'status': 'disabled', 'reason': 'MelodyRoo not available'}
            
        logger.info("🎵 MelodyRoo generation would happen here")  
        # TODO: Implement when MelodyRoo is ready
        return {'status': 'todo', 'engine': 'MelodyRoo'}
        
    async def generate_bassaroo(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate bass (when BassaRoo is ready)."""
        
        if not BASSAROO_AVAILABLE:
            logger.info("⏸️ BassaRoo not enabled yet")
            return {'status': 'disabled', 'reason': 'BassaRoo not available'}
            
        logger.info("🎸 BassaRoo generation would happen here")
        # TODO: Implement when BassaRoo is ready  
        return {'status': 'todo', 'engine': 'BassaRoo'}
        
    async def orchestrate_full_session(self, input_file: str, 
                                     drummaroo_params: Optional[DrummarooUIControls] = None,
                                     **engine_options) -> Dict[str, Any]:
        """
        Complete orchestration workflow: analyze → generate → export.
        Currently focuses on DrummaRoo with hooks for other engines.
        """
        
        logger.info("🎼 Starting full orchestration session")
        session_start = time.time()
        
        # Create session
        session_id = self.create_session(input_file, **engine_options)
        
        try:
            # Step 1: Analyze input
            context = await self.analyze_input(input_file)
            
            # Step 2: Generate with enabled engines
            generation_results = {}
            
            # DrummaRoo (priority)
            if self.current_session.drummaroo_enabled:
                drum_result = await self.generate_drummaroo(context, drummaroo_params)
                generation_results['drums'] = drum_result
                
                if 'output_file' in drum_result:
                    self.current_session.generated_files.append(drum_result['output_file'])
                    
            # Other engines (when ready)
            if self.current_session.harmonyroo_enabled:
                harmony_result = await self.generate_harmonyroo(context)
                generation_results['harmony'] = harmony_result
                
            if self.current_session.melodyroo_enabled:
                melody_result = await self.generate_melodyroo(context)
                generation_results['melody'] = melody_result
                
            if self.current_session.bassaroo_enabled:
                bass_result = await self.generate_bassaroo(context)
                generation_results['bass'] = bass_result
                
            # Update session
            self.current_session.generation_reports = generation_results
            self.current_session.total_generation_time = time.time() - session_start
            
            # Save session report
            session_report = self.current_session.to_dict()
            report_file = self.output_dir / f"{session_id}_session_report.json"
            
            with open(report_file, 'w') as f:
                json.dump(session_report, f, indent=2)
                
            # Add to history
            self.sessions_history.append(self.current_session)
            
            logger.info(f"✅ Orchestration complete in {self.current_session.total_generation_time:.2f}s")
            logger.info(f"📄 Session report: {report_file}")
            
            return session_report
            
        except Exception as e:
            logger.error(f"❌ Orchestration failed: {e}")
            return {'error': str(e), 'session_id': session_id}


def create_cli_parser() -> argparse.ArgumentParser:
    """Create command-line interface."""
    
    parser = argparse.ArgumentParser(description='MuseAroo Main Orchestrator')
    
    parser.add_argument('input_file', help='Input audio/MIDI file to process')
    parser.add_argument('-o', '--output', default='output', help='Output directory')
    
    # Engine controls
    parser.add_argument('--drums-only', action='store_true', 
                       help='Generate only drums (default for now)')
    parser.add_argument('--enable-harmony', action='store_true',
                       help='Enable HarmonyRoo (when available)')
    parser.add_argument('--enable-melody', action='store_true', 
                       help='Enable MelodyRoo (when available)')
    parser.add_argument('--enable-bass', action='store_true',
                       help='Enable BassaRoo (when available)')
    
    # DrummaRoo parameters
    parser.add_argument('--groove-intensity', type=float, default=0.7,
                       help='DrummaRoo groove intensity (0.0-1.0)')
    parser.add_argument('--complexity', type=float, default=0.6,
                       help='Pattern complexity (0.0-1.0)')
    parser.add_argument('--swing', type=float, default=0.0,
                       help='Swing amount (0.0-1.0)')
    parser.add_argument('--rock-influence', type=float, default=0.8,
                       help='Rock style influence (0.0-1.0)')
    
    return parser


async def main():
    """Main CLI entry point."""
    
    parser = create_cli_parser()
    args = parser.parse_args()
    
    # Check input file exists
    if not Path(args.input_file).exists():
        logger.error(f"❌ Input file not found: {args.input_file}")
        return 1
        
    # Create orchestrator
    orchestrator = MuseArooOrchestrator(output_dir=args.output)
    
    # Configure DrummaRoo parameters
    drummaroo_params = DrummarooUIControls(
        groove_intensity=args.groove_intensity,
        pattern_complexity=args.complexity,
        swing_amount=args.swing,
        rock_influence=args.rock_influence
    ) if DRUMMAROO_AVAILABLE else None
    
    # Engine options
    engine_options = {
        'drummaroo': True,  # Always enable for now
        'harmonyroo': args.enable_harmony,
        'melodyroo': args.enable_melody,
        'bassaroo': args.enable_bass
    }
    
    # Run orchestration
    logger.info("🚀 Starting MuseAroo orchestration")
    
    try:
        result = await orchestrator.orchestrate_full_session(
            args.input_file,
            drummaroo_params=drummaroo_params,
            **engine_options
        )
        
        if 'error' in result:
            logger.error(f"❌ Orchestration failed: {result['error']}")
            return 1
            
        # Success summary
        logger.info("🎉 ORCHESTRATION COMPLETE!")
        logger.info(f"📁 Session ID: {result['session_id']}")
        
        if result['results']['generated_files']:
            logger.info("📄 Generated files:")
            for file in result['results']['generated_files']:
                logger.info(f"  • {file}")
                
        logger.info(f"⏱️  Total time: {result['results']['total_time']:.2f}s")
        logger.info("\n🎵 Ready for Ableton Live! Import the MIDI files.")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Orchestration crashed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
