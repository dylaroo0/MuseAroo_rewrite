#!/usr/bin/env python3
"""
MuseAroo Unified Server v2.0
============================
Production-ready server combining the best of all implementations:
- FastAPI for REST endpoints
- WebSocket for real-time communication
- Plugin architecture for extensibility
- Precision timing preservation
- Memory-efficient context management
- Ableton Live integration

This replaces:
- musearoo_master_conductor.py
- enhanced_api_server.py
- websocket_manager.py
- Multiple duplicate servers

To run:
    # Development
    python musearoo_server.py --dev
    
    # Production
    python musearoo_server.py --prod --workers 4
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from contextlib import asynccontextmanager
import uuid

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import MuseAroo components
from conductor.master_conductor import MasterConductor
from context.context_integrator import ContextIntegrator
from context.analysis_context import AnalysisContext
from utils.precision_timing_handler import PrecisionTimingHandler
from plugins.plugin_manager import PluginManager
from engines.base.engine_base import EngineRegistry


# =============================================================================
# CONFIGURATION
# =============================================================================

class ServerConfig:
    """Server configuration."""
    
    # Server settings
    HOST = os.getenv("MUSEAROO_HOST", "0.0.0.0")
    PORT = int(os.getenv("MUSEAROO_PORT", "8000"))
    WORKERS = int(os.getenv("MUSEAROO_WORKERS", "1"))
    
    # Environment
    ENV = os.getenv("MUSEAROO_ENV", "dev")
    IS_DEV = ENV == "dev"
    
    # Paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    TEMP_DIR = BASE_DIR / "temp"
    
    # WebSocket settings
    WS_HEARTBEAT_INTERVAL = 30
    WS_MAX_CONNECTIONS = 100
    
    # File upload settings
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS = {".mp3", ".wav", ".mid", ".midi", ".flac", ".ogg"}
    
    # Logging
    LOG_LEVEL = logging.DEBUG if IS_DEV else logging.INFO
    

# Setup logging
logging.basicConfig(
    level=ServerConfig.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("musearoo.server")


# =============================================================================
# WEBSOCKET MANAGER
# =============================================================================

class ConnectionManager:
    """Manages WebSocket connections and message routing."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_info: Dict[str, Dict[str, Any]] = {}
        self.message_history: Dict[str, List[Dict]] = {}
        
    async def connect(self, websocket: WebSocket, session_id: str):
        """Accept new WebSocket connection."""
        await websocket.accept()
        self.active_connections[session_id] = websocket
        self.session_info[session_id] = {
            "connected_at": datetime.now(),
            "last_activity": datetime.now(),
            "client_info": {}
        }
        self.message_history[session_id] = []
        logger.info(f"WebSocket connected: {session_id}")
        
    async def disconnect(self, session_id: str):
        """Handle WebSocket disconnection."""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            del self.session_info[session_id]
            logger.info(f"WebSocket disconnected: {session_id}")
            
    async def send_message(self, session_id: str, message: Dict[str, Any]):
        """Send message to specific session."""
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            await websocket.send_json(message)
            self.message_history[session_id].append({
                "timestamp": datetime.now().isoformat(),
                "direction": "outbound",
                "message": message
            })
            
    async def broadcast(self, message: Dict[str, Any], exclude: Optional[Set[str]] = None):
        """Broadcast message to all connected sessions."""
        exclude = exclude or set()
        for session_id, websocket in self.active_connections.items():
            if session_id not in exclude:
                await websocket.send_json(message)
                
    async def receive_message(self, session_id: str, message: Dict[str, Any]):
        """Process received message."""
        self.session_info[session_id]["last_activity"] = datetime.now()
        self.message_history[session_id].append({
            "timestamp": datetime.now().isoformat(),
            "direction": "inbound",
            "message": message
        })
        

# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class AnalyzeRequest(BaseModel):
    """Music analysis request."""
    file_path: str
    options: Dict[str, Any] = Field(default_factory=dict)
    session_id: Optional[str] = None
    

class GenerateRequest(BaseModel):
    """Music generation request."""
    engine: str  # drummaroo, bassaroo, melodyroo, harmonyroo
    context_data: Dict[str, Any]
    parameters: Dict[str, Any] = Field(default_factory=dict)
    session_id: Optional[str] = None
    

class EngineStatus(BaseModel):
    """Engine status information."""
    name: str
    version: str
    available: bool
    parameters: Dict[str, Any]
    

class SessionInfo(BaseModel):
    """Session information."""
    session_id: str
    created_at: datetime
    status: str
    engines_used: List[str]
    files_processed: int
    

# =============================================================================
# MAIN APPLICATION
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("🎵 MuseAroo Server Starting...")
    
    # Initialize directories
    ServerConfig.DATA_DIR.mkdir(exist_ok=True)
    ServerConfig.TEMP_DIR.mkdir(exist_ok=True)
    
    # Initialize components
    app.state.conductor = MasterConductor()
    app.state.plugin_manager = PluginManager()
    app.state.engine_registry = EngineRegistry()
    app.state.connection_manager = ConnectionManager()
    
    # Load plugins
    await app.state.plugin_manager.discover_plugins()
    
    # Start background tasks
    app.state.heartbeat_task = asyncio.create_task(heartbeat_task(app))
    
    logger.info("✅ Server ready!")
    
    yield
    
    # Shutdown
    logger.info("🛑 MuseAroo Server Shutting down...")
    
    # Cancel background tasks
    app.state.heartbeat_task.cancel()
    
    # Close connections
    for session_id in list(app.state.connection_manager.active_connections.keys()):
        await app.state.connection_manager.disconnect(session_id)
        
    logger.info("👋 Shutdown complete")
    

# Create FastAPI app
app = FastAPI(
    title="MuseAroo API",
    description="AI-powered music generation and analysis system",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ServerConfig.IS_DEV else ["https://musearoo.ai"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# HEALTH & STATUS ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "MuseAroo",
        "version": "2.0.0",
        "status": "operational",
        "docs": "/docs"
    }
    

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "environment": ServerConfig.ENV,
        "active_sessions": len(app.state.connection_manager.active_connections)
    }
    

@app.get("/engines")
async def list_engines():
    """List available engines."""
    engines = []
    
    for engine_name, engine_class in app.state.engine_registry.engines.items():
        engines.append(EngineStatus(
            name=engine_name,
            version=getattr(engine_class, "VERSION", "1.0"),
            available=True,
            parameters=getattr(engine_class, "PARAMETERS", {})
        ))
        
    return {"engines": engines}
    

@app.get("/plugins")
async def list_plugins():
    """List available plugins."""
    return {
        "plugins": app.state.plugin_manager.get_plugin_list(),
        "total": len(app.state.plugin_manager.plugins)
    }
    

# =============================================================================
# FILE UPLOAD ENDPOINTS
# =============================================================================

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    session_id: Optional[str] = None
):
    """Upload a music file for analysis."""
    # Validate file
    if not any(file.filename.endswith(ext) for ext in ServerConfig.ALLOWED_EXTENSIONS):
        raise HTTPException(400, f"File type not allowed. Supported: {ServerConfig.ALLOWED_EXTENSIONS}")
        
    if file.size > ServerConfig.MAX_FILE_SIZE:
        raise HTTPException(400, f"File too large. Max size: {ServerConfig.MAX_FILE_SIZE} bytes")
        
    # Generate session ID if not provided
    session_id = session_id or str(uuid.uuid4())
    
    # Save file
    file_path = ServerConfig.TEMP_DIR / session_id / file.filename
    file_path.parent.mkdir(exist_ok=True)
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
        
    logger.info(f"File uploaded: {file_path} (session: {session_id})")
    
    return {
        "session_id": session_id,
        "file_path": str(file_path),
        "file_size": len(content),
        "status": "uploaded"
    }
    

# =============================================================================
# ANALYSIS ENDPOINTS
# =============================================================================

@app.post("/analyze")
async def analyze_music(
    request: AnalyzeRequest,
    background_tasks: BackgroundTasks
):
    """Analyze uploaded music file."""
    # Create analysis context
    context = AnalysisContext(
        input_path=request.file_path,
        session_id=request.session_id or str(uuid.uuid4())
    )
    
    # Start analysis in background
    background_tasks.add_task(
        run_analysis,
        app.state.conductor,
        context,
        request.options
    )
    
    return {
        "session_id": context.session_id,
        "status": "analyzing",
        "message": "Analysis started. Connect via WebSocket for updates."
    }
    

# =============================================================================
# GENERATION ENDPOINTS
# =============================================================================

@app.post("/generate")
async def generate_music(
    request: GenerateRequest,
    background_tasks: BackgroundTasks
):
    """Generate music using specified engine."""
    # Validate engine
    if request.engine not in app.state.engine_registry.engines:
        raise HTTPException(400, f"Unknown engine: {request.engine}")
        
    # Create session
    session_id = request.session_id or str(uuid.uuid4())
    
    # Start generation in background
    background_tasks.add_task(
        run_generation,
        app.state.conductor,
        request.engine,
        request.context_data,
        request.parameters,
        session_id
    )
    
    return {
        "session_id": session_id,
        "engine": request.engine,
        "status": "generating",
        "message": "Generation started. Connect via WebSocket for updates."
    }
    

# =============================================================================
# WEBSOCKET ENDPOINT
# =============================================================================

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time communication."""
    manager = app.state.connection_manager
    
    await manager.connect(websocket, session_id)
    
    try:
        # Send welcome message
        await manager.send_message(session_id, {
            "type": "welcome",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        })
        
        # Message loop
        while True:
            data = await websocket.receive_json()
            await manager.receive_message(session_id, data)
            
            # Process message
            await process_websocket_message(app, session_id, data)
            
    except WebSocketDisconnect:
        await manager.disconnect(session_id)
        

# =============================================================================
# BACKGROUND TASKS
# =============================================================================

async def heartbeat_task(app: FastAPI):
    """Send periodic heartbeat to all connections."""
    while True:
        await asyncio.sleep(ServerConfig.WS_HEARTBEAT_INTERVAL)
        
        message = {
            "type": "heartbeat",
            "timestamp": datetime.now().isoformat()
        }
        
        await app.state.connection_manager.broadcast(message)
        

async def run_analysis(conductor: MasterConductor, context: AnalysisContext, options: Dict):
    """Run analysis in background."""
    try:
        # Run analysis
        results = await conductor.analyze(context, options)
        
        # Send results via WebSocket
        await app.state.connection_manager.send_message(context.session_id, {
            "type": "analysis_complete",
            "results": results,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        await app.state.connection_manager.send_message(context.session_id, {
            "type": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })
        

async def run_generation(
    conductor: MasterConductor,
    engine: str,
    context_data: Dict,
    parameters: Dict,
    session_id: str
):
    """Run generation in background."""
    try:
        # Run generation
        results = await conductor.generate(engine, context_data, parameters)
        
        # Send results via WebSocket
        await app.state.connection_manager.send_message(session_id, {
            "type": "generation_complete",
            "engine": engine,
            "results": results,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        await app.state.connection_manager.send_message(session_id, {
            "type": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })
        

async def process_websocket_message(app: FastAPI, session_id: str, message: Dict):
    """Process incoming WebSocket message."""
    msg_type = message.get("type")
    
    if msg_type == "parameter_update":
        # Handle real-time parameter updates
        engine = message.get("engine")
        parameters = message.get("parameters", {})
        
        # Update engine parameters
        if engine in app.state.engine_registry.engines:
            # Send to engine for real-time update
            await app.state.conductor.update_parameters(engine, parameters)
            
            # Acknowledge
            await app.state.connection_manager.send_message(session_id, {
                "type": "parameter_update_ack",
                "engine": engine,
                "timestamp": datetime.now().isoformat()
            })
            
    elif msg_type == "get_status":
        # Send current status
        await app.state.connection_manager.send_message(session_id, {
            "type": "status",
            "engines": list(app.state.engine_registry.engines.keys()),
            "plugins": len(app.state.plugin_manager.plugins),
            "timestamp": datetime.now().isoformat()
        })
        

# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MuseAroo Unified Server")
    parser.add_argument("--dev", action="store_true", help="Run in development mode")
    parser.add_argument("--prod", action="store_true", help="Run in production mode")
    parser.add_argument("--host", default=ServerConfig.HOST, help="Host to bind to")
    parser.add_argument("--port", type=int, default=ServerConfig.PORT, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=ServerConfig.WORKERS, help="Number of workers")
    
    args = parser.parse_args()
    
    # Set environment
    if args.dev:
        os.environ["MUSEAROO_ENV"] = "dev"
    elif args.prod:
        os.environ["MUSEAROO_ENV"] = "prod"
        
    # Configure uvicorn
    config = {
        "app": "musearoo_server:app",
        "host": args.host,
        "port": args.port,
        "workers": args.workers if args.prod else 1,
        "reload": args.dev,
        "log_level": "debug" if args.dev else "info"
    }
    
    # Run server
    logger.info(f"Starting MuseAroo Server on {args.host}:{args.port}")
    uvicorn.run(**config)
    

if __name__ == "__main__":
    main()
