CMD#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
"""
MuseAroo Unified Server — FastAPI + WebSocket + Plugin/Context System
World-class, cutting-edge music AI API for DAW, web, and real-time apps.

- Modern async/await FastAPI for REST endpoints
- WebSocket hub for DAW and UI sync (Ableton, Max4Live, web)
- Advanced plugin/engine/context integration
- Professional logging, error handling, Swagger docs
- Switch between DEV (simple) and PROD (enterprise) modes

To run:
    # Development (reloads, easy debug)
    uvicorn musearoo_server:app --reload

    # Production (scalable, secure)
    uvicorn musearoo_server:app --host 0.0.0.0 --port 8000 --workers 4

Docs: http://localhost:8000/docs
WebSocket: ws://localhost:8000/ws/{session_id}
"""

import os
import sys
import logging
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, APIRouter, Request, Depends, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import project modules (must be in PYTHONPATH)
from context import create_runtime_context, create_analysis_context, create_integrator
from advanced_plugin_architecture import AdvancedPluginManager
from utils.logger_setup import setup_logger
from utils.file_handler import validate_music_file
from engines.base.engine_base_classes import EngineBase
from pydantic import BaseModel

# WebSocketManager for live sync
from websocket_manager import WebSocketManager

# -----------------------------------------------------------------------------
# CONFIGURATION

DEV_MODE = os.environ.get("MUSEAROO_ENV", "dev") == "dev"
logger = setup_logger("musearoo_server", logging.DEBUG if DEV_MODE else logging.INFO)
plugin_manager = AdvancedPluginManager()
ws_manager = WebSocketManager(host="0.0.0.0", port=8765)

# -----------------------------------------------------------------------------
# FASTAPI APP

app = FastAPI(
    title="MuseAroo Unified API",
    description="Plugin-based AI music server for Ableton, DAWs, web, and more.",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set to ["https://yourdomain.com"] for production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# SCHEMA & ROUTES

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """System health check for CI/ops."""
    return HealthResponse(status="ok", timestamp=datetime.now(), version="2.0.0")

@app.get("/plugins", tags=["Plugins"])
async def list_plugins():
    """List all available plugins."""
    return {"plugins": [p.name for p in plugin_manager.list_all_plugins()]}

class GenerateRequest(BaseModel):
    midi_path: str
    session_id: Optional[str] = "default"
    params: Dict[str, Any] = {}

@app.post("/generate", tags=["Generation"])
async def generate_music(req: GenerateRequest):
    """
    Trigger a full music generation workflow.
    """
    # Validate input
    try:
        validate_music_file(req.midi_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    session_id = req.session_id or "default"
    runtime_ctx = create_runtime_context()
    analysis_ctx = create_analysis_context()
    integrator = create_integrator("genius")
    analysis_ctx.features["midi_path"] = req.midi_path
    analysis_ctx.features.update(req.params)

    # Intelligence phase (plugin)
    intelligence_plugins = plugin_manager.get_plugins_by_phase("intelligence")
    if intelligence_plugins:
        await intelligence_plugins[0].run(runtime_ctx, analysis_ctx)
    role_ctxs = integrator.merge(runtime_ctx, analysis_ctx)
    # Generation for each role
    gen_results = {}
    for phase in ("drums", "bass", "harmony", "melody"):
        plugins = plugin_manager.get_plugins_by_phase(phase)
        for plugin in plugins:
            result = await plugin.generate(role_ctxs[phase])
            gen_results[phase] = str(result)
            await ws_manager.broadcast(f"{phase} generated: {result}")
    return {"status": "ok", "generated": gen_results}

@app.get("/sessions", tags=["Session"])
async def get_sessions():
    """List all active sessions."""
    return {"sessions": list(ws_manager.active_sessions.keys())}

# -----------------------------------------------------------------------------
# WEBSOCKET ENDPOINT

@app.websocket("/ws/{session_id}")
async def ws_endpoint(ws: WebSocket, session_id: str):
    """Real-time WebSocket connection for UI/DAW/clients."""
    await ws.accept()
    await ws_manager.connect(session_id, ws)
    try:
        while True:
            msg = await ws.receive_text()
            logger.info(f"[WS {session_id}] Received: {msg}")
            await ws_manager.broadcast(f"[{session_id}] {msg}")
    except WebSocketDisconnect:
        await ws_manager.disconnect(session_id)
        logger.info(f"WS client {session_id} disconnected")

# -----------------------------------------------------------------------------
# ERROR HANDLING

@app.exception_handler(Exception)
async def all_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"error": str(exc), "path": request.url.path})

# -----------------------------------------------------------------------------
# STARTUP TASKS

@app.on_event("startup")
async def on_startup():
    logger.info("MuseAroo server starting up...")
    # Optionally start WebSocket server as a background task
    asyncio.create_task(ws_manager.start_server())
    logger.info("WebSocket manager started.")

@app.on_event("shutdown")
async def on_shutdown():
    logger.info("MuseAroo server shutting down.")
    await ws_manager.stop_server()

# -----------------------------------------------------------------------------
# CLI LAUNCH (for python musearoo_server.py --dev)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    logger.info(f"Launching on http://localhost:{port} (DEV_MODE={DEV_MODE})")
    uvicorn.run("musearoo_server:app", host="0.0.0.0", port=port, reload=DEV_MODE)

