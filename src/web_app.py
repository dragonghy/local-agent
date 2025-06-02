#!/usr/bin/env python3
"""
Web UI for Local LLM Deployment
Provides a simple chat interface for interacting with deployed models
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import asyncio
from datetime import datetime

from fastapi import FastAPI, WebSocket, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import torch
import logging
import platform

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path to import inference module
sys.path.append(str(Path(__file__).parent))
from inference import ModelManager

# Log environment info
logger.info(f"Platform: {platform.system()} {platform.machine()}")
logger.info(f"Using device: {'mps' if torch.backends.mps.is_available() else 'cpu'}")

# Initialize FastAPI app
app = FastAPI(title="Local LLM Chat", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model manager
model_manager = ModelManager()

# Request/Response models
class ChatRequest(BaseModel):
    model: str
    prompt: str
    temperature: float = 0.7
    max_tokens: int = 256
    system_prompt: Optional[str] = None

class ImageRequest(BaseModel):
    model: str
    image_path: str
    prompt: Optional[str] = None

class ModelInfo(BaseModel):
    name: str
    type: str
    loaded: bool
    description: str

# Store uploaded files temporarily
UPLOAD_DIR = Path("temp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.get("/")
async def root():
    """Serve the main chat interface"""
    return HTMLResponse(content=open("static/index.html").read())

@app.get("/api/models")
async def list_models() -> List[ModelInfo]:
    """List available models and their status"""
    models = []
    for name, config in model_manager.model_configs.items():
        models.append(ModelInfo(
            name=name,
            type=config["type"],
            loaded=name in model_manager.loaded_models,
            description=f"{config['type'].title()} model"
        ))
    return models

@app.post("/api/chat")
async def chat(request: ChatRequest) -> Dict[str, Any]:
    """Generate text response from a model"""
    try:
        result = model_manager.generate_text(
            request.model,
            request.prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            system_prompt=request.system_prompt
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """Generate streaming text response from a model"""
    async def generate():
        try:
            # Send initial message
            yield f"data: {json.dumps({'type': 'start', 'model': request.model})}\n\n"
            
            # Generate tokens
            async for token_data in model_manager.generate_text_stream(
                request.model,
                request.prompt,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                system_prompt=request.system_prompt
            ):
                yield f"data: {json.dumps(token_data)}\n\n"
                await asyncio.sleep(0)  # Allow other tasks to run
            
            # Send completion message
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )

@app.post("/api/image/analyze")
async def analyze_image(
    model: str,
    prompt: Optional[str] = None,
    file: UploadFile = File(...)
) -> Dict[str, Any]:
    """Analyze an uploaded image"""
    try:
        # Save uploaded file
        file_path = UPLOAD_DIR / f"{datetime.now().timestamp()}_{file.filename}"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Analyze image
        result = model_manager.analyze_image(
            model,
            str(file_path),
            prompt
        )
        
        # Clean up
        file_path.unlink()
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for streaming responses"""
    await websocket.accept()
    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            # Generate response
            # TODO: Implement streaming generation
            result = model_manager.generate_text(
                data["model"],
                data["prompt"],
                temperature=data.get("temperature", 0.7),
                max_tokens=data.get("max_tokens", 256)
            )
            
            # Send response
            await websocket.send_json(result)
            
    except Exception as e:
        await websocket.close(code=1000)

@app.post("/api/load_model/{model_name}")
async def load_model(model_name: str) -> Dict[str, str]:
    """Load a model into memory"""
    if model_name not in model_manager.model_configs:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    success = model_manager.load_model(model_name)
    if success:
        return {"status": "success", "message": f"Model {model_name} loaded successfully"}
    else:
        raise HTTPException(status_code=500, detail=f"Failed to load model {model_name}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": len(model_manager.loaded_models),
        "available_models": len(model_manager.model_configs)
    }

# Mount static files
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

def main():
    """Run the web application"""
    # Pre-load a default model
    print("Starting Local LLM Web Interface...")
    
    # Try to load a lightweight model by default
    default_models = ["qwen2.5-1.5b", "deepseek-r1-distill-qwen-1.5b"]
    for model in default_models:
        if model in model_manager.model_configs:
            print(f"Loading default model: {model}")
            if model_manager.load_model(model):
                print(f"Successfully loaded {model}")
                break
    
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

if __name__ == "__main__":
    main()