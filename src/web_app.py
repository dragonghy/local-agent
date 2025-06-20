#!/usr/bin/env python3
"""
Web UI for Local LLM Deployment
Provides a simple chat interface for interacting with deployed models
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import asyncio
from datetime import datetime

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, will fall back to system env vars

from fastapi import FastAPI, WebSocket, HTTPException, UploadFile, File, Form
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

class TranscriptionRequest(BaseModel):
    model: str
    
# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Store uploaded files temporarily
UPLOAD_DIR = Path("temp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.get("/")
async def root():
    """Serve the main chat interface"""
    return HTMLResponse(content=open("static/index.html").read())

@app.get("/transcription")
async def transcription_page():
    """Serve the transcription interface"""
    return HTMLResponse(content=open("static/transcription.html").read())

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

@app.post("/api/audio/transcribe")
async def transcribe_audio(file: UploadFile = File(...), model: str = Form("whisper-base")) -> Dict[str, Any]:
    """Transcribe uploaded audio using Whisper model"""
    try:
        logger.info(f"Received transcription request with model: {model}")
        # Validate file type (accept both audio and video mime types for webm)
        valid_types = ['audio/', 'video/webm', 'audio/webm']
        if not file.content_type or not any(file.content_type.startswith(t) for t in valid_types):
            raise HTTPException(status_code=400, detail=f"File must be an audio file. Got: {file.content_type}")
        
        # Save uploaded file with proper extension
        timestamp = datetime.now().timestamp()
        if file.content_type and 'webm' in file.content_type:
            file_path = UPLOAD_DIR / f"{timestamp}_recording.webm"
        else:
            file_path = UPLOAD_DIR / f"{timestamp}_{file.filename}"
            
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"Saved audio file: {file_path} ({file.content_type}, {len(content)} bytes)")
        
        # Transcribe audio using specified Whisper model
        result = model_manager.transcribe_audio(
            model,  # Use the specified model
            str(file_path)
        )
        
        # Clean up
        try:
            file_path.unlink()
        except:
            pass  # Don't fail if cleanup fails
        
        return result
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/transcribe/openai")
async def transcribe_openai(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Transcribe audio using OpenAI Whisper API"""
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=501, detail="OpenAI API key not configured")
    
    try:
        import openai
        from openai import OpenAI
        
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Save uploaded file
        timestamp = datetime.now().timestamp()
        file_path = UPLOAD_DIR / f"{timestamp}_{file.filename}"
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"Transcribing with OpenAI Whisper API: {file_path}")
        start_time = time.time()
        
        # Transcribe with OpenAI
        with open(file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        
        transcription_time = time.time() - start_time
        
        # Clean up
        try:
            file_path.unlink()
        except:
            pass
        
        return {
            "transcription": transcript.strip(),
            "duration": transcription_time,
            "model": "openai-whisper"
        }
        
    except ImportError:
        raise HTTPException(status_code=501, detail="OpenAI library not installed. Install with: pip install openai")
    except Exception as e:
        logger.error(f"OpenAI transcription error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/transcribe/gemini")
async def transcribe_gemini(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Transcribe audio using Google Gemini API"""
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=501, detail="Gemini API key not configured")
    
    try:
        import google.generativeai as genai
        import base64
        
        # Configure Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Read audio file
        content = await file.read()
        
        logger.info(f"Transcribing with Gemini API")
        start_time = time.time()
        
        # Convert audio to base64 for Gemini
        audio_data = base64.b64encode(content).decode()
        
        # Create prompt for transcription
        prompt = "Please transcribe the audio content in this file. Provide only the transcribed text without any additional commentary."
        
        # Send to Gemini (Note: Audio transcription support may vary)
        response = model.generate_content([
            prompt,
            {
                "mime_type": file.content_type or "audio/webm",
                "data": audio_data
            }
        ])
        
        transcription_time = time.time() - start_time
        
        transcription = response.text.strip() if response.text else "No speech detected"
        
        return {
            "transcription": transcription,
            "duration": transcription_time,
            "model": "gemini-speech"
        }
        
    except ImportError:
        raise HTTPException(status_code=501, detail="Google Generative AI library not installed. Install with: pip install google-generativeai")
    except Exception as e:
        logger.error(f"Gemini transcription error: {str(e)}")
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
    import argparse
    
    parser = argparse.ArgumentParser(description="Local LLM Web Interface")
    parser.add_argument("--https", action="store_true", help="Run with HTTPS")
    parser.add_argument("--port", type=int, default=8000, help="Port to run on")
    args = parser.parse_args()
    
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
    
    # Configure SSL if requested
    ssl_config = None
    if args.https:
        ssl_keyfile = Path(__file__).parent.parent / "ssl" / "key.pem"
        ssl_certfile = Path(__file__).parent.parent / "ssl" / "cert.pem"
        
        if ssl_keyfile.exists() and ssl_certfile.exists():
            ssl_config = {
                "ssl_keyfile": str(ssl_keyfile),
                "ssl_certfile": str(ssl_certfile)
            }
            print(f"Using HTTPS with SSL certificates")
        else:
            print("\nSSL certificates not found. Please generate them first:")
            print("mkdir -p ssl")
            print('openssl req -x509 -newkey rsa:4096 -keyout ssl/key.pem -out ssl/cert.pem -days 365 -nodes \\')
            print('  -subj "/C=US/ST=Local/L=Local/O=LocalLLM/CN=localhost"')
            return
    
    # Run the server
    protocol = "https" if ssl_config else "http"
    print(f"Server will be available at {protocol}://localhost:{args.port}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=args.port,
        log_level="info",
        **ssl_config if ssl_config else {}
    )

if __name__ == "__main__":
    main()