#!/usr/bin/env python3
"""
Unified inference wrapper for multiple model types
Supports text generation, vision-language, and speech recognition
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, AsyncGenerator
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, AutoProcessor,
    WhisperProcessor, WhisperForConditionalGeneration,
    BlipProcessor, BlipForConditionalGeneration,
    Blip2Processor, Blip2ForConditionalGeneration,
    pipeline
)
from PIL import Image
import numpy as np
try:
    from .prompt_templates import PromptTemplateFactory, get_model_params
except ImportError:
    from prompt_templates import PromptTemplateFactory, get_model_params

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set device
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")

class ModelManager:
    """Manages loading and inference for multiple model types"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.loaded_models = {}
        self.model_configs = {
            "deepseek-r1-distill-qwen-1.5b": {
                "type": "text",
                "class": "causal_lm",
                "path": self.models_dir / "deepseek-r1-distill-qwen-1.5b"
            },
            "qwen2.5-1.5b": {
                "type": "text",
                "class": "causal_lm",
                "path": self.models_dir / "qwen2.5-1.5b"
            },
            "llama-3.2-3b": {
                "type": "text",
                "class": "causal_lm",
                "path": self.models_dir / "llama-3.2-3b"
            },
            "whisper-base": {
                "type": "speech",
                "class": "whisper",
                "path": self.models_dir / "whisper-base"
            },
            "whisper-small": {
                "type": "speech",
                "class": "whisper",
                "path": self.models_dir / "whisper-small"
            },
            "whisper-medium": {
                "type": "speech",
                "class": "whisper",
                "path": self.models_dir / "whisper-medium"
            },
            "whisper-large-v2": {
                "type": "speech",
                "class": "whisper",
                "path": self.models_dir / "whisper-large-v2"
            },
            "whisper-large-v3": {
                "type": "speech",
                "class": "whisper",
                "path": self.models_dir / "whisper-large-v3"
            },
            "blip2-base": {
                "type": "vision-language",
                "class": "blip2",
                "path": self.models_dir / "blip2-base"
            },
            "blip-base": {
                "type": "vision-language",
                "class": "blip",
                "path": self.models_dir / "blip-base"
            },
            "deepseek-vl2-small": {
                "type": "vision-language",
                "class": "deepseek-vl",
                "path": self.models_dir / "deepseek-vl2-small"
            }
        }
    
    def load_model(self, model_name: str) -> bool:
        """Load a model into memory"""
        if model_name in self.loaded_models:
            logger.info(f"Model {model_name} already loaded")
            return True
        
        if model_name not in self.model_configs:
            logger.error(f"Unknown model: {model_name}")
            return False
        
        config = self.model_configs[model_name]
        model_path = config["path"]
        
        if not model_path.exists():
            logger.error(f"Model path does not exist: {model_path}")
            return False
        
        logger.info(f"Loading {model_name} from {model_path}")
        start_time = time.time()
        
        try:
            if config["class"] == "causal_lm":
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if DEVICE == "mps" else torch.float32,
                    device_map=DEVICE
                )
                self.loaded_models[model_name] = {
                    "model": model,
                    "tokenizer": tokenizer,
                    "type": config["type"]
                }
                
            elif config["class"] == "whisper":
                processor = WhisperProcessor.from_pretrained(model_path)
                model = WhisperForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if DEVICE == "mps" else torch.float32
                ).to(DEVICE)
                self.loaded_models[model_name] = {
                    "model": model,
                    "processor": processor,
                    "type": config["type"]
                }
                
            elif config["class"] == "blip":
                processor = BlipProcessor.from_pretrained(model_path)
                model = BlipForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if DEVICE == "mps" else torch.float32
                ).to(DEVICE)
                self.loaded_models[model_name] = {
                    "model": model,
                    "processor": processor,
                    "type": config["type"]
                }
                
            elif config["class"] == "blip2":
                processor = Blip2Processor.from_pretrained(model_path)
                model = Blip2ForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if DEVICE == "mps" else torch.float32
                ).to(DEVICE)
                self.loaded_models[model_name] = {
                    "model": model,
                    "processor": processor,
                    "type": config["type"]
                }
                
            elif config["class"] == "deepseek-vl":
                # DeepSeek VL uses AutoProcessor and AutoModelForCausalLM
                processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if DEVICE == "mps" else torch.float32,
                    device_map=DEVICE,
                    trust_remote_code=True
                )
                self.loaded_models[model_name] = {
                    "model": model,
                    "processor": processor,
                    "type": config["type"]
                }
            
            load_time = time.time() - start_time
            logger.info(f"Loaded {model_name} in {load_time:.2f} seconds")
            return True
            
        except Exception as e:
            logger.error(f"Error loading {model_name}: {str(e)}")
            return False
    
    async def generate_text_stream(self, model_name: str, prompt: str, **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate text using a language model with streaming"""
        if model_name not in self.loaded_models:
            if not self.load_model(model_name):
                yield {"type": "error", "error": f"Failed to load model {model_name}"}
                return
        
        model_data = self.loaded_models[model_name]
        if model_data["type"] != "text":
            yield {"type": "error", "error": f"Model {model_name} is not a text generation model"}
            return
        
        try:
            model = model_data["model"]
            tokenizer = model_data["tokenizer"]
            
            # Get prompt template
            template = PromptTemplateFactory.get_template(model_name)
            
            # Format prompt with template
            formatted_prompt = template.format_prompt(
                prompt, 
                system_prompt=kwargs.get("system_prompt")
            )
            
            # Set pad token if not set
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id

            # Tokenize input
            inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True).to(DEVICE)
            
            # Start timing
            start_time = time.time()
            tokens_generated = 0
            
            # Generate with streaming
            with torch.no_grad():
                # Get model-specific generation parameters
                gen_kwargs = get_model_params(model_name, {
                    "max_new_tokens": kwargs.get("max_tokens", 256),
                    "temperature": kwargs.get("temperature", 0.7),
                    "do_sample": kwargs.get("do_sample", True),
                    "top_p": kwargs.get("top_p", 0.9),
                })
                gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
                
                # Initialize generation
                input_ids = inputs["input_ids"]
                attention_mask = inputs.get("attention_mask")
                past_key_values = None
                generated_tokens = []
                
                for i in range(gen_kwargs["max_new_tokens"]):
                    # Forward pass
                    if past_key_values is None:
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            use_cache=True
                        )
                    else:
                        outputs = model(
                            input_ids=input_ids[:, -1:],
                            attention_mask=attention_mask,
                            past_key_values=past_key_values,
                            use_cache=True
                        )
                    
                    past_key_values = outputs.past_key_values
                    logits = outputs.logits[:, -1, :]
                    
                    # Apply temperature
                    if gen_kwargs["temperature"] > 0:
                        logits = logits / gen_kwargs["temperature"]
                    
                    # Sample token
                    if gen_kwargs["do_sample"]:
                        probs = torch.nn.functional.softmax(logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        next_token = torch.argmax(logits, dim=-1, keepdim=True)
                    
                    # Decode token
                    token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
                    generated_tokens.append(next_token[0].item())
                    tokens_generated += 1
                    
                    # Calculate metrics
                    elapsed_time = time.time() - start_time
                    tokens_per_second = tokens_generated / elapsed_time if elapsed_time > 0 else 0
                    
                    # Yield token data
                    yield {
                        "type": "token",
                        "token": token_text,
                        "token_id": next_token[0].item(),
                        "tokens_generated": tokens_generated,
                        "tokens_per_second": tokens_per_second,
                        "elapsed_time": elapsed_time
                    }
                    
                    # Check for EOS
                    if next_token[0].item() == tokenizer.eos_token_id:
                        break
                    
                    # Update input_ids and attention_mask
                    input_ids = torch.cat([input_ids, next_token], dim=-1)
                    if attention_mask is not None:
                        attention_mask = torch.cat([
                            attention_mask,
                            torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device)
                        ], dim=-1)
                
                # Final metrics
                total_time = time.time() - start_time
                final_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                # Extract clean response using template
                clean_response = template.extract_response(
                    tokenizer.decode(input_ids[0], skip_special_tokens=True),
                    formatted_prompt
                )
                
                yield {
                    "type": "final",
                    "full_response": clean_response,
                    "tokens_generated": tokens_generated,
                    "generation_time": total_time,
                    "tokens_per_second": tokens_generated / total_time if total_time > 0 else 0
                }
                
        except Exception as e:
            logger.error(f"Error in streaming generation: {str(e)}")
            yield {"type": "error", "error": str(e)}
    
    def generate_text(self, model_name: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate text using a language model"""
        if model_name not in self.loaded_models:
            if not self.load_model(model_name):
                return {"error": f"Failed to load model {model_name}"}
        
        model_data = self.loaded_models[model_name]
        if model_data["type"] != "text":
            return {"error": f"Model {model_name} is not a text generation model"}
        
        try:
            model = model_data["model"]
            tokenizer = model_data["tokenizer"]
            
            # Get prompt template
            template = PromptTemplateFactory.get_template(model_name)
            
            # Format prompt with template
            formatted_prompt = template.format_prompt(
                prompt, 
                system_prompt=kwargs.get("system_prompt")
            )
            
            # Get model-specific generation parameters
            gen_params = get_model_params(model_name, {
                "max_new_tokens": kwargs.get("max_tokens", 256),
                "temperature": kwargs.get("temperature", 0.7),
                "do_sample": kwargs.get("do_sample", True),
                "top_p": kwargs.get("top_p", 0.9),
            })
            
            # Set pad token if not set
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            
            # Tokenize input
            inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True).to(DEVICE)
            
            # Generate
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    **gen_params
                )
            
            generation_time = time.time() - start_time
            
            # Decode output
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract response using template
            response_text = template.extract_response(generated_text, formatted_prompt)
            
            # Calculate tokens per second
            num_tokens = len(outputs[0]) - len(inputs["input_ids"][0])
            tokens_per_second = num_tokens / generation_time if generation_time > 0 else 0
            
            return {
                "model": model_name,
                "prompt": prompt,
                "response": response_text,
                "generation_time": generation_time,
                "tokens_generated": num_tokens,
                "tokens_per_second": tokens_per_second
            }
            
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            return {"error": str(e)}
    
    def transcribe_audio(self, model_name: str, audio_path: str, **kwargs) -> Dict[str, Any]:
        """Transcribe audio using Whisper"""
        logger.info(f"Starting transcription with model: {model_name}, audio: {audio_path}")
        
        if model_name not in self.loaded_models:
            logger.info(f"Model {model_name} not loaded, attempting to load...")
            if not self.load_model(model_name):
                return {"error": f"Failed to load model {model_name}"}
        
        model_data = self.loaded_models[model_name]
        if model_data["type"] != "speech":
            return {"error": f"Model {model_name} is not a speech recognition model"}
        
        try:
            import librosa
            
            model = model_data["model"]
            processor = model_data["processor"]
            
            logger.info(f"Loading audio file: {audio_path}")
            
            # Check if file exists
            if not Path(audio_path).exists():
                return {"error": f"Audio file not found: {audio_path}"}
            
            # Get file size
            file_size = Path(audio_path).stat().st_size
            logger.info(f"Audio file size: {file_size} bytes")
            
            # Load and preprocess audio
            audio_loaded = False
            
            # Check file extension
            file_ext = Path(audio_path).suffix.lower()
            logger.info(f"Audio file extension: {file_ext}")
            
            # For WebM files, we need to convert to WAV first
            if file_ext in ['.webm', '.opus', '.mp4']:
                logger.info("WebM/Opus/MP4 format detected, converting to WAV...")
                try:
                    import subprocess
                    import tempfile
                    
                    # Create temporary WAV file
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
                        wav_path = tmp_wav.name
                    
                    # Try to find ffmpeg in common locations
                    ffmpeg_paths = ['/opt/homebrew/bin/ffmpeg', '/usr/local/bin/ffmpeg', 'ffmpeg']
                    ffmpeg_cmd = None
                    for ffmpeg_path in ffmpeg_paths:
                        if Path(ffmpeg_path).exists():
                            ffmpeg_cmd = ffmpeg_path
                            logger.info(f"Found ffmpeg at: {ffmpeg_cmd}")
                            break
                    
                    if not ffmpeg_cmd:
                        # Try using 'which' to find ffmpeg
                        try:
                            result = subprocess.run(['which', 'ffmpeg'], capture_output=True, text=True)
                            if result.returncode == 0 and result.stdout.strip():
                                ffmpeg_cmd = result.stdout.strip()
                                logger.info(f"Found ffmpeg via which: {ffmpeg_cmd}")
                        except:
                            pass
                    
                    if not ffmpeg_cmd:
                        return {"error": "ffmpeg not found. Please install with: brew install ffmpeg"}
                        
                    cmd = [
                        ffmpeg_cmd, '-i', audio_path,
                        '-acodec', 'pcm_s16le',
                        '-ar', '16000',  # Whisper expects 16kHz
                        '-ac', '1',  # Mono
                        '-y',  # Overwrite
                        wav_path
                    ]
                    
                    logger.info(f"Running ffmpeg command: {' '.join(cmd)}")
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode != 0:
                        logger.error(f"ffmpeg stderr: {result.stderr}")
                        logger.error(f"ffmpeg stdout: {result.stdout}")
                        return {"error": f"ffmpeg conversion failed: {result.stderr}"}
                    
                    logger.info(f"Successfully converted to WAV: {wav_path}")
                    
                    # Load the converted WAV file
                    audio, sr = librosa.load(wav_path, sr=16000)
                    logger.info(f"Audio loaded from converted WAV: shape={audio.shape}, sample_rate={sr}")
                    audio_loaded = True
                    
                    # Clean up temporary file
                    try:
                        Path(wav_path).unlink()
                    except:
                        pass
                        
                except Exception as convert_error:
                    logger.error(f"Error converting audio to WAV: {convert_error}", exc_info=True)
                    return {"error": f"Failed to convert audio: {str(convert_error)}"}
            
            # If not WebM or conversion failed, try direct loading
            if not audio_loaded:
                try:
                    audio, sr = librosa.load(audio_path, sr=16000)  # Whisper expects 16kHz
                    logger.info(f"Audio loaded with librosa: shape={audio.shape}, sample_rate={sr}")
                    audio_loaded = True
                except Exception as load_error:
                    logger.error(f"Error loading audio with librosa: {load_error}")
                    # Try with soundfile as fallback
                    try:
                        import soundfile as sf
                        audio, sr = sf.read(audio_path)
                        # Resample if needed
                        if sr != 16000:
                            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                            sr = 16000
                        logger.info(f"Audio loaded with soundfile: shape={audio.shape}, sample_rate={sr}")
                        audio_loaded = True
                    except Exception as sf_error:
                        logger.error(f"Error loading audio with soundfile: {sf_error}")
            
            if not audio_loaded:
                return {"error": "Failed to load audio file. Supported formats: WAV, MP3, FLAC. For WebM, install ffmpeg: brew install ffmpeg"}
            
            # Process audio - handle long audio by chunking if needed
            logger.info("Processing audio for Whisper model...")
            start_time = time.time()  # Start timing here for both paths
            
            # Check if audio is longer than 30 seconds (480,000 samples at 16kHz)
            max_samples = 30 * sr  # 30 seconds max per chunk
            if len(audio) > max_samples:
                logger.info(f"Audio is {len(audio)/sr:.1f}s long, chunking into 30s segments")
                
                # Process in chunks and concatenate results
                transcription_parts = []
                for i in range(0, len(audio), max_samples):
                    chunk = audio[i:i + max_samples]
                    logger.info(f"Processing chunk {i//max_samples + 1}, length: {len(chunk)/sr:.1f}s")
                    
                    inputs = processor(chunk, sampling_rate=sr, return_tensors="pt")
                    input_features = inputs["input_features"].to(DEVICE)
                    if model.dtype == torch.float16:
                        input_features = input_features.half()
                    
                    # Transcribe this chunk
                    with torch.no_grad():
                        # Remove forced_decoder_ids to avoid conflicts with newer transformers
                        generation_config = model.generation_config
                        if hasattr(generation_config, 'forced_decoder_ids'):
                            generation_config.forced_decoder_ids = None
                        
                        predicted_ids = model.generate(
                            input_features,
                            max_new_tokens=440,  # Set to 440 to stay under 448 limit (448 - 4 - 4 margin)
                            do_sample=False,
                            forced_decoder_ids=None,
                            task="transcribe",
                            language=None,
                            return_timestamps=False
                        )
                    
                    # Decode this chunk
                    chunk_transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                    chunk_text = chunk_transcription.strip()
                    transcription_parts.append(chunk_text)
                
                # Combine all chunks
                full_transcription = " ".join(transcription_parts)
                transcription_time = time.time() - start_time
                logger.info(f"Chunked transcription completed in {transcription_time:.2f} seconds")
                logger.info(f"Transcription result: '{full_transcription}'")
                
                return {
                    "model": model_name,
                    "audio_path": audio_path,
                    "transcription": full_transcription,
                    "transcription_time": transcription_time,
                    "chunks_processed": len(transcription_parts)
                }
            else:
                # Original single-chunk processing for shorter audio
                inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
                logger.info(f"Input features shape: {inputs['input_features'].shape}")
                
                # Move inputs to device and ensure correct dtype
                input_features = inputs["input_features"].to(DEVICE)
                if model.dtype == torch.float16:
                    input_features = input_features.half()
            
            # Transcribe
            logger.info("Starting transcription...")
            with torch.no_grad():
                # Remove forced_decoder_ids to avoid conflicts with newer transformers
                generation_config = model.generation_config
                if hasattr(generation_config, 'forced_decoder_ids'):
                    generation_config.forced_decoder_ids = None
                
                predicted_ids = model.generate(
                    input_features,
                    max_new_tokens=440,  # Set to 440 to stay under Whisper's 448 token limit
                    do_sample=False,
                    forced_decoder_ids=None,  # Explicitly set to None
                    task="transcribe",  # Use transcribe instead of translate
                    language=None,  # Auto-detect language instead of forcing English
                    return_timestamps=False  # Disable timestamps to get full text
                )
            
            transcription_time = time.time() - start_time
            logger.info(f"Transcription completed in {transcription_time:.2f} seconds")
            
            # Decode transcription
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            logger.info(f"Transcription result: '{transcription.strip()}'")
            
            return {
                "model": model_name,
                "audio_path": audio_path,
                "transcription": transcription.strip(),
                "transcription_time": transcription_time
            }
            
        except ImportError as e:
            logger.error(f"Import error: {e}")
            return {"error": "librosa is required for audio processing. Install with: pip install librosa"}
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}", exc_info=True)
            return {"error": str(e)}
    
    def analyze_image(self, model_name: str, image_path: str, prompt: str = None, **kwargs) -> Dict[str, Any]:
        """Analyze image using vision-language model"""
        if model_name not in self.loaded_models:
            if not self.load_model(model_name):
                return {"error": f"Failed to load model {model_name}"}
        
        model_data = self.loaded_models[model_name]
        if model_data["type"] != "vision-language":
            return {"error": f"Model {model_name} is not a vision-language model"}
        
        try:
            model = model_data["model"]
            processor = model_data["processor"]
            
            # Load image
            image = Image.open(image_path)
            
            # Process inputs based on model type
            if "blip2" in model_name:
                # BLIP-2 uses prompt
                if prompt is None:
                    prompt = "What is in this image?"
                inputs = processor(image, prompt, return_tensors="pt").to(DEVICE)
            elif "deepseek-vl" in model_name:
                # DeepSeek VL has a different API
                if prompt is None:
                    prompt = "Describe this image."
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
                inputs = processor(conversation, images=[image], return_tensors="pt").to(DEVICE)
            else:
                # BLIP uses different API
                if prompt and prompt != "What is in this image?":
                    # Conditional captioning
                    text = prompt
                    inputs = processor(image, text, return_tensors="pt").to(DEVICE)
                else:
                    # Unconditional captioning
                    inputs = processor(image, return_tensors="pt").to(DEVICE)
            
            # Generate
            start_time = time.time()
            with torch.no_grad():
                if "blip-base" in model_name and (not prompt or prompt == "What is in this image?"):
                    # Use unconditional generation for BLIP
                    outputs = model.generate(**inputs, max_length=30)
                elif "blip2" in model_name:
                    # BLIP-2 specific generation
                    outputs = model.generate(**inputs, max_new_tokens=50, num_beams=1)
                else:
                    outputs = model.generate(**inputs, max_new_tokens=100)
            
            generation_time = time.time() - start_time
            
            # Decode output
            response = processor.decode(outputs[0], skip_special_tokens=True)
            
            return {
                "model": model_name,
                "image_path": image_path,
                "prompt": prompt,
                "response": response,
                "generation_time": generation_time
            }
            
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            return {"error": str(e)}
    
    def infer(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Unified inference endpoint"""
        model_name = request.get("model_name")
        if not model_name:
            return {"error": "model_name is required"}
        
        # Determine inference type based on model or request
        if model_name not in self.model_configs:
            return {"error": f"Unknown model: {model_name}"}
        
        model_type = self.model_configs[model_name]["type"]
        
        if model_type == "text":
            prompt = request.get("prompt")
            if not prompt:
                return {"error": "prompt is required for text generation"}
            kwargs = {k: v for k, v in request.items() if k not in ["model_name", "prompt"]}
            return self.generate_text(model_name, prompt, **kwargs)
        
        elif model_type == "speech":
            audio_path = request.get("audio_path")
            if not audio_path:
                return {"error": "audio_path is required for speech recognition"}
            kwargs = {k: v for k, v in request.items() if k not in ["model_name", "audio_path"]}
            return self.transcribe_audio(model_name, audio_path, **kwargs)
        
        elif model_type == "vision-language":
            image_path = request.get("image_path")
            if not image_path:
                return {"error": "image_path is required for vision-language models"}
            prompt = request.get("prompt")
            kwargs = {k: v for k, v in request.items() if k not in ["model_name", "image_path", "prompt"]}
            return self.analyze_image(model_name, image_path, prompt, **kwargs)
        
        else:
            return {"error": f"Unknown model type: {model_type}"}


def main():
    """CLI interface for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Inference wrapper for local LLMs")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--prompt", help="Text prompt")
    parser.add_argument("--image", help="Image path")
    parser.add_argument("--audio", help="Audio path")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    
    args = parser.parse_args()
    
    # Create manager
    manager = ModelManager()
    
    # Build request
    request = {
        "model_name": args.model,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature
    }
    
    if args.prompt:
        request["prompt"] = args.prompt
    if args.image:
        request["image_path"] = args.image
    if args.audio:
        request["audio_path"] = args.audio
    
    # Run inference
    result = manager.infer(request)
    
    # Print result
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()