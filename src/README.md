# Source Code

This directory contains the core application code.

## Files

- **inference.py** - Unified model inference wrapper that provides a consistent API for:
  - Text generation (DeepSeek, Qwen)
  - Vision-language understanding (BLIP, BLIP-2, DeepSeek-VL)
  - Speech recognition (Whisper)

## Usage

```bash
# Text generation
python src/inference.py --model qwen2.5-1.5b --prompt "Hello world"

# Image captioning
python src/inference.py --model blip-base --image path/to/image.png

# Visual question answering
python src/inference.py --model blip2-base --image path/to/image.png --prompt "What is in this image?"

# Speech recognition (pending audio implementation)
python src/inference.py --model whisper-base --audio path/to/audio.wav
```

## Architecture

The inference wrapper uses a unified `ModelManager` class that:
1. Dynamically loads models based on configuration
2. Provides consistent input/output format
3. Handles device management (MPS/CPU)
4. Tracks performance metrics

## Future Enhancements

- [ ] Streaming generation support
- [ ] Batch inference
- [ ] Model quantization options
- [ ] Audio processing for Whisper
- [ ] Web API endpoints