# Audio Transcription Guide

## Overview

The local LLM deployment includes a comprehensive audio transcription feature that supports multiple Whisper models and external APIs. Access the transcription interface at `https://localhost:8000/transcription`.

## Features

### 🎤 Recording Interface
- **Browser-based recording** using WebRTC MediaRecorder API
- **Up to 10 minutes** recording duration with automatic cutoff
- **Audio preview** with playback controls before transcription
- **One-step workflow** - Record → Confirm → Transcribe

### 🤖 Multi-Model Support
- **Local Whisper Models:**
  - Whisper Base (145MB) - Fast, basic accuracy
  - Whisper Small (244MB) - Balanced speed/accuracy  
  - Whisper Medium (2.9GB) - Better accuracy
  - Whisper Large-v2 (6.2GB) - High accuracy
  - Whisper Large-v3 (5.8GB) - Best accuracy, latest model

- **External APIs:**
  - OpenAI Whisper API - Cloud-based transcription
  - Google Gemini Speech API - Alternative cloud option

### 🌍 Language Support
- **Automatic language detection** - No manual selection needed
- **Native language output** - Chinese audio → Chinese text
- **Multilingual support** - 99+ languages supported by Whisper
- **No forced English translation** - Preserves original language

### ⚡ Performance Features
- **Automatic chunking** for audio longer than 30 seconds
- **Parallel processing** of multiple models simultaneously
- **Smart token limits** (440 tokens per chunk, under Whisper's 448 limit)
- **Seamless chunk combination** for complete long-form transcription

### 📝 Results & History
- **Side-by-side comparison** of different model outputs
- **Copy buttons** for each transcription result
- **Searchable history** sidebar with previous transcriptions
- **Persistent storage** using browser localStorage
- **Collapsible model selection** to save screen space

## Setup Instructions

### 1. Download Whisper Models

```bash
# Download individual models
python scripts/download_whisper_models.py whisper-base
python scripts/download_whisper_models.py whisper-medium  
python scripts/download_whisper_models.py whisper-large-v3

# Models are saved to ./models/ directory
```

### 2. Configure API Keys (Optional)

For external APIs, create a `.env` file:

```bash
# Copy the example file
cp .env.example .env

# Edit with your API keys
OPENAI_API_KEY=sk-your-openai-key-here
GEMINI_API_KEY=your-gemini-key-here
```

### 3. Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# Additional for external APIs (optional)
pip install openai google-generativeai
```

### 4. HTTPS Requirement

**Important:** Modern browsers require HTTPS for microphone access. The server automatically generates self-signed certificates.

```bash
# Run with HTTPS (required for microphone)
python src/web_app.py --https --port 8000

# Access at: https://localhost:8000/transcription
```

## Usage Guide

### Basic Workflow
1. **Open transcription page** - Navigate to https://localhost:8000/transcription
2. **Select models** - Choose which transcription models to use
3. **Record audio** - Click "🎤 Start Recording"
4. **Confirm recording** - Click "✅ Confirm & Transcribe" when done
5. **Review results** - Compare outputs from different models
6. **Copy text** - Use copy buttons to get transcription text

### Model Selection Tips
- **For speed:** Use Whisper Base (fastest, ~0.3s)
- **For accuracy:** Use Whisper Large-v3 (best quality, ~3-5s)
- **For comparison:** Select multiple models to compare outputs
- **For multilingual:** Large-v3 has best non-English support

### Long Audio Handling
- **30+ seconds:** Automatically chunked into 30-second segments
- **Up to 10 minutes:** Full support with complete transcription
- **Chunk visibility:** Logs show processing progress
- **Seamless output:** All chunks combined into single result

## Technical Details

### Audio Processing Pipeline
1. **Browser recording** → WebM/Opus format
2. **Server upload** → Temporary file storage  
3. **FFmpeg conversion** → WAV format (16kHz, mono)
4. **Model inference** → Whisper processing
5. **Result return** → JSON response with transcription

### Chunking Algorithm
```
For audio > 30 seconds:
├── Split into 30-second chunks
├── Process each chunk independently  
├── Each chunk: max 440 tokens (under 448 limit)
├── Combine all chunks with spaces
└── Return complete transcription
```

### Language Detection
- **Automatic detection** using Whisper's built-in capabilities
- **No language forcing** - preserves original audio language
- **Task=transcribe** instead of task=translate
- **Native output** - Chinese audio produces Chinese text

## Troubleshooting

### Microphone Issues
- **HTTPS required** - Use `--https` flag when starting server
- **Permission denied** - Allow microphone access in browser
- **No audio detected** - Check microphone settings and levels

### Model Issues
- **Model not found** - Download using `scripts/download_whisper_models.py`
- **Out of memory** - Try smaller models (base/small instead of large)
- **Slow performance** - Normal for large models (~3-5s vs ~0.3s for base)

### API Issues  
- **OpenAI 501 error** - Install: `pip install openai`
- **Gemini 501 error** - Install: `pip install google-generativeai`
- **Invalid API key** - Check `.env` file configuration

## Performance Benchmarks

| Model | Size | Speed | Quality | Languages |
|-------|------|-------|---------|-----------|
| Whisper Base | 145MB | ~0.3s | Good | 99+ |
| Whisper Medium | 2.9GB | ~1.0s | Better | 99+ |
| Whisper Large-v3 | 5.8GB | ~3-5s | Best | 99+ |
| OpenAI API | Cloud | ~1-2s | Excellent | 99+ |

*Benchmarks on Apple M4 with MPS acceleration*

## File Structure

```
static/
├── transcription.html    # Main transcription interface
├── transcription.css     # Styling and animations  
├── transcription.js      # Recording and UI logic
└── ...

src/
├── web_app.py           # FastAPI routes and endpoints
├── inference.py         # Whisper model inference
└── ...

.env.example             # API key template
requirements.txt         # Updated dependencies
```