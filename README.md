# Local LLM Agent - Run AI Models Locally on Apple Silicon

A high-performance local AI system designed for Apple Silicon Macs (M1/M2/M3/M4). Features include text generation, speech transcription, and image understanding - all running privately on your device.

**Key Features:**
- 🚀 Optimized for Apple Metal GPU acceleration (17-20 tokens/sec)
- 💬 Chat interface with real-time streaming responses
- 🎤 Professional transcription with multiple Whisper models
- 🖼️ Vision capabilities for image analysis
- 🔒 100% private - all processing happens locally
- 🌐 Optional cloud API integration (OpenAI, Gemini)

## 1. Project Context
- **Motivation**: Explore feasibility of running modern large language models (LLMs) locally or via remote GPU, and assess performance for potential on-device or self-hosted applications.
- **Models Under Consideration**:
  - DeepSeek
  - LLaMA 3
  - Google Gemma 3
- **Target Environment**:
  - Apple Mac mini M4 with 10 GPU cores and Metal 3 support
  - Unified memory architecture (16GB/24GB/32GB variants)
  - PyTorch with MPS (Metal Performance Shaders) backend
- **UI Prototype**:
  - Minimal terminal‐based chat interface (e.g., Python + readline/curses)
  - Optional lightweight web UI (e.g., Flask or FastAPI with a simple HTML/JS front end)

## 2. High-Level Goals
1. **Environment Setup** ✅ COMPLETED 
   - Verified Apple M4 with 10 GPU cores and Metal 3 support
   - Installed Python 3.9.6 with PyTorch 2.7.0 (MPS backend)
   - Set up virtual environment with core ML dependencies
2. **Model Acquisition & Configuration** 📋 RESEARCHED
   - Identified optimal models for Apple M4: LLaMA 3.1 8B, Gemma 3 4B, DeepSeek-Math 7B
   - Documented memory requirements and Apple Silicon compatibility
   - Created comprehensive model selection guide
3. **UI Prototype**  
   - Build a simple “chat” interface to send prompts and receive responses
   - Ensure minimal dependencies so it can run in a terminal; optionally extend to a basic web page
4. **Benchmark & Evaluation**  
   - Measure text generation throughput (tokens/sec) for each model under comparable settings
   - (If applicable) Test any built-in vision or speech features for latency and resource usage
   - Document observed memory/VRAM usage and CPU/GPU utilization
5. **Next‐Phase Exploration**  
   - Based on initial performance, identify feasible local/offline use cases (e.g., lightweight assistant, edge vision tasks, basic speech)
   - Outline potential integration points or product ideas leveraging local inference

## 3. Current Progress Status

### ✅ Completed
- Environment setup with Apple M4 + MPS support
- Model research and selection guide
- Project structure and documentation
- Knowledge base with Apple Silicon specifics
- **Downloaded and configured 5 models successfully** (Total: ~57GB)
- **Created unified inference wrapper supporting text, vision-language, and speech models**
- **Achieved ~17-20 tokens/sec performance on text models**

### 🚀 Models Successfully Deployed
- **Text Models:**
  - DeepSeek-R1-Distill-Qwen 1.5B (3.1GB) - 17-20 tokens/sec
  - Qwen 2.5 1.5B (3.3GB) - 18-20 tokens/sec
- **Vision-Language Models:**
  - BLIP base (446MB) - Image captioning
  - BLIP-2 2.7B (15GB) - Advanced VQA
  - DeepSeek-VL2-Small (35GB) - Multi-modal understanding
- **Speech Models:**
  - Whisper base (145MB) - Fast speech transcription
  - Whisper medium (2.9GB) - Improved accuracy
  - Whisper large-v3 (5.8GB) - Highest accuracy multilingual

### 📋 Pending
- Llama 3.2 3B (awaiting Meta access approval)

### 🌐 Web UI Status
- ✅ **Fully functional chat interface** at https://localhost:8000
- ✅ **Real-time streaming responses** with SSE (tokens appear as generated)
- ✅ **High-quality responses** with prompt templates and system prompts
- ✅ **Proper model selection** - users can switch between loaded models
- ✅ **Server management** via `./scripts/manage_web_ui.sh {start|stop|status}`
- ✅ **Professional transcription interface** at https://localhost:8000/transcription
- ✅ **Multi-model transcription** - Compare local Whisper models vs OpenAI/Gemini APIs
- ✅ **Long audio support** - Handles recordings up to 10 minutes with automatic chunking
- ✅ **Language auto-detection** - Native language output (Chinese, English, etc.)
- ✅ **Transcription history** - Searchable sidebar with previous results

### 📋 Next Steps
- 🟡 **MEDIUM**: Complete benchmarking suite
- 🟡 **MEDIUM**: Add conversation history persistence
- 🟡 **MEDIUM**: Implement model comparison mode
- ⏳ **PENDING**: Llama 3.2 3B (awaiting access)

### 📁 Documentation Created
- `knowledge_base.md` - Apple M4 environment details
- `docs/model_selection_guide.md` - Comprehensive model analysis
- `requirements.txt` - Python dependencies
- `.gitignore` - Excludes model files from version control

See [Tasks.md](./Tasks.md) for detailed tasks breakdown and status.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ 
- macOS with Apple Silicon (M1/M2/M3/M4) for GPU acceleration
- 16GB+ RAM recommended
- 50GB+ free disk space for models
- Homebrew (for system dependencies)

### Installation

```bash
# 1. Install system dependencies
brew install ffmpeg       # Required for audio processing
brew install portaudio    # Required for audio recording (optional)

# 2. Clone the repository
git clone https://github.com/yourusername/local-agent.git
cd local-agent

# 3. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On macOS/Linux

# 4. Install dependencies
pip install -r requirements.txt

# 5. Download models (choose based on your needs)
# Text generation models
python scripts/download_models.py qwen2.5-1.5b          # Recommended starter model
python scripts/download_models.py deepseek-r1-distill-qwen-1.5b

# Speech transcription models  
python scripts/download_whisper_models.py whisper-base   # Fast transcription
python scripts/download_whisper_models.py whisper-large-v3  # Best accuracy

# Vision-language models (optional)
python scripts/download_models.py blip-base              # Image captioning

# 6. Configure API keys (optional, for external services)
cp .env.example .env
# Edit .env with your OpenAI/Gemini API keys

# 7. Generate SSL certificates (required for HTTPS/microphone access)
mkdir -p ssl
openssl req -x509 -newkey rsa:4096 -keyout ssl/key.pem -out ssl/cert.pem -days 365 -nodes \
  -subj "/C=US/ST=Local/L=Local/O=LocalLLM/CN=localhost"

# 8. Run the web server
python src/web_app.py --https --port 8000

# 9. Access the UI
# Chat interface: https://localhost:8000
# Transcription: https://localhost:8000/transcription
```

### First Run Notes
- **HTTPS Warning**: Browser will show security warning due to self-signed certificate - this is normal, click "Advanced" → "Proceed"
- **Model Loading**: First model load takes 5-10 seconds, subsequent loads are faster
- **Memory Usage**: Each model uses 3-15GB RAM depending on size

### Available Models
| Model | Size | Purpose | Download Command |
|-------|------|---------|------------------|
| qwen2.5-1.5b | 3.3GB | Fast text generation | `python scripts/download_models.py qwen2.5-1.5b` |
| deepseek-r1-distill-qwen-1.5b | 3.1GB | Quality text generation | `python scripts/download_models.py deepseek-r1-distill-qwen-1.5b` |
| whisper-base | 145MB | Fast transcription | `python scripts/download_whisper_models.py whisper-base` |
| whisper-large-v3 | 5.8GB | Best transcription | `python scripts/download_whisper_models.py whisper-large-v3` |
| blip-base | 446MB | Image captioning | `python scripts/download_models.py blip-base` |

**Note**: This project uses native deployment to leverage Apple's Metal Performance Shaders (MPS) for GPU acceleration, achieving ~17-20 tokens/sec on M4 hardware. Docker is not recommended as it cannot access MPS, resulting in significantly slower CPU-only performance.

## 📁 Project Structure

```
.
├── src/                    # Core application code
│   └── inference.py       # Unified model inference wrapper
├── scripts/               # Utility scripts
│   └── download_models.py # Model download automation
├── tests/                 # Test files
│   ├── test_models.py    # Model validation tests
│   └── test_image.png    # Sample image for vision models
├── docs/                  # Documentation
│   ├── guides/           # How-to guides
│   │   ├── model_download_guide.md
│   │   └── model_selection_guide.md
│   └── knowledge_base.md # Apple Silicon deployment notes
├── benchmarks/           # Performance results
│   └── initial_results_summary.md
├── models/               # Downloaded models (gitignored)
├── logs/                 # Runtime logs (gitignored)
└── static/               # Web UI assets
```

## 🔧 Troubleshooting

### Common Issues

**"Connection not private" warning in browser**
- This is normal due to self-signed HTTPS certificate
- Click "Advanced" → "Proceed to localhost"

**"SSL certificates not found" error**
- Run the SSL generation command from step 6:
  ```bash
  mkdir -p ssl
  openssl req -x509 -newkey rsa:4096 -keyout ssl/key.pem -out ssl/cert.pem -days 365 -nodes \
    -subj "/C=US/ST=Local/L=Local/O=LocalLLM/CN=localhost"
  ```

**"ffmpeg not found" or audio conversion errors**
- Install ffmpeg: `brew install ffmpeg`
- Verify installation: `ffmpeg -version`

**Model download fails**
- Check internet connection
- Ensure sufficient disk space (models are 0.1-35GB)
- Try downloading one model at a time

**Out of memory errors**
- Close other applications
- Try smaller models (whisper-base instead of whisper-large)
- Restart the server between model switches

**Microphone not working**
- Ensure HTTPS is enabled (`--https` flag)
- Allow microphone permissions in browser
- Check System Preferences → Security & Privacy → Microphone

**Slow performance**
- Verify MPS is detected: Check logs for "Using device: mps"
- First inference is always slower (model loading)
- Consider using smaller models for testing

### Getting Help

- 📖 See [docs/transcription_guide.md](docs/transcription_guide.md) for detailed transcription setup
- 🐛 Report issues: [GitHub Issues](https://github.com/yourusername/local-agent/issues)
- 💡 Check [docs/model_selection_guide.md](docs/model_selection_guide.md) for model recommendations

## 📄 License

This project is open source and available under the MIT License.