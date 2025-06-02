# Local LLM Deployment Prototype

Overall Project Goal

**Deliver a working proof‐of‐concept that demonstrates how DeepSeek, LLaMA 3, and Gemma 3 perform when deployed on local GPU/CPU resources, with a simple chat UI for manual testing. Use empirical benchmarks to decide which model(s) are practical for future local/offline products.**

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
- **Speech Model:**
  - Whisper base (145MB) - Speech transcription

### 📋 Pending
- Llama 3.2 3B (awaiting Meta access approval)

### 🌐 Web UI Status
- ✅ **Fully functional chat interface** at http://localhost:8000
- ✅ **Real-time streaming responses** with SSE (tokens appear as generated)
- ✅ **High-quality responses** with prompt templates and system prompts
- ✅ **Proper model selection** - users can switch between loaded models
- ✅ **Server management** via `./scripts/manage_web_ui.sh {start|stop|status}`

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

### Native Deployment (Leverages Apple Metal GPU)

```bash
# Install dependencies
pip install -r requirements.txt

# Download models
python scripts/download_models.py <model-name>

# Run web UI
python scripts/run_web_ui.py
# Or use the management script:
./scripts/manage_web_ui.sh start

# Access UI at http://localhost:8000
```

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
└── static/               # Web UI assets (future)