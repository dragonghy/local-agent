# Local LLM Deployment - Knowledge Base

## Environment Information

### Hardware Setup
- **Device**: Mac mini M4
- **CPU**: Apple M4 chip
- **GPU**: Apple M4 with 10 GPU cores
- **Metal Support**: Metal 3
- **Architecture**: ARM64 (Apple Silicon)

### Display Configuration
- Primary: DELL S3222DGM (2560 x 1440 @ 144Hz)
- Secondary: Q27G4N (1440 x 2560 @ 72Hz, rotated 90°)

### Software Environment
- **OS**: macOS (Darwin)
- **Python**: 3.9.6 (system default)
- **GPU Framework**: Metal Performance Shaders (MPS) - NOT CUDA
- **PyTorch Backend**: MPS (Metal Performance Shaders)

## Key Technical Considerations

### GPU Acceleration on Apple Silicon
- **No NVIDIA CUDA**: Apple Silicon uses Metal framework, not CUDA
- **PyTorch MPS Backend**: Use `torch.backends.mps.is_available()` to check MPS support
- **Device Selection**: Use `torch.device("mps")` instead of `torch.device("cuda")`
- **Memory Management**: Apple Silicon uses unified memory architecture

### Model Deployment Considerations
- **Memory**: Unified memory shared between CPU and GPU
- **Quantization**: ARM64-optimized quantization methods may differ from x86_64
- **Model Formats**: Ensure model weights are compatible with ARM64 architecture
- **Performance**: Metal-optimized operations may have different performance characteristics

### Dependencies for Apple Silicon
- **PyTorch**: Install with MPS support (`torch` with Metal backend)
- **Transformers**: Hugging Face library with MPS compatibility
- **Accelerate**: For optimized model loading and inference
- **Model-specific SDKs**: Verify ARM64/MPS compatibility

## Environment Setup Commands

### Python Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### Core Dependencies (Apple Silicon)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers accelerate
pip install fastapi uvicorn
pip install numpy pandas matplotlib
```

### Verification Commands
```bash
# Check MPS availability
python -c "import torch; print(torch.backends.mps.is_available())"

# Check Metal GPU info
system_profiler SPDisplaysDataType

# Check CPU info
sysctl -n machdep.cpu.brand_string
```

## Model-Specific Notes

### DeepSeek
- Verify ARM64 compatibility for model weights
- Check if custom inference code supports MPS backend

### LLaMA 3
- Use Hugging Face transformers with MPS backend
- Consider using Apple's MLX framework for optimized inference

### Gemma 3
- Ensure Google's model implementations support Apple Silicon
- May require specific configuration for MPS backend

## Performance Considerations
- **Memory Usage**: Monitor unified memory usage with Activity Monitor
- **Thermal Management**: Apple Silicon thermal throttling under sustained load
- **Batch Size**: Optimize for unified memory architecture
- **Model Size**: 10 GPU cores may limit largest model variants

## Troubleshooting
- If MPS not available, fallback to CPU-only inference
- Use `torch.mps.empty_cache()` for memory management
- Monitor temperature and performance with built-in tools