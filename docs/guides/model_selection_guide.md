# Model Selection Guide for Apple M4 (10 GPU Cores)

## Overview
This guide provides detailed information about available model checkpoints for DeepSeek, LLaMA 3, and Gemma 3, optimized for local deployment on Apple M4 hardware with unified memory architecture.

## Hardware Context
- **Target**: Apple M4 with 10 GPU cores
- **Memory**: Unified memory architecture (16GB/24GB/32GB variants)
- **Framework**: Metal Performance Shaders (MPS) via PyTorch
- **Optimization**: MLX framework for Apple Silicon

---

## DeepSeek Models

### Available Variants
| Model | Parameters | Hugging Face ID | Memory (Q4) | M4 Compatible |
|-------|------------|-----------------|-------------|---------------|
| DeepSeek-V3 | 671B (MoE) | `deepseek-ai/DeepSeek-V3` | 386GB+ | ❌ No |
| DeepSeek-R1 | 671B | `deepseek-ai/DeepSeek-R1` | 386GB+ | ❌ No |
| DeepSeek-Coder V2 | 16B | `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` | ~9GB | ⚠️ 24GB+ |
| DeepSeek-Coder | 33B | `deepseek-ai/deepseek-coder-33b-instruct` | ~20GB | ⚠️ 32GB+ |
| DeepSeek-Math | 7B | `deepseek-ai/deepseek-math-7b-instruct` | ~4GB | ✅ Yes |

### Recommended for M4
- **DeepSeek-Math 7B**: Best for mathematical reasoning
- **DeepSeek-Coder 7B**: Coding tasks (if available)

### Quantized Options
- GGUF format via `unsloth/DeepSeek-R1-GGUF`
- Various quantization levels: Q4_K_M, Q5_K, Q6_K

---

## LLaMA 3 Models

### Available Variants
| Model | Parameters | Hugging Face ID | Memory (Q4) | M4 Compatible |
|-------|------------|-----------------|-------------|---------------|
| LLaMA 3.2 | 1B | `meta-llama/Llama-3.2-1B-Instruct` | ~1GB | ✅ Excellent |
| LLaMA 3.2 | 3B | `meta-llama/Llama-3.2-3B-Instruct` | ~2GB | ✅ Excellent |
| LLaMA 3.1 | 8B | `meta-llama/Llama-3.1-8B-Instruct` | ~5GB | ✅ Excellent |
| LLaMA 3.2 Vision | 11B | `meta-llama/Llama-3.2-11B-Vision-Instruct` | ~7GB | ✅ 24GB+ |
| LLaMA 3.3 | 70B | `meta-llama/Llama-3.3-70B-Instruct` | ~40GB | ⚠️ 32GB+ |

### Apple Silicon Optimization
- Native Core ML support
- MLX framework optimization (~33 tokens/s on M1 Max)
- Metal acceleration for GPU compute

### Recommended for M4
- **LLaMA 3.1 8B**: Best overall performance/quality balance
- **LLaMA 3.2 3B**: Lightweight option for constrained memory
- **LLaMA 3.2 11B Vision**: Multimodal capabilities

---

## Gemma 3 Models

### Available Variants
| Model | Parameters | Hugging Face ID | Memory (Q4) | M4 Compatible |
|-------|------------|-----------------|-------------|---------------|
| Gemma 3 | 1B | `google/gemma-3-1b-it` | ~1GB | ✅ Excellent |
| Gemma 3 | 4B | `google/gemma-3-4b-it` | ~3GB | ✅ Excellent |
| Gemma 3 | 12B | `google/gemma-3-12b-it` | ~7GB | ✅ 24GB+ |
| Gemma 3 | 27B | `google/gemma-3-27b-it` | ~15GB | ⚠️ 32GB+ |

### Features
- 140+ language support
- Context windows: 32K (1B), 128K (4B+)
- Multimodal support (4B+ models)
- Official QAT (Quantization-Aware Training) models

### Recommended for M4
- **Gemma 3 4B**: Best size/performance trade-off
- **Gemma 3 1B**: Ultra-lightweight option

---

## Memory Requirements by M4 Configuration

### 16GB M4 (~12GB effective GPU memory)
**Recommended Models:**
- LLaMA 3.2 1B/3B (Q4): 1-2GB
- Gemma 3 1B/4B (Q4): 1-3GB
- DeepSeek-Math 7B (Q4): ~4GB
- LLaMA 3.1 8B (Q4): ~5GB

### 24GB M4 (~18GB effective GPU memory)
**Additional Options:**
- LLaMA 3.2 11B Vision (Q4): ~7GB
- DeepSeek-Coder 16B (Q4): ~9GB
- Gemma 3 12B (Q4): ~7GB

### 32GB M4 (~24GB effective GPU memory)
**High-End Options:**
- Gemma 3 27B (Q4): ~15GB
- DeepSeek-Coder 33B (Q4): ~20GB
- LLaMA 3.3 70B (Q4): ~40GB (requires optimization)

---

## Installation Methods

### Option 1: Ollama (Recommended for Simplicity)
```bash
# Install
brew install ollama

# Download models
ollama pull llama3.1:8b
ollama pull gemma3:4b
ollama pull deepseek-coder:7b
```

### Option 2: MLX (Apple Silicon Optimized)
```bash
pip install mlx-lm mlx-vlm
# Models automatically optimized for Apple Silicon
```

### Option 3: Hugging Face Transformers
```bash
# Install in our existing environment
source venv/bin/activate
pip install torch torchvision torchaudio transformers accelerate
```

### Option 4: llama.cpp (Maximum Control)
```bash
# Download GGUF quantized models
# Supports custom quantization levels and fine-tuning
```

---

## Deployment Strategy Recommendations

### Phase 1: Quick Start (Any M4)
1. **LLaMA 3.2 3B** - General chat, low memory
2. **Gemma 3 4B** - Balanced performance
3. **DeepSeek-Math 7B** - Specialized reasoning

### Phase 2: Performance Testing (24GB+ M4)
1. **LLaMA 3.1 8B** - Best general performance
2. **LLaMA 3.2 11B Vision** - Multimodal testing
3. **Gemma 3 12B** - High-quality text generation

### Phase 3: Advanced Deployment (32GB M4)
1. **LLaMA 3.3 70B** - State-of-the-art performance
2. **DeepSeek-Coder 33B** - Advanced coding tasks
3. **Gemma 3 27B** - Premium text generation

---

## Performance Considerations

### Quantization Quality Hierarchy
1. **FP16**: Full precision, maximum quality, highest memory
2. **Q8_0**: 8-bit quantization, minimal quality loss
3. **Q6_K**: 6-bit, good quality/size balance
4. **Q5_K_M**: 5-bit, recommended for most use cases
5. **Q4_K_M**: 4-bit, best size/quality trade-off
6. **Q3_K_M**: 3-bit, aggressive compression
7. **Q2_K**: 2-bit, maximum compression, noticeable quality loss

### Apple Silicon Specific Optimizations
- Use MLX framework when available
- Enable Metal acceleration in PyTorch
- Consider unified memory architecture in batch sizing
- Monitor thermal throttling during sustained inference

### Context Length Impact
- Reserve 20-30% additional memory for context processing
- Longer contexts require proportionally more memory
- Consider sliding window attention for very long documents