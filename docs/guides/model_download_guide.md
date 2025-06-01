# Model Download Guide

This guide documents the process, challenges, and workarounds for downloading models on Apple Silicon, based on our experience with the Apple M4 Mac mini.

## Overview

We successfully downloaded 5 models totaling ~57GB. This guide will help you replicate the process and avoid common pitfalls.

## Prerequisites

1. **Hugging Face Authentication**
   ```bash
   pip install huggingface-hub
   huggingface-cli login
   ```
   - Required for gated models like Llama
   - Get your token from https://huggingface.co/settings/tokens

2. **Storage Requirements**
   - Ensure at least 100GB free space (models + temporary downloads)
   - Models use ~57GB when complete
   - Downloads may use 2x space temporarily

## Quick Start

```bash
# Activate virtual environment
source venv/bin/activate

# Download a model
python scripts/download_models.py <model-name>

# Available models:
# - deepseek-r1-distill-qwen-1.5b
# - qwen2.5-1.5b
# - whisper-base
# - blip-base
# - blip2-base
# - deepseek-vl2-small
```

## Model-Specific Instructions

### 1. DeepSeek-R1-Distill-Qwen 1.5B ✅
**Size**: 3.3GB  
**Status**: Downloads without issues
```bash
python scripts/download_models.py deepseek-r1-distill-qwen-1.5b
```

### 2. Qwen 2.5 1.5B ✅
**Size**: 2.9GB  
**Status**: Downloads without issues
```bash
python scripts/download_models.py qwen2.5-1.5b
```

### 3. Whisper Base ✅
**Size**: 1.1GB  
**Status**: Downloads without issues
```bash
python scripts/download_models.py whisper-base
```

### 4. BLIP Base ✅
**Size**: 877MB  
**Status**: Downloads without issues
```bash
python scripts/download_models.py blip-base
```

### 5. BLIP-2 2.7B ⚠️
**Size**: 17GB (2 files: 10GB + 5GB)  
**Challenge**: Large model, download may timeout

**Workaround**: Download files separately using huggingface-cli
```bash
# If timeout occurs, use huggingface-cli to download individual files:
huggingface-cli download Salesforce/blip2-opt-2.7b \
  --include "model-00001-of-00002.safetensors" \
  --local-dir models/blip2-base \
  --local-dir-use-symlinks False

huggingface-cli download Salesforce/blip2-opt-2.7b \
  --include "model-00002-of-00002.safetensors" \
  --local-dir models/blip2-base \
  --local-dir-use-symlinks False
```

### 6. DeepSeek-VL2-Small ⚠️
**Size**: ~30GB (4 files: 8.5GB each)  
**Challenge**: Very large model, downloads often timeout

**Workaround**: Download files individually
```bash
# Download each file separately to avoid timeouts:
for i in {1..4}; do
  huggingface-cli download deepseek-ai/deepseek-vl2-small \
    --include "model-0000${i}-of-000004.safetensors" \
    --local-dir models/deepseek-vl2-small \
    --local-dir-use-symlinks False
done
```

### 7. Llama 3.2 3B ❌
**Size**: ~6GB  
**Challenge**: Gated model requiring approval
**Status**: Access must be requested at https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct

**Process**:
1. Visit the model page
2. Click "Request access"
3. Accept license terms
4. Wait for approval (can take hours/days)
5. Once approved, download normally

## Common Issues and Solutions

### 1. Download Timeouts

**Issue**: Large models timeout after 10 minutes  
**Solution**: Use huggingface-cli to download individual files
```bash
# Check partial downloads
ls -la models/<model-name>/.cache/huggingface/download/*.incomplete

# Resume download of specific files
huggingface-cli download <model-id> --include "*.safetensors" \
  --local-dir models/<model-name> --local-dir-use-symlinks False
```

### 2. Authentication Errors

**Issue**: 401/403 errors for gated models  
**Solution**: 
```bash
# Ensure you're logged in
huggingface-cli whoami

# Re-login if needed
huggingface-cli login
```

### 3. Incomplete Downloads

**Issue**: Model fails to load due to missing files  
**Solution**: Check for required files
```bash
# List downloaded files
ls -la models/<model-name>/*.safetensors

# Check index file for required files
cat models/<model-name>/model.safetensors.index.json | grep -o '"model.*safetensors"' | sort -u
```

### 4. Storage Space

**Issue**: Running out of space during download  
**Solution**: 
- Clear incomplete downloads: `rm -rf models/*/.cache`
- Download models one at a time
- Use external storage if needed

## Performance Tips

1. **Network Speed**: Use ethernet if possible for large models
2. **Parallel Downloads**: Don't download multiple large models simultaneously
3. **Partial Downloads**: huggingface-cli can resume interrupted downloads
4. **Verification**: Always test models after download with test_models.py

## Directory Structure

```
models/
├── deepseek-r1-distill-qwen-1.5b/
├── qwen2.5-1.5b/
├── whisper-base/
├── blip-base/
├── blip2-base/
├── deepseek-vl2-small/
└── llama-3.2-3b/  # Empty until access granted
```

## Testing Downloads

After downloading, verify models work:
```bash
# Test text model
python src/inference.py --model qwen2.5-1.5b --prompt "Hello world"

# Test vision model
python src/inference.py --model blip-base --image tests/test_image.png

# Run all tests
python tests/test_models.py
```

## Lessons Learned

1. **Start Small**: Download smaller models first to test your setup
2. **Gated Models**: Check access requirements before attempting download
3. **Timeout Strategy**: For models >10GB, plan to use huggingface-cli
4. **Disk Space**: Keep 2x the model size available during download
5. **Model Variants**: Some models have multiple formats (.bin, .safetensors) - we use safetensors
6. **Apple Silicon**: All models work with MPS backend, no CUDA needed

## Troubleshooting Checklist

- [ ] Logged into Hugging Face?
- [ ] Sufficient disk space (2x model size)?
- [ ] Using virtual environment?
- [ ] Network connection stable?
- [ ] For large models, using huggingface-cli?
- [ ] For gated models, access approved?

## Next Steps

Once models are downloaded:
1. Test with `python tests/test_models.py`
2. Use `src/inference.py` for individual model testing
3. Implement web UI for easier interaction
4. Run comprehensive benchmarks

## Support

If you encounter issues not covered here:
1. Check model-specific documentation on Hugging Face
2. Verify your PyTorch MPS setup: `python -c "import torch; print(torch.backends.mps.is_available())"`
3. Check available storage: `df -h`
4. Review download logs in terminal output