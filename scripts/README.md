# Scripts

Utility scripts for model management and automation.

## Files

- **download_models.py** - Automated model downloader from Hugging Face Hub

## Usage

```bash
# Download a specific model
python scripts/download_models.py <model-name>

# Available models:
# - deepseek-r1-distill-qwen-1.5b
# - qwen2.5-1.5b  
# - whisper-base
# - blip-base
# - blip2-base
# - deepseek-vl2-small
# - llama-3.2-3b (requires access approval)

# Download all models
python scripts/download_models.py
```

## Features

- Automatic retry on failure
- Progress tracking
- Handles large model files
- Skips already downloaded models
- Provides clear error messages

## Notes

- For large models (>10GB), consider using `huggingface-cli` directly
- Requires authentication for gated models
- Downloads to `models/` directory