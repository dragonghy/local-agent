#!/usr/bin/env python3
"""
Model download script for local LLM deployment
Supports text, vision-language, and speech models
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
import torch

# Model configurations
MODELS = {
    "llama-3.2-3b": {
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "type": "text",
        "description": "Llama 3.2 3B Instruct model"
    },
    "qwen2.5-1.5b": {
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "type": "text",
        "description": "Qwen 2.5 1.5B Instruct model (alternative to Llama)"
    },
    "deepseek-r1-distill-qwen-1.5b": {
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "type": "text", 
        "description": "DeepSeek R1 Distilled Qwen 1.5B model"
    },
    "deepseek-vl2-small": {
        "model_id": "deepseek-ai/deepseek-vl2-small",
        "type": "vision-language",
        "description": "DeepSeek VL2 Small vision-language model"
    },
    "whisper-base": {
        "model_id": "openai/whisper-base",
        "type": "speech",
        "description": "OpenAI Whisper base speech recognition model"
    },
    "blip2-base": {
        "model_id": "Salesforce/blip2-opt-2.7b",
        "type": "vision-language",
        "description": "BLIP-2 base vision-language model"
    },
    "blip-base": {
        "model_id": "Salesforce/blip-image-captioning-base",
        "type": "vision-language",
        "description": "BLIP base image captioning model (smaller)"
    },
    "deepseek-vl2-small": {
        "model_id": "deepseek-ai/deepseek-vl2-small",
        "type": "vision-language",
        "description": "DeepSeek VL2 Small vision-language model"
    }
}

def download_model(model_key, models_dir="models"):
    """Download a model from Hugging Face Hub"""
    if model_key not in MODELS:
        print(f"Unknown model: {model_key}")
        return False
    
    model_info = MODELS[model_key]
    model_path = Path(models_dir) / model_key
    
    print(f"\n{'='*60}")
    print(f"Downloading: {model_info['description']}")
    print(f"Model ID: {model_info['model_id']}")
    print(f"Type: {model_info['type']}")
    print(f"Destination: {model_path}")
    print(f"{'='*60}\n")
    
    try:
        # Create models directory if it doesn't exist
        model_path.parent.mkdir(exist_ok=True)
        
        # Download model
        snapshot_download(
            repo_id=model_info['model_id'],
            local_dir=str(model_path),
            local_dir_use_symlinks=False,
            ignore_patterns=["*.bin"] if model_info['type'] == 'text' else None  # Skip old format for text models
        )
        
        print(f"✅ Successfully downloaded {model_key}")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading {model_key}: {str(e)}")
        return False

def main():
    """Main download function"""
    print("Local LLM Model Downloader")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print("="*60)
    
    # Check if specific models requested
    if len(sys.argv) > 1:
        models_to_download = sys.argv[1:]
    else:
        # Download all models
        models_to_download = list(MODELS.keys())
    
    print(f"\nModels to download: {', '.join(models_to_download)}")
    
    # Download each model
    success_count = 0
    for model_key in models_to_download:
        if download_model(model_key):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Download complete: {success_count}/{len(models_to_download)} models downloaded successfully")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()