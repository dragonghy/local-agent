#!/usr/bin/env python3
"""
Download Whisper models for better transcription accuracy
"""

import os
import sys
from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def download_whisper_model(model_name: str, save_path: Path):
    """Download and save a Whisper model"""
    print(f"Downloading {model_name}...")
    
    # Create directory
    save_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download processor
        processor = WhisperProcessor.from_pretrained(f"openai/{model_name}")
        processor.save_pretrained(save_path)
        print(f"✓ Processor downloaded to {save_path}")
        
        # Download model
        model = WhisperForConditionalGeneration.from_pretrained(f"openai/{model_name}")
        model.save_pretrained(save_path)
        print(f"✓ Model downloaded to {save_path}")
        
        print(f"✅ {model_name} downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download {model_name}: {e}")
        return False

def main():
    """Download Whisper models"""
    models_dir = Path("models")
    
    # Available models (in order of size/accuracy)
    models = {
        "whisper-small": "244M parameters - Good balance of speed/accuracy",
        "whisper-medium": "769M parameters - High accuracy",  
        "whisper-large-v2": "1550M parameters - Best accuracy, handles fast speech",
        "whisper-large-v3": "1550M parameters - Latest model, excellent performance"
    }
    
    print("🎤 Whisper Model Downloader")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        # Download specific model
        model_name = sys.argv[1]
        if model_name in models:
            save_path = models_dir / model_name
            download_whisper_model(model_name, save_path)
        else:
            print(f"❌ Unknown model: {model_name}")
            print(f"Available models: {list(models.keys())}")
    else:
        # Interactive selection
        print("Available models:")
        for i, (model, desc) in enumerate(models.items(), 1):
            print(f"{i}. {model} - {desc}")
        
        print("\nRecommendations:")
        print("• whisper-small: Good upgrade from base, 3x better accuracy")
        print("• whisper-medium: Significant improvement for fast speech")  
        print("• whisper-large-v3: Best quality, handles all speech patterns")
        
        try:
            choice = input("\nEnter model number to download (or 'all' for all): ").strip()
            
            if choice.lower() == 'all':
                for model_name in models.keys():
                    save_path = models_dir / model_name
                    download_whisper_model(model_name, save_path)
            else:
                model_list = list(models.keys())
                if choice.isdigit() and 1 <= int(choice) <= len(model_list):
                    model_name = model_list[int(choice) - 1]
                    save_path = models_dir / model_name
                    download_whisper_model(model_name, save_path)
                else:
                    print("❌ Invalid choice")
                    
        except KeyboardInterrupt:
            print("\n👋 Download cancelled")

if __name__ == "__main__":
    main()