#!/usr/bin/env python3
"""
Basic sanity tests for downloaded models
"""

import json
import subprocess
import sys
import os
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent
INFERENCE_PATH = PROJECT_ROOT / "src" / "inference.py"

def test_deepseek_text():
    """Test DeepSeek text generation"""
    print("\n=== Testing DeepSeek-R1-Distill-Qwen 1.5B ===")
    
    prompts = [
        "What is 2 + 2?",
        "Write a haiku about coding.",
        "Explain quantum computing in one sentence."
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        result = subprocess.run([
            sys.executable, str(INFERENCE_PATH),
            "--model", "deepseek-r1-distill-qwen-1.5b",
            "--prompt", prompt,
            "--max-tokens", "50",
            "--temperature", "0.7"
        ], capture_output=True, text=True, cwd=str(PROJECT_ROOT))
        
        if result.returncode == 0:
            try:
                output = json.loads(result.stdout)
                if "error" in output:
                    print(f"Error: {output['error']}")
                else:
                    print(f"Response: {output['response'][:100]}...")
                    print(f"Tokens/sec: {output['tokens_per_second']:.2f}")
            except json.JSONDecodeError:
                print("Failed to parse output")
        else:
            print(f"Command failed: {result.stderr}")

def test_whisper():
    """Test Whisper (placeholder since audio loading not implemented)"""
    print("\n=== Testing Whisper Base ===")
    print("Note: Audio loading not yet implemented")
    
    result = subprocess.run([
        sys.executable, str(INFERENCE_PATH),
        "--model", "whisper-base",
        "--audio", "test_audio.wav"
    ], capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    
    if result.returncode == 0:
        output = json.loads(result.stdout)
        print(f"Result: {output}")

def test_blip2():
    """Test BLIP-2 with a sample image"""
    print("\n=== Testing BLIP-2 Base ===")
    
    # Create a simple test image
    from PIL import Image
    import numpy as np
    
    # Use the test image in the tests directory
    test_image_path = Path(__file__).parent / "test_image.png"
    if not test_image_path.exists():
        # Create a simple colored rectangle
        img_array = np.zeros((224, 224, 3), dtype=np.uint8)
        img_array[:112, :, 0] = 255  # Red top half
        img_array[112:, :, 2] = 255  # Blue bottom half
        img = Image.fromarray(img_array)
        img.save(test_image_path)
        print(f"Created test image: {test_image_path}")
    
    prompts = [
        "What colors are in this image?",
        "Describe this image.",
        None  # Test default prompt
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt or 'Default'}")
        cmd = [
            sys.executable, str(INFERENCE_PATH),
            "--model", "blip2-base",
            "--image", str(test_image_path)
        ]
        if prompt:
            cmd.extend(["--prompt", prompt])
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
        
        if result.returncode == 0:
            try:
                output = json.loads(result.stdout)
                if "error" in output:
                    print(f"Error: {output['error']}")
                else:
                    print(f"Response: {output.get('response', 'No response')}")
                    print(f"Time: {output.get('generation_time', 0):.2f}s")
            except json.JSONDecodeError:
                print("Failed to parse output")
        else:
            print(f"Command failed: {result.stderr}")

def main():
    """Run all tests"""
    print("Model Sanity Tests")
    print("=" * 60)
    
    # Test text generation
    test_deepseek_text()
    
    # Test speech (placeholder)
    test_whisper()
    
    # Test vision-language
    test_blip2()
    
    print("\n" + "=" * 60)
    print("Tests complete!")

if __name__ == "__main__":
    main()