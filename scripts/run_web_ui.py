#!/usr/bin/env python3
"""
Launch script for the web UI
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))

# Import and run the web app
from web_app import main

if __name__ == "__main__":
    print("Starting Local LLM Web UI...")
    print("Open http://localhost:8000 in your browser")
    print("Press Ctrl+C to stop the server")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)