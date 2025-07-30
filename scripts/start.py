#!/usr/bin/env python3
'''
Cross-platform startup script for AI On-Call Agent
'''

import sys
import subprocess
import os
from pathlib import Path

def start_system(mode="dev"):
    print(f"🚀 Starting AI On-Call Agent - {mode.upper()} Mode")
    
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    if mode == "dev":
        # Development mode
        try:
            print("🌟 Starting development server...")
            subprocess.run([
                sys.executable, "-m", "uvicorn", 
                "src.main:app", 
                "--host", "localhost", 
                "--port", "8000", 
                "--reload"
            ], check=True)
        except KeyboardInterrupt:
            print("\n👋 Development server stopped")
    
    elif mode == "prod":
        # Production mode
        try:
            print("🏭 Starting production server...")
            subprocess.run([
                sys.executable, "-m", "gunicorn",
                "--host", "0.0.0.0",
                "--port", "8000", 
                "--workers", "4",
                "src.main:app"
            ], check=True)
        except KeyboardInterrupt:
            print("\n👋 Production server stopped")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Start AI On-Call Agent")
    parser.add_argument("--mode", choices=["dev", "prod"], default="dev",
                      help="Startup mode")
    args = parser.parse_args()
    
    start_system(args.mode)
