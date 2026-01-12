#!/usr/bin/env python3
"""
Launch script for the Financial Sentiment Analysis demo.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch the Streamlit demo application."""
    demo_path = Path(__file__).parent / "demo" / "app.py"
    
    if not demo_path.exists():
        print(f"Error: Demo file not found at {demo_path}")
        sys.exit(1)
    
    print("🚀 Launching Financial Sentiment Analysis Demo...")
    print("⚠️  Remember: This is for research/educational purposes only - NOT for investment advice!")
    print()
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(demo_path),
            "--server.port", "8501",
            "--server.address", "localhost"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error launching demo: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
        sys.exit(0)

if __name__ == "__main__":
    main()
