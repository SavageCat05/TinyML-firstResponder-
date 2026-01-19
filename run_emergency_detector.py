#!/usr/bin/env python3
"""
Run script for TinyML Emergency Intent Detection System v1

Usage:
    python run_emergency_detector.py run          # Start emergency detection
    python run_emergency_detector.py train        # Train the model
    python run_emergency_detector.py benchmark    # Benchmark performance

Make sure to install dependencies first:
    pip install -r requirements.txt
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from main import main

if __name__ == '__main__':
    main()