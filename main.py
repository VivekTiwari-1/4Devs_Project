"""
Main entry point - Run this to start simulation
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from scripts.run_step_1 import main

if __name__ == "__main__":
    main()