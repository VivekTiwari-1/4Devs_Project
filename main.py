"""
Main entry point for Energy-Efficient Container Cloud Simulator
8th Semester - QMIX Multi-Agent Version
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from run.run_step1 import main

if __name__ == "__main__":
    main()
