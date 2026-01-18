"""
Main Runner Script for Step 1
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.simulator import Simulator

def main():
    print("\n" + "🔬" * 30)
    print("ENERGY-EFFICIENT CONTAINER CLOUD SIMULATOR")
    print("🔬" * 30 + "\n")
    
    sim = Simulator(workload_pattern='bursty', placement_strategy='first_fit')
    sim.run(num_slots=50)
    
    return sim

if __name__ == "__main__":
    main()