"""
Main entry point for Energy-Efficient Container Cloud Simulator
8th Semester - QMIX Multi-Agent Version
"""
# Add src to path

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# from scripts.train_rl import train_rl_agent
# from scripts.compare_all import run_comparison
# from scripts.plot_results import plot_training_progress, plot_comparison

# def main():
#     print("""
#     🎓 Energy-Efficient Container Cloud - Mini Project
    
#     Select mode:
#     1. Train RL Agent (30 episodes)
#     2. Run Comparison (Baseline vs RL vs RL+Migration)
#     3. Generate Plots
#     4. Full Pipeline (Train → Compare → Plot)
#     5. Quick Demo (1 episode)
#     """)
    
#     choice = input("Enter choice (1-5): ").strip()
    
#     if choice == '1':
#         train_rl_agent()
#     elif choice == '2':
#         run_comparison(slots=50, num_runs=3)
#     elif choice == '3':
#         plot_training_progress()
#         plot_comparison()
#     elif choice == '4':
#         print("\n🚀 Running Full Pipeline...\n")
#         train_rl_agent()
#         run_comparison(slots=50, num_runs=3)
#         plot_training_progress()
#         plot_comparison()
#     elif choice == '5':
#         from environment.simulator import Simulator
#         sim = Simulator(enable_rl=True, enable_migration=True)
#         sim.run(num_slots=50)
#     else:
#         print("❌ Invalid choice")


from run.run_step1 import main

if __name__ == "__main__":
    main()
