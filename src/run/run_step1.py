"""
Main Runner Script - QMIX Multi-Agent Version
Run this file to execute the complete simulation with QMIX.
Results are automatically saved to results/ after each run.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.simulator import Simulator, run_simulation
from src.config.config import TOTAL_SIMULATION_SLOTS
from src.utils.qmix_results_saver import QMIXResultsSaver   # ← new saver


def main():
    print("\n" + "🔬" * 30)
    print("ENERGY-EFFICIENT CONTAINER CLOUD - QMIX MULTI-AGENT")
    print("🔬" * 30 + "\n")

    # Configuration
    num_slots          = TOTAL_SIMULATION_SLOTS
    workload_pattern   = 'random'     # Options: 'random', 'poisson', 'bursty'
    placement_strategy = 'first_fit'  # Options: 'first_fit', 'best_fit', 'worst_fit'

    print(f"Configuration:")
    print(f"  - Simulation slots:    {num_slots}")
    print(f"  - Workload pattern:    {workload_pattern}")
    print(f"  - Placement strategy:  {placement_strategy}")
    print(f"  - RL Algorithm:        QMIX Multi-Agent")
    print("\n")

    # Run simulation
    sim = run_simulation(
        num_slots=num_slots,
        workload_pattern=workload_pattern,
        placement_strategy=placement_strategy,
        enable_rl=True,
        enable_migration=True
    )

    # ── Auto-save all results
    saver = QMIXResultsSaver(results_dir="results")
    saver.save_all(sim)

    # ── Print QMIX agent summary
    summary = sim.get_summary_stats()
    if 'qmix_num_agents' in summary:
        print(f"\n🤖 QMIX AGENT INFO:")
        print(f"  ✓ Number of agents:    {summary['qmix_num_agents']}")
        print(f"  ✓ Total Q-entries:     {summary['qmix_total_entries']}")
        print(f"  ✓ Replay buffer size:  {summary['qmix_replay_size']}")
        print(f"  ✓ Total updates:       {summary['qmix_total_updates']}")

    print("\n💡 To generate plots, run:")
    print("   python scripts/plot_qmix_results.py")
    print("   python scripts/plot_qmix_results.py --individual   (saves each graph separately)\n")

    return sim


if __name__ == "__main__":
    sim = main()