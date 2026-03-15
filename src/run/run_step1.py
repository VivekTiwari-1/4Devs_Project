"""
Main Runner Script - QMIX Multi-Agent Version
Run this file to execute the complete simulation with QMIX
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.simulator import Simulator, run_simulation
from src.config.config import TOTAL_SIMULATION_SLOTS


def main():
    """
    Main entry point for running the QMIX simulator.
    """
    print("\n" + "🔬" * 30)
    print("ENERGY-EFFICIENT CONTAINER CLOUD - QMIX MULTI-AGENT")
    print("8TH SEMESTER PROJECT")
    print("🔬" * 30 + "\n")
    
    # Configuration
    num_slots = TOTAL_SIMULATION_SLOTS
    workload_pattern = 'random'  # Options: 'random', 'poisson', 'bursty'
    placement_strategy = 'first_fit'  # Options: 'first_fit', 'best_fit', 'worst_fit'
    
    print(f"Configuration:")
    print(f"  - Simulation slots: {num_slots}")
    print(f"  - Workload pattern: {workload_pattern}")
    print(f"  - Placement strategy: {placement_strategy}")
    print(f"  - RL Algorithm: QMIX Multi-Agent")
    print("\n")
    
    # Run simulation
    sim = run_simulation(
        num_slots=num_slots,
        workload_pattern=workload_pattern,
        placement_strategy=placement_strategy,
        enable_rl=True,
        enable_migration=True
    )
    
    # Get and display summary
    summary = sim.get_summary_stats()
    
    print("\n📈 KEY PERFORMANCE INDICATORS:")
    print(f"  ✓ Containers processed: {summary['total_containers_finished']}/{summary['total_containers_arrived']}")
    print(f"  ✓ Success rate: {(1 - summary['rejection_rate']):.2%}")
    print(f"  ✓ Deadline compliance: {(1 - summary['violation_rate']):.2%}")
    print(f"  ✓ Peak PM utilization: {summary['peak_active_pms']} servers")
    print(f"  ✓ Total energy: {summary['total_energy_kwh']:.6f} kWh")
    print(f"  ✓ Estimated cost: ${summary['estimated_cost_usd']:.4f}")
    
    if 'qmix_num_agents' in summary:
        print(f"\n🤖 QMIX AGENT INFO:")
        print(f"  ✓ Number of agents: {summary['qmix_num_agents']}")
        print(f"  ✓ Total Q-entries: {summary['qmix_total_entries']}")
        print(f"  ✓ Replay buffer size: {summary['qmix_replay_size']}")
    
    print("\n💡 PROJECT STATUS:")
    print("  ✅ 8th Semester - QMIX Multi-Agent Implementation Complete")
    print("  ✅ Upgraded from 7th Semester Single-Agent Q-Learning")
    
    return sim


if __name__ == "__main__":
    sim = main()
