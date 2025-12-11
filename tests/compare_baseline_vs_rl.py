"""
Compare Baseline vs RL Performance
Runs both modes and shows improvements
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.simulator import Simulator


def run_comparison(num_slots=50):
    """
    Run baseline and RL simulations and compare results.
    
    Args:
        num_slots (int): Number of slots to simulate
    """
    print("\n" + "="*80)
    print("🔬 BASELINE vs RL COMPARISON")
    print("="*80 + "\n")
    
    # Run baseline (no RL)
    print("▶️  Running BASELINE simulation (no RL)...")
    print("-"*80)
    baseline_sim = Simulator(enable_rl=False)
    baseline_sim.run(num_slots=num_slots)
    baseline_stats = baseline_sim.get_summary_stats()
    
    print("\n\n")
    
    # Run RL-enabled
    print("▶️  Running RL-ENABLED simulation...")
    print("-"*80)
    rl_sim = Simulator(enable_rl=True)
    rl_sim.run(num_slots=num_slots)
    rl_stats = rl_sim.get_summary_stats()
    
    # Print comparison
    print("\n" + "="*80)
    print("📊 COMPARISON RESULTS")
    print("="*80 + "\n")
    
    print(f"{'Metric':<40} {'Baseline':<20} {'RL':<20} {'Improvement'}")
    print("-"*80)
    
    # Energy
    baseline_energy = baseline_stats['total_energy_kwh']
    rl_energy = rl_stats['total_energy_kwh']
    energy_improvement = ((baseline_energy - rl_energy) / baseline_energy * 100) if baseline_energy > 0 else 0
    print(f"{'Total Energy (kWh)':<40} {baseline_energy:<20.6f} {rl_energy:<20.6f} {energy_improvement:+.2f}%")
    
    # Cost
    baseline_cost = baseline_stats['estimated_cost_usd']
    rl_cost = rl_stats['estimated_cost_usd']
    cost_improvement = ((baseline_cost - rl_cost) / baseline_cost * 100) if baseline_cost > 0 else 0
    print(f"{'Cost ($)':<40} ${baseline_cost:<19.4f} ${rl_cost:<19.4f} {cost_improvement:+.2f}%")
    
    # Violations
    baseline_viol = baseline_stats['total_deadline_violations']
    rl_viol = rl_stats['total_deadline_violations']
    viol_change = rl_viol - baseline_viol
    print(f"{'Deadline Violations':<40} {baseline_viol:<20} {rl_viol:<20} {viol_change:+d}")
    
    # Violation rate
    baseline_viol_rate = baseline_stats['violation_rate'] * 100
    rl_viol_rate = rl_stats['violation_rate'] * 100
    viol_rate_change = rl_viol_rate - baseline_viol_rate
    print(f"{'Violation Rate (%)':<40} {baseline_viol_rate:<20.2f} {rl_viol_rate:<20.2f} {viol_rate_change:+.2f}%")
    
    # Finished containers
    baseline_finished = baseline_stats['total_containers_finished']
    rl_finished = rl_stats['total_containers_finished']
    finished_change = rl_finished - baseline_finished
    print(f"{'Containers Finished':<40} {baseline_finished:<20} {rl_finished:<20} {finished_change:+d}")
    
    # Peak PMs
    baseline_pms = baseline_stats['peak_active_pms']
    rl_pms = rl_stats['peak_active_pms']
    pm_change = rl_pms - baseline_pms
    print(f"{'Peak PMs Used':<40} {baseline_pms:<20} {rl_pms:<20} {pm_change:+d}")
    
    print("\n" + "="*80)
    
    # Summary
    print("\n💡 SUMMARY:")
    if energy_improvement > 0:
        print(f"  ✅ RL saved {energy_improvement:.2f}% energy (${cost_improvement*baseline_cost/100:.4f})")
    else:
        print(f"  ⚠️  RL used {-energy_improvement:.2f}% more energy")
    
    if viol_change <= 0:
        print(f"  ✅ RL maintained/reduced violations ({abs(viol_change)} fewer)")
    else:
        print(f"  ⚠️  RL had {viol_change} more violations")
    
    if finished_change >= 0:
        print(f"  ✅ RL finished {finished_change} more containers")
    else:
        print(f"  ⚠️  RL finished {abs(finished_change)} fewer containers")
    
    # RL learning info
    if 'rl_q_table_size' in rl_stats:
        print(f"\n🤖 RL LEARNING:")
        print(f"  Q-table size: {rl_stats['rl_q_table_size']} states learned")
        print(f"  Total updates: {rl_stats['rl_total_updates']}")
    
    print("\n" + "="*80 + "\n")
    
    return baseline_stats, rl_stats


if __name__ == "__main__":
    # Run comparison
    baseline, rl = run_comparison(num_slots=50)
    
    print("✅ Comparison complete!")
    print("\n💡 Note: RL agent is still learning. Run multiple episodes to see full potential.")