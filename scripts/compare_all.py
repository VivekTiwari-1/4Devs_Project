"""
Compare Baseline vs RL vs RL+Migration
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.simulator import Simulator
import pickle

def run_comparison(slots=50, num_runs=3):
    """
    Run comparison experiments.
    """
    print("📊 RUNNING COMPARISON EXPERIMENTS\n")
    
    scenarios = [
        ('Baseline (No RL, No Migration)', False, False),
        ('RL Only (No Migration)', True, False),
        ('RL + Migration (Full System)', True, True)
    ]
    
    results = {}
    
    for name, enable_rl, enable_migration in scenarios:
        print(f"\n{'='*60}")
        print(f"🧪 {name}")
        print('='*60)
        
        run_stats = []
        
        for run in range(num_runs):
            print(f"\n  Run {run + 1}/{num_runs}...")
            
            sim = Simulator(
                workload_pattern='random',
                enable_rl=enable_rl,
                enable_migration=enable_migration
            )
            
            # Load trained Q-table if RL enabled
            if enable_rl:
                try:
                    with open('results/qtables/trained_agent.pkl', 'rb') as f:
                        sim.rl_agent.q_table = pickle.load(f)
                        sim.rl_agent.epsilon = 0.1  # Mostly exploit
                except FileNotFoundError:
                    print("  ⚠️ No trained Q-table found, using untrained agent")
            
            sim.run(num_slots=slots)
            stats = sim.get_summary_stats()
            run_stats.append(stats)
        
        # Average across runs
        avg_stats = {
            'energy': sum(s['total_energy_kwh'] for s in run_stats) / num_runs,
            'cost': sum(s['estimated_cost_usd'] for s in run_stats) / num_runs,
            'violations': sum(s['total_deadline_violations'] for s in run_stats) / num_runs,
            'violation_rate': sum(s['violation_rate'] for s in run_stats) / num_runs,
            'peak_pms': sum(s['peak_active_pms'] for s in run_stats) / num_runs,
            'migrations': sum(s.get('total_migrations', 0) for s in run_stats) / num_runs if enable_migration else 0
        }
        
        results[name] = avg_stats
        
        print(f"\n  📊 Average Results ({num_runs} runs):")
        print(f"    Energy: {avg_stats['energy']:.2f} kWh")
        print(f"    Cost: ${avg_stats['cost']:.4f}")
        print(f"    Violations: {avg_stats['violations']:.1f}")
        print(f"    Peak PMs: {avg_stats['peak_pms']:.1f}")
        if enable_migration:
            print(f"    Migrations: {avg_stats['migrations']:.1f}")
    
    # Print comparison table
    print("\n" + "="*80)
    print("📈 COMPARISON SUMMARY")
    print("="*80)
    
    baseline_energy = results['Baseline (No RL, No Migration)']['energy']
    baseline_cost = results['Baseline (No RL, No Migration)']['cost']
    
    print(f"\n{'Scenario':<40} {'Energy (kWh)':<15} {'Cost ($)':<12} {'Violations':<12}")
    print("-" * 80)
    
    for name, stats in results.items():
        energy_reduction = ((baseline_energy - stats['energy']) / baseline_energy) * 100
        cost_reduction = ((baseline_cost - stats['cost']) / baseline_cost) * 100
        
        print(f"{name:<40} {stats['energy']:>8.2f} ({energy_reduction:>+5.1f}%) "
              f"${stats['cost']:>6.4f} ({cost_reduction:>+5.1f}%)  {stats['violations']:>5.1f}")
    
    print("="*80)
    
    # Save results
    import json
    os.makedirs('results/logs', exist_ok=True)
    with open('results/logs/comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: results/logs/comparison_results.json")
    
    return results

if __name__ == "__main__":
    results = run_comparison(slots=50, num_runs=3)