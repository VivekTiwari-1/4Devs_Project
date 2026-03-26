"""
Generate plots for mini project report
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import matplotlib.pyplot as plt
import numpy as np

def plot_training_progress():
    """Plot training progress over episodes."""
    with open('results/logs/training_history.json', 'r') as f:
        history = json.load(f)
    
    episodes = range(1, len(history['episode_rewards']) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Rewards
    axes[0].plot(episodes, history['episode_rewards'], marker='o', color='blue')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('RL Agent Learning: Rewards')
    axes[0].grid(True, alpha=0.3)
    
    # Violations
    axes[1].plot(episodes, history['episode_violations'], marker='s', color='red')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Deadline Violations')
    axes[1].set_title('RL Agent Learning: Violations')
    axes[1].grid(True, alpha=0.3)
    
    # Energy
    axes[2].plot(episodes, history['episode_energy'], marker='^', color='green')
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Energy (kWh)')
    axes[2].set_title('RL Agent Learning: Energy')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('results/plots', exist_ok=True)
    plt.savefig('results/plots/training_progress.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: results/plots/training_progress.png")
    plt.show()

def plot_comparison():
    """Plot comparison bar charts."""
    with open('results/logs/comparison_results.json', 'r') as f:
        results = json.load(f)
    
    scenarios = list(results.keys())
    energy = [results[s]['energy'] for s in scenarios]
    cost = [results[s]['cost'] for s in scenarios]
    violations = [results[s]['violations'] for s in scenarios]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    # Energy comparison
    axes[0].bar(range(len(scenarios)), energy, color=colors)
    axes[0].set_xticks(range(len(scenarios)))
    axes[0].set_xticklabels(['Baseline', 'RL Only', 'RL+Migration'], rotation=15, ha='right')
    axes[0].set_ylabel('Energy (kWh)')
    axes[0].set_title('Energy Consumption Comparison')
    axes[0].grid(True, axis='y', alpha=0.3)
    
    # Cost comparison
    axes[1].bar(range(len(scenarios)), cost, color=colors)
    axes[1].set_xticks(range(len(scenarios)))
    axes[1].set_xticklabels(['Baseline', 'RL Only', 'RL+Migration'], rotation=15, ha='right')
    axes[1].set_ylabel('Cost ($)')
    axes[1].set_title('Operational Cost Comparison')
    axes[1].grid(True, axis='y', alpha=0.3)
    
    # Violations comparison
    axes[2].bar(range(len(scenarios)), violations, color=colors)
    axes[2].set_xticks(range(len(scenarios)))
    axes[2].set_xticklabels(['Baseline', 'RL Only', 'RL+Migration'], rotation=15, ha='right')
    axes[2].set_ylabel('Violations')
    axes[2].set_title('Deadline Violations Comparison')
    axes[2].grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/plots/comparison_bars.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: results/plots/comparison_bars.png")
    plt.show()

if __name__ == "__main__":
    print("📊 Generating Plots...\n")
    plot_training_progress()
    plot_comparison()
    print("\n✅ All plots generated!")