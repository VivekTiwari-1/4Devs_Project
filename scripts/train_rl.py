"""
Train RL agent over multiple episodes
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.simulator import Simulator
import pickle

def train_rl_agent(episodes=30, slots_per_episode=50):
    """
    Train RL agent over multiple episodes.
    """
    print(f"🎓 Training RL Agent: {episodes} episodes\n")
    
    # Track progress across episodes
    episode_rewards = []
    episode_violations = []
    episode_energy = []
    
    sim = Simulator(
        workload_pattern='random',
        enable_rl=True,
        enable_migration=True
    )
    
    for episode in range(episodes):
        print(f"\n{'='*60}")
        print(f"📚 EPISODE {episode + 1}/{episodes}")
        print('='*60)
        
        # Reset simulator but keep Q-table
        if episode > 0:
            sim = Simulator(
                workload_pattern='random',
                enable_rl=True,
                enable_migration=True
            )
            sim.rl_agent.q_table = trained_q_table
            sim.rl_agent.epsilon = max(0.1, 0.9 * (0.95 ** episode))
        
        # Run episode
        sim.run(num_slots=slots_per_episode)
        
        # Collect metrics
        stats = sim.get_summary_stats()
        episode_rewards.append(sum(sim.history['rl_rewards']))
        episode_violations.append(stats['total_deadline_violations'])
        episode_energy.append(stats['total_energy_kwh'])
        
        # Save Q-table
        trained_q_table = sim.rl_agent.q_table.copy()
        
        print(f"\n📊 Episode Summary:")
        print(f"  Total Reward: {sum(sim.history['rl_rewards']):.2f}")
        print(f"  Violations: {stats['total_deadline_violations']}")
        print(f"  Energy: {stats['total_energy_kwh']:.2f} kWh")
        print(f"  Q-table size: {len(trained_q_table)}")
    
    # Save trained Q-table
    os.makedirs('results/qtables', exist_ok=True)
    with open('results/qtables/trained_agent.pkl', 'wb') as f:
        pickle.dump(trained_q_table, f)
    
    print(f"\n✅ Training Complete!")
    print(f"Q-table saved to: results/qtables/trained_agent.pkl")
    
    # Save training history
    import json
    history = {
        'episode_rewards': episode_rewards,
        'episode_violations': episode_violations,
        'episode_energy': episode_energy
    }
    
    with open('results/logs/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    return history

if __name__ == "__main__":
    history = train_rl_agent(episodes=30, slots_per_episode=50)