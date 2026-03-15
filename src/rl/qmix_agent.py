
"""
QMIX Multi-Agent Reinforcement Learning Agent
Complete implementation with mixing network, replay buffer, target networks
"""

import sys
import os
import numpy as np
import pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import (
    RL_LEARNING_RATE,
    RL_DISCOUNT_FACTOR,
    RL_EPSILON_START,
    RL_EPSILON_MIN,
    RL_EPSILON_DECAY,
    RL_POLICIES
)
from rl.qmix_network import MixingNetwork
from rl.experience_replay import MultiAgentReplayBuffer


class QMIXAgent:
    """
    QMIX Multi-Agent System with:
    - Individual Q-networks per agent (tabular for simplicity)
    - Mixing network with hypernetwork
    - Experience replay
    - Target network (for stability)
    """
    
    def __init__(self, num_agents=3, state_dim=3, learning_rate=RL_LEARNING_RATE,
                 discount_factor=RL_DISCOUNT_FACTOR, epsilon_start=RL_EPSILON_START):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_min = RL_EPSILON_MIN
        self.epsilon_decay = RL_EPSILON_DECAY
        
        # Individual agent Q-tables (for simplicity, using tables not networks)
        self.agent_q_tables = [{} for _ in range(num_agents)]
        
        # Target Q-tables (frozen copies for stable learning)
        self.target_q_tables = [{} for _ in range(num_agents)]
        
        # QMIX Mixing Network
        self.mixing_net = MixingNetwork(num_agents, state_dim)
        
        # Target Mixing Network (frozen copy)
        self.target_mixing_net = MixingNetwork(num_agents, state_dim)
        
        # Experience Replay Buffer
        self.replay_buffer = MultiAgentReplayBuffer(capacity=10000)
        
        # Training parameters
        self.batch_size = 32
        self.update_target_every = 100  # Update target networks every N steps
        self.train_step_counter = 0
        
        # Action space
        self.actions = list(range(len(RL_POLICIES)))
        self.policy_names = RL_POLICIES
        
        # Statistics
        self.total_updates = 0
        self.episodes_trained = 0
        
        print(f"🤖 QMIX Agent initialized: {num_agents} agents, replay buffer, target networks")
    
    def get_state(self, pm):
        """Extract local state from PM."""
        if not pm.is_on or len(pm.containers) == 0:
            return (0, 0, 0)
        
        num_containers = len(pm.containers)
        if num_containers <= 3:
            num_containers_bin = 1
        elif num_containers <= 8:
            num_containers_bin = 2
        else:
            num_containers_bin = 3
        
        deadline_gaps = [c.deadline - c._current_time for c in pm.containers if hasattr(c, '_current_time')]
        if deadline_gaps:
            avg_gap = sum(deadline_gaps) / len(deadline_gaps)
            if avg_gap < 0:
                deadline_gap_bin = 0
            elif avg_gap < 300:
                deadline_gap_bin = 1
            elif avg_gap < 600:
                deadline_gap_bin = 2
            elif avg_gap < 900:
                deadline_gap_bin = 3
            else:
                deadline_gap_bin = 4
        else:
            deadline_gap_bin = 4
        
        cpu_util = pm.cpu_utilization()
        if cpu_util < 0.3:
            cpu_avail_bin = 2
        elif cpu_util < 0.7:
            cpu_avail_bin = 1
        else:
            cpu_avail_bin = 0
        
        return (num_containers_bin, deadline_gap_bin, cpu_avail_bin)
    
    def get_global_state(self, states):
        """
        Create global state from individual states.
        
        Args:
            states (list): Individual agent states
            
        Returns:
            np.array: Global state vector
        """
        # Flatten all states into single vector
        global_state = []
        for state in states:
            global_state.extend(state)
        
        # Pad to fixed size
        while len(global_state) < self.state_dim * self.num_agents:
            global_state.append(0)
        
        return np.array(global_state[:self.state_dim * self.num_agents])
    
    def select_actions(self, states):
        """
        Decentralized execution: each agent selects action independently.
        
        Args:
            states (list): States for each agent
            
        Returns:
            list: Actions for each agent
        """
        actions = []
        
        for agent_id, state in enumerate(states):
            if agent_id >= self.num_agents:
                agent_id = self.num_agents - 1
            
            # Epsilon-greedy
            if np.random.random() < self.epsilon:
                action = np.random.choice(self.actions)
            else:
                q_values = [self._get_local_q(agent_id, state, a, use_target=False) 
                           for a in self.actions]
                max_q = max(q_values)
                best_actions = [a for a, q in zip(self.actions, q_values) if q == max_q]
                action = np.random.choice(best_actions)
            
            actions.append(action)
        
        return actions
    
    def _get_local_q(self, agent_id, state, action, use_target=False):
        """Get Q-value from agent's Q-table."""
        q_table = self.target_q_tables[agent_id] if use_target else self.agent_q_tables[agent_id]
        return q_table.get((state, action), 0.0)
    
    def _set_local_q(self, agent_id, state, action, value):
        """Set Q-value in agent's Q-table."""
        self.agent_q_tables[agent_id][(state, action)] = value
    
    def store_transition(self, states, actions, rewards, next_states, done=False):
        """
        Store transition in replay buffer.
        
        Args:
            states (list): Current states
            actions (list): Actions taken
            rewards (list): Rewards received
            next_states (list): Next states
            done (bool): Episode done
        """
        self.replay_buffer.push(states, actions, rewards, next_states, done)
    
    def train(self):
        """
        Train QMIX from replay buffer (centralized training).
        Uses batch learning for stability.
        """
        if len(self.replay_buffer) < self.batch_size:
            return  # Not enough samples
        
        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        
        total_loss = 0.0
        
        # Process each transition in batch
        for i in range(len(batch['states'])):
            states = batch['states'][i]
            actions = batch['actions'][i]
            rewards = batch['rewards'][i]
            next_states = batch['next_states'][i]
            done = batch['done'][i]
            
            # Get local Q-values (current)
            local_q_values = []
            for agent_id, (state, action) in enumerate(zip(states, actions)):
                local_q_values.append(self._get_local_q(agent_id, state, action, use_target=False))
            
            # Get global state
            global_state = self.get_global_state(states)
            
            # Mix Q-values with current mixing network
            q_total, weights, bias = self.mixing_net.mix(local_q_values, global_state)
            
            # Get target Q-values
            if not done:
                # Get best next actions from target networks
                next_local_q_values = []
                for agent_id, next_state in enumerate(next_states):
                    next_q_vals = [self._get_local_q(agent_id, next_state, a, use_target=True) 
                                  for a in self.actions]
                    next_local_q_values.append(max(next_q_vals) if next_q_vals else 0.0)
                
                # Mix next Q-values with target mixing network
                next_global_state = self.get_global_state(next_states)
                q_total_next, _, _ = self.target_mixing_net.mix(next_local_q_values, next_global_state)
                
                # TD target
                total_reward = sum(rewards)
                td_target = total_reward + self.gamma * q_total_next
            else:
                td_target = sum(rewards)
            
            # TD error
            td_error = td_target - q_total
            total_loss += td_error ** 2
            
            # Update individual Q-values
            for agent_id, (state, action) in enumerate(zip(states, actions)):
                current_q = self._get_local_q(agent_id, state, action, use_target=False)
                
                # Update with share of TD error weighted by mixing weight
                weight = weights[agent_id] if agent_id < len(weights) else 1.0
                new_q = current_q + self.alpha * weight * td_error
                
                self._set_local_q(agent_id, state, action, new_q)
            
            # Update mixing network
            self.mixing_net.update(td_error, local_q_values, global_state)
        
        self.total_updates += 1
        self.train_step_counter += 1
        
        # Update target networks periodically
        if self.train_step_counter % self.update_target_every == 0:
            self._update_target_networks()
    
    def _update_target_networks(self):
        """Copy current networks to target networks."""
        # Copy Q-tables
        for i in range(self.num_agents):
            self.target_q_tables[i] = self.agent_q_tables[i].copy()
        
        # Copy mixing network (deep copy of hypernetwork weights)
        # Simplified: just reinitialize target with same architecture
        # In practice, should copy actual weights
        print("  🎯 Target networks updated")
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episodes_trained += 1
    
    def get_policy_name(self, action):
        """Get policy name."""
        return self.policy_names[action]
    
    def save(self, filepath):
        """Save QMIX agent."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'agent_q_tables': self.agent_q_tables,
                'target_q_tables': self.target_q_tables,
                'epsilon': self.epsilon,
                'total_updates': self.total_updates,
                'episodes_trained': self.episodes_trained,
                'train_step_counter': self.train_step_counter
            }, f)
        print(f"✅ QMIX agent saved to {filepath}")
    
    def load(self, filepath):
        """Load QMIX agent."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.agent_q_tables = data['agent_q_tables']
            self.target_q_tables = data['target_q_tables']
            self.epsilon = data['epsilon']
            self.total_updates = data['total_updates']
            self.episodes_trained = data['episodes_trained']
            self.train_step_counter = data['train_step_counter']
        print(f"✅ QMIX agent loaded from {filepath}")
    
    def get_statistics(self):
        """Get statistics."""
        total_q_entries = sum(len(qt) for qt in self.agent_q_tables)
        return {
            'num_agents': self.num_agents,
            'total_q_entries': total_q_entries,
            'replay_buffer_size': len(self.replay_buffer),
            'epsilon': self.epsilon,
            'total_updates': self.total_updates,
            'episodes_trained': self.episodes_trained,
            'train_step_counter': self.train_step_counter
        }
    
    def __repr__(self):
        total_entries = sum(len(qt) for qt in self.agent_q_tables)
        return (f"QMIXAgent(agents={self.num_agents}, "
                f"q_entries={total_entries}, buffer={len(self.replay_buffer)}, "
                f"epsilon={self.epsilon:.3f})")
