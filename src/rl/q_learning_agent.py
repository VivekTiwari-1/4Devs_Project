"""
Q-Learning Agent for CPU Allocation
Learns to allocate CPU cores to containers to minimize energy while meeting deadlines
"""

import sys
import os
import random
import pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import (
    RL_LEARNING_RATE,
    RL_DISCOUNT_FACTOR,
    RL_EPSILON_START,
    RL_EPSILON_MIN,
    RL_EPSILON_DECAY,
    RL_DEADLINE_VIOLATION_PENALTY,
    RL_POLICIES
)


class QLearningAgent:
    """
    Q-Learning agent that learns CPU allocation policies for containers.
    
    State: Discretized PM state (num_containers, avg_deadline_gap, cpu_available)
    Action: Select one of the allocation policies
    Reward: -energy - penalties
    """
    
    def __init__(self, learning_rate=RL_LEARNING_RATE, 
                 discount_factor=RL_DISCOUNT_FACTOR,
                 epsilon_start=RL_EPSILON_START,
                 epsilon_min=RL_EPSILON_MIN,
                 epsilon_decay=RL_EPSILON_DECAY):
        """
        Initialize Q-learning agent.
        
        Args:
            learning_rate (float): Alpha - learning rate
            discount_factor (float): Gamma - discount factor
            epsilon_start (float): Initial exploration rate
            epsilon_min (float): Minimum exploration rate
            epsilon_decay (float): Epsilon decay per episode
        """
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Q-table: dict mapping (state, action) -> Q-value
        self.q_table = {}
        
        # Action space: indices of allocation policies
        self.actions = list(range(len(RL_POLICIES)))
        self.policy_names = RL_POLICIES
        
        # Statistics
        self.total_updates = 0
        self.episodes_trained = 0
        
    def get_state(self, pm):
        """
        Extract and discretize state from a Physical Machine.
        
        Args:
            pm (PhysicalMachine): PM object
            
        Returns:
            tuple: Discretized state (num_containers_bin, deadline_gap_bin, cpu_avail_bin)
        """
        if not pm.is_on or len(pm.containers) == 0:
            return (0, 0, 0)  # Empty PM state
        
        # Feature 1: Number of containers (discretized)
        num_containers = len(pm.containers)
        if num_containers == 0:
            num_containers_bin = 0
        elif num_containers <= 3:
            num_containers_bin = 1
        elif num_containers <= 8:
            num_containers_bin = 2
        else:
            num_containers_bin = 3
        
        # Feature 2: Average deadline gap (time until deadline)
        # Get current time from first container
        if pm.containers:
            current_time = pm.containers[0].arrival_time  # Approximation
            deadline_gaps = [c.deadline - current_time for c in pm.containers]
            avg_gap = sum(deadline_gaps) / len(deadline_gaps)
            
            # Discretize deadline gap
            if avg_gap < 0:
                deadline_gap_bin = 0  # Already late
            elif avg_gap < 300:
                deadline_gap_bin = 1  # Urgent (< 5 min)
            elif avg_gap < 600:
                deadline_gap_bin = 2  # Soon (5-10 min)
            elif avg_gap < 900:
                deadline_gap_bin = 3  # Normal (10-15 min)
            else:
                deadline_gap_bin = 4  # Relaxed (> 15 min)
        else:
            deadline_gap_bin = 4
        
        # Feature 3: Available CPU capacity
        cpu_util = pm.cpu_utilization()
        if cpu_util < 0.3:
            cpu_avail_bin = 0  # Low utilization
        elif cpu_util < 0.6:
            cpu_avail_bin = 1  # Medium
        else:
            cpu_avail_bin = 2  # High utilization
        
        return (num_containers_bin, deadline_gap_bin, cpu_avail_bin)
    
    def select_action(self, state):
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state (tuple): Current state
            
        Returns:
            int: Selected action index
        """
        # Exploration: random action
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        
        # Exploitation: best action from Q-table
        q_values = [self.get_q_value(state, action) for action in self.actions]
        max_q = max(q_values)
        
        # Handle ties: randomly select among actions with max Q-value
        best_actions = [action for action, q in zip(self.actions, q_values) if q == max_q]
        return random.choice(best_actions)
    
    def get_q_value(self, state, action):
        """
        Get Q-value for (state, action) pair.
        
        Args:
            state (tuple): State
            action (int): Action
            
        Returns:
            float: Q-value (0 if never seen)
        """
        return self.q_table.get((state, action), 0.0)
    
    def update(self, state, action, reward, next_state):
        """
        Update Q-table using Q-learning update rule.
        
        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        
        Args:
            state (tuple): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (tuple): Next state
        """
        # Current Q-value
        current_q = self.get_q_value(state, action)
        
        # Maximum Q-value for next state
        next_q_values = [self.get_q_value(next_state, a) for a in self.actions]
        max_next_q = max(next_q_values) if next_q_values else 0.0
        
        # Q-learning update
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        
        # Store in Q-table
        self.q_table[(state, action)] = new_q
        
        self.total_updates += 1
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episodes_trained += 1
    
    def get_policy_name(self, action):
        """
        Get human-readable policy name for action.
        
        Args:
            action (int): Action index
            
        Returns:
            str: Policy name
        """
        return self.policy_names[action]
    
    def save(self, filepath):
        """
        Save Q-table to file.
        
        Args:
            filepath (str): Path to save Q-table
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'epsilon': self.epsilon,
                'total_updates': self.total_updates,
                'episodes_trained': self.episodes_trained
            }, f)
        print(f"✅ Q-table saved to {filepath}")
    
    def load(self, filepath):
        """
        Load Q-table from file.
        
        Args:
            filepath (str): Path to load Q-table from
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.q_table = data['q_table']
            self.epsilon = data['epsilon']
            self.total_updates = data['total_updates']
            self.episodes_trained = data['episodes_trained']
        print(f"✅ Q-table loaded from {filepath}")
    
    def get_statistics(self):
        """
        Get agent statistics.
        
        Returns:
            dict: Statistics
        """
        return {
            'q_table_size': len(self.q_table),
            'epsilon': self.epsilon,
            'total_updates': self.total_updates,
            'episodes_trained': self.episodes_trained,
            'learning_rate': self.alpha,
            'discount_factor': self.gamma
        }
    
    def __repr__(self):
        """String representation."""
        return (f"QLearningAgent(q_table_size={len(self.q_table)}, "
                f"epsilon={self.epsilon:.3f}, updates={self.total_updates})")