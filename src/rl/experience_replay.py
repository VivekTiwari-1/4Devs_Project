
"""
Experience Replay Buffer for QMIX
Stores multi-agent transitions
"""

import numpy as np
from collections import deque
import random


class MultiAgentReplayBuffer:
    """
    Experience replay buffer for multi-agent learning.
    Stores (states, actions, rewards, next_states, done) tuples.
    """
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, states, actions, rewards, next_states, done=False):
        """
        Add transition to buffer.
        
        Args:
            states (list): List of states (one per agent)
            actions (list): List of actions
            rewards (list): List of rewards
            next_states (list): List of next states
            done (bool): Episode done flag
        """
        self.buffer.append({
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'done': done
        })
    
    def sample(self, batch_size):
        """
        Sample random batch.
        
        Args:
            batch_size (int): Batch size
            
        Returns:
            dict: Batch of transitions
        """
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        return {
            'states': [t['states'] for t in batch],
            'actions': [t['actions'] for t in batch],
            'rewards': [t['rewards'] for t in batch],
            'next_states': [t['next_states'] for t in batch],
            'done': [t['done'] for t in batch]
        }
    
    def __len__(self):
        return len(self.buffer)
