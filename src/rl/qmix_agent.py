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


# FIX: Each local state has 4 values now (added container load bin)
LOCAL_STATE_SIZE = 4


class QMIXAgent:
    """
    QMIX Multi-Agent System with:
    - Individual Q-networks per agent (tabular for simplicity)
    - Mixing network with hypernetwork
    - Experience replay
    - Target network (for stability)
    """

    def __init__(self, num_agents=3, learning_rate=RL_LEARNING_RATE,
                 discount_factor=RL_DISCOUNT_FACTOR, epsilon_start=RL_EPSILON_START,
                 batch_size=16, update_target_every=10):
        self.num_agents = num_agents
        # --- FIX: state_dim is now correctly LOCAL_STATE_SIZE per agent ---
        self.local_state_size = LOCAL_STATE_SIZE
        self.global_state_size = num_agents * LOCAL_STATE_SIZE  # = 9

        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_min = RL_EPSILON_MIN
        self.epsilon_decay = RL_EPSILON_DECAY

        # Individual agent Q-tables
        self.agent_q_tables = [{} for _ in range(num_agents)]

        # Target Q-tables (frozen copies for stable learning)
        self.target_q_tables = [{} for _ in range(num_agents)]

        # QMIX Mixing Network (lazy init on first train call)
        self.mixing_net = None
        self.target_mixing_net = None

        # Experience Replay Buffer
        self.replay_buffer = MultiAgentReplayBuffer(capacity=10000)

        # Training parameters
        self.batch_size = batch_size                   # wired from config (default 16)
        self.update_target_every = update_target_every  # wired from config (default 10)
        self.train_step_counter = 0

        # Action space
        self.actions = list(range(len(RL_POLICIES)))
        self.policy_names = RL_POLICIES

        # Statistics
        self.total_updates = 0
        self.episodes_trained = 0
        self.reward_history = []

        print(f"🤖 QMIX Agent initialized: {num_agents} agents, replay buffer, target networks")

    def get_state(self, pm):
        """
        Extract local state from PM.
        Returns a 4-tuple: (container_count_bin, deadline_gap_bin, cpu_avail_bin, load_bin)

        FIX: Added load_bin (4th feature) so the Q-table can distinguish between
        a PM with 5 containers and one with 50 — previously both mapped to the
        same 3-tuple despite needing completely different policies.
        """
        if not pm.is_on or len(pm.containers) == 0:
            return (0, 0, 0, 0)

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

        # FIX: 4th feature — container load bin
        # Distinguishes light (few containers) from overloaded (many containers)
        # Critical: same cpu_util can mean very different things at different container counts
        if num_containers <= 10:
            load_bin = 0   # light
        elif num_containers <= 25:
            load_bin = 1   # medium
        elif num_containers <= 50:
            load_bin = 2   # heavy
        else:
            load_bin = 3   # overloaded

        return (num_containers_bin, deadline_gap_bin, cpu_avail_bin, load_bin)

    def get_global_state(self, states):
        """
        Create global state from individual agent states.

        Args:
            states (list): Individual agent states (each is a tuple of 3 values)

        Returns:
            np.array: Global state vector of fixed size (num_agents * LOCAL_STATE_SIZE)
        """
        global_state = []
        for state in states:
            global_state.extend(state)

        # --- FIX: Pad to num_agents * LOCAL_STATE_SIZE (= 9), not state_dim * num_agents ---
        target_size = self.global_state_size
        while len(global_state) < target_size:
            global_state.append(0)

        return np.array(global_state[:target_size], dtype=float)

    def _init_mixing_nets(self, state_dim):
        """Lazily initialize mixing and target mixing networks."""
        self.mixing_net = MixingNetwork(num_agents=self.num_agents, state_dim=state_dim)
        self.target_mixing_net = MixingNetwork(num_agents=self.num_agents, state_dim=state_dim)
        # Start target as a copy of mixing net
        self.target_mixing_net.copy_weights_from(self.mixing_net)

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
        """Store transition in replay buffer."""
        self.replay_buffer.push(states, actions, rewards, next_states, done)

    def train(self):
        """
        Train QMIX from replay buffer (centralized training).
        Uses batch learning for stability.
        """
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)

        total_loss = 0.0

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

            # --- FIX: Lazy init both mixing nets together ---
            if self.mixing_net is None:
                self._init_mixing_nets(state_dim=global_state.shape[-1])

            # Mix Q-values with current mixing network
            q_total, weights, bias = self.mixing_net.mix(local_q_values, global_state)

            # Compute TD target
            if not done:
                next_local_q_values = []
                for agent_id, next_state in enumerate(next_states):
                    next_q_vals = [self._get_local_q(agent_id, next_state, a, use_target=True)
                                   for a in self.actions]
                    next_local_q_values.append(max(next_q_vals) if next_q_vals else 0.0)

                next_global_state = self.get_global_state(next_states)
                q_total_next, _, _ = self.target_mixing_net.mix(next_local_q_values, next_global_state)

                total_reward = sum(rewards)
                td_target = total_reward + self.gamma * q_total_next
            else:
                td_target = sum(rewards)

            # TD error — clipped for stability
            td_error = np.clip(td_target - q_total, -10.0, 10.0)
            total_loss += td_error ** 2

            # Update individual Q-values
            for agent_id, (state, action) in enumerate(zip(states, actions)):
                current_q = self._get_local_q(agent_id, state, action, use_target=False)
                weight = weights[agent_id] if agent_id < len(weights) else 1.0
                new_q = current_q + self.alpha * weight * td_error
                self._set_local_q(agent_id, state, action, new_q)

            # Update mixing network
            self.mixing_net.update(td_error, local_q_values, global_state)

        self.total_updates += 1
        self.train_step_counter += 1

        # --- FIX: Properly copy weights to target network ---
        if self.train_step_counter % self.update_target_every == 0:
            self._update_target_networks()

    def _update_target_networks(self):
        """Copy current networks to target networks (proper weight copy)."""
        # Copy Q-tables
        for i in range(self.num_agents):
            self.target_q_tables[i] = self.agent_q_tables[i].copy()

        # --- FIX: Actually copy mixing network weights instead of reinitializing ---
        if self.mixing_net is not None and self.target_mixing_net is not None:
            self.target_mixing_net.copy_weights_from(self.mixing_net)

        print("  🎯 Target networks updated")

    def decay_epsilon(self):
        """
        Decay exploration rate.
        --- FIX: Call this every TIME SLOT, not just at episode end ---
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episodes_trained += 1

    def record_reward(self, reward):
        """Track reward history for monitoring."""
        self.reward_history.append(reward)

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
                'train_step_counter': self.train_step_counter,
                'reward_history': self.reward_history
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
            self.reward_history = data.get('reward_history', [])
        print(f"✅ QMIX agent loaded from {filepath}")

    def get_statistics(self):
        """Get statistics."""
        total_q_entries = sum(len(qt) for qt in self.agent_q_tables)
        avg_reward = np.mean(self.reward_history[-50:]) if self.reward_history else 0.0
        return {
            'num_agents': self.num_agents,
            'total_q_entries': total_q_entries,
            'replay_buffer_size': len(self.replay_buffer),
            'epsilon': self.epsilon,
            'total_updates': self.total_updates,
            'episodes_trained': self.episodes_trained,
            'train_step_counter': self.train_step_counter,
            'avg_reward_last50': avg_reward
        }

    def __repr__(self):
        total_entries = sum(len(qt) for qt in self.agent_q_tables)
        return (f"QMIXAgent(agents={self.num_agents}, "
                f"q_entries={total_entries}, buffer={len(self.replay_buffer)}, "
                f"epsilon={self.epsilon:.3f})")