"""
QMIX Neural Network Components
Uses simple feed-forward networks (no deep learning libraries needed)
"""

import numpy as np
import pickle


class SimpleNN:
    """Simple feed-forward neural network."""

    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.lr = learning_rate

        # Xavier initialization
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))

        # For backprop
        self.cache = {}

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def forward(self, X):
        """Forward pass with NaN guard."""
        # --- FIX: Replace any NaN/Inf in input before propagating ---
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
        self.cache['X'] = X

        # Hidden layer
        self.cache['Z1'] = np.dot(X, self.W1) + self.b1
        self.cache['A1'] = self.relu(self.cache['Z1'])

        # Output layer
        self.cache['Z2'] = np.dot(self.cache['A1'], self.W2) + self.b2

        return self.cache['Z2']

    def backward(self, grad_output):
        """Backward pass with gradient clipping to prevent explosion."""
        # --- FIX: Clip incoming gradient to prevent explosion ---
        grad_output = np.clip(grad_output, -1.0, 1.0)
        grad_output = np.nan_to_num(grad_output, nan=0.0, posinf=1.0, neginf=-1.0)

        m = self.cache['X'].shape[0]

        # Output layer gradients
        dZ2 = grad_output
        dW2 = np.dot(self.cache['A1'].T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # Hidden layer gradients
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.relu_derivative(self.cache['Z1'])

        # --- FIX: Guard against NaN in hidden layer gradient ---
        dZ1 = np.nan_to_num(dZ1, nan=0.0, posinf=1.0, neginf=-1.0)

        dW1 = np.dot(self.cache['X'].T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # --- FIX: Clip weight gradients before applying ---
        dW1 = np.clip(dW1, -1.0, 1.0)
        dW2 = np.clip(dW2, -1.0, 1.0)
        db1 = np.clip(db1, -1.0, 1.0)
        db2 = np.clip(db2, -1.0, 1.0)

        # Update weights
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2


class HyperNetwork:
    """
    Hypernetwork that generates mixing weights from global state.
    Ensures monotonicity constraint for QMIX.
    """

    def __init__(self, state_dim, num_agents, hidden_dim=32):
        self.num_agents = num_agents

        # Network to generate weights (must be positive)
        self.weight_net = SimpleNN(state_dim, hidden_dim, num_agents)

        # Network to generate bias
        self.bias_net = SimpleNN(state_dim, hidden_dim, 1)

    def forward(self, global_state):
        """
        Generate mixing weights and bias from global state.

        Args:
            global_state (np.array): Global state vector

        Returns:
            tuple: (weights, bias) where weights are non-negative
        """
        # Ensure input is 2D
        if len(global_state.shape) == 1:
            global_state = global_state.reshape(1, -1)

        # Generate weights and apply absolute value for monotonicity
        weights_raw = self.weight_net.forward(global_state)
        weights = np.abs(weights_raw)  # Monotonicity: weights >= 0

        # Generate bias
        bias = self.bias_net.forward(global_state)

        return weights.flatten(), bias.flatten()[0]

    def backward_weight(self, grad):
        """Backpropagate through weight network."""
        self.weight_net.backward(grad)

    def backward_bias(self, grad):
        """Backpropagate through bias network."""
        self.bias_net.backward(grad)


class MixingNetwork:
    """
    QMIX Mixing Network.
    Combines individual Q-values into Q_total with monotonicity.
    """

    def __init__(self, num_agents, state_dim):
        self.num_agents = num_agents
        self.state_dim = state_dim

        # Hypernetwork generates weights and bias
        self.hyper_net = HyperNetwork(state_dim, num_agents)

    def mix(self, agent_q_values, global_state):
        """
        Mix individual Q-values.

        Args:
            agent_q_values (list): Q-values from each agent
            global_state (np.array): Global state

        Returns:
            float: Mixed Q_total
        """
        # Get mixing parameters from hypernetwork
        weights, bias = self.hyper_net.forward(global_state)

        # Ensure we have values for all agents
        q_values = np.array(agent_q_values[:self.num_agents])
        if len(q_values) < self.num_agents:
            q_values = np.pad(q_values, (0, self.num_agents - len(q_values)))

        # Mixing: Q_tot = sum(w_i * Q_i) + b
        q_total = np.dot(weights, q_values) + bias

        return q_total, weights, bias

    def update(self, td_error, agent_q_values, global_state):
        """
        Update mixing network parameters.

        Args:
            td_error (float): TD error from Q-learning
            agent_q_values (list): Individual Q-values
            global_state (np.array): Global state
        """
        # --- FIX: Clip td_error before using in gradient ---
        td_error = np.clip(td_error, -10.0, 10.0)

        q_values = np.array(agent_q_values[:self.num_agents])
        if len(q_values) < self.num_agents:
            q_values = np.pad(q_values, (0, self.num_agents - len(q_values)))

        # Gradient w.r.t weights: dL/dw = td_error * Q_values
        grad_weights = td_error * q_values.reshape(1, -1)

        # Gradient w.r.t bias: dL/db = td_error
        grad_bias = np.array([[td_error]])

        # Backpropagate
        self.hyper_net.backward_weight(grad_weights)
        self.hyper_net.backward_bias(grad_bias)

    def copy_weights_from(self, source):
        """
        Copy weights from another MixingNetwork (for target network update).

        Args:
            source (MixingNetwork): The network to copy weights from
        """
        hn = source.hyper_net
        thn = self.hyper_net

        thn.weight_net.W1 = hn.weight_net.W1.copy()
        thn.weight_net.b1 = hn.weight_net.b1.copy()
        thn.weight_net.W2 = hn.weight_net.W2.copy()
        thn.weight_net.b2 = hn.weight_net.b2.copy()

        thn.bias_net.W1 = hn.bias_net.W1.copy()
        thn.bias_net.b1 = hn.bias_net.b1.copy()
        thn.bias_net.W2 = hn.bias_net.W2.copy()
        thn.bias_net.b2 = hn.bias_net.b2.copy()