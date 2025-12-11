<<<<<<< HEAD
"""
Main Simulator Class - Integrated with RL (Step 3)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import (
    TIME_SLOT_DURATION,
    TOTAL_SIMULATION_SLOTS,
    CORE_SPEED,
    PRINT_EVERY_N_SLOTS,
    DEBUG_MODE,
    ENABLE_RL_SCALING,
    RL_DEADLINE_VIOLATION_PENALTY
)
from environment.pm import PhysicalMachine
from environment.workload_generator import WorkloadGenerator
from environment.energy_model import EnergyModel
from modules.placement import PlacementModule, DelayedPlacementQueue
from rl.q_learning_agent import QLearningAgent
from modules.policies import AllocationPolicies


class Simulator:
    """
    Main simulation engine with RL-based CPU scaling.
    
    Step 3 additions:
    - Q-learning agent for CPU allocation
    - Dynamic policy selection per PM
    - Reward-based learning
    """
    
    def __init__(self, workload_pattern='random', placement_strategy='first_fit', 
                 enable_rl=ENABLE_RL_SCALING):
        """Initialize simulator with optional RL."""
        # Time management
        self.current_time = 0
        self.time_slot_duration = TIME_SLOT_DURATION
        self.current_slot = 0
        
        # Infrastructure
        self.pms = []
        self.pm_counter = 0
        
        # Modules
        self.workload_gen = WorkloadGenerator(pattern=workload_pattern)
        self.placement_module = PlacementModule(strategy=placement_strategy)
        self.delayed_queue = DelayedPlacementQueue()
        self.energy_model = EnergyModel()
        
        # 🤖 RL Agent (Step 3)
        self.enable_rl = enable_rl
        if self.enable_rl:
            self.rl_agent = QLearningAgent()
            self.pm_states = {}  # Track previous state for each PM
        
        # Tracking
        self.total_containers_arrived = 0
        self.total_containers_finished = 0
        self.total_deadline_violations = 0
        self.total_rejections = 0
        
        # History
        self.history = {
            'time': [],
            'active_pms': [],
            'total_containers': [],
            'finished_containers': [],
            'violations': [],
            'avg_cpu_util': [],
            'energy_consumed': [],
            'total_power': [],
            'rl_actions': [],        # 🤖 NEW
            'rl_rewards': [],        # 🤖 NEW
            'rl_epsilon': []         # 🤖 NEW
        }
        
        mode = "RL-ENABLED" if self.enable_rl else "Baseline"
        print(f"✅ Simulator initialized ({mode})")
        print(f"   Workload pattern: {workload_pattern}")
        print(f"   Placement strategy: {placement_strategy}")
        print(f"   Time slot duration: {TIME_SLOT_DURATION}s")
        print(f"   ⚡ Energy model: ENABLED (Quadratic CPU)")
        if self.enable_rl:
            print(f"   🤖 RL Agent: ENABLED (Q-Learning)")
        print("-" * 60)
    
    def _create_new_pm(self):
        """Factory method to create new PM."""
        pm = PhysicalMachine(pm_id=self.pm_counter)
        self.pm_counter += 1
        pm.turn_on(self.current_time)
        return pm
    
    def _process_new_arrivals(self):
        """Generate and place new containers."""
        new_containers = self.workload_gen.generate(self.current_time)
        self.total_containers_arrived += len(new_containers)
        
        delayed_containers = self.delayed_queue.get_ready_containers(self.current_time)
        all_containers = delayed_containers + new_containers
        
        placed = 0
        delayed = 0
        
        for container in all_containers:
            success, pm, is_new_pm = self.placement_module.place_or_start_new_pm(
                container, self.pms, self._create_new_pm
            )
            
            if success:
                if is_new_pm:
                    self.pms.append(pm)
                placed += 1
            else:
                self.delayed_queue.add(container, self.current_time)
                delayed += 1
        
        rejected_containers = self.delayed_queue.get_dropped_containers()
        rejected = len(rejected_containers)
        self.total_rejections += rejected
        
        return placed, delayed, rejected
    
    def _execute_containers_with_rl(self):
        """
        🤖 Execute containers with RL-based CPU allocation.
        
        Returns:
            tuple: (finished, violations, total_reward, actions_taken)
        """
        total_finished = 0
        total_violations = 0
        total_reward = 0.0
        actions_taken = []
        
        for pm in self.pms:
            if not pm.is_on or len(pm.containers) == 0:
                continue
            
            # Update current time estimate for containers
            for container in pm.containers:
                # Use current simulation time for state extraction
                container._current_time = self.current_time
            
            # 🤖 RL: Get state
            state = self.rl_agent.get_state(pm)
            
            # 🤖 RL: Select action (policy)
            action = self.rl_agent.select_action(state)
            actions_taken.append(action)
            
            # 🤖 RL: Apply policy to get CPU allocations
            allocations = AllocationPolicies.apply_policy_by_index(
                action, pm.containers, pm.available_cores()
            )
            
            # Assign cores to containers
            for i, container in enumerate(pm.containers):
                if i < len(allocations):
                    container.assigned_cores = max(0.1, allocations[i])  # Minimum 0.1 cores
            
            # Execute containers
            for container in pm.containers:
                container.execute(CORE_SPEED, self.time_slot_duration)
            
            # Remove finished and count violations
            finished, violations = pm.remove_finished_containers(self.current_time)
            total_finished += finished
            total_violations += violations
            
            # 🤖 RL: Calculate reward
            energy = self.energy_model.compute_pm_energy(pm)
            reward = -energy - (violations * RL_DEADLINE_VIOLATION_PENALTY)
            total_reward += reward
            
            # 🤖 RL: Get next state and update Q-table
            next_state = self.rl_agent.get_state(pm)
            self.rl_agent.update(state, action, reward, next_state)
        
        self.total_containers_finished += total_finished
        self.total_deadline_violations += total_violations
        
        return total_finished, total_violations, total_reward, actions_taken
    
    def _execute_containers_baseline(self):
        """
        Baseline execution without RL (fixed allocation).
        
        Returns:
            tuple: (finished, violations, 0.0, [])
        """
        total_finished = 0
        total_violations = 0
        
        for pm in self.pms:
            if pm.is_on and len(pm.containers) > 0:
                # Fixed allocation: 1 core per container (or fair split)
                available = pm.available_cores()
                cores_per_container = available / len(pm.containers) if pm.containers else 1
                
                for container in pm.containers:
                    container.assigned_cores = max(0.1, cores_per_container)
                    container.execute(CORE_SPEED, self.time_slot_duration)
                
                finished, violations = pm.remove_finished_containers(self.current_time)
                total_finished += finished
                total_violations += violations
        
        self.total_containers_finished += total_finished
        self.total_deadline_violations += total_violations
        
        return total_finished, total_violations, 0.0, []
    
    def _execute_containers(self):
        """Execute containers (with or without RL)."""
        if self.enable_rl:
            return self._execute_containers_with_rl()
        else:
            return self._execute_containers_baseline()
    
    def _compute_energy(self):
        """Compute energy consumption."""
        slot_energy = self.energy_model.compute_datacenter_energy(self.pms)
        total_power = sum(self.energy_model.compute_pm_power(pm) for pm in self.pms)
        return slot_energy, total_power
    
    def _compute_metrics(self, rl_reward, rl_actions):
        """Compute metrics including RL stats."""
        active_pms = sum(1 for pm in self.pms if pm.is_on)
        total_containers = sum(len(pm.containers) for pm in self.pms)
        
        if active_pms > 0:
            avg_cpu_util = sum(pm.cpu_utilization() for pm in self.pms if pm.is_on) / active_pms
        else:
            avg_cpu_util = 0.0
        
        slot_energy, total_power = self._compute_energy()
        
        metrics = {
            'time': self.current_time,
            'slot': self.current_slot,
            'active_pms': active_pms,
            'total_pms': len(self.pms),
            'total_containers': total_containers,
            'avg_cpu_util': avg_cpu_util,
            'containers_arrived': self.total_containers_arrived,
            'containers_finished': self.total_containers_finished,
            'violations': self.total_deadline_violations,
            'rejections': self.total_rejections,
            'slot_energy_kwh': slot_energy,
            'total_power_watts': total_power,
            'rl_reward': rl_reward,                                    # 🤖 NEW
            'rl_actions': rl_actions,                                  # 🤖 NEW
            'rl_epsilon': self.rl_agent.epsilon if self.enable_rl else 0.0  # 🤖 NEW
        }
        
        return metrics
    
    def _update_history(self, metrics):
        """Update history."""
        self.history['time'].append(metrics['time'])
        self.history['active_pms'].append(metrics['active_pms'])
        self.history['total_containers'].append(metrics['total_containers'])
        self.history['finished_containers'].append(metrics['containers_finished'])
        self.history['violations'].append(metrics['violations'])
        self.history['avg_cpu_util'].append(metrics['avg_cpu_util'])
        self.history['energy_consumed'].append(metrics['slot_energy_kwh'])
        self.history['total_power'].append(metrics['total_power_watts'])
        if self.enable_rl:
            self.history['rl_actions'].append(metrics['rl_actions'])
            self.history['rl_rewards'].append(metrics['rl_reward'])
            self.history['rl_epsilon'].append(metrics['rl_epsilon'])
    
    def _print_progress(self, metrics, placed, delayed, rejected, finished, violations):
        """Print progress with RL info."""
        if DEBUG_MODE or self.current_slot % PRINT_EVERY_N_SLOTS == 0:
            print(f"[Slot {self.current_slot:3d} | Time {self.current_time:6.0f}s]")
            print(f"  PMs: {metrics['active_pms']}/{metrics['total_pms']} active | "
                  f"Containers: {metrics['total_containers']} running")
            
            # 🤖 RL Actions
            if self.enable_rl and metrics['rl_actions']:
                action_names = [self.rl_agent.get_policy_name(a) for a in metrics['rl_actions']]
                print(f"  🤖 RL Actions: {action_names}")
                print(f"  🤖 Reward: {metrics['rl_reward']:.2f} | Epsilon: {metrics['rl_epsilon']:.3f}")
            
            print(f"  Placed: {placed} | Delayed: {delayed} | Rejected: {rejected}")
            print(f"  Finished: {finished} | Violations: {violations}")
            print(f"  Avg CPU Util: {metrics['avg_cpu_util']:.2%}")
            print(f"  ⚡ Energy: {metrics['slot_energy_kwh']:.6f} kWh | "
                  f"Power: {metrics['total_power_watts']:.2f} W")
            print("-" * 60)
    
    def run_time_slot(self):
        """Execute one time slot."""
        placed, delayed, rejected = self._process_new_arrivals()
        finished, violations, rl_reward, rl_actions = self._execute_containers()
        metrics = self._compute_metrics(rl_reward, rl_actions)
        self._update_history(metrics)
        self._print_progress(metrics, placed, delayed, rejected, finished, violations)
        
        self.current_time += self.time_slot_duration
        self.current_slot += 1
    
    def run(self, num_slots=TOTAL_SIMULATION_SLOTS):
        """Run simulation."""
        print("\n" + "=" * 60)
        print("🚀 STARTING SIMULATION")
        print("=" * 60 + "\n")
        
        for slot in range(num_slots):
            self.run_time_slot()
        
        # Decay epsilon after episode
        if self.enable_rl:
            self.rl_agent.decay_epsilon()
        
        self._print_final_summary()
    
    def _print_final_summary(self):
        """Print final summary with RL stats."""
        print("\n" + "=" * 60)
        print("✅ SIMULATION COMPLETED")
        print("=" * 60)
        
        # Container statistics
        print(f"\n📊 CONTAINER STATISTICS:")
        print(f"  Total simulation time: {self.current_time}s ({self.current_slot} slots)")
        print(f"  Total containers arrived: {self.total_containers_arrived}")
        print(f"  Total containers finished: {self.total_containers_finished}")
        print(f"  Total deadline violations: {self.total_deadline_violations}")
        print(f"  Total rejections: {self.total_rejections}")
        print(f"  Violation rate: {self.total_deadline_violations/max(1, self.total_containers_finished):.2%}")
        print(f"  Rejection rate: {self.total_rejections/max(1, self.total_containers_arrived):.2%}")
        
        # PM statistics
        print(f"\n💻 SERVER STATISTICS:")
        print(f"  Peak PMs used: {max(self.history['active_pms'])}")
        print(f"  Total PMs created: {len(self.pms)}")
        print(f"  Avg CPU utilization: {sum(self.history['avg_cpu_util'])/len(self.history['avg_cpu_util']):.2%}")
        
        # Energy statistics
        energy_stats = self.energy_model.get_energy_statistics()
        print(f"\n⚡ ENERGY STATISTICS:")
        print(f"  Total energy consumed: {energy_stats['total_energy_kwh']:.6f} kWh")
        print(f"  Average energy per slot: {energy_stats['avg_energy_per_slot']:.6f} kWh")
        print(f"  Peak energy in slot: {energy_stats['peak_energy_slot']:.6f} kWh")
        print(f"  Estimated cost (@ $0.12/kWh): ${self.energy_model.estimate_cost():.4f}")
        
        # 🤖 RL statistics
        if self.enable_rl:
            rl_stats = self.rl_agent.get_statistics()
            print(f"\n🤖 RL AGENT STATISTICS:")
            print(f"  Q-table size: {rl_stats['q_table_size']} state-action pairs")
            print(f"  Total Q-updates: {rl_stats['total_updates']}")
            print(f"  Final epsilon: {rl_stats['epsilon']:.3f}")
            print(f"  Learning rate: {rl_stats['learning_rate']}")
            print(f"  Discount factor: {rl_stats['discount_factor']}")
            
            avg_reward = sum(self.history['rl_rewards']) / len(self.history['rl_rewards'])
            print(f"  Average reward: {avg_reward:.2f}")
        
        # Placement statistics
        placement_stats = self.placement_module.get_statistics()
        print(f"\n📦 PLACEMENT STATISTICS:")
        print(f"  Strategy: {placement_stats['strategy']}")
        print(f"  Total placements: {placement_stats['total_placements']}")
        print(f"  New PMs started: {placement_stats['new_pm_starts']}")
        
        print("\n" + "=" * 60 + "\n")
    
    def save_qtable(self, filepath="results/qtables/qtable.pkl"):
        """Save Q-table."""
        if self.enable_rl:
            self.rl_agent.save(filepath)
    
    def load_qtable(self, filepath="results/qtables/qtable.pkl"):
        """Load Q-table."""
        if self.enable_rl:
            self.rl_agent.load(filepath)
    
    def get_history(self):
        """Get history."""
        return self.history
    
    def get_summary_stats(self):
        """Get summary stats."""
        energy_stats = self.energy_model.get_energy_statistics()
        
        stats = {
            'total_containers_arrived': self.total_containers_arrived,
            'total_containers_finished': self.total_containers_finished,
            'total_deadline_violations': self.total_deadline_violations,
            'total_rejections': self.total_rejections,
            'peak_active_pms': max(self.history['active_pms']) if self.history['active_pms'] else 0,
            'total_pms_created': len(self.pms),
            'violation_rate': self.total_deadline_violations / max(1, self.total_containers_finished),
            'rejection_rate': self.total_rejections / max(1, self.total_containers_arrived),
            'total_energy_kwh': energy_stats['total_energy_kwh'],
            'avg_energy_per_slot': energy_stats['avg_energy_per_slot'],
            'estimated_cost_usd': self.energy_model.estimate_cost()
        }
        
        if self.enable_rl:
            stats['rl_q_table_size'] = len(self.rl_agent.q_table)
            stats['rl_total_updates'] = self.rl_agent.total_updates
        
        return stats


def run_simulation(num_slots=50, workload_pattern='random', 
                   placement_strategy='first_fit', enable_rl=True):
    """Quick simulation runner."""
    sim = Simulator(workload_pattern=workload_pattern, 
                    placement_strategy=placement_strategy,
                    enable_rl=enable_rl)
    sim.run(num_slots=num_slots)
    return sim
=======
a
>>>>>>> 3fdc487e76549e8239d83e40c86355f2a7360963
