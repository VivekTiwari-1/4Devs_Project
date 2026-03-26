"""
Main Simulator Class - With QMIX Multi-Agent RL (8th Semester Upgrade)
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
    ENABLE_MIGRATION,
    NUM_QMIX_AGENTS,
    RL_DEADLINE_VIOLATION_PENALTY,
    RL_REJECTION_PENALTY,
    QMIX_BATCH_SIZE,
    QMIX_UPDATE_TARGET_EVERY
)
from environment.pm import PhysicalMachine
from environment.workload_generator import WorkloadGenerator
from environment.energy_model import EnergyModel
from modules.placement import PlacementModule, DelayedPlacementQueue
from modules.migration import MigrationModule
from modules.policies import AllocationPolicies
from rl.qmix_agent import QMIXAgent


class Simulator:
    """
    Main simulation engine with QMIX Multi-Agent RL.

    8th Semester Upgrade:
    - Multi-agent QMIX replacing single-agent Q-Learning
    - Experience replay buffer
    - Target networks for stability
    - Mixing network with hypernetwork
    """

    def __init__(self, workload_pattern='random', placement_strategy='first_fit',
                 enable_rl=ENABLE_RL_SCALING, enable_migration=ENABLE_MIGRATION):
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

        # QMIX Multi-Agent RL
        self.enable_rl = enable_rl
        if self.enable_rl:
            # FIX: Pass batch_size and update_target_every from config
            self.rl_agent = QMIXAgent(
                num_agents=NUM_QMIX_AGENTS,
                batch_size=QMIX_BATCH_SIZE,
                update_target_every=QMIX_UPDATE_TARGET_EVERY
            )

        # Migration Module
        self.enable_migration = enable_migration
        if self.enable_migration:
            self.migration_module = MigrationModule()

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
            'rl_actions': [],
            'rl_rewards': [],
            'rl_epsilon': [],
            'migrations': []
        }

        mode_parts = []
        if self.enable_rl:
            mode_parts.append("QMIX")
        if self.enable_migration:
            mode_parts.append("Migration")
        mode = "+".join(mode_parts) if mode_parts else "Baseline"

        print(f"✅ Simulator initialized ({mode})")
        print(f"   Workload pattern: {workload_pattern}")
        print(f"   Placement strategy: {placement_strategy}")
        print(f"   Time slot duration: {TIME_SLOT_DURATION}s")
        print(f"   ⚡ Energy model: ENABLED")
        if self.enable_rl:
            print(f"   🤖 QMIX Multi-Agent: ENABLED ({NUM_QMIX_AGENTS} agents)")
        if self.enable_migration:
            print(f"   🔄 Migration: ENABLED")
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

    def _compute_reward(self, pm, violations, rejected=0):
        """
        Compute normalized reward for a single PM/agent.

        FIX: Old reward ranged from -0.83 to -4001 (4800x swing).
        New reward is normalized to roughly [-1, 0] range by dividing
        energy by static_power baseline and using scaled penalties.
        This keeps TD errors small and learning stable.

        Args:
            pm: PhysicalMachine
            violations (int): Deadline violations this slot
            rejected (int): Rejected containers this slot

        Returns:
            float: Normalized reward
        """
        # Normalize energy: divide by baseline (static power * slot duration in hours)
        # Static power = 100W, slot = 30s = 30/3600 h → baseline = 100 * 30/3600 ≈ 0.833 kWh
        energy = self.energy_model.compute_pm_energy(pm)
        baseline_energy = 100 * (self.time_slot_duration / 3600)  # idle PM energy
        normalized_energy = energy / max(baseline_energy, 1e-6)

        # Scaled penalties (from config — already reduced to 10 and 5)
        violation_penalty = violations * RL_DEADLINE_VIOLATION_PENALTY
        rejection_penalty = rejected * RL_REJECTION_PENALTY

        reward = -normalized_energy - violation_penalty - rejection_penalty
        return reward

    def _execute_containers_with_qmix(self):
        """
        Execute containers with QMIX multi-agent RL.

        Returns:
            tuple: (finished, violations, total_reward, actions)
        """
        total_finished = 0
        total_violations = 0

        # Get active PMs with containers
        active_pms = [pm for pm in self.pms if pm.is_on and len(pm.containers) > 0]

        if not active_pms:
            return 0, 0, 0.0, []

        # Update current time for containers
        for pm in active_pms:
            for container in pm.containers:
                container._current_time = self.current_time

        # Get states for all active PMs
        # FIX: Pad states to NUM_QMIX_AGENTS so global state size is always consistent
        states = [self.rl_agent.get_state(pm) for pm in active_pms]
        while len(states) < NUM_QMIX_AGENTS:
            states.append((0, 0, 0, 0))  # idle agent state — 4-tuple

        # QMIX: Decentralized execution
        actions = self.rl_agent.select_actions(states)

        # Apply policies and execute
        rewards = []
        slot_violations = 0

        for i, pm in enumerate(active_pms):
            action = actions[i] if i < len(actions) else 0

            # Apply selected policy
            allocations = AllocationPolicies.apply_policy_by_index(
                action, pm.containers, pm.available_cores()
            )

            # Assign cores
            for j, container in enumerate(pm.containers):
                if j < len(allocations):
                    container.assigned_cores = max(0.1, allocations[j])

            # Execute containers for this time slot
            for container in pm.containers:
                container.execute(CORE_SPEED, self.time_slot_duration)

            # Remove finished containers
            finished, violations = pm.remove_finished_containers(self.current_time)
            total_finished += finished
            total_violations += violations
            slot_violations += violations

            # FIX: Use normalized reward function
            reward = self._compute_reward(pm, violations)
            rewards.append(reward)

        # Pad rewards for idle agents
        while len(rewards) < NUM_QMIX_AGENTS:
            rewards.append(0.0)

        # Get next states (also padded)
        next_states = [self.rl_agent.get_state(pm) for pm in active_pms]
        while len(next_states) < NUM_QMIX_AGENTS:
            next_states.append((0, 0, 0, 0))

        # QMIX: Store ONE transition per slot with all agent info
        self.rl_agent.store_transition(states, actions, rewards, next_states, done=False)

        # QMIX: Centralized training
        self.rl_agent.train()

        self.total_containers_finished += total_finished
        self.total_deadline_violations += total_violations

        total_reward = sum(rewards)
        return total_finished, total_violations, total_reward, actions[:len(active_pms)]

    def _execute_containers_baseline(self):
        """Baseline execution without RL (fixed allocation)."""
        total_finished = 0
        total_violations = 0

        for pm in self.pms:
            if pm.is_on and len(pm.containers) > 0:
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
        """Execute containers (baseline or QMIX)."""
        if self.enable_rl:
            return self._execute_containers_with_qmix()
        else:
            return self._execute_containers_baseline()

    def _perform_migration(self):
        """Perform migration if enabled."""
        if not self.enable_migration:
            return 0
        return self.migration_module.check_and_migrate(self.pms, self.current_time)

    def _compute_energy(self):
        """Compute energy consumption."""
        slot_energy = self.energy_model.compute_datacenter_energy(self.pms)
        total_power = sum(self.energy_model.compute_pm_power(pm) for pm in self.pms)
        return slot_energy, total_power

    def _compute_metrics(self, rl_reward, rl_actions, migrations):
        """Compute metrics."""
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
            'rl_reward': rl_reward,
            'rl_actions': rl_actions,
            'rl_epsilon': self.rl_agent.epsilon if self.enable_rl else 0.0,
            'migrations': migrations
        }

        return metrics

    def _update_history(self, metrics):
        """Update simulation history."""
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
        if self.enable_migration:
            self.history['migrations'].append(metrics['migrations'])

    def _print_progress(self, metrics, placed, delayed, rejected, finished, violations):
        """Print progress."""
        if DEBUG_MODE or self.current_slot % PRINT_EVERY_N_SLOTS == 0:
            print(f"[Slot {self.current_slot:3d} | Time {self.current_time:6.0f}s]")
            print(f"  PMs: {metrics['active_pms']}/{metrics['total_pms']} active | "
                  f"Containers: {metrics['total_containers']} running")

            if self.enable_rl and metrics['rl_actions']:
                action_names = [self.rl_agent.get_policy_name(a) for a in metrics['rl_actions']]
                print(f"  🤖 QMIX Actions: {action_names}")
                print(f"  🤖 Reward: {metrics['rl_reward']:.2f} | Epsilon: {metrics['rl_epsilon']:.3f}")

            if self.enable_migration and metrics['migrations'] > 0:
                print(f"  🔄 Migrations: {metrics['migrations']}")

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

        # Decay epsilon every slot (already correct in original)
        if self.enable_rl:
            self.rl_agent.decay_epsilon()

        migrations = self._perform_migration()

        metrics = self._compute_metrics(rl_reward, rl_actions, migrations)
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

        self._print_final_summary()

    def _print_final_summary(self):
        """Print final summary."""
        print("\n" + "=" * 60)
        print("✅ SIMULATION COMPLETED")
        print("=" * 60)

        print(f"\n📊 CONTAINER STATISTICS:")
        print(f"  Total simulation time: {self.current_time}s ({self.current_slot} slots)")
        print(f"  Total containers arrived:  {self.total_containers_arrived}")
        print(f"  Total containers finished: {self.total_containers_finished}")
        print(f"  Total deadline violations: {self.total_deadline_violations}")
        print(f"  Total rejections: {self.total_rejections}")
        # FIX: violation rate should be violations/arrived, not violations/finished
        print(f"  Violation rate: {self.total_deadline_violations / max(1, self.total_containers_arrived):.2%}")
        print(f"  Rejection rate: {self.total_rejections / max(1, self.total_containers_arrived):.2%}")

        print(f"\n💻 SERVER STATISTICS:")
        print(f"  Peak PMs used: {max(self.history['active_pms'])}")
        print(f"  Total PMs created: {len(self.pms)}")
        print(f"  Avg CPU utilization: {sum(self.history['avg_cpu_util']) / len(self.history['avg_cpu_util']):.2%}")

        energy_stats = self.energy_model.get_energy_statistics()
        print(f"\n⚡ ENERGY STATISTICS:")
        print(f"  Total energy consumed: {energy_stats['total_energy_kwh']:.6f} kWh")
        print(f"  Average energy per slot: {energy_stats['avg_energy_per_slot']:.6f} kWh")
        print(f"  Peak energy in slot: {energy_stats['peak_energy_slot']:.6f} kWh")
        print(f"  Estimated cost (@ $0.12/kWh): ${self.energy_model.estimate_cost():.4f}")

        if self.enable_rl:
            rl_stats = self.rl_agent.get_statistics()
            print(f"\n🤖 QMIX MULTI-AGENT STATISTICS:")
            print(f"  Number of agents: {rl_stats['num_agents']}")
            print(f"  Total Q-entries: {rl_stats['total_q_entries']}")
            print(f"  Replay buffer size: {rl_stats['replay_buffer_size']}")
            print(f"  Training steps: {rl_stats['train_step_counter']}")
            print(f"  Final epsilon: {rl_stats['epsilon']:.3f}")
            print(f"  Total updates: {rl_stats['total_updates']}")
            avg_reward = sum(self.history['rl_rewards']) / len(self.history['rl_rewards']) if self.history['rl_rewards'] else 0
            print(f"  Average reward: {avg_reward:.2f}")
            # FIX: Show reward trend (first 50 vs last 50 slots)
            if len(self.history['rl_rewards']) >= 100:
                early  = sum(self.history['rl_rewards'][:50])  / 50
                late   = sum(self.history['rl_rewards'][-50:]) / 50
                trend  = "📈 IMPROVING" if late > early else "📉 NOT IMPROVING"
                print(f"  Reward trend: {early:.2f} (early) → {late:.2f} (late)  {trend}")

        if self.enable_migration:
            migration_stats = self.migration_module.get_statistics()
            print(f"\n🔄 MIGRATION STATISTICS:")
            print(f"  Total migrations: {migration_stats['total_migrations']}")
            print(f"  PMs turned OFF: {migration_stats['pms_turned_off']}")
            print(f"  PMs turned ON: {migration_stats['pms_turned_on']}")
            print(f"  Failed migrations: {migration_stats['failed_migrations']}")

        placement_stats = self.placement_module.get_statistics()
        print(f"\n📦 PLACEMENT STATISTICS:")
        print(f"  Strategy: {placement_stats['strategy']}")
        print(f"  Total placements: {placement_stats['total_placements']}")
        print(f"  New PMs started: {placement_stats['new_pm_starts']}")

        print("\n" + "=" * 60 + "\n")

        print(f"📈 KEY PERFORMANCE INDICATORS:")
        print(f"  ✓ Containers processed: {self.total_containers_finished}/{self.total_containers_arrived}")
        print(f"  ✓ Success rate: {self.total_containers_finished / max(1, self.total_containers_arrived):.2%}")
        print(f"  ✓ Deadline compliance: {1 - self.total_deadline_violations / max(1, self.total_containers_arrived):.2%}")
        print(f"  ✓ Peak PM utilization: {max(self.history['active_pms'])} servers")
        print(f"  ✓ Total energy: {energy_stats['total_energy_kwh']:.6f} kWh")
        print(f"  ✓ Estimated cost: ${self.energy_model.estimate_cost():.4f}")
        print("\n" + "=" * 60 + "\n")

    def save_qtable(self, filepath="results/qtables/qmix_qtable.pkl"):
        if self.enable_rl:
            self.rl_agent.save(filepath)

    def load_qtable(self, filepath="results/qtables/qmix_qtable.pkl"):
        if self.enable_rl:
            self.rl_agent.load(filepath)

    def get_history(self):
        return self.history

    def get_summary_stats(self):
        energy_stats = self.energy_model.get_energy_statistics()
        stats = {
            'total_containers_arrived':   self.total_containers_arrived,
            'total_containers_finished':  self.total_containers_finished,
            'total_deadline_violations':  self.total_deadline_violations,
            'total_rejections':           self.total_rejections,
            'peak_active_pms':            max(self.history['active_pms']) if self.history['active_pms'] else 0,
            'total_pms_created':          len(self.pms),
            # FIX: both rates denominated by arrived
            'violation_rate':             self.total_deadline_violations / max(1, self.total_containers_arrived),
            'rejection_rate':             self.total_rejections / max(1, self.total_containers_arrived),
            'total_energy_kwh':           energy_stats['total_energy_kwh'],
            'avg_energy_per_slot':        energy_stats['avg_energy_per_slot'],
            'estimated_cost_usd':         self.energy_model.estimate_cost()
        }

        if self.enable_rl:
            rl_stats = self.rl_agent.get_statistics()
            stats['qmix_num_agents']    = rl_stats['num_agents']
            stats['qmix_total_entries'] = rl_stats['total_q_entries']
            stats['qmix_replay_size']   = rl_stats['replay_buffer_size']
            stats['qmix_total_updates'] = rl_stats['total_updates']

        if self.enable_migration:
            migration_stats = self.migration_module.get_statistics()
            stats['total_migrations'] = migration_stats['total_migrations']
            stats['pms_turned_off']   = migration_stats['pms_turned_off']

        return stats


def run_simulation(num_slots=TOTAL_SIMULATION_SLOTS, workload_pattern='random',
                   placement_strategy='first_fit', enable_rl=True, enable_migration=True):
    """Quick simulation runner."""
    sim = Simulator(
        workload_pattern=workload_pattern,
        placement_strategy=placement_strategy,
        enable_rl=enable_rl,
        enable_migration=enable_migration
    )
    sim.run(num_slots=num_slots)
    return sim