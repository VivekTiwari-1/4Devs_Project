"""
QMIX Results Saver
Saves simulation history, Q-tables, and training metrics to disk after each run.
Replaces the old Q-learning save system with QMIX-compatible saving.
"""

import os
import json
import csv
import pickle
import numpy as np
from datetime import datetime


class QMIXResultsSaver:
    """
    Handles all saving of QMIX simulation results to disk.

    Saves:
    - results/simulation_outputs/run_TIMESTAMP.json  (full history)
    - results/simulation_outputs/metrics_TIMESTAMP.csv (per-slot metrics)
    - results/qtables/qmix_qtable_TIMESTAMP.pkl      (Q-tables + network weights)
    - results/simulation_outputs/summary_latest.json  (latest run summary, for quick comparison)
    """

    def __init__(self, results_dir="results"):
        self.results_dir = results_dir
        self.sim_outputs_dir = os.path.join(results_dir, "simulation_outputs")
        self.qtables_dir     = os.path.join(results_dir, "qtables")
        self.logs_dir        = os.path.join(results_dir, "logs")

        # Create all directories
        for d in [self.sim_outputs_dir, self.qtables_dir, self.logs_dir]:
            os.makedirs(d, exist_ok=True)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def save_all(self, sim):
        """
        Save everything after simulation completes.

        Args:
            sim: Simulator instance (after sim.run() has been called)
        """
        print("\n💾 Saving results...")

        self._save_per_slot_csv(sim)
        self._save_full_history_json(sim)
        self._save_summary_json(sim)
        self._save_qtables(sim)

        print(f"✅ All results saved with timestamp: {self.timestamp}")
        print(f"   📁 {self.sim_outputs_dir}/")
        print(f"   📁 {self.qtables_dir}/")

    def _save_per_slot_csv(self, sim):
        """
        Save per-slot metrics to CSV — the main file used for plotting.
        Each row = one time slot.
        """
        filepath = os.path.join(self.sim_outputs_dir, f"metrics_{self.timestamp}.csv")
        latest   = os.path.join(self.sim_outputs_dir, "metrics_latest.csv")

        history = sim.history

        # Build rows
        rows = []
        num_slots = len(history['time'])

        for i in range(num_slots):
            row = {
                'slot':              i,
                'time':              history['time'][i],
                'active_pms':        history['active_pms'][i],
                'total_containers':  history['total_containers'][i],
                'containers_finished': history['finished_containers'][i],
                'violations':        history['violations'][i],
                'avg_cpu_util':      history['avg_cpu_util'][i],
                'energy_kwh':        history['energy_consumed'][i],
                'power_watts':       history['total_power'][i],
            }

            if sim.enable_rl and i < len(history['rl_rewards']):
                row['rl_reward']  = history['rl_rewards'][i]
                row['rl_epsilon'] = history['rl_epsilon'][i]
                # Slot-level violations (derivative of cumulative)
                row['slot_violations'] = (
                    history['violations'][i] - history['violations'][i-1]
                ) if i > 0 else history['violations'][0]
            else:
                row['rl_reward']       = 0.0
                row['rl_epsilon']      = 0.0
                row['slot_violations'] = 0

            if sim.enable_migration and i < len(history['migrations']):
                row['migrations'] = history['migrations'][i]
            else:
                row['migrations'] = 0

            rows.append(row)

        # Write CSV
        if rows:
            fieldnames = list(rows[0].keys())
            for path in [filepath, latest]:
                with open(path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)

        print(f"  ✓ Per-slot CSV: metrics_{self.timestamp}.csv")

    def _save_full_history_json(self, sim):
        """Save complete simulation history as JSON."""
        filepath = os.path.join(self.sim_outputs_dir, f"run_{self.timestamp}.json")

        # Convert numpy types to Python native for JSON serialization
        def to_serializable(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, list):
                return [to_serializable(x) for x in obj]
            return obj

        history_clean = {
            k: to_serializable(v) for k, v in sim.history.items()
        }

        data = {
            'timestamp':   self.timestamp,
            'config': {
                'total_slots':       sim.current_slot,
                'total_time':        sim.current_time,
                'num_qmix_agents':   sim.rl_agent.num_agents if sim.enable_rl else 0,
                'enable_rl':         sim.enable_rl,
                'enable_migration':  sim.enable_migration,
            },
            'final_stats': {
                'containers_arrived':   sim.total_containers_arrived,
                'containers_finished':  sim.total_containers_finished,
                'deadline_violations':  sim.total_deadline_violations,
                'rejections':           sim.total_rejections,
                'violation_rate':       sim.total_deadline_violations / max(1, sim.total_containers_arrived),
                'total_pms_created':    len(sim.pms),
            },
            'history': history_clean
        }

        if sim.enable_rl:
            rl_stats = sim.rl_agent.get_statistics()
            data['qmix_stats'] = {
                'num_agents':        rl_stats['num_agents'],
                'total_q_entries':   rl_stats['total_q_entries'],
                'replay_buffer_size': rl_stats['replay_buffer_size'],
                'total_updates':     rl_stats['total_updates'],
                'final_epsilon':     rl_stats['epsilon'],
                'train_step_counter': rl_stats['train_step_counter'],
            }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"  ✓ Full history JSON: run_{self.timestamp}.json")

    def _save_summary_json(self, sim):
        """Save a compact summary — overwrites latest for easy comparison."""
        filepath = os.path.join(self.sim_outputs_dir, "summary_latest.json")
        archive  = os.path.join(self.sim_outputs_dir, f"summary_{self.timestamp}.json")

        rewards = sim.history.get('rl_rewards', [])
        early_reward = float(np.mean(rewards[:50]))  if len(rewards) >= 50  else 0.0
        late_reward  = float(np.mean(rewards[-50:])) if len(rewards) >= 50  else 0.0

        summary = {
            'timestamp':          self.timestamp,
            'slots':              sim.current_slot,
            'containers_arrived': sim.total_containers_arrived,
            'containers_finished': sim.total_containers_finished,
            'violation_rate':     round(sim.total_deadline_violations / max(1, sim.total_containers_arrived), 4),
            'rejection_rate':     round(sim.total_rejections / max(1, sim.total_containers_arrived), 4),
            'peak_pms':           max(sim.history['active_pms']) if sim.history['active_pms'] else 0,
            'total_pms_created':  len(sim.pms),
            'avg_cpu_util':       round(float(np.mean(sim.history['avg_cpu_util'])), 4),
            'total_energy_kwh':   round(float(np.sum(sim.history['energy_consumed'])), 4),
            'early_avg_reward':   round(early_reward, 4),
            'late_avg_reward':    round(late_reward, 4),
            'reward_improved':    late_reward > early_reward,
            'final_epsilon':      round(sim.rl_agent.epsilon, 4) if sim.enable_rl else 0,
            'total_q_entries':    sim.rl_agent.get_statistics()['total_q_entries'] if sim.enable_rl else 0,
            'total_migrations':   sim.migration_module.get_statistics()['total_migrations'] if sim.enable_migration else 0,
        }

        for path in [filepath, archive]:
            with open(path, 'w') as f:
                json.dump(summary, f, indent=2)

        print(f"  ✓ Summary JSON: summary_latest.json + summary_{self.timestamp}.json")

    def _save_qtables(self, sim):
        """
        Save QMIX Q-tables and mixing network weights.
        This is the QMIX equivalent of Q-learning's qtable save.
        """
        if not sim.enable_rl:
            return

        filepath = os.path.join(self.qtables_dir, f"qmix_qtable_{self.timestamp}.pkl")
        latest   = os.path.join(self.qtables_dir, "qmix_qtable_latest.pkl")

        agent = sim.rl_agent

        # Build save payload
        payload = {
            'agent_q_tables':    agent.agent_q_tables,
            'target_q_tables':   agent.target_q_tables,
            'epsilon':           agent.epsilon,
            'total_updates':     agent.total_updates,
            'episodes_trained':  agent.episodes_trained,
            'train_step_counter': agent.train_step_counter,
            'reward_history':    agent.reward_history,
            'num_agents':        agent.num_agents,
            'global_state_size': agent.global_state_size,
            'timestamp':         self.timestamp,
        }

        # Save mixing network weights if initialized
        if agent.mixing_net is not None:
            hn = agent.mixing_net.hyper_net
            payload['mixing_net_weights'] = {
                'weight_net_W1': hn.weight_net.W1.tolist(),
                'weight_net_b1': hn.weight_net.b1.tolist(),
                'weight_net_W2': hn.weight_net.W2.tolist(),
                'weight_net_b2': hn.weight_net.b2.tolist(),
                'bias_net_W1':   hn.bias_net.W1.tolist(),
                'bias_net_b1':   hn.bias_net.b1.tolist(),
                'bias_net_W2':   hn.bias_net.W2.tolist(),
                'bias_net_b2':   hn.bias_net.b2.tolist(),
            }

        for path in [filepath, latest]:
            with open(path, 'wb') as f:
                pickle.dump(payload, f)

        print(f"  ✓ Q-tables + network weights: qmix_qtable_{self.timestamp}.pkl")
        print(f"  ✓ Latest copy: qmix_qtable_latest.pkl")