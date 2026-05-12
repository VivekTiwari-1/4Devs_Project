"""
scripts/run_comparison.py
Runs all scenarios and saves results for comparison plotting.

Scenarios:
  1. Baseline        — no RL, no migration
  2. QMIX Only       — RL on, migration off
  3. QMIX + Migration — RL on, migration on  (full system)
  4. Q-Learning      — optional, loads from saved qtable pkl

Usage:
    python scripts/run_comparison.py                     # 3 scenarios, 3 runs each, 500 slots
    python scripts/run_comparison.py --slots 200         # shorter run
    python scripts/run_comparison.py --runs 5            # more averaging
    python scripts/run_comparison.py --include-qlearning # adds Q-Learning scenario
    python scripts/run_comparison.py --no-plot           # skip auto-plot
"""

import sys
import os
import json
import argparse
import pickle

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.environment.simulator import Simulator
from src.config.config import TOTAL_SIMULATION_SLOTS


# ── scenario definitions ──────────────────────────────────────────────────────
BASE_SCENARIOS = [
    {
        'key':   'baseline',
        'name':  'Baseline',
        'label': 'Baseline\n(No RL, No Migration)',
        'color': '#78909C',
        'enable_rl':        False,
        'enable_migration': False,
    },
    {
        'key':   'qmix_only',
        'name':  'QMIX Only',
        'label': 'QMIX Only\n(No Migration)',
        'color': '#42A5F5',
        'enable_rl':        True,
        'enable_migration': False,
    },
    {
        'key':   'qmix_migration',
        'name':  'QMIX + Migration',
        'label': 'QMIX + Migration\n(Full System)',
        'color': '#26A69A',
        'enable_rl':        True,
        'enable_migration': True,
    },
]

QLEARNING_SCENARIO = {
    'key':   'qlearning',
    'name':  'Q-Learning',
    'label': 'Q-Learning\n(Previous Version)',
    'color': '#FFA726',
    'enable_rl':        True,
    'enable_migration': True,
    'qlearning_mode':   True,
}


def _load_qmix_qtable(sim, qtable_path):
    """Load saved QMIX Q-tables into agent."""
    if not os.path.exists(qtable_path):
        print(f"    ⚠️  No QMIX qtable at {qtable_path} — using untrained agent")
        return
    with open(qtable_path, 'rb') as f:
        data = pickle.load(f)
    sim.rl_agent.agent_q_tables  = data.get('agent_q_tables',  sim.rl_agent.agent_q_tables)
    sim.rl_agent.target_q_tables = data.get('target_q_tables', sim.rl_agent.target_q_tables)
    sim.rl_agent.epsilon = 0.05   # exploit the trained policy
    print(f"    ✅ Loaded QMIX qtable — epsilon set to 0.05")


def _load_qlearning_qtable(sim, qtable_path):
    """
    Load a Q-Learning q_table into a QMIX simulator.
    Maps single-agent table → Agent 0's table so the simulation still runs.
    Falls back gracefully if file not found or format differs.
    """
    if not os.path.exists(qtable_path):
        print(f"    ⚠️  No Q-Learning qtable at {qtable_path} — using untrained agent")
        return
    try:
        with open(qtable_path, 'rb') as f:
            data = pickle.load(f)
        # Old Q-Learning format: dict of {(state, action): value}
        # or new format with 'q_table' key
        if isinstance(data, dict) and 'q_table' in data:
            qtable = data['q_table']
        elif isinstance(data, dict):
            qtable = data
        else:
            print("    ⚠️  Unrecognised Q-Learning format — using untrained agent")
            return
        # Map to agent 0 — all other agents use default 0.0 Q-values
        sim.rl_agent.agent_q_tables[0] = qtable
        sim.rl_agent.epsilon = 0.05
        print(f"    ✅ Loaded Q-Learning qtable ({len(qtable)} entries) into Agent 0")
    except Exception as e:
        print(f"    ⚠️  Failed to load Q-Learning qtable: {e}")


def run_single(scenario, slots, run_idx, results_dir):
    """Run one simulation and return summary stats dict."""
    sim = Simulator(
        workload_pattern='random',
        placement_strategy='first_fit',
        enable_rl=scenario['enable_rl'],
        enable_migration=scenario['enable_migration'],
    )

    if scenario['enable_rl']:
        if scenario.get('qlearning_mode'):
            qpath = os.path.join(results_dir, 'qtables', 'trained_agent.pkl')
            _load_qlearning_qtable(sim, qpath)
        else:
            qpath = os.path.join(results_dir, 'qtables', 'qmix_qtable_latest.pkl')
            _load_qmix_qtable(sim, qpath)

    sim.run(num_slots=slots)
    return sim.get_summary_stats(), sim.history


def run_comparison(slots=TOTAL_SIMULATION_SLOTS, num_runs=3,
                   include_qlearning=False, results_dir=None):
    """
    Run all scenarios, average over num_runs, save JSON.
    Returns the results dict (keyed by scenario key).
    """
    if results_dir is None:
        results_dir = os.path.join(PROJECT_ROOT, 'results')
    os.makedirs(os.path.join(results_dir, 'logs'), exist_ok=True)

    scenarios = list(BASE_SCENARIOS)
    if include_qlearning:
        scenarios.append(QLEARNING_SCENARIO)

    all_results = {}

    for sc in scenarios:
        print(f"\n{'='*60}")
        print(f"🧪  {sc['name']}")
        print('='*60)

        run_stats_list = []
        all_histories  = []

        for run_idx in range(num_runs):
            print(f"\n  Run {run_idx+1}/{num_runs}...")
            stats, history = run_single(sc, slots, run_idx, results_dir)
            run_stats_list.append(stats)
            all_histories.append(history)

        # ── average numeric fields ────────────────────────────────────────────
        def avg(key, default=0):
            vals = [s.get(key, default) for s in run_stats_list]
            return sum(vals) / len(vals)

        avg_stats = {
            'scenario_key':   sc['key'],
            'scenario_name':  sc['name'],
            'scenario_label': sc['label'],
            'scenario_color': sc['color'],
            'slots':          slots,
            'num_runs':       num_runs,
            # container metrics
            'containers_arrived':   avg('total_containers_arrived'),
            'containers_finished':  avg('total_containers_finished'),
            'deadline_violations':  avg('total_deadline_violations'),
            'violation_rate':       avg('violation_rate'),
            'rejection_rate':       avg('rejection_rate'),
            # server metrics
            'peak_active_pms':  avg('peak_active_pms'),
            'total_pms_created': avg('total_pms_created'),
            'avg_cpu_util':     avg('avg_cpu_util', 0),
            # energy
            'total_energy_kwh': avg('total_energy_kwh'),
            'avg_energy_slot':  avg('avg_energy_per_slot'),
            'estimated_cost':   avg('estimated_cost_usd'),
            # migrations
            'total_migrations': avg('total_migrations'),
            'pms_turned_off':   avg('pms_turned_off'),
            # RL
            'qmix_total_updates': avg('qmix_total_updates'),
            'qmix_total_entries': avg('qmix_total_entries'),
        }

        # ── average reward/epsilon histories across runs ──────────────────────
        if sc['enable_rl']:
            reward_runs = [h.get('rl_rewards', []) for h in all_histories]
            eps_runs    = [h.get('rl_epsilon',  []) for h in all_histories]
            non_empty_r = [r for r in reward_runs if r]
            non_empty_e = [e for e in eps_runs if e]
            if non_empty_r:
                min_len = min(len(r) for r in non_empty_r)
                avg_stats['avg_rewards'] = [
                    sum(r[i] for r in non_empty_r if i < len(r)) / len(non_empty_r)
                    for i in range(min_len)
                ]
            if non_empty_e:
                min_len = min(len(e) for e in non_empty_e)
                avg_stats['avg_epsilon'] = [
                    sum(e[i] for e in non_empty_e if i < len(e)) / len(non_empty_e)
                    for i in range(min_len)
                ]

        # ── average per-slot energy across runs ──────────────────────────────
        energy_runs = [h.get('energy_consumed', []) for h in all_histories]
        non_empty_nrg = [e for e in energy_runs if e]
        if non_empty_nrg:
            min_len = min(len(e) for e in non_empty_nrg)
        if non_empty_nrg and min_len > 0:
            avg_stats['avg_energy_history'] = [
                sum(e[i] for e in energy_runs if i < len(e)) / num_runs
                for i in range(min_len)
            ]

        # ── average per-slot violation history ───────────────────────────────
        viol_runs = [h.get('violations', []) for h in all_histories]
        non_empty_viol = [v for v in viol_runs if v]
        if non_empty_viol:
            min_len = min(len(v) for v in non_empty_viol)
        if non_empty_viol and min_len > 0:
            raw = [
                sum(v[i] for v in viol_runs if i < len(v)) / num_runs
                for i in range(min_len)
            ]
            # convert cumulative → per-slot
            avg_stats['avg_violations_per_slot'] = (
                [raw[0]] + [raw[i] - raw[i-1] for i in range(1, len(raw))]
            )

        all_results[sc['key']] = avg_stats

        print(f"\n  📊 Average over {num_runs} runs:")
        print(f"     Energy:     {avg_stats['total_energy_kwh']:.2f} kWh")
        print(f"     Cost:       ${avg_stats['estimated_cost']:.4f}")
        print(f"     Violations: {avg_stats['deadline_violations']:.1f}")
        print(f"     Peak PMs:   {avg_stats['peak_active_pms']:.1f}")
        if sc['enable_migration']:
            print(f"     Migrations: {avg_stats['total_migrations']:.1f}")

    # ── print comparison table ────────────────────────────────────────────────
    baseline = all_results.get('baseline', {})
    b_energy = baseline.get('total_energy_kwh', 1)
    b_cost   = baseline.get('estimated_cost', 1)
    b_viol   = baseline.get('deadline_violations', 1)

    print(f"\n{'='*80}")
    print("📈  COMPARISON SUMMARY")
    print('='*80)
    print(f"{'Scenario':<30} {'Energy (kWh)':<20} {'Cost ($)':<18} {'Violations':<14}")
    print('-'*80)
    for sc in scenarios:
        s = all_results[sc['key']]
        e_red = (b_energy - s['total_energy_kwh']) / max(b_energy, 1e-9) * 100
        c_red = (b_cost   - s['estimated_cost'])   / max(b_cost,   1e-9) * 100
        v_red = (b_viol   - s['deadline_violations']) / max(b_viol, 1e-9) * 100
        print(
            f"{sc['name']:<30} "
            f"{s['total_energy_kwh']:>7.2f}  ({e_red:>+6.1f}%)   "
            f"${s['estimated_cost']:>7.4f} ({c_red:>+6.1f}%)   "
            f"{s['deadline_violations']:>5.1f} ({v_red:>+6.1f}%)"
        )
    print('='*80)

    # ── save ─────────────────────────────────────────────────────────────────
    out_path = os.path.join(results_dir, 'logs', 'comparison_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅  Results saved to {out_path}")

    return all_results


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run QMIX comparison experiments")
    parser.add_argument('--slots',  type=int, default=TOTAL_SIMULATION_SLOTS,
                        help='Simulation slots per run')
    parser.add_argument('--runs',   type=int, default=3,
                        help='Number of runs to average over')
    parser.add_argument('--include-qlearning', action='store_true',
                        help='Include Q-Learning scenario (needs trained_agent.pkl)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip auto-plotting after comparison')
    args = parser.parse_args()

    print("\n" + "🔬"*30)
    print("QMIX COMPARISON RUNNER")
    print("🔬"*30 + "\n")
    print(f"  Slots per run: {args.slots}")
    print(f"  Runs:          {args.runs}")
    print(f"  Q-Learning:    {'YES' if args.include_qlearning else 'NO'}\n")

    results = run_comparison(
        slots=args.slots,
        num_runs=args.runs,
        include_qlearning=args.include_qlearning,
    )

    if not args.no_plot:
        try:
            from scripts.plot_comparison import plot_all
            plot_all()
        except Exception as e:
            print(f"\n⚠️  Auto-plot failed: {e}")
            print("   Run manually: python scripts/plot_comparison.py")