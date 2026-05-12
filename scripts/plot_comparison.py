"""
scripts/plot_comparison.py
Generates comparison graphs from run_comparison.py results.

Usage:
    python scripts/plot_comparison.py
    python scripts/plot_comparison.py --input results/logs/comparison_results.json
    python scripts/plot_comparison.py --individual   # also saves each plot separately
"""

import os
import sys
import json
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
LOGS_DIR    = os.path.join(RESULTS_DIR, 'logs')
PLOTS_DIR   = os.path.join(RESULTS_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':  '#F8F9FA',
    'axes.facecolor':    '#FFFFFF',
    'axes.grid':         True,
    'grid.color':        '#E8EAED',
    'grid.linewidth':    0.7,
    'grid.alpha':        0.8,
    'font.family':       'DejaVu Sans',
    'font.size':         10,
    'axes.titlesize':    12,
    'axes.titleweight':  'bold',
    'axes.labelsize':    10,
    'legend.fontsize':   9,
    'legend.framealpha': 0.9,
    'lines.linewidth':   2.0,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

DEFAULT_COLORS = {
    'baseline':       '#78909C',
    'qmix_only':      '#42A5F5',
    'qmix_migration': '#26A69A',
    'qlearning':      '#FFA726',
}

DEFAULT_LABELS = {
    'baseline':       'Baseline',
    'qmix_only':      'QMIX Only',
    'qmix_migration': 'QMIX + Migration',
    'qlearning':      'Q-Learning',
}


def _normalise(results):
    """
    Normalise comparison_results.json to a consistent key schema.
    Handles both:
      - New format (from run_comparison.py): total_energy_kwh, deadline_violations, estimated_cost
      - Old format (from original comparison script): energy, violations, cost
    Returns a new dict with guaranteed keys so all plot functions work regardless of source.
    """
    normalised = {}
    for key, d in results.items():
        n = dict(d)

        # energy
        if 'total_energy_kwh' not in n:
            n['total_energy_kwh'] = d.get('energy', d.get('total_energy', 0))
        if 'avg_energy_slot' not in n:
            n['avg_energy_slot'] = d.get('avg_energy_slot', d.get('avg_energy', 0))

        # cost
        if 'estimated_cost' not in n:
            n['estimated_cost'] = d.get('cost', d.get('estimated_cost_usd', 0))

        # violations
        if 'deadline_violations' not in n:
            n['deadline_violations'] = d.get('violations', d.get('total_deadline_violations', 0))
        if 'violation_rate' not in n:
            n['violation_rate'] = d.get('violation_rate', 0)

        # PMs
        if 'peak_active_pms' not in n:
            n['peak_active_pms'] = d.get('peak_pms', d.get('peak_active_pms', 0))
        if 'total_pms_created' not in n:
            n['total_pms_created'] = d.get('total_pms_created', n['peak_active_pms'])

        # migrations
        if 'total_migrations' not in n:
            n['total_migrations'] = d.get('migrations', d.get('total_migrations', 0))

        # containers
        if 'containers_arrived' not in n:
            n['containers_arrived'] = d.get('total_containers_arrived',
                                      d.get('containers_arrived', 0))
        if 'containers_finished' not in n:
            n['containers_finished'] = d.get('total_containers_finished',
                                       d.get('containers_finished', 0))

        # scenario metadata — infer from key if missing
        if 'scenario_name' not in n:
            n['scenario_name'] = DEFAULT_LABELS.get(key, key.replace('_', ' ').title())
        if 'scenario_color' not in n:
            n['scenario_color'] = DEFAULT_COLORS.get(key, '#90A4AE')

        normalised[key] = n
    return normalised


def smooth(values, window=15):
    if len(values) < window:
        return np.array(values)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode='valid')


def _color(sc_key, data):
    return data.get('scenario_color', DEFAULT_COLORS.get(sc_key, '#90A4AE'))


def _label(sc_key, data):
    return data.get('scenario_name', DEFAULT_LABELS.get(sc_key, sc_key))


# ── individual plot functions ─────────────────────────────────────────────────

def plot_energy_bar(ax, results):
    """Bar chart: total energy per scenario."""
    keys   = list(results.keys())
    vals   = [results[k]['total_energy_kwh'] for k in keys]
    colors = [_color(k, results[k]) for k in keys]
    labels = [_label(k, results[k]) for k in keys]

    bars = ax.bar(labels, vals, color=colors, alpha=0.85, edgecolor='white',
                  linewidth=1.5, width=0.55)

    # Value labels on bars
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Reduction arrows vs baseline
    if 'baseline' in results:
        b = results['baseline']['total_energy_kwh']
        for bar, val, key in zip(bars, vals, keys):
            if key != 'baseline' and b > 0:
                pct = (b - val) / b * 100
                color = '#2E7D32' if pct > 0 else '#C62828'
                symbol = '▼' if pct > 0 else '▲'
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() * 0.5,
                        f'{symbol}{abs(pct):.1f}%',
                        ha='center', va='center', fontsize=8,
                        color='white', fontweight='bold')

    ax.set_title('Total Energy Consumption')
    ax.set_ylabel('Energy (kWh)')
    ax.set_ylim(0, max(vals) * 1.2)
    ax.tick_params(axis='x', labelsize=8)


def plot_violation_bar(ax, results):
    """Bar chart: deadline violations per scenario."""
    keys   = list(results.keys())
    vals   = [results[k]['deadline_violations'] for k in keys]
    rates  = [results[k]['violation_rate'] * 100 for k in keys]
    colors = [_color(k, results[k]) for k in keys]
    labels = [_label(k, results[k]) for k in keys]

    bars = ax.bar(labels, vals, color=colors, alpha=0.85, edgecolor='white',
                  linewidth=1.5, width=0.55)

    for bar, val, rate in zip(bars, vals, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                f'{val:.0f}\n({rate:.1f}%)',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_title('Deadline Violations')
    ax.set_ylabel('Total Violations')
    ax.set_ylim(0, max(vals) * 1.25 if max(vals) > 0 else 10)
    ax.tick_params(axis='x', labelsize=8)


def plot_cost_bar(ax, results):
    """Bar chart: estimated cost per scenario."""
    keys   = list(results.keys())
    vals   = [results[k]['estimated_cost'] for k in keys]
    colors = [_color(k, results[k]) for k in keys]
    labels = [_label(k, results[k]) for k in keys]

    bars = ax.bar(labels, vals, color=colors, alpha=0.85, edgecolor='white',
                  linewidth=1.5, width=0.55)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                f'${val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    if 'baseline' in results:
        b = results['baseline']['estimated_cost']
        for bar, val, key in zip(bars, vals, keys):
            if key != 'baseline' and b > 0:
                pct = (b - val) / b * 100
                color = '#2E7D32' if pct > 0 else '#C62828'
                symbol = '▼' if pct > 0 else '▲'
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() * 0.5,
                        f'{symbol}{abs(pct):.1f}%',
                        ha='center', va='center', fontsize=8,
                        color='white', fontweight='bold')

    ax.set_title('Estimated Energy Cost')
    ax.set_ylabel('Cost (USD)')
    ax.set_ylim(0, max(vals) * 1.2)
    ax.tick_params(axis='x', labelsize=8)


def plot_pm_bar(ax, results):
    """Grouped bar: peak PMs used vs total PMs created."""
    keys   = list(results.keys())
    labels = [_label(k, results[k]) for k in keys]
    peak   = [results[k]['peak_active_pms']   for k in keys]
    total  = [results[k].get('total_pms_created', results[k]['peak_active_pms']) for k in keys]
    colors = [_color(k, results[k]) for k in keys]

    x = np.arange(len(keys))
    w = 0.35
    bars1 = ax.bar(x - w/2, peak,  width=w, label='Peak Active',  color=colors, alpha=0.85, edgecolor='white')
    bars2 = ax.bar(x + w/2, total, width=w, label='Total Created', color=colors, alpha=0.45, edgecolor='white', hatch='//')

    for bar, val in zip(list(bars1)+list(bars2), peak+total):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_title('Physical Machine Usage')
    ax.set_ylabel('PM Count')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend(fontsize=8)


def plot_energy_over_time(ax, results):
    """Line chart: energy per slot over time for each scenario."""
    has_data = False
    for key, data in results.items():
        history = data.get('avg_energy_history', [])
        if not history:
            continue
        has_data = True
        slots = np.arange(len(history))
        color = _color(key, data)
        label = _label(key, data)
        ax.plot(slots, history, color=color, alpha=0.25, linewidth=1)
        if len(history) >= 15:
            sm = smooth(history, 15)
            ax.plot(slots[14:], sm, color=color, linewidth=2.5, label=label)
        else:
            ax.plot(slots, history, color=color, linewidth=2.5, label=label)

    if not has_data:
        ax.text(0.5, 0.5,
                'Run  python scripts/run_comparison.py\nto generate per-slot history data',
                ha='center', va='center', transform=ax.transAxes,
                color='#90A4AE', fontsize=10, linespacing=1.8,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F5F5', edgecolor='#E0E0E0'))
    ax.set_title('Energy Consumption Over Time')
    ax.set_xlabel('Time Slot')
    ax.set_ylabel('Energy (kWh)')
    if has_data:
        ax.legend(fontsize=8)


def plot_violations_over_time(ax, results):
    """Line chart: violations per slot over time."""
    has_data = False
    for key, data in results.items():
        history = data.get('avg_violations_per_slot', [])
        if not history:
            continue
        has_data = True
        slots = np.arange(len(history))
        color = _color(key, data)
        label = _label(key, data)
        ax.bar(slots, history, color=color, alpha=0.2, width=1.0)
        if len(history) >= 15:
            sm = smooth(history, 15)
            ax.plot(slots[14:], sm, color=color, linewidth=2.5, label=label)
        else:
            ax.plot(slots, history, color=color, linewidth=2.5, label=label)

    if not has_data:
        ax.text(0.5, 0.5,
                'Run  python scripts/run_comparison.py\nto generate per-slot history data',
                ha='center', va='center', transform=ax.transAxes,
                color='#90A4AE', fontsize=10, linespacing=1.8,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F5F5', edgecolor='#E0E0E0'))
    ax.set_title('Deadline Violations Per Slot')
    ax.set_xlabel('Time Slot')
    ax.set_ylabel('Violations')
    if has_data:
        ax.legend(fontsize=8)


def plot_reward_over_time(ax, results):
    """Line chart: reward over time for RL scenarios."""
    has_data = False
    for key, data in results.items():
        rewards = data.get('avg_rewards', [])
        if not rewards:
            continue
        has_data = True
        slots = np.arange(len(rewards))
        color = _color(key, data)
        label = _label(key, data)

        ax.plot(slots, rewards, color=color, alpha=0.2, linewidth=1)
        if len(rewards) >= 15:
            sm = smooth(rewards, 15)
            ax.plot(slots[14:], sm, color=color, linewidth=2.5, label=label)

    if not has_data:
        ax.text(0.5, 0.5, 'No RL scenarios in results\n(Baseline has no reward)',
                ha='center', va='center', transform=ax.transAxes,
                color='#90A4AE', fontsize=10)
    ax.set_title('RL Reward Over Time (RL scenarios only)')
    ax.set_xlabel('Time Slot')
    ax.set_ylabel('Total Reward')
    if has_data:
        ax.legend(fontsize=8)


def plot_summary_table(ax, results):
    """Formatted summary table as the final panel."""
    ax.axis('off')

    keys = list(results.keys())
    b    = results.get('baseline', {})
    b_e  = b.get('total_energy_kwh', 1)
    b_v  = b.get('deadline_violations', 1)
    b_c  = b.get('estimated_cost', 1)

    col_labels = ['Scenario', 'Energy\n(kWh)', 'vs Base', 'Violations', 'vs Base',
                  'Cost ($)', 'vs Base', 'Peak PMs']
    rows = []
    for key in keys:
        d = results[key]
        e = d['total_energy_kwh']
        v = d['deadline_violations']
        c = d['estimated_cost']
        p = d['peak_active_pms']
        e_r = f"{'▼' if e<b_e else '▲'}{abs((b_e-e)/max(b_e,1e-9)*100):.1f}%" if key!='baseline' else '—'
        v_r = f"{'▼' if v<b_v else '▲'}{abs((b_v-v)/max(b_v,1e-9)*100):.1f}%" if key!='baseline' else '—'
        c_r = f"{'▼' if c<b_c else '▲'}{abs((b_c-c)/max(b_c,1e-9)*100):.1f}%" if key!='baseline' else '—'
        rows.append([_label(key,d).replace('\n',' '),
                     f'{e:.1f}', e_r, f'{v:.0f}', v_r,
                     f'${c:.2f}', c_r, f'{p:.0f}'])

    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)

    # Header style
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor('#1F4E79')
        tbl[0, j].set_text_props(color='white', fontweight='bold')

    # Row styles
    for i, key in enumerate(keys, 1):
        color = _color(key, results[key])
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor(color + '22')  # hex transparency
        # highlight improvement cells
        for j_idx, col_idx in [(2,2),(4,4),(6,6)]:
            txt = tbl[i, j_idx].get_text().get_text()
            if '▼' in txt:
                tbl[i, j_idx].set_facecolor('#E8F5E9')
                tbl[i, j_idx].get_text().set_color('#2E7D32')
            elif '▲' in txt:
                tbl[i, j_idx].set_facecolor('#FFEBEE')
                tbl[i, j_idx].get_text().set_color('#C62828')

    ax.set_title('Summary Comparison Table', fontsize=12, fontweight='bold', pad=15)


# ── dashboard ─────────────────────────────────────────────────────────────────

def plot_all(input_path=None, individual=False):
    input_path = input_path or os.path.join(LOGS_DIR, 'comparison_results.json')

    if not os.path.exists(input_path):
        print(f"❌ No comparison results at {input_path}")
        print("   Run first: python scripts/run_comparison.py")
        return

    with open(input_path) as f:
        results = json.load(f)

    # Normalise key schema — handles both old and new comparison JSON formats
    results = _normalise(results)

    print(f"📊 Plotting {len(results)} scenarios from {input_path}")

    # ── 4×2 dashboard ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 22))
    fig.patch.set_facecolor('#F8F9FA')

    slots = list(results.values())[0].get('slots', '?')
    runs  = list(results.values())[0].get('num_runs', '?')
    fig.suptitle(
        f'QMIX Multi-Agent vs Baseline — Performance Comparison\n'
        f'{len(results)} scenarios | {slots} slots | Averaged over {runs} runs',
        fontsize=14, fontweight='bold', y=0.99
    )

    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.48, wspace=0.32)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    # ax5 = fig.add_subplot(gs[2, 0])
    # ax6 = fig.add_subplot(gs[2, 1])
    # ax7 = fig.add_subplot(gs[3, 0])
    # ax8 = fig.add_subplot(gs[3, 1])

    plot_energy_bar(ax1, results)
    plot_violation_bar(ax2, results)
    plot_cost_bar(ax3, results)
    # plot_pm_bar(ax4, results)
    plot_energy_over_time(ax4, results)
    # plot_violations_over_time(ax6, results)
    # plot_reward_over_time(ax7, results)
    # plot_summary_table(ax8, results)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out      = os.path.join(PLOTS_DIR, f'comparison_{ts}.png')
    latest   = os.path.join(PLOTS_DIR, 'comparison_latest.png')

    for path in [out, latest]:
        fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Dashboard saved: {out}")
    print(f"   Latest copy:    {latest}")

    if individual:
        _save_individual(results, ts)

    return out


def _save_individual(results, ts):
    plots = [
        ('energy_bar',            lambda ax: plot_energy_bar(ax, results)),
        ('violation_bar',         lambda ax: plot_violation_bar(ax, results)),
        ('cost_bar',              lambda ax: plot_cost_bar(ax, results)),
        ('pm_usage',              lambda ax: plot_pm_bar(ax, results)),
        ('energy_over_time',      lambda ax: plot_energy_over_time(ax, results)),
        ('violations_over_time',  lambda ax: plot_violations_over_time(ax, results)),
        ('reward_over_time',      lambda ax: plot_reward_over_time(ax, results)),
        ('summary_table',         lambda ax: plot_summary_table(ax, results)),
    ]
    print("\nSaving individual plots...")
    for name, fn in plots:
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#F8F9FA')
        fn(ax)
        path = os.path.join(PLOTS_DIR, f'{name}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ {name}.png")
    print(f"✅ Individual plots saved to {PLOTS_DIR}/")


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot QMIX comparison results")
    parser.add_argument('--input',      default=None, help='Path to comparison_results.json')
    parser.add_argument('--individual', action='store_true', help='Also save each plot separately')
    args = parser.parse_args()

    print("\n" + "📈"*20)
    print("QMIX COMPARISON PLOTTER")
    print("📈"*20 + "\n")
    plot_all(args.input, args.individual)