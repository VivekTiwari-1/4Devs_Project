"""
QMIX Simulation Plotting Script
Generates all performance graphs from saved simulation results.
Run this after main.py completes:  python scripts/plot_qmix_results.py
"""

import os
import sys
import json
import csv
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend (works without display)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

# ── paths ────────────────────────────────────────────────────────────────────
RESULTS_DIR   = "results"
SIM_DIR       = os.path.join(RESULTS_DIR, "simulation_outputs")
PLOTS_DIR     = os.path.join(RESULTS_DIR, "plots")
QTABLES_DIR   = os.path.join(RESULTS_DIR, "qtables")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── style ─────────────────────────────────────────────────────────────────────
COLORS = {
    'primary':    '#2196F3',
    'success':    '#4CAF50',
    'danger':     '#F44336',
    'warning':    '#FF9800',
    'purple':     '#9C27B0',
    'teal':       '#009688',
    'dark':       '#212121',
    'grid':       '#E0E0E0',
    'bg':         '#FAFAFA',
}

plt.rcParams.update({
    'figure.facecolor':  COLORS['bg'],
    'axes.facecolor':    COLORS['bg'],
    'axes.grid':         True,
    'grid.color':        COLORS['grid'],
    'grid.linewidth':    0.6,
    'font.family':       'DejaVu Sans',
    'font.size':         10,
    'axes.titlesize':    12,
    'axes.labelsize':    10,
    'legend.fontsize':   9,
    'lines.linewidth':   1.8,
})


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(filepath):
    """Load metrics CSV into a dict of lists."""
    data = {}
    with open(filepath, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                data.setdefault(k, [])
                try:
                    data[k].append(float(v))
                except ValueError:
                    data[k].append(v)
    return data


def load_summary(filepath):
    with open(filepath) as f:
        return json.load(f)


def smooth(values, window=10):
    """Moving average smoothing."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode='valid')


def slots_axis(data):
    return np.array(data['slot'])


# ─────────────────────────────────────────────────────────────────────────────
# Individual plot functions
# ─────────────────────────────────────────────────────────────────────────────

def plot_reward_trend(ax, data, summary):
    """Plot 1 — RL reward over time with smoothed trend."""
    slots   = slots_axis(data)
    rewards = np.array(data['rl_reward'])

    ax.plot(slots, rewards, color=COLORS['primary'], alpha=0.3, linewidth=1, label='Raw reward')

    if len(rewards) >= 10:
        sm = smooth(rewards, window=20)
        ax.plot(slots[19:], sm, color=COLORS['primary'], linewidth=2.5, label='Smoothed (w=20)')

    # Early vs late annotation
    early = summary.get('early_avg_reward', 0)
    late  = summary.get('late_avg_reward', 0)
    improved = late > early
    color = COLORS['success'] if improved else COLORS['danger']
    trend_label = f"{'📈' if improved else '📉'} Early: {early:.1f}  →  Late: {late:.1f}"
    ax.set_title(f"QMIX Reward Over Time\n{trend_label}", color=color)
    ax.set_xlabel("Time Slot")
    ax.set_ylabel("Total Reward")
    ax.legend()


def plot_epsilon_decay(ax, data):
    """Plot 2 — Epsilon (exploration rate) decay."""
    slots   = slots_axis(data)
    epsilon = np.array(data['rl_epsilon'])

    ax.plot(slots, epsilon, color=COLORS['warning'], linewidth=2)
    ax.fill_between(slots, epsilon, alpha=0.15, color=COLORS['warning'])
    ax.axhline(y=epsilon[-1], linestyle='--', color=COLORS['danger'],
               linewidth=1, label=f"Final ε = {epsilon[-1]:.3f}")

    # Mark 50% exploitation point
    half_idx = np.argmax(epsilon <= 0.5)
    if half_idx > 0:
        ax.axvline(x=slots[half_idx], linestyle=':', color=COLORS['dark'],
                   linewidth=1, label=f"50% exploit @ slot {int(slots[half_idx])}")

    ax.set_title("Epsilon (Exploration Rate) Decay")
    ax.set_xlabel("Time Slot")
    ax.set_ylabel("Epsilon")
    ax.set_ylim(0, 1)
    ax.legend()


def plot_violations(ax, data, summary):
    """Plot 3 — Deadline violations per slot."""
    slots      = slots_axis(data)
    slot_viol  = np.array(data.get('slot_violations', [0]*len(slots)))

    # Bar chart
    colors = [COLORS['danger'] if v > 0 else COLORS['success'] for v in slot_viol]
    ax.bar(slots, slot_viol, color=colors, alpha=0.7, width=1.0)

    if len(slot_viol) >= 10:
        sm = smooth(slot_viol, window=20)
        ax.plot(slots[19:], sm, color=COLORS['dark'], linewidth=2, label='Trend (w=20)')

    vrate = summary.get('violation_rate', 0) * 100
    ax.set_title(f"Deadline Violations Per Slot\nOverall violation rate: {vrate:.1f}%")
    ax.set_xlabel("Time Slot")
    ax.set_ylabel("Violations")
    ax.legend()


def plot_energy(ax, data):
    """Plot 4 — Energy consumption per slot."""
    slots  = slots_axis(data)
    energy = np.array(data['energy_kwh'])

    ax.plot(slots, energy, color=COLORS['warning'], alpha=0.4, linewidth=1)
    if len(energy) >= 10:
        sm = smooth(energy, window=20)
        ax.plot(slots[19:], sm, color=COLORS['warning'], linewidth=2.5, label='Smoothed')

    ax.fill_between(slots, energy, alpha=0.1, color=COLORS['warning'])
    total = np.sum(energy)
    ax.set_title(f"Energy Consumption Per Slot\nTotal: {total:.1f} kWh")
    ax.set_xlabel("Time Slot")
    ax.set_ylabel("Energy (kWh)")
    ax.legend()


def plot_containers(ax, data):
    """Plot 5 — Running containers and finished containers."""
    slots     = slots_axis(data)
    running   = np.array(data['total_containers'])
    finished  = np.array(data['containers_finished'])

    ax.plot(slots, running,  color=COLORS['primary'], label='Running containers')
    ax.plot(slots, finished, color=COLORS['success'], label='Total finished (cumulative)')
    ax.set_title("Container Population Over Time")
    ax.set_xlabel("Time Slot")
    ax.set_ylabel("Container Count")
    ax.legend()


def plot_cpu_util(ax, data):
    """Plot 6 — Average CPU utilization per slot."""
    slots = slots_axis(data)
    util  = np.array(data['avg_cpu_util'])

    ax.plot(slots, util * 100, color=COLORS['teal'], alpha=0.4, linewidth=1)
    if len(util) >= 10:
        sm = smooth(util * 100, window=20)
        ax.plot(slots[19:], sm, color=COLORS['teal'], linewidth=2.5, label='Smoothed')

    ax.axhline(y=70, linestyle='--', color=COLORS['danger'],
               linewidth=1, alpha=0.7, label='70% threshold')
    ax.axhline(y=30, linestyle='--', color=COLORS['warning'],
               linewidth=1, alpha=0.7, label='30% threshold')
    ax.fill_between(slots, util * 100, alpha=0.1, color=COLORS['teal'])
    ax.set_title("Average CPU Utilization Per Slot")
    ax.set_xlabel("Time Slot")
    ax.set_ylabel("CPU Utilization (%)")
    ax.set_ylim(0, 110)
    ax.legend()


def plot_active_pms(ax, data, summary):
    """Plot 7 — Active PM count over time."""
    slots      = slots_axis(data)
    active_pms = np.array(data['active_pms'])

    ax.step(slots, active_pms, color=COLORS['purple'], linewidth=2, where='post')
    ax.fill_between(slots, active_pms, alpha=0.15, color=COLORS['purple'], step='post')
    peak = summary.get('peak_pms', 0)
    total = summary.get('total_pms_created', 0)
    ax.set_title(f"Active Physical Machines\nPeak: {peak} | Total created: {total}")
    ax.set_xlabel("Time Slot")
    ax.set_ylabel("Active PMs")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))


def plot_migrations(ax, data):
    """Plot 8 — Migrations per slot."""
    slots      = slots_axis(data)
    migrations = np.array(data.get('migrations', [0]*len(slots)))

    ax.bar(slots, migrations, color=COLORS['purple'], alpha=0.6, width=1.0, label='Migrations')
    if len(migrations) >= 10:
        sm = smooth(migrations, window=20)
        ax.plot(slots[19:], sm, color=COLORS['dark'], linewidth=2, label='Trend (w=20)')

    total = int(np.sum(migrations))
    ax.set_title(f"Container Migrations Per Slot\nTotal: {total}")
    ax.set_xlabel("Time Slot")
    ax.set_ylabel("Migrations")
    ax.legend()


def plot_q_table_growth(ax, qtable_path):
    """Plot 9 — Q-table size loaded from pkl (if reward_history available)."""
    if not os.path.exists(qtable_path):
        ax.text(0.5, 0.5, 'No Q-table file found\nRun simulation first',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Q-Table Growth")
        return

    with open(qtable_path, 'rb') as f:
        payload = pickle.load(f)

    q_tables = payload.get('agent_q_tables', [])
    sizes = [len(qt) for qt in q_tables]

    agents = [f"Agent {i+1}" for i in range(len(sizes))]
    bars = ax.bar(agents, sizes, color=[COLORS['primary'], COLORS['teal'], COLORS['purple']])

    for bar, size in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(size), ha='center', va='bottom', fontweight='bold')

    total = sum(sizes)
    ax.set_title(f"Q-Table Entries Per Agent\nTotal: {total} entries")
    ax.set_xlabel("Agent")
    ax.set_ylabel("Q-Table Entries")


def plot_kpi_summary(ax, summary):
    """Plot 10 — KPI summary table."""
    ax.axis('off')

    vrate   = summary.get('violation_rate', 0) * 100
    rrate   = summary.get('rejection_rate', 0) * 100
    energy  = summary.get('total_energy_kwh', 0)
    cost    = energy * 0.12
    epsilon = summary.get('final_epsilon', 0)
    improved = summary.get('reward_improved', False)
    migrations = summary.get('total_migrations', 0)
    q_entries  = summary.get('total_q_entries', 0)

    rows = [
        ["Metric",                    "Value",                    "Status"],
        ["Deadline compliance",        f"{100-vrate:.1f}%",       "✅" if vrate < 10 else "⚠️" if vrate < 25 else "❌"],
        ["Rejection rate",             f"{rrate:.1f}%",           "✅" if rrate == 0 else "⚠️"],
        ["Total energy",               f"{energy:.1f} kWh",       "ℹ️"],
        ["Estimated cost",             f"${cost:.2f}",            "ℹ️"],
        ["Final epsilon",              f"{epsilon:.3f}",          "✅" if epsilon <= 0.1 else "⚠️"],
        ["Reward trend",               "Improving" if improved else "Not improving", "✅" if improved else "❌"],
        ["Total migrations",           str(migrations),           "✅" if migrations < 100 else "⚠️" if migrations < 300 else "❌"],
        ["Q-table entries",            str(q_entries),            "✅" if q_entries > 100 else "⚠️"],
        ["Containers arrived",         str(int(summary.get('containers_arrived', 0))),  "ℹ️"],
        ["Containers finished",        str(int(summary.get('containers_finished', 0))), "ℹ️"],
    ]

    table = ax.table(
        cellText=rows[1:],
        colLabels=rows[0],
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    # Style header
    for j in range(3):
        table[0, j].set_facecolor(COLORS['dark'])
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Style data rows
    for i in range(1, len(rows)):
        status = rows[i][2]
        bg = '#E8F5E9' if '✅' in status else '#FFF3E0' if '⚠️' in status else '#FFEBEE' if '❌' in status else COLORS['bg']
        for j in range(3):
            table[i, j].set_facecolor(bg)

    ax.set_title("📊 KPI Summary", fontsize=13, fontweight='bold', pad=15)


# ─────────────────────────────────────────────────────────────────────────────
# Main dashboard
# ─────────────────────────────────────────────────────────────────────────────

def generate_dashboard(csv_path=None, summary_path=None, qtable_path=None):
    """
    Generate the full 10-panel dashboard.

    Args:
        csv_path: path to metrics CSV (defaults to latest)
        summary_path: path to summary JSON (defaults to latest)
        qtable_path: path to qtable pkl (defaults to latest)
    """
    csv_path     = csv_path     or os.path.join(SIM_DIR,    "metrics_latest.csv")
    summary_path = summary_path or os.path.join(SIM_DIR,    "summary_latest.json")
    qtable_path  = qtable_path  or os.path.join(QTABLES_DIR,"qmix_qtable_latest.pkl")

    if not os.path.exists(csv_path):
        print(f"❌ No CSV found at {csv_path}. Run main.py first.")
        return
    if not os.path.exists(summary_path):
        print(f"❌ No summary JSON found at {summary_path}. Run main.py first.")
        return

    print(f"📊 Loading data from {csv_path}")
    data    = load_csv(csv_path)
    summary = load_summary(summary_path)

    # ── Layout: 5 rows × 2 cols ───────────────────────────────────────────
    fig = plt.figure(figsize=(18, 28))
    fig.suptitle(
        f"QMIX Multi-Agent Simulation Dashboard\n"
        f"Run: {summary.get('timestamp','N/A')} | "
        f"Slots: {summary.get('slots','N/A')} | "
        f"Violation rate: {summary.get('violation_rate',0)*100:.1f}%",
        fontsize=14, fontweight='bold', y=0.99
    )

    gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.45, wspace=0.3)

    ax1  = fig.add_subplot(gs[0, 0])
    ax2  = fig.add_subplot(gs[0, 1])
    ax3  = fig.add_subplot(gs[1, 0])
    ax4  = fig.add_subplot(gs[1, 1])
    ax5  = fig.add_subplot(gs[2, 0])
    ax6  = fig.add_subplot(gs[2, 1])
    ax7  = fig.add_subplot(gs[3, 0])
    ax8  = fig.add_subplot(gs[3, 1])
    ax9  = fig.add_subplot(gs[4, 0])
    ax10 = fig.add_subplot(gs[4, 1])

    plot_reward_trend(ax1,  data, summary)
    plot_epsilon_decay(ax2, data)
    plot_violations(ax3,    data, summary)
    plot_energy(ax4,        data)
    plot_containers(ax5,    data)
    plot_cpu_util(ax6,      data)
    plot_active_pms(ax7,    data, summary)
    plot_migrations(ax8,    data)
    plot_q_table_growth(ax9, qtable_path)
    plot_kpi_summary(ax10,  summary)

    # Save
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path    = os.path.join(PLOTS_DIR, f"qmix_dashboard_{ts}.png")
    latest_path = os.path.join(PLOTS_DIR, "qmix_dashboard_latest.png")

    for path in [out_path, latest_path]:
        fig.savefig(path, dpi=150, bbox_inches='tight')

    plt.close(fig)
    print(f"✅ Dashboard saved:")
    print(f"   {out_path}")
    print(f"   {latest_path}")
    return out_path


def generate_individual_plots(csv_path=None, summary_path=None, qtable_path=None):
    """Generate each plot as a separate file (useful for reports)."""
    csv_path     = csv_path     or os.path.join(SIM_DIR,    "metrics_latest.csv")
    summary_path = summary_path or os.path.join(SIM_DIR,    "summary_latest.json")
    qtable_path  = qtable_path  or os.path.join(QTABLES_DIR,"qmix_qtable_latest.pkl")

    if not os.path.exists(csv_path):
        print(f"❌ No CSV found. Run main.py first.")
        return

    data    = load_csv(csv_path)
    summary = load_summary(summary_path)

    plots = [
        ("reward_trend",   lambda ax: plot_reward_trend(ax, data, summary)),
        ("epsilon_decay",  lambda ax: plot_epsilon_decay(ax, data)),
        ("violations",     lambda ax: plot_violations(ax, data, summary)),
        ("energy",         lambda ax: plot_energy(ax, data)),
        ("containers",     lambda ax: plot_containers(ax, data)),
        ("cpu_util",       lambda ax: plot_cpu_util(ax, data)),
        ("active_pms",     lambda ax: plot_active_pms(ax, data, summary)),
        ("migrations",     lambda ax: plot_migrations(ax, data)),
        ("q_table_growth", lambda ax: plot_q_table_growth(ax, qtable_path)),
        ("kpi_summary",    lambda ax: plot_kpi_summary(ax, summary)),
    ]

    for name, fn in plots:
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor(COLORS['bg'])
        fn(ax)
        path = os.path.join(PLOTS_DIR, f"{name}.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ {name}.png")

    print(f"✅ Individual plots saved to {PLOTS_DIR}/")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot QMIX simulation results")
    parser.add_argument('--individual', action='store_true',
                        help='Also save each plot as a separate PNG')
    parser.add_argument('--csv',     default=None, help='Path to metrics CSV')
    parser.add_argument('--summary', default=None, help='Path to summary JSON')
    parser.add_argument('--qtable',  default=None, help='Path to qtable pkl')
    args = parser.parse_args()

    print("\n" + "📈" * 20)
    print("QMIX RESULTS PLOTTER")
    print("📈" * 20 + "\n")

    generate_dashboard(args.csv, args.summary, args.qtable)

    if args.individual:
        print("\nGenerating individual plots...")
        generate_individual_plots(args.csv, args.summary, args.qtable)