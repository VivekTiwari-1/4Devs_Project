"""
Configuration file for Energy-Efficient Container Cloud Simulator - QMIX Multi-Agent Upgrade
"""

# ============================================
# SIMULATION SETTINGS
# ============================================
TIME_SLOT_DURATION = 30  # seconds per time slot (tau)
TOTAL_SIMULATION_SLOTS = 500  # number of time slots to simulate
RANDOM_SEED = 42  # for reproducibility

# ============================================
# PHYSICAL MACHINE (PM) SPECIFICATIONS
# ============================================
PM_TOTAL_CORES = 68  # total CPU cores per PM
PM_MEMORY_GB = 256  # total memory in GB
PM_NETWORK_GBPS = 10  # network bandwidth in Gbps

# Core processing speed
CORE_SPEED = 5_000_000  # 5M instructions per second per core

# ============================================
# CONTAINER SPECIFICATIONS
# ============================================
CONTAINER_MIN_INSTRUCTIONS = 500_000_000    # 500M
CONTAINER_MAX_INSTRUCTIONS = 2_000_000_000  # 2B

CONTAINER_MIN_DEADLINE_OFFSET = 300   # (5 min) 
CONTAINER_MAX_DEADLINE_OFFSET = 900  #(15 min)

DEFAULT_CORES_PER_CONTAINER = 1

# ============================================
# WORKLOAD GENERATOR SETTINGS
# ============================================
MIN_CONTAINERS_PER_SLOT = 0
MAX_CONTAINERS_PER_SLOT = 8   # FIX: was 15
                               # At 8M instr/sec, 1 core, avg container = 1.25B instr
                               # completes in: 1.25B / (8M × 30) = 5.2 slots
                               # 8 arrivals/slot × ~5 slots each = ~40 active per PM — manageable
                               # Old 15/slot caused 460+ containers to pile up — impossible to finish

# ============================================
# PLACEMENT MODULE SETTINGS
# ============================================
PLACEMENT_CPU_THRESHOLD = 0.75   # FIX: was 0.80 — starts new PM sooner, fewer containers per PM
MAX_CONTAINERS_PER_PM   = 50     # FIX: NEW — hard cap, prevents PM overload regardless of CPU util
PLACEMENT_STRATEGY = "first_fit"

# ============================================
# ENERGY MODEL PARAMETERS
# ============================================
ENERGY_ALPHA_1 = 40
ENERGY_ALPHA_2 = 80
ENERGY_BETA_MEM = 10
ENERGY_GAMMA_NET = 5
ENERGY_STATIC_POWER = 100  # watts when PM is ON but idle

# ============================================
# MIGRATION SETTINGS
# ============================================
MIGRATION_UNDERLOAD_THRESHOLD = 0.15  # FIX: lowered from 0.20 — fewer spurious migrations
MIGRATION_OVERLOAD_THRESHOLD  = 0.85  # FIX: raised from 0.70 — less aggressive overload trigger
MIGRATION_COST_INSTRUCTIONS   = 1_000_000

# ============================================
# QMIX MULTI-AGENT RL SETTINGS
# ============================================
RL_LEARNING_RATE    = 0.1
RL_DISCOUNT_FACTOR  = 0.95
RL_EPSILON_START    = 0.9
RL_EPSILON_MIN      = 0.05   # FIX: lowered from 0.1 — allow more exploitation at end
RL_EPSILON_DECAY    = 0.99   # FIX: was 0.995 — faster decay so agent exploits sooner
                              #      500 slots: 0.9 * (0.99^500) = 0.003 → clamps to 0.05 ✓
                              #      Agent reaches 0.5 epsilon by slot ~62, 0.1 by slot ~204

# Reward shaping
# FIX: penalty reduced from 1000 → 10
# Old: reward ranged from -0.83 to -4001 (4800x range — TD errors explode)
# New: reward ranges from -0.83 to   -21 (25x range  — numerically stable)
RL_DEADLINE_VIOLATION_PENALTY = 10
RL_REJECTION_PENALTY          = 5   # FIX: scaled down from 500 proportionally

# QMIX-specific settings
NUM_QMIX_AGENTS          = 3
QMIX_REPLAY_BUFFER_SIZE  = 10000
QMIX_BATCH_SIZE          = 16    # FIX: reduced from 32 — more frequent diverse updates
QMIX_UPDATE_TARGET_EVERY = 10    # FIX: reduced from 100 — target net actually updates now

# RL action space
RL_POLICIES = ["fair", "deadline_priority", "smallest_work", "conservative"]

# ============================================
# LOGGING AND OUTPUT SETTINGS
# ============================================
LOG_LEVEL = "INFO"
SAVE_LOGS = True
LOG_FILE_PATH = "results/logs/simulation.log"

TRACK_METRICS = [
    "total_energy",
    "deadline_violations",
    "container_rejections",
    "num_migrations",
    "avg_pm_utilization",
    "num_active_pms"
]

# ============================================
# PATHS
# ============================================
RESULTS_DIR             = "results/"
LOGS_DIR                = "results/logs/"
PLOTS_DIR               = "results/plots/"
QTABLES_DIR             = "results/qtables/"
SIMULATION_OUTPUTS_DIR  = "results/simulation_outputs/"

# ============================================
# TESTING AND DEBUGGING
# ============================================
DEBUG_MODE                  = False
VALIDATE_STATE_TRANSITIONS  = True
PRINT_EVERY_N_SLOTS         = 50   # FIX: changed from 10 — less noise for 500-slot runs

# ============================================
# FEATURE FLAGS
# ============================================
ENABLE_RL_SCALING       = True
ENABLE_MIGRATION        = True
ENABLE_ENERGY_MODEL     = True
ENABLE_DELAYED_PLACEMENT = False