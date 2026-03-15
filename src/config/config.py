"""
Configuration file for Energy-Efficient Container Cloud Simulator
8th Semester - QMIX Multi-Agent Upgrade
"""

# ============================================
# SIMULATION SETTINGS
# ============================================
TIME_SLOT_DURATION = 30  # seconds per time slot (tau)
TOTAL_SIMULATION_SLOTS = 50  # number of time slots to simulate
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
# Range for random container workload (instructions)
CONTAINER_MIN_INSTRUCTIONS = 500_000_000    # 500M
CONTAINER_MAX_INSTRUCTIONS = 2_000_000_000  # 2B

# Range for random deadline offset (seconds from arrival)
CONTAINER_MIN_DEADLINE_OFFSET = 300  # 5 minutes
CONTAINER_MAX_DEADLINE_OFFSET = 900  # 15 minutes

# Default resource allocation
DEFAULT_CORES_PER_CONTAINER = 1

# ============================================
# WORKLOAD GENERATOR SETTINGS
# ============================================
MIN_CONTAINERS_PER_SLOT = 0  # minimum new containers per time slot
MAX_CONTAINERS_PER_SLOT = 5  # maximum new containers per time slot

# ============================================
# PLACEMENT MODULE SETTINGS
# ============================================
PLACEMENT_CPU_THRESHOLD = 0.80  # 80% - don't place if PM is above this
PLACEMENT_STRATEGY = "first_fit"  # Options: first_fit, best_fit, worst_fit

# ============================================
# ENERGY MODEL PARAMETERS (Step 2)
# ============================================
# Quadratic CPU energy model: P_cpu = ALPHA_1 * util + ALPHA_2 * util^2
ENERGY_ALPHA_1 = 40  # linear coefficient (watts)
ENERGY_ALPHA_2 = 80  # quadratic coefficient (watts)

# Memory and network energy (linear models)
ENERGY_BETA_MEM = 10  # watts per unit memory utilization
ENERGY_GAMMA_NET = 5  # watts per unit network utilization

# Static power consumption
ENERGY_STATIC_POWER = 100  # watts when PM is ON but idle

# ============================================
# MIGRATION SETTINGS (Step 4)
# ============================================
MIGRATION_UNDERLOAD_THRESHOLD = 0.30  # 30% - migrate out if below
MIGRATION_OVERLOAD_THRESHOLD = 0.90  # 90% - migrate out if above
MIGRATION_COST_INSTRUCTIONS = 1_000_000  # penalty for migration

# ============================================
# QMIX MULTI-AGENT RL SETTINGS (8TH SEMESTER)
# ============================================
# Basic RL parameters
RL_LEARNING_RATE = 0.1  # alpha
RL_DISCOUNT_FACTOR = 0.95  # gamma
RL_EPSILON_START = 0.9  # initial exploration rate
RL_EPSILON_MIN = 0.1  # minimum exploration rate
RL_EPSILON_DECAY = 0.995  # decay rate per episode

# Reward shaping
RL_DEADLINE_VIOLATION_PENALTY = 1000  # large penalty for missing deadlines
RL_REJECTION_PENALTY = 500  # penalty for rejecting containers

# QMIX-specific settings
NUM_QMIX_AGENTS = 3  # Number of agents (one per PM typically)
QMIX_REPLAY_BUFFER_SIZE = 10000  # Experience replay capacity
QMIX_BATCH_SIZE = 32  # Training batch size
QMIX_UPDATE_TARGET_EVERY = 100  # Update target network frequency

# RL action space (allocation policies)
RL_POLICIES = ["fair", "deadline_priority", "smallest_work", "conservative"]

# ============================================
# LOGGING AND OUTPUT SETTINGS
# ============================================
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
SAVE_LOGS = True
LOG_FILE_PATH = "results/logs/simulation.log"

# Metrics to track
TRACK_METRICS = [
    "total_energy",
    "deadline_violations",
    "container_rejections",
    "num_migrations",
    "avg_pm_utilization",
    "num_active_pms"
]

# ============================================
# PATHS (aligned with project structure)
# ============================================
RESULTS_DIR = "results/"
LOGS_DIR = "results/logs/"
PLOTS_DIR = "results/plots/"
QTABLES_DIR = "results/qtables/"
SIMULATION_OUTPUTS_DIR = "results/simulation_outputs/"

# ============================================
# TESTING AND DEBUGGING
# ============================================
DEBUG_MODE = False  # Set True for verbose output
VALIDATE_STATE_TRANSITIONS = True  # Check for inconsistencies
PRINT_EVERY_N_SLOTS = 10  # print summary every N slots

# ============================================
# FEATURE FLAGS
# ============================================
ENABLE_RL_SCALING = True  # Enable QMIX RL-based scaling
ENABLE_MIGRATION = True  # Enable container migration
ENABLE_ENERGY_MODEL = True  # Enable energy tracking
ENABLE_DELAYED_PLACEMENT = False  # Advanced feature
