"""
Configuration file for Energy-Efficient Container Cloud Simulator
Contains all hyperparameters and system settings
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
CORE_SPEED = 5_000_000  # ✅ FIXED: 5M instructions per second per core (was 1M)

# ============================================
# CONTAINER SPECIFICATIONS
# ============================================
# Range for random container workload (instructions)
CONTAINER_MIN_INSTRUCTIONS = 2_000_000_000   # Was 500M - longer tasks
CONTAINER_MAX_INSTRUCTIONS = 8_000_000_000   # Was 2B

# Range for random deadline offset (seconds from arrival)
CONTAINER_MIN_DEADLINE_OFFSET = 300  # 5 minutes
CONTAINER_MAX_DEADLINE_OFFSET = 900  # 15 minutes

# Default resource allocation (will be overridden by RL later)
DEFAULT_CORES_PER_CONTAINER = 1

# ============================================
# WORKLOAD GENERATOR SETTINGS
# ============================================
MIN_CONTAINERS_PER_SLOT = 0  # minimum new containers per time slot
MAX_CONTAINERS_PER_SLOT = 15  # maximum new containers per time slot

# ============================================
# PLACEMENT MODULE SETTINGS
# ============================================
PLACEMENT_CPU_THRESHOLD = 0.80  # 80% - don't place if PM is above this
PLACEMENT_STRATEGY = "first_fit"  # Options: first_fit, best_fit (for future)

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
# MIGRATION SETTINGS (for Step 4)
# ============================================
MIGRATION_UNDERLOAD_THRESHOLD = 0.20  # 30% - migrate out if below
MIGRATION_OVERLOAD_THRESHOLD = 0.70  # 90% - migrate out if above
MIGRATION_COST_INSTRUCTIONS = 1_000_000  # penalty for migration

# ============================================
# RL AGENT SETTINGS (for Step 3)
# ============================================
# Q-Learning parameters
RL_LEARNING_RATE = 0.1  # alpha
RL_DISCOUNT_FACTOR = 0.95  # gamma
RL_EPSILON_START = 0.9  # initial exploration rate
RL_EPSILON_MIN = 0.1  # minimum exploration rate
RL_EPSILON_DECAY = 0.995  # decay rate per episode

# Reward shaping
RL_DEADLINE_VIOLATION_PENALTY = 1000  # large penalty for missing deadlines
RL_REJECTION_PENALTY = 500  # penalty for rejecting containers

# State discretization bins (for Q-table)
STATE_NUM_CONTAINERS_BINS = [0, 1, 4, 8, float('inf')]  # [0, 1-3, 4-8, >8]
STATE_REMAINING_WORK_BINS = 5  # number of bins for work discretization
STATE_DEADLINE_GAP_BINS = 5  # number of bins for deadline gap
STATE_CPU_AVAIL_BINS = 5  # number of bins for CPU availability

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
# FUTURE EXTENSIONS (placeholders)
# ============================================
# These will be used in later steps
ENABLE_RL_SCALING = True  # Step 3
ENABLE_MIGRATION = True  # Step 4
ENABLE_ENERGY_MODEL = True  # Step 2 ✅ ENABLED
ENABLE_DELAYED_PLACEMENT = False  # Advanced feature