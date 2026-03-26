"""
Test migration functionality with heavy load
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.simulator import Simulator
from config import config

# Temporarily increase load
config.MAX_CONTAINERS_PER_SLOT = 15
config.CONTAINER_MIN_INSTRUCTIONS = 2_000_000_000
config.CONTAINER_MAX_INSTRUCTIONS = 8_000_000_000
config.MIGRATION_UNDERLOAD_THRESHOLD = 0.25
config.MIGRATION_OVERLOAD_THRESHOLD = 0.75

print("🧪 Testing Migration with Heavy Load\n")

sim = Simulator(
    workload_pattern='bursty',
    placement_strategy='first_fit',
    enable_rl=True,
    enable_migration=True
)

sim.run(num_slots=50)

stats = sim.get_summary_stats()

print("\n📊 Migration Test Results:")
print(f"  Total migrations: {stats.get('total_migrations', 0)}")
print(f"  PMs turned OFF: {stats.get('pms_turned_off', 0)}")
print(f"  Peak PMs: {stats['peak_active_pms']}")