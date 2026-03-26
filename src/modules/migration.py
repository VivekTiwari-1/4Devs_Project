"""
Migration Module - Container Migration and PM Power Management
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import (
    MIGRATION_UNDERLOAD_THRESHOLD,
    MIGRATION_OVERLOAD_THRESHOLD,
    MIGRATION_COST_INSTRUCTIONS
)


class MigrationModule:
    """
    Handles container migration between PMs and PM power management.

    Key fixes:
    - Migration cooldown per container (prevents thrashing)
    - Consolidation-first destination selection (bin-packing)
    - Skip containers near deadline (migration would cause violation)
    - Underload threshold lowered via config
    """

    def __init__(self, underload_threshold=MIGRATION_UNDERLOAD_THRESHOLD,
                 overload_threshold=MIGRATION_OVERLOAD_THRESHOLD,
                 migration_cost=MIGRATION_COST_INSTRUCTIONS):
        self.underload_threshold = underload_threshold
        self.overload_threshold = overload_threshold
        self.migration_cost = migration_cost

        # --- FIX: Cooldown tracking per container (by id) ---
        # Container won't be migrated again for COOLDOWN_SLOTS slots
        self.COOLDOWN_SLOTS = 5
        self.migration_cooldown = {}   # {container_id: last_migrated_slot}
        self.current_slot = 0

        # --- FIX: Limit migrations per slot to avoid thrashing ---
        self.MAX_MIGRATIONS_PER_SLOT = 3

        # Statistics
        self.total_migrations = 0
        self.pms_turned_off = 0
        self.pms_turned_on = 0
        self.failed_migrations = 0

    def check_and_migrate(self, pms, current_time):
        """
        Check all PMs and perform migrations if needed.

        Args:
            pms (list): List of PhysicalMachine objects
            current_time (float): Current simulation time

        Returns:
            int: Number of migrations performed
        """
        # Track slot number for cooldown
        self.current_slot = int(current_time / 30)

        migrations_this_round = 0

        # --- FIX: Cap total migrations per slot ---
        remaining_budget = self.MAX_MIGRATIONS_PER_SLOT

        # First handle overloaded PMs (higher priority)
        done, remaining_budget = self._handle_overloaded_pms(pms, current_time, remaining_budget)
        migrations_this_round += done

        # Then handle underloaded PMs only if budget remains
        if remaining_budget > 0:
            done, remaining_budget = self._handle_underloaded_pms(pms, current_time, remaining_budget)
            migrations_this_round += done

        return migrations_this_round

    def _can_migrate(self, container, current_time):
        """
        Check if a container is eligible for migration.

        Args:
            container: Container object
            current_time: Current simulation time

        Returns:
            bool: True if migration is allowed
        """
        container_id = id(container)

        # --- FIX: Cooldown check ---
        last_slot = self.migration_cooldown.get(container_id, -self.COOLDOWN_SLOTS)
        if self.current_slot - last_slot < self.COOLDOWN_SLOTS:
            return False

        # --- FIX: Don't migrate containers close to their deadline ---
        # If deadline is within 2 time slots (60s), skip migration
        if hasattr(container, 'deadline'):
            time_left = container.deadline - current_time
            if time_left < 60:
                return False

        return True

    def _record_migration(self, container):
        """Record that a container was just migrated (for cooldown)."""
        self.migration_cooldown[id(container)] = self.current_slot

    def _handle_underloaded_pms(self, pms, current_time, budget):
        """
        Migrate containers from underloaded PMs and turn them off.

        Args:
            pms (list): List of PMs
            current_time (float): Current time
            budget (int): Max migrations allowed

        Returns:
            tuple: (migrations_done, remaining_budget)
        """
        migrations = 0

        for pm in pms:
            if budget <= 0:
                break

            if not pm.is_on or len(pm.containers) == 0:
                continue

            if pm.cpu_utilization() >= self.underload_threshold:
                continue

            containers_to_migrate = list(pm.containers)

            for container in containers_to_migrate:
                if budget <= 0:
                    break

                # --- FIX: Check cooldown and deadline before migrating ---
                if not self._can_migrate(container, current_time):
                    continue

                dest_pm = self._find_destination_pm(container, pms, pm)

                if dest_pm:
                    success = self._migrate_container(container, pm, dest_pm)
                    if success:
                        migrations += 1
                        budget -= 1
                        self._record_migration(container)
                    else:
                        self.failed_migrations += 1

            # Only turn off if truly empty now
            if len(pm.containers) == 0:
                pm.turn_off(current_time)
                self.pms_turned_off += 1

        return migrations, budget

    def _handle_overloaded_pms(self, pms, current_time, budget):
        """
        Migrate containers from overloaded PMs.

        Args:
            pms (list): List of PMs
            current_time (float): Current time
            budget (int): Max migrations allowed

        Returns:
            tuple: (migrations_done, remaining_budget)
        """
        migrations = 0

        for pm in pms:
            if budget <= 0:
                break

            if not pm.is_on or len(pm.containers) == 0:
                continue

            if pm.cpu_utilization() <= self.overload_threshold:
                continue

            container = self._select_container_to_migrate(pm)

            if container and self._can_migrate(container, current_time):
                dest_pm = self._find_destination_pm(container, pms, pm)

                if dest_pm:
                    success = self._migrate_container(container, pm, dest_pm)
                    if success:
                        migrations += 1
                        budget -= 1
                        self._record_migration(container)
                    else:
                        self.failed_migrations += 1

        return migrations, budget

    def _find_destination_pm(self, container, pms, source_pm):
        """
        Find suitable destination PM using consolidation strategy (bin-packing).
        Prefers the MOST loaded PM that can still fit the container,
        to consolidate workloads and free up PMs for shutdown.

        Args:
            container: Container to migrate
            pms (list): All PMs
            source_pm: Source PM

        Returns:
            PhysicalMachine or None
        """
        best_pm = None
        # --- FIX: Pick most-loaded PM (consolidation), not least-loaded ---
        max_util_after = -1

        for pm in pms:
            if pm == source_pm or not pm.is_on:
                continue

            future_cores = sum(c.assigned_cores for c in pm.containers) + container.assigned_cores
            future_util = future_cores / pm.total_cores

            # Must stay below overload threshold with some headroom
            if future_util <= (self.overload_threshold - 0.05):
                if future_util > max_util_after:
                    max_util_after = future_util
                    best_pm = pm

        return best_pm

    def _select_container_to_migrate(self, pm):
        """
        Select which container to migrate from overloaded PM.
        Strategy: smallest container (least disruption).

        Args:
            pm: PhysicalMachine

        Returns:
            Container or None
        """
        if not pm.containers:
            return None

        # --- FIX: Migrate smallest container (was largest) ---
        # Smallest causes least disruption and migration overhead
        return min(pm.containers, key=lambda c: c.assigned_cores)

    def _migrate_container(self, container, source_pm, dest_pm):
        """
        Perform actual container migration.

        Args:
            container: Container to migrate
            source_pm: Source PM
            dest_pm: Destination PM

        Returns:
            bool: Success
        """
        if not source_pm.remove_container(container):
            return False

        # Add migration cost
        container.remaining_instructions += self.migration_cost

        if dest_pm.add_container(container):
            self.total_migrations += 1
            return True
        else:
            # Rollback
            source_pm.add_container(container)
            return False

    def turn_on_pm_if_needed(self, pms, pending_containers):
        """
        Turn on OFF PMs if needed for pending containers.

        Args:
            pms (list): All PMs
            pending_containers (list): Containers waiting for placement

        Returns:
            PhysicalMachine or None: Turned on PM
        """
        if not pending_containers:
            return None

        for pm in pms:
            if not pm.is_on:
                pm.turn_on(0)
                self.pms_turned_on += 1
                return pm

        return None

    def get_statistics(self):
        """Get migration statistics."""
        return {
            'total_migrations': self.total_migrations,
            'pms_turned_off': self.pms_turned_off,
            'pms_turned_on': self.pms_turned_on,
            'failed_migrations': self.failed_migrations
        }

    def reset_statistics(self):
        """Reset statistics."""
        self.total_migrations = 0
        self.pms_turned_off = 0
        self.pms_turned_on = 0
        self.failed_migrations = 0