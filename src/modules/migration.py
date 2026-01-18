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
    """
    
    def __init__(self, underload_threshold=MIGRATION_UNDERLOAD_THRESHOLD,
                 overload_threshold=MIGRATION_OVERLOAD_THRESHOLD,
                 migration_cost=MIGRATION_COST_INSTRUCTIONS):
        self.underload_threshold = underload_threshold
        self.overload_threshold = overload_threshold
        self.migration_cost = migration_cost
        
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
        migrations_this_round = 0
        
        # First handle underloaded PMs
        migrations_this_round += self._handle_underloaded_pms(pms, current_time)
        
        # Then handle overloaded PMs
        migrations_this_round += self._handle_overloaded_pms(pms, current_time)
        
        return migrations_this_round
    
    def _handle_underloaded_pms(self, pms, current_time):
        """
        Migrate containers from underloaded PMs and turn them off.
        
        Args:
            pms (list): List of PMs
            current_time (float): Current time
            
        Returns:
            int: Number of migrations
        """
        migrations = 0
        
        for pm in pms:
            if not pm.is_on or len(pm.containers) == 0:
                continue
            
            # Check if underloaded
            if pm.cpu_utilization() < self.underload_threshold:
                # Try to migrate all containers
                containers_to_migrate = list(pm.containers)
                
                for container in containers_to_migrate:
                    # Find destination PM
                    dest_pm = self._find_destination_pm(container, pms, pm)
                    
                    if dest_pm:
                        # Perform migration
                        success = self._migrate_container(container, pm, dest_pm)
                        if success:
                            migrations += 1
                        else:
                            self.failed_migrations += 1
                
                # If PM is now empty, turn it off
                if len(pm.containers) == 0:
                    pm.turn_off(current_time)
                    self.pms_turned_off += 1
        
        return migrations
    
    def _handle_overloaded_pms(self, pms, current_time):
        """
        Migrate containers from overloaded PMs.
        
        Args:
            pms (list): List of PMs
            current_time (float): Current time
            
        Returns:
            int: Number of migrations
        """
        migrations = 0
        
        for pm in pms:
            if not pm.is_on or len(pm.containers) == 0:
                continue
            
            # Check if overloaded
            if pm.cpu_utilization() > self.overload_threshold:
                # Migrate one container (largest first)
                container = self._select_container_to_migrate(pm)
                
                if container:
                    dest_pm = self._find_destination_pm(container, pms, pm)
                    
                    if dest_pm:
                        success = self._migrate_container(container, pm, dest_pm)
                        if success:
                            migrations += 1
                        else:
                            self.failed_migrations += 1
        
        return migrations
    
    def _find_destination_pm(self, container, pms, source_pm):
        """
        Find suitable destination PM for container.
        
        Args:
            container: Container to migrate
            pms (list): All PMs
            source_pm: Source PM
            
        Returns:
            PhysicalMachine or None
        """
        best_pm = None
        min_util_after = float('inf')
        
        for pm in pms:
            # Skip source PM and OFF PMs
            if pm == source_pm or not pm.is_on:
                continue
            
            # Check if PM can accommodate
            future_cores = sum(c.assigned_cores for c in pm.containers) + container.assigned_cores
            future_util = future_cores / pm.total_cores
            
            # Must stay below threshold
            if future_util <= 0.85:  # Leave some headroom
                # Prefer PM with lowest utilization after migration
                if future_util < min_util_after:
                    min_util_after = future_util
                    best_pm = pm
        
        return best_pm
    
    def _select_container_to_migrate(self, pm):
        """
        Select which container to migrate from overloaded PM.
        Strategy: largest container (most cores)
        
        Args:
            pm: PhysicalMachine
            
        Returns:
            Container or None
        """
        if not pm.containers:
            return None
        
        # Select container with most assigned cores
        return max(pm.containers, key=lambda c: c.assigned_cores)
    
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
        # Remove from source
        if not source_pm.remove_container(container):
            return False
        
        # Add migration cost to container's remaining work
        container.remaining_instructions += self.migration_cost
        
        # Add to destination
        if dest_pm.add_container(container):
            self.total_migrations += 1
            return True
        else:
            # Failed to add to destination, add back to source
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
        
        # Find an OFF PM
        for pm in pms:
            if not pm.is_on:
                pm.turn_on(0)  # Turn on
                self.pms_turned_on += 1
                return pm
        
        return None
    
    def get_statistics(self):
        """
        Get migration statistics.
        
        Returns:
            dict: Statistics
        """
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