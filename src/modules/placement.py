"""
Placement Module
Handles container placement decisions on Physical Machines
"""

import sys
sys.path.append('..')
from config.config import PLACEMENT_CPU_THRESHOLD, PLACEMENT_STRATEGY


class PlacementModule:
    """
    Manages container placement on Physical Machines.
    
    Placement strategies:
    - first_fit: Place on first PM with available resources
    - best_fit: Place on PM with least remaining capacity (pack tightly)
    - worst_fit: Place on PM with most remaining capacity (spread load)
    """
    
    def __init__(self, strategy=PLACEMENT_STRATEGY, threshold=PLACEMENT_CPU_THRESHOLD):
        """
        Initialize placement module.
        
        Args:
            strategy (str): Placement strategy
            threshold (float): CPU utilization threshold (0.0 to 1.0)
        """
        self.strategy = strategy
        self.threshold = threshold
        
        # Statistics
        self.placements = 0
        self.new_pm_starts = 0
        self.placement_failures = 0
        
    def place_container(self, container, pms):
        """
        Place a container on an appropriate PM.
        
        Args:
            container (Container): Container to place
            pms (list): List of PhysicalMachine objects
            
        Returns:
            tuple: (success: bool, pm: PhysicalMachine or None)
        """
        if self.strategy == 'first_fit':
            return self._first_fit_placement(container, pms)
        elif self.strategy == 'best_fit':
            return self._best_fit_placement(container, pms)
        elif self.strategy == 'worst_fit':
            return self._worst_fit_placement(container, pms)
        else:
            raise ValueError(f"Unknown placement strategy: {self.strategy}")
    
    def _first_fit_placement(self, container, pms):
        """
        First Fit: place on first PM that can accommodate.
        
        Args:
            container (Container): Container to place
            pms (list): List of available PMs
            
        Returns:
            tuple: (success, pm)
        """
        # Try to place on existing active PMs
        for pm in pms:
            if pm.is_on and pm.can_accommodate(container, self.threshold):
                if pm.add_container(container):
                    self.placements += 1
                    return True, pm
        
        # No suitable PM found
        return False, None
    
    def _best_fit_placement(self, container, pms):
        """
        Best Fit: place on PM with least remaining capacity after placement.
        This packs containers tightly to minimize number of active PMs.
        
        Args:
            container (Container): Container to place
            pms (list): List of available PMs
            
        Returns:
            tuple: (success, pm)
        """
        best_pm = None
        min_remaining_cores = float('inf')
        
        # Find PM with minimum remaining cores after placement
        for pm in pms:
            if pm.is_on and pm.can_accommodate(container, self.threshold):
                remaining_after = pm.available_cores() - container.assigned_cores
                if remaining_after < min_remaining_cores:
                    min_remaining_cores = remaining_after
                    best_pm = pm
        
        # Place on best PM if found
        if best_pm is not None:
            if best_pm.add_container(container):
                self.placements += 1
                return True, best_pm
        
        return False, None
    
    def _worst_fit_placement(self, container, pms):
        """
        Worst Fit: place on PM with most remaining capacity.
        This spreads load across PMs evenly.
        
        Args:
            container (Container): Container to place
            pms (list): List of available PMs
            
        Returns:
            tuple: (success, pm)
        """
        worst_pm = None
        max_remaining_cores = -1
        
        # Find PM with maximum remaining cores
        for pm in pms:
            if pm.is_on and pm.can_accommodate(container, self.threshold):
                remaining = pm.available_cores()
                if remaining > max_remaining_cores:
                    max_remaining_cores = remaining
                    worst_pm = pm
        
        # Place on worst-fit PM if found
        if worst_pm is not None:
            if worst_pm.add_container(container):
                self.placements += 1
                return True, worst_pm
        
        return False, None
    
    def place_or_start_new_pm(self, container, pms, pm_factory):
        """
        Try to place container, or start a new PM if needed.
        
        Args:
            container (Container): Container to place
            pms (list): List of existing PMs
            pm_factory (callable): Function that creates a new PM
            
        Returns:
            tuple: (success: bool, pm: PhysicalMachine, is_new_pm: bool)
        """
        # Try placement on existing PMs
        success, pm = self.place_container(container, pms)
        
        if success:
            return True, pm, False
        
        # Need to start new PM
        new_pm = pm_factory()
        
        if new_pm.add_container(container):
            self.placements += 1
            self.new_pm_starts += 1
            return True, new_pm, True
        else:
            # This should rarely happen (new PM should have capacity)
            self.placement_failures += 1
            return False, None, False
    
    def get_statistics(self):
        """
        Get placement statistics.
        
        Returns:
            dict: Placement statistics
        """
        return {
            'total_placements': self.placements,
            'new_pm_starts': self.new_pm_starts,
            'placement_failures': self.placement_failures,
            'strategy': self.strategy,
            'threshold': self.threshold
        }
    
    def reset_statistics(self):
        """Reset placement statistics."""
        self.placements = 0
        self.new_pm_starts = 0
        self.placement_failures = 0


class DelayedPlacementQueue:
    """
    Queue for containers that couldn't be placed immediately.
    They will be tried again in the next time slot.
    
    This is mentioned in the paper as an optimization technique.
    """
    
    def __init__(self, max_delay_slots=2):
        """
        Initialize delayed placement queue.
        
        Args:
            max_delay_slots (int): Maximum number of slots to delay
        """
        self.queue = []
        self.max_delay_slots = max_delay_slots
        
    def add(self, container, current_time):
        """
        Add container to delayed queue.
        
        Args:
            container (Container): Container to delay
            current_time (float): Current simulation time
        """
        self.queue.append({
            'container': container,
            'queued_time': current_time,
            'retry_count': 0
        })
    
    def get_ready_containers(self, current_time):
        """
        Get containers ready for placement retry.
        
        Args:
            current_time (float): Current simulation time
            
        Returns:
            list: Containers to retry placement
        """
        ready = []
        remaining = []
        
        for item in self.queue:
            # Try placement in next slot
            if item['retry_count'] < self.max_delay_slots:
                ready.append(item['container'])
                item['retry_count'] += 1
                
                # Keep in queue for potential next retry
                if item['retry_count'] < self.max_delay_slots:
                    remaining.append(item)
            # else: exceeded max delay, drop (counted as rejection)
        
        self.queue = remaining
        return ready
    
    def get_dropped_containers(self):
        """
        Get containers that exceeded max delay (rejections).
        
        Returns:
            list: Rejected containers
        """
        dropped = [item['container'] for item in self.queue 
                   if item['retry_count'] >= self.max_delay_slots]
        
        # Remove dropped from queue
        self.queue = [item for item in self.queue 
                      if item['retry_count'] < self.max_delay_slots]
        
        return dropped
    
    def size(self):
        """Get number of containers in queue."""
        return len(self.queue)