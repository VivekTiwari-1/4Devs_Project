"""
Placement Module
Handles container placement decisions on Physical Machines

Key fix:
- Added MAX_CONTAINERS_PER_PM hard cap so PMs never accumulate
  hundreds of containers that can never finish before their deadlines.
- can_accommodate() now checks BOTH cpu threshold AND container count cap.
"""

import sys
sys.path.append('..')
from config.config import PLACEMENT_CPU_THRESHOLD, PLACEMENT_STRATEGY

# FIX: Import the new cap — gracefully fall back to 50 if config not updated yet
try:
    from config.config import MAX_CONTAINERS_PER_PM
except ImportError:
    MAX_CONTAINERS_PER_PM = 50


class PlacementModule:
    """
    Manages container placement on Physical Machines.

    Placement strategies:
    - first_fit: Place on first PM with available resources
    - best_fit: Place on PM with least remaining capacity (pack tightly)
    - worst_fit: Place on PM with most remaining capacity (spread load)
    """

    def __init__(self, strategy=PLACEMENT_STRATEGY, threshold=PLACEMENT_CPU_THRESHOLD,
                 max_containers_per_pm=MAX_CONTAINERS_PER_PM):
        self.strategy = strategy
        self.threshold = threshold
        self.max_containers_per_pm = max_containers_per_pm  # FIX: hard cap

        # Statistics
        self.placements = 0
        self.new_pm_starts = 0
        self.placement_failures = 0

    def _can_accommodate(self, pm, container):
        """
        Check if PM can accommodate a new container.
        Enforces BOTH cpu threshold AND container count cap.

        Args:
            pm: PhysicalMachine
            container: Container to place

        Returns:
            bool: True if PM can take the container
        """
        if not pm.is_on:
            return False

        # FIX: Hard cap on container count — prevents PM from accumulating
        # hundreds of containers that physically can't finish before deadline
        if len(pm.containers) >= self.max_containers_per_pm:
            return False

        # CPU utilization threshold check
        future_used = sum(c.assigned_cores for c in pm.containers) + container.assigned_cores
        future_util = future_used / pm.total_cores
        return future_util <= self.threshold

    def place_container(self, container, pms):
        """
        Place a container on an appropriate PM.

        Args:
            container: Container to place
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
        for pm in pms:
            if self._can_accommodate(pm, container):
                if pm.add_container(container):
                    self.placements += 1
                    return True, pm
        return False, None

    def _best_fit_placement(self, container, pms):
        best_pm = None
        min_remaining = float('inf')

        for pm in pms:
            if self._can_accommodate(pm, container):
                remaining = pm.available_cores() - container.assigned_cores
                if remaining < min_remaining:
                    min_remaining = remaining
                    best_pm = pm

        if best_pm is not None and best_pm.add_container(container):
            self.placements += 1
            return True, best_pm
        return False, None

    def _worst_fit_placement(self, container, pms):
        worst_pm = None
        max_remaining = -1

        for pm in pms:
            if self._can_accommodate(pm, container):
                remaining = pm.available_cores()
                if remaining > max_remaining:
                    max_remaining = remaining
                    worst_pm = pm

        if worst_pm is not None and worst_pm.add_container(container):
            self.placements += 1
            return True, worst_pm
        return False, None

    def place_or_start_new_pm(self, container, pms, pm_factory):
        """
        Try to place container on existing PMs, or start a new PM if needed.

        Args:
            container: Container to place
            pms (list): List of existing PMs
            pm_factory (callable): Function that creates a new PM

        Returns:
            tuple: (success: bool, pm: PhysicalMachine, is_new_pm: bool)
        """
        success, pm = self.place_container(container, pms)

        if success:
            return True, pm, False

        # Start new PM
        new_pm = pm_factory()

        if new_pm.add_container(container):
            self.placements += 1
            self.new_pm_starts += 1
            return True, new_pm, True
        else:
            self.placement_failures += 1
            return False, None, False

    def get_statistics(self):
        return {
            'total_placements': self.placements,
            'new_pm_starts': self.new_pm_starts,
            'placement_failures': self.placement_failures,
            'strategy': self.strategy,
            'threshold': self.threshold,
            'max_containers_per_pm': self.max_containers_per_pm,
        }

    def reset_statistics(self):
        self.placements = 0
        self.new_pm_starts = 0
        self.placement_failures = 0


class DelayedPlacementQueue:
    """
    Queue for containers that couldn't be placed immediately.
    Retried in the next time slot.
    """

    def __init__(self, max_delay_slots=2):
        self.queue = []
        self.max_delay_slots = max_delay_slots

    def add(self, container, current_time):
        self.queue.append({
            'container': container,
            'queued_time': current_time,
            'retry_count': 0
        })

    def get_ready_containers(self, current_time):
        ready = []
        remaining = []

        for item in self.queue:
            if item['retry_count'] < self.max_delay_slots:
                ready.append(item['container'])
                item['retry_count'] += 1
                if item['retry_count'] < self.max_delay_slots:
                    remaining.append(item)

        self.queue = remaining
        return ready

    def get_dropped_containers(self):
        dropped = [item['container'] for item in self.queue
                   if item['retry_count'] >= self.max_delay_slots]
        self.queue = [item for item in self.queue
                      if item['retry_count'] < self.max_delay_slots]
        return dropped

    def size(self):
        return len(self.queue)