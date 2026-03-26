"""
CPU Allocation Policies
Different strategies for distributing CPU cores among containers.

Key fixes:
- Use container._current_time instead of arrival_time for deadline calculations
- Conservative policy now uses minimum 1 core per container, not 20% of available
- All policies guarantee minimum cores so containers can actually finish
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Minimum cores guaranteed to every container regardless of policy
# At 5M instructions/sec, 1 core processes 150M instructions per 30s slot.
# A 2B instruction container needs ~14 slots minimum at 1 core.
# This ensures every container makes progress every slot.
MIN_CORES_PER_CONTAINER = 1.0


class AllocationPolicies:
    """
    Collection of CPU allocation policies.
    Each policy takes containers and available cores, returns allocation vector.

    All policies now guarantee MIN_CORES_PER_CONTAINER to every container
    so that no container is starved and deadline violations are minimized.
    """

    @staticmethod
    def _get_current_time(containers):
        """
        Safely get current simulation time from containers.
        Uses _current_time if set by simulator, falls back to arrival_time.
        """
        for c in containers:
            if hasattr(c, '_current_time') and c._current_time > 0:
                return c._current_time
        # Fallback — should not happen if simulator sets _current_time correctly
        return containers[0].arrival_time if containers else 0

    @staticmethod
    def _guarantee_minimum(allocations, num_containers, available_cores):
        """
        Ensure every container gets at least MIN_CORES_PER_CONTAINER.
        Scales down proportionally if not enough cores available.

        Args:
            allocations (list): Raw allocation vector
            num_containers (int): Number of containers
            available_cores (float): Total available cores

        Returns:
            list: Adjusted allocation vector
        """
        if not allocations:
            return []

        min_cores = min(MIN_CORES_PER_CONTAINER, available_cores / max(num_containers, 1))

        # Clamp each allocation to at least min_cores
        adjusted = [max(a, min_cores) for a in allocations]

        # If total exceeds available, scale down proportionally
        total = sum(adjusted)
        if total > available_cores and total > 0:
            scale = available_cores / total
            adjusted = [a * scale for a in adjusted]

        return adjusted

    @staticmethod
    def fair_allocation(containers, available_cores):
        """
        Fair allocation: distribute cores equally among all containers.

        Args:
            containers (list): List of Container objects
            available_cores (float): Number of available CPU cores

        Returns:
            list: Allocation vector [cores for each container]
        """
        if not containers or available_cores <= 0:
            return [0] * len(containers)

        cores_per_container = available_cores / len(containers)
        allocations = [cores_per_container] * len(containers)

        return AllocationPolicies._guarantee_minimum(allocations, len(containers), available_cores)

    @staticmethod
    def deadline_priority_allocation(containers, available_cores):
        """
        Deadline priority: allocate more cores to containers with closer deadlines.
        Containers near deadline get proportionally more CPU.

        Args:
            containers (list): List of Container objects
            available_cores (float): Number of available CPU cores

        Returns:
            list: Allocation vector [cores for each container]
        """
        if not containers or available_cores <= 0:
            return [0] * len(containers)

        # FIX: Use _current_time set by simulator, not arrival_time
        current_time = AllocationPolicies._get_current_time(containers)

        urgencies = []
        for container in containers:
            time_to_deadline = max(1, container.deadline - current_time)
            urgency = 1.0 / time_to_deadline
            urgencies.append(urgency)

        total_urgency = sum(urgencies)
        if total_urgency == 0:
            return AllocationPolicies.fair_allocation(containers, available_cores)

        normalized = [u / total_urgency for u in urgencies]
        allocations = [u * available_cores for u in normalized]

        return AllocationPolicies._guarantee_minimum(allocations, len(containers), available_cores)

    @staticmethod
    def smallest_remaining_work_allocation(containers, available_cores):
        """
        Smallest remaining work first: prioritize containers with less work left.
        Helps finish containers quickly to free resources and avoid violations.

        Args:
            containers (list): List of Container objects
            available_cores (float): Number of available CPU cores

        Returns:
            list: Allocation vector [cores for each container]
        """
        if not containers or available_cores <= 0:
            return [0] * len(containers)

        priorities = []
        for container in containers:
            priority = 1.0 / max(1, container.remaining_instructions)
            priorities.append(priority)

        total_priority = sum(priorities)
        if total_priority == 0:
            return AllocationPolicies.fair_allocation(containers, available_cores)

        normalized = [p / total_priority for p in priorities]
        allocations = [p * available_cores for p in normalized]

        return AllocationPolicies._guarantee_minimum(allocations, len(containers), available_cores)

    @staticmethod
    def conservative_allocation(containers, available_cores):
        """
        Conservative: balance energy saving with deadline compliance.

        FIX: Old version used only 20% of cores when not urgent — containers
        with 2B instructions at 0.068 cores could never finish.

        New behaviour:
        - Always guarantee MIN_CORES_PER_CONTAINER per container
        - Urgent containers (deadline < 5 min) get full fair share
        - Non-urgent containers get 50% share (saves energy without starving)
        - Uses _current_time, not arrival_time

        Args:
            containers (list): List of Container objects
            available_cores (float): Number of available CPU cores

        Returns:
            list: Allocation vector [cores for each container]
        """
        if not containers or available_cores <= 0:
            return [0] * len(containers)

        # FIX: Use _current_time set by simulator, not arrival_time
        current_time = AllocationPolicies._get_current_time(containers)

        urgent_threshold = 300  # 5 minutes

        allocations = []
        for container in containers:
            time_to_deadline = container.deadline - current_time
            if time_to_deadline < urgent_threshold:
                # Urgent: give full fair share
                cores = available_cores / len(containers)
            else:
                # Not urgent: give 50% of fair share (saves energy, but not 20%)
                # FIX: was 0.2 * available / n — now 0.5 * available / n
                cores = (available_cores * 0.5) / len(containers)
            allocations.append(cores)

        return AllocationPolicies._guarantee_minimum(allocations, len(containers), available_cores)

    @staticmethod
    def apply_policy(policy_name, containers, available_cores):
        """
        Apply a named policy.

        Args:
            policy_name (str): Name of policy
            containers (list): List of containers
            available_cores (float): Available cores

        Returns:
            list: Allocation vector
        """
        policy_map = {
            "fair":             AllocationPolicies.fair_allocation,
            "deadline_priority": AllocationPolicies.deadline_priority_allocation,
            "smallest_work":    AllocationPolicies.smallest_remaining_work_allocation,
            "conservative":     AllocationPolicies.conservative_allocation,
        }
        fn = policy_map.get(policy_name, AllocationPolicies.fair_allocation)
        return fn(containers, available_cores)

    @staticmethod
    def apply_policy_by_index(policy_index, containers, available_cores):
        """
        Apply policy by index (for RL agent action space).

        Args:
            policy_index (int): Index 0-3
            containers (list): List of containers
            available_cores (float): Available cores

        Returns:
            list: Allocation vector
        """
        policy_names = ["fair", "deadline_priority", "smallest_work", "conservative"]
        policy_name = policy_names[policy_index] if 0 <= policy_index < len(policy_names) else "fair"
        return AllocationPolicies.apply_policy(policy_name, containers, available_cores)