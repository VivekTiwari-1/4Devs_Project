"""
CPU Allocation Policies
Different strategies for distributing CPU cores among containers
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class AllocationPolicies:
    """
    Collection of CPU allocation policies.
    Each policy takes containers and available cores, returns allocation vector.
    """
    
    @staticmethod
    def fair_allocation(containers, available_cores):
        """
        Fair allocation: distribute cores equally among all containers.
        
        Args:
            containers (list): List of Container objects
            available_cores (int): Number of available CPU cores
            
        Returns:
            list: Allocation vector [cores for each container]
        """
        if not containers or available_cores <= 0:
            return [0] * len(containers)
        
        # Divide equally
        cores_per_container = available_cores / len(containers)
        
        return [cores_per_container] * len(containers)
    
    @staticmethod
    def deadline_priority_allocation(containers, available_cores):
        """
        Deadline priority: allocate more cores to containers with closer deadlines.
        
        Args:
            containers (list): List of Container objects
            available_cores (int): Number of available CPU cores
            
        Returns:
            list: Allocation vector [cores for each container]
        """
        if not containers or available_cores <= 0:
            return [0] * len(containers)
        
        # Calculate urgency (inverse of time to deadline)
        # Use arrival_time as proxy for current_time
        current_time = containers[0].arrival_time if containers else 0
        
        urgencies = []
        for container in containers:
            time_to_deadline = max(1, container.deadline - current_time)  # Avoid division by zero
            urgency = 1.0 / time_to_deadline
            urgencies.append(urgency)
        
        # Normalize urgencies to sum to 1
        total_urgency = sum(urgencies)
        if total_urgency == 0:
            # All equal urgency - fall back to fair
            return AllocationPolicies.fair_allocation(containers, available_cores)
        
        normalized_urgencies = [u / total_urgency for u in urgencies]
        
        # Allocate cores proportionally to urgency
        allocations = [u * available_cores for u in normalized_urgencies]
        
        return allocations
    
    @staticmethod
    def smallest_remaining_work_allocation(containers, available_cores):
        """
        Smallest remaining work first: prioritize containers with less work left.
        This helps finish containers quickly to free resources.
        
        Args:
            containers (list): List of Container objects
            available_cores (int): Number of available CPU cores
            
        Returns:
            list: Allocation vector [cores for each container]
        """
        if not containers or available_cores <= 0:
            return [0] * len(containers)
        
        # Calculate inverse of remaining work (smaller work = higher priority)
        priorities = []
        for container in containers:
            # Inverse priority (smaller remaining = higher priority)
            priority = 1.0 / max(1, container.remaining_instructions)
            priorities.append(priority)
        
        # Normalize priorities
        total_priority = sum(priorities)
        if total_priority == 0:
            return AllocationPolicies.fair_allocation(containers, available_cores)
        
        normalized_priorities = [p / total_priority for p in priorities]
        
        # Allocate cores proportionally
        allocations = [p * available_cores for p in normalized_priorities]
        
        return allocations
    
    @staticmethod
    def conservative_allocation(containers, available_cores):
        """
        Conservative: allocate minimum cores (save energy) unless deadline is urgent.
        Give 20% of available cores spread fairly, unless deadline < 5 minutes.
        
        Args:
            containers (list): List of Container objects
            available_cores (int): Number of available CPU cores
            
        Returns:
            list: Allocation vector [cores for each container]
        """
        if not containers or available_cores <= 0:
            return [0] * len(containers)
        
        current_time = containers[0].arrival_time if containers else 0
        allocations = []
        
        # Check if any container is urgent
        urgent_threshold = 300  # 5 minutes
        has_urgent = any((c.deadline - current_time) < urgent_threshold for c in containers)
        
        if has_urgent:
            # Urgent: allocate more (60% of available)
            cores_to_use = available_cores * 0.6
        else:
            # Not urgent: conserve energy (20% of available)
            cores_to_use = available_cores * 0.2
        
        # Distribute the cores fairly
        cores_per_container = cores_to_use / len(containers)
        allocations = [cores_per_container] * len(containers)
        
        return allocations
    
    @staticmethod
    def apply_policy(policy_name, containers, available_cores):
        """
        Apply a named policy.
        
        Args:
            policy_name (str): Name of policy
            containers (list): List of containers
            available_cores (int): Available cores
            
        Returns:
            list: Allocation vector
        """
        if policy_name == "fair":
            return AllocationPolicies.fair_allocation(containers, available_cores)
        elif policy_name == "deadline_priority":
            return AllocationPolicies.deadline_priority_allocation(containers, available_cores)
        elif policy_name == "smallest_work":
            return AllocationPolicies.smallest_remaining_work_allocation(containers, available_cores)
        elif policy_name == "conservative":
            return AllocationPolicies.conservative_allocation(containers, available_cores)
        else:
            # Default to fair
            return AllocationPolicies.fair_allocation(containers, available_cores)
    
    @staticmethod
    def apply_policy_by_index(policy_index, containers, available_cores):
        """
        Apply policy by index (for RL agent).
        
        Args:
            policy_index (int): Index of policy (0-3)
            containers (list): List of containers
            available_cores (int): Available cores
            
        Returns:
            list: Allocation vector
        """
        policy_names = ["fair", "deadline_priority", "smallest_work", "conservative"]
        policy_name = policy_names[policy_index] if 0 <= policy_index < len(policy_names) else "fair"
        return AllocationPolicies.apply_policy(policy_name, containers, available_cores)