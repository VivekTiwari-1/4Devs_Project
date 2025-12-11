"""
Container Class
Represents a containerized workload with instructions, deadline, and resource allocation
"""

import sys
sys.path.append('..')
from config.config import DEFAULT_CORES_PER_CONTAINER


class Container:
    """
    Represents a single container in the cloud environment.
    
    Attributes:
        cid (str): Unique container identifier
        remaining_instructions (int): Number of instructions left to execute
        total_instructions (int): Original total instructions (for progress tracking)
        deadline (float): Absolute time (in seconds) when container must finish
        arrival_time (float): Time when container arrived in the system
        assigned_cores (int): Number of CPU cores currently assigned
        finish_time (float): Time when container completed (None if not finished)
        pm_id (int): ID of PM where container is running (None if not placed)
    """
    
    def __init__(self, cid, instructions, deadline, arrival_time):
        """
        Initialize a new container.
        
        Args:
            cid (str): Container ID
            instructions (int): Total instructions to execute
            deadline (float): Absolute deadline time (seconds)
            arrival_time (float): Time when container arrived
        """
        self.cid = cid
        self.remaining_instructions = instructions
        self.total_instructions = instructions
        self.deadline = deadline
        self.arrival_time = arrival_time
        
        # Resource allocation (will be managed by RL agent in future)
        self.assigned_cores = DEFAULT_CORES_PER_CONTAINER
        
        # Execution tracking
        self.finish_time = None
        self.pm_id = None  # PM where this container is running
        
    def execute(self, core_speed, tau):
        """
        Execute container for one time slot.
        
        Args:
            core_speed (int): Instructions per second per core
            tau (int): Time slot duration in seconds
            
        Returns:
            bool: True if container finished execution, False otherwise
        """
        # Calculate instructions executed this time slot
        executed_instructions = self.assigned_cores * core_speed * tau
        
        # Update remaining work
        self.remaining_instructions -= executed_instructions
        
        # Check if finished
        if self.remaining_instructions <= 0:
            self.remaining_instructions = 0  # clamp to zero
            return True  # finished
        
        return False  # still running
    
    def is_finished(self):
        """Check if container has completed execution."""
        return self.remaining_instructions <= 0
    
    def is_deadline_violated(self, current_time):
        """
        Check if container missed its deadline.
        
        Args:
            current_time (float): Current simulation time
            
        Returns:
            bool: True if deadline is missed, False otherwise
        """
        if self.finish_time is None:
            # Still running - check if deadline has passed
            return current_time > self.deadline
        else:
            # Finished - check if finished after deadline
            return self.finish_time > self.deadline
    
    def get_progress_percentage(self):
        """
        Calculate execution progress percentage.
        
        Returns:
            float: Progress percentage (0-100)
        """
        completed = self.total_instructions - self.remaining_instructions
        return (completed / self.total_instructions) * 100
    
    def get_remaining_time_to_deadline(self, current_time):
        """
        Calculate time remaining until deadline.
        
        Args:
            current_time (float): Current simulation time
            
        Returns:
            float: Seconds until deadline (negative if deadline passed)
        """
        return self.deadline - current_time
    
    def estimate_completion_time(self, current_time, core_speed):
        """
        Estimate when container will finish with current allocation.
        
        Args:
            current_time (float): Current simulation time
            core_speed (int): Instructions per second per core
            
        Returns:
            float: Estimated completion time (absolute time)
        """
        if self.assigned_cores == 0:
            return float('inf')  # will never finish
        
        instructions_per_second = self.assigned_cores * core_speed
        seconds_needed = self.remaining_instructions / instructions_per_second
        
        return current_time + seconds_needed
    
    def __repr__(self):
        """String representation for debugging."""
        status = "FINISHED" if self.is_finished() else "RUNNING"
        return (f"Container({self.cid}, status={status}, "
                f"progress={self.get_progress_percentage():.1f}%, "
                f"cores={self.assigned_cores})")
    
    def get_state_dict(self):
        """
        Get container state as dictionary (useful for logging/serialization).
        
        Returns:
            dict: Container state information
        """
        return {
            'cid': self.cid,
            'remaining_instructions': self.remaining_instructions,
            'total_instructions': self.total_instructions,
            'deadline': self.deadline,
            'arrival_time': self.arrival_time,
            'assigned_cores': self.assigned_cores,
            'finish_time': self.finish_time,
            'pm_id': self.pm_id,
            'progress_pct': self.get_progress_percentage()
        }