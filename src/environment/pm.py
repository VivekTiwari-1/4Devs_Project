"""
Physical Machine (PM) Class
Represents a server/host machine that runs containers
"""

import sys
sys.path.append('..')
from config.config import PM_TOTAL_CORES, PM_MEMORY_GB, PM_NETWORK_GBPS


class PhysicalMachine:
    """
    Represents a physical server in the data center.
    
    Attributes:
        pm_id (int): Unique PM identifier
        total_cores (int): Total CPU cores available
        total_memory (int): Total memory in GB
        total_network (int): Network bandwidth in Gbps
        is_on (bool): Power state (ON/OFF)
        containers (list): List of Container objects running on this PM
        state_history (list): History of state transitions (for debugging)
    """
    
    def __init__(self, pm_id, total_cores=PM_TOTAL_CORES, 
                 total_memory=PM_MEMORY_GB, total_network=PM_NETWORK_GBPS):
        """
        Initialize a Physical Machine.
        
        Args:
            pm_id (int): Unique identifier
            total_cores (int): Total CPU cores
            total_memory (int): Total memory in GB
            total_network (int): Network bandwidth in Gbps
        """
        self.pm_id = pm_id
        self.total_cores = total_cores
        self.total_memory = total_memory
        self.total_network = total_network
        
        # Power state
        self.is_on = True  # PM starts in ON state when created
        
        # Container management
        self.containers = []
        
        # State tracking (for future RL/migration)
        self.state_history = []
        self.turn_on_time = None
        self.turn_off_time = None
        
    def available_cores(self):
        """
        Calculate number of available CPU cores.
        
        Returns:
            int: Number of free cores
        """
        used_cores = sum(container.assigned_cores for container in self.containers)
        return self.total_cores - used_cores
    
    def cpu_utilization(self):
        """
        Calculate CPU utilization as a fraction (0.0 to 1.0).
        
        Returns:
            float: CPU utilization ratio
        """
        used_cores = sum(container.assigned_cores for container in self.containers)
        return used_cores / self.total_cores if self.total_cores > 0 else 0.0
    
    def memory_utilization(self):
        """
        Placeholder for memory utilization calculation.
        Will be implemented when memory requirements are added to containers.
        
        Returns:
            float: Memory utilization ratio (0.0 to 1.0)
        """
        # TODO: Implement when containers have memory requirements
        return 0.0
    
    def network_utilization(self):
        """
        Placeholder for network utilization calculation.
        Will be implemented when network requirements are added.
        
        Returns:
            float: Network utilization ratio (0.0 to 1.0)
        """
        # TODO: Implement when containers have network requirements
        return 0.0
    
    def can_accommodate(self, container, threshold=1.0):
        """
        Check if PM can accommodate a new container.
        
        Args:
            container (Container): Container to check
            threshold (float): Maximum allowed utilization (0.0 to 1.0)
            
        Returns:
            bool: True if container can be placed, False otherwise
        """
        # Check if PM is ON
        if not self.is_on:
            return False
        
        # Check if adding container would exceed threshold
        future_used_cores = sum(c.assigned_cores for c in self.containers) + container.assigned_cores
        future_utilization = future_used_cores / self.total_cores
        
        return future_utilization <= threshold
    
    def add_container(self, container):
        """
        Add a container to this PM.
        
        Args:
            container (Container): Container to add
            
        Returns:
            bool: True if successfully added, False otherwise
        """
        # Check if there are enough available cores
        if self.available_cores() >= container.assigned_cores:
            self.containers.append(container)
            container.pm_id = self.pm_id  # update container's PM reference
            return True
        return False
    
    def remove_container(self, container):
        """
        Remove a specific container from this PM.
        
        Args:
            container (Container): Container to remove
            
        Returns:
            bool: True if removed, False if not found
        """
        if container in self.containers:
            self.containers.remove(container)
            container.pm_id = None
            return True
        return False
    
    def remove_finished_containers(self, current_time):
        """
        Remove all finished containers from this PM.
        
        Args:
            current_time (float): Current simulation time
            
        Returns:
            tuple: (num_finished, num_violations)
        """
        finished_containers = []
        violations = 0
        
        for container in self.containers:
            if container.is_finished():
                # Set finish time if not already set
                if container.finish_time is None:
                    container.finish_time = current_time
                
                # Check for deadline violation
                if container.is_deadline_violated(current_time):
                    violations += 1
                
                finished_containers.append(container)
        
        # Remove finished containers
        for container in finished_containers:
            self.remove_container(container)
        
        return len(finished_containers), violations
    
    def turn_on(self, current_time):
        """
        Turn on the PM.
        
        Args:
            current_time (float): Current simulation time
        """
        if not self.is_on:
            self.is_on = True
            self.turn_on_time = current_time
            self.state_history.append(('ON', current_time))
    
    def turn_off(self, current_time):
        """
        Turn off the PM (must have no containers running).
        
        Args:
            current_time (float): Current simulation time
            
        Returns:
            bool: True if turned off, False if containers still running
        """
        if len(self.containers) == 0:
            self.is_on = False
            self.turn_off_time = current_time
            self.state_history.append(('OFF', current_time))
            return True
        return False
    
    def is_underloaded(self, threshold=0.3):
        """
        Check if PM is underloaded (for migration decisions).
        
        Args:
            threshold (float): Underload threshold (default 30%)
            
        Returns:
            bool: True if underloaded
        """
        return self.cpu_utilization() < threshold and len(self.containers) > 0
    
    def is_overloaded(self, threshold=0.9):
        """
        Check if PM is overloaded (for migration decisions).
        
        Args:
            threshold (float): Overload threshold (default 90%)
            
        Returns:
            bool: True if overloaded
        """
        return self.cpu_utilization() > threshold
    
    def get_num_containers(self):
        """Get number of containers running on this PM."""
        return len(self.containers)
    
    def __repr__(self):
        """String representation for debugging."""
        state = "ON" if self.is_on else "OFF"
        return (f"PM(id={self.pm_id}, state={state}, "
                f"containers={len(self.containers)}, "
                f"cpu_util={self.cpu_utilization():.2%})")
    
    def get_state_dict(self):
        """
        Get PM state as dictionary (useful for logging/serialization).
        
        Returns:
            dict: PM state information
        """
        return {
            'pm_id': self.pm_id,
            'is_on': self.is_on,
            'num_containers': len(self.containers),
            'cpu_utilization': self.cpu_utilization(),
            'available_cores': self.available_cores(),
            'total_cores': self.total_cores
        }
