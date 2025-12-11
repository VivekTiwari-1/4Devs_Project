<<<<<<< HEAD
"""
Workload Generator
Generates random container arrivals for simulation
"""

import random
import sys
sys.path.append('..')
from config.config import (
    MIN_CONTAINERS_PER_SLOT,
    MAX_CONTAINERS_PER_SLOT,
    CONTAINER_MIN_INSTRUCTIONS,
    CONTAINER_MAX_INSTRUCTIONS,
    CONTAINER_MIN_DEADLINE_OFFSET,
    CONTAINER_MAX_DEADLINE_OFFSET,
    RANDOM_SEED
)
from environment.container import Container


class WorkloadGenerator:
    """
    Generates container workloads for the simulator.
    
    Supports multiple workload patterns:
    - Random: containers arrive randomly each time slot
    - Poisson: arrival follows Poisson distribution
    - Bursty: periodic bursts of high load
    - Trace-based: load from real workload traces (future extension)
    """
    
    def __init__(self, pattern='random', seed=RANDOM_SEED):
        """
        Initialize workload generator.
        
        Args:
            pattern (str): Workload pattern ('random', 'poisson', 'bursty')
            seed (int): Random seed for reproducibility
        """
        self.pattern = pattern
        self.container_counter = 0  # global counter for unique IDs
        
        # Set random seed for reproducibility
        random.seed(seed)
        
    def generate(self, current_time):
        """
        Generate new containers for current time slot.
        
        Args:
            current_time (float): Current simulation time
            
        Returns:
            list: List of new Container objects
        """
        if self.pattern == 'random':
            return self._generate_random(current_time)
        elif self.pattern == 'poisson':
            return self._generate_poisson(current_time)
        elif self.pattern == 'bursty':
            return self._generate_bursty(current_time)
        else:
            raise ValueError(f"Unknown workload pattern: {self.pattern}")
    
    def _generate_random(self, current_time):
        """
        Generate random number of containers each time slot.
        
        Args:
            current_time (float): Current simulation time
            
        Returns:
            list: List of Container objects
        """
        num_containers = random.randint(MIN_CONTAINERS_PER_SLOT, MAX_CONTAINERS_PER_SLOT)
        
        containers = []
        for i in range(num_containers):
            container = self._create_container(current_time)
            containers.append(container)
        
        return containers
    
    def _generate_poisson(self, current_time, lambda_rate=2.5):
        """
        Generate containers using Poisson arrival process.
        
        Args:
            current_time (float): Current simulation time
            lambda_rate (float): Average arrival rate per time slot
            
        Returns:
            list: List of Container objects
        """
        # Use Poisson distribution for number of arrivals
        num_containers = min(
            random.poisson(lambda_rate),
            MAX_CONTAINERS_PER_SLOT  # cap at maximum
        )
        
        containers = []
        for i in range(num_containers):
            container = self._create_container(current_time)
            containers.append(container)
        
        return containers
    
    def _generate_bursty(self, current_time, burst_period=300):
        """
        Generate bursty workload (high load every N seconds).
        
        Args:
            current_time (float): Current simulation time
            burst_period (int): Time between bursts (seconds)
            
        Returns:
            list: List of Container objects
        """
        # Check if we're in a burst period (every burst_period seconds)
        time_in_cycle = current_time % burst_period
        
        if time_in_cycle < 60:  # burst for first 60 seconds of cycle
            num_containers = random.randint(4, MAX_CONTAINERS_PER_SLOT)
        else:
            num_containers = random.randint(MIN_CONTAINERS_PER_SLOT, 2)
        
        containers = []
        for i in range(num_containers):
            container = self._create_container(current_time)
            containers.append(container)
        
        return containers
    
    def _create_container(self, current_time):
        """
        Create a single container with random parameters.
        
        Args:
            current_time (float): Current simulation time (arrival time)
            
        Returns:
            Container: New container object
        """
        # Generate unique container ID
        cid = f"C_{self.container_counter}"
        self.container_counter += 1
        
        # Random workload size (instructions)
        instructions = random.randint(
            CONTAINER_MIN_INSTRUCTIONS,
            CONTAINER_MAX_INSTRUCTIONS
        )
        
        # Random deadline offset from arrival time
        deadline_offset = random.randint(
            CONTAINER_MIN_DEADLINE_OFFSET,
            CONTAINER_MAX_DEADLINE_OFFSET
        )
        deadline = current_time + deadline_offset
        
        # Create container
        container = Container(
            cid=cid,
            instructions=instructions,
            deadline=deadline,
            arrival_time=current_time
        )
        
        return container
    
    def reset(self):
        """Reset the generator (useful for running multiple experiments)."""
        self.container_counter = 0
        random.seed(RANDOM_SEED)
    
    def get_statistics(self):
        """
        Get workload generator statistics.
        
        Returns:
            dict: Statistics about generated workload
        """
        return {
            'pattern': self.pattern,
            'total_containers_generated': self.container_counter
        }


# Convenience function for quick generation
def generate_workload(current_time, pattern='random'):
    """
    Quick function to generate workload without instantiating class.
    
    Args:
        current_time (float): Current simulation time
        pattern (str): Workload pattern
        
    Returns:
        list: List of new containers
    """
    generator = WorkloadGenerator(pattern=pattern)
    return generator.generate(current_time)
=======
a
>>>>>>> 3fdc487e76549e8239d83e40c86355f2a7360963
