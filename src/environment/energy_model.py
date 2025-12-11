<<<<<<< HEAD
"""
Energy Model - Quadratic CPU Energy Calculation
Implements: P_cpu = α₁·μ + α₂·μ²
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import (
    ENERGY_ALPHA_1,
    ENERGY_ALPHA_2,
    ENERGY_BETA_MEM,
    ENERGY_GAMMA_NET,
    ENERGY_STATIC_POWER,
    TIME_SLOT_DURATION
)


class EnergyModel:
    """
    Energy model for Physical Machines using quadratic CPU model.
    
    Formula:
        P_total = P_static + P_cpu + P_mem + P_net
        P_cpu = α₁·μ + α₂·μ²
        
    Where:
        μ = CPU utilization (0 to 1)
        α₁ = linear coefficient (watts)
        α₂ = quadratic coefficient (watts)
    """
    
    def __init__(self):
        """Initialize energy model with parameters from config."""
        self.alpha_1 = ENERGY_ALPHA_1
        self.alpha_2 = ENERGY_ALPHA_2
        self.beta_mem = ENERGY_BETA_MEM
        self.gamma_net = ENERGY_GAMMA_NET
        self.static_power = ENERGY_STATIC_POWER
        self.tau = TIME_SLOT_DURATION
        
        # Statistics tracking
        self.total_energy_consumed = 0.0  # kWh
        self.energy_history = []
        
    def compute_cpu_power(self, cpu_utilization):
        """
        Compute CPU power consumption using quadratic model.
        
        Args:
            cpu_utilization (float): CPU utilization ratio (0.0 to 1.0)
            
        Returns:
            float: CPU power in watts
        """
        # Quadratic formula: P_cpu = α₁·μ + α₂·μ²
        mu = max(0.0, min(1.0, cpu_utilization))  # Clamp between 0 and 1
        P_cpu = self.alpha_1 * mu + self.alpha_2 * (mu ** 2)
        return P_cpu
    
    def compute_memory_power(self, memory_utilization):
        """
        Compute memory power consumption (linear model).
        
        Args:
            memory_utilization (float): Memory utilization ratio (0.0 to 1.0)
            
        Returns:
            float: Memory power in watts
        """
        mu_mem = max(0.0, min(1.0, memory_utilization))
        P_mem = self.beta_mem * mu_mem
        return P_mem
    
    def compute_network_power(self, network_utilization):
        """
        Compute network power consumption (linear model).
        
        Args:
            network_utilization (float): Network utilization ratio (0.0 to 1.0)
            
        Returns:
            float: Network power in watts
        """
        mu_net = max(0.0, min(1.0, network_utilization))
        P_net = self.gamma_net * mu_net
        return P_net
    
    def compute_pm_power(self, pm):
        """
        Compute total power consumption for a Physical Machine.
        
        Args:
            pm (PhysicalMachine): PM object
            
        Returns:
            float: Total power in watts
        """
        if not pm.is_on:
            return 0.0  # PM is OFF, no power consumption
        
        # CPU power (quadratic)
        P_cpu = self.compute_cpu_power(pm.cpu_utilization())
        
        # Memory power (linear) - placeholder for now
        P_mem = self.compute_memory_power(pm.memory_utilization())
        
        # Network power (linear) - placeholder for now
        P_net = self.compute_network_power(pm.network_utilization())
        
        # Static power (idle power when ON)
        P_static = self.static_power
        
        # Total instantaneous power
        P_total = P_static + P_cpu + P_mem + P_net
        
        return P_total
    
    def compute_pm_energy(self, pm):
        """
        Compute energy consumed by a PM during one time slot.
        
        Args:
            pm (PhysicalMachine): PM object
            
        Returns:
            float: Energy consumed in kWh
        """
        # Power in watts
        power_watts = self.compute_pm_power(pm)
        
        # Energy = Power × Time
        # Convert time from seconds to hours: tau / 3600
        energy_kwh = power_watts * (self.tau / 3600)
        
        return energy_kwh
    
    def compute_datacenter_energy(self, pms):
        """
        Compute total energy consumed by all PMs in one time slot.
        
        Args:
            pms (list): List of PhysicalMachine objects
            
        Returns:
            float: Total energy consumed in kWh
        """
        total_energy = 0.0
        
        for pm in pms:
            energy = self.compute_pm_energy(pm)
            total_energy += energy
        
        # Update statistics
        self.total_energy_consumed += total_energy
        self.energy_history.append(total_energy)
        
        return total_energy
    
    def get_power_breakdown(self, pm):
        """
        Get detailed power breakdown for a PM.
        
        Args:
            pm (PhysicalMachine): PM object
            
        Returns:
            dict: Power breakdown in watts
        """
        if not pm.is_on:
            return {
                'static': 0.0,
                'cpu': 0.0,
                'memory': 0.0,
                'network': 0.0,
                'total': 0.0
            }
        
        P_static = self.static_power
        P_cpu = self.compute_cpu_power(pm.cpu_utilization())
        P_mem = self.compute_memory_power(pm.memory_utilization())
        P_net = self.compute_network_power(pm.network_utilization())
        P_total = P_static + P_cpu + P_mem + P_net
        
        return {
            'static': P_static,
            'cpu': P_cpu,
            'memory': P_mem,
            'network': P_net,
            'total': P_total
        }
    
    def get_energy_statistics(self):
        """
        Get energy consumption statistics.
        
        Returns:
            dict: Energy statistics
        """
        return {
            'total_energy_kwh': self.total_energy_consumed,
            'total_energy_joules': self.total_energy_consumed * 3.6e6,  # kWh to J
            'num_slots_tracked': len(self.energy_history),
            'avg_energy_per_slot': (self.total_energy_consumed / len(self.energy_history) 
                                   if self.energy_history else 0.0),
            'peak_energy_slot': max(self.energy_history) if self.energy_history else 0.0,
            'min_energy_slot': min(self.energy_history) if self.energy_history else 0.0
        }
    
    def reset_statistics(self):
        """Reset energy tracking statistics."""
        self.total_energy_consumed = 0.0
        self.energy_history = []
    
    def estimate_cost(self, price_per_kwh=0.12):
        """
        Estimate electricity cost based on energy consumption.
        
        Args:
            price_per_kwh (float): Price per kWh in USD (default: $0.12)
            
        Returns:
            float: Estimated cost in USD
        """
        return self.total_energy_consumed * price_per_kwh
    
    def __repr__(self):
        """String representation."""
        return (f"EnergyModel(α₁={self.alpha_1}, α₂={self.alpha_2}, "
                f"total_energy={self.total_energy_consumed:.4f} kWh)")


# Convenience function for quick energy calculation
def calculate_energy(pm, tau=TIME_SLOT_DURATION):
    """
    Quick function to calculate energy for a PM.
    
    Args:
        pm (PhysicalMachine): PM object
        tau (int): Time slot duration in seconds
        
    Returns:
        float: Energy consumed in kWh
    """
    model = EnergyModel()
    return model.compute_pm_energy(pm)
=======
a
>>>>>>> 3fdc487e76549e8239d83e40c86355f2a7360963
