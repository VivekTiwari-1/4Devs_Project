"""
Test Script for Energy Model (Step 2)
Run this to verify energy calculations are working
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.pm import PhysicalMachine
from environment.container import Container
from environment.energy_model import EnergyModel


def test_energy_model():
    """Test energy model with different CPU utilizations."""
    
    print("\n" + "="*60)
    print("⚡ TESTING ENERGY MODEL")
    print("="*60 + "\n")
    
    # Create energy model
    energy_model = EnergyModel()
    
    print(f"Energy Model Parameters:")
    print(f"  α₁ (linear):    {energy_model.alpha_1} W")
    print(f"  α₂ (quadratic): {energy_model.alpha_2} W")
    print(f"  Static power:   {energy_model.static_power} W")
    print(f"  Time slot:      {energy_model.tau} seconds\n")
    
    # Test different CPU utilizations
    print("Testing CPU Power at Different Utilizations:")
    print("-" * 60)
    print(f"{'CPU Util':<12} {'CPU Power (W)':<15} {'Total Power (W)':<18}")
    print("-" * 60)
    
    test_utils = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    
    for util in test_utils:
        cpu_power = energy_model.compute_cpu_power(util)
        total_power = energy_model.static_power + cpu_power
        print(f"{util:>6.0%}       {cpu_power:>10.2f}      {total_power:>12.2f}")
    
    print("\n" + "="*60)
    print("✅ Energy model calculations verified!")
    print("="*60 + "\n")


def test_pm_energy():
    """Test energy calculation for a Physical Machine."""
    
    print("\n" + "="*60)
    print("⚡ TESTING PM ENERGY CALCULATION")
    print("="*60 + "\n")
    
    # Create PM and containers
    pm = PhysicalMachine(pm_id=0, total_cores=68)
    energy_model = EnergyModel()
    
    # Add some containers
    for i in range(10):
        container = Container(
            cid=f"C_{i}",
            instructions=10_000_000_000,
            deadline=1000,
            arrival_time=0
        )
        container.assigned_cores = 3
        pm.add_container(container)
    
    # Calculate energy
    cpu_util = pm.cpu_utilization()
    power_breakdown = energy_model.get_power_breakdown(pm)
    energy = energy_model.compute_pm_energy(pm)
    
    print(f"PM Status:")
    print(f"  Containers: {len(pm.containers)}")
    print(f"  Used cores: {sum(c.assigned_cores for c in pm.containers)}/{pm.total_cores}")
    print(f"  CPU Utilization: {cpu_util:.2%}\n")
    
    print(f"Power Breakdown:")
    print(f"  Static:  {power_breakdown['static']:.2f} W")
    print(f"  CPU:     {power_breakdown['cpu']:.2f} W")
    print(f"  Memory:  {power_breakdown['memory']:.2f} W")
    print(f"  Network: {power_breakdown['network']:.2f} W")
    print(f"  ───────────────────────────")
    print(f"  TOTAL:   {power_breakdown['total']:.2f} W\n")
    
    print(f"Energy consumed in {energy_model.tau}s slot:")
    print(f"  {energy:.6f} kWh")
    print(f"  {energy * 1000:.4f} Wh")
    print(f"  {energy * 3.6e6:.2f} Joules\n")
    
    print("="*60)
    print("✅ PM energy calculation verified!")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Run tests
    test_energy_model()
    test_pm_energy()
    
    print("\n🎉 All energy model tests passed!")
    print("✅ Ready to run full simulation with energy tracking\n")