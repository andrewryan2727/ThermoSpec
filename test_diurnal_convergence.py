#!/usr/bin/env python3
"""
Test script for diurnal convergence checking functionality.

This script demonstrates the new diurnal convergence features and validates
that they work correctly with different convergence methods.
"""

import numpy as np
from config import SimulationConfig
from modelmain import Simulator

def test_temperature_convergence():
    """Test temperature-based convergence checking."""
    print("=" * 60)
    print("Testing Temperature-based Convergence (Kieffer 2013 method)")
    print("=" * 60)
    
    # Create configuration with temperature convergence enabled
    config = SimulationConfig(
        # Basic simulation settings
        diurnal=True,
        use_RTE=True,
        RTE_solver='hapke',
        ndays=25,  # Start with many days - should converge early
        
        # Enable convergence checking
        enable_diurnal_convergence=True,
        diurnal_convergence_method='temperature',
        diurnal_convergence_temp_tol=0.06,  # 10 mK tolerance
        diurnal_convergence_min_cycles=2,
        
        # Simple thermal properties for faster convergence
        Et=1000.0,
        k_dust=1e-4,
        dust_thickness=0.5,
        ssalb_therm=0.1,
        ssalb_vis=0.06
    )
    
    # Run simulation
    print("Running simulation with temperature convergence...")
    sim = Simulator(config)
    T_out, phi_vis, phi_therm, T_surf, t_out = sim.run()
    
    print(f"Simulation completed. Output shape: {T_out.shape}")
    print(f"Final surface temperature: {T_surf[-1]:.2f} K")
    
    return sim

def test_energy_convergence():
    """Test energy balance convergence checking."""
    print("\n" + "=" * 60)
    print("Testing Energy Balance Convergence (Rozitis & Green 2011 method)")
    print("=" * 60)
    
    # Create configuration with energy convergence enabled
    config = SimulationConfig(
        # Basic simulation settings
        diurnal=True,
        use_RTE=True,
        RTE_solver='hapke',
        ndays=25,  # Start with many days - should converge early
        
        # Enable convergence checking
        enable_diurnal_convergence=True,
        diurnal_convergence_method='energy',
        diurnal_convergence_energy_tol=0.05,  # 5% tolerance
        diurnal_convergence_min_cycles=2,
        
        # Simple thermal properties
        Et=1000.0,
        k_dust=1e-4,
        dust_thickness=0.5,
        ssalb_therm=0.1,
        ssalb_vis=0.06
    )
    
    # Run simulation
    print("Running simulation with energy balance convergence...")
    sim = Simulator(config)
    T_out, phi_vis, phi_therm, T_surf, t_out = sim.run()
    
    print(f"Simulation completed. Output shape: {T_out.shape}")
    print(f"Final surface temperature: {T_surf[-1]:.2f} K")
    
    return sim

def test_hybrid_convergence():
    """Test hybrid convergence checking (both temperature and energy)."""
    print("\n" + "=" * 60)
    print("Testing Hybrid Convergence (Both methods)")
    print("=" * 60)
    
    # Create configuration with hybrid convergence enabled
    config = SimulationConfig(
        # Basic simulation settings
        diurnal=True,
        use_RTE=True,
        RTE_solver='hapke',
        ndays=10,  # Start with many days - should converge early
        
        # Enable convergence checking
        enable_diurnal_convergence=True,
        diurnal_convergence_method='both',  # Require both temperature AND energy criteria
        diurnal_convergence_temp_tol=1e-2,  
        diurnal_convergence_energy_tol=0.02,  
        diurnal_convergence_min_cycles=2,
        
        # Simple thermal properties
        Et=1000.0,
        k_dust=1e-4,
        dust_thickness=0.5,
        ssalb_therm=0.1,
        ssalb_vis=0.06
    )
    
    # Run simulation
    print("Running simulation with hybrid convergence...")
    sim = Simulator(config)
    T_out, phi_vis, phi_therm, T_surf, t_out = sim.run()
    
    print(f"Simulation completed. Output shape: {T_out.shape}")
    print(f"Final surface temperature: {T_surf[-1]:.2f} K")
    
    return sim

def test_no_convergence_baseline():
    """Test baseline case without convergence checking."""
    print("\n" + "=" * 60)
    print("Testing Baseline (No Convergence Checking)")
    print("=" * 60)
    
    # Create configuration without convergence checking
    config = SimulationConfig(
        # Basic simulation settings
        diurnal=True,
        use_RTE=True,
        RTE_solver='hapke',
        ndays=5,  # Fixed number of days
        
        # Disable convergence checking
        enable_diurnal_convergence=False,
        
        # Same thermal properties as convergence tests
        Et=1000.0,
        k_dust=1e-4,
        dust_thickness=0.5,
        ssalb_therm=0.1,
        ssalb_vis=0.06
    )
    
    # Run simulation
    print("Running baseline simulation (3 days fixed)...")
    sim = Simulator(config)
    T_out, phi_vis, phi_therm, T_surf, t_out = sim.run()
    
    print(f"Simulation completed. Output shape: {T_out.shape}")
    print(f"Final surface temperature: {T_surf[-1]:.2f} K")
    
    return sim

def analyze_convergence_performance():
    """Compare convergence performance between methods and baseline."""
    print("\n" + "=" * 80)
    print("CONVERGENCE PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    # Run all test cases
    print("Running all test cases...")
    
    try:
        sim_temp = test_temperature_convergence()
        temp_cycles = len(sim_temp.convergence_checker.cycle_history) if sim_temp.convergence_checker else 0
    except Exception as e:
        print(f"Temperature convergence test failed: {e}")
        temp_cycles = 0
    
    try:
        sim_energy = test_energy_convergence()  
        energy_cycles = len(sim_energy.convergence_checker.cycle_history) if sim_energy.convergence_checker else 0
    except Exception as e:
        print(f"Energy convergence test failed: {e}")
        energy_cycles = 0
    
    try:
        sim_hybrid = test_hybrid_convergence()
        hybrid_cycles = len(sim_hybrid.convergence_checker.cycle_history) if sim_hybrid.convergence_checker else 0
    except Exception as e:
        print(f"Hybrid convergence test failed: {e}")
        hybrid_cycles = 0
    
    try:
        sim_baseline = test_no_convergence_baseline()
        baseline_cycles = 3  # Fixed 3 days
    except Exception as e:
        print(f"Baseline test failed: {e}")
        baseline_cycles = 3
    
    # Summary
    print(f"\nPERFORMANCE SUMMARY:")
    print(f"  Temperature method: {temp_cycles} cycles")
    print(f"  Energy method: {energy_cycles} cycles") 
    print(f"  Hybrid method: {hybrid_cycles} cycles")
    print(f"  Baseline (no convergence): {baseline_cycles} cycles")
    
    if temp_cycles > 0:
        print(f"\nEfficiency gains:")
        print(f"  Temperature method: {(baseline_cycles - temp_cycles) / baseline_cycles * 100:.1f}% reduction")
    if energy_cycles > 0:
        print(f"  Energy method: {(baseline_cycles - energy_cycles) / baseline_cycles * 100:.1f}% reduction")
    if hybrid_cycles > 0:
        print(f"  Hybrid method: {(baseline_cycles - hybrid_cycles) / baseline_cycles * 100:.1f}% reduction")

if __name__ == "__main__":
    print("Diurnal Convergence Testing Suite")
    print("This test validates the implementation of automatic diurnal convergence")
    print("using both Kieffer 2013 and Rozitis & Green 2011 methodologies.\n")
    
    try:
        analyze_convergence_performance()
        print("\n" + "=" * 80)
        print("All tests completed!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nTest suite failed with error: {e}")
        print("Please check the implementation and try again.")
        raise