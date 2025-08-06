"""
Diurnal convergence checking for thermal model simulations.

This module implements two approaches to check for diurnal convergence:
1. Temperature Change Method (Kieffer 2013, section 3.2.6)
2. Energy Balance Method (Rozitis & Green 2011)

Both methods can be used independently or in combination to automatically
terminate diurnal simulations when thermal equilibrium is achieved.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from config import SimulationConfig


@dataclass
class CycleData:
    """Data container for a single diurnal cycle."""
    cycle_number: int
    temperatures: np.ndarray  # Surface temperatures throughout cycle
    times: np.ndarray  # Time points for temperature data
    energy_in: float  # Total absorbed solar energy for the cycle
    energy_out: float  # Total emitted thermal energy for the cycle
    local_solar_times: np.ndarray  # Local solar time for each temperature point


class DiurnalConvergenceChecker(ABC):
    """
    Base class for diurnal convergence checking.
    
    Provides common functionality for tracking simulation state across
    diurnal cycles and determining when thermal equilibrium is achieved.
    """
    
    def __init__(self, config: SimulationConfig):
        """
        Initialize convergence checker with simulation configuration.
        
        Args:
            config: Simulation configuration containing convergence parameters
        """
        self.cfg = config
        self.cycle_history: List[CycleData] = []
        self.current_cycle_number = 0
        self.converged = False
        self.convergence_message = ""
        
    def add_cycle_data(self, cycle_data: CycleData) -> None:
        """
        Add completed cycle data to history.
        
        Args:
            cycle_data: Data from a completed diurnal cycle
        """
        self.cycle_history.append(cycle_data)
        self.current_cycle_number += 1
        
    def should_check_convergence(self) -> bool:
        """
        Determine if convergence should be checked at this cycle.
        
        Returns:
            True if convergence checking should be performed
        """
        if len(self.cycle_history) < self.cfg.diurnal_convergence_min_cycles:
            return False
            
        return (self.current_cycle_number % self.cfg.diurnal_convergence_check_interval) == 0
    
    @abstractmethod
    def check_convergence(self) -> bool:
        """
        Check if convergence has been achieved.
        
        Returns:
            True if the simulation has converged
        """
        pass
    
    def get_convergence_diagnostics(self) -> Dict[str, Any]:
        """
        Get diagnostic information about convergence status.
        
        Returns:
            Dictionary containing convergence diagnostics
        """
        return {
            'method': self.__class__.__name__,
            'cycles_completed': len(self.cycle_history),
            'converged': self.converged,
            'convergence_message': self.convergence_message,
            'config_tolerance': self._get_tolerance_info()
        }
    
    @abstractmethod
    def _get_tolerance_info(self) -> Dict[str, float]:
        """Return tolerance information specific to the checker type."""
        pass


class TemperatureChangeChecker(DiurnalConvergenceChecker):
    """
    Temperature-based convergence checker (Kieffer 2013 method).
    
    Compares surface temperatures at equivalent local solar times between
    consecutive diurnal cycles. Convergence is achieved when the maximum
    absolute temperature difference falls below the specified tolerance.
    """
    
    def __init__(self, config: SimulationConfig):
        """Initialize temperature-based convergence checker."""
        super().__init__(config)
        self.tolerance = config.diurnal_convergence_temp_tol
        self.reference_local_times = np.array([0.0, 6.0, 12.0, 18.0, 24.0])  # Dawn, morning, noon, evening, midnight (hours)
        
    def check_convergence(self) -> bool:
        """
        Check convergence based on temperature changes between cycles.
        
        Compares temperatures at reference local solar times between
        the current cycle and previous cycles in the averaging window.
        
        Returns:
            True if temperature changes are below tolerance
        """
        if not self.should_check_convergence():
            return False
            
        n_cycles = len(self.cycle_history)
        window_size = min(self.cfg.diurnal_convergence_window, n_cycles - 1)
        
        if window_size < 1:
            return False
            
        # Get the most recent cycle
        current_cycle = self.cycle_history[-1]
        
        # Interpolate current cycle temperatures to reference local times
        current_temps = self._interpolate_to_reference_times(current_cycle)
        
        max_temp_difference = 0.0
        
        # Compare with previous cycles in the window
        for i in range(1, window_size + 1):
            previous_cycle = self.cycle_history[-(i + 1)]
            previous_temps = self._interpolate_to_reference_times(previous_cycle)
            
            # Calculate maximum absolute temperature difference
            temp_differences = np.abs(current_temps - previous_temps)
            cycle_max_diff = np.max(temp_differences)
            max_temp_difference = max(max_temp_difference, cycle_max_diff)
        
        # Check convergence
        print(f"Max temperature difference over last {window_size} cycles: {max_temp_difference:.6f} K")
        converged = max_temp_difference < self.tolerance
        
        if converged:
            self.converged = True
            self.convergence_message = (
                f"Temperature convergence achieved at cycle {self.current_cycle_number}. "
                f"Max temperature difference: {max_temp_difference:.6f} K "
                f"(tolerance: {self.tolerance:.6f} K)"
            )
        
        return converged
    
    def _interpolate_to_reference_times(self, cycle_data: CycleData) -> np.ndarray:
        """
        Interpolate cycle temperatures to reference local solar times.
        
        Args:
            cycle_data: Cycle data containing temperatures and local solar times
            
        Returns:
            Interpolated temperatures at reference times
        """
        # Handle wrap-around for local solar time (0-24 hours)
        local_times = cycle_data.local_solar_times % 24.0
        temperatures = cycle_data.temperatures
        
        # Sort by local time for interpolation
        sort_indices = np.argsort(local_times)
        local_times_sorted = local_times[sort_indices]
        temperatures_sorted = temperatures[sort_indices]
        
        # Add wrap-around points for smooth interpolation
        if local_times_sorted[0] > 0.1:  # Add point at beginning
            local_times_sorted = np.concatenate([[local_times_sorted[-1] - 24.0], local_times_sorted])
            temperatures_sorted = np.concatenate([[temperatures_sorted[-1]], temperatures_sorted])
        
        if local_times_sorted[-1] < 23.9:  # Add point at end
            local_times_sorted = np.concatenate([local_times_sorted, [local_times_sorted[0] + 24.0]])
            temperatures_sorted = np.concatenate([temperatures_sorted, [temperatures_sorted[0]]])
        
        # Interpolate to reference times
        interpolated_temps = np.interp(self.reference_local_times, local_times_sorted, temperatures_sorted)
        
        return interpolated_temps
    
    def _get_tolerance_info(self) -> Dict[str, float]:
        """Return temperature convergence tolerance information."""
        return {'temperature_tolerance_K': self.tolerance}


class EnergyBalanceChecker(DiurnalConvergenceChecker):
    """
    Energy balance convergence checker (Rozitis & Green 2011 method).
    
    Compares absorbed solar energy with emitted thermal energy for each
    diurnal cycle. Convergence is achieved when the relative energy imbalance
    falls below the specified tolerance.
    """
    
    def __init__(self, config: SimulationConfig):
        """Initialize energy balance convergence checker."""
        super().__init__(config)
        self.tolerance = config.diurnal_convergence_energy_tol
        
    def check_convergence(self) -> bool:
        """
        Check convergence based on energy balance.
        
        Compares energy input (absorbed solar) with energy output (thermal emission)
        for recent cycles. Convergence is achieved when the relative energy
        imbalance is below the tolerance.
        
        Returns:
            True if energy balance is within tolerance
        """
        if not self.should_check_convergence():
            return False
            
        n_cycles = len(self.cycle_history)
        window_size = min(self.cfg.diurnal_convergence_window, n_cycles)
        
        if window_size < 1:
            return False
            
        # Calculate average energy balance over the window
        #recent_cycles = self.cycle_history[-window_size:]
        
        #total_energy_in = sum(cycle.energy_in for cycle in recent_cycles)
        #total_energy_out = sum(cycle.energy_out for cycle in recent_cycles)
        #Get energy in and out from most recent cycle
        total_energy_in = self.cycle_history[-1].energy_in
        total_energy_out = self.cycle_history[-1].energy_out
        print(f"Total energy in and out over last cycles: {total_energy_in:.6f} J, {total_energy_out:.6f} J")
        
        # Calculate relative energy imbalance
        energy_imbalance = abs(total_energy_in - total_energy_out) / max(total_energy_in, total_energy_out)

            
        # Check convergence
        converged = energy_imbalance < self.tolerance
        
        if converged:
            self.converged = True
            self.convergence_message = (
                f"Energy balance convergence achieved at cycle {self.current_cycle_number}. "
                f"Relative energy imbalance: {energy_imbalance:.6f} "
                f"(tolerance: {self.tolerance:.6f})"
            )
        
        return converged
    
    def _get_tolerance_info(self) -> Dict[str, float]:
        """Return energy balance tolerance information."""
        return {'energy_balance_tolerance': self.tolerance}


class HybridChecker(DiurnalConvergenceChecker):
    """
    Hybrid convergence checker combining temperature and energy methods.
    
    Uses both temperature change and energy balance criteria. Convergence
    can be achieved when either both criteria are met (AND logic) or when
    either criterion is met (OR logic), depending on configuration.
    """
    
    def __init__(self, config: SimulationConfig, require_both: bool = True):
        """
        Initialize hybrid convergence checker.
        
        Args:
            config: Simulation configuration
            require_both: If True, both temperature and energy criteria must be met.
                         If False, either criterion is sufficient.
        """
        super().__init__(config)
        self.temperature_checker = TemperatureChangeChecker(config)
        self.energy_checker = EnergyBalanceChecker(config)
        self.require_both = require_both
        
    def add_cycle_data(self, cycle_data: CycleData) -> None:
        """Add cycle data to both constituent checkers."""
        super().add_cycle_data(cycle_data)
        self.temperature_checker.add_cycle_data(cycle_data)
        self.energy_checker.add_cycle_data(cycle_data)
        
    def check_convergence(self) -> bool:
        """
        Check convergence using both temperature and energy criteria.
        
        Returns:
            True if convergence criteria are met (based on require_both setting)
        """
        if not self.should_check_convergence():
            return False
            
        temp_converged = self.temperature_checker.check_convergence()
        energy_converged = self.energy_checker.check_convergence()
        
        if self.require_both:
            converged = temp_converged and energy_converged
            criteria_text = "both temperature AND energy"
        else:
            converged = temp_converged or energy_converged
            criteria_text = "either temperature OR energy"
            
        if converged:
            self.converged = True
            self.convergence_message = (
                f"Hybrid convergence achieved at cycle {self.current_cycle_number} "
                f"using {criteria_text} criteria. "
                f"Temperature converged: {temp_converged}, Energy converged: {energy_converged}"
            )
            
        return converged
    
    def _get_tolerance_info(self) -> Dict[str, Any]:
        """Return tolerance information for both methods."""
        return {
            'temperature_tolerance_K': self.temperature_checker.tolerance,
            'energy_balance_tolerance': self.energy_checker.tolerance,
            'require_both_criteria': self.require_both
        }


def create_convergence_checker(config: SimulationConfig) -> Optional[DiurnalConvergenceChecker]:
    """
    Factory function to create appropriate convergence checker based on configuration.
    
    Args:
        config: Simulation configuration
        
    Returns:
        Convergence checker instance or None if convergence checking is disabled
    """
    if not config.enable_diurnal_convergence:
        return None
        
    method = config.diurnal_convergence_method.lower()
    
    if method == 'temperature':
        return TemperatureChangeChecker(config)
    elif method == 'energy':
        return EnergyBalanceChecker(config)
    elif method == 'both':
        return HybridChecker(config, require_both=True)
    elif method == 'either':
        return HybridChecker(config, require_both=False)
    else:
        raise ValueError(f"Unknown diurnal convergence method: {method}. "
                        f"Valid options: 'temperature', 'energy', 'both', 'either'")