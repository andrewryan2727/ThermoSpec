"""
DEPRECATED: This module has been moved to radiance_processor.py

All crater radiance functionality has been consolidated into the unified radiance processing system.
Please update your imports to use radiance_processor instead.

This file provides backward compatibility imports but will be removed in a future version.
"""

import warnings
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union

# Issue deprecation warning
warnings.warn(
    "observer_radiance.py is deprecated and will be removed in a future version. "
    "Use radiance_processor module instead:\n\n"
    "OLD: from observer_radiance import ObserverRadianceCalculator\n"
    "NEW: from radiance_processor import CraterRadianceProcessor\n\n"
    "Or use the high-level functions:\n"
    "- calculate_crater_radiance_from_sim()\n"
    "- calculate_dual_radiance_from_sim()\n"
    "- calculate_radiances_from_results() with surface_type='crater'",
    DeprecationWarning,
    stacklevel=2
)

# Import the new classes for backward compatibility
try:
    from radiance_processor import CraterRadianceProcessor as ObserverRadianceCalculator
    from radiance_processor import (
        calculate_crater_radiance_from_sim,
        calculate_dual_radiance_from_sim, 
        extract_crater_data_from_sim,
        calculate_radiances_from_results
    )

    # Make sure the backward compatible class has the same interface
    class ObserverRadianceCalculator(CraterRadianceProcessor):
        """
        DEPRECATED: Use CraterRadianceProcessor from radiance_processor module instead.

        This class provides backward compatibility but will be removed in a future version.
        """

        def __init__(self, cfg, grid, observer_vectors):
            # Issue another deprecation warning when instantiated
            warnings.warn(
                "ObserverRadianceCalculator is deprecated. Use CraterRadianceProcessor from radiance_processor instead.",
                DeprecationWarning,
                stacklevel=2
            )
            # Call parent constructor with renamed parameters to match new interface
            super().__init__(config=cfg, grid=grid, observer_vectors=observer_vectors)

            # Store old parameter names for backward compatibility
            self.cfg = cfg

except ImportError as e:
    # Fallback if there are import issues
    warnings.warn(
        f"Could not import from radiance_processor: {e}. "
        "Please ensure radiance_processor.py is available and properly configured.",
        ImportWarning,
        stacklevel=2
    )
    
    # Define a placeholder class that raises an informative error
    class ObserverRadianceCalculator:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "ObserverRadianceCalculator is deprecated and the new radiance_processor module "
                "could not be imported. Please check your installation and update your code to use "
                "the new radiance processing interface."
            )
    
    # Define placeholder functions
    def calculate_crater_radiance_from_sim(*args, **kwargs):
        raise ImportError("radiance_processor module could not be imported")
        
    def calculate_dual_radiance_from_sim(*args, **kwargs):
        raise ImportError("radiance_processor module could not be imported")
        
    def extract_crater_data_from_sim(*args, **kwargs):
        raise ImportError("radiance_processor module could not be imported")
        
    def calculate_radiances_from_results(*args, **kwargs):
        raise ImportError("radiance_processor module could not be imported")


# Additional backward compatibility aliases
ObserverRadiance = ObserverRadianceCalculator  # Alternative name that might have been used


def migration_guide():
    """
    Print a migration guide for updating code from observer_radiance to radiance_processor.
    """
    print("""
MIGRATION GUIDE: observer_radiance.py → radiance_processor.py
================================================================

The crater radiance functionality has been moved to a unified radiance processing system.

OLD CODE:
---------
from observer_radiance import ObserverRadianceCalculator

# Create calculator
calc = ObserverRadianceCalculator(cfg, grid, observer_vectors)

# Calculate crater radiance
radiances = calc.compute_all_observers(
    T_crater_facets, crater_mesh, crater_shadowtester, 
    mu_sun, F_sun, sun_vec, crater_radtrans, 
    therm_flux_facets, illuminated, albedo, emissivity
)

NEW CODE (Option 1 - High-level functions):
-------------------------------------------
from radiance_processor import calculate_crater_radiance_from_sim

# Calculate directly from simulation object
crater_results = calculate_crater_radiance_from_sim(
    sim, observer_vectors, spectral_mode='hybrid'
)

NEW CODE (Option 2 - Low-level interface):
------------------------------------------
from radiance_processor import CraterRadianceProcessor

# Create processor (note: parameter names changed)
processor = CraterRadianceProcessor(config=cfg, grid=grid, observer_vectors=observer_vectors)

# Calculate crater radiance (same interface)
radiances = processor.compute_all_observers(
    T_crater_facets, crater_mesh, crater_shadowtester,
    mu_sun, F_sun, sun_vec, crater_radtrans,
    therm_flux_facets, illuminated, albedo, emissivity
)

NEW CODE (Option 3 - Unified interface with dual outputs):
----------------------------------------------------------
from radiance_processor import calculate_radiances_from_results

# Calculate both smooth and crater radiance
dual_results = calculate_radiances_from_results(
    thermal_results, config,
    surface_type='both',
    observer_angles=[0, 30, 60],  # For smooth surface
    observer_vectors=[[0,0,1], [0.5,0,1]],  # For crater
    crater_data=crater_data
)

BENEFITS OF NEW SYSTEM:
-----------------------
✓ Unified interface for all radiance calculations
✓ Support for dual outputs (crater + smooth comparison)  
✓ Better integration with post-processing workflows
✓ Consistent spectral mode handling
✓ Enhanced documentation and examples

""")


if __name__ == "__main__":
    migration_guide()