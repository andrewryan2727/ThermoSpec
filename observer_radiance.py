import numpy as np
from rte_disort import DisortRTESolver
from config import SimulationConfig
from grid import LayerGrid

class ObserverRadianceCalculator:
    """
    Calculate summed crater radiance as seen by observers at different geometries.
    Creates DISORT instances dynamically for each facet with correct viewing angles.
    """
    
    def __init__(self, cfg: SimulationConfig, grid: LayerGrid, observer_vectors):
        """
        Store configuration and observer vectors for dynamic DISORT creation.
        
        Args:
            cfg: Simulation configuration
            grid: Spatial grid
            observer_vectors: list of [x,y,z] observer direction vectors
        """
        self.cfg = cfg
        self.grid = grid
        self.observer_vectors = [np.array(vec) / np.linalg.norm(vec) for vec in observer_vectors]
        
        # Store observers without pre-creating DISORT instances
        self.observers = []
        for obs_vec in self.observer_vectors:
            self.observers.append({'vector': obs_vec})
    
    def compute_facet_visibility(self, crater_shadowtester, observer_vec):
        """
        Use existing ShadowTester to determine facet visibility from observer.
        
        Args:
            crater_shadowtester: ShadowTester instance from crater module
            observer_vec: normalized observer direction vector [x,y,z]
            
        Returns:
            visibility: fractional visibility (0-1) for each facet
        """
        # Use the existing illuminated_facets method with observer vector instead of sun vector
        return crater_shadowtester.illuminated_facets(observer_vec)
    
    def compute_facet_observer_angles(self, crater_mesh, observer_vec):
        """
        Calculate local viewing angles for each facet relative to observer.
        Uses precomputed local coordinate system from crater_mesh for consistency.
        
        Args:
            crater_mesh: CraterMesh object with precomputed tangent vectors
            observer_vec: normalized observer direction vector
            
        Returns:
            facet_mu: cosine of local viewing angle for each facet
            facet_phi: local azimuth angle for each facet (in facet reference frame)
        """
        # Dot product for mu
        facet_mu = np.dot(crater_mesh.normals, observer_vec)
        facet_mu[facet_mu < 0] = 0.0
        
        # Use precomputed tangent vectors for phi calculation
        facet_phi = np.zeros(len(crater_mesh.normals))
        
        for i in range(len(crater_mesh.normals)):
            if facet_mu[i] <= 0:
                continue
                
            # Project observer into facet plane
            obs_in_plane = observer_vec - facet_mu[i] * crater_mesh.normals[i]
            obs_in_plane_norm = np.linalg.norm(obs_in_plane)
            
            if obs_in_plane_norm > 1e-10:
                obs_in_plane = obs_in_plane / obs_in_plane_norm
                cos_phi = np.dot(obs_in_plane, crater_mesh.tangent1[i])
                sin_phi = np.dot(obs_in_plane, crater_mesh.tangent2[i])
                facet_phi[i] = np.arctan2(sin_phi, cos_phi)
                if facet_phi[i] < 0:
                    facet_phi[i] += 2 * np.pi
            else:
                # Observer is along the normal direction, phi is arbitrary
                facet_phi[i] = 0.0
        
        return facet_mu, facet_phi
    
    def create_disort_for_facet(self, mu, phi):
        """
        Create DISORT instances for a specific viewing geometry.
        
        Args:
            mu: cosine of viewing angle
            phi: azimuth angle (radians)
            
        Returns:
            tuple: (disort_thermal, disort_vis) instances
        """
        disort_thermal = DisortRTESolver(self.cfg, self.grid, n_cols=1,
                                       output_radiance=True, planck=True,
                                       observer_mu=mu, observer_phi=phi)
        
        disort_vis = None
        if not self.cfg.multi_wave:
            disort_vis = DisortRTESolver(self.cfg, self.grid, n_cols=1,
                                       output_radiance=True, planck=False,
                                       observer_mu=mu, observer_phi=phi)
        
        return disort_thermal, disort_vis
    
    def compute_crater_radiance(self, T_crater_facets, crater_mesh, crater_shadowtester, 
                               observer_idx, mu_sun=0.0, F_sun=0.0):
        """
        Calculate total crater radiance as seen by a specific observer.
        Creates DISORT instances dynamically for each facet with correct viewing angles.
        
        Args:
            T_crater_facets: temperature profiles for all facets [depth, n_facets]
            crater_mesh: CraterMesh object
            crater_shadowtester: ShadowTester object 
            observer_idx: index of observer
            mu_sun: solar cosine (for scattered light calculation)
            F_sun: solar illumination flag
            
        Returns:
            total_radiance: area-weighted summed radiance from all visible facets
                          - For multi_wave: array of shape [n_waves]
                          - For two-wave: scalar value
        """
        observer_vec = self.observers[observer_idx]['vector']
        
        # Get fractional visibility for each facet using existing ShadowTester
        visibility = self.compute_facet_visibility(crater_shadowtester, observer_vec)
        
        # Get local viewing angles for each facet
        facet_mu, facet_phi = self.compute_facet_observer_angles(crater_mesh, observer_vec)
        
        # Initialize radiance arrays based on multi_wave mode
        if self.cfg.multi_wave:
            # We need to get n_waves by creating a temporary DISORT instance
            temp_disort = DisortRTESolver(self.cfg, self.grid, n_cols=1, output_radiance=True, planck=True)
            n_waves = len(temp_disort.wavenumbers)
            total_radiance = np.zeros(n_waves)
        else:
            total_radiance = 0.0
            
        total_projected_area = 0.0
        
        for i in range(len(crater_mesh.normals)):
            # Skip facets that are not visible or facing away from observer
            if visibility[i] <= 0 or facet_mu[i] <= 0:
                continue
                
            # Temperature profile for this facet
            T_facet = T_crater_facets[:, i]
            
            # Create DISORT instances for this facet's local viewing angles
            try:
                disort_thermal, disort_vis = self.create_disort_for_facet(facet_mu[i], facet_phi[i])
                
                # Calculate radiance from this facet using facet-specific DISORT
                if self.cfg.multi_wave:
                    # Multi-wave case: return full spectral radiance
                    radiance = disort_thermal.disort_run(T_facet, mu_sun, F_sun)
                    if hasattr(radiance, 'numpy'):
                        radiance = radiance.numpy()
                    # radiance should be shape [n_waves] or [n_waves, 1]
                    if radiance.ndim > 1:
                        facet_radiance = radiance[:, 0]  # Take first column if 2D
                    else:
                        facet_radiance = radiance
                else:
                    # Two-wave case: thermal + visible, return single value
                    rad_thermal = disort_thermal.disort_run(T_facet, mu_sun, F_sun)
                    rad_vis = disort_vis.disort_run(T_facet, mu_sun, F_sun)
                    
                    if hasattr(rad_thermal, 'numpy'):
                        rad_thermal = rad_thermal.numpy()
                    if hasattr(rad_vis, 'numpy'):
                        rad_vis = rad_vis.numpy()
                        
                    # Ensure scalar values
                    rad_thermal = rad_thermal.item() if hasattr(rad_thermal, 'item') else rad_thermal
                    rad_vis = rad_vis.item() if hasattr(rad_vis, 'item') else rad_vis
                    facet_radiance = rad_thermal + rad_vis
                    
            except Exception as e:
                print(f"Warning: DISORT failed for facet {i} (mu={facet_mu[i]:.3f}, phi={facet_phi[i]:.3f}): {e}")
                if self.cfg.multi_wave:
                    facet_radiance = np.zeros(n_waves)
                else:
                    facet_radiance = 0.0
            
            # Weight by facet area, projected area (cosine factor), and visibility
            facet_area = crater_mesh.areas[i]
            projected_area = facet_area * facet_mu[i] * visibility[i]
            
            total_radiance += facet_radiance * projected_area
            total_projected_area += projected_area
        
        # Return area-averaged radiance
        if total_projected_area > 0:
            return total_radiance / total_projected_area
        else:
            if self.cfg.multi_wave:
                return np.zeros(n_waves)
            else:
                return 0.0
    
    def compute_all_observers(self, T_crater_facets, crater_mesh, crater_shadowtester,
                            mu_sun=0.0, F_sun=0.0):
        """
        Calculate crater radiance for all observers.
        
        Args:
            T_crater_facets: temperature profiles [depth, n_facets]
            crater_mesh: CraterMesh object
            crater_shadowtester: ShadowTester object
            mu_sun: solar cosine
            F_sun: solar illumination flag
            
        Returns:
            radiances: array of radiance for each observer
                      - For multi_wave: shape [n_observers, n_waves]  
                      - For two-wave: shape [n_observers]
        """
        if self.cfg.multi_wave:
            # Get number of wavelengths by creating a temporary DISORT instance
            temp_disort = DisortRTESolver(self.cfg, self.grid, n_cols=1, output_radiance=True, planck=True)
            n_waves = len(temp_disort.wavenumbers)
            radiances = np.zeros((len(self.observers), n_waves))
        else:
            radiances = np.zeros(len(self.observers))
        
        for i in range(len(self.observers)):
            radiances[i] = self.compute_crater_radiance(
                T_crater_facets, crater_mesh, crater_shadowtester, i, mu_sun, F_sun
            )
            
        return radiances