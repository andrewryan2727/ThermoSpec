import numpy as np
import trimesh
from scipy.sparse import csr_matrix
from scipy.constants import sigma
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

#Code to load in a hemispherical crater as a roughness element for thermal modeling. 

# ------------------------- Data Loaders -------------------------

class CraterMesh:
    def __init__(self, mesh_file, nvtx=61, nfaces=100):
        self.vertices,  self.faces = self.load_mesh(mesh_file,nvtx,nfaces)
        self.sub_vertices, self.sub_faces, self.sub_face_index = self.subdivide()
        self.normals, self.areas, self.centroids, self.sub_normals, self.sub_areas, self.sub_centroids = self.compute_geometry()
        # Precompute local coordinate systems for consistent solar/observer angle calculations
        self.tangent1, self.tangent2 = self._compute_local_coordinates()

    def load_mesh(self, mesh_file,nvtx,nfaces):
        vertices, faces = [], []
        with open(mesh_file, 'r') as f:
            lines = f.readlines()
        vert_lines, face_lines = [], []
        for idx,line in enumerate(lines):
            if idx<nvtx:
                vert_lines.append(line)
            else:
                face_lines.append(line)
        for line in vert_lines:
            vertices.append([float(x) for x in line.strip().split()])
        for line in face_lines:
            faces.append([int(x) for x in line.strip().split() if x.isdigit()])
        return np.array(vertices), np.array(faces)

    def subdivide(self):
        sub_vertices,sub_faces, sub_face_index = trimesh.remesh.subdivide(self.vertices,self.faces, return_index=True)
        return sub_vertices,sub_faces, sub_face_index

    def compute_geometry(self):
        #Compute quantities for regular mesh. 
        v = self.vertices
        f = self.faces
        v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        normals = np.cross(v1 - v0, v2 - v0)
        areas = 0.5 * np.linalg.norm(normals, axis=1)
        normals = normals / np.linalg.norm(normals, axis=1)[:, None]
        centroids = (v0 + v1 + v2) / 3

        #Compute quantities for subdivided mesh. 
        v = self.sub_vertices
        f = self.sub_faces
        v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        sub_normals = np.cross(v1 - v0, v2 - v0)
        sub_areas = 0.5 * np.linalg.norm(sub_normals, axis=1)
        sub_normals = sub_normals / np.linalg.norm(sub_normals, axis=1)[:, None]
        sub_centroids = (v0 + v1 + v2) / 3
        return normals, areas, centroids, sub_normals, sub_areas, sub_centroids
    
    def _compute_local_coordinates(self):
        """
        Precompute local coordinate system (tangent vectors) for each facet.
        Uses same logic as observer_radiance.py for consistency between 
        solar illumination and observer viewing angle calculations.
        
        Returns:
            tangent1: first tangent vector for each facet [n_facets, 3]
            tangent2: second tangent vector for each facet [n_facets, 3]
        """
        n_facets = len(self.normals)
        tangent1 = np.zeros((n_facets, 3))
        tangent2 = np.zeros((n_facets, 3))
        
        for i, normal in enumerate(self.normals):
            # Create local coordinate system for this facet
            # Normal is the local z-axis, need to define x and y axes
            if abs(normal[2]) < 0.9:
                # If normal is not too close to [0,0,1], use [0,0,1] x normal as reference
                tangent1[i] = np.cross([0, 0, 1], normal)
            else:
                # If normal is close to [0,0,1], use [1,0,0] x normal as reference  
                tangent1[i] = np.cross([1, 0, 0], normal)
            
            tangent1[i] = tangent1[i] / np.linalg.norm(tangent1[i])
            tangent2[i] = np.cross(normal, tangent1[i])
        
        return tangent1, tangent2


def compute_solar_angles_all_facets(crater_mesh, sun_vec):
    """
    Calculate mu and phi for sun vector in each facet's local coordinate system.
    Uses same coordinate system as observer calculations for consistency.
    
    Args:
        crater_mesh: CraterMesh object with precomputed tangent vectors
        sun_vec: solar direction vector [x, y, z] (pointing towards sun)
        
    Returns:
        mu_solar: cosine of solar incidence angle for each facet [n_facets]
        phi_solar: solar azimuth angle in local coordinates for each facet [n_facets]
    """
    normals = crater_mesh.normals
    tangent1 = crater_mesh.tangent1
    tangent2 = crater_mesh.tangent2
    n_facets = len(normals)
    
    # Solar incidence angles (dot product with normals)
    mu_solar = np.dot(normals, sun_vec)
    mu_solar[mu_solar < 0] = 0.0  # Only facets facing sun have positive mu
    
    # Solar azimuth angles in local coordinate systems
    phi_solar = np.zeros(n_facets)
    
    for i in range(n_facets):
        if mu_solar[i] <= 0:
            continue
            
        # Project sun vector into facet plane
        sun_in_plane = sun_vec - mu_solar[i] * normals[i]
        sun_in_plane_norm = np.linalg.norm(sun_in_plane)
        
        if sun_in_plane_norm > 1e-10:
            sun_in_plane = sun_in_plane / sun_in_plane_norm
            
            # Calculate azimuth in local coordinate system
            cos_phi = np.dot(sun_in_plane, tangent1[i])
            sin_phi = np.dot(sun_in_plane, tangent2[i])
            phi_solar[i] = np.arctan2(sin_phi, cos_phi)
            
            if phi_solar[i] < 0:
                phi_solar[i] += 2 * np.pi
        else:
            # Sun is along the normal direction, phi is arbitrary
            phi_solar[i] = 0.0
    
    return mu_solar, phi_solar


class SelfHeatingList:
    def __init__(self, fname):
        self.indices, self.view_factors = [], []
        with open(fname, 'r') as f:
            for line in f:
                parts = line.strip().split()
                n = int(parts[0])
                idxs = [int(x)-1 for x in parts[1:n+1]]
                vfs = [float(x) for x in parts[n+1:2*n+1]]
                self.indices.append(np.array(idxs))
                self.view_factors.append(np.array(vfs))
        self.indices = np.array(self.indices, dtype=object)
        self.view_factors = np.array(self.view_factors, dtype=object)

    def as_view_matrix(self, N):
        # Builds full (N,N) view factor matrix (dense, can make sparse if needed)
        V = np.zeros((N, N))
        for i, (idxs, vfs) in enumerate(zip(self.indices, self.view_factors)):
            V[i, list(idxs)] = vfs
        return V

# ---------------------- Shadow Tester ----------------------

class ShadowTester:
    def __init__(self, mesh: CraterMesh):
        self.mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
        self.sub_mesh = trimesh.Trimesh(vertices=mesh.sub_vertices, faces=mesh.sub_faces, process=False)
        self.sub_centroids = mesh.sub_centroids
        self.centroids = mesh.centroids
        self.mapping = mesh.sub_face_index
        self.sub_normals = mesh.sub_normals

    def illuminated_facets(self, sun_vec):
        n_facets = self.sub_centroids.shape[0]
        offsets = 40.0 * sun_vec #move the source of the solar vector sufficiently far away so that it is outside of the crater. 
        origins = self.sub_centroids + offsets
        directions = np.tile(-sun_vec / np.linalg.norm(sun_vec), (n_facets, 1))
        index_tri = self.sub_mesh.ray.intersects_first(origins, directions)
        index_ray = np.arange(n_facets)
        illuminated = np.zeros(len(self.centroids))
        match = index_ray==index_tri
        dot = np.dot(self.sub_normals,sun_vec)>0
        illum = match & dot
        illuminated = np.zeros(len(self.centroids))
        for facet_index in np.arange(len(self.centroids)):
            illuminated[facet_index] = np.sum(illum[self.mapping[facet_index]])
            #for i in self.mapping[facet_index]:
                #if (index_tri[i] == index_ray[i] and np.dot(self.sub_normals[i], sun_vec) > 0):
                #    illuminated[facet_index] += 1.0
        illuminated /= len(index_tri) / len(self.mapping)  # normalize so that 1 = fully illuminated.
        return illuminated

# ------------------ Radiative Source Terms + Multiple Scattering ------------------

class CraterRadiativeTransfer:
    def __init__(self, mesh, selfheating):
        self.mesh = mesh
        self.selfheating = selfheating
        self.view_matrix = self.selfheating.as_view_matrix(len(self.mesh.normals))

    def compute_fluxes(self, sun_vec, illuminated, therm_flux, albedo, emissivity,F_sun, n_waves=1,multiple_scatter=True, max_iter=100, tol=1e-6):
        #therm_flux should be equivalent to emissivity*sigma*T**4 for broadband, or the appropriate narrowband integrated value. 
        n_facets = len(self.mesh.normals)
        areas = self.mesh.areas
        # Solar incidence angle
        sun_vec = sun_vec / np.linalg.norm(sun_vec)
        cosines = np.dot(self.mesh.normals, sun_vec)
        cosines[cosines < 0] = 0.0
        cosines = np.tile(cosines[:,None],(1,n_waves))
        illuminated = np.tile(illuminated[:,None],(1,n_waves))
        albedo = np.tile(albedo,(n_facets,1))

        if(np.any(illuminated>0)):
            # Direct solar absorption
            Q_direct = np.zeros((n_facets,n_waves))
            mask = (illuminated[:,0] > 0) & (cosines[:,0] > 0)
            Q_direct[mask] = (1 - albedo[mask]) * F_sun * cosines[mask] * illuminated[mask]

            # Multiple scattered sunlight 
            if multiple_scatter:
                F_sun = F_sun
                Q_scattered = compute_multiple_scattered_sunlight(
                    albedo, F_sun, illuminated, cosines, self.view_matrix,
                    max_iter=max_iter, tol=tol
                )
            else:
                # Single scattering only 
                Q_scattered = np.zeros((n_facets,n_waves))
                for i in range(n_facets):
                    idxs = self.selfheating.indices[i]
                    vfs = self.selfheating.view_factors[i]
                    for j_idx, vf in zip(idxs, vfs):
                        if illuminated[j_idx] > 0 and cosines[j_idx] > 0:
                            Q_scattered[i] += albedo * F_sun * illuminated[j_idx]* cosines[j_idx] * vf
        else:
            Q_scattered =np.zeros((n_facets,n_waves))
            Q_direct = np.zeros((n_facets,n_waves))

        # Self-heating (thermal IR)
        Q_selfheat = np.zeros((n_facets,n_waves))
        for i in range(n_facets):
            idxs = self.selfheating.indices[i]
            if(therm_flux.ndim == 1):
                #Should be of size [ncols, nwave] or [ncols, 1]
                vfs = self.selfheating.view_factors[i]
            else:
                vfs = self.selfheating.view_factors[i][:,None]
            #Q_selfheat[i] = emissivity * sigma * np.sum((T_surface[list(idxs)] ** 4) * vfs)
            Q_selfheat[i] = np.sum(therm_flux[list(idxs)] * vfs,axis=0)
        if(n_waves==1):
            return Q_direct[:,0], Q_scattered[:,0], Q_selfheat[:,0], cosines[:,0]
        else:
            return Q_direct, Q_scattered, Q_selfheat, cosines

# ---------- Multiple Scattering (Rozitis & Green, Eq 18–20, Iterative) ----------

def compute_multiple_scattered_sunlight(
    Alb, F_sun, illum_frac, sun_cosines, view_matrix, max_iter=100, tol=1e-5
):
    N = len(illum_frac)
    G = Alb * F_sun * illum_frac * sun_cosines  # Initial guess
    for iteration in range(max_iter):
        G_new = np.zeros_like(G)
        for i in range(N):
            sum_vf = np.dot(view_matrix[i], G)
            G_new[i] = Alb[i] * (F_sun * illum_frac[i] * sun_cosines[i] + sum_vf)
        if np.allclose(G_new, G, rtol=tol, atol=tol):
            break
        G = G_new
    F_SCAT = G.copy()
    if(Alb.shape[1]==F_SCAT.shape[1]): 
        F_SCAT[Alb>0.0] /= Alb[Alb>0.0]
    else:
        F_SCAT /= Alb
    return F_SCAT

def compute_multiple_scattered_sunlight_gs(
    Alb, F_sun, illum_frac, sun_cosines, view_matrix, max_iter=100, tol=1e-6
):
    N = len(illum_frac)
    G = Alb * F_sun * illum_frac * sun_cosines  # Initial guess
    #Gauss-Seidel, as in Rozitis and Green 2011. Result is within
    for iteration in range(max_iter):
        converged = True
        for i in range(N):
            beforesum = np.dot(view_matrix[i, :i], G[:i]) if i > 0 else 0.0
            aftersum = np.dot(view_matrix[i, i+1:], G[i+1:]) if i < N-1 else 0.0
            newval = Alb * (F_sun * illum_frac[i] * sun_cosines[i] + beforesum + aftersum)
            if abs(newval - G[i]) > tol:
                converged = False
            G[i] = newval
        if converged:
            break
    F_SCAT = G.copy() 
    F_SCAT[Alb>0.0] /= Alb[Alb>0.0]
    return F_SCAT




def plot_crater_temperature(mesh, temperatures, time_label=""):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')

    verts = mesh.vertices
    faces = mesh.faces
    temp = temperatures  # length = number of faces

    # Map temperature to color
    norm = plt.Normalize(vmin=np.min(temp), vmax=np.max(temp))
    facecolors = plt.cm.inferno(norm(temp))

    poly3d = [verts[face] for face in faces]
    pc = Poly3DCollection(poly3d, facecolors=facecolors, linewidths=0.05, edgecolors='gray', alpha=1.0)
    ax.add_collection3d(pc)

    ax.set_xlim([verts[:,0].min(), verts[:,0].max()])
    ax.set_ylim([verts[:,1].min(), verts[:,1].max()])
    ax.set_zlim([verts[:,2].min(), verts[:,2].max()])
    ax.set_box_aspect([2,2,1])

    mappable = plt.cm.ScalarMappable(cmap='inferno', norm=norm)
    mappable.set_array(temp)
    cbar = plt.colorbar(mappable, ax=ax, shrink=0.5, pad=0.05)
    cbar.set_label("Surface Temperature [K]")

    ax.set_title(f"Crater Temperature Distribution {time_label}")
    plt.tight_layout()
    plt.show()

