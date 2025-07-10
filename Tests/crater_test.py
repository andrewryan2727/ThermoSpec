import numpy as np
import trimesh
from scipy.sparse import csr_matrix
from scipy.constants import sigma
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time

#Code to load in a hemispherical crater as a roughness element for thermal modeling. 

# ------------------------- Data Loaders -------------------------

class CraterMesh:
    def __init__(self, mesh_file, nvtx=61, nfaces=100):
        self.vertices,  self.faces = self.load_mesh(mesh_file,nvtx,nfaces)
        self.sub_vertices, self.sub_faces, self.sub_face_index = self.subdivide()
        self.normals, self.areas, self.centroids, self.sub_normals, self.sub_areas, self.sub_centroids = self.compute_geometry()

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
    


class Photovectors:
    def __init__(self, fname):
        self.vectors = np.loadtxt(fname)

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
        self.centroids = mesh.sub_centroids
        self.mapping = mesh.sub_face_index

    def illuminated_facets(self, sun_vec):
        n_facets = mesh.sub_centroids.shape[0]
        offsets = 40.0*sun_vec
        origins = mesh.sub_centroids + offsets
        directions = np.tile(-sun_vec / np.linalg.norm(sun_vec), (n_facets, 1))
        #_, index_ray, index_tri = self.sub_mesh.ray.intersects_location(
        #    ray_origins=origins, ray_directions=directions, multiple_hits=False
        #)
        index_tri = self.sub_mesh.ray.intersects_first(origins,directions)
        index_ray = np.arange(n_facets)
        illuminated = np.zeros(len(mesh.centroids))
        for facet_index in np.arange(len(mesh.centroids)):
            for i in self.mapping[facet_index]:
                if(index_tri[i] == index_ray[i] and np.dot(mesh.sub_normals[i],sun_vec)>0):
                    illuminated[facet_index] += 1.0
        illuminated /= len(index_tri)/len(self.mapping) #normalize so that 1 = fully illuminated. 
        return illuminated

# ---------------------- Subsurface Model (Explicit) ----------------------

class SubsurfaceModel:
    def __init__(self, n_facets, n_z=10, dz=0.01, k=0.01, rho=1500, c=800, T_init=150):
        self.n_facets = n_facets
        self.n_z = n_z
        self.dz = dz
        self.k = k
        self.rho = rho
        self.c = c
        self.T = np.full((n_facets, n_z), T_init, dtype=float)
        self.alpha = k / (rho * c)
        self.C = rho * c * dz
        self.z = np.arange(n_z) * dz

    def step(self, Q_surface, dt, T_bottom=None):
        T = self.T
        n, nz = self.n_facets, self.n_z
        alpha, dz = self.alpha, self.dz
        T_prev = T.copy()
        T[:,1:-1] = T_prev[:,1:-1] + alpha * dt / dz**2 * (T_prev[:,2:] - 2*T_prev[:,1:-1] + T_prev[:,:-2])
        if T_bottom is not None:
            T[:,-1] = T_bottom
        else:
            T[:,-1] = T[:,-2]
        T[:,0] = T_prev[:,0] + dt/self.C * (self.k/dz * (T_prev[:,1] - T_prev[:,0]) + Q_surface)
        self.T = T
        return T

# ------------------ Radiative Source Terms + Multiple Scattering ------------------

class RadiativeTransfer:
    def __init__(self, mesh, selfheating, albedo=0.4, emissivity=0.9, solar_constant=1361):
        self.mesh = mesh
        self.selfheating = selfheating
        self.albedo = albedo
        self.eps = emissivity
        self.solar_constant = solar_constant
        self.view_matrix = self.selfheating.as_view_matrix(len(self.mesh.normals))

    def compute_fluxes(self, sun_vec, illuminated, T_surface, multiple_scatter=True, max_iter=100, tol=1e-6):
        n_facets = len(self.mesh.normals)
        areas = self.mesh.areas
        # Solar incidence angle
        sun_vec = sun_vec / np.linalg.norm(sun_vec)
        cosines = np.dot(self.mesh.normals, sun_vec)
        cosines[cosines < 0] = 0.0

        # Direct solar absorption
        Q_direct = np.zeros(n_facets)
        mask = (illuminated > 0) & (cosines > 0)
        Q_direct[mask] = (1 - self.albedo) * self.solar_constant * cosines[mask]

        # Multiple scattered sunlight (Rozitis & Green 2011)
        if multiple_scatter:
            F_sun = self.solar_constant
            Q_scat = compute_multiple_scattered_sunlight(
                self.albedo, F_sun, illuminated, cosines, self.view_matrix,
                max_iter=max_iter, tol=tol
            )
            # Only the absorbed (1 - A) part goes into heat:
            Q_scattered = (1 - self.albedo) * Q_scat
        else:
            # Single scattering only (previous approach)
            Q_scattered = np.zeros(n_facets)
            for i in range(n_facets):
                idxs = self.selfheating.indices[i]
                vfs = self.selfheating.view_factors[i]
                for j_idx, vf in zip(idxs, vfs):
                    if illuminated[j_idx] > 0 and cosines[j_idx] > 0:
                        Q_scattered[i] += self.albedo * self.solar_constant * illuminated[j_idx]* cosines[j_idx] * vf

        # Self-heating (thermal IR)
        Q_selfheat = np.zeros(n_facets)
        for i in range(n_facets):
            idxs = self.selfheating.indices[i]
            vfs = self.selfheating.view_factors[i]
            Q_selfheat[i] = self.eps * sigma * np.sum((T_surface[list(idxs)] ** 4) * vfs)

        # Emitted
        Q_emit = self.eps * sigma * (T_surface ** 4)
        Q_net = Q_direct + Q_scattered + Q_selfheat - Q_emit
        return Q_net, Q_direct, Q_scattered, Q_selfheat, Q_emit

# ---------- Multiple Scattering (Rozitis & Green, Eq 18–20, Iterative) ----------

def compute_multiple_scattered_sunlight(
    AB, F_sun, illum_frac, sun_cosines, view_matrix, max_iter=100, tol=1e-6
):
    N = len(illum_frac)
    G = AB * F_sun * illum_frac * sun_cosines  # Initial guess
    for iteration in range(max_iter):
        G_new = np.zeros_like(G)
        for i in range(N):
            sum_vf = np.dot(view_matrix[i], G)
            G_new[i] = AB * (F_sun * illum_frac[i] * sun_cosines[i] + sum_vf)
        if np.allclose(G_new, G, rtol=tol, atol=tol):
            break
        G = G_new
    F_SCAT = G / AB
    return F_SCAT

def compute_multiple_scattered_sunlight_gs(
    AB, F_sun, illum_frac, sun_cosines, view_matrix, max_iter=100, tol=1e-6
):
    N = len(illum_frac)
    G = AB * F_sun * illum_frac * sun_cosines  # Initial guess
    #Gauss-Seidel, as in Rozitis and Green 2011. Result is within
    for iteration in range(max_iter):
        converged = True
        for i in range(N):
            beforesum = np.dot(view_matrix[i, :i], G[:i]) if i > 0 else 0.0
            aftersum = np.dot(view_matrix[i, i+1:], G[i+1:]) if i < N-1 else 0.0
            newval = AB * (F_sun * illum_frac[i] * sun_cosines[i] + beforesum + aftersum)
            if abs(newval - G[i]) > tol:
                converged = False
            G[i] = newval
        if converged:
            break
    F_SCAT = G / AB
    return F_SCAT

# --------------- Main Thermal Crater Model + Time Loop ---------------

class ThermalCraterModel:
    def __init__(self, mesh, selfheating, subsurface, radtrans, shadowtester):
        self.mesh = mesh
        #self.photovectors = photovectors
        self.selfheating = selfheating
        self.subsurface = subsurface
        self.radtrans = radtrans
        self.shadowtester = shadowtester

    def run(self, n_steps, dt, sun_vector_func, observer_vector_func, output_every=1,T_bottom=None):
        n_facets = len(self.mesh.normals)
        n_depth = self.subsurface.n_z
        T_history = []
        F_obs_history = []

        for t_idx in range(n_steps):
            time = t_idx * dt

            # 1. Sun and observer geometry (can be functions of time)
            sun_vec = sun_vector_func(time)
            obs_vec = observer_vector_func(time)

            # 2. Determine solar illumination fraction for each crater facet. 
            illuminated = self.shadowtester.illuminated_facets(sun_vec)

            # 3. Radiative fluxes (with multiple scattering)
            T_surface = self.subsurface.T[:, 0]
            Q_net, Q_dir, Q_scat, Q_selfheat, Q_emit = self.radtrans.compute_fluxes(
                sun_vec, illuminated, T_surface
            )

            # 5. Advance subsurface temperatures
            T_new = self.subsurface.step(Q_net, dt,T_bottom = T_bottom)
            # Save every output_every steps
            if t_idx % output_every == 0:
                T_history.append(T_new.copy())
                # 6. Compute observed flux (sum of emission in obs direction)
                obs_vec_norm = obs_vec / np.linalg.norm(obs_vec)
                facet_to_obs = np.dot(self.mesh.normals, obs_vec_norm)
                facet_to_obs[facet_to_obs < 0] = 0.0
                F_emit = self.radtrans.eps * sigma * (self.subsurface.T[:, 0] ** 4)
                F_obs = np.sum(
                    self.mesh.areas * F_emit * facet_to_obs
                )  # Sum emission visible to observer
                F_obs_history.append(F_obs)
        # Outputs
        return np.array(T_history), np.array(F_obs_history), illuminated



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




# --------------- Example Driver Script ---------------

if __name__ == "__main__":
    # --- File paths (update as needed) ---
    mesh_file = "new_crater2.txt"
    #photovector_file = "new_crater2_photovectors.txt"
    selfheating_file = "new_crater2_selfheating_list.txt"

    # --- Load data ---
    mesh = CraterMesh(mesh_file)
    #photovectors = Photovectors(photovector_file)
    selfheating = SelfHeatingList(selfheating_file)

    # --- Subsurface and radiative properties ---
    n_facets = len(mesh.normals)
    subsurface = SubsurfaceModel(n_facets, n_z=10, dz=0.01, k=0.01, rho=1500, c=800, T_init=300)
    radtrans = RadiativeTransfer(mesh, selfheating, albedo=0.4, emissivity=0.9, solar_constant=1361)
    shadowtester = ShadowTester(mesh)

    # --- Solar and observer geometry functions ---
    def sun_vector_func(t):
        # Example: fixed Sun at +z (zenith)
        vec = np.array([0, 0.6, 0.5])
        return vec/np.linalg.norm(vec)
    def observer_vector_func(t):
        # Example: observer at +z (nadir)
        vec = np.array([0.5,0, 0.5])
        return vec/np.linalg.norm(vec)

    # --- Run model ---
    model = ThermalCraterModel(mesh, selfheating, subsurface, radtrans, shadowtester)
    n_steps = 1000
    dt = 10.0  # [s]
    T_history, F_obs_history, illum = model.run(
        n_steps, dt, sun_vector_func, observer_vector_func, output_every=10, T_bottom = 300.0
    )

    # Example plot usage: 
    time_idx = -1  # or any time step
    plot_crater_temperature(mesh, T_history[time_idx,:,0], time_label=f"t = {time_idx * dt} s")

    time_idx = -1  # or any time step
    plot_crater_temperature(mesh, illum, time_label=f"t = {time_idx * dt} s")

    # --- Save or plot output ---
    np.save("crater_T_history.npy", T_history)
    np.save("crater_F_obs_history.npy", F_obs_history)
    print("Simulation complete. Temperatures and observed flux saved.")

