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

    def compute_fluxes(self, sun_vec, illuminated, T_surface, albedo, emissivity,solar_constant, multiple_scatter=True, max_iter=100, tol=1e-6):
        n_facets = len(self.mesh.normals)
        areas = self.mesh.areas
        # Solar incidence angle
        sun_vec = sun_vec / np.linalg.norm(sun_vec)
        cosines = np.dot(self.mesh.normals, sun_vec)
        cosines[cosines < 0] = 0.0

        if(sun_vec[2]>0.001):
            # Direct solar absorption
            Q_direct = np.zeros(n_facets)
            mask = (illuminated > 0) & (cosines > 0)
            Q_direct[mask] = (1 - albedo) * solar_constant * cosines[mask] * illuminated[mask]

            # Multiple scattered sunlight 
            if multiple_scatter:
                F_sun = solar_constant
                Q_scattered = compute_multiple_scattered_sunlight(
                    albedo, F_sun, illuminated, cosines, self.view_matrix,
                    max_iter=max_iter, tol=tol
                )
            else:
                # Single scattering only 
                Q_scattered = np.zeros(n_facets)
                for i in range(n_facets):
                    idxs = self.selfheating.indices[i]
                    vfs = self.selfheating.view_factors[i]
                    for j_idx, vf in zip(idxs, vfs):
                        if illuminated[j_idx] > 0 and cosines[j_idx] > 0:
                            Q_scattered[i] += albedo * solar_constant * illuminated[j_idx]* cosines[j_idx] * vf
        else:
            Q_scattered = np.zeros(n_facets)
            Q_direct = np.zeros(n_facets)

        # Self-heating (thermal IR)
        Q_selfheat = np.zeros(n_facets)
        for i in range(n_facets):
            idxs = self.selfheating.indices[i]
            vfs = self.selfheating.view_factors[i]
            Q_selfheat[i] = emissivity * sigma * np.sum((T_surface[list(idxs)] ** 4) * vfs)

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
            G_new[i] = Alb * (F_sun * illum_frac[i] * sun_cosines[i] + sum_vf)
        if np.allclose(G_new, G, rtol=tol, atol=tol):
            break
        G = G_new
    F_SCAT = G / Alb
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
    F_SCAT = G / Alb
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

