import torch
import trimesh
from jaxtyping import Float32, Array

class RayMeshIntersector:
    def __init__(self, 
                 mesh: trimesh.Trimesh):
        self.mesh_raw = mesh
        self.mesh_vertices = torch.from_numpy(mesh.vertices, dtype=torch.float32).cuda()
        self.mesh_faces = torch.from_numpy(mesh.faces, dtype=torch.int32).cuda()
