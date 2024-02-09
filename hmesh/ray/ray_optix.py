import torch
import trimesh
from jaxtyping import Float32, Int32
import hmesh.backend.ops as hops

class RayMeshIntersector:
    def __init__(self, 
                 mesh: trimesh.Trimesh):
        # original mesh on the host memory
        self.mesh_raw = mesh
        # mesh vertices
        # [n, 3] float32 on the device
        self.mesh_vertices = torch.from_numpy(mesh.vertices).float().contiguous().cuda()
        # [n, 3] int32 on the device
        self.mesh_faces = torch.from_numpy(mesh.faces).int().contiguous().cuda()
        # build acceleration structure
        self.as_wrapper = OptixAccelStructureWrapper()
        self.as_wrapper.build_accel_structure(self.mesh_vertices, self.mesh_faces)
    
    def intersects_any(self,
                       origins: torch.Tensor,
                       dirs: torch.Tensor) -> torch.Tensor:
        return hops.intersects_any(self.as_wrapper, origins, dirs)


class OptixAccelStructureWrapper:
    def __init__(self):
        self._inner = hops.get_module().OptixAccelStructureWrapperCPP()
    
    def build_accel_structure(self,
                              vertices: Float32[torch.Tensor, "nvert 3"],
                              faces: Int32[torch.Tensor, "nface 3"]):
        self._inner.buildAccelStructure(vertices, faces)
