import torch
import trimesh
from typing import Tuple
from jaxtyping import Float32, Int32, Bool
import hmesh.backend.ops as hops


class RayMeshIntersector:
    def __init__(self, mesh: trimesh.Trimesh):
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

    def intersects_any(
        self,
        origins: Float32[torch.Tensor, "*b 3"],
        directions: Float32[torch.Tensor, "*b 3"],
    ) -> Bool[torch.Tensor, "*b"]:
        return hops.intersects_any(self.as_wrapper, origins, directions)

    def intersects_first(
        self,
        origins: Float32[torch.Tensor, "*b 3"],
        directions: Float32[torch.Tensor, "*b 3"],
    ) -> Bool[torch.Tensor, "*b"]:
        return hops.intersects_first(self.as_wrapper, origins, directions)

    def intersects_closest(
        self,
        origins: Float32[torch.Tensor, "*b 3"],
        directions: Float32[torch.Tensor, "*b 3"],
        stream_compaction: bool = False,
    ) -> (
        Tuple[
            Bool[torch.Tensor, "*b"],  # hit
            Bool[torch.Tensor, "*b"],  # front
            Int32[torch.Tensor, "*b"],  # triangle index
            Float32[torch.Tensor, "*b 3"],  # intersect location
            Float32[torch.Tensor, "*b 2"],  # uv
        ]
        | Tuple[
            Bool[torch.Tensor, "*b"],  # hit
            Bool[torch.Tensor, "h"],  # front
            Int32[torch.Tensor, "h"],  # ray index
            Int32[torch.Tensor, "h"],  # triangle index
            Float32[torch.Tensor, "h 3"],  # intersect location
            Float32[torch.Tensor, "h 2"],  # uv:
        ]
    ):
        hit, front, tri_idx, loc, uv = hops.intersects_closest(
            self.as_wrapper, origins, directions
        )
        if stream_compaction:
            ray_idx = torch.arange(0, hit.shape.numel()).cuda().int()[hit.reshape(-1)]
            return hit, front[hit], ray_idx, tri_idx[hit], loc[hit], uv[hit]
        else:
            return hit, front, tri_idx, loc, uv

    def intersects_location(
        self,
        origins: Float32[torch.Tensor, "*b 3"],
        directions: Float32[torch.Tensor, "*b 3"],
    ) -> Tuple[
        Float32[torch.Tensor, "h 3"], Int32[torch.Tensor, "h"], Int32[torch.Tensor, "h"]
    ]:
        return hops.intersects_location(self.as_wrapper, origins, directions)


class OptixAccelStructureWrapper:
    def __init__(self):
        self._inner = hops.get_module().OptixAccelStructureWrapperCPP()

    def build_accel_structure(
        self,
        vertices: Float32[torch.Tensor, "nvert 3"],
        faces: Int32[torch.Tensor, "nface 3"],
    ):
        self._inner.buildAccelStructure(vertices, faces)
