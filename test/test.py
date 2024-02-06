
import torch
import trimesh
import hmesh.ray.ray_optix

m = trimesh.creation.icosphere()
r = hmesh.ray.ray_optix.RayMeshIntersector(m)
a = torch.Tensor([1,2,3])

print(r.mesh_faces.dtype)
print(r.intersects_any(a, a))
