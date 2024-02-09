
import torch
import trimesh
import hmesh.ray.ray_optix

f = torch.Tensor([[0.5, -0.5, 0], [0, 0.5, 0], [-0.5, -0.5, 0]])
i = torch.Tensor([[0, 1, 2]]).int()
m = trimesh.Trimesh(vertices=f, faces=i)
o = torch.Tensor([[0, 0, 4], [10, 10, 10]]).cuda()
d = torch.Tensor([[0, 0, -1], [0, 1, 0]]).cuda()
r = hmesh.ray.ray_optix.RayMeshIntersector(m)
print(r.intersects_any(o, d))
