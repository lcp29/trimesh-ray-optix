
import torch
import trimesh
import matplotlib.pyplot as plt
from hmesh.ray.ray_optix import RayMeshIntersector

f = torch.Tensor([[0.5, -0.5, 0], [0, 0.5, 0], [-0.5, -0.5, 0]])
i = torch.Tensor([[0, 1, 2]]).int()
m = trimesh.Trimesh(vertices=f, faces=i)
o = torch.Tensor([[0, 0, 4], [10, 10, 10]]).cuda()
d = torch.Tensor([[0, 0, -1], [0, 1, 0]]).cuda()
r = RayMeshIntersector(m)
print(r.intersects_any(o, d))
print(r.intersects_first(o, d))

s = trimesh.creation.icosphere()
sr = RayMeshIntersector(s)
x, y = torch.meshgrid([torch.linspace(-1, 1, 1280),
                       torch.linspace(-1, 1, 1280)],
                       indexing='ij')

z = -torch.ones_like(x)

dirs = torch.stack([x, -y, z], dim=-1).float().cuda().contiguous()
origin = torch.Tensor([[0, 0, 3]]).float().cuda().broadcast_to(dirs.shape).contiguous()

result = sr.intersects_first(origin, dirs)
plt.imshow(result.cpu())
plt.show()
