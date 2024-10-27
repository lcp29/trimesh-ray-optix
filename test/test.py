import torch
import trimesh
import matplotlib.pyplot as plt
from triro.ray.ray_optix import RayMeshIntersector

f = torch.Tensor([[0.5, -0.5, 0], [0, 0.5, 0], [-0.5, -0.5, 0]])
i = torch.Tensor([[0, 1, 2]]).int()
m = trimesh.Trimesh(vertices=f, faces=i)
o = torch.Tensor([[0, 0, 4], [10, 10, 10]]).cuda()
d = torch.Tensor([[0, 0, -1], [0, 1, 0]]).cuda()
r = RayMeshIntersector(mesh=m)
print(r.intersects_any(o, d))
print(r.intersects_first(o, d))

s = trimesh.creation.icosphere()
sr = RayMeshIntersector(mesh=s)
x, y = torch.meshgrid(
    [torch.linspace(-1, 1, 800), torch.linspace(-1, 1, 800)], indexing="ij"
)
z = -torch.ones_like(x)

dirs = torch.stack([x, -y, z], dim=-1).cuda()
print(f'dirs: {dirs.shape}, stride: {dirs.stride()}')
origin = torch.Tensor([[0, 0, 3]]).cuda().broadcast_to(dirs.shape)

hit, front, ray_idx, tri_idx, location, uv = sr.intersects_closest(
    origin, dirs, stream_compaction=True
)
loc = torch.zeros([*hit.shape, 3]).cuda().float()
loc[hit] = location
print(front)
plt.imshow(loc.cpu())
plt.show()

mesh_normals = torch.from_numpy(s.vertex_normals).cuda().float()
mesh_faces = torch.from_numpy(s.faces).cuda().int()

tri_v = mesh_faces[tri_idx]
tri_norm = mesh_normals[tri_v]
print(tri_v)
hit_norm = uv[:, :1] * tri_norm[:, 0] + uv[:, 1:] * tri_norm[:, 1] + (1 - uv[:, :1] - uv[:, 1:]) * tri_norm[:, 2]
loc[hit] = hit_norm
plt.imshow(loc.cpu())
plt.show()


f = torch.Tensor(
    [
        [0.5, -0.5, 0],
        [0, 0.5, 0],
        [-0.5, -0.5, 0],
        [0.5, -0.5, -1],
        [0, 0.5, -1],
        [-0.5, -0.5, -1],
    ]
)
i = torch.Tensor([[0, 1, 2], [3, 4, 5]]).int()
m = trimesh.Trimesh(vertices=f, faces=i)
o = torch.Tensor([[0, 0, -4], [0, 0.1, 4]]).cuda()
d = torch.Tensor([[0, 0, 1], [0, 0, -1]]).cuda()
r = RayMeshIntersector(m)
print(r.intersects_location(o, d))
print(r.intersects_count(o, d))
print(sr.contains_points(torch.Tensor([[0, 0, 0.999]]).cuda()))
