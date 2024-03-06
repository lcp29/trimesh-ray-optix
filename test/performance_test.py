import time
import torch
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from triro.ray.ray_optix import RayMeshIntersector


def gen_rays(cam_mat, w, h, f):
    y, x = torch.meshgrid(
        [torch.linspace(0, h - 1, h), torch.linspace(0, w - 1, w)], indexing="ij"
    )
    x = x - (w - 1) / 2
    y = y - (h - 1) / 2
    z = -torch.ones_like(x) * f
    dirs = torch.stack([x, y, z], dim=-1).cuda()
    dirs /= torch.norm(dirs, dim=-1, keepdim=True)
    dirs = dirs @ torch.transpose(cam_mat, 0, 1)
    return dirs

cam_mat = torch.Tensor(
    [
        [5.6272650e-01, 2.7091104e-01, 7.8099048e-01],
        [8.2602328e-01, -1.4769979e-01, -5.4393965e-01],
        [3.2007132e-02, -9.5120555e-01, 3.0689341e-01],
    ]
).cuda()
rw = 640
rh = int(rw * 9 / 16)
rf = int(rw * 25 / 36)

GPU_ITER = 100000
CPU_ITER = 100

cam_origin = torch.Tensor([4.3092918e01, -2.9232937e01, 3.7687759e01])

# GPU
ray_dirs = gen_rays(cam_mat, rw, rh, rf)
ray_origins = (
    cam_origin
    .cuda()
    .broadcast_to(ray_dirs.shape)
)
mesh = trimesh.load('test/models/iscv2.obj', force='mesh')
r = RayMeshIntersector(mesh)

gpu_start_time = time.time()
for i in range(GPU_ITER):
    result = r.intersects_closest(ray_origins, ray_dirs)
gpu_end_time = time.time()
gpu_time = gpu_end_time - gpu_start_time

print(f'GPU time: {gpu_time:.3f} s / {GPU_ITER} iters')

loc = torch.norm(result[3].reshape(-1, 3) - cam_origin.cuda(), dim=-1)
loc -= loc.min(dim=0, keepdim=True)[0]
loc /= loc.max(dim=0, keepdim=True)[0]
loc = loc.reshape((rh, rw))

plt.imshow(loc.cpu(), cmap='gray')
plt.show()

# CPU
ray_dirs = ray_dirs.cpu().reshape(-1, 3)
ray_origins = ray_origins.cpu().reshape(-1, 3)
cpu_start_time = time.time()
for i in range(CPU_ITER):
    result = mesh.ray.intersects_location(ray_origins, ray_dirs, False)
cpu_end_time = time.time()
cpu_time = cpu_end_time - cpu_start_time
print(f'Trimesh & PyEmbree CPU time: {cpu_time:.3f} s / {CPU_ITER} iters')

loc = torch.zeros((rh, rw, 3)).reshape(-1, 3)
loc_tight = torch.from_numpy(result[0]).float()
ray_idx = torch.from_numpy(result[1])[..., None].broadcast_to(loc_tight.shape)
loc = torch.scatter(loc, 0, ray_idx, loc_tight)
loc = torch.norm(loc - cam_origin, dim=-1)
loc -= loc.min(dim=0, keepdim=True)[0]
loc /= loc.max(dim=0, keepdim=True)[0]
loc = loc.reshape(rh, rw)
plt.imshow(loc, cmap='gray')
plt.show()

print(f'speedup: {int((cpu_time / CPU_ITER) / (gpu_time / GPU_ITER))}x')
