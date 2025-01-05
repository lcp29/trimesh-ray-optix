# Triro - A Python Ray-Mesh Intersector in OptiX

Triro is a mesh ray tracing library implemented with NVIDIA OptiX. It has compatible interface with [trimesh.ray](https://trimesh.org/trimesh.ray.html) and provides other convenient functions.

## 🔧️ Installation
> You may need to enable <a href="https://superuser.com/questions/1715715/can-i-enable-unicode-utf-8-worldwide-support-in-windows-11-but-set-another-enco">unicode UTF-8 global support</a> in Windows for some character set problems.

You need an OptiX SDK (>=7.7) installed to get to use Triro. If you are running Windows you also need an MSVC installation. First
```sh
# if you are running in Windows set the system variable
export OptiX_INSTALL_DIR=<Your Optix SDK installation directory>
```
Then
```sh
pip install git+https://github.com/lcp29/trimesh-ray-optix
```
or
```sh
git clone https://github.com/lcp29/trimesh-ray-optix
cd trimesh-ray-optix
pip install .
```
## 📖️ Example
```python
import torch
import trimesh
import matplotlib.pyplot as plt
from triro.ray.ray_optix import RayMeshIntersector

# creating mesh and intersector
mesh = trimesh.creation.icosphere()
intersector = RayMeshIntersector(mesh=mesh)

# generating rays
y, x = torch.meshgrid([torch.linspace(1, -1, 800), 
                       torch.linspace(-1, 1, 800)], indexing='ij')
z = -torch.ones_like(x)
ray_directions = torch.stack([x, y, z], dim=-1).cuda()
ray_origins = torch.Tensor([0, 0, 3]).cuda().broadcast_to(ray_directions.shape)

# OptiX, Launch!
hit, front, ray_idx, tri_idx, location, uv = intersector.intersects_closest(
    ray_origins, ray_directions, stream_compaction=True
)

# drawing result
locs = torch.zeros((800, 800, 3)).cuda()
locs[hit] = location
plt.imshow(locs.cpu())
plt.show()
```
The above code generates the following result:

![](assets/location.png)

## 🕊️ TODOs

 - [x] Installation on Windows
 - [x] Supporting Tensor strides

## 🚀️ Performance Comparison

Scene closest-hit ray tracing tested under Ubuntu 22.04, i5-13490F and RTX 3090 ([performance_test.py](test/performance_test.py)):
```
GPU time: 8.362 s / 100000 iters
Trimesh & PyEmbree CPU time: 18.175 s / 100 iters
speedup: 2173x
```

![](assets/testcase.png)

## 🔍️ Documentation

The Trimesh document is [here](https://trimesh.org/trimesh.ray.html).

For detailed document see [here](docs/api.md) or the [wiki](https://github.com/lcp29/trimesh-ray-optix/wiki).
