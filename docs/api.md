# RayMeshIntersector

A class for performing ray-mesh intersection tests using OptiX acceleration structure.

This class provides methods for checking if rays intersect with a mesh, finding the closest intersection,
retrieving intersection locations, counting intersections, and more.

## Method: `__init__(self, mesh: trimesh.Trimesh)`

Initialize the `RayMeshIntersector` class.

### Parameters

- `mesh` (`trimesh.Trimesh`): The mesh to be used for intersection tests.

## Method: `intersects_any`

Check if any ray intersects with the mesh.

### Parameters

- `origins` (`Float32[torch.Tensor, "*b 3"]`): The origins of the rays.
- `directions` (`Float32[torch.Tensor, "*b 3"]`): The directions of the rays.

### Returns

- `Bool[torch.Tensor, "*b"]`: A boolean tensor indicating if each ray intersects with the mesh.

## Method: `intersects_first`

Find the index of the first intersection for each ray.

### Parameters

- `origins` (`Float32[torch.Tensor, "*b 3"]`): The origins of the rays.
- `directions` (`Float32[torch.Tensor, "*b 3"]`): The directions of the rays.

### Returns

- `Int32[torch.Tensor, "*b"]`: The index of the first intersection for each ray.

## Method: `intersects_closest`

Find the closest intersection for each ray.

### Parameters

- `origins` (`Float32[torch.Tensor, "*b 3"]`): The origins of the rays.
- `directions` (`Float32[torch.Tensor, "*b 3"]`): The directions of the rays.
- `stream_compaction` (`bool`, optional): Whether to perform stream compaction. Defaults to `False`.

### Returns

If `stream_compaction` is `False`:
- `hit` (`Bool[torch.Tensor, "*b"]`): A boolean tensor indicating if each ray intersects with the mesh.
- `front` (`Bool[torch.Tensor, "*b"]`): A boolean tensor indicating if the intersection is from the front face of the mesh.
- `triangle index` (`Int32[torch.Tensor, "*b"]`): The index of the triangle that was intersected by each ray.
- `intersect location` (`Float32[torch.Tensor, "*b 3"]`): The 3D coordinates of the intersection point for each ray.
- `uv` (`Float32[torch.Tensor, "*b 2"]`): The UV coordinates of the intersection point for each ray.

If `stream_compaction` is `True`:
- `hit` (`Bool[torch.Tensor, "*b"]`): A boolean tensor indicating if each ray intersects with the mesh.
- `front` (`Bool[torch.Tensor, "h"]`): A boolean tensor indicating if the intersection is from the front face of the mesh.
- `ray index` (`Int32[torch.Tensor, "h"]`): The index of the ray that had the closest intersection.
- `triangle index` (`Int32[torch.Tensor, "h"]`): The index of the triangle that was intersected by the closest ray.
- `intersect location` (`Float32[torch.Tensor, "h 3"]`): The 3D coordinates of the closest intersection point.
- `uv` (`Float32[torch.Tensor, "h 2"]`): The UV coordinates of the closest intersection point.

## Method: `intersects_location`

Find the intersection location for each ray.

### Parameters

- `origins` (`Float32[torch.Tensor, "*b 3"]`): The origins of the rays.
- `directions` (`Float32[torch.Tensor, "*b 3"]`): The directions of the rays.

### Returns

- `Tuple`:
  - `intersection locations` (`Float32[torch.Tensor, "h 3"]`): The 3D coordinates of the intersection points for each ray.
  - `hit ray indices` (`Int32[torch.Tensor, "h"]`): The indices of the rays that had intersections.
  - `triangle indices` (`Int32[torch.Tensor, "h"]`): The indices of the triangles that were intersected by the rays.
 return hops.intersects_count(self.as_wrapper, origins, directions)

## Method: `intersects_count`

Count the number of intersections for each ray.

### Parameters

- `origins` (`Float32[torch.Tensor, "*b 3"]`): The origins of the rays.
- `directions` (`Float32[torch.Tensor, "*b 3"]`): The directions of the rays.

### Returns

- `Int32[torch.Tensor, "*b 3"]`: The number of intersections for each ray.

## Method: `intersects_id`

Find the intersection indices for each ray.

### Parameters

- `origins` (`Float32[torch.Tensor, "*b 3"]`): The origins of the rays.
- `directions` (`Float32[torch.Tensor, "*b 3"]`): The directions of the rays.
- `return_locations` (`bool`, optional): Whether to return the intersection locations. Defaults to `False`.
- `multiple_hits` (`bool`, optional): Whether to allow multiple intersections per ray. Defaults to `True`.

### Returns

- `Tuple`:
  - `hit triangle indices` (`Int32[torch.Tensor, "h"]`): The indices of the triangles that were hit by the rays.
  - `ray indices` (`Int32[torch.Tensor, "h"]`): The indices of the rays that had intersections.
  - `hit location` (`Float32[torch.Tensor, "h 3"]`): The 3D coordinates of the intersection points for each ray.
    (Only returned if `return_locations` is set to `True`)

## Method: `contains_points`

Check if points are inside the mesh.

### Parameters

- `points` (`Float32[torch.Tensor, "*b 3"]`): The points to be checked.
- `check_direction` (`Optional[Float32[torch.Tensor, "3"]]`, optional): The direction of the rays used for checking. Defaults to None.

### Returns

- `Bool[torch.Tensor, "*b 3"]`: A boolean tensor indicating if each point is inside the mesh.

