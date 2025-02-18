import torch
import trimesh
from typing import Tuple, Optional
from jaxtyping import Float32, Int32, Bool
import triro.backend.ops as hops


## @class RayMeshIntersector
# @brief A class for performing ray-mesh intersection tests using OptiX acceleration structure.
#
# This class provides methods for checking if rays intersect with a mesh, finding the closest intersection,
# retrieving intersection locations, counting intersections, and more.
#
# This class has similar functionality as the `RayMeshIntersector` class in `trimesh.ray`.
#
# @param kwargs: Keyword arguments for initializing the intersector.
#                Either 'mesh' or 'vertices' and 'faces' must be provided.
class RayMeshIntersector:
    
    ## @brief Initialize the RayMeshIntersector class.
    #
    # @param kwargs: Keyword arguments for initializing the intersector.
    #                Either 'mesh' or 'vertices' and 'faces' must be provided.
    def __init__(self, **kwargs):
        if 'mesh' in kwargs:
            mesh = kwargs['mesh']
            # mesh vertices
            # [n, 3] float32 on the device
            self.mesh_vertices = torch.from_numpy(mesh.vertices).float().contiguous().cuda()
            # [n, 3] int32 on the device
            self.mesh_faces = torch.from_numpy(mesh.faces).int().contiguous().cuda()
        elif 'vertices' in kwargs and 'faces' in kwargs:
            vertices = kwargs['vertices']
            faces = kwargs['faces']
            # mesh vertices
            # [n, 3] float32 on the device
            self.mesh_vertices = vertices.float().contiguous().cuda()
            # [n, 3] int32 on the device
            self.mesh_faces = faces.int().contiguous().cuda()
        else:
            raise ValueError("Either 'mesh' or 'vertices' and 'faces' must be provided.")
        # ([3], [3])
        self.mesh_aabb = (
            torch.min(self.mesh_vertices, dim=0)[0],
            torch.max(self.mesh_vertices, dim=0)[0],
        )
        # build acceleration structure
        self.as_wrapper = OptixAccelStructureWrapper()
        self.as_wrapper.build_accel_structure(self.mesh_vertices, self.mesh_faces)

    ## @brief Update the raw mesh data.
    #
    # @param vertices (Float32[torch.Tensor, "n 3"]): The vertices of the mesh.
    # @param faces (Int32[torch.Tensor, "f 3"]): The faces of the mesh.
    def update_raw(self, 
                   vertices: Float32[torch.Tensor, "n 3"], 
                   faces: Int32[torch.Tensor, "f 3"]):
        # mesh vertices
        # [n, 3] float32 on the device
        self.mesh_vertices = vertices.float().contiguous().cuda()
        # [f, 3] int32 on the device
        self.mesh_faces = faces.int().contiguous().cuda()
        # ([3], [3])
        self.mesh_aabb = (
            torch.min(self.mesh_vertices, dim=0)[0],
            torch.max(self.mesh_vertices, dim=0)[0],
        )
        # build acceleration structure
        self.as_wrapper.build_accel_structure(self.mesh_vertices, self.mesh_faces)

    ## @brief Check if any intersections occur for each ray.
    #
    # @param origins (Float32[torch.Tensor, "*b 3"]): The origins of the rays.
    # @param directions (Float32[torch.Tensor, "*b 3"]): The directions of the rays.
    #
    # @return Bool[torch.Tensor, "*b"]: A boolean tensor indicating if each ray intersects with the mesh.
    def intersects_any(
        self,
        origins: Float32[torch.Tensor, "*b 3"],
        directions: Float32[torch.Tensor, "*b 3"],
    ) -> Bool[torch.Tensor, "*b"]:
        return hops.intersects_any(self.as_wrapper, origins, directions)

    ## @brief Find the index of the first intersection for each ray.
    #
    # @param origins (Float32[torch.Tensor, "*b 3"]): The origins of the rays.
    # @param directions (Float32[torch.Tensor, "*b 3"]): The directions of the rays.
    #
    # @return Int32[torch.Tensor, "*b"]: The index of the first intersection for each ray.
    def intersects_first(
        self,
        origins: Float32[torch.Tensor, "*b 3"],
        directions: Float32[torch.Tensor, "*b 3"],
    ) -> Int32[torch.Tensor, "*b"]:
        return hops.intersects_first(self.as_wrapper, origins, directions)

    ## @brief Find the closest intersection for each ray.
    #
    # @param origins (Float32[torch.Tensor, "*b 3"]): The origins of the rays.
    # @param directions (Float32[torch.Tensor, "*b 3"]): The directions of the rays.
    # @param stream_compaction (bool, optional): Whether to perform stream compaction. Defaults to False.
    #
    # @return Tuple: A tuple containing the following elements:
    #                - If `stream_compaction` is False:
    #                  - hit (Bool[torch.Tensor, "*b"]): A boolean tensor indicating if each ray intersects with the mesh.
    #                  - front (Bool[torch.Tensor, "*b"]): A boolean tensor indicating if the intersection is from the front face of the mesh.
    #                  - triangle index (Int32[torch.Tensor, "*b"]): The index of the triangle that was intersected by each ray.
    #                  - intersect location (Float32[torch.Tensor, "*b 3"]): The 3D coordinates of the closest intersection point.
    #                  - uv (Float32[torch.Tensor, "*b 2"]): The UV coordinates of the closest intersection point.
    #                - If `stream_compaction` is True:
    #                  - hit (Bool[torch.Tensor, "*b"]): A boolean tensor indicating if each ray intersects with the mesh.
    #                  - front (Bool[torch.Tensor, "h"]): A boolean tensor indicating if the intersection is from the front face of the mesh.
    #                  - ray index (Int32[torch.Tensor, "h"]): The index of the ray that had the closest intersection.
    #                  - triangle index (Int32[torch.Tensor, "h"]): The index of the triangle that was intersected by the closest ray.
    #                  - intersect location (Float32[torch.Tensor, "h 3"]): The 3D coordinates of the closest intersection point.
    #                  - uv (Float32[torch.Tensor, "h 2"]): The UV coordinates of the closest intersection point.
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

    ## @brief Find the intersection location for each ray.
    #
    # @param origins (Float32[torch.Tensor, "*b 3"]): The origins of the rays.
    # @param directions (Float32[torch.Tensor, "*b 3"]): The directions of the rays.
    #
    # @return Tuple: A tuple containing the following elements:
    #                - intersection locations (Float32[torch.Tensor, "h 3"]): The 3D coordinates of the intersection points for each ray.
    #                - hit ray indices (Int32[torch.Tensor, "h"]): The indices of the rays that had intersections.
    #                - triangle indices (Int32[torch.Tensor, "h"]): The indices of the triangles that were intersected by the rays.
    def intersects_location(
        self,
        origins: Float32[torch.Tensor, "*b 3"],
        directions: Float32[torch.Tensor, "*b 3"],
    ) -> Tuple[
        Float32[torch.Tensor, "h 3"], Int32[torch.Tensor, "h"], Int32[torch.Tensor, "h"]
    ]:
        return hops.intersects_location(self.as_wrapper, origins, directions)

    ## @brief Count the number of intersections for each ray.
    #
    # @param origins (Float32[torch.Tensor, "*b 3"]): The origins of the rays.
    # @param directions (Float32[torch.Tensor, "*b 3"]): The directions of the rays.
    #
    # @return Int32[torch.Tensor, "*b 3"]: The number of intersections for each ray.
    def intersects_count(
        self,
        origins: Float32[torch.Tensor, "*b 3"],
        directions: Float32[torch.Tensor, "*b 3"],
    ) -> Int32[torch.Tensor, "*b 3"]:
        return hops.intersects_count(self.as_wrapper, origins, directions)

    ## @brief Find the intersection indices for each ray.
    #
    # @param origins (Float32[torch.Tensor, "*b 3"]): The origins of the rays.
    # @param directions (Float32[torch.Tensor, "*b 3"]): The directions of the rays.
    # @param return_locations (bool, optional): Whether to return the intersection locations. Defaults to False.
    # @param multiple_hits (bool, optional): Whether to allow multiple intersections per ray. Defaults to True.
    #
    # @return Tuple: A tuple containing the following elements:
    #                - hit triangle indices (Int32[torch.Tensor, "h"]): The indices of the triangles that were hit by the rays.
    #                - ray indices (Int32[torch.Tensor, "h"]): The indices of the rays that had intersections.
    #                - hit location (Float32[torch.Tensor, "h 3"]): The 3D coordinates of the intersection points for each ray.
    #                  (Only returned if `return_locations` is set to True)
    def intersects_id(
        self,
        origins: Float32[torch.Tensor, "*b 3"],
        directions: Float32[torch.Tensor, "*b 3"],
        return_locations: bool = False,
        multiple_hits: bool = True,
    ) -> (
        Tuple[
            Int32[torch.Tensor, "h"],  # hit triangle indices
            Int32[torch.Tensor, "h"],  # ray indices
            Float32[torch.Tensor, "h 3"],  # hit location
        ]
        | Tuple[
            Int32[torch.Tensor, "h"], Int32[torch.Tensor, "h"]
        ]  # hit triangle indices and ray indices
    ):
        if multiple_hits:
            loc, ray_idx, tri_idx = hops.intersects_location(
                self.as_wrapper, origins, directions
            )
            if return_locations:
                return tri_idx, ray_idx, loc
            else:
                return tri_idx, ray_idx
        else:
            hit, _, tri_idx, loc, _ = hops.intersects_closest(
                self.as_wrapper, origins, directions
            )
            ray_idx = torch.arange(0, hit.shape.numel()).cuda().int()[hit.reshape(-1)]
            if return_locations:
                return tri_idx[hit], ray_idx, loc[hit]
            else:
                return tri_idx[hit], ray_idx

    ## @brief Check if the points are contained within the mesh.
    #
    # @param points (Float32[torch.Tensor, "*b 3"]): The points to check.
    # @param check_direction (Optional[Float32[torch.Tensor, "3"]], optional): The direction to check. Defaults to None.
    #
    # @return Bool[torch.Tensor, "*b 3"]: A boolean tensor indicating if each point is contained within the mesh.
    def contains_points(
        self,
        points: Float32[torch.Tensor, "*b 3"],
        check_direction: Optional[Float32[torch.Tensor, "3"]] = None,
    ) -> Bool[torch.Tensor, "*b 3"]:
        contains = torch.zeros(points.shape[:-1], dtype=torch.bool, device=points.device)
        # check if points are in the aabb
        inside_aabb = ~(
            (~(points > self.mesh_aabb[0])).any(dim=1)
            | (~(points < self.mesh_aabb[1])).any(dim=1)
        )

        if not inside_aabb.any():
            return contains
        default_direction = torch.Tensor(
            [0.4395064455, 0.617598629942, 0.652231566745]
        ).cuda()
        # overwrite default direction
        if check_direction is None:
            ray_directions = torch.tile(default_direction, [*contains.shape, 1])
        else:
            ray_directions = torch.tile(check_direction, [*contains.shape, 1])
        # ray trace in two directions
        hit_count = torch.stack(
            [
                hops.intersects_count(self.as_wrapper, points, ray_directions),
                hops.intersects_count(self.as_wrapper, points, -ray_directions),
            ],
            dim=0,
        )
        
        # if hit count in two directions are all odd number then the point is likely to be inside the mesh
        hit_count_mod_2 = torch.remainder(hit_count, 2)
        agree = torch.all(hit_count_mod_2, dim=0)
        contain = inside_aabb & agree & hit_count_mod_2[0] == 1

        broken_mask = ~agree & (hit_count == 0).any(dim=0)
    
        if not broken_mask.any():
            return contain

        if check_direction is None:
            new_direction = (torch.rand(3) - 0.5).cuda()
            contains = contain.cuda()
            broken_mask = broken_mask.cuda()
            points = points.cuda()
            contains[broken_mask] = self.contains_points(points[broken_mask], new_direction)

        return contains


class OptixAccelStructureWrapper:
    def __init__(self):
        self._inner = hops.get_module().OptixAccelStructureWrapperCPP()

    def __del__(self):
        self._inner.freeAccelStructure()

    def build_accel_structure(
        self,
        vertices: Float32[torch.Tensor, "nvert 3"],
        faces: Int32[torch.Tensor, "nface 3"],
    ):
        self._inner.buildAccelStructure(vertices, faces)
