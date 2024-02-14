import os
import torch
from typing import Tuple
from jaxtyping import Float32, Bool, Int32
import importlib
import torch.utils.cpp_extension
from triro.ray.ray_optix import OptixAccelStructureWrapper

triro_module = None


def get_module():
    """
    Get the triro module by compiling it if necessary and importing it.

    Returns:
        The triro module.
    """
    global triro_module
    if triro_module is not None:
        return triro_module
    # compile module
    # source file
    source_files = ["base.cpp", "binding.cpp", "ray.cpp"]
    # optix install location
    optix_install_dir = os.environ["OptiX_INSTALL_DIR"]
    # include optix
    cflags = [f"-I{optix_install_dir}/include"]
    # link with cuda lib
    ldflags = ["-lcuda"]
    # full source path
    source_paths = [os.path.join(os.path.dirname(__file__), fn) for fn in source_files]
    torch.utils.cpp_extension.load(
        name="triro",
        sources=source_paths,
        extra_cflags=cflags,
        extra_ldflags=ldflags,
        with_cuda=True,
        verbose=False,
    )
    triro_module = importlib.import_module("triro")
    return triro_module


def init_optix():
    """
    Initialize OptiX.
    """
    get_module().initOptix()


def create_optix_context():
    """
    Create an OptiX context.
    """
    get_module().createOptixContext()


def create_optix_module():
    """
    Create an OptiX module.
    """
    get_module().createOptixModule()


def create_optix_pipelines():
    """
    Create OptiX pipelines.
    """
    get_module().createOptixPipelines()


def build_sbts():
    """
    Build the SBTs (Shader Binding Tables).
    """
    get_module().buildSBT()


def intersects_any(
    accel_structure: OptixAccelStructureWrapper,
    origins: Float32[torch.Tensor, "*b 3"],
    dirs: Float32[torch.Tensor, "*b 3"],
) -> Bool[torch.Tensor, "*b"]:
    """
    Check if any ray intersects with the acceleration structure.

    Args:
        accel_structure: The acceleration structure.
        origins: The origins of the rays.
        dirs: The directions of the rays.

    Returns:
        A boolean tensor indicating if each ray intersects with the acceleration structure.
    """
    return get_module().intersectsAny(accel_structure._inner, origins, dirs)


def intersects_first(
    accel_structure: OptixAccelStructureWrapper,
    origins: Float32[torch.Tensor, "*b 3"],
    dirs: Float32[torch.Tensor, "*b 3"],
) -> Int32[torch.Tensor, "*b"]:
    """
    Find the index of the first intersection for each ray.

    Args:
        accel_structure: The acceleration structure.
        origins: The origins of the rays.
        dirs: The directions of the rays.

    Returns:
        An integer tensor indicating the index of the first intersection for each ray.
    """
    return get_module().intersectsFirst(accel_structure._inner, origins, dirs)


def intersects_closest(
    accel_structure: OptixAccelStructureWrapper,
    origins: Float32[torch.Tensor, "*b 3"],
    dirs: Float32[torch.Tensor, "*b 3"],
) -> Tuple[
    Bool[torch.Tensor, "*b"],  # hit
    Bool[torch.Tensor, "*b"],  # front
    Int32[torch.Tensor, "*b"],  # triangle index
    Float32[torch.Tensor, "*b 3"],  # intersect location
    Float32[torch.Tensor, "*b 2"],  # uv
]:
    """
    Find the closest intersection for each ray.

    Args:
        accel_structure: The acceleration structure.
        origins: The origins of the rays.
        dirs: The directions of the rays.

    Returns:
        A tuple containing the following tensors:
        - A boolean tensor indicating if each ray hits an object.
        - A boolean tensor indicating if each ray hits the front face of an object.
        - An integer tensor indicating the index of the triangle that each ray intersects with.
        - A float tensor indicating the location of the intersection for each ray.
        - A float tensor indicating the UV coordinates of the intersection for each ray.
    """
    return get_module().intersectsClosest(accel_structure._inner, origins, dirs)


def intersects_count(
    accel_structure: OptixAccelStructureWrapper,
    origins: Float32[torch.Tensor, "*b 3"],
    dirs: Float32[torch.Tensor, "*b 3"],
) -> Int32[torch.Tensor, "*b"]:
    """
    Count the number of intersections for each ray.

    Args:
        accel_structure: The acceleration structure.
        origins: The origins of the rays.
        dirs: The directions of the rays.

    Returns:
        An integer tensor indicating the number of intersections for each ray.
    """
    return get_module().intersectsCount(accel_structure._inner, origins, dirs)


def intersects_location(
    accel_structure: OptixAccelStructureWrapper,
    origins: Float32[torch.Tensor, "*b 3"],
    dirs: Float32[torch.Tensor, "*b 3"],
) -> Tuple[
    Float32[torch.Tensor, "h 3"], Int32[torch.Tensor, "h"], Int32[torch.Tensor, "h"]
]:
    """
    Find the location of intersections for each ray.

    Args:
        accel_structure: The acceleration structure.
        origins: The origins of the rays.
        dirs: The directions of the rays.

    Returns:
        A tuple containing the following tensors:
        - A float tensor indicating the location of the intersection for each ray.
        - The index of the ray that had the intersection.
        - An integer tensor indicating the index of the instance that each ray intersects with.
    """
    return get_module().intersectsLocation(accel_structure._inner, origins, dirs)
