import os
import torch
from typing import Tuple
from jaxtyping import Float32, Bool, Int32
import importlib
import torch.utils.cpp_extension
from hmesh.ray.ray_optix import OptixAccelStructureWrapper

hmesh_module = None


def get_module():
    global hmesh_module
    if hmesh_module is not None:
        return hmesh_module
    # compile module
    # source file
    source_files = ["base.cpp", "binding.cpp", "ray.cpp", "program.cpp"]
    # optix install location
    optix_install_dir = os.environ["OptiX_INSTALL_DIR"]
    # include optix
    cflags = [f"-I{optix_install_dir}/include"]
    # link with cuda lib
    ldflags = ["-lcuda"]
    # full source path
    source_paths = [os.path.join(os.path.dirname(__file__), fn) for fn in source_files]
    torch.utils.cpp_extension.load(
        name="hmesh",
        sources=source_paths,
        extra_cflags=cflags,
        extra_ldflags=ldflags,
        with_cuda=True,
        verbose=False,
    )
    hmesh_module = importlib.import_module("hmesh")
    return hmesh_module


def init_optix():
    get_module().initOptix()


def create_optix_context():
    get_module().createOptixContext()


def create_optix_module():
    get_module().createOptixModule()


def create_optix_pipelines():
    get_module().createOptixPipelines()


def build_sbts():
    get_module().buildSBT()


def intersects_any(
    accel_structure: OptixAccelStructureWrapper,
    origins: Float32[torch.Tensor, "*b 3"],
    dirs: Float32[torch.Tensor, "*b 3"],
) -> Bool[torch.Tensor, "*b"]:
    return get_module().intersectsAny(accel_structure._inner, origins, dirs)


def intersects_first(
    accel_structure: OptixAccelStructureWrapper,
    origins: Float32[torch.Tensor, "*b 3"],
    dirs: Float32[torch.Tensor, "*b 3"],
) -> Int32[torch.Tensor, "*b"]:
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
    return get_module().intersectsClosest(accel_structure._inner, origins, dirs)

def intersects_count(accel_structure: OptixAccelStructureWrapper,
    origins: Float32[torch.Tensor, "*b 3"],
    dirs: Float32[torch.Tensor, "*b 3"],) -> Int32[torch.Tensor, "*b"]:
    return get_module().intersectsCount(accel_structure._inner, origins, dirs)

def intersects_location(
    accel_structure: OptixAccelStructureWrapper,
    origins: Float32[torch.Tensor, "*b 3"],
    dirs: Float32[torch.Tensor, "*b 3"],
) -> Tuple[
    Float32[torch.Tensor, "h 3"], Int32[torch.Tensor, "h"], Int32[torch.Tensor, "h"]
]:
    return get_module().intersectsLocation(accel_structure._inner, origins, dirs)
