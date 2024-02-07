
import os
import torch
from jaxtyping import Float32, Bool
import importlib
import torch.utils.cpp_extension

hmesh_module = None
optix_inited = False

def get_module():
    global hmesh_module
    if hmesh_module is not None:
        return hmesh_module
    # compile module
    # source file
    source_files = [
        'base.cpp',
        'binding.cpp',
        'ray.cpp',
        'program.cpp'
    ]
    # optix install location
    optix_install_dir = os.environ['OptiX_INSTALL_DIR']
    # include optix
    cflags = [
        f'-I{optix_install_dir}/include'
    ]
    # link with cuda lib
    ldflags = [
        '-lcuda'
    ]
    # full source path
    source_paths = [os.path.join(os.path.dirname(__file__), fn) for fn in source_files]
    torch.utils.cpp_extension.load(name='hmesh', sources=source_paths, extra_cflags=cflags, extra_ldflags=ldflags, with_cuda=True, verbose=False)
    hmesh_module = importlib.import_module('hmesh')
    return hmesh_module

def init_optix():
    global optix_inited
    get_module().initOptix()
    optix_inited = True

def create_optix_context():
    global optix_inited
    if optix_inited:
        get_module().createOptixContext()

def intersects_any(origins: Float32[torch.Tensor, "n 3"],
                   dirs: Float32[torch.Tensor, "n 3"]) -> Bool[torch.Tensor, "n"]:
    return get_module().intersectsAny(origins, dirs)
