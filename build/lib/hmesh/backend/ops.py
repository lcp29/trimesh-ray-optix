
import os
import torch
import importlib
import torch.utils.cpp_extension

backend = None

def get_module():
    global backend
    if backend is not None:
        return backend
    # compile module
    source_files = [
        'base.cpp',
        'bindings.cpp'
    ]

    optix_install_dir = os.environ['OptiX_INSTALL_DIR']

    cflags = [
        f'-I{optix_install_dir}/include',
        '-DNVDR_TORCH'
    ]

    ldflags = [
        '-lcuda'
    ]

    source_paths = [os.path.join(os.path.dirname(__file__), fn) for fn in source_files]
    torch.utils.cpp_extension.load(name='hmesh', sources=source_paths, extra_cflags=cflags, extra_ldflags=ldflags, with_cuda=True, verbose=False)
    backend = importlib.import_module('hmesh')
    return backend
