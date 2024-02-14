
import triro.backend.ops as hops

import os

if os.name == 'nt':
    # add cl.exe to path
    from setuptools import msvc
    msvc_env = msvc.EnvironmentInfo('amd64')
    msvc_dir_candidates = msvc_env.VCTools
    for d in msvc_dir_candidates:
        if 'bin' in d:
            msvc_dir = d
            break
    os.environ['Path'] = os.environ['Path'] + f';{msvc_dir}'

# initialize optix
hops.init_optix()
hops.create_optix_context()
hops.create_optix_module()
hops.create_optix_pipelines()
hops.build_sbts()
