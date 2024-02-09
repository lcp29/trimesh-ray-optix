
import hmesh.backend.ops as hops

# initialize optix
hops.init_optix()
hops.create_optix_context()
hops.create_optix_module()
hops.create_optix_pipelines()
hops.build_sbts()
