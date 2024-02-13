#pragma once

#include "optix8.h"
#include "sbtdef.h"

namespace hmesh {

extern CUstream cuStream;
extern OptixDeviceContext optixContext;
extern OptixShaderBindingTable sbts[SBTType::count];
extern OptixPipeline optixPipelines[SBTType::count];

} // namespace hmesh




