#pragma once

#include <vector_types.h>
#include <optix_types.h>

namespace hmesh {

struct RayInput {
    // ray count
    size_t nray;
    // ray origins
    float3 *origins;
    // ray directions
    float3 *directions;
};

struct LPResult {
    float3 *location;
    int *triIdx;
    bool *hit;
    float2 *uv;
};

extern "C" struct LaunchParams {
    // input ray info
    RayInput rays;
    // output buffer
    LPResult results;
    // acceleration structure handle
    OptixTraversableHandle traversable;
};

} // namespace hmesh
