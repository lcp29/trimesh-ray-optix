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
    float *location;
    int *triIdx;
    bool *hit;
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
