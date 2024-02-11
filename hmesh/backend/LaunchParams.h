#pragma once

#include <optix_types.h>

namespace hmesh {

struct RayInput {
    // ray count
    size_t nray;
    // ray origins
    float *origins;
    // ray directions
    float *directions;
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
