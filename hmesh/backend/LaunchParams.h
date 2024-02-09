#pragma once

#include <optix_types.h>

namespace hmesh {

extern "C" struct LaunchParams {
    // ray count
    size_t nray;
    // ray origins
    float *origins;
    // ray directions
    float *dirs;
    // output buffer
    bool *result;

    // acceleration structure handle
    OptixTraversableHandle traversable;
};

} // namespace hmesh
