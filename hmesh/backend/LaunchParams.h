#pragma once

#include <optix_types.h>

namespace hmesh {

extern "C" struct LaunchParams {
    // ray count
    size_t nray;
    // ray origins
    CUdeviceptr origins;
    // ray directions
    CUdeviceptr dirs;
    // output buffer
    CUdeviceptr result;
};

} // namespace hmesh
