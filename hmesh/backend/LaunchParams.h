#pragma once

#include <vector_types.h>
#include <optix_types.h>

namespace hmesh {

constexpr int MAX_ANYHIT_SIZE = 8;

struct RayInput {
    // ray count
    size_t nray;
    // ray origins
    float3 *origins;
    // ray directions
    float3 *directions;
    // hit counts
    int *hitCounts;
    // global index
    int *globalIdx;
};

struct LPResult {
    float3 *location;
    int *triIdx;
    bool *hit;
    bool *front;
    float2 *uv;
    int *hitCount;
    int *rayIdx;
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
