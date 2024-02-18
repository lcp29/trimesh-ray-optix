#pragma once

#include <vector_types.h>
#include <optix_types.h>

namespace hmesh {

constexpr int MAX_ANYHIT_SIZE = 8;
constexpr int MAX_SIZE_LENGTH = 4;

struct RayInput {
    // ray count
    size_t nray;
    // ray shape
    int64_t rayShape[MAX_SIZE_LENGTH];
    // ray origins
    float *origins;
    // ray origins stride
    int64_t originsStride[MAX_SIZE_LENGTH];
    // ray directions
    float *directions;
    // ray directions stride
    int64_t directionsStride[MAX_SIZE_LENGTH];
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
