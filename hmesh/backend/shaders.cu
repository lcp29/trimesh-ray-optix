
#include "LaunchParams.h"
#include "optix_types.h"
#include <optix_device.h>
#include <tuple>

namespace hmesh {

extern "C" __constant__ LaunchParams launchParams;

__forceinline__ __device__ std::tuple<unsigned int, unsigned int>
setPayloadPointer(void *p) {
    unsigned int u0 = (unsigned long long)p & 0xFFFFFFFFllu;
    unsigned int u1 = ((unsigned long long)p >> 32) & 0xFFFFFFFFllu;
    return {u0, u1};
}

template <typename T>
__forceinline__ __host__ __device__ T *getPayloadPointer() {
    unsigned int u0 = optixGetPayload_0();
    unsigned int u1 = optixGetPayload_1();
    void *p = (void *)(((unsigned long long)u1 << 32) + u0);
    return (T *)p;
}

// intersects_any

extern "C" __global__ void __miss__intersectsAny() {
    bool *result_pt = getPayloadPointer<bool>();
    *result_pt = false;
}

extern "C" __global__ void __closesthit__intersectsAny() {
    bool *result_pt = getPayloadPointer<bool>();
    *result_pt = true;
}

extern "C" __global__ void __raygen__intersectsAny() {
    // thread index, ranging in [0, N)
    int idx = optixGetLaunchIndex().x;
    // intersection result, to be overwritten by the shader
    bool isect_result = false;
    // ray info
    float3 ray_origin = launchParams.rays.origins[idx];
    float3 ray_dir = launchParams.rays.directions[idx];
    // result pointer
    auto [u0, u1] = setPayloadPointer(&isect_result);
    optixTrace(launchParams.traversable, ray_origin, ray_dir, 1e-4, 1e7, 0,
               OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, 0, 0, 0, u0, u1);
    launchParams.results.hit[idx] = isect_result;
}

// intersects_first

extern "C" __global__ void __miss__intersectsFirst() {
    int *result_pt = getPayloadPointer<int>();
    *result_pt = -1;
}

extern "C" __global__ void __closesthit__intersectsFirst() {
    int *result_pt = getPayloadPointer<int>();
    *result_pt = optixGetPrimitiveIndex();
}

extern "C" __global__ void __raygen__intersectsFirst() {
    // thread index, ranging in [0, N)
    int idx = optixGetLaunchIndex().x;
    // first hit triangle index, to be overwritten by the shader
    int ch_idx = -1;
    // ray info
    float3 ray_origin = launchParams.rays.origins[idx];
    float3 ray_dir = launchParams.rays.directions[idx];
    // result pointer
    auto [u0, u1] = setPayloadPointer(&ch_idx);
    optixTrace(launchParams.traversable, ray_origin, ray_dir, 1e-4, 1e7, 0,
               OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, 0, 0, 0, u0, u1);
    launchParams.results.triIdx[idx] = ch_idx;
}

// intersects_closest
// todo

struct WBData {
    bool hit;
    bool front;
    int triIdx;
    float3 loc;
    float2 uv;
};

extern "C" __global__ void __miss__intersectsClosest() {
    WBData *result = getPayloadPointer<WBData>();
    result->hit = false;
    result->triIdx = -1;
    result->uv = {0, 0};
    result->loc = {0, 0, 0};
    result->front = false;
}

extern "C" __global__ void __closesthit__intersectsClosest() {
    WBData *result = getPayloadPointer<WBData>();
    float2 uv = optixGetTriangleBarycentrics();
    int triIdx = optixGetPrimitiveIndex();
    float3 verts[3];
    optixGetTriangleVertexData(launchParams.traversable, triIdx, 0, 0,
                               verts);
    float3 isectLoc = {
        uv.x * verts[0].x + uv.y * verts[1].x + (1 - uv.x - uv.y) * verts[2].x,
        uv.x * verts[0].y + uv.y * verts[1].y + (1 - uv.x - uv.y) * verts[2].y,
        uv.x * verts[0].z + uv.y * verts[1].z + (1 - uv.x - uv.y) * verts[2].z};

    result->triIdx = triIdx;
    result->uv = uv;
    result->hit = true;
    result->front = optixIsFrontFaceHit();
    result->loc = isectLoc;
}

extern "C" __global__ void __raygen__intersectsClosest() {
    // thread index, ranging in [0, N)
    int idx = optixGetLaunchIndex().x;
    WBData wbdata;
    // ray info
    float3 ray_origin = launchParams.rays.origins[idx];
    float3 ray_dir = launchParams.rays.directions[idx];
    // result pointer
    auto [u0, u1] = setPayloadPointer(&wbdata);
    optixTrace(launchParams.traversable, ray_origin, ray_dir, 1e-4, 1e7, 0,
               OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, 0, 0, 0, u0, u1);
    // write back to the buffers
    launchParams.results.hit[idx] = wbdata.hit;
    launchParams.results.front[idx] = wbdata.front;
    launchParams.results.location[idx] = wbdata.loc;
    launchParams.results.triIdx[idx] = wbdata.triIdx;
    launchParams.results.uv[idx] = wbdata.uv;
}

} // namespace hmesh
