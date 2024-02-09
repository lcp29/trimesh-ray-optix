
#include "LaunchParams.h"
#include "optix_types.h"
#include <optix_device.h>

namespace hmesh {

extern "C" __constant__ LaunchParams launchParams;

// intersects_any

extern "C" __global__ void __miss__intersectsAny() {
    unsigned long long ptvalue =
        ((unsigned long long)optixGetPayload_1() << 32) + optixGetPayload_0();
    bool *result_pt = (bool *)ptvalue;
    *result_pt = false;
}

extern "C" __global__ void __closesthit__intersectsAny() {
    unsigned long long ptvalue =
        ((unsigned long long)optixGetPayload_1() << 32) + optixGetPayload_0();
    bool *result_pt = (bool *)ptvalue;
    *result_pt = true;
}

extern "C" __global__ void __raygen__intersectsAny() {
    // thread index, ranging in [0, N)
    int idx = optixGetLaunchIndex().x;
    // intersection result, to be overwritten by the shader
    bool isect_result = false;
    // ray info
    float3 ray_origin =
        *(float3 *)(launchParams.origins + idx * 3);
    float3 ray_dir = *(float3 *)(launchParams.dirs + idx * 3);
    // result pointer
    unsigned int u0 = (unsigned long long)(&isect_result) & 0xFFFFFFFFllu;
    unsigned int u1 =
        ((unsigned long long)(&isect_result) >> 32) & 0xFFFFFFFFllu;
    optixTrace(launchParams.traversable, ray_origin, ray_dir, 1e-4, 1e7, 0,
               OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, 0, 0, 0, u0, u1);
    launchParams.result[idx] = isect_result;
    printf("ray origin: %f, %f, %f, ray dir: %f, %f, %f, idx: %i\n",
           ray_origin.x, ray_origin.y, ray_origin.z,
           ray_dir.x, ray_dir.y, ray_dir.z, idx);
}

} // namespace hmesh
