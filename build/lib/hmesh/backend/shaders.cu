
#include "LaunchParams.h"
#include <optix_device.h>

namespace hmesh {

extern "C" __constant__ LaunchParams launchParams;

// intersects_any

extern "C" __global__ void __miss__intersectsAny() {

}

extern "C" __global__ void __anyhit__intersectsAny() {

}

extern "C" __global__ void __raygen__intersectsAny() {
    // thread index, ranging in [0, N)
    int idx = optixGetLaunchIndex().x;

    // intersection result, to be overwritten by the shader
    bool isect_result = false;

    printf("%lu", sizeof(isect_result));
}

} // namespace hmesh
