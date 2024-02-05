
#include "CUDABuffer.h"
#include "optix8.h"
#include "optix_host.h"
#include "optix_types.h"
#include <iostream>

CUcontext cuCtx;
CUstream cuStream;
OptixDeviceContext optixContext;

static void context_log_cb(unsigned int level,
                           const char *tag,
                           const char *message,
                           void *)
{
    std::cerr << "[" << (int)level << "][" << tag << "]: " << message << "\n";
}

void createOptixContext()
{
    OPTIX_CHECK(optixInit());
    CUDA_CHECK(StreamCreate(&cuStream));
    CUresult res = cuCtxGetCurrent(&cuCtx);
    if (res != CUDA_SUCCESS)
        std::cerr << "Error getting current CUDA context: error code " << res << "\n";
    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, nullptr, &optixContext));
    OPTIX_CHECK(optixDeviceContextSetLogCallback(optixContext, context_log_cb, nullptr, 4))
}
