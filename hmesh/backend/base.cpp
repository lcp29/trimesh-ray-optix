
#include "optix8.h"
#include <optix_function_table_definition.h>
#include <torch/extension.h>
#include <iostream>

namespace hmesh {

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

void initOptix()
{
    optixInit();
}

void createOptixContext()
{
    cudaStreamCreate(&cuStream);
    CUresult res = cuCtxGetCurrent(&cuCtx);
    if (res != CUDA_SUCCESS)
        std::cerr << "Error getting current CUDA context: error code " << res << "\n";
    optixDeviceContextCreate(cuCtx, nullptr, &optixContext);
    optixDeviceContextSetLogCallback(optixContext, context_log_cb, nullptr, 4);
}

void buildAcclerationStructure(torch::Tensor vertices, torch::Tensor faces)
{
    
}

} // namespace hmesh
