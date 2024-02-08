
#include "embedded/shaders_embedded.h"
#include "optix8.h"
#include "optix_host.h"
#include "optix_types.h"
#include <iostream>
#include <optix_function_table_definition.h>
#include <torch/extension.h>

namespace hmesh {

CUcontext cuCtx;
CUstream cuStream;
OptixDeviceContext optixContext;
OptixModule optixModule;

static void context_log_cb(unsigned int level, const char *tag,
                           const char *message, void *) {
    std::cerr << "[" << (int)level << "][" << tag << "]: " << message << "\n";
}

void initOptix() { optixInit(); }

void createOptixContext() {
    cudaStreamCreate(&cuStream);
    CUresult res = cuCtxGetCurrent(&cuCtx);
    if (res != CUDA_SUCCESS)
        std::cerr << "Error getting current CUDA context: error code " << res
                  << "\n";
    optixDeviceContextCreate(cuCtx, nullptr, &optixContext);
    optixDeviceContextSetLogCallback(optixContext, context_log_cb, nullptr, 4);
}

void createOptixModule() {
    // create module
    OptixModuleCompileOptions moduleCompileOptions = {};
    moduleCompileOptions.maxRegisterCount =
        OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
    moduleCompileOptions.numPayloadTypes = 0;
    moduleCompileOptions.payloadTypes = nullptr;

    OptixPipelineCompileOptions pipelineCompileOptions = {};
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.traversableGraphFlags =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipelineCompileOptions.allowOpacityMicromaps = false;
    pipelineCompileOptions.numAttributeValues = 2;
    pipelineCompileOptions.numPayloadValues = 2;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.usesPrimitiveTypeFlags =
        OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "launchParams";

    char logString[2048];
    size_t logStringSize = 2048;
    OPTIX_CHECK(optixModuleCreate(optixContext, &moduleCompileOptions,
                                  &pipelineCompileOptions, (const char *)shader_code,
                                  shader_length, logString,
                                  &logStringSize, &optixModule));
}

} // namespace hmesh
