
#include "CUDABuffer.h"
#include "embedded/shaders_embedded.h"
#include "optix8.h"
#include "optix_host.h"
#include "optix_types.h"
#include "sbtdef.h"
#include <cstddef>
#include <iostream>
#include <optix_function_table_definition.h>
#include <torch/extension.h>

namespace hmesh {

CUcontext cuCtx;
CUstream cuStream;
OptixDeviceContext optixContext;
OptixModule optixModule;
OptixPipelineCompileOptions pipelineCompileOptions = {};

// SBTs and pipelines for each function
OptixShaderBindingTable sbts[SBTType::count];
OptixPipeline optixPipelines[SBTType::count];
OptixProgramGroup optixProgramGroups[SBTType::count][3];

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

    pipelineCompileOptions.usesMotionBlur = false;
    // !must be ALLOW_SINGLE_GAS for only one GAS
    pipelineCompileOptions.traversableGraphFlags =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.allowOpacityMicromaps = false;
    // 0: hitT
    // 1: hitKind
    pipelineCompileOptions.numAttributeValues = 2;
    // 0: resultpointer low 32 bits
    // 1: resultpointer high 32 bits
    pipelineCompileOptions.numPayloadValues = 2;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.usesPrimitiveTypeFlags =
        OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "launchParams";

    char logString[2048];
    size_t logStringSize = 2048;
    OPTIX_CHECK(optixModuleCreate(optixContext, &moduleCompileOptions,
                                  &pipelineCompileOptions,
                                  (const char *)shader_code, shader_length,
                                  logString, &logStringSize, &optixModule));
}

void createPipelines() {
    for (int t = 0; t < SBTType::count; t++) {
        // program affix
        const std::string &prgName = std::get<0>(programInfos[t]);
        const std::string raygenName = std::string("__raygen__") + prgName;
        const std::string anyhitName = std::string("__anyhit__") + prgName;
        const std::string closesthitName =
            std::string("__closesthit__") + prgName;
        const std::string intersectionName =
            std::string("__intersection__") + prgName;
        const std::string missName = std::string("__miss__") + prgName;
        int prgMask = std::get<1>(programInfos[t]);
        // program group descriptors
        // { RAYGEN, HITGROUP, MISS }
        OptixProgramGroupDesc pgDescs[3] = {};
        // raygen program group
        pgDescs[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        pgDescs[0].raygen.module = optixModule;
        pgDescs[0].raygen.entryFunctionName = raygenName.c_str();
        // hitgroup program group
        pgDescs[1].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        if (prgMask & PRG_AH) {
            pgDescs[1].hitgroup.moduleAH = optixModule;
            pgDescs[1].hitgroup.entryFunctionNameAH = anyhitName.c_str();
        }
        if (prgMask & PRG_CH) {
            pgDescs[1].hitgroup.moduleCH = optixModule;
            pgDescs[1].hitgroup.entryFunctionNameCH = closesthitName.c_str();
        }
        if (prgMask & PRG_IS) {
            pgDescs[1].hitgroup.moduleIS = optixModule;
            pgDescs[1].hitgroup.entryFunctionNameIS = intersectionName.c_str();
        }
        // miss program group
        pgDescs[2].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        if (prgMask & PRG_MS) {
            pgDescs[2].miss.module = optixModule;
            pgDescs[2].miss.entryFunctionName = missName.c_str();
        }
        // program group options
        OptixProgramGroupOptions pgOptions[3] = {};
        // create program group
        char logString[2048];
        size_t logStringSize = 2048;
        OptixProgramGroup *pg = optixProgramGroups[t];
        OPTIX_CHECK(optixProgramGroupCreate(optixContext, pgDescs, 3, pgOptions,
                                            logString, &logStringSize, pg));
        // create pipeline
        OptixPipelineLinkOptions pipelineLinkOptions = {};
        pipelineLinkOptions.maxTraceDepth = 1;

        OPTIX_CHECK(optixPipelineCreate(optixContext, &pipelineCompileOptions,
                                        &pipelineLinkOptions, pg, 3, logString,
                                        &logStringSize, &optixPipelines[t]));
    }
}

void buildSBT() {
    for (int t = 0; t < SBTType::count; t++) {
        // create SBT header
        SBTRecordEmpty tmpRec;
        CUDABuffer rgRecDevice;
        optixSbtRecordPackHeader(optixProgramGroups[t][0], &tmpRec);
        rgRecDevice.alloc_and_upload(&tmpRec, 1);
        CUDABuffer hgRecDevice;
        optixSbtRecordPackHeader(optixProgramGroups[t][1], &tmpRec);
        hgRecDevice.alloc_and_upload(&tmpRec, 1);
        CUDABuffer msRecDevice;
        optixSbtRecordPackHeader(optixProgramGroups[t][2], &tmpRec);
        msRecDevice.alloc_and_upload(&tmpRec, 1);
        // fill sbt
        OptixShaderBindingTable &sbt = sbts[t];
        sbt.raygenRecord = rgRecDevice.d_pointer();
        sbt.hitgroupRecordBase = hgRecDevice.d_pointer();
        sbt.hitgroupRecordCount = 1;
        sbt.hitgroupRecordStrideInBytes = sizeof(SBTRecordEmpty);
        sbt.missRecordBase = msRecDevice.d_pointer();
        sbt.missRecordCount = 1;
        sbt.missRecordStrideInBytes = sizeof(SBTRecordEmpty);
    }
}

} // namespace hmesh
