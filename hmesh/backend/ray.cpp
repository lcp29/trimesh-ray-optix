/**
 * @file ray.cpp
 * @author helmholtz
 * @brief evrything per-instance
 *
 */

#include "ray.h"
#include "CUDABuffer.h"
#include "LaunchParams.h"
#include "base.h"
#include "c10/core/ScalarType.h"
#include "c10/core/TensorOptions.h"
#include "optix8.h"
#include "optix_host.h"
#include "optix_types.h"
#include "sbtdef.h"
#include "type.h"

namespace hmesh {

void OptixAccelStructureWrapperCPP::buildAccelStructure(torch::Tensor vertices,
                                                        torch::Tensor faces) {
    OptixAccelBuildOptions buildOptions = {};
    OptixBuildInput buildInput = {};

    // CUdeviceptr tempBuffer, outputBuffer;
    size_t tempBufferSizeInBytes, outputBufferSizeInBytes;

    buildOptions.buildFlags =
        OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
    buildOptions.motionOptions.numKeys = 1;

    CUdeviceptr pVert = (CUdeviceptr)vertices.data_ptr();
    CUdeviceptr pFace = (CUdeviceptr)faces.data_ptr();

    buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    buildInput.triangleArray.vertexBuffers = &pVert;
    buildInput.triangleArray.numVertices = vertices.size(0);
    buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    buildInput.triangleArray.vertexStrideInBytes = sizeof(vec3f);
    buildInput.triangleArray.indexBuffer = pFace;
    buildInput.triangleArray.numIndexTriplets = faces.size(0);
    buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    buildInput.triangleArray.indexStrideInBytes = sizeof(vec3i);
    buildInput.triangleArray.preTransform = 0;

    buildInput.triangleArray.numSbtRecords = 1;
    buildInput.triangleArray.sbtIndexOffsetBuffer = 0;
    buildInput.triangleArray.sbtIndexOffsetSizeInBytes = 0;
    buildInput.triangleArray.sbtIndexOffsetStrideInBytes = 0;

    uint32_t triangleBuildFlags = 0;
    buildInput.triangleArray.flags = &triangleBuildFlags;

    OptixAccelBufferSizes bufferSizes = {};
    OPTIX_CHECK(optixAccelComputeMemoryUsage(optixContext, &buildOptions,
                                             &buildInput, 1, &bufferSizes));

    CUDABuffer tempBuffer;
    CUDABuffer accelStructureBuffer;
    accelStructureBuffer.alloc(bufferSizes.outputSizeInBytes);
    tempBuffer.alloc(bufferSizes.tempSizeInBytes);

    CUDABuffer compactedSizeBuffer;
    compactedSizeBuffer.alloc(sizeof(uint64_t));
    OptixAccelEmitDesc emitDesc;
    emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = compactedSizeBuffer.d_pointer();

    OPTIX_CHECK(optixAccelBuild(
        optixContext, cuStream, &buildOptions, &buildInput, 1,
        (CUdeviceptr)tempBuffer.d_ptr, tempBuffer.sizeInBytes,
        (CUdeviceptr)accelStructureBuffer.d_ptr,
        accelStructureBuffer.sizeInBytes, &asHandle, &emitDesc, 1));

    CUDA_SYNC_CHECK();

    uint64_t compactedSize;
    compactedSizeBuffer.download(&compactedSize, 1);
    asBuffer.resize(compactedSize);

    OPTIX_CHECK(optixAccelCompact(optixContext, cuStream, asHandle,
                                  asBuffer.d_pointer(),
                                  compactedSize, &asHandle));

    CUDA_SYNC_CHECK();

    compactedSizeBuffer.free();
    tempBuffer.free();
    accelStructureBuffer.free();
}

torch::Tensor intersectsAny(OptixAccelStructureWrapperCPP as,
                            torch::Tensor origins, torch::Tensor dirs) {
    if (!(origins.is_cuda() && dirs.is_cuda())) {
        std::cerr << "error in file " << __FILE__ << " line " << __LINE__
                  << ": input tensors must reside on cuda device.\n";
        return torch::Tensor();
    }
    // output buffer
    auto options =
        torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA);
    auto result = torch::empty({origins.size(0)}, options);
    // fill launch params
    LaunchParams lp = {};
    lp.origins = origins.data_ptr<float>();
    lp.dirs = dirs.data_ptr<float>();
    lp.nray = origins.size(0);
    lp.traversable = as.asHandle;
    lp.result = result.data_ptr<bool>();
    CUDABuffer lpBuffer;
    lpBuffer.alloc_and_upload(&lp, 1);
    optixLaunch(optixPipelines[SBTType::INTERSECTS_ANY], cuStream,
                lpBuffer.d_pointer(), sizeof(LaunchParams),
                &sbts[SBTType::INTERSECTS_ANY], lp.nray, 1, 1);
    lpBuffer.free();
    return result;
}

} // namespace hmesh
