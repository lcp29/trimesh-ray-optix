
#include "ray.h"
#include "CUDABuffer.h"
#include "optix8.h"
#include "optix_host.h"
#include "optix_types.h"
#include "type.h"

namespace hmesh {

extern CUstream cuStream;
extern OptixDeviceContext optixContext;

void OptixAccelStructureWrapperCPP::buildAccelStructure(torch::Tensor vertices,
                                                        torch::Tensor faces) {
    OptixAccelBuildOptions buildOptions = {};
    OptixBuildInput buildInput;

    // CUdeviceptr tempBuffer, outputBuffer;
    size_t tempBufferSizeInBytes, outputBufferSizeInBytes;

    memset(&buildOptions, 0, sizeof(buildOptions));
    buildOptions.buildFlags =
        OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
    buildOptions.motionOptions.numKeys = 0;

    memset(&buildInput, 0, sizeof(buildInput));

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

    uint64_t compactedSize;
    compactedSizeBuffer.download(&compactedSize, 1);

    OPTIX_CHECK(optixAccelCompact(optixContext, cuStream, asHandle,
                                  accelStructureBuffer.d_pointer(),
                                  compactedSize, &asHandle));

    CUDA_SYNC_CHECK();

    compactedSizeBuffer.free();
    tempBuffer.free();
    accelStructureBuffer.free();
}

torch::Tensor intersectsAny(torch::Tensor origins, torch::Tensor dirs) {
    
    return torch::Tensor();
}

} // namespace hmesh
