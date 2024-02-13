/**
 * @file ray.cpp
 * @author helmholtz
 * @brief evrything per-instance
 *
 */

#include "ray.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/where.h"
#include "CUDABuffer.h"
#include "LaunchParams.h"
#include "base.h"
#include "c10/core/ScalarType.h"
#include "c10/core/TensorOptions.h"
#include "c10/util/ArrayRef.h"
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

    buildOptions.buildFlags = OPTIX_BUILD_FLAG_NONE |
                              OPTIX_BUILD_FLAG_ALLOW_COMPACTION |
                              OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
    buildOptions.motionOptions.numKeys = 0;

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

    uint32_t triangleBuildFlags =
        OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL;
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
                                  asBuffer.d_pointer(), compactedSize,
                                  &asHandle));

    CUDA_SYNC_CHECK();

    compactedSizeBuffer.free();
    tempBuffer.free();
    accelStructureBuffer.free();
}

void OptixAccelStructureWrapperCPP::freeAccelStructure() { asBuffer.free(); }

template <typename... Ts> inline bool tensorInputCheck(Ts... ts) {
    bool valid = true;
    (
        [&] {
            if (!ts.is_cuda()) {
                std::cerr << "error in file " << __FILE__ << " line "
                          << __LINE__
                          << ": input tensors must reside in cuda device.\n";
                valid = false;
            }
            if (!ts.is_contiguous()) {
                std::cerr << "error in file " << __FILE__ << " line "
                          << __LINE__
                          << ": input tensors must be contiguous.\n";
                valid = false;
            }
        }(),
        ...);
    return valid;
}

inline std::vector<long> removeLastDim(const c10::IntArrayRef dims) {
    auto ref = dims.vec();
    ref.pop_back();
    return ref;
}

inline size_t prod(const std::vector<long> &dims) {
    size_t p = 1;
    for (auto s : dims)
        p *= s;
    return p;
}

inline std::vector<long> changeLastDim(const c10::IntArrayRef dims,
                                       size_t value) {
    std::vector<long> dimsVec;
    for (auto s : dims)
        dimsVec.push_back(s);
    *(dimsVec.end() - 1) = value;
    return dimsVec;
}

template <typename T> inline T *data_ptr(torch::Tensor t) {
    return (T *)t.data_ptr();
}

torch::Tensor intersectsAny(OptixAccelStructureWrapperCPP as,
                            const torch::Tensor &origins,
                            const torch::Tensor &directions) {
    if (!tensorInputCheck(origins, directions))
        return {};
    // output buffer
    auto options =
        torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA);
    auto resultSize = removeLastDim(origins.sizes());
    auto nray = prod(resultSize);
    auto result = torch::empty(resultSize, options);
    // fill launch params
    LaunchParams lp = {};
    lp.rays.origins = data_ptr<float3>(origins);
    lp.rays.directions = data_ptr<float3>(directions);
    lp.rays.nray = nray;
    lp.traversable = as.asHandle;
    lp.results.hit = data_ptr<bool>(result);
    CUDABuffer lpBuffer;
    lpBuffer.alloc_and_upload(&lp, 1);
    optixLaunch(optixPipelines[SBTType::INTERSECTS_ANY], cuStream,
                lpBuffer.d_pointer(), sizeof(lp),
                &sbts[SBTType::INTERSECTS_ANY], lp.rays.nray, 1, 1);
    lpBuffer.free();
    return result;
}

torch::Tensor intersectsFirst(OptixAccelStructureWrapperCPP as,
                              const torch::Tensor &origins,
                              const torch::Tensor &directions) {
    if (!tensorInputCheck(origins, directions))
        return {};
    // output buffer
    auto options =
        torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA);
    auto resultSize = removeLastDim(origins.sizes());
    auto nray = prod(resultSize);
    auto result = torch::empty(resultSize, options);
    // fill launch params
    LaunchParams lp = {};
    lp.rays.origins = data_ptr<float3>(origins);
    lp.rays.directions = data_ptr<float3>(directions);
    lp.rays.nray = nray;
    lp.traversable = as.asHandle;
    lp.results.triIdx = data_ptr<int>(result);
    CUDABuffer lpBuffer;
    lpBuffer.alloc_and_upload(&lp, 1);
    optixLaunch(optixPipelines[SBTType::INTERSECTS_FIRST], cuStream,
                lpBuffer.d_pointer(), sizeof(lp),
                &sbts[SBTType::INTERSECTS_FIRST], lp.rays.nray, 1, 1);
    lpBuffer.free();
    return result;
}

/**
 * @brief Find if ray hits any triangle and return ray index, triangle index,
 * hit location and uv.
 *
 * @param as
 * @param origins
 * @param directions
 * @return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
 * torch::Tensor, torch::Tensor>
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
intersectsClosest(OptixAccelStructureWrapperCPP as, torch::Tensor origins,
                  torch::Tensor directions) {
    if (!tensorInputCheck(origins, directions))
        return {};
    // output buffers
    // hitmask buffer
    auto hitbufOptions =
        torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA);
    auto hitbufSize = removeLastDim(origins.sizes());
    auto hitbuf = torch::empty(hitbufSize, hitbufOptions);
    // front hit buffer
    auto frontbuf = torch::empty(hitbufSize, hitbufOptions);
    // triangle index buffer
    auto tibufOptions =
        torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA);
    auto tibufSize = removeLastDim(origins.sizes());
    auto tibuf = torch::empty(tibufSize, tibufOptions);
    // intersect location buffer
    auto locbufOptions =
        torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    auto locbufSize = changeLastDim(origins.sizes(), 3);
    auto locbuf = torch::empty(locbufSize, locbufOptions);
    // uv buffer
    auto uvbufOptions =
        torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    auto uvbufSize = changeLastDim(origins.sizes(), 2);
    auto uvbuf = torch::empty(uvbufSize, uvbufOptions);
    auto nray = prod(hitbufSize);

    // fill and upload launchParams
    LaunchParams lp = {};
    lp.rays.nray = nray;
    lp.rays.origins = data_ptr<float3>(origins);
    lp.rays.directions = data_ptr<float3>(directions);

    lp.results.hit = data_ptr<bool>(hitbuf);
    lp.results.location = data_ptr<float3>(locbuf);
    lp.results.triIdx = data_ptr<int>(tibuf);
    lp.results.uv = data_ptr<float2>(uvbuf);
    lp.results.front = data_ptr<bool>(frontbuf);

    lp.traversable = as.asHandle;

    CUDABuffer lpBuffer;
    lpBuffer.alloc_and_upload(&lp, 1);

    // 启动！
    optixLaunch(optixPipelines[SBTType::INTERSECTS_CLOSEST], cuStream,
                lpBuffer.d_pointer(), sizeof(lp),
                &sbts[SBTType::INTERSECTS_CLOSEST], nray, 1, 1);

    lpBuffer.free();
    return {hitbuf, frontbuf, tibuf, locbuf, uvbuf};
}

torch::Tensor intersectsCount(OptixAccelStructureWrapperCPP as,
                              torch::Tensor origins, torch::Tensor directions) {
    if (!tensorInputCheck(origins, directions))
        return {};
    // first pass - get the intersection count
    auto hitCountBufSize = removeLastDim(origins.sizes());
    auto hitCountBufOptions =
        torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA);
    auto hitCountBuf =
        torch::zeros(hitCountBufSize, hitCountBufOptions).contiguous();
    auto nray = prod(hitCountBufSize);

    LaunchParams lp = {};
    lp.rays.nray = nray;
    lp.rays.origins = data_ptr<float3>(origins);
    lp.rays.directions = data_ptr<float3>(directions);
    lp.results.hitCount = data_ptr<int>(hitCountBuf);
    lp.traversable = as.asHandle;

    CUDABuffer lpBuffer;
    lpBuffer.alloc_and_upload(&lp, 1);

    optixLaunch(optixPipelines[SBTType::INTERSECTS_COUNT], cuStream,
                lpBuffer.d_pointer(), sizeof(lp),
                &sbts[SBTType::INTERSECTS_COUNT], nray, 1, 1);

    lpBuffer.free();
    return hitCountBuf;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
intersectsLocation(OptixAccelStructureWrapperCPP as, torch::Tensor origins,
                   torch::Tensor directions) {
    if (!tensorInputCheck(origins, directions))
        return {};
    // first pass - get the intersection count
    auto hitCountBuf = intersectsCount(as, origins, directions);

    // second pass
    hitCountBuf = hitCountBuf.flatten();
    hitCountBuf = torch::where(hitCountBuf <= MAX_ANYHIT_SIZE, hitCountBuf,
                               MAX_ANYHIT_SIZE);
    auto globalIdxBuf = hitCountBuf.cumsum(0);
    auto globalIdxBufOptions =
        torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA);
    auto nhits = globalIdxBuf[-1].item<int>();
    globalIdxBuf = torch::cat({torch::zeros({1}, globalIdxBufOptions),
                               torch::slice(hitCountBuf, 0, 0, -1)});
    // hit location
    auto locbufOptions =
        torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    auto locbuf = torch::empty({nhits, 3}, locbufOptions);
    auto idxbufOptions =
        torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA);
    auto tibuf = torch::empty({nhits}, idxbufOptions);
    auto ribuf = torch::empty({nhits}, idxbufOptions);

    auto nray = prod(removeLastDim(origins.sizes()));

    LaunchParams lp = {};
    lp.traversable = as.asHandle;
    lp.rays.nray = nray;
    lp.rays.origins = data_ptr<float3>(origins);
    lp.rays.directions = data_ptr<float3>(directions);
    lp.rays.hitCounts = data_ptr<int>(hitCountBuf);
    lp.rays.globalIdx = data_ptr<int>(globalIdxBuf);
    lp.results.hitCount = data_ptr<int>(hitCountBuf);
    lp.results.location = data_ptr<float3>(locbuf);
    lp.results.triIdx = data_ptr<int>(tibuf);
    lp.results.rayIdx = data_ptr<int>(ribuf);

    CUDABuffer lpBuffer;
    lpBuffer.alloc_and_upload(&lp, 1);

    optixLaunch(optixPipelines[SBTType::INTERSECTS_LOCATION],
                cuStream, lpBuffer.d_pointer(), sizeof(lp),
                &sbts[SBTType::INTERSECTS_LOCATION], nray, 1, 1);

    lpBuffer.free();
    return {locbuf, ribuf, tibuf};
}

} // namespace hmesh
