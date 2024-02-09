#pragma once

#include "CUDABuffer.h"
#include "base.h"
#include "optix_types.h"
#include <torch/extension.h>

namespace hmesh {

struct OptixAccelStructureWrapperCPP {
    OptixTraversableHandle asHandle = 0;
    CUDABuffer asBuffer;
    void buildAccelStructure(torch::Tensor vertices, torch::Tensor faces);
};

extern torch::Tensor intersectsAny(OptixAccelStructureWrapperCPP as,
                                   torch::Tensor origins, torch::Tensor dirs);

} // namespace hmesh
