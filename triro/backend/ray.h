#pragma once

#include "CUDABuffer.h"
#include "base.h"
#include "optix_types.h"
#include <torch/extension.h>
#include <limits>

namespace hmesh {

struct OptixAccelStructureWrapperCPP {
    OptixTraversableHandle asHandle = 0;
    CUDABuffer asBuffer;
    void buildAccelStructure(torch::Tensor vertices, torch::Tensor faces);
    void freeAccelStructure();
};

extern torch::Tensor intersectsAny(OptixAccelStructureWrapperCPP as,
                                   const torch::Tensor &origins,
                                   const torch::Tensor &dirs);
extern torch::Tensor intersectsFirst(OptixAccelStructureWrapperCPP as,
                                     const torch::Tensor &origins,
                                     const torch::Tensor &dirs);
extern std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                  torch::Tensor>
intersectsClosest(OptixAccelStructureWrapperCPP as, torch::Tensor origins,
                  torch::Tensor directions);
extern torch::Tensor intersectsCount(OptixAccelStructureWrapperCPP as,
                                     torch::Tensor origins,
                                     torch::Tensor directions);
extern std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
intersectsLocation(OptixAccelStructureWrapperCPP as, torch::Tensor origins,
                   torch::Tensor directions);

} // namespace hmesh
