#pragma once

#include "CUDABuffer.h"
#include "optix_types.h"
#include <torch/extension.h>

namespace hmesh {

struct OptixAccelStructureWrapperCPP {
private:
    OptixTraversableHandle asHandle = 0;

public:
    void buildAccelStructure(torch::Tensor vertices, torch::Tensor faces);
};

} // namespace hmesh
