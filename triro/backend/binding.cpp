
#include "ray.h"
#include <torch/extension.h>

namespace hmesh {

extern void initOptix();
extern void createOptixContext();
extern void createOptixModule();
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
extern void createPipelines();
extern void buildSBT();

} // namespace hmesh

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<hmesh::OptixAccelStructureWrapperCPP>(
        m, "OptixAccelStructureWrapperCPP")
        .def(pybind11::init<>())
        .def("buildAccelStructure",
             &hmesh::OptixAccelStructureWrapperCPP::buildAccelStructure)
        .def("freeAccelStructure",
             &hmesh::OptixAccelStructureWrapperCPP::freeAccelStructure);

    m.def("initOptix", &hmesh::initOptix, "Initialize Optix");
    m.def("createOptixContext", &hmesh::createOptixContext,
          "Create Optix context");
    m.def("createOptixModule", &hmesh::createOptixModule,
          "Create Optix module");
    m.def("createOptixPipelines", &hmesh::createPipelines,
          "Create Optix pipelines for each function type.");
    m.def("buildSBT", &hmesh::buildSBT, "Build SBT for each function type.");
    m.def("intersectsAny", &hmesh::intersectsAny,
          "Find out if each ray hit any triangle on the mesh.");
    m.def("intersectsFirst", &hmesh::intersectsFirst,
          "Find the index of the first triangle a ray hits.");
    m.def("intersectsClosest", &hmesh::intersectsClosest,
          "Find if ray hits any triangle and return ray index, triangle index, "
          "hit location and uv.");
    m.def("intersectsCount", &hmesh::intersectsCount,
          "Find the intersection count.");
    m.def("intersectsLocation", &hmesh::intersectsLocation,
          "Find all intersection locations.");
}
