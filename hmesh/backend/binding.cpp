
#include "ray.h"
#include <torch/extension.h>

namespace hmesh {

extern void initOptix();
extern void createOptixContext();
extern void createOptixModule();
extern torch::Tensor intersectsAny(OptixAccelStructureWrapperCPP, torch::Tensor,
                                   torch::Tensor);
extern torch::Tensor intersectsFirst(OptixAccelStructureWrapperCPP,
                                     torch::Tensor, torch::Tensor);
extern void createPipelines();
extern void buildSBT();

} // namespace hmesh

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<hmesh::OptixAccelStructureWrapperCPP>(
        m, "OptixAccelStructureWrapperCPP")
        .def(pybind11::init<>())
        .def("buildAccelStructure",
             &hmesh::OptixAccelStructureWrapperCPP::buildAccelStructure);

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
}
