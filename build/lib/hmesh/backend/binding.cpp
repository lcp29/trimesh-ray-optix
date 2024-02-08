
#include "ray.h"
#include <torch/extension.h>

namespace hmesh {

extern void initOptix();
extern void createOptixContext();
extern void createOptixModule();
extern torch::Tensor intersectsAny(torch::Tensor, torch::Tensor);

} // namespace hmesh


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<hmesh::OptixAccelStructureWrapperCPP>(m, "OptixAccelStructureWrapperCPP")
        .def(pybind11::init<>())
        .def("buildAccelStructure", &hmesh::OptixAccelStructureWrapperCPP::buildAccelStructure);

    m.def("initOptix", &hmesh::initOptix, "Initialize Optix");
    m.def("createOptixContext", &hmesh::createOptixContext, "Create Optix context");
    m.def("createOptixModule", &hmesh::createOptixModule, "Create Optix module");
    m.def("intersectsAny", &hmesh::intersectsAny, "Find out if each ray hit any triangle on the mesh.");
}
