#include <pybind11/pybind11.h>

void createOptixContext();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("createOptixContext", &createOptixContext, "Initialize Optix and create Optix context");
}
