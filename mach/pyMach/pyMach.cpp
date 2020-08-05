#include <pybind11/pybind11.h>

namespace py = pybind11;

void initVector(py::module &);
void initGridFunction(py::module &);
void initSolver(py::module &);
void initMesh(py::module &);

PYBIND11_MODULE(pyMach, m) {
   initVector(m);
   initGridFunction(m);
   initSolver(m);
   initMesh(m);
}