#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_vector(py::module &);
void init_solver(py::module &);

PYBIND11_MODULE(_pyMach, m) {
   init_vector(m);
   init_solver(m);
}