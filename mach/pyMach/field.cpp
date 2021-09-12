#include <string>
#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>

#include "mfem.hpp"

namespace py = pybind11;

using namespace mfem;

void initField(py::module &m)
{
   py::class_<HypreParVector, Vector>(m, "Field");          // NOLINT
   py::class_<ParGridFunction, Vector>(m, "GridFunction");  // NOLINT
}
