#include <pybind11/pybind11.h>

#include "utils.hpp"

namespace py = pybind11;
using namespace mach;

void initExceptions(py::module &m)
{
   py::register_exception<NotImplementedException>(
       m, "MachNotImplementedError", PyExc_NotImplementedError);
}
