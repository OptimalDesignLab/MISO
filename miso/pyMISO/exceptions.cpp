#include <pybind11/pybind11.h>

#include "utils.hpp"

namespace py = pybind11;
using namespace miso;

void initExceptions(py::module &m)
{
   py::register_exception<NotImplementedException>(
       m, "MISONotImplementedError", PyExc_NotImplementedError);
}
