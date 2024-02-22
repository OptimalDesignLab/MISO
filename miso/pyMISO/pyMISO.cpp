#include <pybind11/pybind11.h>

namespace py = pybind11;

void initExceptions(py::module &);
// void initField(py::module &);
void initSolver(py::module &);
// void initMesh(py::module &);
void initMeshWarper(py::module &);
void initPDESolver(py::module &);
// void initVector(py::module &);

PYBIND11_MODULE(pyMISO, m)
{
   initExceptions(m);
   // initField(m);
   initSolver(m);
   // initMesh(m);
   initMeshWarper(m);
   initPDESolver(m);
   // initVector(m);
}