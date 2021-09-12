#include <pybind11/pybind11.h>

namespace py = pybind11;

void initVector(py::module &);
void initField(py::module &);
void initSolver(py::module &);
void initMesh(py::module &);
void initMeshMotion(py::module &);

PYBIND11_MODULE(pyMach, m)
{
   initVector(m);
   initField(m);
   initSolver(m);
   initMesh(m);

   auto mesh_motion =
       m.def_submodule("MeshMovement", "Handles support for mesh movement");

   initMeshMotion(mesh_motion);
}