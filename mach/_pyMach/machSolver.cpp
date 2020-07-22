#include <pybind11/pybind11.h>

#include "mfem.hpp"

#include "mpi4py_comm.hpp"

#include "solver.hpp"
#include "magnetostatic.hpp"
#include "euler.hpp"

namespace py = pybind11;

using namespace mfem;
using namespace mach;

void init_solver(py::module &m)
{
   py::class_<AbstractSolver>(m, "machSolver")
      .def(py::init([](const std::string &type,
                       const std::string &opt_file_name,
                       mpi4py_comm comm)
      {
         if (type == "Magnetostatic")
         {
            return createSolver<MagnetostaticSolver>(opt_file_name, nullptr, comm);
         }
         else if (type == "Euler")
         {
            return createSolver<EulerSolver<2, false>>(opt_file_name, nullptr, comm);
         }
         else
         {
            throw std::runtime_error("Unknown solver type!\n"
            "\tKnown types are:\n"
            "\t\tMagnetostatic\n"
            "\t\tEuler\n");
         }
      }))
      .def("getMeshSize", &AbstractSolver::getMeshSize)
      .def("getMeshNodalCoordinates", &AbstractSolver::getMeshNodalCoordinates,
            py::return_value_policy::reference)
      .def("setMeshNodalCoordinates", &AbstractSolver::setMeshNodalCoordinates)

      /// TODO:
      // .def("calcResidual", &AbstractSolver::calcResidual)
      // .def("calcState", &AbstractSolver::calcState)
      // .def("multStateJacTranspose", &AbstractSolver::multStateJacTranspose)
      // .def("multMeshJacTranspose", &AbstractSolver::multMeshJacTranspose)
      // .def("invertStateJacTranspose", &AbstractSolver::invertStateJacTranspose)
   ;
}