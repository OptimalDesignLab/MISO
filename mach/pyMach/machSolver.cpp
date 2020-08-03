#include <pybind11/pybind11.h>
#include <mpi4py/mpi4py.h>
#include <iostream>

#include "mfem.hpp"
#include "json.hpp"
#include "pybind11_json.hpp"

#include "mpi4py_comm.hpp"

#include "solver.hpp"
#include "magnetostatic.hpp"
#include "euler.hpp"

namespace py = pybind11;

using namespace mfem;
// using namespace nlohmann;
using namespace mach;


namespace
{
using SolverPtr = std::unique_ptr<AbstractSolver>;

/// \brief convert templated C++ factory method to one usable from python
/// \param type - the solver type to create
/// \param json_options - options dictionary containing solver options
/// \param comm - MPI communicator for parallel operations
/// \param entvar - flag to use the entropy variables in the integrators
SolverPtr initSolver(const std::string &type,
                     const nlohmann::json &json_options,
                     mpi4py_comm comm,
                     bool entvar)
{
   std::cout << "initSolver called with json options: " << json_options << "\n";
   if (type == "Magnetostatic")
   {
      return createSolver<MagnetostaticSolver>(json_options, nullptr, comm);
   }
   else if (type == "Euler")
   {
      if (entvar)
         return createSolver<EulerSolver<2, true>>(json_options, nullptr, comm);
      else
         return createSolver<EulerSolver<2, false>>(json_options, nullptr, comm);
   }
   else
   {
      throw std::runtime_error("Unknown solver type!\n"
      "\tKnown types are:\n"
      "\t\tMagnetostatic\n"
      "\t\tEuler\n");
   }
}

} // anonymous namespace

void initSolver(py::module &m)
{
   /// imports mpi4py's C interface
   if (import_mpi4py() < 0) return;

   py::class_<AbstractSolver>(m, "MachSolver")
      .def(py::init([](const std::string &type,
                       const std::string &opt_file_name,
                       mpi4py_comm comm,
                       bool entvar)
      {
         nlohmann::json json_options;
         std::ifstream options_file(opt_file_name);
         options_file >> json_options;
         return initSolver(type, json_options, comm, entvar);
         // setting keyword + default arguments
      }), py::arg("type"),
          py::arg("opt_file_name"),
          py::arg("comm") = mpi4py_comm(MPI_COMM_WORLD),
          py::arg("entvar") = false)

      .def(py::init([](const std::string &type,
                       const nlohmann::json &json_options,
                       mpi4py_comm comm,
                       bool entvar)
      {
         return initSolver(type, json_options, comm, entvar);
         // setting keyword + default arguments
      }), py::arg("type"),
          py::arg("json_options"),
          py::arg("comm") = mpi4py_comm(MPI_COMM_WORLD),
          py::arg("entvar") = false)

      .def("getMeshSize", &AbstractSolver::getMeshSize)
      .def("getMeshNodalCoordinates", &AbstractSolver::getMeshNodalCoordinates,
            py::return_value_policy::reference)
      .def("setMeshNodalCoordinates", &AbstractSolver::setMeshNodalCoordinates)

      .def("setInitialCondition", (void (AbstractSolver::*)
            (std::function<double(const mfem::Vector &)>))
            &AbstractSolver::setInitialCondition,
            "Initializes the state vector to a given scalar function.")

      /// TODO:
      // .def("calcResidual", &AbstractSolver::calcResidual)
      // .def("calcState", &AbstractSolver::calcState)
      // .def("multStateJacTranspose", &AbstractSolver::multStateJacTranspose)
      // .def("multMeshJacTranspose", &AbstractSolver::multMeshJacTranspose)
      // .def("invertStateJacTranspose", &AbstractSolver::invertStateJacTranspose)
   ;
}