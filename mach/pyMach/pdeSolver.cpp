#include <iostream>
#include <memory>
#include <utility>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "mfem.hpp"
#include "nlohmann/json.hpp"
#include "pybind11_json.hpp"

#include "abstract_solver.hpp"
#include "mach_input.hpp"
#include "magnetostatic.hpp"
#include "mpi_comm.hpp"
#include "py_mach_utils.hpp"
#include "thermal.hpp"

namespace py = pybind11;
using namespace mach;

namespace
{
/// \brief Construct a PDE solver based on the given type and options
/// \param solver_type - dictionary containing solver type info
/// \param solver_options - options dictionary containing solver options
/// \param comm - MPI communicator for parallel operations
std::unique_ptr<PDESolver> initSolver(nlohmann::json solver_type,
                                      nlohmann::json solver_options,
                                      MPI_Comm comm)
{
   solver_options["solver-type"] = solver_type;
   auto type = solver_type["type"].get<std::string>();

   if (type == "magnetostatic")
   {
      return std::make_unique<MagnetostaticSolver>(
          comm, solver_options, nullptr);
   }
   else if (type == "thermal")
   {
      return std::make_unique<ThermalSolver>(comm, solver_options, nullptr);
   }
   else
   {
      throw std::runtime_error(
          "Unknown solver type! Known types are:\n"
          "\tmagnetostatic\n"
          "\tthermal\n");
   }
}

/// \brief Construct a PDE solver based on the given type and options
/// \param type - string indicating PDE solver type
/// \param solver_options - options dictionary containing solver options
/// \param comm - MPI communicator for parallel operations
std::unique_ptr<PDESolver> initSolver(std::string type,
                                      nlohmann::json solver_options,
                                      MPI_Comm comm)
{
   nlohmann::json solver_type{{"type", std::move(type)}};
   return initSolver(std::move(solver_type), std::move(solver_options), comm);
}

}  // anonymous namespace

void initPDESolver(py::module &m)
{
   /// imports mpi4py's C interface
   if (import_mpi4py() < 0)
   {
      return;
   }

   py::class_<PDESolver, AbstractSolver2>(m, "PDESolver")
       .def(py::init(
                [](std::string type,
                   nlohmann::json solver_options,
                   mpi_comm comm) {
                   return initSolver(
                       std::move(type), std::move(solver_options), comm);
                }),
            py::arg("type"),
            py::arg("solver_options"),
            py::arg("comm") = mpi_comm(MPI_COMM_WORLD))
       .def(py::init(
                [](nlohmann::json type,
                   nlohmann::json solver_options,
                   mpi_comm comm) {
                   return initSolver(
                       std::move(type), std::move(solver_options), comm);
                }),
            py::arg("type"),
            py::arg("solver_options"),
            py::arg("comm") = mpi_comm(MPI_COMM_WORLD))
       .def("getNumStates", &PDESolver::getNumStates)
       .def(
           "getMeshCoordinates",
           [](PDESolver &self, const py::array_t<double> &mesh_coords)
           {
              auto mesh_coords_vec = npBufferToMFEMVector(mesh_coords);
              return self.getMeshCoordinates(mesh_coords_vec);
           },
           py::arg("mesh_coords"));
}
