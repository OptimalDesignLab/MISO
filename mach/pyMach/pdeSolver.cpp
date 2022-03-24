#include <iostream>
#include <memory>
#include <utility>

#include <pybind11/pybind11.h>

#include "mfem.hpp"
#include "nlohmann/json.hpp"
#include "pybind11_json.hpp"

#include "abstract_solver.hpp"
#include "mach_input.hpp"
#include "magnetostatic.hpp"
#include "mpi_comm.hpp"

namespace py = pybind11;
using namespace mach;

namespace
{
/// \brief Construct a PDE solver based on the given options
/// \param json_options - options dictionary containing solver options
/// \param comm - MPI communicator for parallel operations
std::unique_ptr<PDESolver> initSolver(const nlohmann::json &json_options,
                                      MPI_Comm comm)
{
   std::string solver_name;
   if (json_options["solver"].type() == nlohmann::json::value_t::string)
   {
      solver_name = json_options["solver"].get<std::string>();
   }
   else if (json_options["solver"].type() == nlohmann::json::value_t::object)
   {
      solver_name = json_options["solver"]["type"].get<std::string>();
   }

   if (solver_name == "magnetostatic")
   {
      return std::make_unique<MagnetostaticSolver>(comm, json_options, nullptr);
   }
   else
   {
      throw std::runtime_error(
          "Unknown solver type!\n"
          "\tKnown types are:\n"
          "\t\tmagnetostatic\n");
   }
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
                [](const std::string &opt_file_name, mpi_comm comm)
                {
                   nlohmann::json json_options;
                   std::ifstream options_file(opt_file_name);
                   options_file >> json_options;
                   return initSolver(json_options, comm);
                }),
            py::arg("opt_file_name"),
            py::arg("comm") = mpi_comm(MPI_COMM_WORLD))
       .def(py::init([](const nlohmann::json &json_options, mpi_comm comm)
                     { return initSolver(json_options, comm); }),
            py::arg("json_options"),
            py::arg("comm") = mpi_comm(MPI_COMM_WORLD))
       .def("getNumStates", &PDESolver::getNumStates);
}
