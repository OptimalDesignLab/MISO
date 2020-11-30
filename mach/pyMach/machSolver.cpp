#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <mpi4py/mpi4py.h>
#include <iostream>

#include "mfem.hpp"
#include "json.hpp"
#include "pybind11_json.hpp"

#include "mpi4py_comm.hpp"

#include "solver.hpp"
#include "magnetostatic.hpp"
#include "thermal.hpp"
#include "euler.hpp"
#include "mesh_movement.hpp"

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
   // std::cout << "initSolver called with json options: " << json_options << "\n";
   if (type == "Magnetostatic")
   {
      return createSolver<MagnetostaticSolver>(json_options, nullptr, comm);
   }
   else if (type == "Thermal")
   {
      return createSolver<ThermalSolver>(json_options, nullptr, comm);
   }
   else if (type == "Euler")
   {
      if (entvar)
         return createSolver<EulerSolver<2, true>>(json_options, nullptr, comm);
      else
         return createSolver<EulerSolver<2, false>>(json_options,
                                                    nullptr,
                                                    comm);
   }
   else if (type == "MeshMovement")
   {
      return createSolver<LEAnalogySolver>(json_options, nullptr, comm);
   }
   else
   {
      throw std::runtime_error("Unknown solver type!\n"
      "\tKnown types are:\n"
      "\t\tMagnetostatic\n"
      "\t\tThermal\n"
      "\t\tEuler\n"
      "\t\tMeshMovement\n");
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

      .def("getOptions", [](AbstractSolver &self) -> nlohmann::json
      {
         return self.getOptions();
      })

      .def("getMeshSize", &AbstractSolver::getMeshSize)
      .def("getMeshCoordinates", &AbstractSolver::getMeshCoordinates,
            py::return_value_policy::reference)
      .def("setResidualInput", [](
         AbstractSolver &self,
         const std::string &field,
         py::array_t<double> data)
      {
         py::buffer_info info = data.request();

         /* Some sanity checks ... */
         if (info.format != py::format_descriptor<double>::format())
         {
            throw std::runtime_error("Incompatible format:\n"
                                       "\texpected a double array!");
         }
         if (info.ndim != 1)
         {
            throw std::runtime_error("Incompatible dimensions:\n"
                                       "\texpected a 1D array!");
         }
         
         if (info.shape[0] != self.getFieldSize(field))
         {
            std::string err("Incompatible size:\n"
            "\tattempting to set field \"");
            err += field;
            err += "\" (size: ";
            err += self.getFieldSize(field);
            err += ") with numpy vector of size: ";
            err += info.shape[0];
            throw std::runtime_error(err);
         }
         return self.setResidualInput(field, (double*)info.ptr);
      })

      .def("setScalarInitialCondition", (void (AbstractSolver::*)
            (mfem::ParGridFunction &state, 
            const std::function<double(const mfem::Vector &)>&))
            &AbstractSolver::setInitialCondition,
            "Initializes the state vector to a given scalar function.")

      .def("setInitialFieldValue", [](
         AbstractSolver &self,
         mfem::ParGridFunction &state,
         double u_init)
      {
         self.setInitialCondition(state, u_init);
      },
      "Initializes the state vector to a given value.")
      .def("setInitialFieldVectorValue", [](
         AbstractSolver &self,
         mfem::ParGridFunction &state,
         const mfem::Vector &u_init)
      {
         self.setInitialCondition(state, u_init);
      },
      "Initializes the state vector to a given vector value.")
      .def("setInitialFieldFunction", [](
         AbstractSolver &self,
         mfem::ParGridFunction &state,
         const std::function<double(const mfem::Vector &)> &u_init)
      {
         self.setInitialCondition(state, u_init);
      },
      "Initializes the state vector to a given scalar function.")
      .def("setInitialCondition", [](
         AbstractSolver& self,
         mfem::ParGridFunction &state,
         std::function<void(const mfem::Vector &, mfem::Vector *const)> u_init)
      {
         self.setInitialCondition(state, [u_init](const mfem::Vector &x, mfem::Vector &u)
         {
            u_init(x, &u);
         });
      },
      "Initializes the state vector to a given function.")

      .def("setInitialField", [](
         AbstractSolver& self,
         mfem::ParGridFunction &state,
         const mfem::ParGridFunction &u_init)
      {
         self.setInitialCondition(state, u_init);
      },
      "Initializes the state vector to a given field.")


      .def("getNewField", [] (
         AbstractSolver &self,
         py::array_t<double> data)
      {
         py::buffer_info info = data.request();

         // std::cout << "ptr: " << info.ptr << "\n";
         // std::cout << "format: " << info.format << "\n";
         // std::cout << "ndim: " << info.ndim << "\n";
         

         /* Some sanity checks ... */
         if (info.format != py::format_descriptor<double>::format())
         {
            throw std::runtime_error("Incompatible format:\n"
                                       "\texpected a double array!");
         }

         if (info.ndim > 1)
         {
            throw std::runtime_error("Incompatible dimensions:\n"
                                       "\texpected a 1D array!");
         }

         if (info.ndim == 0)
         {
            return self.getNewField(nullptr);
         }
         else
         {
            if (info.shape[0] != self.getStateSize())
            {
               std::string err("Incompatible size:\n"
               "\tattempting to construct state vector (size ");
               err += self.getStateSize();
               err += ") with numpy vector of size ";
               err += info.shape[0];
               throw std::runtime_error(err);
            }
            return self.getNewField((double*)info.ptr);
         }
         
         
      }, py::arg("data") = py::none())

      .def("solveForState", (void (AbstractSolver::*)(mfem::ParGridFunction&))
         &AbstractSolver::solveForState,
         py::arg("state"))

      .def("calcL2Error", [](
         AbstractSolver &self,
         mfem::ParGridFunction &state,
         std::function<void(const mfem::Vector &, mfem::Vector *const)> u_exact,
         int entry)
      {
         return self.calcL2Error(&state, [u_exact](const mfem::Vector &x, mfem::Vector &u)
         {
            u_exact(x, &u);
         }, entry);
      })

      .def("printMesh", &AbstractSolver::printMesh)

      .def("printField", &AbstractSolver::printField,
         py::arg("filename"),
         py::arg("field"),
         py::arg("name"),
         py::arg("refine") = -1)

      .def("printFields", &AbstractSolver::printFields,
         py::arg("filename"),
         py::arg("fields"),
         py::arg("names"),
         py::arg("refine") = -1)

      .def("calcResidual",
         (void (AbstractSolver::*)(const mfem::ParGridFunction &,
                                   mfem::ParGridFunction&) const)
         &AbstractSolver::calcResidual,
         py::arg("state"),
         py::arg("residual"))

      .def("calcFunctional",
         (double (AbstractSolver::*)(const mfem::ParGridFunction &,
                                     const std::string &))
         &AbstractSolver::calcOutput,
         py::arg("state"),
         py::arg("func"))

      .def("getStateSize", &AbstractSolver::getStateSize)
      .def("getFieldSize", &AbstractSolver::getFieldSize)

      /// TODO:
      // .def("linearize", &AbstractSolver::linearize)
      // .def("multStateJacTranspose", &AbstractSolver::multStateJacTranspose)
      // .def("multMeshJacTranspose", &AbstractSolver::multMeshJacTranspose)
      // .def("invertStateJacTranspose", &AbstractSolver::invertStateJacTranspose)
   ;
}