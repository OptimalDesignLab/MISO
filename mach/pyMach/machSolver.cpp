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

#ifdef BUILD_TESTING
#include "test_mach_inputs.hpp"
#endif

#include "mach_input.hpp"

namespace py = pybind11;

using namespace mfem;
// using namespace nlohmann;
using namespace mach;


/// should getNewField create an entry in `res_fields`?
/// would need to specify name of field
/// could have optional options input to construct different fes/fec for the field
/// 
/// this does not address that integrators need to have a reference to the field
/// when they're constructed
/// what about getting reference to named field?
///
/// I think in general this is a bad idea
/// -- just make a field implicitly convertable to a buffer
/// or should getNewField just return the buffer with no gf?
///   -- anywhere that would use it would need to know its size a priori
///   ^ we could not use mfem functions without the gf object
/// ** just make implicitly castable to buffer (not from) **

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
#ifdef BUILD_TESTING
   else if (type == "TestMachInput")
   {
      return createSolver<TestMachInputSolver>(json_options, nullptr, comm);
   }
#endif
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

MachInputs pyDictToMachInputs(const py::dict &py_inputs)
{
   MachInputs inputs(py_inputs.size());

   for (auto &input : py_inputs)
   {
      const auto &key = input.first.cast<std::string>();

      const char *val_name = input.second.ptr()->ob_type->tp_name;
      bool is_float = strncmp("float", val_name, 5) == 0;
      if (is_float)
      {
         const auto &value = input.second.cast<double>();
         inputs.emplace(key, value);
      }
      else
      {
         const auto &value = input.second.cast<py::buffer>();
         /* Request a buffer descriptor from Python */
         py::buffer_info buffer = value.request();

         /* Some sanity checks ... */
         if (buffer.format != py::format_descriptor<double>::format())
            throw std::runtime_error("Incompatible format:\n"
                                    "\texpected a double array!");
         if (buffer.ndim != 1)
            throw std::runtime_error("Incompatible dimensions:\n"
                                    "\texpected a 1D array!");

         if (buffer.shape[0] == 1)
            inputs.emplace(key, *(double*)buffer.ptr);
         else
            inputs.emplace(key, (double*)buffer.ptr);
      }
   }
   return inputs;
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

      .def("setFieldValue", [](
         AbstractSolver &self,
         mfem::HypreParVector &state,
         const double u_init)
      {
         self.setFieldValue(state, u_init);

      },
      "Sets the field to a given value.")

      // .def("setFieldValue",
      // (void (AbstractSolver::*)
      //    (mfem::HypreParVector &field, 
      //    const std::function<double(const mfem::Vector &)>&))
      // &AbstractSolver::setInitialCondition,
      // "Sets the field to a given scalar function.")
      .def("setFieldValue", [](
         AbstractSolver &self,
         mfem::HypreParVector &state,
         const std::function<double(const mfem::Vector &)> &u_init)
      {
         self.setFieldValue(state, u_init);
      },
      "Sets the field to a given scalar function.")

      .def("setFieldValue", [](
         AbstractSolver &self,
         mfem::HypreParVector &field,
         const mfem::Vector &u_init)
      {
         self.setFieldValue(field, u_init);
      },
      "Sets the vector field to a given vector value.")
      .def("setFieldValue", [](
         AbstractSolver &self,
         mfem::HypreParVector &field,
         py::array_t<double> u_init_data)
      {
         py::buffer_info info = u_init_data.request();
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
         mfem::Vector u_init((double*)info.ptr, info.shape[0]);
         self.setFieldValue(field, u_init);
      },
      "Sets the vector field to a given vector value.")
   
      .def("setFieldValue", [](
         AbstractSolver& self,
         mfem::HypreParVector &field,
         std::function<void(const mfem::Vector &, mfem::Vector *const)> u_init)
      {
         self.setFieldValue(field, [u_init](const mfem::Vector &x, mfem::Vector &u)
         {
            u_init(x, &u);
         });
      },
      "Sets the vector field to a given vector-valued function.")

      .def("setField", [](
         AbstractSolver& self,
         mfem::HypreParVector &field,
         const mfem::HypreParVector &u_init)
      {
         // self.setInitialCondition(state, u_init);
         field = u_init;
      },
      "Sets the field to equal a given field.")


      .def("getNewField", [] (
         AbstractSolver &self,
         py::array_t<double> data)
      {
         py::buffer_info info = data.request();         

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

      .def("solveForState", [](AbstractSolver &self,
                            const py::dict &py_inputs,
                            mfem::HypreParVector &state)
         {
            
            return self.solveForState(pyDictToMachInputs(py_inputs), state);
         },
         py::arg("inputs"),
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
         py::arg("refine") = -1,
         py::arg("cycle") = 0)

      .def("printFields", &AbstractSolver::printFields,
         py::arg("filename"),
         py::arg("fields"),
         py::arg("names"),
         py::arg("refine") = -1,
         py::arg("cycle") = 0)

      .def("getField", &AbstractSolver::getField,
         py::arg("field_name"))

      .def("calcResidual", [](AbstractSolver &self,
                              const py::dict &py_inputs,
                              mfem::HypreParVector &residual)
         {
            self.calcResidual(pyDictToMachInputs(py_inputs), residual);
         },
         py::arg("inputs"),
         py::arg("residual"))

      .def("createOutput", 
         static_cast<void (AbstractSolver::*)(const std::string &fun)>
         (&AbstractSolver::createOutput),
         "Initialize the nonlinear form for the functional",
         py::arg("fun"))

      .def("createOutput", 
         static_cast<void (AbstractSolver::*)(const std::string &fun,
                                              const nlohmann::json &options)>
         (&AbstractSolver::createOutput),
         "Initialize the nonlinear form for the functional with options",
         py::arg("fun"),
         py::arg("options"))

      .def("calcOutput", [](AbstractSolver &self,
                            const std::string &fun,
                            const py::dict &py_inputs)
         {
            return self.calcOutput(fun, pyDictToMachInputs(py_inputs));
         },
         "Evaluates and returns the output functional specifed by `fun`",
         py::arg("fun"),
         py::arg("inputs"))

      .def("calcOutputPartial", [](AbstractSolver &self,
                            const std::string &of,
                            const std::string &wrt,
                            const py::dict &py_inputs,
                            py::array_t<double> partial)
         {
            /* Request a buffer descriptor from Python */
            py::buffer_info buffer = partial.request();

            /* Some sanity checks ... */
            if (buffer.format != py::format_descriptor<double>::format())
               throw std::runtime_error("Incompatible format:\n"
                                       "\texpected a double array!\n");
            if (buffer.ndim != 1)
               throw std::runtime_error("Incompatible dimensions:\n"
                                       "\texpected a 1D array!\n");

            if (buffer.shape[0] == 1)
               // self.calcOutputPartial(of, wrt,
               //                        pyDictToMachInputs(py_inputs),
               //                        *(double*)buffer.ptr);
               throw std::runtime_error("calcOutputPartial not supported for "
                                        "scalar derivative!\n")
            else
               self.calcOutputPartial(of, wrt,
                                      pyDictToMachInputs(py_inputs),
                                      (double*)buffer.ptr);
         },
         "Evaluates and returns the partial derivative of output functional "
         "specifed by `of` with respect to the input specified by `wrt`",
         py::arg("of"),
         py::arg("wrt"),
         py::arg("inputs"),
         py::arg("partial"))
      
      .def("setOutputOptions",
         static_cast<void (AbstractSolver::*)(const std::string &fun,
                                              const nlohmann::json &options)>
         (&AbstractSolver::setOutputOptions),
         "Set options for the output functional specified by \"fun\"",
         py::arg("fun"),
         py::arg("options"))

      .def("getStateSize", &AbstractSolver::getStateSize)
      .def("getFieldSize", &AbstractSolver::getFieldSize)

      /// TODO:
      // .def("linearize", &AbstractSolver::linearize)
      // .def("multStateJacTranspose", &AbstractSolver::multStateJacTranspose)
      // .def("multMeshJacTranspose", &AbstractSolver::multMeshJacTranspose)
      // .def("invertStateJacTranspose", &AbstractSolver::invertStateJacTranspose)
   ;
}