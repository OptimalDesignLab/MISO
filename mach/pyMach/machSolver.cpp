#include <iostream>
#include <memory>
#include <utility>

#include <mpi4py/mpi4py.h>

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "mfem.hpp"
#include "nlohmann/json.hpp"
#include "pybind11_json.hpp"

#include "abstract_solver.hpp"
#include "mach_input.hpp"
#include "magnetostatic.hpp"
#include "mpi_comm.hpp"

// #ifdef BUILD_TESTING
// #include "test_mach_inputs.hpp"
// #endif

namespace py = pybind11;
using namespace mach;

namespace
{
/// \brief convert templated C++ factory method to one usable from python
/// \param type - the solver type to create
/// \param json_options - options dictionary containing solver options
/// \param comm - MPI communicator for parallel operations
std::unique_ptr<AbstractSolver2> initSolver(const nlohmann::json &json_options,
                                            MPI_Comm comm)
{
   auto physics = json_options["physics"].get<std::string>();
   if (physics == "magnetostatic")
   {
      return std::make_unique<MagnetostaticSolver>(comm, json_options, nullptr);
   }
   //    if (type == "Thermal")
   //    {
   //       return createSolver<ThermalSolver>(json_options, nullptr, comm);
   //    }
   //    else if (type == "Euler")
   //    {
   //       if (entvar)
   //       {
   //          return createSolver<EulerSolver<2, true>>(json_options, nullptr,
   //          comm);
   //       }
   //       else
   //       {
   //          return createSolver<EulerSolver<2, false>>(
   //              json_options, nullptr, comm);
   //       }
   //    }
   //    else if (type == "MeshMovement")
   //    {
   //       return createSolver<LEAnalogySolver>(json_options, nullptr, comm);
   //    }
   // #ifdef BUILD_TESTING
   //    else if (type == "TestMachInput")
   //    {
   //       return createSolver<TestMachInputSolver>(json_options, nullptr,
   //       comm);
   //    }
   // #endif
   else
   {
      throw std::runtime_error(
          "Unknown solver type!\n"
          "\tKnown types are:\n"
          "\t\tmagnetostatic\n"
          //  "\t\tThermal\n"
          //  "\t\tEuler\n"
          //  "\t\tMeshMovement\n"
      );
   }
}

double *npBufferToDoubleArray(const py::array_t<double> &buffer,
                              std::vector<pybind11::ssize_t> &shape,
                              int expected_dim = 1)
{
   auto info = buffer.request();

   /* Some sanity checks ... */
   if (info.format != py::format_descriptor<double>::format())
   {
      throw std::runtime_error(
          "Incompatible format:\n"
          "\texpected a double array!");
   }
   if (info.ndim != expected_dim)
   {
      throw std::runtime_error(
          "Incompatible dimensions:\n"
          "\texpected a 1D array!");
   }
   shape = std::move(info.shape);
   return static_cast<double *>(info.ptr);
}

double *npBufferToDoubleArray(const py::array_t<double> &buffer,
                              int expected_dim = 1)
{
   std::vector<pybind11::ssize_t> shape;
   return npBufferToDoubleArray(buffer, shape, expected_dim);
}

mfem::Vector npBufferToMFEMVector(const py::array_t<double> &buffer)
{
   auto info = buffer.request();
   /* Some sanity checks ... */
   if (info.format != py::format_descriptor<double>::format())
   {
      throw std::runtime_error(
          "Incompatible format:\n"
          "\texpected a double array!");
   }
   if (info.ndim != 1)
   {
      throw std::runtime_error(
          "Incompatible dimensions:\n"
          "\texpected a 1D array!");
   }
   return {static_cast<double *>(info.ptr), static_cast<int>(info.shape[0])};
}

MachInputs pyDictToMachInputs(const py::dict &py_inputs)
{
   MachInputs inputs(py_inputs.size());

   for (const auto &input : py_inputs)
   {
      const auto &key = input.first.cast<std::string>();

      const char *val_name = input.second.ptr()->ob_type->tp_name;
      bool is_number = strncmp("float", val_name, 5) == 0 ||
                       strncmp("int", val_name, 3) == 0;
      if (is_number)
      {
         const auto &value = input.second.cast<double>();
         inputs.emplace(key, value);
      }
      else
      {
         const auto &value_buffer = input.second.cast<py::array_t<double>>();
         std::vector<pybind11::ssize_t> shape;
         auto *value = npBufferToDoubleArray(value_buffer, shape);

         if (shape[0] == 1)
         {
            inputs.emplace(key, *value);
         }
         else
         {
            inputs.emplace(key, InputVector(value, shape[0]));
         }
      }
   }
   return inputs;
}

}  // anonymous namespace

void initSolver(py::module &m)
{
   /// imports mpi4py's C interface
   if (import_mpi4py() < 0)
   {
      return;
   }

   py::class_<AbstractSolver2>(m, "MachSolver")
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
                {
                   return initSolver(json_options, comm);
                 }),
            py::arg("json_options"),
            py::arg("comm") = mpi_comm(MPI_COMM_WORLD))

       .def(
           "setState",
           [](AbstractSolver2 &self,
              std::function<void(mfem::Vector &)> fun,
              const py::array_t<double> &state,
              const std::string &name)
           {
              auto state_vec = npBufferToMFEMVector(state);
              self.setState(fun, state_vec, name);
           },
           py::arg("fun"),
           py::arg("state"),
           py::arg("name") = "state")
       .def(
           "setState",
           [](AbstractSolver2 &self,
              std::function<double(const mfem::Vector &)> fun,
              const py::array_t<double> &state,
              const std::string &name)
           {
              auto state_vec = npBufferToMFEMVector(state);
              self.setState(fun, state_vec, name);
           },
           py::arg("fun"),
           py::arg("state"),
           py::arg("name") = "state")
       .def(
           "setState",
           [](AbstractSolver2 &self,
              std::function<void(const mfem::Vector &, mfem::Vector *const)>
                  fun,
              const py::array_t<double> &state,
              const std::string &name)
           {
              auto state_vec = npBufferToMFEMVector(state);
              self.setState([&fun](const mfem::Vector &x, mfem::Vector &u)
                            { fun(x, &u); },
                            state_vec,
                            name);
           },
           py::arg("fun"),
           py::arg("state"),
           py::arg("name") = "state")

       .def(
           "calcStateError",
           [](AbstractSolver2 &self,
              std::function<void(mfem::Vector &)> ex_sol,
              const py::array_t<double> &state,
              const std::string &name) {
              return self.calcStateError(
                  ex_sol, npBufferToMFEMVector(state), name);
           },
           py::arg("ex_sol"),
           py::arg("state"),
           py::arg("name") = "state")
       .def(
           "calcStateError",
           [](AbstractSolver2 &self,
              std::function<double(const mfem::Vector &)> ex_sol,
              const py::array_t<double> &state,
              const std::string &name)
           { self.calcStateError(ex_sol, npBufferToMFEMVector(state), name); },
           py::arg("ex_sol"),
           py::arg("state"),
           py::arg("name") = "state")
       .def(
           "calcStateError",
           [](AbstractSolver2 &self,
              std::function<void(const mfem::Vector &, mfem::Vector *const)>
                  ex_sol,
              const py::array_t<double> &state,
              const std::string &name)
           {
              self.calcStateError([&ex_sol](const mfem::Vector &x,
                                            mfem::Vector &u) { ex_sol(x, &u); },
                                  npBufferToMFEMVector(state),
                                  name);
           },
           py::arg("ex_sol"),
           py::arg("state"),
           py::arg("name") = "state")

       .def(
           "solveForState",
           [](AbstractSolver2 &self, const py::array_t<double> &state)
           {
              auto state_vec = npBufferToMFEMVector(state);
              self.solveForState(state_vec);
           },
           py::arg("state"))
       .def(
           "solveForState",
           [](AbstractSolver2 &self,
              const py::dict &py_inputs,
              const py::array_t<double> &state)
           {
              auto state_vec = npBufferToMFEMVector(state);
              self.solveForState(pyDictToMachInputs(py_inputs), state_vec);
           },
           py::arg("inputs"),
           py::arg("state"))

       .def(
           "calcResidual",
           [](AbstractSolver2 &self,
              const py::array_t<double> &state,
              const py::array_t<double> &residual)
           {
              auto state_vec = npBufferToMFEMVector(state);
              auto res_vec = npBufferToMFEMVector(residual);
              self.calcResidual(state_vec, res_vec);
           },
           py::arg("state"),
           py::arg("residual"))
       .def(
           "calcResidual",
           [](AbstractSolver2 &self,
              const py::dict &py_inputs,
              const py::array_t<double> &residual)
           {
              auto res_vec = npBufferToMFEMVector(residual);
              self.calcResidual(pyDictToMachInputs(py_inputs), res_vec);
           },
           py::arg("inputs"),
           py::arg("residual"))

       .def(
           "calcResidualNorm",
           [](AbstractSolver2 &self, const py::array_t<double> &state)
           {
              auto state_vec = npBufferToMFEMVector(state);
              return self.calcResidualNorm(state_vec);
           },
           py::arg("state"))
       .def(
           "calcResidualNorm",
           [](AbstractSolver2 &self, const py::dict &py_inputs)
           { return self.calcResidualNorm(pyDictToMachInputs(py_inputs)); },
           py::arg("inputs"))

       .def("getStateSize", &AbstractSolver2::getStateSize)
       .def("getFieldSize", &AbstractSolver2::getFieldSize, py::arg("name"))

       .def("createOutput",
            static_cast<void (AbstractSolver2::*)(const std::string &)>(
                &AbstractSolver2::createOutput),
            "Initialize the given output",
            py::arg("output"))
       .def("createOutput",
            static_cast<void (AbstractSolver2::*)(const std::string &,
                                                  const nlohmann::json &)>(
                &AbstractSolver2::createOutput),
            "Initialize the given output with options",
            py::arg("output"),
            py::arg("options"))

       .def("setOutputOptions",
            &AbstractSolver2::setOutputOptions,
            "Set options for the output specified by \"output\"",
            py::arg("output"),
            py::arg("options"))

       .def(
           "calcOutput",
           [](AbstractSolver2 &self,
              const std::string &output,
              const py::dict &py_inputs)
           { return self.calcOutput(output, pyDictToMachInputs(py_inputs)); },
           "Calculate the output specified by \"output\" using \"inputs\"",
           py::arg("output"),
           py::arg("inputs"))
       .def(
           "calcOutput",
           [](AbstractSolver2 &self,
              const std::string &output,
              const py::dict &py_inputs,
              const py::array_t<double> &out)
           {
              auto out_vec = npBufferToMFEMVector(out);
              return self.calcOutput(
                  output, pyDictToMachInputs(py_inputs), out_vec);
           },
           "Calculate the vector-valued output specified by \"output\" using "
           "\"inputs\"",
           py::arg("output"),
           py::arg("inputs"),
           py::arg("out_vec"))

       .def(
           "calcOutputPartial",
           [](AbstractSolver2 &self,
              const std::string &of,
              const std::string &wrt,
              const py::dict &py_inputs,
              const py::array_t<double> &partial_buffer)
           {
              std::vector<pybind11::ssize_t> shape;
              auto *partial = npBufferToDoubleArray(partial_buffer, shape);

              auto inputs = pyDictToMachInputs(py_inputs);
              if (shape[0] == 1)
              {
                 self.calcOutputPartial(of, wrt, inputs, *partial);
              }
              else
              {
                 auto partial_vec = mfem::Vector(partial, shape[0]);
                 self.calcOutputPartial(of, wrt, inputs, partial_vec);
              }
           },
           "Evaluates and returns the partial derivative of output functional "
           "specifed by \"of\" with respect to the input specified by \"wrt\"",
           py::arg("of"),
           py::arg("wrt"),
           py::arg("inputs"),
           py::arg("partial"))

       //  .def(
       //      "linearize",
       //      [](AbstractSolver &self, const py::dict &py_inputs)
       //      {
       // self.linearize(pyDictToMachInputs(py_inputs)); },
       //      py::arg("inputs"))

       //  .def(
       //      "vectorJacobianProduct",
       //      [](AbstractSolver &self,
       //         const py::array_t<double> &res_bar_buffer,
       //         const std::string &wrt,
       //         const py::array_t<double> &wrt_bar_buffer)
       //      {
       // auto *res_bar = npBufferToDoubleArray(res_bar_buffer);

       // std::vector<pybind11::ssize_t> shape;
       // auto *wrt_bar = npBufferToDoubleArray(wrt_bar_buffer, shape);
       // if (shape[0] == 1)
       // {
       //    *wrt_bar += self.vectorJacobianProduct(res_bar, wrt);
       // }
       // else
       // {
       //    self.vectorJacobianProduct(res_bar, wrt, wrt_bar);
       // }
       //      },
       //      py::arg("res_bar"),
       //      py::arg("wrt"),
       //      py::arg("wrt_bar"))

       //  .def("calcL2Error",
       //       [](AbstractSolver &self,
       //          mfem::ParGridFunction &state,
       //          const std::function<void(const mfem::Vector &,
       //                                   mfem::Vector *const)> &u_exact,
       //          int entry)
       //       {
       // return self.calcL2Error(
       //     &state,
       //     [u_exact](const mfem::Vector &x, mfem::Vector &u) { u_exact(x,
       //     &u); }, entry);
       //       })

       //  .def("printMesh", &AbstractSolver::printMesh)

       //  .def("printField",
       //       static_cast<void (AbstractSolver::*)(
       //           const std::string &, const std::string &, int, int)>(
       //           &AbstractSolver::printField),
       //       py::arg("filename"),
       //       py::arg("fieldname"),
       //       py::arg("refine") = -1,
       //       py::arg("cycle") = 0)

       //  .def("printField",
       //       static_cast<void (AbstractSolver::*)(const std::string &,
       //                                            mfem::ParGridFunction &,
       //                                            const std::string &,
       //                                            int,
       //                                            int)>(
       //           &AbstractSolver::printField),
       //       py::arg("filename"),
       //       py::arg("field"),
       //       py::arg("name"),
       //       py::arg("refine") = -1,
       //       py::arg("cycle") = 0)

       //  .def("printFields",
       //       &AbstractSolver::printFields,
       //       py::arg("filename"),
       //       py::arg("fields"),
       //       py::arg("names"),
       //       py::arg("refine") = -1,
       //       py::arg("cycle") = 0)

       //  .def(
       //      "getField",
       //      [](AbstractSolver &self,
       //         const std::string &name,
       //         const py::array_t<double> &field_buffer)
       //      {
       // self.getField(name, npBufferToDoubleArray(field_buffer)); },
       //      py::arg("name"),
       //      py::arg("field"))

       //  .def(
       //      "calcResidual",
       //      [](AbstractSolver &self,
       //         const py::dict &py_inputs,
       //         const py::array_t<double> &residual)
       //      {
       // self.calcResidual(pyDictToMachInputs(py_inputs),
       //                   npBufferToDoubleArray(residual));
       //      },
       //      py::arg("inputs"),
       //      py::arg("residual"))

       //  .def("createOutput",
       //       static_cast<void (AbstractSolver::*)(const std::string &fun)>(
       //           &AbstractSolver::createOutput),
       //       "Initialize the nonlinear form for the functional",
       //       py::arg("fun"))

       //  .def("createOutput",
       //       static_cast<void (AbstractSolver::*)(
       //           const std::string &fun, const nlohmann::json &options)>(
       //           &AbstractSolver::createOutput),
       //       "Initialize the nonlinear form for the functional with options",
       //       py::arg("fun"),
       //       py::arg("options"))

       //  .def(
       //      "calcOutput",
       //      [](AbstractSolver &self,
       //         const std::string &fun,
       //         const py::dict &py_inputs)
       //      {
       // return self.calcOutput(fun, pyDictToMachInputs(py_inputs)); },
       //      "Evaluates and returns the output functional specifed by `fun`",
       //      py::arg("fun"),
       //      py::arg("inputs"))

       //  .def(
       //      "calcOutputPartial",
       //      [](AbstractSolver &self,
       //         const std::string &of,
       //         const std::string &wrt,
       //         const py::dict &py_inputs,
       //         const py::array_t<double> &partial_buffer)
       //      {
       // std::vector<pybind11::ssize_t> shape;
       // auto *partial = npBufferToDoubleArray(partial_buffer, shape);

       // auto inputs = pyDictToMachInputs(py_inputs);
       // if (shape[0] == 1)
       // {
       //    // *partial = self.calcOutputPartial(of, wrt, inputs);
       //    throw std::runtime_error(
       //        "calcOutputPartial not supported for "
       //        "scalar derivative!\n");
       // }
       // else
       // {
       //    self.calcOutputPartial(of, wrt, inputs, partial);
       // }
       //      },
       //      "Evaluates and returns the partial derivative of output
       //      functional " "specifed by `of` with respect to the input
       //      specified by `wrt`", py::arg("of"), py::arg("wrt"),
       //      py::arg("inputs"),
       //      py::arg("partial"))

       //  .def("setOutputOptions",
       //       static_cast<void (AbstractSolver::*)(
       //           const std::string &fun, const nlohmann::json &options)>(
       //           &AbstractSolver::setOutputOptions),
       //       "Set options for the output functional specified by \"fun\"",
       //       py::arg("fun"),
       //       py::arg("options"))

       //  .def("getStateSize", &AbstractSolver::getStateSize)
       //  .def("getFieldSize", &AbstractSolver::getFieldSize)

       /// TODO:
       // .def("linearize", &AbstractSolver::linearize)
       // .def("multStateJacTranspose", &AbstractSolver::multStateJacTranspose)
       // .def("multMeshJacTranspose", &AbstractSolver::multMeshJacTranspose)
       // .def("invertStateJacTranspose",
       // &AbstractSolver::invertStateJacTranspose)
       ;
}
