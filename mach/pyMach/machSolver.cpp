#include <iostream>
#include <memory>
#include <utility>

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "mfem.hpp"
#include "nlohmann/json.hpp"
#include "pybind11_json.hpp"

#include "abstract_solver.hpp"
#include "mach_input.hpp"
#include "mpi_comm.hpp"
#include "py_mach_utils.hpp"

namespace py = pybind11;
using namespace mach;

void initSolver(py::module &m)
{
   /// imports mpi4py's C interface
   if (import_mpi4py() < 0)
   {
      return;
   }

   py::class_<AbstractSolver2>(m, "MachSolver")
       //  .def(py::init(
       //           [](const std::string &opt_file_name, mpi_comm comm)
       //           {
       //              nlohmann::json json_options;
       //              std::ifstream options_file(opt_file_name);
       //              options_file >> json_options;
       //              return initSolver(json_options, comm);
       //           }),
       //       py::arg("opt_file_name"),
       //       py::arg("comm") = mpi_comm(MPI_COMM_WORLD))
       //  .def(py::init([](const nlohmann::json &json_options, mpi_comm comm)
       //                { return initSolver(json_options, comm); }),
       //       py::arg("json_options"),
       //       py::arg("comm") = mpi_comm(MPI_COMM_WORLD))
       .def("getOptions", &AbstractSolver2::getOptions)
       .def(
           "setState",
           [](AbstractSolver2 &self,
              const std::function<void(mfem::Vector &)> &fun,
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
              const std::function<double(const mfem::Vector &)> &fun,
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
              const std::function<void(mfem::Vector &)> &ex_sol,
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
              const std::function<double(const mfem::Vector &)> &ex_sol,
              const py::array_t<double> &state,
              const std::string &name)
           { self.calcStateError(ex_sol, npBufferToMFEMVector(state), name); },
           py::arg("ex_sol"),
           py::arg("state"),
           py::arg("name") = "state")
       .def(
           "calcStateError",
           [](AbstractSolver2 &self,
              const std::function<void(const mfem::Vector &,
                                       mfem::Vector *const)> &ex_sol,
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
           "solveForAdjoint",
           [](AbstractSolver2 &self,
              const py::array_t<double> &state,
              const py::array_t<double> &state_bar,
              const py::array_t<double> &adjoint)
           {
              auto state_vec = npBufferToMFEMVector(state);
              auto state_bar_vec = npBufferToMFEMVector(state_bar);
              auto adjoint_vec = npBufferToMFEMVector(adjoint);
              self.solveForAdjoint(state_vec, state_bar_vec, adjoint_vec);
           },
           py::arg("state"),
           py::arg("state_bar"),
           py::arg("adjoint"))
       .def(
           "solveForAdjoint",
           [](AbstractSolver2 &self,
              const py::dict &py_inputs,
              const py::array_t<double> &state_bar,
              const py::array_t<double> &adjoint)
           {
              auto inputs = pyDictToMachInputs(py_inputs);
              auto state_bar_vec = npBufferToMFEMVector(state_bar);
              auto adjoint_vec = npBufferToMFEMVector(adjoint);
              self.solveForAdjoint(inputs, state_bar_vec, adjoint_vec);
           },
           py::arg("inputs"),
           py::arg("state_bar"),
           py::arg("adjoint"))
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

       .def("getOutputSize", &AbstractSolver2::getOutputSize, py::arg("output"))

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
           R"(Calculate the output specified by "output" using "inputs")",
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
              self.calcOutput(output, pyDictToMachInputs(py_inputs), out_vec);
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

       .def(
           "outputJacobianVectorProduct",
           [](AbstractSolver2 &self,
              const std::string &of,
              const py::dict &py_inputs,
              const py::array_t<double> &wrt_dot_buffer,
              const std::string &wrt,
              const py::array_t<double> &out_dot_buffer)
           {
              auto inputs = pyDictToMachInputs(py_inputs);
              auto wrt_dot = npBufferToMFEMVector(wrt_dot_buffer);
              auto out_dot = npBufferToMFEMVector(out_dot_buffer);
              self.outputJacobianVectorProduct(
                  of, inputs, wrt_dot, wrt, out_dot);
           },
           py::arg("of"),
           py::arg("inputs"),
           py::arg("wrt_dot"),
           py::arg("wrt"),
           py::arg("out_dot"))

       .def(
           "outputVectorJacobianProduct",
           [](AbstractSolver2 &self,
              const std::string &of,
              const py::dict &py_inputs,
              const py::array_t<double> &out_bar_buffer,
              const std::string &wrt,
              const py::array_t<double> &wrt_bar_buffer)
           {
              auto inputs = pyDictToMachInputs(py_inputs);
              auto out_bar = npBufferToMFEMVector(out_bar_buffer);
              auto wrt_bar = npBufferToMFEMVector(wrt_bar_buffer);
              self.outputVectorJacobianProduct(
                  of, inputs, out_bar, wrt, wrt_bar);
           },
           py::arg("of"),
           py::arg("inputs"),
           py::arg("out_bar"),
           py::arg("wrt"),
           py::arg("wrt_bar"))

       .def(
           "linearize",
           [](AbstractSolver2 &self, const py::dict &py_inputs)
           { self.linearize(pyDictToMachInputs(py_inputs)); },
           py::arg("inputs"))
       .def(
           "jacobianVectorProduct",
           [](AbstractSolver2 &self,
              const py::array_t<double> &wrt_dot_buffer,
              const std::string &wrt,
              const py::array_t<double> &res_dot_buffer)
           {
              auto wrt_dot = npBufferToMFEMVector(wrt_dot_buffer);
              auto res_dot = npBufferToMFEMVector(res_dot_buffer);

              if (res_dot.Size() == 1)
              {
                 res_dot(0) += self.jacobianVectorProduct(wrt_dot, wrt);
              }
              else
              {
                 self.jacobianVectorProduct(wrt_dot, wrt, res_dot);
              }
           },
           py::arg("wrt_dot"),
           py::arg("wrt"),
           py::arg("res_dot"))

       .def(
           "vectorJacobianProduct",
           [](AbstractSolver2 &self,
              const py::array_t<double> &res_bar_buffer,
              const std::string &wrt,
              const py::array_t<double> &wrt_bar_buffer)
           {
              auto res_bar = npBufferToMFEMVector(res_bar_buffer);
              auto wrt_bar = npBufferToMFEMVector(wrt_bar_buffer);

              if (wrt_bar.Size() == 1)
              {
                 wrt_bar(0) += self.vectorJacobianProduct(res_bar, wrt);
              }
              else
              {
                 self.vectorJacobianProduct(res_bar, wrt, wrt_bar);
              }
           },
           py::arg("res_bar"),
           py::arg("wrt"),
           py::arg("wrt_bar"))

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
