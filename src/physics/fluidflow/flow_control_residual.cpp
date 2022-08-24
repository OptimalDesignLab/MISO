#include <cmath>
#include <fstream>

#include "adept.h"

#include "flow_control_residual.hpp"
#include "mfem_extensions.hpp"
#include "utils.hpp"

using namespace std;
using namespace mfem;

namespace mach
{
ControlResidual::ControlResidual(MPI_Comm incomm,
                                 const nlohmann::json &control_options)
 : Kp(0.0),
   Td(0.0),
   Ti(0.0),
   beta(0.0),
   eta(0.0),
   target_entropy(0.0),
   closed_loop(true),
   time(0.0),
   boundary_entropy(0.0)
{
   MPI_Comm_dup(incomm, &comm);
   MPI_Comm_rank(comm, &rank);
   rank == 0 ? num_var = 2 : num_var = 0;
   x.SetSize(num_var);
   work.SetSize(num_var);
   P = std::make_unique<mfem::DenseMatrix>(num_var);
   mass_mat = std::make_unique<mfem::DenseMatrix>(num_var);
   Jac = std::make_unique<mfem::DenseMatrix>(num_var);
   prec = std::make_unique<mfem::DenseMatrixInverse>();
   if (rank == 0)
   {
      (*P) = 0.0;
      (*mass_mat) = 0.0;
      (*mass_mat)(0, 0) = 1.0;
      (*mass_mat)(1, 1) = 1.0;
   }
   test_ode = control_options["test-ode"];
}

void setInputs(ControlResidual &residual, const MachInputs &inputs)
{
   // Set state, time, and boundary entropy value
   setValueFromInputs(inputs, "time", residual.time);
   setValueFromInputs(inputs, "boundary-entropy", residual.boundary_entropy);
   setVectorFromInputs(inputs, "state", residual.x);
   // Set control parameters
   setValueFromInputs(inputs, "Kp", residual.Kp);
   setValueFromInputs(inputs, "Td", residual.Td);
   setValueFromInputs(inputs, "Ti", residual.Ti);
   setValueFromInputs(inputs, "eta", residual.eta);
   setValueFromInputs(inputs, "beta", residual.beta);
   setValueFromInputs(inputs, "target-entropy", residual.target_entropy);
   double closed_double = 1.0;
   setValueFromInputs(inputs, "closed-loop", closed_double);
   residual.closed_loop =
       fabs(closed_double) >= numeric_limits<double>::epsilon();

   // if (residual.rank == 0)
   // {
   //    // Update the P matrix
   //    double &Td = residual.Td;
   //    double &Ti = residual.Ti;
   //    double &eta = residual.eta;
   //    double &beta = residual.beta;

   //    double sigma = -0.5*(beta*Ti + eta*Td);
   //    double rho = beta*eta*Ti*Td - sigma*(beta*Ti + eta*Td);
   //    (*residual.P)(0,0) = rho;
   //    (*residual.P)(0,1) = sigma;
   //    (*residual.P)(1,0) = sigma;
   //    (*residual.P)(1,1) = 1.0;
   // }

   if ((residual.rank == 0) && (inputs.find("P-matrix") != inputs.end()))
   {
      Vector p_vector(4);
      setVectorFromInputs(inputs, "P-matrix", p_vector);
      (*residual.P)(0, 0) = p_vector(0);
      (*residual.P)(0, 1) = p_vector(1);
      (*residual.P)(1, 0) = p_vector(1);
      (*residual.P)(1, 1) = p_vector(3);
      // check for symmetry
      if (fabs(p_vector(1) - p_vector(2)) >
          10000.0 * numeric_limits<double>::epsilon())
      {
         throw MachException(
             "setInputs(ControlResidual, inputs): "
             "P-matrix is not symmetric!");
      }
   }
}

void setOptions(ControlResidual &residual, const nlohmann::json &options)
{
   // set options here
}

void evaluate(ControlResidual &residual,
              const MachInputs &inputs,
              mfem::Vector &res_vec)
{
   setInputs(residual, inputs);
   res_vec.SetSize(residual.num_var);
   if (residual.rank == 0)
   {
      if (residual.test_ode)
      {
         // This is a simple non-dissipative, decoupled system for testing
         // Residual is defined on left-hand side!
         res_vec(0) = -0.05 * residual.x(1);
         res_vec(1) = 0.05 * residual.x(0);
         double error = residual.target_entropy - residual.boundary_entropy;
         res_vec(1) -= error;
      }
      else
      {
         // define some aliases
         double &Kp = residual.Kp;
         double &Td = residual.Td;
         double &Ti = residual.Ti;
         double &eta = residual.eta;
         double &beta = residual.beta;

         // control-state coupling
         double fac = 1 / (beta * eta * Ti * Td);
         res_vec(0) = fac * residual.x(1);
         res_vec(1) =
             -residual.x(0) + fac * (beta * Ti + eta * Td) * residual.x(1);

         if (residual.closed_loop)
         {
            double error = residual.target_entropy - residual.boundary_entropy;
            double scaled_error = error * Kp / (eta * Ti * Td);
            res_vec(0) += (1.0 / (eta * beta) - 1.0) * scaled_error;
            res_vec(1) += ((beta * Ti + eta * Td) / (eta * beta) - (Ti + Td)) *
                          scaled_error;
         }
      }
   }
}

Operator &getJacobian(ControlResidual &residual,
                      const MachInputs &inputs,
                      const std::string &wrt)
{
   setInputs(residual, inputs);
   if (residual.rank == 0)
   {
      if (residual.test_ode)
      {
         // Jacobian for the simple problem
         (*residual.Jac) = 0.0;
         (*residual.Jac)(0, 1) = -0.05;
         (*residual.Jac)(1, 0) = 0.05;
      }
      else
      {
         // define some aliases
         double &Td = residual.Td;
         double &Ti = residual.Ti;
         double &eta = residual.eta;
         double &beta = residual.beta;

         DenseMatrix &Jac = *residual.Jac;
         Jac = 0.0;

         double fac = 1 / (beta * eta * Ti * Td);
         Jac(0, 1) = fac;
         Jac(1, 0) = -1.0;
         Jac(1, 1) = fac * (beta * Ti + eta * Td);
      }
   }
   return *residual.Jac;
}

double calcEntropy(ControlResidual &residual, const MachInputs &inputs)
{
   setInputs(residual, inputs);
   double ent = 0.0;
   if (residual.rank == 0)
   {
      if (residual.test_ode)
      {
         ent += residual.x(0) * residual.x(0) + residual.x(1) * residual.x(1);
      }
      else
      {
         Vector &x = residual.x;
         ent += 0.5 * residual.P->InnerProduct(x, x);
      }
   }
   MPI_Bcast(&ent, 1, MPI_DOUBLE, 0, residual.comm);
   return ent;
}

double calcEntropyChange(ControlResidual &residual, const MachInputs &inputs)
{
   // This only sets residual.x, time, and boundary_entropy
   setInputs(residual, inputs);
   Vector dxdt;
   setVectorFromInputs(inputs, "state_dot", dxdt, false, true);
   double ent_change = 0.0;
   if (residual.rank == 0)
   {
      // Note: since k = dxdt = -Res, we need to include a negative sign
      if (residual.test_ode)
      {
         ent_change -= residual.x * dxdt;
      }
      else
      {
         ent_change -= residual.P->InnerProduct(residual.x, dxdt);
      }
   }
   MPI_Bcast(&ent_change, 1, MPI_DOUBLE, 0, residual.comm);
   return ent_change;
}

double ControlResidual::getControlVelocity(const MachInputs &inputs)
{
   setInputs(*this, inputs);
   double vel = 0.0;
   if (rank == 0)
   {
      if (test_ode)
      {
         // For the simple controller
         vel = x(1);
      }
      else
      {
         vel += x(1);  // x should be set in setInputs above
         if (closed_loop)
         {
            double error = target_entropy - boundary_entropy;
            vel += Kp * error / eta;
         }
      }
   }
   MPI_Bcast(&vel, 1, MPI_DOUBLE, 0, comm);
   return vel;
}

template <int dim, bool entvar>
FlowControlResidual<dim, entvar>::FlowControlResidual(
    const nlohmann::json &options,
    ParFiniteElementSpace &pfes,
    std::map<std::string, FiniteElementState> &fields,
    adept::Stack &diff_stack,
    std::ostream &outstream)
 : out(outstream),
   flow_res(options, pfes, fields, diff_stack),
   control_res(pfes.GetComm(), options),
   boundary_entropy(
       flow_res.constructOutput("boundary-entropy",
                                options["outputs"]["boundary-entropy"]))
{
   setOptions(*this, options);
   // offsets mark the start of each row/column block; note that offsets must
   // be a unique pointer because move semantics are not set up for mfem::Array
   offsets = make_unique<Array<int>>(3);
   (*offsets)[0] = 0;
   (*offsets)[1] = num_control();
   (*offsets)[2] = (*offsets)[1] + num_flow();
   // create the mass operator
   mass_mat = make_unique<BlockOperator>(*offsets);
   auto *control_mass = getMassMatrix(control_res, options);
   auto flow_mass = getMassMatrix(flow_res, options);
   mass_mat->SetDiagonalBlock(0, control_mass);
   mass_mat->SetDiagonalBlock(1, flow_mass);
   // create the preconditioner
   prec = make_unique<BlockJacobiPreconditioner>(*offsets);
   auto *control_prec = getPreconditioner(control_res);
   auto *flow_prec = getPreconditioner(flow_res);
   prec->SetDiagonalBlock(0, control_prec);
   prec->SetDiagonalBlock(1, flow_prec);

   // check for consistency between the boundary-entropy and the control bc
   // NOTE: ideally, we would use the control BCs to define the functional.
   if (options["bcs"].contains("control"))
   {
      auto bc_marker = options["bcs"]["control"].get<vector<int>>();
      auto fun_marker = options["outputs"]["boundary-entropy"]["boundaries"]
                            .get<vector<int>>();
      if (bc_marker != fun_marker)
      {
         throw MachException(
             "FlowControlResidual:\n"
             "control bc and boundary entropy markers are inconsistent!\n");
      }
   }
   else
   {
      throw MachException("FlowControlResidual must have control BCs!\n");
   }
}

template <int dim, bool entvar>
void FlowControlResidual<dim, entvar>::extractStates(const Vector &state,
                                                     Vector &control_state,
                                                     Vector &flow_state) const
{
   control_state.NewDataAndSize(state.GetData(), num_control());
   flow_state.NewDataAndSize(state.GetData() + num_control(), num_flow());
}

template <int dim, bool entvar>
void FlowControlResidual<dim, entvar>::extractStates(const MachInputs &inputs,
                                                     Vector &control_state,
                                                     Vector &flow_state) const
{
   Vector state;
   setVectorFromInputs(inputs, "state", state, false, true);
   extractStates(state, control_state, flow_state);
}

template <int dim, bool entvar>
void FlowControlResidual<dim, entvar>::setInputs_(const MachInputs &inputs)
{
   auto control_inputs = MachInputs(inputs);
   auto flow_inputs = MachInputs(inputs);
   if (inputs.find("state") != inputs.end())
   {
      extractStates(inputs, control_ref, flow_ref);
      control_inputs.at("state") = control_ref;
      flow_inputs.at("state") = flow_ref;
   }
   setInputs(control_res, control_inputs);
   setInputs(flow_res, flow_inputs);
   setVectorFromInputs(inputs, "x-actuator", x_actuator);
}

template <int dim, bool entvar>
void FlowControlResidual<dim, entvar>::evaluate_(const MachInputs &inputs,
                                                 Vector &res_vec)
{
   setInputs_(inputs);
   Vector control_res_vec(res_vec.GetData() + 0, num_control());
   Vector flow_res_vec(res_vec.GetData() + num_control(), num_flow());
   // double time;
   // setValueFromInputs(inputs, "time", time, true);
   // double time = std::get<double>(inputs.at("time"));

   // get the coupling variables/outputs
   auto flow_inputs =
       MachInputs({{"state", flow_ref}, {"x-actuator", x_actuator}});
   double bndry_ent = calcOutput(boundary_entropy, flow_inputs);
   auto control_inputs =
       MachInputs({{"state", control_ref}, {"boundary-entropy", bndry_ent}});
   double control_vel = control_res.getControlVelocity(control_inputs);

   // evaluate the residuals
   flow_inputs.emplace("control", control_vel);
   evaluate(flow_res, flow_inputs, flow_res_vec);
   evaluate(control_res, control_inputs, control_res_vec);
}

template <int dim, bool entvar>
double FlowControlResidual<dim, entvar>::calcEntropy_(const MachInputs &inputs)
{
   // extract flow and control states to compute entropy
   // extractStates(inputs, control_ref, flow_ref);
   setInputs_(inputs);
   auto flow_inputs = MachInputs({{"state", flow_ref}});
   auto control_inputs = MachInputs({{"state", control_ref}});
   return calcEntropy(flow_res, flow_inputs) +
          calcEntropy(control_res, control_inputs);
}

template <int dim, bool entvar>
double FlowControlResidual<dim, entvar>::calcEntropyChange_(
    const MachInputs &inputs)
{
   // extract flow and control states to compute entropy
   extractStates(inputs, control_ref, flow_ref);
   Vector dxdt;
   Vector control_dxdt;
   Vector flow_dxdt;
   setVectorFromInputs(inputs, "state_dot", dxdt, false, true);
   extractStates(dxdt, control_dxdt, flow_dxdt);

   // extract time and time-step size
   double time = NAN;
   double dt = NAN;
   setValueFromInputs(inputs, "time", time, true);
   setValueFromInputs(inputs, "dt", dt, true);

   // get the control velocity; for this we need the boundary entropy at the
   // new state, so compute that first
   auto flow_inputs =
       MachInputs({{"state", flow_ref}, {"x-actuator", x_actuator}});
   double bndry_ent = calcOutput(boundary_entropy, flow_inputs);
   auto control_inputs =
       MachInputs({{"state", control_ref}, {"boundary-entropy", bndry_ent}});
   double control_vel = control_res.getControlVelocity(control_inputs);

   flow_inputs = MachInputs({{"state", flow_ref},
                             {"state_dot", flow_dxdt},
                             {"control", control_vel},
                             {"time", time},
                             {"dt", dt}});
   control_inputs = MachInputs({{"state", control_ref},
                                {"state_dot", control_dxdt},
                                {"time", time},
                                {"dt", dt}});
   return calcEntropyChange(flow_res, flow_inputs) +
          calcEntropyChange(control_res, control_inputs);
}

template <int dim, bool entvar>
mfem::Operator &FlowControlResidual<dim, entvar>::getJacobianBlock_(
    const MachInputs &inputs,
    int i)
{
   setInputs_(inputs);
   if (i == 0)
   {
      auto control_inputs = MachInputs({{"state", control_ref}});
      return getJacobian(control_res, control_inputs, "state");
   }
   else if (i == 1)
   {
      auto flow_inputs = MachInputs({{"state", flow_ref}});
      return getJacobian(flow_res, flow_inputs, "state");
   }
   else
   {
      throw MachException(
          "FlowControlResidual::GetJacobianBlock: \n"
          "invalid block index (must be 0 or 1)!\n");
   }
}

template <int dim, bool entvar>
double FlowControlResidual<dim, entvar>::minCFLTimeStep(
    double cfl,
    const ParGridFunction &state)
{
   return flow_res.minCFLTimeStep(cfl, state);
}

template <int dim, bool entvar>
MachOutput FlowControlResidual<dim, entvar>::constructOutput(
    const std::string &fun,
    const nlohmann::json &options)
{
   if (fun == "drag" || fun == "lift" || fun == "boundary-entropy")
   {
      return flow_res.constructOutput(fun, options);
   }
   else if (fun == "entropy")
   {
      // global entropy
      EntropyOutput<FlowControlResidual<dim, entvar>> fun_out(*this);
      return fun_out;
   }
   else
   {
      throw MachException("Output with name " + fun +
                          " not supported by "
                          "FlowControlResidual!\n");
   }
}

// explicit instantiation
template class FlowControlResidual<1, true>;
template class FlowControlResidual<1, false>;
template class FlowControlResidual<2, true>;
template class FlowControlResidual<2, false>;
template class FlowControlResidual<3, true>;
template class FlowControlResidual<3, false>;

}  // namespace mach