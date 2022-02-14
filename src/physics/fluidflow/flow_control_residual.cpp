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
void setInputs(ControlResidual &residual, const MachInputs &inputs)
{
   // setValueFromInputs(inputs, "Kp", residual.Kp);
   // setValueFromInputs(inputs, "Td", residual.Td);
   // setValueFromInputs(inputs, "Ti", residual.Ti);
   // setValueFromInputs(inputs, "alpha", residual.alpha);
   // setValueFromInputs(inputs, "beta", residual.beta);
   // setValueFromInputs(inputs, "entropy_targ", residual.entropy_targ);
   setValueFromInputs(inputs, "time", residual.time);
   setValueFromInputs(inputs, "boundary-entropy", residual.boundary_entropy);
   setVectorFromInputs(inputs, "state", residual.x);
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
      // This is a simple non-dissipative, decoupled system for testing
      // Residual is defined on left-hand side!
      res_vec(0) = -0.05 * residual.x(1);
      res_vec(1) = 0.05 * residual.x(0) + residual.boundary_entropy;
   }

   // // define some aliases
   // double &Kp = residual.Kp;
   // double &Td = residual.Td;
   // double &Ti = residual.Ti;
   // double &alpha = residual.alpha;
   // double &beta = residual.beta;
   // double &entropy_targ = residual.entropy_targ;
   // // extract the control variables ("state") and entropy from inputs
   // const bool error_if_not_found = true;
   // setVectorFromInputs(inputs, "state", residual.x, error_if_not_found);
   // double entropy = NAN;
   // setValueFromInputs(inputs, "entropy", entropy, error_if_not_found);
   // // evaluate the residual
   // double e = entropy_targ - entropy;
   // res_vec(0) = -((1.0 - 1.0/(alpha*beta))*Kp*e -
   // residual.x(1)/beta)/(alpha*Ti*Td); res_vec(1) = -( ((Td + Ti) - (alpha*Td
   // + beta*Ti)/(alpha*beta))*Kp*e /
   //                 (alpha*Ti*Td) + residual.x(0) - residual.x(1)*(alpha*Td +
   //                 beta*Ti) / (alpha*beta*Ti*Td) );
}

Operator &getJacobian(ControlResidual &residual,
                      const MachInputs &inputs,
                      std::string wrt)
{
   setInputs(residual, inputs);
   if (residual.rank == 0)
   {
      (*residual.Jac) = 0.0;
      (*residual.Jac)(0, 1) = -0.05;
      (*residual.Jac)(1, 0) = 0.05;
   }
   return *residual.Jac;
}

double calcEntropy(ControlResidual &residual, const MachInputs &inputs)
{
   setInputs(residual, inputs);
   double ent = 0.0;
   if (residual.rank == 0)
   {
      ent += residual.x(0) * residual.x(0) + residual.x(1) * residual.x(1);
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
      ent_change += residual.x * dxdt;
   }
   MPI_Bcast(&ent_change, 1, MPI_DOUBLE, 0, residual.comm);
   return ent_change;
}

template <int dim, bool entvar>
FlowControlResidual<dim, entvar>::FlowControlResidual(
    const nlohmann::json &options,
    ParFiniteElementSpace &pfes,
    adept::Stack &diff_stack,
    std::ostream &outstream)
 : out(outstream),
   flow_res(options, pfes, diff_stack),
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
   auto control_mass = getMassMatrix(control_res, options);
   auto flow_mass = getMassMatrix(flow_res, options);
   mass_mat->SetDiagonalBlock(0, control_mass);
   mass_mat->SetDiagonalBlock(1, flow_mass);
   // create the preconditioner
   auto prec_opts = options["lin-prec"];
   prec = make_unique<BlockJacobiPreconditioner>(*offsets);
   auto control_prec = getPreconditioner(control_res, prec_opts);
   auto flow_prec = getPreconditioner(flow_res, prec_opts);
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
   auto control_inputs = MachInputs({{"state", control_ref}});
   double control_vel = control_res.getControlVelocity(control_ref);

   // evaluate the residuals
   flow_inputs.emplace("control", control_vel);
   control_inputs.emplace("boundary-entropy", bndry_ent);
   evaluate(flow_res, flow_inputs, flow_res_vec);
   evaluate(control_res, control_inputs, control_res_vec);
}

template <int dim, bool entvar>
double FlowControlResidual<dim, entvar>::calcEntropy_(const MachInputs &inputs)
{
   // extract flow and control states to compute entropy
   extractStates(inputs, control_ref, flow_ref);
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
   Vector dxdt, control_dxdt, flow_dxdt;
   setVectorFromInputs(inputs, "state_dot", dxdt, false, true);
   extractStates(dxdt, control_dxdt, flow_dxdt);

   // extract time and time-step size
   double time = NAN;
   double dt = NAN;
   setValueFromInputs(inputs, "time", time, true);
   setValueFromInputs(inputs, "dt", dt, true);

   // get the control velocity for input to the calcEntropyChange for the flow; 
   // note that the boundary entropy is not needed for the control entropy 
   // change, because we can use `state_dot` directly.
   control_work.SetSize(control_ref.Size());
   add(control_ref, dt, control_dxdt, control_work);
   double control_vel = control_res.getControlVelocity(control_work);

   auto flow_inputs = MachInputs({{"state", flow_ref},
                                  {"state_dot", flow_dxdt},
                                  {"control", control_vel},
                                  {"time", time},
                                  {"dt", dt}});
   auto control_inputs = MachInputs({{"state", control_ref}, 
                                     {"state_dot", control_dxdt},
                                     {"time", time},
                                     {"dt", dt}});
   return calcEntropyChange(flow_res, flow_inputs) +
          calcEntropyChange(control_res, control_inputs);

//    // extract time and time-step size
//    double time = NAN;
//    double dt = NAN;
//    setValueFromInputs(inputs, "time", time, true);
//    setValueFromInputs(inputs, "dt", dt, true);

//    // get the coupling variables/outputs; these need to be computed at the
//    // updated state!!!
//    control_work.SetSize(control_ref.Size());
//    flow_work.SetSize(flow_ref.Size());
//    add(control_ref, dt, control_dxdt, control_work);
//    add(flow_ref, dt, flow_dxdt, flow_work);
//    auto flow_inputs =
//        MachInputs({{"state", flow_work}, {"x-actuator", x_actuator}});
//    double bndry_ent = calcOutput(boundary_entropy, flow_inputs);
//    double control_vel = control_res.getControlVelocity(control_work);

//    // set inputs for flow and control residuals and evaluate change
//    flow_inputs = MachInputs({{"state", flow_ref},
//                              {"state_dot", flow_dxdt},
//                              {"x-actuator", x_actuator},
//                              {"control", control_vel},
//                              {"time", time},
//                              {"dt", dt}});
//    auto control_inputs = MachInputs({{"state", control_ref},
//                                      {"state_dot", control_dxdt},
//                                      {"boundary-entropy", bndry_ent},
//                                      {"time", time},
//                                      {"dt", dt}});
//    return calcEntropyChange(flow_res, flow_inputs) +
//           calcEntropyChange(control_res, control_inputs);
// 

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
   if (fun == "drag" || fun == "lift")
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