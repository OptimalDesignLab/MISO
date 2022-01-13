#include <cmath>
#include <fstream>

#include "adept.h"

#include "utils.hpp"
#include "flow_control_residual.hpp"

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
   setVectorFromInputs(inputs, "state", residual.x);
}

void setOptions(ControlResidual &residual, const nlohmann::json &options)
{
   // set options here 
}

void evaluate(ControlResidual &residual, const MachInputs &inputs, 
              mfem::Vector &res_vec)
{
   setInputs(residual, inputs);
   // This is a simple non-dissipative, decoupled system for testing
   // Residual is defined on left-hand side!
   res_vec.SetSize(2);
   res_vec(0) = -residual.x(1);
   res_vec(1) = residual.x(0);

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
   // res_vec(0) = -((1.0 - 1.0/(alpha*beta))*Kp*e - residual.x(1)/beta)/(alpha*Ti*Td);
   // res_vec(1) = -( ((Td + Ti) - (alpha*Td + beta*Ti)/(alpha*beta))*Kp*e / 
   //                 (alpha*Ti*Td) + residual.x(0) - residual.x(1)*(alpha*Td + beta*Ti) /
   //                 (alpha*beta*Ti*Td) );
}

Operator &getJacobian(ControlResidual &residual,
                      const MachInputs &inputs,
                      std::string wrt)
{
   setInputs(residual, inputs);
   residual.Jac = 0.0;
   residual.Jac(0,1) = -1.0;
   residual.Jac(1,0) = 1.0;
   return residual.Jac;
}

double calcEntropy(ControlResidual &residual, const MachInputs &inputs)
{
   setInputs(residual, inputs);
   return residual.x(0)*residual.x(0) + residual.x(1)*residual.x(1);
}

double calcEntropyChange(ControlResidual &residual, const MachInputs &inputs)
{
   // This only sets residual.x and time for now
   setInputs(residual, inputs);
   Vector dxdt;
   setVectorFromInputs(inputs, "state_dot", dxdt, false, true);
   double dt = NAN;
   setValueFromInputs(inputs, "dt", dt, true);
   auto &y = residual.work;
   add(residual.x, dt, dxdt, y);
   return -y(0)*y(1) + y(1)*y(0);
}

template <int dim, bool entvar>
FlowControlResidual<dim, entvar>::FlowControlResidual(
    const nlohmann::json &options,
    mfem::ParFiniteElementSpace &pfes,
    adept::Stack &diff_stack)
 : flow_res(options, pfes, diff_stack), control_res(options)
{
   // flow-control specific set-up?
}

template <int dim, bool entvar>
void FlowControlResidual<dim, entvar>::extractStatesFromInputs(
    const MachInputs &inputs,
    mfem::Vector &control_state,
    mfem::Vector &flow_state)
{
   Vector state;
   setVectorFromInputs(inputs, "state", state, false, true);
   control_state.NewDataAndSize(state.begin(), num_control());
   flow_state.NewDataAndSize(state.begin() + num_control(), num_flow());
}

template <int dim, bool entvar>
void FlowControlResidual<dim, entvar>::evaluate_(const MachInputs &inputs,
                                                 mfem::Vector &res_vec)
{
   Vector control_res_vec(res_vec.GetData() + 0, getSize(control_res));
   Vector flow_res_vec(res_vec.GetData() + num_control,
                       getSize(flow_res));

   // Compute the outputs of the flow and control problems needed by the other
   double time = std::get<double>(inputs.at("time"));

   Vector control_state, flow_state;
   extractStatesFromInputs(inputs, control_state, flow_state);

   // double * control_state_ptr = inputs.at("state").getField();
   // double * flow_state_ptr = inputs.at("state").getField() + num_control;
   // auto flow_inputs = MachInputs({// {"state", flow_state_ptr},
   //                                {"state", flow_state},
   //                                {"time", time}});
   // double entropy = calcOutput(residual.boundary_entropy, flow_inputs);
   // auto control_inputs = MachInputs({// {"state", control_state_ptr}
   //                                   {"state", control_state}});
   // double velocity = calcOutput(residual.velocity, control_inputs);

   // // Evaluate the flow residual
   // flow_inputs.emplace("input-velocity", velocity);
   // // WARNING: as of writing, evaluate does not call setInputs on itself
   // evaluate(residual.flow_res, flow_inputs, flow_res_vec);

   // // Evaluate the control residual
   // control_inputs.emplace("entropy", entropy);
   // evaluate(residual.control_res, control_inputs, control_res_vec);
}

template <int dim, bool entvar>
Operator &FlowControlResidual<dim, entvar>::getJacobian_(
                      const MachInputs &inputs,
                      const std::string &wrt)
{
   // if (wrt != "state")
   // {
   //    throw MachException(
   //        "Unsupported value for wrt in getJacobian(FlowControlResidual)!\n"
   //        "\tvalue provided for wrt was " + wrt "\n");
   // }
   // double time = inputs.at("time");

   // Vector control_state, flow_state;
   // extractStatesFromInputs(inputs, control_state, flow_state);

   // // double * control_state_ptr = inputs.at("state").getField();
   // // double * flow_state_ptr = inputs.at("state").getField() + num_control;
   // // set inputs for flow Jacobian and get a reference to it
   // auto flow_inputs = MachInputs({
   //    {"state", flow_state}, {"time", time}, {"control", control_state}
   // });
   // Operator &flow_jac = getJacobian(residual.flow_res, flow_inputs, "state");

   // // Use a Block Operator?
   // // Probably easier to use Matrix-Free approach, or a hybrid approach
   // // For preconditioner, we can use block Jacobi and ignore the coupling terms
}

template <int dim, bool entvar>
double FlowControlResidual<dim, entvar>::calcEntropy_(const MachInputs &inputs)
{
   // extract flow and control states to compute entropy
   Vector control_state, flow_state;
   extractStatesFromInputs(inputs, control_state, flow_state);
   auto flow_inputs = MachInputs(
       {{"state", flow_state}, {"time", time}, {"control", control_state}});
   auto control_inputs = MachInputs({{"state", control_state}, {"time", time}});
   return calcEntropy(flow_res, flow_inputs) +
          calcEntropy(control_res, control_inputs);
}

template <int dim, bool entvar>
double FlowControlResidual<dim, entvar>::calcEntropyChange_(
   const MachInputs &inputs)
{
   // Vector x;
   // setVectorFromInputs(inputs, "state", x, false, true);
   // Vector dxdt;
   // setVectorFromInputs(inputs, "state_dot", dxdt, false, true);
   // double dt = NAN;
   // double time = NAN;
   // setValueFromInputs(inputs, "time", time, true);
   // setValueFromInputs(inputs, "dt", dt, true);
   // auto &y = work;
   // add(x, dt, dxdt, y);
   // auto form_inputs = MachInputs({{"state", y}, {"time", time + dt}});
   // return calcFormOutput(res, form_inputs);
   return 0.0;
}


} // namespace mach