#include <fstream>

#include "adept.h"

#include "utils.hpp"
#include "flow_control.hpp"

using namespace std;
using namespace mfem;

namespace mach 
{

void setInputs(PassiveControlResidual &residual, const MachInputs &inputs)
{
   setValueFromInputs(inputs, "Kp", residual.Kp);
   setValueFromInputs(inputs, "Td", residual.Td);
   setValueFromInputs(inputs, "Ti", residual.Ti);
   setValueFromInputs(inputs, "alpha", residual.alpha);
   setValueFromInputs(inputs, "beta", residual.beta);
   setValueFromInputs(inputs, "entropy_targ", residual.entropy_targ);
   setVectorFromInputs(inputs, "state", residual.x);
}

void setOptions(PassiveControlResidual &residual, const nlohmann::json &options)
{
   // set options here 
}

void evaluate(PassiveControlResidual &residual, const MachInputs &inputs, 
              mfem::Vector &res_vec)
{
   // define some aliases 
   double &Kp = residual.Kp;
   double &Td = residual.Td;
   double &Ti = residual.Ti;
   double &alpha = residual.alpha;
   double &beta = residual.beta;
   double &entropy_targ = residual.entropy_targ;
   // extract the control variables ("state") and entropy from inputs
   const bool error_if_not_found = true;
   setVectorFromInputs(inputs, "state", residual.x, error_if_not_found);
   double entropy;
   setValueFromInputs(inputs, "entropy", entropy, error_if_not_found);
   // evaluate the residual
   double e = entropy_targ - entropy;
   res_vec(0) = -((1.0 - 1.0/(alpha*beta))*Kp*e - x(1)/beta)/(alpha*Ti*Td);
   res_vec(1) = -( ((Td + Ti) - (alpha*Td + beta*Ti)/(alpha*beta))*Kp*e / 
                   (alpha*Ti*Td) + x(0) - x(1)*(alpha*Td + beta*Ti) /
                   (alpha*beta*Ti*Td) );
}

FlowControlResidual::FlowControlResidual(
    mfem::ParFiniteElementSpace &pfes,
    std::unordered_map<std::string, mfem::ParGridFunction> &fields,
    const nlohmann::json &flow_options, 
    const nlohmann::json &space_dis_options,
    const nlohmann::json &control_options) 
    : flow_res(pfes, fields), control_res(control_options)
{
   addFlowResidualIntegrators(flow_res, flow_options, space_dis_options);
   // TODO: need to add integrators for entropy output
}

int getSize(const FlowControlResidual &residual) 
{
   return getSize(residual.flow_res) + getSize(residual.control_res);
}

void setInputs(FlowControlResidual &residual, const MachInputs &inputs)
{
   // set inputs here
}

void setOptions(FlowControlResidual &residual, const nlohmann::json &options)
{
   // set options here 
}

void evaluate(FlowControlResidual &residual, const MachInputs &inputs, 
              mfem::Vector &res_vec)
{
   Vector control_res_vec(res_vec.GetData() + 0, getSize(residual.control_res));
   Vector flow_res_vec(res_vec.GetData() + num_control,
                       getSize(residual.flow_res));

   // Compute the outputs of the flow and control problems needed by the other
   double time = std::get<double>(inputs.at("time"));

   Vector state;
   getVectorFromInput(inputs.at("state"), state);
   Vector control_state(state.begin(), num_control);
   Vector flow_state(state.begin() + num_control, num_flow_state);

   // double * control_state_ptr = inputs.at("state").getField();
   // double * flow_state_ptr = inputs.at("state").getField() + num_control;
   auto flow_inputs = MachInputs({
      // {"state", flow_state_ptr},
      {"state", flow_state},
      {"time", time}
   });
   double entropy = calcOutput(residual.boundary_entropy, flow_inputs);
   auto control_inputs = MachInputs({
      // {"state", control_state_ptr}
      {"state", control_state}
   });
   double velocity = calcOutput(residual.velocity, control_inputs);

   // Evaluate the flow residual 
   flow_inputs.emplace("input-velocity", velocity);
   // WARNING: as of writing, evaluate does not call setInputs on itself
   evaluate(residual.flow_res, flow_inputs, flow_res_vec);

   // Evaluate the control residual 
   control_inputs.emplace("entropy", entropy);
   evaluate(residual.control_res, control_inputs, control_res_vec);
}

Operator &getJacobian(FlowControlResidual &residual, const MachInputs &inputs,
                      std::string wrt)
{
   if (wrt != "state")
   {
      throw MachException(
          "Unsupported value for wrt in getJacobian(FlowControlResidual)!\n"
          "\tvalue provided for wrt was " + wrt "\n");
   }
   double time = inputs.at("time");

   Vector state;
   getVectorFromInput(inputs.at("state"), state);
   Vector control_state(state.begin(), num_control);
   Vector flow_state(state.begin() + num_control, num_flow_state);

   // double * control_state_ptr = inputs.at("state").getField();
   // double * flow_state_ptr = inputs.at("state").getField() + num_control;
   // set inputs for flow Jacobian and get a reference to it
   auto flow_inputs = MachInputs({
      {"state", flow_state}, {"time", time}, {"control", control_state}
   });
   Operator &flow_jac = getJacobian(residual.flow_res, flow_inputs, "state");

   // Use a Block Operator?
   // Probably easier to use Matrix-Free approach, or a hybrid approach
   // For preconditioner, we can use block Jacobi and ignore the coupling terms
}


} // namespace mach