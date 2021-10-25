#include <fstream>

#include "adept.h"

#include "utils.hpp"
#include "flow_control.hpp"

using namespace std;
using namespace mfem;

namespace mach 
{

FlowControlResidual::FlowControlResidual(
    mfem::ParFiniteElementSpace &pfes,
    std::unordered_map<std::string, mfem::ParGridFunction> &fields,
    const nlohmann::json &flow_options, 
    const nlohmann::json &space_dis_options,
    const nlohmann::json &control_options) : flow_res(pfes, fields)
{
   addFlowResidualIntegrators(flow_res, flow_options, space_dis_options);
   num_control = control_options["num-control"];
}

int getSize(const FlowControlResidual &residual) 
{
   return getSize(residual.flow_res) + num_control;
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
   Vector control_res_vec(res_vec.GetData() + 0, num_control);
   Vector flow_res_vec(res_vec.GetData() + num_control,
                       getSize(residual.flow_res));
   double time = inputs.at("time");
   double * control_state_ptr = inputs.at("state").getField();
   double * flow_state_ptr = inputs.at("state").getField() + num_control;
   // set inputs for flow residual and evaluate it 
   auto flow_inputs = MachInputs({
      {"state", flow_state_ptr}, {"time", time}, {"control", control_state_ptr}
   });
   // WARNING: as of writing, evaluate does not call setInputs on itself
   evaluate(residual.flow_res, flow_inputs, flow_res_vec);

   // set inputs for ODE residual and evaluate it 
   // probably need a MachOutput to compute flow value need by ODE


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
   double * control_state_ptr = inputs.at("state").getField();
   double * flow_state_ptr = inputs.at("state").getField() + num_control;
   // set inputs for flow Jacobian and get a reference to it
   auto flow_inputs = MachInputs({
      {"state", flow_state_ptr}, {"time", time}, {"control", control_state_ptr}
   });
   Operator &flow_jac = getJacobian(residual.flow_res, flow_inputs, "state");

   // Use a Block Operator?
   // Probably easier to use Matrix-Free approach, or a hybrid approach
   // For preconditioner, we can use block Jacobi and ignore the coupling terms
}


} // namespace mach