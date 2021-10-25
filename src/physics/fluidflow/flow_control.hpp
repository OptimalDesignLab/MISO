#ifndef MACH_FLOW_CONTROL
#define MACH_FLOW_CONTROL

#include "mfem.hpp"
#include "nlohmann/json.hpp"
#include "mach_nonlinearform.hpp"


class FlowControlResidual final
{
public:
   /// Gets the number of equations/unknowns; # of FEM unknowns + # control vars
   /// \param[inout] residual - the residual whose size is being queried
   /// \returns the number of equations/unknowns
   friend int getSize(const FlowControlResidual &residual);

   /// Set inputs in the underlying residual
   /// \param[inout] residual - the flow-control residual being assigned inputs
   /// \param[in] inputs - the inputs that are being assigned
   friend void setInputs(FlowControlResidual &residual,
                         const MachInputs &inputs);

   /// Set options in the underlying residual type
   /// \param[inout] residual - flow-control residual whose options are set
   /// \param[in] options - the options that are being assigned
   friend void setOptions(FlowControlResidual &residual,
                          const nlohmann::json &options);

   /// Evaluate the flow-control residual at the inputs and return as `res_vec`
   /// \param[inout] residual - the flow-control residual being evaluated
   /// \param[in] inputs - the independent variables at which to evaluate `res`
   /// \param[out] res_vec - the dependent variable, the output from `residual`
   friend void evaluate(FlowControlResidual &residual,
                        const MachInputs &inputs,
                        mfem::Vector &res_vec);

   /// Compute the Jacobian of the flow-control residual and return a reference
   /// \param[inout] residual - flow-control residual whose Jacobian we want
   /// \param[in] inputs - the variables needed to evaluate the Jacobian
   /// \param[in] wrt - the input we are differentiating with respect to
   /// \returns a reference to the residual's Jacobian with respect to `wrt`
   /// \note the underlying `Operator` is owned by `residual`
   friend mfem::Operator &getJacobian(FlowControlResidual &residual,
                                      const MachInputs &inputs,
                                      std::string wrt);

   /// Evaluate the entropy/Lyapunov functional at the given state
   /// \param[inout] residual - flow-control residual whose entropy we want
   /// \param[in] inputs - the variables needed to evaluate the entropy
   /// \return the entropy functional
   /// \note optional, but must be implemented for relaxation RK
   friend double calcEntropy(FlowControlResidual &residual,
                             const MachInputs &inputs);

   /// Evaluate the flow-control residual weighted by the entropy variables
   /// \param[inout] residual - function with an associated entropy
   /// \param[in] inputs - the variables needed to evaluate the entropy
   /// \return the product `w^T res`
   /// \note `w` and `res` are evaluated at `state + dt*state_dot` and time
   /// `t+dt` as provided by `inputs`.
   /// \note optional, but must be implemented for relaxation RK
   friend double calcEntropyChange(FlowControlResidual &residual,
                                   const MachInputs &inputs);

   FlowControlResidual(
       mfem::ParFiniteElementSpace &pfes,
       std::unordered_map<std::string, mfem::ParGridFunction> &fields,
       const nlohmann::json &flow_options, 
       const nlohmann::json &space_dis_options,
       const nlohmann::json &control_options);

private:
   /// Defines the CFD discretization of the problem
   MachNonlinearForm flow_res;




};