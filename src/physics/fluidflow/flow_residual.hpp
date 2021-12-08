#ifndef FLOW_RESIDUAL
#define FLOW_RESIDUAL

#include "mfem.hpp"
#include "nlohmann/json.hpp"
#include "adept.h"

#include "mach_input.hpp"
#include "mach_nonlinearform.hpp"
#include "functional_output.hpp"

namespace mach
{
/// Class for flow equations that follows the MachResidual API
class FlowResidual final
{
public:
   /// Constructor for flow equations
   FlowResidual(const nlohmann::json &options,
                 mfem::ParFiniteElementSpace &fespace,
                 adept::Stack &diff_stack);

   /// Returns the number of equations/unknowns in the flow system
   /// \param[in] residual - flow residual being queried
   friend int getSize(const FlowResidual &residual);

   /// Set inputs in the given flow residual
   /// \param[inout] residual - flow residual whose inputs are set
   /// \param[in] inputs - defines values and fields being set
   friend void setInputs(FlowResidual &residual, const MachInputs &inputs);

   /// Set options in the given flow residual
   /// \param[inout] residual - flow residual whose options are being set
   /// \param[in] options - object containing the options
   friend void setOptions(FlowResidual &residual,
                          const nlohmann::json &options);

   /// Evaluate the fully-discrete flow residual equations
   /// \param[inout] residual - defines the flow residual being evaluated
   /// \param[in] inputs - defines values and fields needed for the evaluation
   /// \param[out] res_vec - where the resulting residual is stored
   /// \note The behavior of evaluate can be changed depending on whether the
   /// residual is used in an explicit or implicit time integration.  This
   /// behavior is controlled by setting `options["implicit"]` to true and
   /// passing this to `setOptions`.
   friend void evaluate(FlowResidual &residual,
                        const MachInputs &inputs,
                        mfem::Vector &res_vec);

   /// Returns the Jacobian of the fully-discrete flow residual equations
   /// \param[inout] residual - the flow residual whose Jacobian is sought
   /// \param[in] inputs - defines values and fields needed for the Jacobian
   /// \param[out] wrt - variable that we want to differentiate with respect to
   /// \note The behavior of getJacobian can be changed depending on whether
   /// the residual is used in an explicit or implicit time integration.  This
   /// behavior is controlled by setting `options["implicit"]` to true and
   /// passing this to `setOptions`.
   friend mfem::Operator &getJacobian(FlowResidual &residual,
                                      const MachInputs &inputs,
                                      const std::string &wrt);

   /// Returns the total integrated entropy over the domain
   /// \param[inout] residual - the flow residual, which helps compute entropy
   /// \param[in] inputs - defines values and fields needed for the entropy
   /// \note The entropy depends only on the state, but the residual helps
   /// distinguish if conservative or entropy-variables are used for the state.
   friend double calcEntropy(FlowResidual &residual, const MachInputs &inputs);

   /// Evaluate the residual weighted by the entropy variables
   /// \param[inout] residual - function with an associated entropy
   /// \param[in] inputs - the variables needed to evaluate the entropy
   /// \return the product `w^T res`
   /// \note `w` and `res` are evaluated at `state + dt*state_dot` and time
   /// `t+dt` \note optional, but must be implemented for relaxation RK
   friend double calcEntropyChange(FlowResidual &residual,
                                   const MachInputs &inputs);

private:
   /// free-stream Mach number
   double mach_fs;
   /// free-stream angle of attack
   double aoa_fs;
   /// index of dimension corresponding to nose to tail axis
   int iroll;
   /// index of "vertical" dimension in body frame
   int ipitch;
   /// if true, the states passed in are assumed to be the entropy variables
   bool state_is_entvar = false;
   /// Determines if the residual is for explicit or implicit time-marching
   bool is_implicit;
   /// Finite-element space associated with inputs to the residual
   mfem::ParFiniteElementSpace &fes;
   /// stack used for algorithmic differentiation
   adept::Stack &stack;
   /// TBD
   std::unique_ptr<std::unordered_map<std::string, mfem::ParGridFunction>>
       fields;
   /// Defines the nonlinear form used to compute the residual and its Jacobian
   mach::MachNonlinearForm res;
   /// Defines the output used to evaluate the entropy
   mach::FunctionalOutput ent;
   /// Work vector
   mfem::Vector work;

   template <int dim>
   void addFlowIntegrators(const nlohmann::json &options);

   template <int dim, bool entvar = false>
   void addFlowDomainIntegrators(const nlohmann::json &flow,
                                  const nlohmann::json &space_dis);

   template <int dim, bool entvar = false>
   void addFlowInterfaceIntegrators(const nlohmann::json &flow,
                                     const nlohmann::json &space_dis);

   template <int dim, bool entvar = false>
   void addFlowBoundaryIntegrators(const nlohmann::json &flow,
                                    const nlohmann::json &space_dis,
                                    const nlohmann::json &bcs);

   template <int dim, bool entvar = false>
   void addEntropyIntegrators();
};

}  // namespace mach

#endif  // FLOW_RESIDUAL