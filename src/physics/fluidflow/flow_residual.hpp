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
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \tparam entvar - if true, the entropy variables are used in the integrators
/// \note We do not use friend functions with this class because it is 
/// templated, which would require a large number of forward declaration.
/// Instead we define member functions needed by the MachResidual interface, 
/// and then use these in non-friend functions.
template <int dim, bool entvar = false>
class FlowResidual final
{
public:
   /// Constructor for flow equations
   FlowResidual(const nlohmann::json &options,
                mfem::ParFiniteElementSpace &fespace,
                adept::Stack &diff_stack);

   /// Returns the number of equations/unknowns in the flow system
   int getSize_() const;

   /// Set inputs in the given flow residual object
   /// \param[in] inputs - defines values and fields being set
   void setInputs_(const MachInputs &inputs);

   /// Set options in the given flow residual object
   /// \param[in] options - object containing the options
   void setOptions_(const nlohmann::json &options);

   /// Evaluate the fully-discrete flow residual equations
   /// \param[in] inputs - defines values and fields needed for the evaluation
   /// \param[out] res_vec - where the resulting residual is stored
   /// \note The behavior of evaluate can be changed depending on whether the
   /// residual is used in an explicit or implicit time integration.  This
   /// behavior is controlled by setting `options["implicit"]` to true and
   /// passing this to `setOptions`.
   void evaluate_(const MachInputs &inputs, mfem::Vector &res_vec);

   /// Returns the Jacobian of the fully-discrete flow residual equations
   /// \param[in] inputs - defines values and fields needed for the Jacobian
   /// \param[out] wrt - variable that we want to differentiate with respect to
   /// \returns a reference to an mfem Operator that defines the Jacobian
   /// \note The behavior of getJacobian can be changed depending on whether
   /// the residual is used in an explicit or implicit time integration.  This
   /// behavior is controlled by setting `options["implicit"]` to true and
   /// passing this to `setOptions`.
   mfem::Operator &getJacobian_(const MachInputs &inputs,
                                const std::string &wrt);

   /// Returns the total integrated entropy over the domain
   /// \param[in] inputs - defines values and fields needed for the entropy
   /// \returns the total entropy over the domain
   /// \note The entropy depends only on the state, but the residual helps
   /// distinguish if conservative or entropy-variables are used for the state.
   double calcEntropy_(const MachInputs &inputs);

   /// Evaluate the residual weighted by the entropy variables
   /// \param[in] inputs - the variables needed to evaluate the entropy
   /// \return the product `w^T res`
   /// \note `w` and `res` are evaluated at `state + dt*state_dot` and time
   /// `t+dt`
   /// \note optional, but must be implemented for relaxation RK
   double calcEntropyChange_(const MachInputs &inputs);

   /// Returns the minimum time step for a given state and CFL number
   /// \param[in] cfl - the target maximum allowable CFL number
   /// \param[in] state - the state which defines the velocity field
   double minCFLTimeStep(double cfl, const mfem::ParGridFunction &state);

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

   void addFlowIntegrators(const nlohmann::json &options);

   void addFlowDomainIntegrators(const nlohmann::json &flow,
                                 const nlohmann::json &space_dis);

   void addFlowInterfaceIntegrators(const nlohmann::json &flow,
                                    const nlohmann::json &space_dis);

   void addFlowBoundaryIntegrators(const nlohmann::json &flow,
                                   const nlohmann::json &space_dis,
                                   const nlohmann::json &bcs);

   void addEntropyIntegrators();
};

/// Returns the number of equations/unknowns in the flow system
/// \param[in] residual - flow residual being queried
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \tparam entvar - if true, the entropy variables are used in the integrators
template <int dim, bool entvar>
int getSize(const FlowResidual<dim, entvar> &residual)
{
   return residual.getSize_();
}

/// Set inputs in the given flow residual
/// \param[inout] residual - flow residual whose inputs are set
/// \param[in] inputs - defines values and fields being set
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \tparam entvar - if true, the entropy variables are used in the integrators
template <int dim, bool entvar>
void setInputs(FlowResidual<dim, entvar> &residual, const MachInputs &inputs)
{
   residual.setInputs_(inputs);
}

/// Set options in the given flow residual
/// \param[inout] residual - flow residual whose options are being set
/// \param[in] options - object containing the options
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \tparam entvar - if true, the entropy variables are used in the integrators
template <int dim, bool entvar>
void setOptions(FlowResidual<dim, entvar> &residual,
                const nlohmann::json &options)
{
   residual.setOptions_(options);
}

/// Evaluate the fully-discrete flow residual equations
/// \param[inout] residual - defines the flow residual being evaluated
/// \param[in] inputs - defines values and fields needed for the evaluation
/// \param[out] res_vec - where the resulting residual is stored
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \tparam entvar - if true, the entropy variables are used in the integrators
/// \note The behavior of evaluate can be changed depending on whether the
/// residual is used in an explicit or implicit time integration.  This
/// behavior is controlled by setting `options["implicit"]` to true and
/// passing this to `setOptions`.
template <int dim, bool entvar>
void evaluate(FlowResidual<dim, entvar> &residual, const MachInputs &inputs,
              mfem::Vector &res_vec)
{
   residual.evaluate_(inputs, res_vec);
}

/// Returns the Jacobian of the fully-discrete flow residual equations
/// \param[inout] residual - the flow residual whose Jacobian is sought
/// \param[in] inputs - defines values and fields needed for the Jacobian
/// \param[out] wrt - variable that we want to differentiate with respect to
/// \returns a reference to an mfem Operator that defines the Jacobian
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \tparam entvar - if true, the entropy variables are used in the integrators
/// \note The behavior of getJacobian can be changed depending on whether
/// the residual is used in an explicit or implicit time integration.  This
/// behavior is controlled by setting `options["implicit"]` to true and
/// passing this to `setOptions`.
template <int dim, bool entvar>
mfem::Operator &getJacobian(FlowResidual<dim, entvar> &residual,
                            const MachInputs &inputs,
                            const std::string &wrt)
{
   return residual.getJacobian_(inputs, wrt);
}

/// Returns the total integrated entropy over the domain
/// \param[inout] residual - the flow residual, which helps compute entropy
/// \param[in] inputs - defines values and fields needed for the entropy
/// \returns the total entropy over the domain
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \tparam entvar - if true, the entropy variables are used in the integrators
/// \note The entropy depends only on the state, but the residual helps
/// distinguish if conservative or entropy-variables are used for the state.
template <int dim, bool entvar>
double calcEntropy(FlowResidual<dim, entvar> &residual,
                   const MachInputs &inputs)
{
   return residual.calcEntropy_(inputs);
}

/// Evaluate the residual weighted by the entropy variables
/// \param[inout] residual - function with an associated entropy
/// \param[in] inputs - the variables needed to evaluate the entropy
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \tparam entvar - if true, the entropy variables are used in the integrators
/// \return the product `w^T res`
/// \note `w` and `res` are evaluated at `state + dt*state_dot` and time
/// `t+dt` \note optional, but must be implemented for relaxation RK
template <int dim, bool entvar>
double calcEntropyChange(FlowResidual<dim, entvar> &residual,
                         const MachInputs &inputs)
{
   return residual.calcEntropyChange_(inputs);
}

}  // namespace mach

#endif  // FLOW_RESIDUAL