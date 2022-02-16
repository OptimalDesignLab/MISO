#ifndef MACH_FLOW_CONTROL
#define MACH_FLOW_CONTROL

#include "mfem.hpp"
#include "nlohmann/json.hpp"
#include "adept.h"

#include "flow_residual.hpp"
#include "mach_input.hpp"
#include "matrix_operators.hpp"
#include "mfem_extensions.hpp"

namespace mach
{
/// Defines the ODE for passive system control-law
class ControlResidual final
{
public:

   /// Constructor
   /// \param[in] control_options - options used to define the residual
   /// \note the number of control variables is hard-coded.
   ControlResidual(MPI_Comm incomm, const nlohmann::json &control_options);

   /// Gets the number of control ODEs/unknwons
   /// \param[inout] residual - the residual whose size is being queried
   /// \returns the number of equations/unknowns
   friend int getSize(const ControlResidual &residual)
   {
      return residual.num_var;
   }

   /// Set inputs in the underlying passive-control residual
   /// \param[inout] residual - passive-control residual being assigned inputs
   /// \param[in] inputs - the inputs that are being assigned
   friend void setInputs(ControlResidual &residual, const MachInputs &inputs);

   /// Set options in the passive-control residual
   /// \param[inout] residual - passive-control residual whose options are set
   /// \param[in] options - the options that are being assigned
   friend void setOptions(ControlResidual &residual,
                          const nlohmann::json &options);

   /// Evaluate the passive-control residual using inputs and return `res_vec`
   /// \param[inout] residual - passive-control residual being evaluated
   /// \param[in] inputs - the independent variables at which to evaluate res
   /// \param[out] res_vec - the dependent variable, the output from `residual`
   friend void evaluate(ControlResidual &residual,
                        const MachInputs &inputs,
                        mfem::Vector &res_vec);

   /// Compute the "Jacobian" of the passive-control residual and return it
   /// \param[inout] residual - passive-control residual whose Jacobian we want
   /// \param[in] inputs - the variables needed to evaluate the Jacobian
   /// \param[in] wrt - the input we are differentiating with respect to
   /// \returns a reference to the residual's Jacobian with respect to `wrt`
   /// \note the underlying `Operator` is owned by `residual`
   friend mfem::Operator &getJacobian(ControlResidual &residual,
                                      const MachInputs &inputs,
                                      std::string wrt);

   /// Evaluate the entropy/Lyapunov functional at the given state
   /// \param[inout] residual - passive-control residual whose entropy we want
   /// \param[in] inputs - the variables needed to evaluate the entropy
   /// \return the entropy functional
   /// \note optional, but must be implemented for relaxation RK
   friend double calcEntropy(ControlResidual &residual,
                             const MachInputs &inputs);

   /// Evaluate the passive-control residual weighted by the entropy variables
   /// \param[inout] residual - function with an associated entropy
   /// \param[in] inputs - the variables needed to evaluate the entropy
   /// \return the product `w^T res`
   /// \note `w` and `res` are evaluated at `state` and time `t + dt`
   /// \note `res` is equal to `-state_dot`, but may be recomputed if necessary
   /// \note optional, but must be implemented for relaxation RK
   friend double calcEntropyChange(ControlResidual &residual,
                                   const MachInputs &inputs);

   /// Return mass matrix (operator) for the control equations
   /// \param[inout] residual - residual whose mass matrix is desired
   /// \param[in] mass_options - options (not presently used)
   /// \return pointer to mass matrix for the control equations
   /// \note The returned pointer is owned by the residual
   friend mfem::Operator *getMassMatrix(ControlResidual &residual,
                                        const nlohmann::json &mass_options)
   {
      return residual.mass_mat.get();
   }

   /// Return a preconditioner for the control state Jacobian
   /// \param[inout] residual - residual whose preconditioner is desired
   /// \param[in] prec_options - options specific to the preconditioner
   /// \return pointer to preconditioner for the state Jacobian
   /// \note The preconditioner's operator is set and factored when SetOperator
   /// is called by the linear solver's SetOperator
   friend mfem::Solver *getPreconditioner(ControlResidual &residual,
                                          const nlohmann::json &prec_options)
   {
      return residual.prec.get();
   }

   /// Return the control velocity, which can be then fed into a flow solver
   /// \param[in] inputs - holds the control state vector and boundary entropy
   /// \return the control velocity
   double getControlVelocity(const MachInputs &inputs);

private:
   /// communicator used by MPI group for communication
   MPI_Comm comm;
   /// MPI process rank
   int rank;
   /// number of control ODE variables/equations on this process
   int num_var;
   /// parameters in the control law
   double Kp, Td, Ti, beta, eta;
   /// desired, or target, flow entropy 
   double target_entropy;
   /// norm matrix that defines the Lyapunov function 
   std::unique_ptr<mfem::DenseMatrix> P;
   /// turn on/off closed loop term
   bool closed_loop;
   /// Stores the current simulation time
   double time;
   /// Stores the boundary entropy functional; provided by the flow solver
   double boundary_entropy;
   /// work vector to store the "state" (i.e. the control variables)
   mfem::Vector x;
   /// generic work vector
   mfem::Vector work;
   /// Mass matrix for the ODE
   /// \note cannot use the identity matrix because we need to add this to Jac
   std::unique_ptr<mfem::DenseMatrix> mass_mat;
   /// Jacobian of the ODE right-hand side
   std::unique_ptr<mfem::DenseMatrix> Jac;
   /// Preconditioner for the ODE Jacobian
   std::unique_ptr<mfem::DenseMatrixInverse> prec;

   static const bool test_ode = false;
};

/// Class for flow-control equations that follows the MachResidual API
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \tparam entvar - if true, the entropy variables are used in the integrators
/// \note We do not use friend functions in all cases with this class because
/// it is templated and would require a large number of forward declarations.
/// Instead, for more involved functions, we define member functions needed by
/// the MachResidual interface and then use these in non-friend functions.
template <int dim, bool entvar = false>
class FlowControlResidual final
{
public:
   /// Construct a flow-control residual object
   /// \param[in] options - options that define the flow and control problems
   /// \param[in] pfes - defines the finite-element space needed by flow
   /// \param[in] diff_stack - used for algorithmic differentiation
   /// \param[in] outstream - for writing information to screen/file
   FlowControlResidual(const nlohmann::json &options,
                       mfem::ParFiniteElementSpace &pfes,
                       adept::Stack &diff_stack,
                       std::ostream &outstream = std::cout);

   friend MPI_Comm getMPIComm(const FlowControlResidual &residual)
   {
      return getMPIComm(residual.flow_res);
   }

   /// Gets the number of equations/unknowns; # of FEM unknowns + # control vars
   /// \param[inout] residual - the residual whose size is being queried
   /// \returns the number of equations/unknowns
   friend int getSize(const FlowControlResidual &residual)
   {
      return getSize(residual.flow_res) + getSize(residual.control_res);
   }

   /// Set inputs in the underlying residual
   /// \param[in] inputs - the inputs that are being assigned
   void setInputs_(const MachInputs &inputs);

   /// Set options in the underlying residual type
   /// \param[inout] residual - flow-control residual whose options are set
   /// \param[in] options - the options that are being assigned
   friend void setOptions(FlowControlResidual &residual,
                          const nlohmann::json &options)
   {
      setOptions(residual.flow_res, options);
      setOptions(residual.control_res, options);
   }

   /// Evaluate the flow-control residual at the inputs and return as `res_vec`
   /// \param[inout] residual - the flow-control residual being evaluated
   /// \param[in] inputs - the independent variables at which to evaluate `res`
   /// \param[out] res_vec - the dependent variable, the output from `residual`
   /// \note This assumes that inputs like `time` have already been set using a
   /// call to `setInputs`
   void evaluate_(const MachInputs &inputs, mfem::Vector &res_vec);

   /// Compute the Jacobian of the flow-control residual and return a reference
   /// \param[in] inputs - the variables needed to evaluate the Jacobian
   /// \param[in] wrt - the input we are differentiating with respect to
   /// \returns a reference to the residual's Jacobian with respect to `wrt`
   /// \note the underlying `Operator` is owned by `residual`
   // mfem::Operator &getJacobian_(const MachInputs &inputs,
   //                             const std::string &wrt);
   friend mfem::Operator &getJacobian(FlowControlResidual &residual,
                                      const MachInputs &inputs,
                                      const std::string &wrt)
   {
      throw MachException(
          "getJacobian not implemented for FlowControlResidual!\n");
   }

   /// Evaluate the entropy/Lyapunov functional at the given state
   /// \param[in] inputs - the variables needed to evaluate the entropy
   /// \return the entropy functional
   /// \note optional, but must be implemented for relaxation RK
   double calcEntropy_(const MachInputs &inputs);

   /// Evaluate the flow-control residual weighted by the entropy variables
   /// \param[in] inputs - the variables needed to evaluate the entropy
   /// \return the product `w^T res`
   /// \note `w` and `res` are evaluated at `state + dt*state_dot` and time
   /// `t+dt` as provided by `inputs`.
   /// \note optional, but must be implemented for relaxation RK
   double calcEntropyChange_(const MachInputs &inputs);

   /// Return mass matrix (operator) for the flow-control equations
   /// \param[inout] residual - residual whose mass matrix is desired
   /// \param[in] mass_options - options (not presently used)
   /// \return pointer to mass matrix for the flow-control equations
   /// \note The returned pointer is owned by the residual
   friend mfem::Operator *getMassMatrix(FlowControlResidual &residual,
                                        const nlohmann::json &mass_options)
   {
      return residual.mass_mat.get();
   }

   /// Return a preconditioner for the flow-control state Jacobian
   /// \param[inout] residual - residual whose preconditioner is desired
   /// \param[in] prec_options - options specific to the preconditioner
   /// \return pointer to preconditioner for the state Jacobian
   /// \note Constructs the preconditioner, and the returned pointer is owned
   /// by the `residual`
   friend mfem::Solver *getPreconditioner(FlowControlResidual &residual,
                                          const nlohmann::json &prec_options)
   {
      return residual.prec.get();
   }

   /// Returns either the control (`i==0`) or flow Jacobian (`i==1`)
   /// \param[in] inputs - the variables needed to evaluate the Jacobian
   /// \param[in] i - selects either the control or flow Jacobian to return
   /// \returns the Jacobian operator for the control or flow residual
   mfem::Operator &getJacobianBlock_(const MachInputs &inputs, int i);

   /// Returns the minimum time step for a given state and CFL number
   /// \param[in] cfl - the target maximum allowable CFL number
   /// \param[in] state - the state which defines the velocity field
   /// \note The `state` here refers to the flow state only (it is a
   /// `ParGridFunction`)
   double minCFLTimeStep(double cfl, const mfem::ParGridFunction &state);

   /// Returns an output capable of evaluting `fun`
   /// \param[in] fun - the name of the output to be constructed
   /// \param[in] options - used to define the output
   /// \returns a `MachOutput`, which is **not** owned by residual
   MachOutput constructOutput(const std::string &fun,
                              const nlohmann::json &options);

   /// Helper function that separates `state` into control and flow states
   /// \param[in] state - vector holding both control and flow states
   /// \param[out] control_state - on exit, holds the control state vector
   /// \param[out] flow_state - on exit, holds the flow state vector
   /// \note No memory is allocated for the output states, they simply wrap the
   /// data passed in by state
   void extractStates(const mfem::Vector &state,
                      mfem::Vector &control_state,
                      mfem::Vector &flow_state) const;

   /// Helper function that extracts state input into separate vectors
   /// \param[in] inputs - must include a state input
   /// \param[out] control_state - on exit, holds the control state vector
   /// \param[out] flow_state - on exit, holds the flow state vector
   /// \note No memory is allocated for the output states, they simply wrap the
   /// data passed in by inputs.
   /// \note An exception is raised if `inputs` does not hold a `state` element.
   void extractStates(const MachInputs &inputs,
                      mfem::Vector &control_state,
                      mfem::Vector &flow_state) const;

   /// Set the given vector to the free-stream value
   /// \param[out] qfar - used to hold the free-stream state upon return
   void getFreeStreamState(mfem::Vector &qfar)
   {
      flow_res.getFreeStreamState(qfar);
   }

private:
   /// print object
   std::ostream &out;
   /// Offsets to mark the start of each row/column block; must be unique_ptr
   std::unique_ptr<mfem::Array<int>> offsets;
   /// position of the actuator
   mfem::Vector x_actuator;
   /// Defines the CFD discretization of the problem
   FlowResidual<dim, entvar> flow_res;
   /// Defines the control problem
   ControlResidual control_res;
   /// Block operator for the mass-matrix operator
   std::unique_ptr<mfem::BlockOperator> mass_mat;
   /// Preconditioner for the Jacobian
   std::unique_ptr<BlockJacobiPreconditioner> prec;
   /// Defines the flow output fed into the control
   mach::MachOutput boundary_entropy;
   /// The Jacobian-free operator
   // JacobianFree jac;
   /// Reference to control state-sized array; memory not owned
   mfem::Vector control_ref;
   /// Reference to flow state-sized array; memory not owned
   mfem::Vector flow_ref;
   /// Work vector for the control state
   mfem::Vector control_work;
   /// Work vector for the flow state
   mfem::Vector flow_work;

   // These could be public
   int num_control() const { return getSize(control_res); }
   int num_flow() const { return getSize(flow_res); }
};

/// Set inputs in the flow-control residual
/// \param[inout] residual - the flow-control residual being assigned inputs
/// \param[in] inputs - the inputs that are being assigned
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \tparam entvar - if true, the entropy variables are used in the integrators
template <int dim, bool entvar>
void setInputs(FlowControlResidual<dim, entvar> &residual,
               const MachInputs &inputs)
{
   residual.setInputs_(inputs);
}

/// Evaluate the fully-discrete flow-control equations
/// \param[inout] residual - defines the flow-control residual being evaluated
/// \param[in] inputs - defines values and fields needed for the evaluation
/// \param[out] res_vec - where the resulting residual is stored
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \tparam entvar - if true, the entropy variables are used in the integrators
/// \note The behavior of evaluate can be changed depending on whether the
/// residual is used in an explicit or implicit time integration.  This
/// behavior is controlled by setting `options["implicit"]` to true and
/// passing this to `setOptions`.
template <int dim, bool entvar>
void evaluate(FlowControlResidual<dim, entvar> &residual,
              const MachInputs &inputs,
              mfem::Vector &res_vec)
{
   residual.evaluate_(inputs, res_vec);
}

/// Returns the Jacobian of the fully-discrete flow-control equations
/// \param[inout] residual - the flow-control residual whose Jacobian is sought
/// \param[in] inputs - defines values and fields needed for the Jacobian
/// \param[out] wrt - variable that we want to differentiate with respect to
/// \returns a reference to an mfem Operator that defines the Jacobian
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \tparam entvar - if true, the entropy variables are used in the integrators
/// \note The behavior of getJacobian can be changed depending on whether
/// the residual is used in an explicit or implicit time integration.  This
/// behavior is controlled by setting `options["implicit"]` to true and
/// passing this to `setOptions`.
// template <int dim, bool entvar>
// mfem::Operator &getJacobian(FlowControlResidual<dim, entvar> &residual,
//                             const MachInputs &inputs,
//                             const std::string &wrt)
// {
//    return residual.getJacobian_(inputs, wrt);
// }

/// Returns the total integrated entropy for the flow-control problem
/// \param[inout] residual - the flow-control residual; helps compute entropy
/// \param[in] inputs - defines values and fields needed for the entropy
/// \returns the total entropy over the domain
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \tparam entvar - if true, the entropy variables are used in the integrators
/// \note The entropy depends only on the state, but the residual helps
/// distinguish if conservative or entropy-variables are used for the state.
template <int dim, bool entvar>
double calcEntropy(FlowControlResidual<dim, entvar> &residual,
                   const MachInputs &inputs)
{
   return residual.calcEntropy_(inputs);
}

/// Evaluate the residual weighted by the entropy variables
/// \param[inout] residual - flow-control residual with an associated entropy
/// \param[in] inputs - the variables needed to evaluate the entropy
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \tparam entvar - if true, the entropy variables are used in the integrators
/// \return the product `w^T res`
/// \note `w` and `res` are evaluated at `state` and time ` t + dt`
/// \note `res` is equal to `-state_dot`, but may be recomputed if necessary
/// \note optional, but must be implemented for relaxation RK
template <int dim, bool entvar>
double calcEntropyChange(FlowControlResidual<dim, entvar> &residual,
                         const MachInputs &inputs)
{
   return residual.calcEntropyChange_(inputs);
}

/// Returns either the control (`i==0`) or flow Jacobian (`i==1`)
/// \param[inout] residual - flow-control residual whose block we want
/// \param[in] inputs - the variables needed to evaluate the Jacobian
/// \param[in] i - selects either the control or flow Jacobian to return
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \tparam entvar - if true, the entropy variables are used in the integrators
/// \returns the Jacobian operator for the control or flow residual
template <int dim, bool entvar>
mfem::Operator &getJacobianBlock(FlowControlResidual<dim, entvar> &residual,
                                 const MachInputs &inputs,
                                 int i)
{
   return residual.getJacobianBlock_(inputs, i);
}

}  // namespace mach

#endif
