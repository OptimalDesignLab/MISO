#ifndef MACH_LINEAR_EVOLVER
#define MACH_LINEAR_EVOLVER

#include "mfem.hpp"

#include "adept.h"
#include "mach_types.hpp"
#include "inexact_newton.hpp"
#include "mach_load.hpp"

namespace mach
{
/// For systems that are equipped with a non-increasing entropy function
class EntropyConstrainedOperator : public mfem::TimeDependentOperator
{
public:
   /// Default constructor
   EntropyConstrainedOperator(int n,
                              double start_time,
                              mfem::TimeDependentOperator::Type type)
    : TimeDependentOperator(n, start_time, type)
   { }

   /// Evaluate the entropy functional at the given state
   /// \param[in] state - the state at which to evaluate the entropy
   /// \returns the entropy functional
   virtual double Entropy(const mfem::Vector &state) = 0;

   /// Evaluate the residual weighted by the entropy variables
   /// \praam[in] dt - evaluate residual at t+dt
   /// \param[in] state - previous time step state
   /// \param[in] k - the approximate time derivative, `du/dt`
   /// \returns the product `w^T res`
   /// \note `w` and `res` are evaluated at `state + dt*k` and time `t+dt`.
   virtual double EntropyChange(double dt,
                                const mfem::Vector &state,
                                const mfem::Vector &k) = 0;

   using mfem::TimeDependentOperator::ImplicitSolve;

   /// Variant of `mfem::ImplicitSolve` for entropy constrained systems
   /// \param[in] dt_stage - the full step size
   /// \param[in] dt - a partial step, `dt` < `dt_stage`.
   /// \param[in] x - baseline state
   /// \param[out] k - the desired slope
   /// \note This may need to be generalized further
   virtual void ImplicitSolve(const double dt_stage,
                              const double dt,
                              const mfem::Vector &x,
                              mfem::Vector &k) = 0;
};

/// Class that can handle implicit or explicit time marching of linear or
/// nonlinear ODEs
class MachEvolver : public EntropyConstrainedOperator
{
public:
   /// Serves as an base class for linear/nonlinear explicit/implicit time
   /// marching problems
   /// \param[in] ess_bdr - flags for essential boundary
   /// \param[in] nonlinear_mass - nonlinear mass operator (not owned)
   /// \param[in] mass - bilinear form for mass matrix (not owned)
   /// \param[in] res - nonlinear residual operator (not owned)
   /// \param[in] stiff - bilinear form for stiffness matrix (not owned)
   /// \param[in] load - load vector (not owned)
   /// \param[in] ent - nonlinear form for entropy/energy (not owned)
   /// \param[in] out - outstream to use pointer (not owned)
   /// \param[in] start_time - time to start integration from
   ///                         (important for time-variant sources)
   /// \param[in] type - solver type; explicit or implicit
   MachEvolver(mfem::Array<int> &ess_bdr,
               NonlinearFormType *nonlinear_mass,
               BilinearFormType *mass,
               NonlinearFormType *res,
               BilinearFormType *stiff,
               MachLoad *load,
               NonlinearFormType *ent,
               std::ostream &outstream,
               double start_time,
               mfem::TimeDependentOperator::Type type = EXPLICIT,
               bool abort_on_no_convergence = true);

   /// Perform the action of the operator: y = k = f(x, t), where k solves
   /// the algebraic equation F(x, k, t) = G(x, t) and t is the current time.
   /// Compute k = M^-1(R(x,t) + Kx + l)
   void Mult(const mfem::Vector &x, mfem::Vector &y) const override;

   /// Solve the implicit equation: k = f(x + dt k, t), for the unknown k at
   /// the current time t.
   /// Currently implemented for the implicit midpoit method
   void ImplicitSolve(const double dt,
                      const mfem::Vector &x,
                      mfem::Vector &k) override;

   /// Variant of `mfem::ImplicitSolve` for entropy constrained systems
   /// \param[in] dt_stage - the full step size
   /// \param[in] dt - a partial step, `dt` < `dt_stage`.
   /// \param[in] x - baseline state
   /// \param[out] k - the desired slope
   /// \note This may need to be generalized further
   void ImplicitSolve(const double dt_stage,
                      const double dt,
                      const mfem::Vector &x,
                      mfem::Vector &k) override;

   /// Set the linear solver to be used for implicit methods
   /// \param[in] linsolver - pointer to configured linear solver (not owned)
   void SetLinearSolver(mfem::Solver *linsolver);

   /// Set the newton solver to be used for implicit methods
   /// \param[in] newton - pointer to configured newton solver (not owned)
   void SetNewtonSolver(mfem::NewtonSolver *newton);

   /// Return a reference to the Jacobian of the combined operator
   /// \param[in] x - the current state
   mfem::Operator &GetGradient(const mfem::Vector &x) const override;

   /// Evaluate the entropy functional at the given state
   /// \param[in] x - the state at which to evaluate the entropy
   /// \returns the entropy functional
   /// \note optional, but must be implemented for relaxation RK
   double Entropy(const mfem::Vector &x) override;

   /// Evaluate the residual weighted by the entropy variables
   /// \praam[in] dt - evaluate residual at t+dt
   /// \param[in] x - previous time step state
   /// \param[in] k - the approximate time derivative, `du/dt`
   /// \returns the product `w^T res`
   /// \note `w` and `res` are evaluated at `x + dt*k` and time `t+dt`
   /// \note optional, but must be implemented for relaxation RK
   double EntropyChange(double dt,
                        const mfem::Vector &x,
                        const mfem::Vector &k) override;

   virtual ~MachEvolver();

protected:
   /// pointer to nonlinear mass bilinear form (not owned)
   NonlinearFormType *nonlinear_mass;
   /// pointer to mass bilinear form (not owned)
   mfem::OperatorHandle mass;
   /// pointer to nonlinear form (not owned)
   NonlinearFormType *res;
   /// pointer to stiffness bilinear form (not owned)
   mfem::OperatorHandle stiff;
   /// pointer to load vector (not owned)
   MachLoad *load;
   /// pointer to a form for computing the entropy  (not owned)
   NonlinearFormType *ent;
   /// outstream for printing
   std::ostream &out;
   /// solver for inverting mass matrix for explicit solves
   /// \note supports partially assembled mass bilinear form
   mfem::CGSolver mass_solver;
   /// preconditioner for inverting mass matrix
   std::unique_ptr<mfem::Solver> mass_prec;
   /// Linear solver for implicit problems (not owned)
   mfem::Solver *linsolver;
   /// Newton solver for implicit problems (not owned)
   mfem::NewtonSolver *newton;
   /// essential degrees of freedom
   mfem::Array<int> ess_tdof_list;
   /// work vectors
   mutable mfem::Vector x_work, r_work1, r_work2;
   /// flag that determines if program should abort if Newton's method does not
   /// converge
   bool abort_on_no_converge;

   /// pointer-to-implementation idiom
   /// Hides implementation details of this operator, and because it's private,
   /// it doesn't pollute the mach namespace
   class SystemOperator;
   /// Operator that combines the linear/nonlinear spatial discretization with
   /// the load vector into one operator used for implicit solves
   std::unique_ptr<SystemOperator> combined_oper;

   /// sets the state and dt for the combined operator
   /// \param[in] dt - time increment
   /// \param[in] x - the current state
   /// \param[in] dt_stage - time step for full stage/step
   void setOperParameters(double dt,
                          const mfem::Vector *x,
                          double dt_stage = -1.0);
};

}  // namespace mach

#endif
