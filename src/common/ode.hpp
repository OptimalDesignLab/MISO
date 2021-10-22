#ifndef MACH_ODE
#define MACH_ODE

#include "mfem.hpp"

#include "equation_solver.hpp"
#include "evolver.hpp"
#include "mach_residual.hpp"

namespace mach
{
// class FirstOrderODE final : public EntropyConstrainedOperator
class FirstOrderODE final : public mfem::TimeDependentOperator
{
public:
   /** \brief Constructor defining the size and specific system of ordinary
      differential equations to be solved

      \param[in] residual - reference to the underlying residual that defines
                 the dynamics of the ODE
      \param[in] ode_options - options for the construction of the ode solver
      \param[in] solver - the solver that operates on
                 the residual

      Implements mfem::TimeDependentOperator::Mult and
      mfem::TimeDependentOperator::ImplicitSolve (described in more detail here:
      https://mfem.github.io/doxygen/html/classmfem_1_1TimeDependentOperator.html)

      where

      mfem::TimeDependentOperator::Mult corresponds to the case where dt is zero
      mfem::TimeDependentOperator::ImplicitSolve corresponds to the case where
      dt is nonzero */
   FirstOrderODE(MachResidual &residual,
                 const nlohmann::json &ode_options,
                 const EquationSolver &solver);

   /// \brief Performs a time step
   /// \param[inout] u - the predicted solution
   /// \param[inout] time - the current time
   /// \param[inout] dt - the desired time step
   /// \see mfem::ODESolver::Step
   void step(mfem::Vector &u, double &time, double &dt)
   {
      ode_solver_->Step(u, time, dt);
   }

   /// \brief Solves the equation du_dt = f(u, t)
   /// \param[in] u - the state true DOFs
   /// \param[in] du_dt - the first time derivative of u
   void Mult(const mfem::Vector &u, mfem::Vector &du_dt) const
   {
      solve(0.0, u, du_dt);
   }

   /// \brief Solves the equation du_dt = f(u + dt * du_dt, t)
   /// \param[in] dt - the time step
   /// \param[in] u - the state true DOFs
   /// \param[in] du_dt - the first time derivative of u
   void ImplicitSolve(const double dt,
                      const mfem::Vector &u,
                      mfem::Vector &du_dt)
   {
      solve(dt, u, du_dt);
   }

private:
   /// \brief reference to the underlying residual that defines the dynamics of
   /// the ODE
   MachResidual &residual_;

   /// \brief reference to the equation solver used to solve for du_dt
   const EquationSolver &solver_;

   /// \brief MFEM solver object for first-order ODEs
   std::unique_ptr<mfem::ODESolver> ode_solver_;

   mfem::Vector zero_;

   /// \brief Internal implementation used for mfem::TDO::Mult and
   /// mfem::TDO::ImplicitSolve \param[in] dt The time step \param[in] u The
   /// true DOFs \param[in] du_dt The first time derivative of u
   virtual void solve(const double dt,
                      const mfem::Vector &u,
                      mfem::Vector &du_dt) const;

   /// \brief Set the time integration method
   /// \param[in] ode_options - options for the construction of the ode solver
   void setTimestepper(const nlohmann::json &ode_options);

   /// \brief Work vectors for ODE
   mutable mfem::Vector U_minus_;
   mutable mfem::Vector U_;
   mutable mfem::Vector U_plus_;
   mutable mfem::Vector dU_dt_;
};

}  // namespace mach

#endif
