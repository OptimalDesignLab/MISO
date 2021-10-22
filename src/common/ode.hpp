#ifndef MACH_ODE
#define MACH_ODE

#include "mfem.hpp"

#include "equation_solver.hpp"
#include "evolver.hpp"
#include "mach_residual.hpp"

namespace mach
{
class TimeDependentResidual final
{
public:
   friend int getSize(const TimeDependentResidual &residual);
   friend void setInputs(TimeDependentResidual &residual,
                         const mach::MachInputs &inputs);

   friend void setOptions(TimeDependentResidual &residual,
                          const nlohmann::json &options);

   friend void evaluate(TimeDependentResidual &residual,
                        const mach::MachInputs &inputs,
                        mfem::Vector &res_vec);

   friend mfem::Operator &getJacobian(TimeDependentResidual &residual,
                                      const mach::MachInputs &inputs,
                                      std::string wrt);

   friend double calcEntropy(TimeDependentResidual &residual,
                             const MachInputs &inputs);

   friend double calcEntropyChange(TimeDependentResidual &residual,
                                   const MachInputs &inputs);

   // template<typename T>
   // TimeDependentResidual(T spatial_res,
   //                       mfem::Operator *mass_matrix = nullptr);
   template <typename T>
   TimeDependentResidual(T spatial_res, mfem::Operator *mass_matrix = nullptr)
    : res_(std::move(spatial_res)),
      mass_matrix_(mass_matrix),
      work(getSize(res_))
   {
      if (mass_matrix_ == nullptr)
      {
         identity_ = std::make_unique<mfem::IdentityOperator>(getSize(res_));
         mass_matrix_ = identity_.get();
      }
      auto *hypre_mass = dynamic_cast<mfem::HypreParMatrix *>(mass_matrix_);
      auto *iden_mass = dynamic_cast<mfem::IdentityOperator *>(mass_matrix_);
      if (hypre_mass != nullptr)
      {
         jac_ = std::make_unique<mfem::HypreParMatrix>(*hypre_mass);
      }
      else if (iden_mass != nullptr)
      {
         jac_ = std::make_unique<mfem::DenseMatrix>(getSize(res_));
      }
   }

private:
   /// \brief reference to the residual that defines the dynamics of the ODE
   MachResidual res_;
   /// \brief pointer to mass matrix used for ODE integration
   mfem::Operator *mass_matrix_;
   /// \brief default mass matrix if none provided
   std::unique_ptr<mfem::IdentityOperator> identity_;
   /// \brief jacobian of combined spatial and temporal residual
   std::unique_ptr<mfem::Operator> jac_ = nullptr;

   double dt;
   double time;
   mfem::Vector state;
   mfem::Vector state_dot;

   mfem::Vector work;
};

class FirstOrderODE final : public EntropyConstrainedOperator
// class FirstOrderODE final : public mfem::TimeDependentOperator
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
   void Mult(const mfem::Vector &u, mfem::Vector &du_dt) const override
   {
      solve(0.0, u, du_dt);
   }

   /// \brief Solves the equation du_dt = f(u + dt * du_dt, t)
   /// \param[in] dt - the time step
   /// \param[in] u - the state true DOFs
   /// \param[in] du_dt - the first time derivative of u
   void ImplicitSolve(const double dt,
                      const mfem::Vector &u,
                      mfem::Vector &du_dt) override
   {
      solve(dt, u, du_dt);
   }

   /// \brief Evaluate the entropy functional at the given state
   /// \param[in] u - the state true DOFs at which to evaluate the entropy
   /// \returns the entropy functional
   double Entropy(const mfem::Vector &u) override
   {
      MachInputs input{{"state", u.GetData()}};
      return calcEntropy(residual_, input);
   }

   /// \brief Evaluate the residual weighted by the entropy variables
   /// \praam[in] dt - evaluate residual at t+dt
   /// \param[in] u - previous time step state
   /// \param[in] du_dt - the first time derivative of u
   /// \returns the product `w^T res`
   /// \note `w` and `res` are evaluated at `u + dt*du_dt` and time `t+dt`.
   double EntropyChange(double dt,
                        const mfem::Vector &u,
                        const mfem::Vector &du_dt) override
   {
      MachInputs inputs{
          {"state", u.GetData()}, {"state_dot", du_dt.GetData()}, {"dt", dt}};
      return calcEntropyChange(residual_, inputs);
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
