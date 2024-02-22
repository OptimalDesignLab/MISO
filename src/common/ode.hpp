#ifndef MISO_ODE
#define MISO_ODE

#include "mfem.hpp"

#include "evolver.hpp"
#include "miso_residual.hpp"
#include "matrix_operators.hpp"

namespace miso
{
/// Builds a space-time residual out of a given spatial residual
/// \note The class assumes the mass matrix is a HypreParMatrix.  If we need to
/// consider nonlinear mass operators (e.g. entropy-stable DGD), we can provide
/// the mass operator as a separate residual.
class TimeDependentResidual final
{
public:
   friend int getSize(const TimeDependentResidual &residual);
   friend void setInputs(TimeDependentResidual &residual,
                         const miso::MISOInputs &inputs);

   friend void setOptions(TimeDependentResidual &residual,
                          const nlohmann::json &options);

   /// Evaluates the residual `M du_dt + R(u, p, t) = 0`
   friend void evaluate(TimeDependentResidual &residual,
                        const miso::MISOInputs &inputs,
                        mfem::Vector &res_vec);

   /// Returns the Jacobian of the residual with respect to `du_dt`
   friend mfem::Operator &getJacobian(TimeDependentResidual &residual,
                                      const miso::MISOInputs &inputs,
                                      const std::string &wrt);

   friend double calcEntropy(TimeDependentResidual &residual,
                             const MISOInputs &inputs);

   friend double calcEntropyChange(TimeDependentResidual &residual,
                                   const MISOInputs &inputs);

   friend double calcSupplyRate(TimeDependentResidual &residual,
                                const MISOInputs &inputs);

   friend mfem::Solver *getPreconditioner(TimeDependentResidual &residual,
                                          const nlohmann::json &options)
   {
      return getPreconditioner(residual.spatial_res_);
   }

   /** \brief constructs a time dependent residual of the form
            M du_dt + R(u, p, t) = 0

       \param[in] spatial_res - Reference to externally owned spatial residual
       \param[in] mass_matrix - Non-owning pointer to the optional mass matrix
       \tparam T - The concrete type of the spatial residual
       \note If no mass matrix is provided an IdentityOperator will be used */
   template <typename T>
   TimeDependentResidual(T &spatial_res, mfem::Operator *mass_matrix = nullptr)
    : spatial_res_(spatial_res),
      mass_matrix_(mass_matrix),
      work(getSize(spatial_res_))
   {
      /// If no mass matrix is provided, we'll use an IdentityOperator
      if (mass_matrix_ == nullptr)
      {
         identity_ =
             std::make_unique<mfem::IdentityOperator>(getSize(spatial_res_));
         mass_matrix_ = identity_.get();
      }

      /// Determine what type of mass matrix we're using and pre-allocate the
      /// Jacobian
      auto *hypre_mass = dynamic_cast<mfem::HypreParMatrix *>(mass_matrix_);
      auto *iden_mass = dynamic_cast<mfem::IdentityOperator *>(mass_matrix_);
      auto *block_mass = dynamic_cast<mfem::BlockOperator *>(mass_matrix_);
      if (hypre_mass != nullptr)
      {
         jac_ = std::make_unique<mfem::HypreParMatrix>(*hypre_mass);
      }
      else if (iden_mass != nullptr)
      {
         jac_ = std::make_unique<mfem::DenseMatrix>(getSize(spatial_res_));
      }
      else if (block_mass != nullptr)
      {
         jac_ = std::make_unique<JacobianFree>(spatial_res_, *mass_matrix_);
      }
   }

private:
   /// \brief reference to the residual that defines the dynamics of the ODE
   MISOResidual &spatial_res_;
   /// \brief pointer to mass matrix used for ODE integration
   mfem::Operator *mass_matrix_;
   /// \brief default mass matrix if none provided
   std::unique_ptr<mfem::IdentityOperator> identity_ = nullptr;
   /// \brief jacobian of combined spatial and temporal residual
   std::unique_ptr<mfem::Operator> jac_ = nullptr;

   double dt = NAN;
   double time = NAN;
   mfem::Vector state;
   mfem::Vector state_dot;

   mfem::Vector work;
};

class FirstOrderODE final : public EntropyConstrainedOperator
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
       mfem::TimeDependentOperator::ImplicitSolve (described in more detail
      here:
       https://mfem.github.io/doxygen/html/classmfem_1_1TimeDependentOperator.html)

       where

       mfem::TimeDependentOperator::Mult corresponds to the case where dt is
       zero mfem::TimeDependentOperator::ImplicitSolve corresponds to the case
       where dt is nonzero */
   FirstOrderODE(MISOResidual &residual,
                 const nlohmann::json &ode_options,
                 mfem::Solver &solver,
                 std::ostream *out_stream = nullptr);

   /// \brief Performs a time step
   /// \param[inout] u - the predicted solution
   /// \param[inout] time - the current time
   /// \param[inout] dt - the desired time step
   /// \see mfem::ODESolver::Step
   void step(mfem::Vector &u, double &time, double &dt)
   {
      ode_solver_->Step(u, time, dt);
   }

   /// \brief Solves the equation `M du_dt + R(u, p, t) = 0` for du_dt
   /// \param[in] u - the state true DOFs
   /// \param[in] du_dt - the first time derivative of u
   void Mult(const mfem::Vector &u, mfem::Vector &du_dt) const override
   {
      solve(0.0, u, du_dt);
   }

   /// \brief Solves the equation `M du_dt + R(u + dt * du_dt, p, t + dt) = 0`
   /// for `du_dt`
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
   /// \return the entropy functional
   double Entropy(const mfem::Vector &u) override
   {
      MISOInputs input{{"state", u}};
      return calcEntropy(residual_, input);
   }

   /// \brief Evaluate the spatial residual weighted by the entropy variables
   /// \param[in] dt - evaluate residual at t+dt
   /// \param[in] u - previous time step state
   /// \param[in] du_dt - the first time derivative of u
   /// \return the product `w^T R(u + dt * du_dt, p, t + dt)`
   /// \note The entropy variables, `w`, are evaluated at `u`, and `R` is
   /// equal to `-du_dt`.  Or, if necessary, `u` and `dt` can be used to
   /// evaluate `R`.
   double EntropyChange(double dt,
                        const mfem::Vector &u,
                        const mfem::Vector &du_dt) override
   {
      MISOInputs inputs{
          {"state", u}, {"state_dot", du_dt}, {"time", t}, {"dt", dt}};
      return calcEntropyChange(residual_, inputs);
   }

   double SupplyRate(double dt,
                     const mfem::Vector &u,
                     const mfem::Vector &du_dt) override
   {
      MISOInputs inputs{
          {"state", u}, {"state_dot", du_dt}, {"time", t}, {"dt", dt}};
      return calcSupplyRate(residual_, inputs);
   }

private:
   /// \brief reference to the underlying residual that defines the dynamics of
   /// the ODE
   MISOResidual &residual_;

   /// \brief reference to the equation solver used to solve for du_dt
   mfem::Solver &solver_;

   /// \brief MFEM solver object for first-order ODEs
   std::unique_ptr<mfem::ODESolver> ode_solver_;

   /// print object
   std::ostream *out;

   mfem::Vector zero_;

   /// \brief Internal implementation used for mfem::TDO::Mult and
   /// mfem::TDO::ImplicitSolve
   /// \param[in] dt The time step
   /// \param[in] u The true DOFs
   /// \param[in] du_dt The first time derivative of u
   virtual void solve(const double dt,
                      const mfem::Vector &u,
                      mfem::Vector &du_dt) const;

   /// \brief Set the time integration method
   /// \param[in] ode_options - options for the construction of the ode solver
   void setTimestepper(const nlohmann::json &ode_options);
};

}  // namespace miso

#endif
