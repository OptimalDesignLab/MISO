#include "mfem.hpp"

#include "mfem_extensions.hpp"
#include "utils.hpp"

#include "ode.hpp"

namespace mach
{
/// \brief Helper function to add A + dt*B = C
/// Supports case where A, B, and C are all HypreParMatrix
/// or when A is an IdentityOperator and B and C are DenseMatrix
void addJacobians(mfem::Operator &A,
                  double dt,
                  mfem::Operator &B,
                  mfem::Operator &C)
{
   auto *hypre_A = dynamic_cast<mfem::HypreParMatrix *>(&A);
   auto *hypre_B = dynamic_cast<mfem::HypreParMatrix *>(&B);
   if (hypre_A != nullptr && hypre_B != nullptr)
   {
      auto *hypre_C = dynamic_cast<mfem::HypreParMatrix *>(&C);
      *hypre_C = 0.0;
      *hypre_C += *hypre_A;
      hypre_C->Add(dt, *hypre_B);
      return;
   }
   auto *iden_A = dynamic_cast<mfem::IdentityOperator *>(&A);
   auto *dense_B = dynamic_cast<mfem::DenseMatrix *>(&B);
   if (iden_A != nullptr && dense_B != nullptr)
   {
      auto *dense_C = dynamic_cast<mfem::DenseMatrix *>(&C);
      dense_C->Diag(1.0, dense_B->Width());
      dense_C->Add(dt, *dense_B);
      return;
   }
}

}  // namespace mach

namespace mach
{
int getSize(const TimeDependentResidual &residual)
{
   return getSize(residual.res_);
}

void setInputs(TimeDependentResidual &residual, const mach::MachInputs &inputs)
{
   // auto it = inputs.find("state");
   // if (it != inputs.end())
   // {
   //    residual.state.SetDataAndSize(it->second.getField(),
   //    getSize(residual));
   // }
   // it = inputs.find("state_dot");
   // if (it != inputs.end())
   // {
   //    residual.state_dot.SetDataAndSize(it->second.getField(),
   //                                      getSize(residual));
   // }
   // it = inputs.find("dt");
   // if (it != inputs.end())
   // {
   //    residual.dt = it->second.getValue();
   // }
   // it = inputs.find("time");
   // if (it != inputs.end())
   // {
   //    residual.time = it->second.getValue();
   // }

   setVectorFromInputs(inputs, "state", residual.state);
   setVectorFromInputs(inputs, "state_dot", residual.state_dot);
   setValueFromInputs(inputs, "dt", residual.dt);
   setValueFromInputs(inputs, "time", residual.time);
   setInputs(residual.res_, inputs);
}

void setOptions(TimeDependentResidual &residual, const nlohmann::json &options)
{
   setOptions(residual.res_, options);
}

void evaluate(TimeDependentResidual &residual,
              const mach::MachInputs &inputs,
              mfem::Vector &res_vec)
{
   setInputs(residual, inputs);
   auto &dt = residual.dt;
   auto &state = residual.state;
   auto &state_dot = residual.state_dot;
   auto &work = residual.work;
   if (dt == 0.0)
   {
      MachInputs input{{"state", state}};
      evaluate(residual.res_, input, res_vec);
   }
   else
   {
      add(state, dt, state_dot, work);
      MachInputs input{{"state", work}};
      evaluate(residual.res_, input, res_vec);
   }
   residual.mass_matrix_->Mult(residual.state_dot, residual.work);
   res_vec += residual.work;
}

mfem::Operator &getJacobian(TimeDependentResidual &residual,
                            const mach::MachInputs &inputs,
                            const std::string &wrt)
{
   auto &dt = residual.dt;
   /// dt is zero when using explicit time marching
   if (dt == 0.0)
   {
      return *residual.mass_matrix_;
   }

   auto &state = residual.state;
   auto &state_dot = residual.state_dot;
   auto &work = residual.work;
   add(state, dt, state_dot, work);
   MachInputs input{{"state", work}};
   auto &spatial_jac = getJacobian(residual.res_, input, wrt);
   addJacobians(*residual.mass_matrix_, dt, spatial_jac, *residual.jac_);
   return *residual.jac_;
}

double calcEntropy(TimeDependentResidual &residual, const MachInputs &inputs)
{
   return calcEntropy(residual.res_, inputs);
}

double calcEntropyChange(TimeDependentResidual &residual,
                         const MachInputs &inputs)
{
   return calcEntropyChange(residual.res_, inputs);
}

// template <typename T>
// TimeDependentResidual::TimeDependentResidual(T spatial_res,
//                                              mfem::Operator *mass_matrix)
//  : res_(std::move(spatial_res)), mass_matrix_(mass_matrix),
//  work(getSize(res_))
// {
//    if (mass_matrix_ == nullptr)
//    {
//       identity_ = std::make_unique<mfem::IdentityOperator>(getSize(res_));
//       mass_matrix_ = identity_.get();
//    }
//    auto *hypre_mass = dynamic_cast<mfem::HypreParMatrix *>(mass_matrix_);
//    auto *iden_mass = dynamic_cast<mfem::IdentityOperator *>(mass_matrix_);
//    if (hypre_mass != nullptr)
//    {
//       jac_ = std::make_unique<mfem::HypreParMatrix>(*hypre_mass);
//    }
//    else if (iden_mass != nullptr)
//    {
//       jac_ = std::make_unique<mfem::DenseMatrix>(getSize(res_));
//    }
// }

FirstOrderODE::FirstOrderODE(MachResidual &residual,
                             const nlohmann::json &ode_options,
                             mfem::Solver &solver)
 : EntropyConstrainedOperator(getSize(residual), 0.0),
   //  : TimeDependentOperator(getSize(residual), 0.0),
   residual_(residual),
   solver_(solver)
// zero_(getSize(residual))
{
   solver_.iterative_mode = false;
   // zero_ = 0.0;
   setTimestepper(ode_options);
}

void FirstOrderODE::setTimestepper(const nlohmann::json &ode_options)
{
   if (ode_options.contains("ode-solver"))
   {
      throw MachException(
          "The option key \"ode-solver\" is deprecated.\n"
          "Please use the key \"type\" instead.");
   }
   auto timestepper = ode_options["type"].get<std::string>();
   if (timestepper == "RK1")
   {
      ode_solver_ = std::make_unique<mfem::ForwardEulerSolver>();
   }
   else if (timestepper == "RK4")
   {
      ode_solver_ = std::make_unique<mfem::RK4Solver>();
   }
   else if (timestepper == "MIDPOINT")
   {
      ode_solver_ = std::make_unique<mfem::ImplicitMidpointSolver>();
   }
   else if (timestepper == "RRK")
   {
      ode_solver_ = std::make_unique<mach::RRKImplicitMidpointSolver>();
   }
   else if (timestepper == "PTC")
   {
      ode_solver_ = std::make_unique<mach::PseudoTransientSolver>();
   }
   else if (timestepper == "steady")
   {
      ode_solver_ = std::make_unique<mach::SteadyODESolver>();
      solver_.iterative_mode = true;
   }
   else
   {
      throw MachException("Unknown ODE solver type: " +
                          ode_options["ode-solver"].get<std::string>());
      // TODO: parallel exit
   }
   ode_solver_->Init(*this);
}

void FirstOrderODE::solve(const double dt,
                          const mfem::Vector &u,
                          mfem::Vector &du_dt) const
{
   MachInputs inputs{
       {"state", u}, {"state_dot", du_dt}, {"dt", dt}, {"time", t}};

   setInputs(residual_, inputs);
   solver_.Mult(zero_, du_dt);
   // SLIC_WARNING_ROOT_IF(!solver_.NonlinearSolver().GetConverged(), "Newton
   // Solver did not converge.");
}

}  // namespace mach
