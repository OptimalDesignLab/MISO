#include "mfem_extensions.hpp"
#include "utils.hpp"

#include "ode.hpp"

namespace mach
{
FirstOrderODE::FirstOrderODE(MachResidual &residual,
                             const nlohmann::json &ode_options,
                             const EquationSolver &solver)
 //  : EntropyConstrainedOperator(getSize(residual), 0.0),
 : TimeDependentOperator(getSize(residual), 0.0),
   residual_(residual),
   solver_(solver),
   zero_(getSize(residual))
{
   zero_ = 0.0;
   setTimestepper(ode_options);
}

void FirstOrderODE::setTimestepper(const nlohmann::json &ode_options)
{
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
   MachInputs inputs{{"state", u.GetData()},
                     {"state_dot", du_dt.GetData()},
                     {"dt", dt},
                     {"time", t}};

   setInputs(residual_, inputs);
   solver_.Mult(zero_, du_dt);
   // SLIC_WARNING_ROOT_IF(!solver_.NonlinearSolver().GetConverged(), "Newton
   // Solver did not converge.");
}

}  // namespace mach
