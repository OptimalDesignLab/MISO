#include "euler_integ.hpp"
#include "flow_solver.hpp"
#include "flow_residual.hpp"
#include "mfem_extensions.hpp"
#include "euler_fluxes.hpp"

#include "diag_mass_integ.hpp"  /// TEMP !!!!

using namespace std;
using namespace mfem;

/// Return the number of flow state variables
/// \param[in] solver_options - helps define/determine the number of states
/// \param[in] space_dim - the number of spatial dimensions, consistent mesh
/// \todo update for any RANS models, since this assumes no additional states
int getNumFlowStates(const nlohmann::json &solver_options, int space_dim)
{
   return space_dim + 2;
}

namespace miso
{
template <int dim, bool entvar>
FlowSolver<dim, entvar>::FlowSolver(MPI_Comm incomm,
                                    const nlohmann::json &solver_options,
                                    std::unique_ptr<mfem::Mesh> smesh)
 : PDESolver(incomm, solver_options, getNumFlowStates, std::move(smesh))
{
   // Check for consistency between the template parameters, mesh, and options
   if (mesh().SpaceDimension() != dim)
   {
      throw MISOException(
          "FlowSolver<dim,entvar> constructor:\n"
          "\tMesh space dimension does not match template"
          "parameter dim");
   }
   bool ent_state = options["flow-param"]["entropy-state"];
   if (ent_state != entvar)
   {
      throw MISOException(
          "FlowSolver<dim,entvar> constructor:\n"
          "\tentropy-state option is inconsistent with entvar"
          "template parameter");
   }
   if ((entvar) && (!options["time-dis"]["steady"]))
   {
      throw MISOException(
          "FlowSolver<dim,entvar> constructor:\n"
          "\tnot set up for using entropy-variables as states for unsteady "
          "problem (need nonlinear mass-integrator).");
   }

   // Construct spatial residual and the space-time residual
   spatial_res = std::make_unique<miso::MISOResidual>(
       FlowResidual<dim, entvar>(options, fes(), fields, diff_stack, *out));
   auto *mass_matrix = getMassMatrix(*spatial_res, options);
   space_time_res = std::make_unique<miso::MISOResidual>(
       miso::TimeDependentResidual(*spatial_res, mass_matrix));

   // get the preconditioner, and construct the linear solver and nonlinear
   // solver
   auto *prec = getPreconditioner(*spatial_res);
   const auto &lin_solver_opts = options["lin-solver"];
   linear_solver = constructLinearSolver(comm, lin_solver_opts, prec);
   const auto &nonlin_solver_opts = options["nonlin-solver"];
   nonlinear_solver =
       constructNonlinearSolver(comm, nonlin_solver_opts, *linear_solver);
   nonlinear_solver->SetOperator(*space_time_res);

   // construct the ODE solver (also used for pseudo-transient continuation)
   const auto &ode_opts = options["time-dis"];
   ode = make_unique<FirstOrderODE>(
       *space_time_res, ode_opts, *nonlinear_solver, out);

   auto log_inital_state  = options["paraview"].value("intial-state",false);
   auto log_final_state   = options["paraview"].value("final-state",false);
   auto log_each_timestep = options["paraview"].value("each-timestep",false);

   if (options["paraview"].at("each-timestep") || log_inital_state || log_final_state)
   {
      ParaViewLogger paraview(options["paraview"]["directory"], &mesh());
      paraview.registerField("state", fields.at("state").gridFunc());
      addLogger(std::move(paraview), {.initial_state = log_inital_state, 
                              .each_timestep = log_each_timestep, .final_state = log_final_state});
   }
}

template <int dim, bool entvar>
void FlowSolver<dim, entvar>::derivedPDEInitialHook(const Vector &state)
{
   // AbstractSolver2::initialHook(state);
   if (options["time-dis"]["steady"])
   {
      // res_norm0 is used to compute the time step in PTC
      res_norm0 = calcResidualNorm(state);
   }
   if (options["time-dis"]["entropy-log"])
   {
      double t0 = options["time-dis"]["t-initial"];  // Should be passed in!!!
      auto inputs = MISOInputs({{"time", t0}, {"state", state}});
      double entropy = calcEntropy(*spatial_res, inputs);
      if (rank == 0)
      {
         *out << "before time stepping, entropy is " << entropy << endl;
         remove("entropy-log.txt");
         entropy_log.open("entropy-log.txt", fstream::app);
         entropy_log << setprecision(16);
      }
   }
}

template <int dim, bool entvar>
void FlowSolver<dim, entvar>::derivedPDEIterationHook(int iter,
                                                      double t,
                                                      double dt,
                                                      const Vector &state)
{
   if (options["time-dis"]["entropy-log"])
   {
      auto inputs = MISOInputs({{"time", t}, {"state", state}});
      double entropy = calcEntropy(*spatial_res, inputs);
      if (rank == 0)
      {
         entropy_log << t << ' ' << entropy << endl;
      }
   }
}

template <int dim, bool entvar>
double FlowSolver<dim, entvar>::calcStepSize(int iter,
                                             double t,
                                             double t_final,
                                             double dt_old,
                                             const Vector &state) const
{
   if (options["time-dis"]["steady"])
   {
      // ramp up time step for pseudo-transient continuation
      // TODO: the l2 norm of the weak residual is probably not ideal here
      // A better choice might be the l1 norm
      double res_norm = calcResidualNorm(state);
      double exponent = options["time-dis"]["res-exp"];
      double dt = options["time-dis"]["dt"].template get<double>() *
                  pow(res_norm0 / res_norm, exponent);
      return max(dt, dt_old);
   }
   if (!options["time-dis"]["const-cfl"])
   {
      return AbstractSolver2::calcStepSize(iter, t, t_final, dt_old, state);
   }
   // Otherwise, use a constant CFL condition
   auto cfl = options["time-dis"]["cfl"];
   // here we call the FlowResidual method for the min time step, which needs
   // the current state; this is provided by the state field of PDESolver,
   // which we access with getState()
   return getConcrete<FlowResidual<dim, entvar>>(*spatial_res)
       .minCFLTimeStep(cfl, getState().gridFunc());
}

template <int dim, bool entvar>
bool FlowSolver<dim, entvar>::iterationExit(int iter,
                                            double t,
                                            double t_final,
                                            double dt,
                                            const Vector &state) const
{
   if (options["time-dis"]["steady"])
   {
      double norm = calcResidualNorm(state);
      if (norm <= options["time-dis"]["steady-abstol"])
      {
         return true;
      }
      if (norm <=
          res_norm0 *
              options["time-dis"]["steady-reltol"].template get<double>())
      {
         return true;
      }
      return false;
   }
   else
   {
      return AbstractSolver2::iterationExit(iter, t, t_final, dt, state);
   }
}

template <int dim, bool entvar>
void FlowSolver<dim, entvar>::derivedPDETerminalHook(int iter,
                                                     double t_final,
                                                     const mfem::Vector &state)
{
   if (options["time-dis"]["entropy-log"])
   {
      auto inputs = MISOInputs({{"time", t_final}, {"state", state}});
      double entropy = calcEntropy(*spatial_res, inputs);
      if (rank == 0)
      {
         entropy_log << t_final << ' ' << entropy << endl;
         entropy_log.close();
      }
   }
}

template <int dim, bool entvar>
double FlowSolver<dim, entvar>::calcConservativeVarsL2Error(
    void (*u_exact)(const mfem::Vector &, mfem::Vector &),
    int entry)
{
   return getConcrete<FlowResidual<dim, entvar>>(*spatial_res)
       .calcConservativeVarsL2Error(getState().gridFunc(), u_exact, entry);
}

template <int dim, bool entvar>
void FlowSolver<dim, entvar>::addOutput(const std::string &fun,
                                        const nlohmann::json &options)
{
   FlowResidual<dim, entvar> &flow_res =
       getConcrete<FlowResidual<dim, entvar>>(*spatial_res);
   outputs.emplace(fun, flow_res.constructOutput(fun, options));
}

// explicit instantiation
template class FlowSolver<1, true>;
template class FlowSolver<1, false>;
template class FlowSolver<2, true>;
template class FlowSolver<2, false>;
template class FlowSolver<3, true>;
template class FlowSolver<3, false>;

/*
Notes:
ode will call nonlinear_solver->Mult, which will use the residual
space_time_res. The residual will be passed references to {"state", u},
{"state_dot", du_dt}, {"dt", dt}, {"time", t} via MISOInputs before the
nonlinear solve happens.  The residual can use this information to decide if
this is an explicit or implicit solve.

Explict: solves `M du_dt + R(u, p, t) = 0` for du_dt.
Implicit: solves `M du_dt + R(u + dt * du_dt, p, t + dt) = 0` for `du_dt`

Depending on which of the above we are dealing with, the linear solver and
preconditioner may be very different.

Explicit: linear solver for SPD mass and corresponding preconditioner
Implicit: nonlinear solver for Jacobian of (time-dependent) residual

Q: who should own the mass matrix?  This solver.

Q: should the FlowResidual be responsible for both the spatial and temporal
residuals?  Note that AbstractSolver2.calcResidualNorm uses `evaluate`, so we
either have to provide ode with a different residual, or support both types of
residuals.

Maybe the short-term solution should be to use the TimeDependentOperator; if we
end up needing a nonlinear mass matrix, then we can create a more generalized
TimeDependentOperator that takes in a second residual (?).

*/

}  // namespace miso