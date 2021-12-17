
#include "diag_mass_integ.hpp"
#include "flow_solver.hpp"
#include "flow_residual.hpp"
#include "mfem_extensions.hpp"

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

namespace mach
{
FlowSolver::FlowSolver(MPI_Comm incomm,
                       const nlohmann::json &solver_options,
                       std::unique_ptr<mfem::Mesh> smesh)
 : PDESolver(incomm, solver_options, getNumFlowStates, std::move(smesh)),
   mass(&fes())
{
   // Construct space-time residual from spatial residual and mass matrix
   spatial_res = std::make_unique<mach::MachResidual>(
       FlowResidual(solver_options, fes(), diff_stack));
   const char *name = fes().FEColl()->Name();
   if ((strncmp(name, "SBP", 3) == 0) || (strncmp(name, "DSBP", 4) == 0))
   {
      mass.AddDomainIntegrator(new DiagMassIntegrator(fes().GetVDim()));
   }
   else
   {
      mass.AddDomainIntegrator(new MassIntegrator());
   }
   mass.Assemble(0);  // May want to consider AssembleDiagonal(Vector &diag)
   mass.Finalize(0);
   mass_mat.reset(mass.ParallelAssemble());
   space_time_res = std::make_unique<mach::MachResidual>(
       mach::TimeDependentResidual(*spatial_res, mass_mat.get()));

   // construct the preconditioner, linear solver, and nonlinear solver
   auto prec_solver_opts = options["lin-prec"];
   prec = constructPreconditioner(prec_solver_opts);
   auto lin_solver_opts = options["lin-solver"];
   linear_solver = constructLinearSolver(comm, lin_solver_opts, prec.get());
   auto nonlin_solver_opts = options["nonlin-solver"];
   nonlinear_solver =
       constructNonlinearSolver(comm, nonlin_solver_opts, *linear_solver);
   nonlinear_solver->SetOperator(*space_time_res);

   // construct the ODE solver (also used for pseudo-transient continuation)
   auto ode_opts = options["time-dis"];
   ode =
       make_unique<FirstOrderODE>(*space_time_res, ode_opts, *nonlinear_solver);
}

unique_ptr<Solver> FlowSolver::constructPreconditioner(
    nlohmann::json &prec_options)
{
   std::string prec_type = prec_options["type"].get<std::string>();
   unique_ptr<Solver> precond;
   if (prec_type == "hypreeuclid")
   {
      precond = std::make_unique<HypreEuclid>(comm);
      // TODO: need to add HYPRE_EuclidSetLevel to odl branch of mfem
      *out << "WARNING! Euclid fill level is hard-coded"
           << "(see AbstractSolver::constructLinearSolver() for details)"
           << endl;
      // int fill = options["lin-solver"]["filllevel"].get<int>();
      // HYPRE_EuclidSetLevel(dynamic_cast<HypreEuclid*>(precond.get())->GetPrec(),
      // fill);
   }
   else if (prec_type == "hypreilu")
   {
      precond = std::make_unique<HypreILU>();
      auto *ilu = dynamic_cast<HypreILU *>(precond.get());
      HYPRE_ILUSetType(*ilu, prec_options["ilu-type"].get<int>());
      HYPRE_ILUSetLevelOfFill(*ilu, prec_options["lev-fill"].get<int>());
      HYPRE_ILUSetLocalReordering(*ilu, prec_options["ilu-reorder"].get<int>());
      HYPRE_ILUSetPrintLevel(*ilu, prec_options["printlevel"].get<int>());
      // Just listing the options below in case we need them in the future
      // HYPRE_ILUSetSchurMaxIter(ilu, schur_max_iter);
      // HYPRE_ILUSetNSHDropThreshold(ilu, nsh_thres); needs type = 20,21
      // HYPRE_ILUSetDropThreshold(ilu, drop_thres);
      // HYPRE_ILUSetMaxNnzPerRow(ilu, nz_max);
   }
   else if (prec_type == "hypreams")
   {
      precond = std::make_unique<HypreAMS>(&(fes()));
      auto *ams = dynamic_cast<HypreAMS *>(precond.get());
      ams->SetPrintLevel(prec_options["printlevel"].get<int>());
      ams->SetSingularProblem();
   }
   else if (prec_type == "hypreboomeramg")
   {
      precond = std::make_unique<HypreBoomerAMG>();
      auto *amg = dynamic_cast<HypreBoomerAMG *>(precond.get());
      amg->SetPrintLevel(prec_options["printlevel"].get<int>());
   }
   else if (prec_type == "blockilu")
   {
      precond = std::make_unique<BlockILU>(fes().GetVDim());
   }
   else
   {
      throw MachException(
          "Unsupported preconditioner type!\n"
          "\tavilable options are: HypreEuclid, HypreILU, HypreAMS,"
          " HypreBoomerAMG.\n");
   }
   return precond;
}

void FlowSolver::derivedPDEinitialHook()
{
   // AbstractSolver2::initialHook(state);
   if (options["time-dis"]["steady"].template get<bool>())
   {
      // res_norm0 is used to compute the time step in PTC
      res_norm0 = calcResidualNorm(getState());
   }
   // TODO: this should only be output if necessary
   // double entropy = ent->GetEnergy(state);
   //*out << "before time stepping, entropy is " << entropy << endl;
   // remove("entropylog.txt");
   // entropylog.open("entropylog.txt", fstream::app);
   // entropylog << setprecision(14);
}

/*
Notes:
ode will call nonlinear_solver->Mult, which will use the residual
space_time_res. The residual will be passed references to {"state", u},
{"state_dot", du_dt}, {"dt", dt}, {"time", t} via MachInputs before the
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

}  // namespace mach