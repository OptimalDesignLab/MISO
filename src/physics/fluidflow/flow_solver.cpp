
#include "diag_mass_integ.hpp"
#include "flow_solver.hpp"
#include "flow_residual.hpp"
#include "mfem_extensions.hpp"
#include "euler_fluxes.hpp"

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

/// Return the appropriate function for computing the spectral radius
/// \param[in] dim - dimension of the problem
/// \param[in] entvar - if true, the states are the entropy variables
std::function<double(const double *, const double *)> getSpectralRadiusFunction(
    int dim,
    bool entvar)
{
   switch (dim)
   {
   case 1:
      return (entvar ? mach::calcSpectralRadius<double, 1, true>
                     : mach::calcSpectralRadius<double, 1>);
   case 2:
      return (entvar ? mach::calcSpectralRadius<double, 2, true>
                     : mach::calcSpectralRadius<double, 2>);
   case 3:
      return (entvar ? mach::calcSpectralRadius<double, 3, true>
                     : mach::calcSpectralRadius<double, 3>);
   default:
      throw mach::MachException("Invalid space dimension!\n");
   }
}

/// Return the node error for either conservative or entropy-based state
/// \param[in] discrete - the numerical solution at the node
/// \param[in] exact - the exact solution at the node
/// \param[in] entry - the entry to compute the error for
/// \returns the error in the conservative variables
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \tparam entvar - if true, the entropy variables are used in the integrators
template <int dim, bool entvar = false>
double getNodeError(const mfem::Vector &discrete,
                    const mfem::Vector &exact,
                    int entry)
{
   mfem::Vector qdiscrete(dim + 2);
   mfem::Vector qexact(dim + 2);
   if (entvar)
   {
      mach::calcConservativeVars<double, dim>(discrete.GetData(),
                                              qdiscrete.GetData());
      mach::calcConservativeVars<double, dim>(exact.GetData(),
                                              qexact.GetData());
   }
   else
   {
      qdiscrete = discrete;
      qexact = exact;
   }
   double err = 0.0;
   if (entry < 0)
   {
      for (int i = 0; i < dim + 2; ++i)
      {
         double dq = qdiscrete(i) - qexact(i);
         err += dq * dq;
      }
   }
   else
   {
      err = qdiscrete(entry) - qexact(entry);
      err = err * err;
   }
   return err;
}

/// Return a function to compute the node error
/// \param[in] dim - dimension of the problem
/// \param[in] entvar - if true, convert to entropy vars before computing error
/// \returns function to compute the error in the conservative variables
std::function<double(const mfem::Vector &, const mfem::Vector &, int)>
getNodeErrorFunction(int dim, bool entvar)
{
   switch (dim)
   {
   case 1:
      return (entvar ? getNodeError<1, true> : getNodeError<1>);
   case 2:
      return (entvar ? getNodeError<2, true> : getNodeError<2>);
   case 3:
      return (entvar ? getNodeError<3, true> : getNodeError<3>);
   default:
      throw mach::MachException("Invalid space dimension!\n");
   }
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

void FlowSolver::derivedPDEInitialHook()
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

double FlowSolver::calcStepSize(int iter, double t, double t_final,
                                double dt_old, const Vector &state) const
{
   if (options["time-dis"]["steady"].template get<bool>())
   {
      // ramp up time step for pseudo-transient continuation
      // TODO: the l2 norm of the weak residual is probably not ideal here
      // A better choice might be the l1 norm
      double res_norm = calcResidualNorm(state);
      double exponent = options["time-dis"]["res-exp"];
      double dt = options["time-dis"]["dt"].get<double>() *
                  pow(res_norm0 / res_norm, exponent);
      return max(dt, dt_old);
   }
   if (!options["time-dis"]["const-cfl"].get<bool>())
   {
      return AbstractSolver2::calcStepSize(iter, t, t_final, dt_old, state);
   }
   // Otherwise, use a constant CFL condition
   auto cfl = options["time-dis"]["cfl"].get<double>();
   return calcCFLTimeStep(cfl);
}

double FlowSolver::calcCFLTimeStep(double cfl) const
{
   int dim = mesh_->SpaceDimension();
   bool entvar = options["flow-param"].value("entropy-state", false);
   auto calcSpec = getSpectralRadiusFunction(dim, entvar);
   double dt_local = 1e100;
   Vector xi(dim);
   Vector dxij(dim);
   Vector ui;
   Vector dxidx;
   DenseMatrix uk;
   DenseMatrix adjJt(dim);
   const mfem::ParGridFunction &state_gf = getState().gridFunc();
   for (int k = 0; k < fes().GetNE(); k++)
   {
      // get the element, its transformation, and the state values on element
      const FiniteElement *fe = fes().GetFE(k);
      const IntegrationRule *ir = &(fe->GetNodes());
      ElementTransformation *trans = fes().GetElementTransformation(k);
      state_gf.GetVectorValues(*trans, *ir, uk);
      for (int i = 0; i < fe->GetDof(); ++i)
      {
         trans->SetIntPoint(&fe->GetNodes().IntPoint(i));
         trans->Transform(fe->GetNodes().IntPoint(i), xi);
         CalcAdjugateTranspose(trans->Jacobian(), adjJt);
         uk.GetColumnReference(i, ui);
         for (int j = 0; j < fe->GetDof(); ++j)
         {
            if (j == i)
            {
               continue;
            }
            trans->Transform(fe->GetNodes().IntPoint(j), dxij);
            dxij -= xi;
            double dx = dxij.Norml2();
            dt_local =
                min(dt_local,
                    cfl * dx * dx /
                        calcSpec(dxij, ui));  // extra dx is to normalize dxij
         }
      }
   }
   double dt_min = NAN;
   MPI_Allreduce(&dt_local, &dt_min, 1, MPI_DOUBLE, MPI_MIN, comm);
   return dt_min;
}

bool FlowSolver::iterationExit(int iter,
                               double t,
                               double t_final,
                               double dt,
                               const mfem::Vector &state) const
{
   if (options["time-dis"]["steady"].get<bool>())
   {
      double norm = calcResidualNorm(state);
      if (norm <= options["time-dis"]["steady-abstol"].get<double>())
      {
         return true;
      }
      if (norm <=
          res_norm0 * options["time-dis"]["steady-reltol"].get<double>())
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

double FlowSolver::calcConservativeVarsL2Error(
    void (*u_exact)(const mfem::Vector &, mfem::Vector &),
    int entry)
{
   int dim = mesh_->SpaceDimension();
   bool entvar = options["flow-param"].value("entropy-state", false);
   // Following function, defined earlier in file, computes the error at a node
   // Beware: this is not particularly efficient, and **NOT thread safe!**
   auto node_error = getNodeErrorFunction(dim, entvar);
   VectorFunctionCoefficient exsol(dim + 2, u_exact);
   DenseMatrix vals;
   DenseMatrix exact_vals;
   Vector u_j;
   Vector exsol_j;
   double loc_norm = 0.0;
   const mfem::ParGridFunction &state_gf = getState().gridFunc();
   for (int i = 0; i < fes().GetNE(); i++)
   {
      const FiniteElement *fe = fes().GetFE(i);
      const IntegrationRule *ir = &(fe->GetNodes());
      ElementTransformation *T = fes().GetElementTransformation(i);
      state_gf.GetVectorValues(*T, *ir, vals);
      exsol.Eval(exact_vals, *T, *ir);
      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         T->SetIntPoint(&ip);
         vals.GetColumnReference(j, u_j);
         exact_vals.GetColumnReference(j, exsol_j);
         loc_norm += ip.weight * T->Weight() * node_error(u_j, exsol_j, entry);
      }
   }
   double norm = NAN;
   MPI_Allreduce(&loc_norm, &norm, 1, MPI_DOUBLE, MPI_SUM, comm);
   if (norm < 0.0)  // This was copied from mfem...should not happen for us
   {
      return -sqrt(-norm);
   }
   return sqrt(norm);
}

void FlowSolver::addOutput(const std::string &fun,
                           const nlohmann::json &options)
{
   int dim = mesh_->SpaceDimension();
   double mach_fs = getConcrete<FlowResidual>(*spatial_res).getMach();
   double aoa_fs = getConcrete<FlowResidual>(*spatial_res).getAoA();
   int iroll = getConcrete<FlowResidual>(*spatial_res).getIRoll();
   int ipitch = getConcrete<FlowResidual>(*spatial_res).getIPitch();
   if (fun == "drag")
   {
      // drag on the specified boundaries
      auto bdrs = options["boundaries"].get<vector<int>>();
      Vector drag_dir(dim);
      drag_dir = 0.0;
      if (dim == 1)
      {
         drag_dir(0) = 1.0;
      }
      else
      {
         drag_dir(iroll) = cos(aoa_fs);
         drag_dir(ipitch) = sin(aoa_fs);
      }
      drag_dir *= 1.0 / pow(mach_fs, 2.0);  // to get non-dimensional Cd
      FunctionalOutput out(fes(), res_fields);
      out.addOutputBdrFaceIntegrator(
          new PressureForce<dim, entvar>(diff_stack, fec().get(), drag_dir),
          std::move(bdrs));
      outputs.emplace(fun, std::move(out));
   }
   else if (fun == "lift")
   {
      // lift on the specified boundaries
      auto bdrs = options["boundaries"].get<vector<int>>();
      Vector lift_dir(dim);
      lift_dir = 0.0;
      if (dim == 1)
      {
         lift_dir(0) = 0.0;
      }
      else
      {
         lift_dir(iroll) = -sin(aoa_fs);
         lift_dir(ipitch) = cos(aoa_fs);
      }
      lift_dir *= 1.0 / pow(mach_fs, 2.0);  // to get non-dimensional Cl

      FunctionalOutput out(fes(), res_fields);
      out.addOutputBdrFaceIntegrator(
          new PressureForce<dim, entvar>(diff_stack, fec.get(), lift_dir),
          std::move(bdrs));
      outputs.emplace(fun, std::move(out));
   }
   else
   {
      throw MachException("Output with name " + fun +
                          " not supported by "
                          "FlowSolver!\n");
   }
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