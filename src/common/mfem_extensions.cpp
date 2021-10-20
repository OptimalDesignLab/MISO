#include <iostream>

#include "mfem.hpp"

#include "evolver.hpp"
#include "utils.hpp"
#include "mfem_extensions.hpp"

using namespace mfem;
using namespace std;
using namespace mach;

namespace mach
{
void PseudoTransientSolver::Init(TimeDependentOperator &_f)
{
   ODESolver::Init(_f);
}

void PseudoTransientSolver::Step(Vector &x, double &t, double &dt)
{
   f->SetTime(t + dt);
   k.SetSize(x.Size(), mem_type);
   f->ImplicitSolve(dt, x, k);
   x.Add(dt, k);
   t += dt;
}

void RRKImplicitMidpointSolver::Init(TimeDependentOperator &_f)
{
   ODESolver::Init(_f);
}

void RRKImplicitMidpointSolver::Step(Vector &x, double &t, double &dt)
{
   auto *f_ode = dynamic_cast<EntropyConstrainedOperator *>(f);
   f_ode->SetTime(t + dt / 2);
   k.SetSize(x.Size(), mem_type);
   f_ode->ImplicitSolve(dt, dt / 2, x, k);

   // Set-up and solve the scalar nonlinear problem for the relaxation gamma
   // cout << "x size is " << x.Size() << '\n';
   // cout << "x is empty? == " << x.GetMemory().Empty() << '\n';
   double delta_entropy = f_ode->EntropyChange(dt / 2, x, k);
   // double delta_entropy = f_ode->EntropyChange(dt, x, k);
   *out << "delta_entropy is " << delta_entropy << '\n';
   double entropy_old = f_ode->Entropy(x);
   *out << "old entropy is " << entropy_old << '\n';
   mfem::Vector x_new(x.Size());
   // cout << "x_new size is " << x_new.Size() << '\n';
   auto entropyFun = [&](double gamma)
   {
      *out << "In lambda function: " << std::setprecision(14);
      add(x, gamma * dt, k, x_new);
      double entropy = f_ode->Entropy(x_new);
      *out << "gamma = " << gamma << ": ";
      *out << "residual = "
           << entropy - entropy_old + gamma * dt * delta_entropy << endl;
      // cout << "new entropy is " << entropy << '\n';
      return entropy - entropy_old + gamma * dt * delta_entropy;
   };
   // TODO: tolerances and maxiter should be provided in some other way
   const double ftol = 1e-12;
   const double xtol = 1e-12;
   const int maxiter = 30;
   // double gamma = bisection(entropyFun, 0.50, 1.5, ftol, xtol, maxiter);
   double gamma = secant(entropyFun, 0.99, 1.01, ftol, xtol, maxiter);
   *out << "\tgamma = " << gamma << endl;
   x.Add(gamma * dt, k);
   t += gamma * dt;
}

unique_ptr<Solver> constructPreconditioner(nlohmann::json &options,
                                           MPI_Comm comm)
{
   std::string prec_type = options["type"].get<std::string>();
   unique_ptr<Solver> precond;
   if (prec_type == "hypreeuclid")
   {
      precond.reset(new HypreEuclid(comm));
      // TODO: need to add HYPRE_EuclidSetLevel to odl branch of mfem
      std::cout << "WARNING! Euclid fill level is hard-coded"
           << "(see AbstractSolver::constructLinearSolver() for details)"
           << endl;
      // int fill = options["lin-solver"]["filllevel"].get<int>();
      // HYPRE_EuclidSetLevel(dynamic_cast<HypreEuclid*>(precond.get())->GetPrec(),
      // fill);
   }
   else if (prec_type == "hypreilu")
   {
      precond.reset(new HypreILU());
      auto *ilu = dynamic_cast<HypreILU *>(precond.get());
      HYPRE_ILUSetType(*ilu, options["ilu-type"].get<int>());
      HYPRE_ILUSetLevelOfFill(*ilu, options["lev-fill"].get<int>());
      HYPRE_ILUSetLocalReordering(*ilu, options["ilu-reorder"].get<int>());
      HYPRE_ILUSetPrintLevel(*ilu, options["printlevel"].get<int>());
      // cout << "Just after Hypre options" << endl;
      // Just listing the options below in case we need them in the future
      // HYPRE_ILUSetSchurMaxIter(ilu, schur_max_iter);
      // HYPRE_ILUSetNSHDropThreshold(ilu, nsh_thres); needs type = 20,21
      // HYPRE_ILUSetDropThreshold(ilu, drop_thres);
      // HYPRE_ILUSetMaxNnzPerRow(ilu, nz_max);
   }
   else if (prec_type == "hypreams")
   {
      precond.reset(new HypreAMS(fes.get()));
      auto *ams = dynamic_cast<HypreAMS *>(precond.get());
      ams->SetPrintLevel(options["printlevel"].get<int>());
      ams->SetSingularProblem();
   }
   else if (prec_type == "hypreboomeramg")
   {
      precond.reset(new HypreBoomerAMG());
      auto *amg = dynamic_cast<HypreBoomerAMG *>(precond.get());
      amg->SetPrintLevel(options["printlevel"].get<int>());
   }
   else if (prec_type == "blockilu")
   {
      precond.reset(new BlockILU(getNumState()));
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


unique_ptr<Solver> constructLinearSolver(nlohmann::json &options,
                                         mfem::Solver &_prec,
                                         MPI_Comm comm)
{
   std::string solver_type = options["type"].get<std::string>();
   auto reltol = options["reltol"].get<double>();
   int maxiter = options["maxiter"].get<int>();
   int ptl = options["printlevel"].get<int>();
   int kdim = options.value("kdim", -1);

   unique_ptr<Solver> lin_solver;
   if (solver_type == "hypregmres")
   {
      lin_solver.reset(new HypreGMRES(comm));
      auto *gmres = dynamic_cast<HypreGMRES *>(lin_solver.get());
      gmres->SetTol(reltol);
      gmres->SetMaxIter(maxiter);
      gmres->SetPrintLevel(ptl);
      gmres->SetPreconditioner(dynamic_cast<HypreSolver &>(_prec));
      if (kdim != -1)
      {
         gmres->SetKDim(kdim);  // set GMRES subspace size
      }
   }
   else if (solver_type == "gmres")
   {
      lin_solver.reset(new GMRESSolver(comm));
      auto *gmres = dynamic_cast<GMRESSolver *>(lin_solver.get());
      gmres->SetRelTol(reltol);
      gmres->SetMaxIter(maxiter);
      gmres->SetPrintLevel(ptl);
      gmres->SetPreconditioner(dynamic_cast<Solver &>(_prec));
      if (kdim != -1)
      {
         gmres->SetKDim(kdim);  // set GMRES subspace size
      }
   }
   else if (solver_type == "hyprefgmres")
   {
      lin_solver.reset(new HypreFGMRES(comm));
      auto *fgmres = dynamic_cast<HypreFGMRES *>(lin_solver.get());
      fgmres->SetTol(reltol);
      fgmres->SetMaxIter(maxiter);
      fgmres->SetPrintLevel(ptl);
      fgmres->SetPreconditioner(dynamic_cast<HypreSolver &>(_prec));
      if (kdim != -1)
      {
         fgmres->SetKDim(kdim);  // set FGMRES subspace size
      }
   }
   else if (solver_type == "hyprepcg")
   {
      lin_solver.reset(new HyprePCG(comm));
      auto *pcg = dynamic_cast<HyprePCG *>(lin_solver.get());
      pcg->SetTol(reltol);
      pcg->SetMaxIter(maxiter);
      pcg->SetPrintLevel(ptl);
      pcg->SetPreconditioner(dynamic_cast<HypreSolver &>(_prec));
   }
   else if (solver_type == "pcg")
   {
      lin_solver.reset(new CGSolver(comm));
      auto *pcg = dynamic_cast<CGSolver *>(lin_solver.get());
      pcg->SetRelTol(reltol);
      pcg->SetMaxIter(maxiter);
      pcg->SetPrintLevel(ptl);
      pcg->SetPreconditioner(dynamic_cast<Solver &>(_prec));
   }
   else if (solver_type == "minres")
   {
      lin_solver.reset(new MINRESSolver(comm));
      auto *minres = dynamic_cast<MINRESSolver *>(lin_solver.get());
      minres->SetRelTol(reltol);
      minres->SetMaxIter(maxiter);
      minres->SetPrintLevel(ptl);
      minres->SetPreconditioner(dynamic_cast<Solver &>(_prec));
   }
   else
   {
      throw MachException(
          "Unsupported iterative solver type!\n"
          "\tavilable options are: hypregmres, gmres, hyprefgmres,\n"
          "\thyprepcg, pcg, minres");
   }
   return lin_solver;
}

unique_ptr<NewtonSolver> constructNonlinearSolver(nlohmann::json &options,
                                                  mfem::Solver &_lin_solver,
                                                  MPI_Comm comm)
{
   std::string solver_type = options["type"].get<std::string>();
   auto abstol = options["abstol"].get<double>();
   auto reltol = options["reltol"].get<double>();
   int maxiter = options["maxiter"].get<int>();
   int ptl = options["printlevel"].get<int>();
   unique_ptr<NewtonSolver> nonlin_solver;
   if (solver_type == "newton")
   {
      nonlin_solver.reset(new NewtonSolver(comm));
   }
   else if (solver_type == "inexactnewton")
   {
      nonlin_solver.reset(new NewtonSolver(comm));
      auto *newton = dynamic_cast<NewtonSolver *>(nonlin_solver.get());

      /// use defaults from SetAdaptiveLinRtol unless specified
      int type = options.value("inexacttype", 2);
      double rtol0 = options.value("rtol0", 0.5);
      double rtol_max = options.value("rtol_max", 0.9);
      double alpha = options.value("alpha", (0.5) * ((1.0) + sqrt((5.0))));
      double gamma = options.value("gamma", 1.0);
      newton->SetAdaptiveLinRtol(type, rtol0, rtol_max, alpha, gamma);
   }
   else
   {
      throw MachException(
          "Unsupported nonlinear solver type!\n"
          "\tavilable options are: newton, inexactnewton\n");
   }

   nonlin_solver->iterative_mode = true;
   nonlin_solver->SetSolver(dynamic_cast<Solver &>(_lin_solver));
   nonlin_solver->SetPrintLevel(ptl);
   nonlin_solver->SetRelTol(reltol);
   nonlin_solver->SetAbsTol(abstol);
   nonlin_solver->SetMaxIter(maxiter);

   return nonlin_solver;
}

unique_ptr<ODESolver> constructODESolver(nlohmann::json &options)
{
   std::string ode_solver_type = options["type"].get<std::string>();
   unique_ptr<ODESolver> ode_solver;
   if (ode_solver_type == "RK1")
   {
      ode_solver.reset(new ForwardEulerSolver);
   }
   else if (ode_solver_type == "RK4")
   {
      ode_solver.reset(new RK4Solver);
   }
   else if (ode_solver_type == "MIDPOINT")
   {
      ode_solver.reset(new ImplicitMidpointSolver);
   }
   else if (ode_solver_type == "RRK")
   {
      ode_solver.reset(new RRKImplicitMidpointSolver());
   }
   else if (ode_solver_type == "PTC")
   {
      ode_solver.reset(new PseudoTransientSolver());
   }
   else
   {
      throw MachException("Unknown ODE solver type " + ode_solver_type);
   }
   return ode_solver;
}

}  // namespace mach
