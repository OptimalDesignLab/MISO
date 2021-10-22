#include "utils.hpp"
#include "equation_solver.hpp"

namespace mach
{
EquationSolver::EquationSolver(
    MPI_Comm comm,
    const nlohmann::json &lin_options,
    std::unique_ptr<mfem::Solver> prec,
    const std::optional<nlohmann::json> &nonlin_options)
{
   prec_ = std::move(prec);
   lin_solver_ = constructLinearSolver(comm, lin_options);
   if (nonlin_options)
   {
      nonlin_solver_ = constructNonlinearSolver(comm, *nonlin_options);
   }
}

// std::unique_ptr<mfem::Solver> EquationSolver::constructPreconditioner(
//     MPI_Comm comm,
//     const nlohmann::json &prec_options)
// {
//    std::string prec_type = prec_options["type"].get<std::string>();
//    if (prec_type == "hypreeuclid")
//    {
//       return std::make_unique<mfem::HypreEuclid>(comm);
//       // TODO: need to add HYPRE_EuclidSetLevel to odl branch of mfem
//       // *out << "WARNING! Euclid fill level is hard-coded"
//       //      << "(see AbstractSolver::constructLinearSolver() for details)"
//       //      << endl;
//       // int fill = options["lin-solver"]["filllevel"].get<int>();
//       //
//       HYPRE_EuclidSetLevel(dynamic_cast<HypreEuclid*>(precond.get())->GetPrec(),
//       // fill);
//    }
//    else if (prec_type == "hypreilu")
//    {
//       auto ilu = std::make_unique<mfem::HypreILU>();
//       HYPRE_ILUSetType(*ilu, prec_options["ilu-type"].get<int>());
//       HYPRE_ILUSetLevelOfFill(*ilu, prec_options["lev-fill"].get<int>());
//       HYPRE_ILUSetLocalReordering(*ilu,
//       prec_options["ilu-reorder"].get<int>()); HYPRE_ILUSetPrintLevel(*ilu,
//       prec_options["printlevel"].get<int>());
//       // Listing the full options below in case we need them in the future
//       // HYPRE_ILUSetSchurMaxIter(ilu, schur_max_iter);
//       // HYPRE_ILUSetNSHDropThreshold(ilu, nsh_thres); needs type = 20,21
//       // HYPRE_ILUSetDropThreshold(ilu, drop_thres);
//       // HYPRE_ILUSetMaxNnzPerRow(ilu, nz_max);
//       return ilu;
//    }
//    else if (prec_type == "hypreams")
//    {
//       // precond.reset(new HypreAMS(fes.get()));
//       // auto *ams = dynamic_cast<HypreAMS *>(precond.get());
//       // ams->SetPrintLevel(prec_options["printlevel"].get<int>());
//       // ams->SetSingularProblem();
//    }
//    else if (prec_type == "hypreboomeramg")
//    {
//       auto amg = std::make_unique<mfem::HypreBoomerAMG>();
//       amg->SetPrintLevel(prec_options["printlevel"].get<int>());
//       return amg;
//    }
//    else if (prec_type == "blockilu")
//    {
//       auto block_size = prec_options["blocksize"].get<int>();
//       return std::make_unique<mfem::BlockILU>();
//    }
//    else
//    {
//       throw MachException(
//           "Unsupported preconditioner type!\n"
//           "\tavilable options are: HypreEuclid, HypreILU, HypreAMS,"
//           " HypreBoomerAMG.\n");
//    }
// }

std::unique_ptr<mfem::Solver> EquationSolver::constructLinearSolver(
    MPI_Comm comm,
    const nlohmann::json &lin_options)
{
   std::string solver_type = lin_options["type"].get<std::string>();
   auto reltol = lin_options["reltol"].get<double>();
   auto abstol = lin_options.value("reltol", 0.0);
   int maxiter = lin_options["maxiter"].get<int>();
   int ptl = lin_options["printlevel"].get<int>();
   int kdim = lin_options.value("kdim", -1);

   if (solver_type == "hypregmres")
   {
      auto gmres = std::make_unique<mfem::HypreGMRES>(comm);
      gmres->SetTol(reltol);
      gmres->SetAbsTol(abstol);
      gmres->SetMaxIter(maxiter);
      gmres->SetPrintLevel(ptl);
      if (kdim != -1)
      {
         gmres->SetKDim(kdim);  // set GMRES subspace size
      }
      // gmres->SetPreconditioner(dynamic_cast<mfem::HypreSolver &>(*prec_));
      return gmres;
   }
   else if (solver_type == "gmres")
   {
      auto gmres = std::make_unique<mfem::GMRESSolver>(comm);
      gmres->SetRelTol(reltol);
      gmres->SetAbsTol(abstol);
      gmres->SetMaxIter(maxiter);
      gmres->SetPrintLevel(ptl);
      if (kdim != -1)
      {
         gmres->SetKDim(kdim);  // set GMRES subspace size
      }
      // gmres->SetPreconditioner(*prec_);
      return gmres;
   }
   else if (solver_type == "hyprefgmres")
   {
      auto fgmres = std::make_unique<mfem::HypreFGMRES>(comm);
      fgmres->SetTol(reltol);
      fgmres->SetMaxIter(maxiter);
      fgmres->SetPrintLevel(ptl);
      if (kdim != -1)
      {
         fgmres->SetKDim(kdim);  // set FGMRES subspace size
      }
      // fgmres->SetPreconditioner(dynamic_cast<mfem::HypreSolver &>(*prec_));
      return fgmres;
   }
   else if (solver_type == "fgmres")
   {
      auto fgmres = std::make_unique<mfem::FGMRESSolver>(comm);
      fgmres->SetRelTol(reltol);
      fgmres->SetAbsTol(abstol);
      fgmres->SetMaxIter(maxiter);
      fgmres->SetPrintLevel(ptl);
      if (kdim != -1)
      {
         fgmres->SetKDim(kdim);  // set FGMRES subspace size
      }
      // fgmres->SetPreconditioner(*prec_);
      return fgmres;
   }
   else if (solver_type == "hyprepcg")
   {
      auto pcg = std::make_unique<mfem::HyprePCG>(comm);
      pcg->SetTol(reltol);
      pcg->SetAbsTol(abstol);
      pcg->SetMaxIter(maxiter);
      pcg->SetPrintLevel(ptl);
      // pcg->SetPreconditioner(dynamic_cast<mfem::HypreSolver &>(*prec_));
      return pcg;
   }
   else if (solver_type == "pcg")
   {
      auto pcg = std::make_unique<mfem::CGSolver>(comm);
      pcg->SetRelTol(reltol);
      pcg->SetAbsTol(abstol);
      pcg->SetMaxIter(maxiter);
      pcg->SetPrintLevel(ptl);
      // pcg->SetPreconditioner(*prec_);
      return pcg;
   }
   else if (solver_type == "minres")
   {
      auto minres = std::make_unique<mfem::MINRESSolver>(comm);
      minres->SetRelTol(reltol);
      minres->SetAbsTol(abstol);
      minres->SetMaxIter(maxiter);
      minres->SetPrintLevel(ptl);
      // minres->SetPreconditioner(*prec_);
      return minres;
   }
   else
   {
      throw MachException(
          "Unsupported iterative solver type!\n"
          "\tavilable options are: hypregmres, gmres, hyprefgmres, fgmres,\n"
          "\thyprepcg, pcg, minres");
   }
}

std::unique_ptr<mfem::NewtonSolver> EquationSolver::constructNonlinearSolver(
    MPI_Comm comm,
    const nlohmann::json &nonlin_options)
{
   auto solver_type = nonlin_options["type"].get<std::string>();
   auto abstol = nonlin_options["abstol"].get<double>();
   auto reltol = nonlin_options["reltol"].get<double>();
   auto maxiter = nonlin_options["maxiter"].get<int>();
   auto ptl = nonlin_options["printlevel"].get<int>();

   std::unique_ptr<mfem::NewtonSolver> nonlin_solver;
   if (solver_type == "newton")
   {
      nonlin_solver = std::make_unique<mfem::NewtonSolver>(comm);
   }
   else if (solver_type == "inexactnewton")
   {
      nonlin_solver = std::make_unique<mfem::NewtonSolver>(comm);

      /// use defaults from SetAdaptiveLinRtol unless specified
      int type = nonlin_options.value("inexacttype", 2);
      double rtol0 = nonlin_options.value("rtol0", 0.5);
      double rtol_max = nonlin_options.value("rtol_max", 0.9);
      double alpha =
          nonlin_options.value("alpha", (0.5) * ((1.0) + sqrt((5.0))));
      double gamma = nonlin_options.value("gamma", 1.0);
      nonlin_solver->SetAdaptiveLinRtol(type, rtol0, rtol_max, alpha, gamma);
   }
   else
   {
      throw MachException(
          "Unsupported nonlinear solver type!\n"
          "\tavilable options are: newton, inexactnewton\n");
   }

   nonlin_solver->iterative_mode = true;
   nonlin_solver->SetPrintLevel(ptl);
   nonlin_solver->SetRelTol(reltol);
   nonlin_solver->SetAbsTol(abstol);
   nonlin_solver->SetMaxIter(maxiter);

   return nonlin_solver;
}

void EquationSolver::SetOperator(const mfem::Operator &op)
{
   if (nonlin_solver_)
   {
      nonlin_solver_->SetOperator(op);

      // Now that the nonlinear solver knows about the operator, we can set its
      // linear solver
      if (!nonlin_solver_set_solver_called_)
      {
         nonlin_solver_->SetSolver(LinearSolver());
         nonlin_solver_set_solver_called_ = true;
      }
   }
   else
   {
      lin_solver_->SetOperator(op);
   }
   height = op.Height();
   width = op.Width();
}

void EquationSolver::Mult(const mfem::Vector &b, mfem::Vector &x) const
{
   if (nonlin_solver_)
   {
      nonlin_solver_->Mult(b, x);
   }
   else
   {
      lin_solver_->Mult(b, x);
   }
}

}  // namespace mach
