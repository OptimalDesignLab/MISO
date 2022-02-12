#include <iostream>

#include "mfem.hpp"

#include "evolver.hpp"
#include "utils.hpp"
#include "matrix_operators.hpp"
#include "mfem_extensions.hpp"

using namespace mfem;

namespace mach
{
void SteadyODESolver::Step(Vector &x, double &t, double &dt)
{
   f->SetTime(t + dt);
   k.SetSize(x.Size(), mem_type);
   k = x;
   f->ImplicitSolve(dt, x, k);
   x = k;
   t += dt;
}

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
   f_ode->ImplicitSolve(dt / 2, x, k);

   // Set-up and solve the scalar nonlinear problem for the relaxation gamma
   // cout << "x size is " << x.Size() << '\n';
   // cout << "x is empty? == " << x.GetMemory().Empty() << '\n';
   double delta_entropy = f_ode->EntropyChange(dt / 2, x, k);
   // double delta_entropy = f_ode->EntropyChange(dt, x, k);
   if (out != nullptr)
   {
      *out << "delta_entropy is " << delta_entropy << '\n';
   }
   double entropy_old = f_ode->Entropy(x);
   if (out != nullptr)
   {
      *out << "old entropy is " << entropy_old << '\n';
   }
   mfem::Vector x_new(x.Size());
   // cout << "x_new size is " << x_new.Size() << '\n';
   auto entropyFun = [&](double gamma)
   {
      if (out != nullptr)
      {
         *out << "In lambda function: " << std::setprecision(14);
      }
      add(x, gamma * dt, k, x_new);
      double entropy = f_ode->Entropy(x_new);
      if (out != nullptr)
      {
         *out << "gamma = " << gamma << ": ";
         *out << "residual = "
              << entropy - entropy_old + gamma * dt * delta_entropy
              << std::endl;
      }
      // cout << "new entropy is " << entropy << '\n';
      return entropy - entropy_old + gamma * dt * delta_entropy;
   };
   // TODO: tolerances and maxiter should be provided in some other way
   const double ftol = 1e-12;
   const double xtol = 1e-12;
   const int maxiter = 30;
   // double gamma = bisection(entropyFun, 0.50, 1.5, ftol, xtol, maxiter);
   double gamma = secant(entropyFun, 0.99, 1.01, ftol, xtol, maxiter);
   if (out != nullptr)
   {
      *out << "\tgamma = " << gamma << std::endl;
   }
   x.Add(gamma * dt, k);
   t += gamma * dt;
}

BlockJacobiPreconditioner::BlockJacobiPreconditioner(const Array<int> &offsets_)
 : Solver(offsets_.Last()),
   owns_blocks(0),
   nBlocks(offsets_.Size() - 1),
   offsets(0),
   op(nBlocks)
{
   op = static_cast<Solver *>(NULL);
   offsets.MakeRef(offsets_);
}

void BlockJacobiPreconditioner::SetDiagonalBlock(int iblock, Solver *opt)
{
   // Cannot check for consistency here, since some preconditioners do not
   // provide Width() and Height()
   // MFEM_VERIFY(offsets[iblock+1] - offsets[iblock] == opt->Height() &&
   //            offsets[iblock+1] - offsets[iblock] == opt->Width(),
   //            "incompatible Operator dimensions");
   if (owns_blocks && op[iblock])
   {
      delete op[iblock];
   }
   op[iblock] = opt;
}

void BlockJacobiPreconditioner::SetOperator(const Operator &input_op)
{
   auto block_op = dynamic_cast<const BlockOperator *>(&input_op);
   if (block_op != nullptr)
   {
      // input_op is a BlockOperator
      for (int i = 0; i < nBlocks; ++i)
      {
         if (op[i])
         {
            op[i]->SetOperator(block_op->GetBlock(i, i));
         }
      }
      return;
   }
   auto jacfree_op = dynamic_cast<const JacobianFree *>(&input_op);
   if (jacfree_op != nullptr)
   {
      // jacfree_op->print("jac-free-matrix.dat");
      // throw(-1);

      // input op is a JacobianFree operator
      for (int i = 0; i < nBlocks; ++i)
      {
         if (op[i])
         {
            op[i]->SetOperator(jacfree_op->getDiagonalBlock(i));
         }
      }
      return;
   }
   // if we get here, input_op was neither a BlockOperator nor a JacobianFree
   throw MachException(
       "BlockJacobiPreconditioner::SetOperator:\n"
       "input operator must be castable to"
       "mfem::BlockOperator or JacobianFree!\n");
}

void BlockJacobiPreconditioner::Mult(const Vector &x, Vector &y) const
{
   MFEM_ASSERT(x.Size() == width, "incorrect input Vector size");
   MFEM_ASSERT(y.Size() == height, "incorrect output Vector size");

   x.Read();
   y.Write();
   y = 0.0;

   xblock.Update(const_cast<Vector &>(x), offsets);
   yblock.Update(y, offsets);

   for (int i = 0; i < nBlocks; ++i)
   {
      if (op[i])
      {
         op[i]->Mult(xblock.GetBlock(i), yblock.GetBlock(i));
      }
      else
      {
         yblock.GetBlock(i) = xblock.GetBlock(i);
      }
   }

   for (int i = 0; i < nBlocks; ++i)
   {
      yblock.GetBlock(i).SyncAliasMemory(y);
   }
}

// Action of the transpose operator
void BlockJacobiPreconditioner::MultTranspose(const Vector &x, Vector &y) const
{
   MFEM_ASSERT(x.Size() == height, "incorrect input Vector size");
   MFEM_ASSERT(y.Size() == width, "incorrect output Vector size");

   x.Read();
   y.Write();
   y = 0.0;

   xblock.Update(const_cast<Vector &>(x), offsets);
   yblock.Update(y, offsets);

   for (int i = 0; i < nBlocks; ++i)
   {
      if (op[i])
      {
         (op[i])->MultTranspose(xblock.GetBlock(i), yblock.GetBlock(i));
      }
      else
      {
         yblock.GetBlock(i) = xblock.GetBlock(i);
      }
   }

   for (int i = 0; i < nBlocks; ++i)
   {
      yblock.GetBlock(i).SyncAliasMemory(y);
   }
}

BlockJacobiPreconditioner::~BlockJacobiPreconditioner()
{
   if (owns_blocks)
   {
      for (int i = 0; i < nBlocks; ++i)
      {
         delete op[i];
      }
   }
}

std::unique_ptr<mfem::Solver> constructLinearSolver(
    MPI_Comm comm,
    const nlohmann::json &lin_options,
    mfem::Solver *prec)
{
   std::string solver_type = lin_options["type"].get<std::string>();
   auto reltol = lin_options["reltol"].get<double>();
   auto abstol = lin_options.value("abstol", 0.0);
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
      if (prec != nullptr)
      {
         gmres->SetPreconditioner(dynamic_cast<mfem::HypreSolver &>(*prec));
      }
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
      if (prec != nullptr)
      {
         gmres->SetPreconditioner(*prec);
      }
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
      if (prec != nullptr)
      {
         fgmres->SetPreconditioner(dynamic_cast<mfem::HypreSolver &>(*prec));
      }
      return fgmres;
   }
   else if (solver_type == "fgmres")
   {
      auto fgmres = std::make_unique<mfem::FGMRESSolver>(comm);
      fgmres->SetRelTol(reltol);
      fgmres->SetAbsTol(abstol);
      fgmres->SetMaxIter(maxiter);
      if (ptl > 0)
      {
         fgmres->SetPrintLevel(IterativeSolver::PrintLevel().Iterations());
      }
      if (kdim != -1)
      {
         fgmres->SetKDim(kdim);  // set FGMRES subspace size
      }
      if (prec != nullptr)
      {
         fgmres->SetPreconditioner(*prec);
      }
      return fgmres;
   }
   else if (solver_type == "hyprepcg")
   {
      auto pcg = std::make_unique<mfem::HyprePCG>(comm);
      pcg->SetTol(reltol);
      pcg->SetAbsTol(abstol);
      pcg->SetMaxIter(maxiter);
      pcg->SetPrintLevel(ptl);
      if (prec != nullptr)
      {
         pcg->SetPreconditioner(dynamic_cast<mfem::HypreSolver &>(*prec));
      }
      return pcg;
   }
   else if (solver_type == "pcg")
   {
      auto pcg = std::make_unique<mfem::CGSolver>(comm);
      pcg->SetRelTol(reltol);
      pcg->SetAbsTol(abstol);
      pcg->SetMaxIter(maxiter);
      pcg->SetPrintLevel(ptl);
      if (prec != nullptr)
      {
         pcg->SetPreconditioner(*prec);
      }
      return pcg;
   }
   else if (solver_type == "minres")
   {
      auto minres = std::make_unique<mfem::MINRESSolver>(comm);
      minres->SetRelTol(reltol);
      minres->SetAbsTol(abstol);
      minres->SetMaxIter(maxiter);
      minres->SetPrintLevel(ptl);
      if (prec != nullptr)
      {
         minres->SetPreconditioner(*prec);
      }
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

std::unique_ptr<mfem::NewtonSolver> constructNonlinearSolver(
    MPI_Comm comm,
    const nlohmann::json &nonlin_options,
    mfem::Solver &lin_solver)
{
   auto solver_type = nonlin_options["type"].get<std::string>();
   auto abstol = nonlin_options["abstol"].get<double>();
   auto reltol = nonlin_options["reltol"].get<double>();
   auto maxiter = nonlin_options["maxiter"].get<int>();
   auto ptl = nonlin_options["printlevel"].get<int>();

   std::unique_ptr<mfem::NewtonSolver> nonlin_solver;
   // allow not constructing nonlinear solver by specifying type as `none`
   if (solver_type == "none")
   {
      return nullptr;
   }
   else if (solver_type == "newton")
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

   nonlin_solver->iterative_mode = false;
   nonlin_solver->SetSolver(lin_solver);
   nonlin_solver->SetPrintLevel(ptl);
   nonlin_solver->SetRelTol(reltol);
   nonlin_solver->SetAbsTol(abstol);
   nonlin_solver->SetMaxIter(maxiter);

   return nonlin_solver;
}

}  // namespace mach
