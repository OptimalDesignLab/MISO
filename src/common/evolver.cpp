#include <iostream>

#include "utils.hpp"
#include "evolver.hpp"

using namespace mfem;

using namespace mach;

namespace mach
{

class MachEvolver::SystemOperator : public mfem::Operator
{
public:
   /// Nonlinear operator of the form that combines the mass, res, stiff,
   /// and load elements for implicit/explicit ODE integration
   /// \param[in] nonlinear_mass - nonlinear mass matrix operator (not owned)
   /// \param[in] mass - bilinear form for mass matrix (not owned)
   /// \param[in] res - nonlinear residual operator (not owned)
   /// \param[in] stiff - bilinear form for stiffness matrix (not owned)
   /// \param[in] load - load vector (not owned)
   /// \note The mfem::NewtonSolver class requires the operator's width and
   /// height to be the same; here we use `GetTrueVSize()` to find the process
   /// local height=width
   SystemOperator(Array<int> &ess_bdr, NonlinearFormType *_nonlinear_mass,
                  BilinearFormType *_mass, NonlinearFormType *_res,
                  BilinearFormType *_stiff, mfem::Vector *_load)
       : Operator(((_nonlinear_mass != nullptr)
                       ? _nonlinear_mass->FESpace()->GetTrueVSize()
                       : _mass->FESpace()->GetTrueVSize())),
         nonlinear_mass(_nonlinear_mass), mass(_mass),
         res(_res), stiff(_stiff), load(_load), jac(nullptr),
         dt(0.0), x(nullptr), x_work(width), r_work(height)
   {
      if (_mass)
      {
         _mass->ParFESpace()->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      }
      else if (_stiff)
      {
         _stiff->FESpace()->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      }
   }

   /// Compute r = N(x + dt_stage*k,t + dt) - N(x,t) + M@k + R(x + dt*k,t) + K@(x+dt*k) + l
   /// (with `@` denoting matrix-vector multiplication)
   /// \param[in] k - dx/dt 
   /// \param[out] r - the residual
   /// \note the signs on each operator must be accounted for elsewhere
   void Mult(const mfem::Vector &k, mfem::Vector &r) const override
   {
      r = 0.0;
      if (nonlinear_mass)
      {
         add(1.0, *x, dt_stage, k, x_work);
         nonlinear_mass->Mult(x_work, r);
         nonlinear_mass->Mult(*x, r_work); // TODO: This could be precomputed
         r -= r_work;
         r *= 1/dt_stage;
      }
      // x_work = x + dt*k = x + dt*dx/dt = x + dx
      add(1.0, *x, dt, k, x_work);
      if (res)
      {
         res->Mult(x_work, r_work);
         r += r_work;
      }
      // if (stiff)
      // {
      //    stiff->TrueAddMult(x_work, r);
      //    r.SetSubVector(ess_tdof_list, 0.0);
      // }
      if (load)
      {
         r += *load;
      }
      if (mass)
      {
         mass->TrueAddMult(k, r);
         r.SetSubVector(ess_tdof_list, 0.0);
      }
   }

   /// Compute J = grad(N(x + dt_stage*k)) + M + dt * grad(R(x + dt*k, t)) + dt * K
   /// \param[in] k - dx/dt 
   mfem::Operator &GetGradient(const mfem::Vector &k) const override
   {
      delete jac;
      jac = nullptr;

      if (mass)
         jac = mass->ParallelAssemble();
      // if (stiff)
      // {
      //    HypreParMatrix *stiffmat = stiff->ParallelAssemble();
      //    jac == nullptr ? jac = stiffmat : jac = Add(1.0, *jac, dt, *stiffmat);
      // }
      if (nonlinear_mass)
      {
         add(*x, dt_stage, k, x_work);
         MatrixType* massjac = dynamic_cast<MatrixType*>(
            &nonlinear_mass->GetGradient(x_work));
         jac == nullptr ? jac = massjac : jac = ParAdd(jac, massjac);
      }
      if (res)
      {
         // x_work = x + dt*k = x + dt*dx/dt = x + dx
         add(1.0, *x, dt, k, x_work);
         MatrixType *resjac =
             dynamic_cast<MatrixType *>(&res->GetGradient(x_work));
         *resjac *= dt;
         jac == nullptr ? jac = resjac : jac = ParAdd(jac, resjac);
      }
      HypreParMatrix *Je = jac->EliminateRowsCols(ess_tdof_list);
      delete Je;

      return *jac;
   }

   /// Set current dt and x values - needed to compute action and Jacobian.
   /// \param[in] _dt - the step used to define where RHS is evaluated
   /// \praam[in] _x - current state
   /// \param[in] _dt_stage - the step for this entire stage/step
   /// \note `_dt` is the step usually assumed in mfem.  `_dt_stage` is needed
   /// by the nonlinear mass form and can be ignored if not needed.
   void setParameters(double _dt, const mfem::Vector *_x, double _dt_stage = -1.0)
   {
      dt = _dt;
      x = _x;
      dt_stage = _dt_stage;
   };

   ~SystemOperator() { delete jac; };

private:
   NonlinearFormType *nonlinear_mass;
   BilinearFormType *mass;
   NonlinearFormType *res;
   BilinearFormType *stiff;
   mfem::Vector *load;
   mutable MatrixType *jac;
   double dt;
   double dt_stage;
   const mfem::Vector *x;

   mutable mfem::Vector x_work;
   mutable mfem::Vector r_work;

   Array<int> ess_tdof_list;
};

MachEvolver::MachEvolver(
    Array<int> &ess_bdr, NonlinearFormType *_nonlinear_mass,
    BilinearFormType *_mass, NonlinearFormType *_res, BilinearFormType *_stiff,
    Vector *_load, NonlinearFormType *_ent, std::ostream &outstream,
    double start_time, TimeDependentOperator::Type type)
    : EntropyConstrainedOperator((_nonlinear_mass != nullptr)
                                     ? _nonlinear_mass->FESpace()->GetTrueVSize()
                                     : _mass->FESpace()->GetTrueVSize(),
                                 start_time, type),
      nonlinear_mass(_nonlinear_mass), res(_res), load(_load), ent(_ent),
      out(outstream), x_work(width), r_work1(height), r_work2(height)
{
   if ( (_mass != nullptr) && (_nonlinear_mass != nullptr) )
   {
      throw MachException("Cannot use a linear and nonlinear mass operator "
                          "simultaneously");
   }

   if (_mass != nullptr)
   {
      Array<int> mass_ess_tdof_list;
      _mass->FESpace()->GetEssentialTrueDofs(ess_bdr, mass_ess_tdof_list);

      AssemblyLevel mass_assem;
      mass_assem = _mass->GetAssemblyLevel();
      if (mass_assem == AssemblyLevel::PARTIAL)
      {
         mass.Reset(_mass, false);
         mass_prec.reset(new OperatorJacobiSmoother(*_mass, ess_tdof_list));
      }
      else if (mass_assem == AssemblyLevel::LEGACYFULL)
      {
         auto *Mmat = _mass->ParallelAssemble();
         auto *Me = Mmat->EliminateRowsCols(ess_tdof_list);
         delete Me;
         mass.Reset(Mmat, true);
         mass_prec.reset(new HypreSmoother(*mass.As<HypreParMatrix>(),
                                           HypreSmoother::Jacobi));
      }
      else
      {
         throw MachException("Unsupported assembly level for mass matrix!");
      }
      mass_solver = CGSolver(_mass->ParFESpace()->GetComm());
      mass_solver.SetPreconditioner(*mass_prec);
      mass_solver.SetOperator(*mass);
      mass_solver.SetRelTol(1e-9);
      mass_solver.SetAbsTol(0.0);
      mass_solver.SetMaxIter(100);
      mass_solver.SetPrintLevel(0);
   }

   if (_stiff != nullptr)
   {
      // Array<int> ess_tdof_list;
      _stiff->FESpace()->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

      AssemblyLevel stiff_assem;
      stiff_assem = _stiff->GetAssemblyLevel();
      if (stiff_assem == AssemblyLevel::PARTIAL)
      {
         stiff.Reset(_stiff, false);
      }
      else if (stiff_assem == AssemblyLevel::LEGACYFULL)
      {
         auto *Smat = _stiff->ParallelAssemble();
         auto *Se = Smat->EliminateRowsCols(ess_tdof_list);
         delete Se;
         stiff.Reset(Smat, true);      }
      else
      {
         throw MachException("Unsupported assembly level"
                                                      "for stiffness matrix!");
      }
   }
   combined_oper.reset(new SystemOperator(ess_bdr, _nonlinear_mass, _mass, _res,
                                          _stiff, _load));
}

MachEvolver::~MachEvolver() = default;

void MachEvolver::Mult(const mfem::Vector &x, mfem::Vector &y) const
{
   if (nonlinear_mass != nullptr)
   {
      throw MachException("Cannot use MachEvolver::Mult with nonlinear mass");
   }

   if (res)
   {
      res->Mult(x, r_work1);
   }

   if (stiff.Ptr())
   {
      // stiff->AddMult(x, work); // <-- Cannot do AddMult with ParBilinearForm
      stiff->Mult(x, r_work2);
      add(r_work1, r_work2, r_work1);
   }
   if (load)
   {
      r_work1 += *load;
   }
   mass_solver.Mult(r_work1, y);
   y *= -1.0;
}

void MachEvolver::ImplicitSolve(const double dt, const Vector &x,
                                Vector &k)
{
   setOperParameters(dt, &x);
   Vector zero; // empty vector is interpreted as zero r.h.s. by NewtonSolver

   // set iterative mode to false so k is only zeroed once, but set back after
   auto iter_mode = newton->iterative_mode;
   newton->iterative_mode = false;
   newton->Mult(zero, k);
   newton->iterative_mode = iter_mode;
   
   MFEM_VERIFY(newton->GetConverged(), "Newton solver did not converge!");
}

void MachEvolver::ImplicitSolve(const double dt_stage, const double dt,
                                const Vector &x, Vector &k)
{
   setOperParameters(dt, &x, dt_stage);
   Vector zero; // empty vector is interpreted as zero r.h.s. by NewtonSolver
   k = 0.0; // In case iterative mode is set to true
   newton->Mult(zero, k);
   MFEM_VERIFY(newton->GetConverged(), "Newton solver did not converge!");
}

void MachEvolver::SetLinearSolver(Solver *_linsolver)
{
   linsolver = _linsolver;
}

void MachEvolver::SetNewtonSolver(NewtonSolver *_newton)
{
   newton = _newton;
   newton->SetOperator(*combined_oper);
}

mfem::Operator& MachEvolver::GetGradient(const mfem::Vector &x) const
{
   return combined_oper->GetGradient(x);
}

double MachEvolver::Entropy(const mfem::Vector &x)
{
   if (!ent)
   {
      throw MachException("MachEvolver::Entropy(): ent member not defined!");
   }
   return ent->GetEnergy(x);
}

double MachEvolver::EntropyChange(double dt, const mfem::Vector &x,
                                  const mfem::Vector &k)
{
   if (!ent) // even though it is not used here, ent should be defined
   {
      throw MachException("MachEvolver::EntropyChange(): ent not defined!");
   }
   add(x, dt, k, x_work);
   return res->GetEnergy(x_work);
}

void MachEvolver::setOperParameters(double dt, const mfem::Vector *x,
                                    double dt_stage)
{
   combined_oper->setParameters(dt, x, dt_stage);
}

} // namespace mach
