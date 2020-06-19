#include "evolver.hpp"
#include "utils.hpp"
#include <iostream>
using namespace mfem;
using namespace std;
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
         res(_res), stiff(_stiff), load(_load), Jacobian(NULL),
         dt(0.0), x(NULL), x_work(width), r_work(height)
   {
      if ((_mass) && (ess_bdr))
      {
         _mass->ParFESpace()->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      }
      // x = 0.0;
   }

   /// Compute r = N(x + dt*k,t + dt) - N(x,t) + M@k + R(x + dt*k,t) + K@(x+dt*k) + l
   /// (with `@` denoting matrix-vector multiplication)
   /// \param[in] k - dx/dt 
   /// \param[out] r - the residual
   /// \note the signs on each operator must be accounted for elsewhere
   void Mult(const mfem::Vector &k, mfem::Vector &r) const override
   {
      /// work = x+dt*k = x+dt*dx/dt = x+dx
      add(1.0, *x, dt, k, x_work);

      r = 0.0;
      if (nonlinear_mass)
      {
         cout << "This needs to be fixed" << endl;
         throw(-1);
#if 0
         add(1.0, *x, 2.0*dt, k, work2);
         nonlinear_mass->Mult(work2, r);
         nonlinear_mass->Mult(*x, work2); // TODO: This could be precomputed
         r -= work2;
         r *= (0.5/dt);
#endif
      }
      if (res)
      {
         res->Mult(x_work, r_work);
         r += r_work;
      }
      if (stiff)
      {
         stiff->TrueAddMult(x_work, r);
      }
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

   /// Compute J = grad(N(k,x + dt*k)) + M + dt * grad(R(x + dt*k, t)) + dt * K
   /// \param[in] k - dx/dt 
   mfem::Operator &GetGradient(const mfem::Vector &k) const override
   {
//       delete Jacobian;
//       SparseMatrix *localJ;
//       if (stiff)
//       {
//          localJ = Add(-alpha, mass->SpMat(), dt, stiff->SpMat());
//       }
//       else
//       {
//          localJ = new SparseMatrix(mass->SpMat()); //, dt, S->SpMat());
//          *localJ *= -alpha;
//       }

//       if (res)
//       {
//          /// work = x+dt*k = x+dt*dx/dt = x+dx
//          add(1.0, *x, dt, k, work);
//          // Vector work(x);
//          work.Add(dt, k);  // work = x + dt * k
// #ifdef MFEM_USE_MPI
//          localJ->Add(dt, res->GetLocalGradient(work));
// #else
//          SparseMatrix *grad_H = dynamic_cast<SparseMatrix *>(&res->GetGradient(work));
//          localJ->Add(dt, *grad_H);
// #endif
//       }

// #ifdef MFEM_USE_MPI
//       Jacobian = mass->ParallelAssemble(localJ);
//       delete localJ;
//       HypreParMatrix *Je = Jacobian->EliminateRowsCols(ess_tdof_list);
//       delete Je;
// #else
//       Jacobian = localJ;
// #endif
//       return *Jacobian;

      MatrixType *jac = nullptr;
      if (mass)
         jac = mass->ParallelAssemble();
      if (stiff)
      {
         if (jac != nullptr)
         {
            jac->Add(dt, *(stiff->ParallelAssemble()));
         }
         else
         {
            jac = stiff->ParallelAssemble();
            *jac *= dt;
         }
      }
 
      if ( (nonlinear_mass) || (res) )
      {
         /// work = x+dt*k = x+dt*dx/dt = x+dx
         add(1.0, *x, dt, k, x_work);
      }

      if (nonlinear_mass)
      {
         cout << "This needs to be fixed" << endl;
         throw(-1);
#if 0
         add(*x, 2.0*dt, k, work2);
         MatrixType* massjac = dynamic_cast<MatrixType*>(
            &nonlinear_mass->GetGradient(work2));

         //MatrixType* massjac = dynamic_cast<MatrixType*>(
         //   &nonlinear_mass->GetGradient(work));

         if (jac == nullptr)
         {
            jac = massjac;
         }
         else
         {
            jac = ParAdd(jac, massjac);
         }
#endif

      }

      if (res)
      {
         MatrixType *resjac =
             dynamic_cast<MatrixType *>(&res->GetGradient(x_work));
         *resjac *= dt;
         if (jac == nullptr)
         { 
            jac = resjac;
         }
         else
         {
            jac = ParAdd(jac, resjac);
         }
      }
      return *jac;
   }

   /// Set current dt and x values - needed to compute action and Jacobian.
   void setParameters(double _dt, const mfem::Vector *_x)
   {
      dt = _dt;
      x = _x;
   };

   ~SystemOperator() { delete Jacobian; };

private:
   NonlinearFormType *nonlinear_mass;
   BilinearFormType *mass;
   NonlinearFormType *res;
   BilinearFormType *stiff;
   mfem::Vector *load;
   mutable MatrixType *Jacobian;
   double dt;
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
      Array<int> ess_tdof_list;

      AssemblyLevel mass_assem;
      mass_assem = _mass->GetAssemblyLevel();
      if (mass_assem == AssemblyLevel::PARTIAL)
      {
         mass.Reset(_mass, false);
         mass_prec.reset(new OperatorJacobiSmoother(*_mass, ess_tdof_list));
      }
      else if (mass_assem == AssemblyLevel::FULL)
      {
         mass.Reset(_mass->ParallelAssemble(), true);
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
      AssemblyLevel stiff_assem;
      stiff_assem = _stiff->GetAssemblyLevel();
      if (stiff_assem == AssemblyLevel::PARTIAL)
      {
         stiff.Reset(_stiff, false);
      }
      else if (stiff_assem == AssemblyLevel::FULL)
      {
         stiff.Reset(_stiff->ParallelAssemble(), true);
      }
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
   // TODO: if using conservative variables, need to convert
   // if using entropy variables, do nothing
   //abs_solver->convertToEntvar(vec1);
   res->Mult(x_work, r_work1);
   return x_work * r_work1;
}

void MachEvolver::setOperParameters(double dt, const mfem::Vector *x)
{
   combined_oper->setParameters(dt, x);
}

} // namespace mach
