#include <iostream>

#include "utils.hpp"
#include "mach_load.hpp"
#include "evolver.hpp"
using namespace mfem;

using namespace mach;

using namespace std;
namespace mach
{
void ODESystemOperator::Mult(const mfem::Vector &k, mfem::Vector &r) const
{
   r = 0.0;
   // Use x_work to store x + dt*k
   add(1.0, *x, dt, k, x_work);
   auto inputs =
       MachInputs({{"state", x_work.GetData()}, {"dxdt", k.GetData()}});
   evaluate(*res, inputs, r);
}

Operator &ODESystemOperator::GetGradient(const mfem::Vector &k) const
{
   // Use x_work to store x + dt*k
   add(1.0, *x, dt, k, x_work);
   auto inputs = MachInputs(
       {{"dt", dt}, {"state", x_work.GetData()}, {"dxdt", k.GetData()}});
   return getJacobian(*res, inputs, "dxdt");
}

class MachEvolver::SystemOperator : public mfem::Operator
{
public:
   /// Nonlinear operator of the form that combines the mass, res, stiff,
   /// and load elements for implicit/explicit ODE integration
   /// \param[in] nonlinear_mass - nonlinear mass matrix operator (not owned)
   /// \param[in] mass - bilinear form for mass matrix (not owned)
   /// \param[in] res - nonlinear residual operator (not owned)
   /// \param[in] load - load vector (not owned)
   /// \note The mfem::NewtonSolver class requires the operator's width and
   /// height to be the same; here we use `GetTrueVSize()` to find the process
   /// local height=width
   SystemOperator(Array<int> &ess_bdr,
                  ParNonlinearForm *_nonlinear_mass,
                  ParBilinearForm *_mass,
                  ParNonlinearForm *_res,
                  MachLoad *_load)
    : Operator(((_nonlinear_mass != nullptr)
                    ? _nonlinear_mass->FESpace()->GetTrueVSize()
                : (_mass != nullptr) ? _res->FESpace()->GetTrueVSize()
                                     : _res->FESpace()->GetTrueVSize())),
      pfes(((_nonlinear_mass != nullptr) ? _nonlinear_mass->ParFESpace()
            : (_mass != nullptr)         ? _mass->ParFESpace()
                                         : _res->ParFESpace())),
      nonlinear_mass(_nonlinear_mass),
      mass(_mass),
      res(_res),
      load(_load),
      jac(Operator::Hypre_ParCSR),
      dt(0.0),
      dt_stage(dt),
      x(nullptr),
      x_work(width),
      r_work(height)
   {
      if (_mass != nullptr)
      {
         //    std::cout << "is this called yet ? " << std::endl;
         //   _mass->ParFESpace()->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
         //    std::cout << "no problem here ? " << std::endl;
      }
      // else if (_stiff)
      // {
      //    _stiff->FESpace()->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      // }
      else if (_res != nullptr)
      {
         std::cout << "how about res ? " << std::endl;
         _res->ParFESpace()->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      }
      // if (load)
      //    load_tv = _res->ParFESpace()->NewTrueDofVector();
      // else
      //    load_tv = nullptr;
   }

   /// Compute r = N(x + dt_stage*k,t + dt) - N(x,t) + M@k + R(x + dt*k,t) +
   /// K@(x+dt*k) + l (with `@` denoting matrix-vector multiplication)
   /// \param[in] k - dx/dt
   /// \param[out] r - the residual
   /// \note the signs on each operator must be accounted for elsewhere
   void Mult(const mfem::Vector &k, mfem::Vector &r) const override
   {
      r = 0.0;
      if (nonlinear_mass != nullptr)
      {
         add(1.0, *x, dt_stage, k, x_work);
         nonlinear_mass->Mult(x_work, r);
         nonlinear_mass->Mult(*x, r_work);  // TODO: This could be precomputed
         r -= r_work;
         r *= 1 / dt_stage;
      }
      // x_work = x + dt*k = x + dt*dx/dt = x + dx
      add(1.0, *x, dt, k, x_work);
      // if (res)
      // {
      r_work = 0.0;
      res->Mult(x_work, r_work);
      r += r_work;
      // }
      // if (stiff)
      // {
      //    stiff->TrueAddMult(x_work, r);
      //    r.SetSubVector(ess_tdof_list, 0.0);
      // }
      if (load != nullptr)
      {
         mach::addLoad(*load, r);
         r.SetSubVector(ess_tdof_list, 0.0);
      }
      if (mass != nullptr)
      {
         mass->TrueAddMult(
             k, r);  /// @sigh - the prolongation operator gets used here
         r.SetSubVector(ess_tdof_list, 0.0);
      }
   }

   /// Compute J = grad(N(x + dt_stage*k)) + M + dt * grad(R(x + dt*k, t)) + dt
   /// * K \param[in] k - dx/dt
   mfem::Operator &GetGradient(const mfem::Vector &k) const override
   {
      jac.Clear();

      const SparseMatrix *mass_local_jac = nullptr;
      /// @warning - mass_local_jac needs to be modified if using `GD` (needs to
      /// be prolongated)
      if (mass != nullptr)
      {
         mass_local_jac = &mass->SpMat();
      }
      cout << "mass_local_jac size " << mass_local_jac->Size() << endl;
      if (nonlinear_mass != nullptr)
      {
         add(*x, dt_stage, k, x_work);
         mass_local_jac = &nonlinear_mass->GetLocalGradient(x_work);
      }

      add(1.0, *x, dt, k, x_work);
      const auto &res_local_jac = res->GetLocalGradient(x_work);

      std::unique_ptr<SparseMatrix> local_jac;

      if (mass_local_jac != nullptr)
      {
         local_jac.reset(Add(1.0, *mass_local_jac, dt, res_local_jac));
      }
      else
      {
         local_jac.reset(new SparseMatrix(res_local_jac, false));
      }

      /// TODO: this is taken from ParNonlinearForm::GetGradient
      {
         OperatorHandle dA(jac.Type());
         OperatorHandle Ph(jac.Type());

         /// TODO -> this will not work for shared face terms
         dA.MakeSquareBlockDiag(pfes->GetComm(),
                                pfes->GlobalVSize(),
                                pfes->GetDofOffsets(),
                                local_jac.get());

         // TODO - construct Dof_TrueDof_Matrix directly in the pGrad format
         Ph.ConvertFrom(pfes->Dof_TrueDof_Matrix());
         jac.MakePtAP(dA, Ph);

         // Impose b.c. on jac
         OperatorHandle jac_e;
         jac_e.EliminateRowsCols(jac, ess_tdof_list);
      }

      return *jac.Ptr();
   }

   /// Set current dt and x values - needed to compute action and Jacobian.
   /// \param[in] _dt - the step used to define where RHS is evaluated
   /// \praam[in] _x - current state
   /// \param[in] _dt_stage - the step for this entire stage/step
   /// \note `_dt` is the step usually assumed in mfem.  `_dt_stage` is needed
   /// by the nonlinear mass form and can be ignored if not needed.
   void setParameters(double _dt,
                      const mfem::Vector *_x,
                      double _dt_stage = -1.0)
   {
      dt = _dt;
      x = _x;
      dt_stage = _dt_stage;
   }

private:
   ParFiniteElementSpace *pfes;
   ParNonlinearForm *nonlinear_mass;
   ParBilinearForm *mass;
   ParNonlinearForm *res;
   // BilinearFormType *stiff;
   MachLoad *load;
   // mfem::HypreParVector *load_tv;
   // mutable HypreParMatrix *jac;
   mutable OperatorHandle jac;
   double dt;
   double dt_stage;
   const mfem::Vector *x;

   mutable mfem::Vector x_work;
   mutable mfem::Vector r_work;

   Array<int> ess_tdof_list;
};

MachEvolver::MachEvolver(Array<int> &ess_bdr,
                         NonlinearFormType *_nonlinear_mass,
                         BilinearFormType *_mass,
                         NonlinearFormType *_res,
                         BilinearFormType *_stiff,
                         MachLoad *_load,
                         NonlinearFormType *_ent,
                         std::ostream &outstream,
                         double start_time,
                         TimeDependentOperator::Type type,
                         bool _abort_on_no_converge)
 : EntropyConstrainedOperator(
       (_nonlinear_mass != nullptr) ? _nonlinear_mass->FESpace()->GetTrueVSize()
       : (_mass != nullptr)         ? _mass->ParFESpace()->GetTrueVSize()
                                    : _res->ParFESpace()->GetTrueVSize(),
       start_time,
       type),
   nonlinear_mass(_nonlinear_mass),
   res(_res),
   load(_load),
   ent(_ent),
   out(outstream),
   linsolver(nullptr),
   newton(nullptr),
   x_work(width),
   r_work1(height),
   r_work2(height),
   pfes(_res->ParFESpace()),
   abort_on_no_converge(_abort_on_no_converge)
{
   if ((_mass != nullptr) && (_nonlinear_mass != nullptr))
   {
      throw MachException(
          "Cannot use a linear and nonlinear mass operator "
          "simultaneously");
   }

   if (_mass != nullptr)
   {
      std::cout << "problem here " << std::endl;
      // Array<int> mass_ess_tdof_list;
      // _mass->ParFESpace()->GetEssentialTrueDofs(ess_bdr, mass_ess_tdof_list);
      // std::cout << "Nope " << std::endl;
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
         cout << "mass size after ParallelAssemble(): " << Mmat->Height()
              << " x " << Mmat->Width() << endl;
         std::ofstream msave("hypremass_dg.txt");
         Mmat->PrintMatlab(msave, 0, 0);
/// test to construct hypre matrix from sparse matrix
#if 0
         // ====================================================================================================
         hypre_mass = SparseToHypre(mass_matrix.get());
         cout << "hypre_mass size " << hypre_mass->Height() << " x "
              << hypre_mass->Width() << endl;
         std::ofstream hmsave("sparsehypremass.txt");
         hypre_mass->PrintMatlab(hmsave, 0, 0);

         mass.Reset(hypre_mass, true);
         // ====================================================================================================
#endif
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
      mass_solver.SetRelTol(1e-10);
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
         stiff.Reset(Smat, true);
      }
      else
      {
         throw MachException(
             "Unsupported assembly level"
             "for stiffness matrix!");
      }
   }
   combined_oper.reset(
       new SystemOperator(ess_bdr, _nonlinear_mass, _mass, _res, _load));
}

MachEvolver::~MachEvolver() = default;

void MachEvolver::Mult(const mfem::Vector &x, mfem::Vector &y) const
{
   if (nonlinear_mass != nullptr)
   {
      throw MachException("Cannot use MachEvolver::Mult with nonlinear mass");
   }

   if (res != nullptr)
   {
      res->Mult(x, r_work1);
   }

   if (stiff.Ptr() != nullptr)
   {
      // stiff->AddMult(x, work); // <-- Cannot do AddMult with ParBilinearForm
      stiff->Mult(x, r_work2);
      add(r_work1, r_work2, r_work1);
   }
   if (load != nullptr)
   {
      // r_work1 += *load;
      addLoad(*load, r_work1);
   }
   mass_solver.Mult(r_work1, y);
   y *= -1.0;
}

void MachEvolver::ImplicitSolve(const double dt, const Vector &x, Vector &k)
{
   setOperParameters(dt, &x);
   Vector zero;  // empty vector is interpreted as zero r.h.s. by NewtonSolver

   // set iterative mode to false so k is only zeroed once, but set back after
   auto iter_mode = newton->iterative_mode;
   newton->iterative_mode = false;
   newton->Mult(zero, k);
   newton->iterative_mode = iter_mode;

   if (abort_on_no_converge)
   {
      MFEM_VERIFY(newton->GetConverged(), "Newton solver did not converge!");
   }
}

HypreParMatrix *MachEvolver::SparseToHypre(SparseMatrix *A_local)
{
   cout << "^^^ Inside SparseToHypre() ^^^" << endl;
   OperatorHandle A(Operator::Hypre_ParCSR);
   A.Clear();

   if (A_local == NULL)
   {
      A.SetOperatorOwner(false);
      return A.As<HypreParMatrix>();
   }
   MFEM_VERIFY(A_local->Finalized(), "the local matrix must be finalized");

   OperatorHandle dA(A.Type()), Ph(A.Type()), hdA;
   HYPRE_Int row_size = A_local->Height();
   HYPRE_Int col_size = A_local->Width();
   cout << " row_size " << row_size << endl;
   HYPRE_Int *mat_row_idx;
   HYPRE_Int *mat_col_idx;
   mat_row_idx = new HYPRE_Int[2];
   mat_col_idx = new HYPRE_Int[2];
   mat_row_idx[0] = 0;
   mat_row_idx[1] = A_local->Height();
   mat_col_idx[0] = 0;
   mat_col_idx[1] = A_local->Width();
   hdA.Reset(new HypreParMatrix(
       pfes->GetComm(), row_size, col_size, mat_row_idx, mat_col_idx, A_local));

#if 0 
   // handle the case when 'a' contains off-diagonal
   int lvsize = pfes->GetVSize();
   const HYPRE_BigInt *face_nbr_glob_ldof = pfes->GetFaceNbrGlobalDofMap();
   HYPRE_BigInt ldof_offset = pfes->GetMyDofOffset();
   Array<HYPRE_BigInt> glob_J(A_local->NumNonZeroElems());
   int *J = A_local->GetJ();
   for (int i = 0; i < glob_J.Size(); i++)
   {
      if (J[i] < lvsize)
      {
         glob_J[i] = J[i] + ldof_offset;
      }
      else
      {
         glob_J[i] = face_nbr_glob_ldof[J[i] - lvsize];
      }
   }

   // TODO - construct dA directly in the A format
   hdA.Reset(new HypreParMatrix(pfes->GetComm(),
                                lvsize,
                                pfes->GlobalVSize(),
                                pfes->GlobalVSize(),
                                A_local->GetI(),
                                glob_J,
                                A_local->GetData(),
                                pfes->GetDofOffsets(),
                                pfes->GetDofOffsets()));
   // - hdA owns the new HypreParMatrix
   // - the above constructor copies all input arrays
   glob_J.DeleteAll();
#endif
   dA.ConvertFrom(hdA);

   // TODO - assemble the Dof_TrueDof_Matrix directly in the required format?
   Ph.ConvertFrom(pfes->Dof_TrueDof_Matrix());
   // TODO: When Ph.Type() == Operator::ANY_TYPE we want to use the Operator
   // returned by pfes->GetProlongationMatrix(), however that Operator is a
   // const Operator, so we cannot store it in OperatorHandle. We need a
   // const version of class OperatorHandle, e.g. ConstOperatorHandle.
   cout << "No pblm here: 1" << endl;
   A.MakePtAP(dA, Ph);

   cout << "No pblm here: 2" << endl;
   A.SetOperatorOwner(false);
   return A.As<HypreParMatrix>();
}

void MachEvolver::ImplicitSolve(const double dt_stage,
                                const double dt,
                                const Vector &x,
                                Vector &k)
{
   setOperParameters(dt, &x, dt_stage);
   Vector zero;  // empty vector is interpreted as zero r.h.s. by NewtonSolver
   k = 0.0;      // In case iterative mode is set to true
   newton->Mult(zero, k);
   if (abort_on_no_converge)
   {
      MFEM_VERIFY(newton->GetConverged(), "Newton solver did not converge!");
   }
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

mfem::Operator &MachEvolver::GetGradient(const mfem::Vector &x) const
{
   return combined_oper->GetGradient(x);
}

double MachEvolver::Entropy(const mfem::Vector &x)
{
   if (ent == nullptr)
   {
      throw MachException("MachEvolver::Entropy(): ent member not defined!");
   }
   return ent->GetEnergy(x);
}

double MachEvolver::EntropyChange(double dt,
                                  const mfem::Vector &x,
                                  const mfem::Vector &k)
{
   /// even though it is not used here, ent should be defined
   if (ent == nullptr)
   {
      throw MachException("MachEvolver::EntropyChange(): ent not defined!");
   }
   add(x, dt, k, x_work);
   return res->GetEnergy(x_work);
}

void MachEvolver::setOperParameters(double dt,
                                    const mfem::Vector *x,
                                    double dt_stage)
{
   combined_oper->setParameters(dt, x, dt_stage);
}

}  // namespace mach
