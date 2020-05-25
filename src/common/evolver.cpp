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
   SystemOperator(Array<int> &ess_bdr, NonlinearFormType *_nonlinear_mass,
                  BilinearFormType *_mass, NonlinearFormType *_res,
                  BilinearFormType *_stiff, mfem::Vector *_load)
      : Operator(_res->Height()), nonlinear_mass(_nonlinear_mass), mass(_mass),
        res(_res), stiff(_stiff), load(_load), Jacobian(NULL),
        dt(0.0), x(NULL), work(height), work2(height)
   {
      if (_mass)
      {
#ifdef MFEM_USE_MPI
         _mass->ParFESpace()->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
#else
         _mass->FESpace()->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
#endif
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
      add(1.0, *x, dt, k, work);
      // Vector work(x);
      // Vector work2(x.Size());
      // Vector work3(x.Size());
      // work2 = work3 = 0.0;

      // work.Add(dt, k);  // work = x + dt * k
      r = 0.0;
      if (nonlinear_mass)
      {
         add(1.0, *x, 2.0*dt, k, work2);
         nonlinear_mass->Mult(work2, r);
         nonlinear_mass->Mult(*x, work2); // TODO: This could be precomputed
         r -= work2;
         r *= (0.5/dt);

         //nonlinear_mass->Mult(work, r);
         //nonlinear_mass->Mult(*x, work2); // TODO: This could be precomputed
         //r -= work2;
         //r *= (1/dt);
      }
      if (res)
      {
         res->Mult(work, work2);
         r += work2;
      }
      if (stiff)
      {
#ifdef MFEM_USE_MPI
         stiff->TrueAddMult(work, r);
         // r += work2;
#else
         stiff->AddMult(work, r);
#endif
      }
      if (load)
      {
         r += *load;
      }
      if (mass)
      {
#ifdef MFEM_USE_MPI
         mass->TrueAddMult(k, r);
         // r += work3;
         r.SetSubVector(ess_tdof_list, 0.0);
#else
         mass->AddMult(k, r);
#endif
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
#ifdef MFEM_USE_MPI
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
#else
      if (mass)
         jac = mass->SpMat();
      if (stiff)
      {
         if (jac != nullptr)
         {
            jac->Add(dt, *(stiff->SpMat()));
         }
         else
         {
            jac = stiff->SpMat();
            *jac *= dt;
         }
      }
#endif
 
      if ( (nonlinear_mass) || (res) )
      {
         /// work = x+dt*k = x+dt*dx/dt = x+dx
         add(1.0, *x, dt, k, work);
      }

      if (nonlinear_mass)
      {
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
#ifdef MFEM_USE_MPI
            jac = ParAdd(jac, massjac);
#else
            jac = Add(*jac, *massjac);
#endif
         }
      }

      if (res)
      {
         MatrixType* resjac = dynamic_cast<MatrixType*>(&res->GetGradient(work));
         *resjac *= dt;
         if (jac == nullptr)
         { 
            jac == resjac;
         }
         else
         {
#ifdef MFEM_USE_MPI
            jac = ParAdd(jac, resjac);
#else
            jac = Add(*jac, *resjac);
#endif
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

   mutable mfem::Vector work, work2;

   Array<int> ess_tdof_list;
};

MachEvolver::MachEvolver(
    Array<int> &ess_bdr, NonlinearFormType *_nonlinear_mass,
    BilinearFormType *_mass, NonlinearFormType *_res, BilinearFormType *_stiff,
    Vector *_load, NonlinearFormType *_ent, std::ostream &outstream,
    double start_time, TimeDependentOperator::Type type)
    : EntropyConstrainedOperator(_res->Height(), start_time, type),
      nonlinear_mass(_nonlinear_mass), res(_res), load(_load), ent(_ent),
      out(outstream), work(height), work2(height)
{
   outstream << "MachEvolver constructor" << endl;
   outstream.flush();
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
#ifdef MFEM_USE_MPI
         mass.Reset(_mass->ParallelAssemble(), true);
         mass_prec.reset(new HypreSmoother(*mass.As<HypreParMatrix>(),
                                           HypreSmoother::Jacobi));
         // mass_prec = HypreSmoother();
         // mass_prec.SetType(HypreSmoother::Jacobi);
#else
         mass.reset(_mass->SpMat(), true);
#endif
      }
      else
      {
         throw MachException("Unsupported assembly level for mass matrix!");
      }
#ifdef MFEM_USE_MPI
      mass_solver = CGSolver(_mass->ParFESpace()->GetComm());
#else
      mass_solver = CGSolver();
#endif
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
   #ifdef MFEM_USE_MPI
         stiff.Reset(_stiff->ParallelAssemble(), true);
   #else
         stiff.reset(_stiff->SpMat(), true);
   #endif
      }
      else
      {
         throw MachException("Unsupported assembly level"
                                                      "for stiffness matrix!");
      }
   }

   outstream << "MachEvolver constructor before combined_oper" << endl;
   combined_oper.reset(new SystemOperator(ess_bdr, _nonlinear_mass, _mass, _res,
                                          _stiff, _load));
   outstream << "MachEvolver constructor after combined_oper" << endl;
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
      res->Mult(x, work);
   }

   if (stiff.Ptr())
   {
      // stiff->AddMult(x, work); // <-- Cannot do AddMult with ParBilinearForm
      stiff->Mult(x, work2);
      add(work, work2, work);
   }
   if (load)
   {
      work += *load;
   }
   mass_solver.Mult(work, y);
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
   add(x, dt, k, work);
   // TODO: if using conservative variables, need to convert
   // if using entropy variables, do nothing
   //abs_solver->convertToEntvar(vec1);
   res->Mult(work, work2);
   return work * work2;
}

void MachEvolver::setOperParameters(double dt, const mfem::Vector *x)
{
   combined_oper->setParameters(dt, x);
}

// ImplicitNonlinearEvolver::ImplicitNonlinearEvolver(NewtonSolver *_newton,
//                                                    BilinearFormType *mass,
//                                                    NonlinearFormType *res,
//                                                    BilinearFormType *stiff,
//                                                    mfem::Vector *load,
//                                                    std::ostream &outstream,
//                                                    double start_time)
//    : MachEvolver(mass, res, stiff, load, outstream, start_time, IMPLICIT),
//      newton(_newton)
// {

// }

// void ImplicitNonlinearEvolver::Mult(const Vector &k, Vector &y) const
// {
//    Vector vec1(x);
//    Vector vec2(x.Size());
//    vec1.Add(dt, k);  // vec1 = x + dt * k
//    res.Mult(vec1, y); // y = f(vec1)
//    mass.Mult(k, vec2);  // vec2 = M * k
//    y += vec2;  // y = f(x + dt * k) - M * k
// }

// Operator &ImplicitNonlinearEvolver::GetGradient(const mfem::Vector &k) const
// {
//    MatrixType *jac;
//    Vector vec1(x);
//    vec1.Add(dt, k);
//    jac = dynamic_cast<MatrixType*>(&res.GetGradient(vec1)); 
//    jac->Add( dt-1.0, *jac );
//    jac->Add(1.0, mass);
//    return *jac;
// }

// void ImplicitNonlinearEvolver::ImplicitSolve(const double dt, const Vector &x,
//                                              Vector &k)
// {
//    SetParameters(dt, x);
//    mfem::Vector zero;
//    newton_solver->Mult(zero, k);
//    MFEM_ASSERT(newton_solver->GetConverged()==1, "Fail to solve dq/dx implicitly.\n");
// }

// LinearEvolver::LinearEvolver(MatrixType &m, MatrixType &k, ostream &outstream)
//    : TimeDependentOperator(m.Height(), 0.0, EXPLICIT), out(outstream), mass(m),
//      stiff(k), z(m.Height())
// {
//     // Here we extract the diagonal from the mass matrix and invert it
//     //M.GetDiag(z);
//     //cout << "minimum of z = " << z.Min() << endl;
//     //cout << "maximum of z = " << z.Max() << endl;
//     //ElementInv(z, Minv);
// #ifdef MFEM_USE_MPI
//    mass_prec.SetType(HypreSmoother::Jacobi);
//    mass_solver.reset(new CGSolver(mass.GetComm()));
// #else
//    mass_solver.reset(new CGSolver());
// #endif
//    mass_solver->SetPreconditioner(mass_prec);
//    mass_solver->SetOperator(mass);
//    mass_solver->iterative_mode = false; // do not use second arg of Mult as guess
//    mass_solver->SetRelTol(1e-9);
//    mass_solver->SetAbsTol(0.0);
//    mass_solver->SetMaxIter(100);
//    mass_solver->SetPrintLevel(0);
// }

// void LinearEvolver::Mult(const Vector &x, Vector &y) const
// {
//    // y = M^{-1} (K x)
//    //HadamardProd(Minv, x, y);
//    stiff.Mult(x, z);
//    mass_solver->Mult(z, y);
//    //HadamardProd(Minv, z, y);
// }

// NonlinearEvolver::NonlinearEvolver(MatrixType &m, NonlinearFormType &r,
//                                    double a)
//    : TimeDependentOperator(m.Height(), 0.0, EXPLICIT), mass(m), res(r),
//      z(m.Height()), alpha(a)
// {
// #ifdef MFEM_USE_MPI
//    mass_prec.SetType(HypreSmoother::Jacobi);
//    mass_solver.reset(new CGSolver(mass.GetComm()));
// #else
//    mass_solver.reset(new CGSolver());
// #endif
//    mass_solver->SetPreconditioner(mass_prec);
//    mass_solver->SetOperator(mass);
//    mass_solver->iterative_mode = false; // do not use second arg of Mult as guess
//    mass_solver->SetRelTol(1e-9);
//    mass_solver->SetAbsTol(0.0);
//    mass_solver->SetMaxIter(100);
//    mass_solver->SetPrintLevel(0);
// }

// void NonlinearEvolver::Mult(const Vector &x, Vector &y) const
// {
//    res.Mult(x, z);
//    mass_solver->Mult(z, y);
//    y *= alpha;
// }

// // /// TODO: rewrite this
// // ImplicitLinearEvolver::ImplicitLinearEvolver(const std::string &opt_file_name,
// //                                              MatrixType &m,
// //                                              MatrixType &k, 
// //                                              Vector b,
// //                                              std::ostream &outstream)
// //    : TimeDependentOperator(m.Height(), 0.0, IMPLICIT), out(outstream),
// //      force(move(b)), mass(m), stiff(k), z(m.Height())
// // {
// //    // t = m + dt*k

// //    // get options
// //    nlohmann::json file_options;
// //    ifstream opts(opt_file_name);
// //    opts >> file_options;
// //    options.merge_patch(file_options);

// // 	std::cout << "Setting Up Linear Solver..." << std::endl;
// //    t_prec.reset(new HypreSmoother());
// //    m_prec.reset(new HypreSmoother());
// // #ifdef MFEM_USE_MPI
// //    t_prec->SetType(HypreSmoother::Jacobi);
// //    t_solver.reset(new CGSolver(stiff.GetComm()));
// //    m_prec->SetType(HypreSmoother::Jacobi);
// //    m_solver.reset(new CGSolver(stiff.GetComm()));
// // #else
// //    t_solver.reset(new CGSolver());
// //    m_solver.reset(new CGSolver());
// // #endif
// //    // set parameters for the linear solver


// //    t_solver->iterative_mode = false;
// //    t_solver->SetRelTol(options["lin-solver"]["rel-tol"].get<double>());
// //    t_solver->SetAbsTol(options["lin-solver"]["abs-tol"].get<double>());
// //    t_solver->SetMaxIter(options["lin-solver"]["max-iter"].get<int>());
// //    t_solver->SetPrintLevel(options["lin-solver"]["print-lvl"].get<int>());
// //    t_solver->SetPreconditioner(*t_prec);

// //    m_solver->iterative_mode = false;
// //    m_solver->SetRelTol(options["lin-solver"]["rel-tol"].get<double>());
// //    m_solver->SetAbsTol(options["lin-solver"]["abs-tol"].get<double>());
// //    m_solver->SetMaxIter(options["lin-solver"]["max-iter"].get<int>());
// //    m_solver->SetPrintLevel(options["lin-solver"]["print-lvl"].get<int>());
// //    m_solver->SetPreconditioner(*m_prec);
// //    m_solver->SetOperator(mass);
// // }

// // void ImplicitLinearEvolver::Mult(const mfem::Vector &x, mfem::Vector &k) const
// // {
// //    stiff.Mult(x, z);
// //    z.Neg();  
// //    z.Add(-1, *rhs);
// //    m_solver->Mult(z, k);
// // }

// // void ImplicitLinearEvolver::ImplicitSolve(const double dt, const Vector &x, Vector &k)
// // {
// //    // if (T == NULL)
// //    // {
// //    T = Add(1.0, mass, dt, stiff);
// //    t_solver->SetOperator(*T);
// //    //}
// //    stiff.Mult(x, z);
// //    z.Neg();  
// //    z.Add(-1, *rhs);
// //    t_solver->Mult(z, k); 
// //    T = NULL;
// // }

// // ImplicitNonlinearEvolver::ImplicitNonlinearEvolver(MatrixType &m,
// //                                             NonlinearFormType &r,
// //                                             double a)
// //    : TimeDependentOperator(m.Height(), 0.0, IMPLICIT), alpha(a), mass(m),
// //      res(r)
// // {
// // #ifdef MFEM_USE_MPI
// // #ifdef MFEM_USE_PETSC
// //    // using petsc gmres solver
// //    linear_solver.reset(new mfem::PetscLinearSolver(mass.GetComm(), "solver_", 0));
// //    prec.reset(new mfem::PetscPreconditioner(mass.GetComm(), "prec_"));
// //    dynamic_cast<mfem::PetscLinearSolver *>(linear_solver.get())->SetPreconditioner(*prec);
// //    dynamic_cast<mfem::PetscSolver *>(linear_solver.get())->SetAbsTol(1e-10);
// //    dynamic_cast<mfem::PetscSolver *>(linear_solver.get())->SetRelTol(1e-10);
// //    dynamic_cast<mfem::PetscSolver *>(linear_solver.get())->SetMaxIter(100);
// //    dynamic_cast<mfem::PetscSolver *>(linear_solver.get())->SetPrintLevel(2);
// // #else
// //    //using hypre solver instead
// //    linear_solver.reset(new mfem::HypreGMRES(mass.GetComm()));
// //    dynamic_cast<mfem::HypreGMRES *>(linear_solver.get())->SetTol(1e-10);
// //    dynamic_cast<mfem::HypreGMRES *>(linear_solver.get())->SetPrintLevel(1);
// //    dynamic_cast<mfem::HypreGMRES *>(linear_solver.get())->SetMaxIter(100);
// // #endif
// //    newton_solver.reset(new mfem::NewtonSolver(mass.GetComm()));
// //    //newton_solver.reset(new mfem::InexactNewton(mass.GetComm(), 1e-4, 1e-1, 1e-4));
// // #else
// //    linear_solver.reset(new mfem::GMRESSolver());
// //    newton_solver.reset(new mfem::NewtonSolver());
// // #endif

// //    // set paramters for the newton solver
// //    newton_solver->SetRelTol(1e-10);
// //    newton_solver->SetAbsTol(1e-10);
// //    newton_solver->SetPrintLevel(1);
// //    newton_solver->SetMaxIter(30);
// //    // set linear solver and operator
// //    newton_solver->SetSolver(*linear_solver);
// //    newton_solver->SetOperator(*this);
// //    newton_solver->iterative_mode = false;
// // }


// // Operator &ImplicitNonlinearEvolver::GetGradient(const mfem::Vector &k) const
// // {
// //    MatrixType *jac;
// //    Vector vec1(x);
// //    vec1.Add(dt, k);
// //    jac = dynamic_cast<MatrixType*>(&res.GetGradient(vec1)); 
// //    jac->Add( dt-1.0, *jac );
// //    jac->Add(1.0, mass);
// //    return *jac;
// // }

// // void ImplicitNonlinearEvolver::ImplicitSolve(const double dt, const Vector &x,
// //                                              Vector &k)
// // {
// //    SetParameters(dt, x);
// //    mfem::Vector zero;
// //    newton_solver->Mult(zero, k);
// //    MFEM_ASSERT(newton_solver->GetConverged()==1, "Fail to solve dq/dx implicitly.\n");
// // }

} // namespace mach
