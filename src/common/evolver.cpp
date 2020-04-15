#include "evolver.hpp"
#include "utils.hpp"

using namespace mfem;
using namespace std;

namespace mach
{

class MachEvolver::SystemOperator : public mfem::Operator
{
public:
   /// Nonlinear operator of the form that combines the mass, res, stiff,
   /// and load elements for implicit/explicit ODE integration
   /// \param[in] mass - bilinear form for mass matrix (not owned)
   /// \param[in] res - nonlinear residual operator (not owned)
   /// \param[in] stiff - bilinear form for stiffness matrix (not owned)
   /// \param[in] load - load vector (not owned)
   SystemOperator(BilinearFormType *_mass, NonlinearFormType *_res,
                  BilinearFormType *_stiff, mfem::Vector *_load)
      : Operator(_mass->Height()), mass(_mass), res(_res), stiff(_stiff),
        load(_load), dt(0.0), x(nullptr), work(height) {}

   /// Compute r = M@k + R(x + dt*k,t) + K@(x+dt*k) + l
   /// (with `@` denoting matrix-vector multiplication)
   /// \param[in] k - dx/dt 
   /// \param[out] r - the residual
   /// \note the signs on each operator must be accounted for elsewhere
   void Mult(const mfem::Vector &k, mfem::Vector &r) const override
   {
      /// work = x+dt*k = x+dt*dx/dt = x+dx
      add(*x, dt, k, work);
      if (res)
         res->Mult(work, r);
      if (stiff)
         stiff->AddMult(work, r);
      mass->AddMult(k, r);
      if (load)
         r += *load;
   }

   /// Compute J = M + dt * grad(R(x + dt*k, t)) + dt * K
   /// \param[in] k - dx/dt 
   mfem::Operator &GetGradient(const mfem::Vector &k) const override
   {
      MatrixType *jac;
#ifdef MFEM_USE_MPI
      jac = mass->ParallelAssemble();
      if (stiff)
         jac->Add(dt, *(stiff->ParallelAssemble()));
#else
      jac = mass->SpMat();
      if (stiff)
         jac->Add(dt, *(stiff->SpMat()));
#endif
      if (res)
      {
         /// work = x+dt*k = x+dt*dx/dt = x+dx
         add(*x, dt, k, work);
         jac->Add(dt, *dynamic_cast<MatrixType*>(&res->GetGradient(work)));
      } 
      return *jac;
   }

   /// Set current dt and x values - needed to compute action and Jacobian.
   void setParameters(double _dt, const mfem::Vector *_x)
   {
      dt = _dt;
      x = _x;
   };

   ~SystemOperator() = default;

private:
   BilinearFormType *mass;
   NonlinearFormType *res;
   BilinearFormType *stiff;
   mfem::Vector *load;

   double dt;
   const mfem::Vector *x;
   mutable mfem::Vector work;
};

MachEvolver::MachEvolver(BilinearFormType *_mass, NonlinearFormType *_res,
                         BilinearFormType *_stiff, Vector *_load,
                         std::ostream &outstream, double start_time,
                         TimeDependentOperator::Type type)
   : TimeDependentOperator(_mass->Height(), start_time, type),
     res(_res), stiff(_stiff), load(_load), out(outstream)
{

   Array<int> ess_tdof_list;

   AssemblyLevel assem;
   assem = _mass->GetAssemblyLevel();
   if (assem == AssemblyLevel::PARTIAL)
   {
      mass.Reset(_mass, false);
      mass_prec.reset(new OperatorJacobiSmoother(*_mass, ess_tdof_list));
   }
   else if (assem == AssemblyLevel::FULL)
   {
#ifdef MFEM_USE_MPI
      mass.Reset(_mass->ParallelAssemble(), true);
      mass_prec.reset(new HypreSmoother(*mass.As<HypreParMatrix>(),
                                        HypreSmoother::Jacobi));
      // mass_prec = HypreSmoother();
      // mass_prec.SetType(HypreSmoother::Jacobi);
#else
      mass->reset(_mass->SpMat(), true);
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

   combined_oper.reset(new SystemOperator(_mass, _res, _stiff, _load));
}

MachEvolver::~MachEvolver() = default;

void MachEvolver::Mult(const mfem::Vector &x, mfem::Vector &y) const
{
   if (res)
   {
      res->Mult(x, work);
   }
   if (stiff)
   {
      // out << "have stiff\n";
      stiff->AddMult(x, work);
   }
   if (load)
   {
      // out << "have load\n";
      work += *load;
   }
   mass_solver.Mult(work, y);
   y *= -1.0;
}

void MachEvolver::ImplicitSolve(const double dt, const Vector &x,
                                Vector &k)
{
   setOperParameters(dt, &x);
   // combined_oper->setParameters(dt, &x);
   Vector zero; // empty vector is interpreted as zero r.h.s. by NewtonSolver
   newton->Mult(zero, k);
   MFEM_VERIFY(newton->GetConverged(), "Newton solver did not converge!");
}

void MachEvolver::SetNewtonSolver(NewtonSolver *_newton)
{
   newton = _newton;
   newton->SetOperator(*combined_oper);
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
