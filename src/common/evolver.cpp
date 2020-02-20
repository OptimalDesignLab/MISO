#include "evolver.hpp"
#include "utils.hpp"

using namespace mfem;
using namespace std;

namespace mach
{

LinearEvolver::LinearEvolver(MatrixType &m, MatrixType &k, ostream &outstream)
   : out(outstream), TimeDependentOperator(m.Height()), mass(m), stiff(k), z(m.Height())
{
    // Here we extract the diagonal from the mass matrix and invert it
    //M.GetDiag(z);
    //cout << "minimum of z = " << z.Min() << endl;
    //cout << "maximum of z = " << z.Max() << endl;
    //ElementInv(z, Minv);
#ifdef MFEM_USE_MPI
   mass_prec.SetType(HypreSmoother::Jacobi);
   mass_solver.reset(new CGSolver(mass.GetComm()));
#else
   mass_solver.reset(new CGSolver());
#endif
   mass_solver->SetPreconditioner(mass_prec);
   mass_solver->SetOperator(mass);
   mass_solver->iterative_mode = false; // do not use second arg of Mult as guess
   mass_solver->SetRelTol(1e-9);
   mass_solver->SetAbsTol(0.0);
   mass_solver->SetMaxIter(100);
   mass_solver->SetPrintLevel(0);
}

void LinearEvolver::Mult(const Vector &x, Vector &y) const
{
   // y = M^{-1} (K x)
   //HadamardProd(Minv, x, y);
   stiff.Mult(x, z);
   mass_solver->Mult(z, y);
   //HadamardProd(Minv, z, y);
}

NonlinearEvolver::NonlinearEvolver(MatrixType &m, NonlinearFormType &r,
                                   double a)
   : TimeDependentOperator(m.Height()), mass(m), res(r), z(m.Height()), alpha(a)
{
#ifdef MFEM_USE_MPI
   mass_prec.SetType(HypreSmoother::Jacobi);
   mass_solver.reset(new CGSolver(mass.GetComm()));
#else
   mass_solver.reset(new CGSolver());
#endif
   mass_solver->SetPreconditioner(mass_prec);
   mass_solver->SetOperator(mass);
   mass_solver->iterative_mode = false; // do not use second arg of Mult as guess
   mass_solver->SetRelTol(1e-9);
   mass_solver->SetAbsTol(0.0);
   mass_solver->SetMaxIter(100);
   mass_solver->SetPrintLevel(0);
}

void NonlinearEvolver::Mult(const Vector &x, Vector &y) const
{
   res.Mult(x, z);
   mass_solver->Mult(z, y);
   y *= alpha;
}

ImplicitLinearEvolver::ImplicitLinearEvolver(const std::string &opt_file_name,
                                             MatrixType &m,
                                             MatrixType &k, 
                                             std::unique_ptr<LinearForm> b,
                                             std::ostream &outstream)
   : out(outstream),  TimeDependentOperator(m.Height()), mass(m), stiff(k), 
                                             force(move(b)), z(m.Height())
{
   // t = m + dt*k

   // get options
   nlohmann::json file_options;
   ifstream opts(opt_file_name);
   opts >> file_options;
   options.merge_patch(file_options);

	std::cout << "Setting Up Linear Solver..." << std::endl;
   t_prec.reset(new HypreSmoother());
#ifdef MFEM_USE_MPI
   t_prec->SetType(HypreSmoother::Jacobi);
   t_solver.reset(new CGSolver(stiff.GetComm()));
#else
   t_solver.reset(new CGSolver());
#endif
   // set parameters for the linear solver


   t_solver->iterative_mode = false;
   t_solver->SetRelTol(options["lin-solver"]["rel-tol"].get<double>());
   t_solver->SetAbsTol(options["lin-solver"]["abs-tol"].get<double>());
   t_solver->SetMaxIter(options["lin-solver"]["max-iter"].get<int>());
   t_solver->SetPrintLevel(options["lin-solver"]["print-lvl"].get<int>());
   t_solver->SetPreconditioner(*t_prec);
}

void ImplicitLinearEvolver::ImplicitSolve(const double dt, const Vector &x, Vector &k)
{
   // if (T == NULL)
   // {
      T = Add(1.0, mass, dt, stiff);
      t_solver->SetOperator(*T);
   //}
   stiff.Mult(x, z);
   z.Neg();  
   z.Add(-1, *rhs);
   t_solver->Mult(z, k); 
   T = NULL;
}

ImplicitNonlinearEvolver::ImplicitNonlinearEvolver(MatrixType &m,
                                            NonlinearFormType &r,
                                            double a)
   : TimeDependentOperator(m.Height()), mass(m), res(r), alpha(a)
{
#ifdef MFEM_USE_MPI
#ifdef MFEM_USE_PETSC
   // using petsc gmres solver
   linear_solver.reset(new mfem::PetscLinearSolver(mass.GetComm(), "solver_", 0));
   prec.reset(new mfem::PetscPreconditioner(mass.GetComm(), "prec_"));
   dynamic_cast<mfem::PetscLinearSolver *>(linear_solver.get())->SetPreconditioner(*prec);
   dynamic_cast<mfem::PetscSolver *>(linear_solver.get())->SetAbsTol(1e-10);
   dynamic_cast<mfem::PetscSolver *>(linear_solver.get())->SetRelTol(1e-10);
   dynamic_cast<mfem::PetscSolver *>(linear_solver.get())->SetMaxIter(100);
   dynamic_cast<mfem::PetscSolver *>(linear_solver.get())->SetPrintLevel(2);
#else
   //using hypre solver instead
   linear_solver.reset(new mfem::HypreGMRES(mass.GetComm()));
   dynamic_cast<mfem::HypreGMRES *>(linear_solver.get())->SetTol(1e-10);
   dynamic_cast<mfem::HypreGMRES *>(linear_solver.get())->SetPrintLevel(1);
   dynamic_cast<mfem::HypreGMRES *>(linear_solver.get())->SetMaxIter(100);
#endif
   newton_solver.reset(new mfem::NewtonSolver(mass.GetComm()));
   //newton_solver.reset(new mfem::InexactNewton(mass.GetComm(), 1e-4, 1e-1, 1e-4));
#else
   linear_solver.reset(new mfem::GMRESSolver());
   newton_solver.reset(new mfem::NewtonSolver());
#endif

   // set paramters for the newton solver
   newton_solver->SetRelTol(1e-10);
   newton_solver->SetAbsTol(1e-10);
   newton_solver->SetPrintLevel(1);
   newton_solver->SetMaxIter(30);
   // set linear solver and operator
   newton_solver->SetSolver(*linear_solver);
   newton_solver->SetOperator(*this);
   newton_solver->iterative_mode = false;
}

void ImplicitNonlinearEvolver::Mult(const Vector &k, Vector &y) const
{
   Vector vec1(x);
   Vector vec2(x.Size());
   vec1.Add(dt, k);  // vec1 = x + dt * k
   res.Mult(vec1, y); // y = f(vec1)
   mass.Mult(k, vec2);  // vec2 = M * k
   y += vec2;  // y = f(x + dt * k) - M * k
}

Operator &ImplicitNonlinearEvolver::GetGradient(const mfem::Vector &k) const
{
   MatrixType *jac;
   Vector vec1(x);
   vec1.Add(dt, k);
   jac = dynamic_cast<MatrixType*>(&res.GetGradient(vec1)); 
   jac->Add( dt-1.0, *jac );
   jac->Add(1.0, mass);
   return *jac;
}

void ImplicitNonlinearEvolver::ImplicitSolve(const double dt, const Vector &x,
                                             Vector &k)
{
   SetParameters(dt, x);
   mfem::Vector zero;
   newton_solver->Mult(zero, k);
   MFEM_ASSERT(newton_solver->GetConverged()==1, "Fail to solve dq/dx implicitly.\n");
}

} // end of mach namespace
