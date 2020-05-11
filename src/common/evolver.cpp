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
   // mass_prec.reset(new HypreEuclid(mass.GetComm()));
   // mass_solver.reset(new HypreGMRES(mass.GetComm()));
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

ImplicitNonlinearMassEvolver::ImplicitNonlinearMassEvolver(NonlinearFormType &nm,
                                 NonlinearFormType &r, double a)
   : TimeDependentOperator(nm.Height()), mass(nm), res(r), alpha(a)
{
#ifdef MFEM_USE_MPI
#ifdef MFEM_USE_PETSC
   // using petsc gmres solver
   linear_solver.reset(new mfem::PetscLinearSolver(mass.ParFESpace()->GetComm(), "solver_", 0));
   prec.reset(new mfem::PetscPreconditioner(mass.ParFESpace()->GetComm(), "prec_"));
   dynamic_cast<mfem::PetscLinearSolver *>(linear_solver.get())->SetPreconditioner(*prec);
   dynamic_cast<mfem::PetscSolver *>(linear_solver.get())->SetAbsTol(1e-10);
   dynamic_cast<mfem::PetscSolver *>(linear_solver.get())->SetRelTol(1e-2);
   dynamic_cast<mfem::PetscSolver *>(linear_solver.get())->SetMaxIter(100);
   dynamic_cast<mfem::PetscSolver *>(linear_solver.get())->SetPrintLevel(0);
#else
   //using hypre solver instead
   linear_solver.reset(new mfem::HypreGMRES(mass.ParFESpcace()->GetComm()));
   prec.reset(new HypreEuclid(mass.ParFESpace()->GetComm()));
   dynamic_cast<mfem::HypreGMRES *>(linear_solver.get())->SetTol(1e-10);
   dynamic_cast<mfem::HypreGMRES *>(linear_solver.get())->SetPrintLevel(0);
   dynamic_cast<mfem::HypreGMRES *>(linear_solver.get())->SetMaxIter(100);
   dynamic_cast<mfem::HypreGMRES*> (linear_solver.get())->SetPreconditioner(*dynamic_cast<HypreSolver*>(prec.get()));
#endif
   newton_solver.reset(new mfem::NewtonSolver(mass.ParFESpace()->GetComm()));
   //newton_solver.reset(new mfem::InexactNewton(mass.GetComm(), 1e-4, 1e-1, 1e-4));
#else
   linear_solver.reset(new mfem::GMRESSolver());
   newton_solver.reset(new mfem::NewtonSolver());
#endif

   // set paramters for the newton solver
   newton_solver->SetRelTol(1e-10);
   newton_solver->SetAbsTol(1e-10);
   newton_solver->SetPrintLevel(-1);
   newton_solver->SetMaxIter(30);
   // set linear solver and operator
   newton_solver->SetSolver(*linear_solver);
   newton_solver->SetOperator(*this);
   newton_solver->iterative_mode = false;
}

void ImplicitNonlinearMassEvolver::Mult(const Vector &k, Vector &y) const
{
   Vector vec1(x);
   Vector vec2(x.Size());
   vec1.Add(dt, k);  // vec1 = x + dt * k
   res.Mult(vec1, y); // y = f(vec1)
   mass.Mult(k, vec2);
   y += vec2;  // y = f(x + dt * k) + M(k)
}

Operator &ImplicitNonlinearMassEvolver::GetGradient(const mfem::Vector &k) const
{
   MatrixType *jac1, *jac2; 
   Vector vec1(x);
   vec1.Add(dt, k);
   jac1 = dynamic_cast<MatrixType*>(&res.GetGradient(vec1));
   *jac1 *= dt; // jac1 = dt * f'(x + dt * k) 
   jac2 = dynamic_cast<MatrixType*>(&mass.GetGradient(k)); // jac2 = M'(k);
   jac1->Add(1.0, *jac2);
   return *jac1;
}

void ImplicitNonlinearMassEvolver::ImplicitSolve(const double dt, const Vector &x,
                                             Vector &k)
{
   SetParameters(dt, x);
   mfem::Vector zero;
   newton_solver->Mult(zero, k);
   MFEM_ASSERT(newton_solver->GetConverged()==1, "Fail to solve dq/dx implicitly.\n");
}

void ImplicitNonlinearMassEvolver::checkJacobian(
    void (*pert_fun)(const mfem::Vector &, mfem::Vector &))
{
   cout << "evolver check jac is called.\n";
   // initialize some variables
   const double delta = 1e-5;
   GridFunType u_plus(*u);
   GridFunType u_minus(*u);
   GridFunType pert_vec(fes.get());
   VectorFunctionCoefficient up(num_state, pert_fun);
   pert_vec.ProjectCoefficient(up);

   // perturb in the positive and negative pert_vec directions
   u_plus.Add(delta, pert_vec);
   u_minus.Add(-delta, pert_vec);

   // Get the product using a 2nd-order finite-difference approximation
   GridFunType res_plus(fes.get());
   GridFunType res_minus(fes.get());
#ifdef MFEM_USE_MPI 
   HypreParVector *u_p = u_plus.GetTrueDofs();
   HypreParVector *u_m = u_minus.GetTrueDofs();
   HypreParVector *res_p = res_plus.GetTrueDofs();
   HypreParVector *res_m = res_minus.GetTrueDofs();
#else 
   GridFunType *u_p = &u_plus;
   GridFunType *u_m = &u_minus;
   GridFunType *res_p = &res_plus;
   GridFunType *res_m = &res_minus;
#endif
   this->Mult(*u_p, *res_p);
   this->Mult(*u_m, *res_m);
#ifdef MFEM_USE_MPI
   res_plus.SetFromTrueDofs(*res_p);
   res_minus.SetFromTrueDofs(*res_m);
#endif
   // res_plus = 1/(2*delta)*(res_plus - res_minus)
   subtract(1/(2*delta), res_plus, res_minus, res_plus);

   // Get the product directly using Jacobian from GetGradient
   GridFunType jac_v(fes.get());
#ifdef MFEM_USE_MPI
   HypreParVector *u_true = u->GetTrueDofs();
   HypreParVector *pert = pert_vec.GetTrueDofs();
   HypreParVector *prod = jac_v.GetTrueDofs();
#else
   GridFunType *u_true = u.get();
   GridFunType *pert = &pert_vec;
   GridFunType *prod = &jac_v;
#endif
   mfem::Operator &jac = this->GetGradient(*u_true);
   jac.Mult(*pert, *prod);
#ifdef MFEM_USE_MPI 
   jac_v.SetFromTrueDofs(*prod);
#endif 

   // check the difference norm
   jac_v -= res_plus;
   double error = calcInnerProduct(jac_v, jac_v);
   cout << "The Jacobian product error norm is " << sqrt(error) << endl;
}

} // end of mach namespace
