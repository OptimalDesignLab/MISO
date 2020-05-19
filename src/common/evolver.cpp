#include "evolver.hpp"
#include "utils.hpp"
#include <iostream>
using namespace mfem;
using namespace std;
using namespace mach;

namespace mach
{


void RRKImplicitMidpointSolver::Init(TimeDependentOperator &_f)
{
   ODESolver::Init(_f);
   k.SetSize(f->Width(), mem_type);
}

void RRKImplicitMidpointSolver::Step(Vector &x, double &t, double &dt)
{
   f->SetTime(t + dt/2);
   f->ImplicitSolve(dt/2, x, k);
   //f->ImplicitSolve(dt, x, k);
   //cout << "equation solved using regular midpont solver\n";
   // Set-up and solve the scalar nonlinear problem for the relaxation gamma
   EntropyConstrainedOperator *f_ode =
       dynamic_cast<EntropyConstrainedOperator *>(f);
   //cout << "x size is " << x.Size() << '\n';
   //cout << "x is empty? == " << x.GetMemory().Empty() << '\n';
   double delta_entropy = f_ode->EntropyChange(dt/2, x, k);
   //double delta_entropy = f_ode->EntropyChange(dt, x, k);
   //cout << "delta_entropy is " << delta_entropy << '\n';
   double entropy_old = f_ode->Entropy(x);
   //cout << "old entropy is " << entropy_old << '\n';
   mfem::Vector x_new(x.Size());
   //cout << "x_new size is " << x_new.Size() << '\n';
   auto entropyFun = [&](double gamma)
   {
      cout << "In lambda function: " << std::setprecision(14); 
      add(x, gamma*dt, k, x_new);
      // x_new = k;
      // x_new *= (gamma*dt);
      // x_new += x;
      double entropy = f_ode->Entropy(x_new);
      cout << "gamma = " << gamma << ": ";
      cout << "residual = " << entropy - entropy_old + gamma*dt*delta_entropy << endl;
      //cout << "new entropy is " << entropy << '\n';
      return entropy - entropy_old + gamma*dt*delta_entropy;
   };
   // TODO: tolerances and maxiter should be provided in some other way
   const double ftol = 1e-12;
   const double xtol = 1e-12;
   const int maxiter = 30;
   //double gamma = bisection(entropyFun, 0.50, 1.5, ftol, xtol, maxiter);
   double gamma = secant(entropyFun, 1.9, 2.1, ftol, xtol, maxiter);
   cout << "\tgamma = " << gamma << endl;
   x.Add(gamma*dt, k);
   t += gamma*dt;
}

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
                                            AbstractSolver *abs,
                                            double a)
   : EntropyConstrainedOperator(m.Height()), 
     mass(m), res(r), abs_solver(abs), alpha(a)
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
   linear_solver.reset(new UMFPackSolver());
   dynamic_cast<UMFPackSolver *>(linear_solver.get())->Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
   dynamic_cast<UMFPackSolver *>(linear_solver.get())->SetPrintLevel(1);
   newton_solver.reset(new mfem::NewtonSolver());
#endif

   // set paramters for the newton solver
   newton_solver->SetRelTol(1e-10);
   newton_solver->SetAbsTol(1e-10);
   newton_solver->SetPrintLevel(0);
   newton_solver->SetMaxIter(30);
   // set linear solver and operator
   newton_solver->SetSolver(*linear_solver);
   newton_solver->SetOperator(*this);
   newton_solver->iterative_mode = false;
}

double ImplicitNonlinearEvolver::Entropy(const Vector &state)
{
   return abs_solver->GetOutput().at("entropy").GetEnergy(state);
}

double ImplicitNonlinearEvolver::EntropyChange(double dt, const Vector &state,
                                               const Vector &k)
{
   Vector vec1(state), vec2(k.Size());
   vec1.Add(dt, k);
   // if using conservative variables, need to convert
   // if using entropy variables, do nothing
   abs_solver->convertToEntvar(vec1);
   res.Mult(vec1, vec2);
   return vec1 * vec2;
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
   *jac *= dt;
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

void ImplicitNonlinearEvolver::checkJacobian(
    void (*pert_fun)(const mfem::Vector &, mfem::Vector &))
{
   cout << "evolver check jac is called.\n";
   // this is a specific version for gd_serial_mfem
   // dont accept incoming changes
   // initialize some variables
   const double delta = 1e-5;
   Vector u_plus(x);
   Vector u_minus(x);
   CentGridFunction pert_vec(res.FESpace());
   VectorFunctionCoefficient up(4, pert_fun);
   pert_vec.ProjectCoefficient(up);

   // perturb in the positive and negative pert_vec directions
   u_plus.Add(delta, pert_vec);
   u_minus.Add(-delta, pert_vec);

   // Get the product using a 2nd-order finite-difference approximation
   CentGridFunction res_plus(res.FESpace());
   CentGridFunction res_minus(res.FESpace());
   this->Mult(u_plus, res_plus);
   this->Mult(u_minus, res_minus);
   // res_plus = 1/(2*delta)*(res_plus - res_minus)
   subtract(1/(2*delta), res_plus, res_minus, res_plus);
   // Get the product directly using Jacobian from GetGradient
   CentGridFunction jac_v(res.FESpace());

   CentGridFunction *pert = &pert_vec;
   CentGridFunction *prod = &jac_v;

   mfem::Operator &jac = this->GetGradient(x);
   jac.Mult(*pert, *prod);

   // check the difference norm
   jac_v -= res_plus;
   //double error = calcInnerProduct(jac_v, jac_v);
   double error = jac_v * jac_v;
   std::cout << "The Jacobian product error norm is " << sqrt(error) << endl;
}


ImplicitNonlinearMassEvolver::ImplicitNonlinearMassEvolver(
    NonlinearFormType &nm, NonlinearFormType &r, NonlinearFormType &e, double a)
    : EntropyConstrainedOperator(nm.Height()), mass(nm), res(r), ent(e),
      alpha(a)
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
   dynamic_cast<mfem::PetscSolver *>(linear_solver.get())->SetPrintLevel(1);
#else
   //using hypre solver instead
   linear_solver.reset(new mfem::HypreGMRES(mass.ParFESpace()->GetComm()));
   prec.reset(new HypreEuclid(mass.ParFESpace()->GetComm()));
   dynamic_cast<mfem::HypreGMRES *>(linear_solver.get())->SetTol(1e-10);
   dynamic_cast<mfem::HypreGMRES *>(linear_solver.get())->SetPrintLevel(0);
   dynamic_cast<mfem::HypreGMRES *>(linear_solver.get())->SetMaxIter(100);
   dynamic_cast<mfem::HypreGMRES*> (linear_solver.get())->SetPreconditioner(*dynamic_cast<HypreSolver*>(prec.get()));
#endif
   newton_solver.reset(new mfem::NewtonSolver(mass.ParFESpace()->GetComm()));
   //newton_solver.reset(new mfem::InexactNewton(mass.GetComm(), 1e-4, 1e-1, 1e-4));
#else
   linear_solver.reset(new UMFPackSolver());
   dynamic_cast<UMFPackSolver *>(linear_solver.get())->Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
   dynamic_cast<UMFPackSolver *>(linear_solver.get())->SetPrintLevel(1);
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

void ImplicitNonlinearMassEvolver::Mult(const Vector &k, Vector &y) const
{   
   Vector vec1(x);
   Vector vec2(x.Size());
   //cout << "vec1&2 size is " << vec1.Size()<< '\n';
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

double ImplicitNonlinearMassEvolver::Entropy(const Vector &state)
{
   return ent.GetEnergy(state);
}

double ImplicitNonlinearMassEvolver::EntropyChange(double dt, const Vector &state,
                                                   const Vector &k)
{
   Vector vec1(state), vec2(k.Size());
   vec1.Add(dt, k);
   // if using conservative variables, need to convert
   // if using entropy variables, do nothing
   //abs_solver->convertToEntvar(vec1);
   res.Mult(vec1, vec2);
   return vec1 * vec2;
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
   // this is a specific version for gd_serial_mfem
   // dont accept incoming changes
   // initialize some variables
   const double delta = 1e-5;
   Vector u_plus(x);
   Vector u_minus(x);
   CentGridFunction pert_vec(mass.FESpace());
   VectorFunctionCoefficient up(4, pert_fun);
   pert_vec.ProjectCoefficient(up);

   // perturb in the positive and negative pert_vec directions
   u_plus.Add(delta, pert_vec);
   u_minus.Add(-delta, pert_vec);

   // Get the product using a 2nd-order finite-difference approximation
   CentGridFunction res_plus(mass.FESpace());
   CentGridFunction res_minus(mass.FESpace());
   this->Mult(u_plus, res_plus);
   this->Mult(u_minus, res_minus);
   // res_plus = 1/(2*delta)*(res_plus - res_minus)
   subtract(1/(2*delta), res_plus, res_minus, res_plus);
   // Get the product directly using Jacobian from GetGradient
   CentGridFunction jac_v(mass.FESpace());

   CentGridFunction *pert = &pert_vec;
   CentGridFunction *prod = &jac_v;

   mfem::Operator &jac = this->GetGradient(x);
   jac.Mult(*pert, *prod);

   // check the difference norm
   jac_v -= res_plus;
   //double error = calcInnerProduct(jac_v, jac_v);
   double error = jac_v * jac_v;
   std::cout << "The Jacobian product error norm is " << sqrt(error) << endl;
}

} // end of mach namespace
