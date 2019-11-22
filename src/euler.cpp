#include "euler.hpp"
#include "euler_fluxes.hpp"
#include "sbp_fe.hpp"
#include "diag_mass_integ.hpp"
#include "inviscid_integ.hpp"
#include "evolver.hpp"

using namespace mfem;
using namespace std;

namespace mach
{

EulerSolver::EulerSolver(const string &opt_file_name,
                         unique_ptr<mfem::Mesh> smesh, int dim)
    : AbstractSolver(opt_file_name, move(smesh))
{
   // set the finite-element space and create (but do not initialize) the
   // state GridFunction
   num_state = dim + 2;
   fes.reset(new SpaceType(mesh.get(), fec.get(), num_state, Ordering::byVDIM));
   u.reset(new GridFunType(fes.get()));
#ifdef MFEM_USE_MPI
   cout << "Number of finite element unknowns: "
        << fes->GlobalTrueVSize() << endl;
#else
   cout << "Number of finite element unknowns: "
        << fes->GetTrueVSize() << endl;
#endif

   // set up the mass matrix
   mass.reset(new BilinearFormType(fes.get()));
   mass->AddDomainIntegrator(new DiagMassIntegrator(num_state));
   mass->Assemble();
   mass->Finalize();

   // set up the spatial semi-linear form
   // TODO: should decide between one-point and two-point fluxes using options
   double alpha = 1.0;
   res.reset(new NonlinearFormType(fes.get()));

   res->AddDomainIntegrator(new IsmailRoeIntegrator<2>(diff_stack, alpha));

   //res->AddDomainIntegrator(new EulerIntegrator<2>(diff_stack, alpha));

   // add the LPS stabilization
   double lps_coeff = options["space-dis"]["lps-coeff"].get<double>();
   res->AddDomainIntegrator(new EntStableLPSIntegrator<2>(diff_stack, alpha,
                                                          lps_coeff));

   // boundary face integrators are handled in their own function
   addBoundaryIntegrators(alpha, dim);

   // define the time-dependent operator
#ifdef MFEM_USE_MPI
   // The parallel bilinear forms return a pointer that this solver owns
   mass_matrix.reset(mass->ParallelAssemble());
#else
   mass_matrix.reset(new MatrixType(mass->SpMat()));
#endif
   evolver.reset(new NonlinearEvolver(*mass_matrix, *res, -1.0));

   //A.reset(res->ParallelAssemble());
}

void EulerSolver::addBoundaryIntegrators(double alpha, int dim)
{
   auto &bcs = options["bcs"];
   bndry_marker.resize(bcs.size());
   int idx = 0;
   if (bcs.find("vortex") != bcs.end())
   { // isentropic vortex BC
      vector<int> tmp = bcs["vortex"].get<vector<int>>();
      bndry_marker[idx].SetSize(tmp.size(), 0);
      bndry_marker[idx].Assign(tmp.data());
      res->AddBdrFaceIntegrator(
          new IsentropicVortexBC<2>(diff_stack, fec.get(), alpha),
          bndry_marker[idx]);
      idx++;
   }
   if (bcs.find("slip-wall") != bcs.end())
   { // slip-wall boundary condition
      vector<int> tmp = bcs["slip-wall"].get<vector<int>>();
      bndry_marker[idx].SetSize(tmp.size(), 0);
      bndry_marker[idx].Assign(tmp.data());
      switch (dim)
      {
         case 1:
            res->AddBdrFaceIntegrator(
                new SlipWallBC<1>(diff_stack, fec.get(), alpha),
                bndry_marker[idx]);
            break;
         case 2:
            res->AddBdrFaceIntegrator(
               new SlipWallBC<2>(diff_stack, fec.get(), alpha),
               bndry_marker[idx]);
            break;
         case 3:
            res->AddBdrFaceIntegrator(
               new SlipWallBC<3>(diff_stack, fec.get(), alpha),
               bndry_marker[idx]);
            break;
      }
      idx++;
   }
   for (int k = 0; k < bndry_marker.size(); ++k)
   {
      cout << "boundary_marker[" << k << "]: ";
      for (int i = 0; i < bndry_marker[k].Size(); ++i)
      {
         cout << bndry_marker[k][i] << " ";
      }
      cout << endl;
   }
}

double EulerSolver::calcResidualNorm()
{
   GridFunType r(fes.get());
   res->Mult(*u, r); // TODO: option to recompute only if necessary
   double res_norm = r * r;
#ifdef MFEM_USE_MPI
   double loc_norm = res_norm;
   MPI_Allreduce(&loc_norm, &res_norm, 1, MPI_DOUBLE, MPI_SUM, comm);
#endif
   res_norm = sqrt(res_norm);
   return res_norm;
}

double EulerSolver::calcStepSize(double cfl) const
{
   double (*calcSpect)(const double *dir, const double *q);
   if (num_dim == 1)
   {
      calcSpect = calcSpectralRadius<double, 1>;
   }
   else if (num_dim == 2)
   {
      calcSpect = calcSpectralRadius<double, 2>;
   }
   else
   {
      calcSpect = calcSpectralRadius<double, 3>;
   }

   double dt_local = 1e100;
   Vector xi(num_dim);
   Vector dxij(num_dim);
   Vector ui, dxidx;
   DenseMatrix uk;
   DenseMatrix adjJt(num_dim);
   for (int k = 0; k < fes->GetNE(); k++)
   {
      // get the element, its transformation, and the state values on element
      const FiniteElement *fe = fes->GetFE(k);
      const IntegrationRule *ir = &(fe->GetNodes());
      ElementTransformation *trans = fes->GetElementTransformation(k);
      u->GetVectorValues(*trans, *ir, uk);
      for (int i = 0; i < fe->GetDof(); ++i)
      {
         trans->SetIntPoint(&fe->GetNodes().IntPoint(i));
         trans->Transform(fe->GetNodes().IntPoint(i), xi);
         CalcAdjugateTranspose(trans->Jacobian(), adjJt);
         uk.GetColumnReference(i, ui);
         for (int j = 0; j < fe->GetDof(); ++j)
         {
            if (j == i)
               continue;
            trans->Transform(fe->GetNodes().IntPoint(j), dxij);
            dxij -= xi;
            double dx = dxij.Norml2();
            dt_local = min(dt_local, cfl * dx * dx / calcSpect(dxij, ui)); // extra dx is to normalize dxij
         }
      }
   }
   double dt_min;
#ifdef MFEM_USE_MPI
   MPI_Allreduce(&dt_local, &dt_min, 1, MPI_DOUBLE, MPI_MIN, comm);
#else
   dt_min = dt_local;
#endif
   return dt_min;
}

/// Solve for the steady problem
void EulerSolver::solveSteady()
{
   // // Hypre solver section
   // prec.reset( new HypreBoomerAMG() );
   // prec->SetPrintLevel(0);
   // std::cout << "preconditioner is set.\n";
   // solver.reset( new HypreGMRES(fes->GetComm()) );
   // solver->SetTol(1e-10);
   // solver->SetMaxIter(100);
   // solver->SetPrintLevel(0);
   // //solver->SetPreconditioner(*prec);
   // std::cout << "Inner solver is set.\n";
   // newton_solver.iterative_mode = true;
   // newton_solver.SetSolver(*solver);
   // newton_solver.SetOperator(*res);
   // newton_solver.SetPrintLevel(1);
   // newton_solver.SetRelTol(1e-10);
   // newton_solver.SetAbsTol(1e-10);
   // newton_solver.SetMaxIter(50);
   // std::cout << "Newton solver is set.\n";
   // mfem::Vector b;
   // newton_solver.Mult(b,  *u);
   // MFEM_VERIFY(newton_solver.GetConverged(), "Newton solver did not converge.");

   // Petsc Solver section
   // std::cout << "Steady solver is called.\n";
   // const char *petscrc_file="eulersteady";
   // MFEMInitializePetsc(NULL, NULL, petscrc_file, NULL);
   // solver.reset(new PetscGMRESSolver(fes->GetComm(), petscrc_file));
   // dynamic_cast<mfem::PetscSolver*>(solver.get())->SetAbsTol(1e-12);
   // dynamic_cast<mfem::PetscSolver*>(solver.get())->SetRelTol(1e-12);
   // dynamic_cast<mfem::PetscSolver*>(solver.get())->SetMaxIter(100);
   // dynamic_cast<mfem::PetscSolver*>(solver.get())->SetPrintLevel(2);
   // std::cout << "Inner solver is set.\n";

   // newton_solver.iterative_mode = true;
   // newton_solver.SetSolver(*solver);
   // newton_solver.SetOperator(*res);
   // newton_solver.SetPrintLevel(1);
   // newton_solver.SetRelTol(1e-10);
   // newton_solver.SetAbsTol(1e-10);
   // newton_solver.SetMaxIter(50);
   // std::cout << "Newton solver is set.\n";

   // mfem::Vector b;
   // newton_solver.Mult(b,  *u);
   // MFEM_VERIFY(newton_solver.GetConverged(), "Newton solver did not converge.");

   // Before solving the nonlinear problem, solve the simple linear problem.
   mfem::Vector r(fes->GlobalTrueVSize());
   res->Mult(*u, r);
   const char *petscrc_file="eulersteady";
   MFEMInitializePetsc(NULL, NULL, petscrc_file, NULL);
   mfem::PetscLinearSolver* psolver = new 
               mfem::PetscLinearSolver(fes->GetComm(), petscrc_file, 0);
   psolver->iterative_mode = true;
   // prec = new PetscPreconditioner(res->GetGradient(*u),"solver_");
   
   std::cout << "The linear system is set.\n";
   psolver->SetAbsTol(1e-10);
   psolver->SetRelTol(1e-10);
   psolver->SetPrintLevel(2);
   psolver->SetMaxIter(100);
   psolver->SetOperator(res->GetGradient(*u));
   //psolver->SetPreconditioner(*prec);
   mfem::Vector c(fes->GlobalTrueVSize());
   //psolver->Mult(r, c);
   c.Print(std::cout, 4);
   delete psolver;
   MFEMFinalizePetsc();
}

void EulerSolver::jacobiancheck()
{
   // initialize the variables
   const double delta = 1e-5;
   std::unique_ptr<GridFunType> u_plus;
   std::unique_ptr<GridFunType> u_minus;
   std::unique_ptr<GridFunType> perturbation_vec;
   perturbation_vec.reset(new GridFunType(fes.get()));
   VectorFunctionCoefficient up(num_state, perturb_fun);
   perturbation_vec->ProjectCoefficient(up);
   u_plus.reset(new GridFunType(fes.get()));
   u_minus.reset(new GridFunType(fes.get()));

   // set uplus and uminus to the current state
   *u_plus = *u;
   *u_minus = *u;
   u_plus->Add(delta, *perturbation_vec);
   u_minus->Add(-delta, *perturbation_vec);

   std::cout << setprecision(14);
   // std::cout << "Check u:\n";
   //u->Save(std::cout);
   // std::cout << '\n';
   // std::cout << "Check u_plus:\n";
   //u_plus->Save(std::cout);
   // std::cout << '\n';
   // std::cout << "Check u_minus:\n";
   //u_minus->Save(std::cout);
   //std::cout << '\n';

   std::unique_ptr<GridFunType> res_plus;
   std::unique_ptr<GridFunType> res_minus;
   res_plus.reset(new GridFunType(fes.get()));
   res_minus.reset(new GridFunType(fes.get()));

   res->Mult(*u_plus, *res_plus);
   res->Mult(*u_minus, *res_minus);

   res_plus->Add(-1.0, *res_minus);
   res_plus->Set(1/(2*delta), *res_plus);
   // std::cout << "The residual difference is:\n";
   // res_plus->Save(std::cout);

   // result from GetGradient(x)
   std::unique_ptr<GridFunType> jac_v;
   jac_v.reset(new GridFunType(fes.get()));
   mfem::Operator &jac = res->GetGradient(*u);
   //jac.PrintMatlab(std::cout);
   jac.Mult(*perturbation_vec, *jac_v);
   //std::cout << "Resuelts from GetGradient(x):\n";
   //jac_v->Save(std::cout);
   // check the difference norm
   jac_v->Add(-1.0, *res_plus);
   std::cout << "The difference norm is " << jac_v->Norml2()<<'\n';
}

} // namespace mach
