#include "euler.hpp"
//#include "euler_fluxes.hpp"
#include "sbp_fe.hpp"
#include "diag_mass_integ.hpp"
#include "euler_integ.hpp"
#include "evolver.hpp"
#include <fstream>
#include <iostream>

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
   // add the integrators based on if discretization is continuous or discrete
   if (options["space-dis"]["basis-type"].get<string>() == "csbp")
   {
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
      // res->AddDomainIntegrator(new EntStableLPSShockIntegrator<2>(diff_stack, alpha,
      //                                                        lps_coeff));

      // boundary face integrators are handled in their own function
      addBoundaryIntegrators(alpha, dim);
   }
   else if (options["space-dis"]["basis-type"].get<string>() == "dsbp")
   {
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
      // add the LPS stabilization
      //TODO: how to get good solution without LPS integrator
      double lps_coeff = options["space-dis"]["lps-coeff"].get<double>();
      res->AddDomainIntegrator(new EntStableLPSIntegrator<2>(diff_stack, alpha,
                                                             lps_coeff));
      res->AddInteriorFaceIntegrator(new InterfaceIntegrator<2>(diff_stack,
                                                                fec.get(), alpha));
      addBoundaryIntegrators(alpha, dim);
   }

   // define the time-dependent operator
#ifdef MFEM_USE_MPI
   // The parallel bilinear forms return a pointer that this solver owns
   mass_matrix.reset(mass->ParallelAssemble());
#else
   mass_matrix.reset(new MatrixType(mass->SpMat()));
#endif
   evolver.reset(new NonlinearEvolver(*mass_matrix, *res, -1.0));
}

void EulerSolver::addBoundaryIntegrators(double alpha, int dim)
{
   auto &bcs = options["bcs"];
   bndry_marker.resize(bcs.size());
   int idx = 0;
   if (bcs.find("wedge-shock") != bcs.end())
   { // shock capturing wedge problem BC
      vector<int> tmp = bcs["wedge-shock"].get<vector<int>>();
      bndry_marker[idx].SetSize(tmp.size(), 0);
      bndry_marker[idx].Assign(tmp.data());
      res->AddBdrFaceIntegrator(
          new WedgeShockBC<2>(diff_stack, fec.get(), alpha),
          bndry_marker[idx]);
      idx++;
   }
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
   HypreParVector *U = u->GetTrueDofs();
   res->Mult(*U, r);   
   double loc_norm = r*r;
   MPI_Allreduce(&loc_norm, &res_norm, 1, MPI_DOUBLE, MPI_SUM, comm);
#else
   res->Mult(*u, r);
   res_norm = r*r;
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
   // // use Newton method with GMRES to solve for steady state problem
   // mfem::NewtonSolver * ntsolver = new NewtonSolver();
   // mfem::GMRESSolver * gmres = new GMRESSolver();

   // // set the members of GMRES solver
   // gmres->SetRelTol(1e-12);
   // gmres->SetAbsTol(1e-12);
   // gmres->SetMaxIter(50);
   // //mfem::GSSmoother gs_prec = new GSSmoother();
   // //gmres->SetPreconditioner(*gs_prec);
   // gmres->iterative_mode = false;

   // // // set the members for NewtonSolver
   // ntsolver->SetRelTol(1e-10);
   // ntsolver->SetAbsTol(1e-12);
   // ntsolver->SetMaxIter(50);
   // ntsolver->SetSolver(*gmres);
   // ntsolver->SetOperator(*res);

   // mfem::Vector zero;
   // ntsolver->Mult(zero, *u);
   // MFEM_ASSERT(ntsolver.GetConverged(), "Newton Solver didn't get converged.\n");
   // delete gmres;`
   // delete ntsolver;
   // delete gs_prec;

   // below are petsc solver

// #ifndef MFEM_USE_PETSC
// #error This function requires MFEM_USE_PETSC defined
// #endif
//    const char *petscrc_file="eulersteady";
//    std::cout << "EulerSolver::solveSteady() is called.\n";
//    MFEMInitializePetsc(NULL, NULL, petscrc_file, NULL);

//    PetscNonlinearSolver *pnewton_solver = new 
//                      PetscNonlinearSolver(fes->GetComm(), *res);
//    //mfem::PetscPreconditonerFactory * j_prec = new 
//    //                   PetscPreconditionerFactory();
//    pnewton_solver->SetPrintLevel(1); // print Newton iterations
//    pnewton_solver->SetRelTol(1e-6);
//    pnewton_solver->SetAbsTol(1e-6);
//    pnewton_solver->SetMaxIter(30);
//    //pnewton_solver->SetPreconditionerFactory(*j_prec);

//    mfem::Vector zero;
//    pnewton_solver->Mult(zero,*u);
//    std::cout << "\nConverged? :"<<pnewton_solver->GetConverged() << std::endl;
//    MFEM_ASSERT(pnewton_solver->GetConverged(), "Newton solver didn't converge.\n");
//    MFEMFinalizePetsc();

   // Use hypre solve to solve the problem
   std::cout << "steady solve is called.\n";

   prec.reset( new HypreBoomerAMG() );
   prec->SetPrintLevel(0);

   std::cout << "preconditioner is set.\n";
   solver.reset( new HypreGMRES(fes->GetComm()) );
   //solver.reset(new CGSolver());
   solver->SetTol(1e-10);
   solver->SetMaxIter(100);
   solver->SetPrintLevel(1);
   //solver->SetPreconditioner(*prec);
   std::cout << "Inner solver is set.\n";

   newton_solver.iterative_mode = true;
   newton_solver.SetSolver(*solver);
   newton_solver.SetOperator(*res);
   newton_solver.SetPrintLevel(1);
   newton_solver.SetRelTol(1e-10);
   newton_solver.SetAbsTol(1e-10);
   newton_solver.SetMaxIter(20);
   std::cout << "Newton solver is set.\n";
   std::cout << "Print the initial condition\n";
   //u->Print(std::cout,4);
   // std::cout << "try a different initial guess using Nonlinearform mult.\n";
   // mfem::Vector r2(fes->GlobalTrueVSize());
   // mfem::Vector x0(fes->GlobalTrueVSize());
   // x0 = 0.0;
   // res->Mult(x0, r2);
   // r2.Print();

   mfem::Vector b;
   newton_solver.Mult(b,  *u);

   // Save the final solution. This output can be viewed later using GLVis:
   // glvis -m unitGridTestMesh.msh -g adv-final.gf".
   int precision = 8;

   {
      ofstream osol("wedge-shock-final.gf");
      osol.precision(precision);
      u->Save(osol);
   }
   // write the solution to vtk file
   if (options["space-dis"]["basis-type"].get<string>() == "csbp")
   {
      ofstream sol_ofs("wedge_shock_cg.vtk");
      sol_ofs.precision(14);
      mesh->PrintVTK(sol_ofs, options["space-dis"]["degree"].get<int>() + 1);
      u->SaveVTK(sol_ofs, "Solution", options["space-dis"]["degree"].get<int>() + 1);
      sol_ofs.close();
      printSolution("final");
   }
   else if (options["space-dis"]["basis-type"].get<string>() == "dsbp")
   {
      ofstream sol_ofs("wedge_shock_dg.vtk");
      sol_ofs.precision(14);
      mesh->PrintVTK(sol_ofs, options["space-dis"]["degree"].get<int>() + 1);
      u->SaveVTK(sol_ofs, "Solution", options["space-dis"]["degree"].get<int>() + 1);
      sol_ofs.close();
      printSolution("final");
   }
      MFEM_VERIFY(newton_solver.GetConverged(), "Newton solver did not converge.");

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
   jac.PrintMatlab(std::cout);
   jac.Mult(*perturbation_vec, *jac_v);
   //std::cout << "Resuelts from GetGradient(x):\n";
   //jac_v->Save(std::cout);
   // check the difference norm
   jac_v->Add(-1.0, *res_plus);
   std::cout << "The difference norm is " << jac_v->Norml2()<<'\n';
}

} // namespace mach
