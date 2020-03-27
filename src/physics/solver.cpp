#include <fstream>
#include <iostream>

#include "default_options.hpp"
#include "solver.hpp"
#include "centgridfunc.hpp"
#include "sbp_fe.hpp"
#include "evolver.hpp"
#include "diag_mass_integ.hpp"
#include "solver.hpp"
#include "galer_diff.hpp"

using namespace std;
using namespace mfem;

namespace mach
{

adept::Stack AbstractSolver::diff_stack;

AbstractSolver::AbstractSolver(const string &opt_file_name,
                               unique_ptr<Mesh> smesh)
{
// Set the options; the defaults are overwritten by the values in the file
// using the merge_patch method
#ifdef MFEM_USE_MPI
   comm = MPI_COMM_WORLD; // TODO: how to pass as an argument?
   MPI_Comm_rank(comm, &rank);
#else
   rank = 0; // serial case
#endif
   out = getOutStream(rank);
   options = default_options;
   nlohmann::json file_options;
   ifstream options_file(opt_file_name);
   options_file >> file_options;
   options.merge_patch(file_options);
   //*out << setw(3) << options << endl;

   // Construct Mesh
   constructMesh(move(smesh));
   int dim = mesh->Dimension();

   // Define the ODE solver used for time integration (possibly not used)
   ode_solver = NULL;
   *out << "ode-solver type = "
        << options["time-dis"]["ode-solver"].get<string>() << endl;
   if (options["time-dis"]["ode-solver"].get<string>() == "RK1")
   {
      ode_solver.reset(new ForwardEulerSolver);
   }
   else if (options["time-dis"]["ode-solver"].get<string>() == "RK4")
   {
      ode_solver.reset(new RK4Solver);
   }
   else if (options["time-dis"]["ode-solver"].get<string>() == "MIDPOINT")
   {
      ode_solver.reset(new ImplicitMidpointSolver);
   }
   else
   {
      throw MachException("Unknown ODE solver type " +
                          options["time-dis"]["ode-solver"].get<string>());
      // TODO: parallel exit
   }

   // Refine the mesh here, or have a separate member function?
   // for (int l = 0; l < options["mesh"]["refine"].get<int>(); l++)
   // {
   //    mesh->UniformRefinement();
   // }

   // Define the SBP elements and finite-element space; eventually, we will want
   // to have a case or if statement here for both CSBP and DSBP, and (?) standard FEM.
   // and here it is for first two
   if (options["GD"]["degree"].get<int>() >= 0)
   {
      fec.reset(new DSBPCollection(options["space-dis"]["degree"].get<int>(),
                                   dim));
   }
   else if (options["space-dis"]["basis-type"].get<string>() == "csbp")
   {
      fec.reset(new SBPCollection(options["space-dis"]["degree"].get<int>(),
                                  dim));
   }
   else if (options["space-dis"]["basis-type"].get<string>() == "dsbp")
   {
      fec.reset(new DSBPCollection(options["space-dis"]["degree"].get<int>(),
                                   dim));
   }
}

void AbstractSolver::initDerived()
{
   // define the number of states, the fes, and the state grid function
   num_state = this->getNumState(); // <--- this is a virtual fun
   *out << "Num states = " << num_state << endl;
   if (options["GD"]["degree"].get<int>() >= 0)
   {
      int gd_degree = options["GD"]["degree"].get<int>();
      fes.reset(new GalerkinDifference(comm, mesh.get(), fec.get(), num_state,
                                       Ordering::byVDIM, gd_degree, pumi_mesh));
      dynamic_cast<GalerkinDifference *>(fes.get())->BuildGDProlongation();
      uc.reset(new CentGridFunction(fes.get()));
      u.reset(new GridFunType(fes.get()));
      cout << "GD space is set, uc size is " << uc->Size() << ", u size is " << u->Size() << '\n';
   }
else
   {
      fes.reset(new SpaceType(mesh.get(), fec.get(), num_state,
                              Ordering::byVDIM));
      u.reset(new GridFunType(fes.get()));
   }

#ifdef MFEM_USE_MPI
   cout << "Number of finite element unknowns: " << fes->GetTrueVSize() << endl;
   //fes->GlobalTrueVSize()
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
   double alpha = 1.0;
   cout << "before set res.\n";
   res.reset(new NonlinearFormType(fes.get()));
   if (0 == rank)
   {
      std::cout << "In rank " << rank << ": fes Vsize " << fes->GetVSize() << ". fes TrueVsize " << fes->GetTrueVSize();
      std::cout << ". fes ndofs is " << fes->GetNDofs() << ". res size " << res->Width() << ". u size " << u->Size();
      std::cout << ". uc size is " << uc->Size() << '\n';
   }
   MPI_Barrier(comm);

   // Add integrators; this can be simplified if we template the entire class
   addVolumeIntegrators(alpha);
   auto &bcs = options["bcs"];
   bndry_marker.resize(bcs.size()); // need to set this before next method
   addBoundaryIntegrators(alpha);
   addInterfaceIntegrators(alpha);

   // This just lists the boundary markers for debugging purposes
   if (0 == rank)
   {
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

   // define the time-dependent operator
#ifdef MFEM_USE_MPI
   // The parallel bilinear forms return a pointer that this solver owns
   //mass_matrix.reset(mass->ParallelAssemble());
   mass_matrix.reset(new MatrixType(mass->SpMat()));
#else
   mass_matrix.reset(new MatrixType(mass->SpMat()));
#endif
   const string odes = options["time-dis"]["ode-solver"].get<string>();
   std::cout << "ode solver is " << odes << std::endl;
   if (odes == "RK1" || odes == "RK4")
   {
      //evolver.reset(new NonlinearEvolver(*mass_matrix, *res, -1.0));
   }
   else
   {
      //evolver.reset(new ImplicitNonlinearEvolver(*mass_matrix, *res, -1.0));
   }

   // add the output functional QoIs
   auto &fun = options["outputs"];
   output_bndry_marker.resize(fun.size());
   addOutputs(); // virtual function
}

AbstractSolver::~AbstractSolver()
{
   *out << "Deleting Abstract Solver..." << endl;
}

void AbstractSolver::constructMesh(unique_ptr<Mesh> smesh)
{
   cout << "In construct Mesh:\n";
#ifdef MFEM_USE_MPI
   comm = MPI_COMM_WORLD;
   MPI_Comm_rank(comm, &rank);
#ifdef MFEM_USE_PUMI
   if (smesh != nullptr)
   {
      throw MachException("AbstractSolver::constructMesh(smesh)\n"
                          "\tdo not provide smesh when using PUMI!");
   }
   // problem with using these in loadMdsMesh
   cout << "Construct pumi mesh.\n";
   std::string model_file = options["model-file"].get<string>().c_str();
   std::string mesh_file = options["mesh"]["file"].get<string>().c_str();
   std::cout << "model file " << model_file.c_str() << '\n';
   std::cout << "mesh file " << mesh_file.c_str() << '\n';
   pumi_mesh = apf::loadMdsMesh(model_file.c_str(), mesh_file.c_str());
   cout << "pumi mesh is constructed from file.\n";
   int mesh_dim = pumi_mesh->getDimension();
   int nEle = pumi_mesh->count(mesh_dim);
   //int ref_levels = (int)floor(log(10000. / nEle) / log(2.) / mesh_dim);
   pumi_mesh->verify();
   //mesh.reset(new MeshType(comm, pumi_mesh));
   // currently do this in serial in the MPI configuration because of gd and gridfunction is not
   //    complete
   mesh.reset(new MeshType(pumi_mesh, 1, 0));
   ofstream savemesh("annulus_fine.mesh");
   ofstream savevtk("annulus_fine.vtk");
   mesh->Print(savemesh);
   mesh->PrintVTK(savevtk);
   savemesh.close();
   savevtk.close();
   cout << "PumiMesh is constructed from pumi_mesh.\n";
#else
   mesh.reset(new MeshType(comm, *smesh));
#endif // end of MFEM_USE_PUMI
#else
   mesh.reset(new MeshType(*smesh));
#endif // end of MFEM_USE_MPI
}

void AbstractSolver::setInitialCondition(
    void (*u_init)(const Vector &, Vector &))
{
   // TODO: Need to verify that this is ok for scalar fields
   VectorFunctionCoefficient u0(num_state, u_init);
   u->ProjectCoefficient(u0);
   uc->ProjectCoefficient(u0);

   // cout << "\nCheck exact solution:\n";
   // u->Print(cout, num_state);
   // cout << "\n\nCheck center values:\n";
   // uc->Print(cout, num_state);

   Vector u_test(fes->GetVSize());
   fes->GetProlongationMatrix()->Mult(*uc, u_test);
   // cout << "\n\nCheck the prolongated results:\n";
   // u_test.Print(cout,4);
   u_test -= *u;
   cout << "After projection, the difference norm is " << u_test.Norml2() << '\n';
   // ofstream u_write("u.txt");
   // ofstream uc_write("uc.txt");
   // u->Print(u_write, 1);
   // uc->Print(uc_write,1);
   // u_write.close();
   // uc_write.close();
   // DenseMatrix vals;
   // Vector uj;
   // for (int i = 0; i < fes->GetNE(); i++)
   // {
   //    const FiniteElement *fe = fes->GetFE(i);
   //    const IntegrationRule *ir = &(fe->GetNodes());
   //    ElementTransformation *T = fes->GetElementTransformation(i);
   //    u->GetVectorValues(*T, *ir, vals);
   //    for (int j = 0; j < ir->GetNPoints(); j++)
   //    {
   //       vals.GetColumnReference(j, uj);
   //       cout << "uj = " << uj(0) << ", " << uj(1) << ", " << uj(2) << ", " << uj(3) << endl;
   //    }
   // }
}

void AbstractSolver::setInitialCondition(const Vector &uic)
{
   // TODO: Need to verify that this is ok for scalar fields
   VectorConstantCoefficient u0(uic);
   u->ProjectCoefficient(u0);
}

double AbstractSolver::calcL2Error(
    void (*u_exact)(const Vector &, Vector &), int entry)
{
   // TODO: need to generalize to parallel
   VectorFunctionCoefficient exsol(num_state, u_exact);
   //return u->ComputeL2Error(ue);

   double loc_norm = 0.0;
   const FiniteElement *fe;
   ElementTransformation *T;
   DenseMatrix vals, exact_vals;
   Vector loc_errs;

   if (entry < 0)
   {
      fes->GetProlongationMatrix()->Mult(*uc, *u);
      // sum up the L2 error over all states
      for (int i = 0; i < fes->GetNE(); i++)
      {
         fe = fes->GetFE(i);
         const IntegrationRule *ir = &(fe->GetNodes());
         T = fes->GetElementTransformation(i);
         u->GetVectorValues(*T, *ir, vals);
         exsol.Eval(exact_vals, *T, *ir);
         vals -= exact_vals;
         loc_errs.SetSize(vals.Width());
         vals.Norm2(loc_errs);
         for (int j = 0; j < ir->GetNPoints(); j++)
         {
            const IntegrationPoint &ip = ir->IntPoint(j);
            T->SetIntPoint(&ip);
            loc_norm += ip.weight * T->Weight() * (loc_errs(j) * loc_errs(j));
         }
      }
   }
   else
   {
      // calculate the L2 error for component index `entry`
      fes->GetProlongationMatrix()->Mult(*uc,*u);
      for (int i = 0; i < fes->GetNE(); i++)
      {
         fe = fes->GetFE(i);
         const IntegrationRule *ir = &(fe->GetNodes());
         T = fes->GetElementTransformation(i);
         u->GetVectorValues(*T, *ir, vals);
         exsol.Eval(exact_vals, *T, *ir);
         vals -= exact_vals;
         loc_errs.SetSize(vals.Width());
         vals.GetRow(entry, loc_errs);
         for (int j = 0; j < ir->GetNPoints(); j++)
         {
            const IntegrationPoint &ip = ir->IntPoint(j);
            T->SetIntPoint(&ip);
            loc_norm += ip.weight * T->Weight() * (loc_errs(j) * loc_errs(j));
         }
      }
   }
   double norm;
#ifdef MFEM_USE_MPI
   MPI_Allreduce(&loc_norm, &norm, 1, MPI_DOUBLE, MPI_SUM, comm);
#else
   norm = loc_norm;
#endif
   if (norm < 0.0) // This was copied from mfem...should not happen for us
   {
      return -sqrt(-norm);
   }
   return sqrt(norm);
}

double AbstractSolver::calcResidualNorm()
{
   //GridFunType r(fes.get());
   CentGridFunction r(fes.get());
   double res_norm;
#ifdef MFEM_USE_MPI
   // HypreParVector *U = u->GetTrueDofs();
   // HypreParVector *R = r.GetTrueDofs();
   // cout << "U size is " << U->Size() << '\n';
   // cout << "R size is " << R->Size() << '\n';
   // res->Mult(*U, *R);
   // double loc_norm = (*R) * (*R);
   // MPI_Allreduce(&loc_norm, &res_norm, 1, MPI_DOUBLE, MPI_SUM, comm);
   res->Mult(*uc, r);
   // cout << "Residual now is:\n";
   // r.Print(cout, 4);
   res_norm = r * r;
#else
   res->Mult(*u, r);
   res_norm = r * r;
#endif
   res_norm = sqrt(res_norm);
   return res_norm;
}

double AbstractSolver::calcStepSize(double cfl) const
{
   throw MachException("AbstractSolver::calcStepSize(cfl)\n"
                       "\tis not implemented for this class!");
}

void AbstractSolver::printSolution(const std::string &file_name,
                                   int refine)
{
   // TODO: These mfem functions do not appear to be parallelized
   ofstream sol_ofs(file_name + ".vtk");
   sol_ofs.precision(14);
   if (refine == -1)
   {
      refine = options["space-dis"]["degree"].get<int>() + 1;
   }
   mesh->PrintVTK(sol_ofs, refine);
   fes->GetProlongationMatrix()->Mult(*uc, *u);
   u->SaveVTK(sol_ofs, "Solution", refine);
   sol_ofs.close();
}

void AbstractSolver::printResidual(const std::string &file_name,
                                   int refine)
{

   GridFunType r(fes.get());
#ifdef MFEM_USE_MPI
   // HypreParVector *U = u->GetTrueDofs();
   // res->Mult(*U, r);
   cout << "Print residual is called, in MPI, uc is feed into Mult, size: " << uc->Size() << '\n';
   res->Mult(*uc, r);
#else
   cout << "Print residual is called, not in MPI, uc is feed into Mult, size: " << uc->Size() << '\n';
   res->Mult(*u, r);
#endif
   // TODO: These mfem functions do not appear to be parallelized
   ofstream res_ofs(file_name + ".vtk");
   res_ofs.precision(14);
   if (refine == -1)
   {
      refine = options["space-dis"]["degree"].get<int>() + 1;
   }
   mesh->PrintVTK(res_ofs, refine);
   r.SaveVTK(res_ofs, "Residual", refine);
   res_ofs.close();
}

void AbstractSolver::solveForState()
{
   if (options["steady"].get<bool>() == true)
   {
      solveSteady();
   }
   else
   {
      solveUnsteady();
   }
}

void AbstractSolver::solveSteady()
{
#ifdef MFEM_USE_MPI
#ifdef MFEM_USE_PETSC
   // Currently need to use serial GMRES solver
   double abstol = options["petscsolver"]["abstol"].get<double>();
   double reltol = options["petscsolver"]["reltol"].get<double>();
   int maxiter = options["petscsolver"]["maxiter"].get<int>();
   int ptl = options["petscsolver"]["printlevel"].get<int>();
   // solver.reset(new GMRESSolver());
   // prec.reset(new GSSmoother(1,1));
   // dynamic_cast<mfem::IterativeSolver *>(solver.get())->SetAbsTol(abstol);
   // dynamic_cast<mfem::IterativeSolver *>(solver.get())->SetRelTol(reltol);
   // dynamic_cast<mfem::IterativeSolver *>(solver.get())->SetMaxIter(maxiter);
   // dynamic_cast<mfem::IterativeSolver *>(solver.get())->SetPrintLevel(ptl);
   // dynamic_cast<mfem::IterativeSolver *>(solver.get())->SetPreconditioner(*prec);

   //solver.reset(new KLUSolver());
   solver.reset(new UMFPackSolver());
   dynamic_cast<UMFPackSolver*>(solver.get())->Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
   dynamic_cast<UMFPackSolver*>(solver.get())->SetPrintLevel(1);
   double nabstol = options["newton"]["abstol"].get<double>();
   double nreltol = options["newton"]["reltol"].get<double>();
   int nmaxiter = options["newton"]["maxiter"].get<int>();
   int nptl = options["newton"]["printlevel"].get<int>();
   newton_solver.reset(new mfem::NewtonSolver());
   newton_solver->iterative_mode = true;
   newton_solver->SetSolver(*solver);
   newton_solver->SetOperator(*res);
   newton_solver->SetAbsTol(nabstol);
   newton_solver->SetRelTol(nreltol);
   newton_solver->SetMaxIter(nmaxiter);
   newton_solver->SetPrintLevel(nptl);
   std::cout << "Newton solver is set.\n";
   mfem::Vector b;
   newton_solver->Mult(b, *uc);
   MFEM_VERIFY(newton_solver->GetConverged(), "Newton solver did not converge.");

   // // Get the PetscSolver option
   // double abstol = options["petscsolver"]["abstol"].get<double>();
   // double reltol = options["petscsolver"]["reltol"].get<double>();
   // int maxiter = options["petscsolver"]["maxiter"].get<int>();
   // int ptl = options["petscsolver"]["printlevel"].get<int>();

   // solver.reset(new mfem::PetscLinearSolver(dynamic_cast<GalerkinDifference*>(fes.get())->GetComm(), "solver_", 0));
   // prec.reset(new mfem::PetscPreconditioner(dynamic_cast<GalerkinDifference*>(fes.get())->GetComm(), "prec_"));
   // dynamic_cast<mfem::PetscLinearSolver *>(solver.get())->SetPreconditioner(*prec);

   // dynamic_cast<mfem::PetscSolver *>(solver.get())->SetAbsTol(abstol);
   // dynamic_cast<mfem::PetscSolver *>(solver.get())->SetRelTol(reltol);
   // dynamic_cast<mfem::PetscSolver *>(solver.get())->SetMaxIter(maxiter);
   // dynamic_cast<mfem::PetscSolver *>(solver.get())->SetPrintLevel(ptl);
   // std::cout << "Petsc Solver set.\n";
   // //Get the newton solver options
   // double nabstol = options["newton"]["abstol"].get<double>();
   // double nreltol = options["newton"]["reltol"].get<double>();
   // int nmaxiter = options["newton"]["maxiter"].get<int>();
   // int nptl = options["newton"]["printlevel"].get<int>();
   // newton_solver.reset(new mfem::NewtonSolver(dynamic_cast<GalerkinDifference*>(fes.get())->GetComm())  );
   // newton_solver->iterative_mode = true;
   // newton_solver->SetSolver(*solver);
   // newton_solver->SetOperator(*res);
   // newton_solver->SetAbsTol(nabstol);
   // newton_solver->SetRelTol(nreltol);
   // newton_solver->SetMaxIter(nmaxiter);
   // newton_solver->SetPrintLevel(nptl);
   // std::cout << "Newton solver is set.\n";
   // // Solve the nonlinear problem with r.h.s at 0
   // mfem::Vector b;
   // mfem::Vector u_true;
   // u->GetTrueDofs(u_true);
   // newton_solver->Mult(b, *uc);
   // MFEM_VERIFY(newton_solver->GetConverged(), "Newton solver did not converge.");
   //u->SetFromTrueDofs(u_true);
#else
   // Hypre solver section
   cout << "HypreGMRESSolver with HypreEuclid preconditioner.\n";
   prec.reset(new HypreEuclid(dynamic_cast<GalerkinDifference*>(fes.get())->GetComm()));
   double tol = options["lin-solver"]["tol"].get<double>();
   int maxiter = options["lin-solver"]["maxiter"].get<int>();
   int ptl = options["lin-solver"]["printlevel"].get<int>();
   solver.reset(new HypreGMRES(dynamic_cast<GalerkinDifference*>(fes.get())->GetComm()));
   dynamic_cast<mfem::HypreGMRES *>(solver.get())->SetTol(tol);
   dynamic_cast<mfem::HypreGMRES *>(solver.get())->SetMaxIter(maxiter);
   dynamic_cast<mfem::HypreGMRES *>(solver.get())->SetPrintLevel(ptl);
   dynamic_cast<mfem::HypreGMRES *>(solver.get())->SetPreconditioner(* dynamic_cast<mfem::HypreSolver*>(prec.get()));

   double nabstol = options["newton"]["abstol"].get<double>();
   double nreltol = options["newton"]["reltol"].get<double>();
   int nmaxiter = options["newton"]["maxiter"].get<int>();
   int nptl = options["newton"]["printlevel"].get<int>();
   newton_solver.reset(new mfem::NewtonSolver(dynamic_cast<GalerkinDifference*>(fes.get())->GetComm()));
   newton_solver->iterative_mode = true;
   newton_solver->SetSolver(*solver);
   newton_solver->SetOperator(*res);
   newton_solver->SetPrintLevel(nptl);
   newton_solver->SetRelTol(nreltol);
   newton_solver->SetAbsTol(nabstol);
   newton_solver->SetMaxIter(nmaxiter);

   // Solve the nonlinear problem with r.h.s at 0
   mfem::Vector b;
   mfem::Vector u_true;
   u->GetTrueDofs(u_true);
   newton_solver->Mult(b, *uc);
   MFEM_VERIFY(newton_solver->GetConverged(), "Newton solver did not converge.");
   //u->SetFromTrueDofs(u_true);
#endif
#else
   // serial
   // linear solver
   double tol = options["lin-solver"]["tol"].get<double>();
   int maxiter = options["lin-solver"]["maxiter"].get<int>();
   int ptl = options["lin-solver"]["printlevel"].get<int>();
   prec.reset(new mfem::GSSmoother());
   solver.reset(new mfem::SLISolver());
   dynamic_cast<mfem::SLISolver *>(solver.get())->SetRelTol(tol);
   dynamic_cast<mfem::SLISolver *>(solver.get())->SetAbsTol(tol);
   dynamic_cast<mfem::SLISolver *>(solver.get())->SetMaxIter(maxiter);
   dynamic_cast<mfem::SLISolver *>(solver.get())->SetPrintLevel(ptl);
   //dynamic_cast<mfem::GMRESSolver *>(solver.get())->SetPreconditioner(*prec);
   dynamic_cast<mfem::SLISolver *>(solver.get())->iterative_mode = false;

   // newton solver
   double nabstol = options["newton"]["abstol"].get<double>();
   double nreltol = options["newton"]["reltol"].get<double>();
   int nmaxiter = options["newton"]["maxiter"].get<int>();
   int nptl = options["newton"]["printlevel"].get<int>();
   newton_solver.reset(new mfem::NewtonSolver());
   newton_solver->iterative_mode = true;
   newton_solver->SetSolver(*solver);
   newton_solver->SetOperator(*res);
   newton_solver->SetPrintLevel(nptl);
   newton_solver->SetRelTol(nreltol);
   newton_solver->SetAbsTol(nabstol);
   newton_solver->SetMaxIter(nmaxiter);
   // Solve the nonlinear problem with r.h.s at 0
   mfem::Vector b;
   mfem::Vector u_true;
   u->GetTrueDofs(u_true);
   newton_solver->Mult(b, *u);
   MFEM_VERIFY(newton_solver->GetConverged(), "Newton solver did not converge.");
   u->SetFromTrueDofs(u_true);
#endif
}

void AbstractSolver::solveUnsteady()
{
   // TODO: This is not general enough.

   double t = 0.0;
   evolver->SetTime(t);
   ode_solver->Init(*evolver);

   // output the mesh and initial condition
   // TODO: need to swtich to vtk for SBP
   int precision = 8;
   {
      ofstream omesh("initial.mesh");
      omesh.precision(precision);
      mesh->Print(omesh);
      ofstream osol("initial-sol.gf");
      osol.precision(precision);
      u->Save(osol);
   }

   printSolution("init");

   bool done = false;
   double t_final = options["time-dis"]["t-final"].get<double>();
   std::cout << "t_final is " << t_final << '\n';
   double dt = options["time-dis"]["dt"].get<double>();
   bool calc_dt = options["time-dis"]["const-cfl"].get<bool>();
   for (int ti = 0; !done;)
   {
      if (calc_dt)
      {
         dt = calcStepSize(options["time-dis"]["cfl"].get<double>());
      }
      double dt_real = min(dt, t_final - t);
      if (ti % 10 == 0)
      {
         *out << "iter " << ti << ": time = " << t << ": dt = " << dt_real
              << " (" << round(10 * t / t_final) << "% complete)" << endl;
      }
#ifdef MFEM_USE_MPI
      // HypreParVector *U = u->GetTrueDofs();
      // ode_solver->Step(*U, t, dt_real);
      // *u = *U;
      ode_solver->Step(*u, t, dt_real);
#else
      ode_solver->Step(*u, t, dt_real);
#endif
      ti++;
      done = (t >= t_final - 1e-8 * dt);
      //std::cout << "t_final is " << t_final << ", done is " << done << std::endl;
      /*       if (done || ti % vis_steps == 0)
      {
         cout << "time step: " << ti << ", time: " << t << endl;

         if (visualization)
         {
            sout << "solution\n" << mesh << u << flush;
         }

         if (visit)
         {
            dc->SetCycle(ti);
            dc->SetTime(t);
            dc->Save();
         }
      } */
   }

   // Save the final solution. This output can be viewed later using GLVis:
   // glvis -m unitGridTestMesh.msh -g adv-final.gf".
   {
      ofstream osol("final.gf");
      osol.precision(precision);
      u->Save(osol);
   }
   // write the solution to vtk file
   if (options["space-dis"]["basis-type"].get<string>() == "csbp")
   {
      ofstream sol_ofs("final_cg.vtk");
      sol_ofs.precision(14);
      mesh->PrintVTK(sol_ofs, options["space-dis"]["degree"].get<int>() + 1);
      u->SaveVTK(sol_ofs, "Solution", options["space-dis"]["degree"].get<int>() + 1);
      sol_ofs.close();
      printSolution("final");
   }
   else if (options["space-dis"]["basis-type"].get<string>() == "dsbp")
   {
      ofstream sol_ofs("final_dg.vtk");
      sol_ofs.precision(14);
      mesh->PrintVTK(sol_ofs, options["space-dis"]["degree"].get<int>() + 1);
      u->SaveVTK(sol_ofs, "Solution", options["space-dis"]["degree"].get<int>() + 1);
      sol_ofs.close();
      printSolution("final");
   }
   // TODO: These mfem functions do not appear to be parallelized
}

double AbstractSolver::calcOutput(const std::string &fun)
{
   try
   {
      if (output.find(fun) == output.end())
      {
         cout << "Did not find " << fun << " in output map?" << endl;
      }
      return output.at(fun).GetEnergy(*u);
   }
   catch (const std::out_of_range &exception)
   {
      std::cerr << exception.what() << endl;
   }
}

void AbstractSolver::jacobianCheck()
{
   // initialize the variables
   const double delta = 1e-5;
   std::unique_ptr<CentGridFunction> u_plus;
   std::unique_ptr<CentGridFunction> u_minus;
   std::unique_ptr<CentGridFunction> perturbation_vec;

   perturbation_vec.reset(new CentGridFunction(fes.get()));
   VectorFunctionCoefficient up(num_state, perturb_fun);
   perturbation_vec->ProjectCoefficient(up);
   u_plus.reset(new CentGridFunction(fes.get()));
   u_minus.reset(new CentGridFunction(fes.get()));

   // set uplus and uminus to the current state
   *u_plus = 0.0; *u_minus = 0.0;
   u_plus->Add(1.0, *uc);
   u_minus->Add(1.0, *uc);
   u_plus->Add(delta, *perturbation_vec);
   u_minus->Add(-delta, *perturbation_vec);

   std::unique_ptr<CentGridFunction> res_plus;
   std::unique_ptr<CentGridFunction> res_minus;
   res_plus.reset(new CentGridFunction(fes.get()));
   res_minus.reset(new CentGridFunction(fes.get()));
   
   res->Mult(*u_plus, *res_plus);
   res->Mult(*u_minus, *res_minus);

   res_plus->Add(-1.0, *res_minus);
   res_plus->Set(1 / (2 * delta), *res_plus);
   // result from GetGradient(x)
   std::unique_ptr<CentGridFunction> jac_v;
   jac_v.reset(new CentGridFunction(fes.get()));
   mfem::Operator &jac = res->GetGradient(*uc);
   jac.Mult(*perturbation_vec, *jac_v);
   // check the difference norm
   jac_v->Add(-1.0, *res_plus);
   std::cout << "The (jav * v) difference norm is " << jac_v->Norml2() << '\n';
}

} // namespace mach
