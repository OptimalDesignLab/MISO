#include <fstream>
#include <iostream>
#include "default_options.hpp"
#include "solver.hpp"
#include "sbp_fe.hpp"
#include "diag_mass_integ.hpp"
#include "evolver.hpp"

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
   *out << setw(3) << options << endl;
   constructMesh(move(smesh));
   int dim = mesh->Dimension();
   *out << "problem space dimension = " << dim << endl;

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
   for (int l = 0; l < options["mesh"]["refine"].get<int>(); l++)
   {
      mesh->UniformRefinement();
   }

   // Define the SBP elements and finite-element space; eventually, we will want
   // to have a case or if statement here for both CSBP and DSBP, and (?) standard FEM.
   // and here it is for first two
   if (options["space-dis"]["basis-type"].get<string>() == "csbp")
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
   fes.reset(new SpaceType(mesh.get(), fec.get(), num_state,
                   Ordering::byVDIM));
   u.reset(new GridFunType(fes.get()));
#ifdef MFEM_USE_MPI
   cout << "Number of finite element unknowns: " << fes->GlobalTrueVSize() << endl;
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
   res.reset(new NonlinearFormType(fes.get()));
   // Add integrators; this can be simplified if we template the entire class
   addVolumeIntegrators(alpha);
   auto &bcs = options["bcs"];
   bndry_marker.resize(bcs.size()); // need to set this before next method
   addBoundaryIntegrators(alpha);
   addInterfaceIntegrators(alpha);

   // This just lists the boundary markers for debugging purposes
   if (0==rank)
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
   mass_matrix.reset(mass->ParallelAssemble());
#else
   mass_matrix.reset(new MatrixType(mass->SpMat()));
#endif
   const string odes = options["time-dis"]["ode-solver"].get<string>();
   std::cout << "ode solver is " << odes << std::endl;
   if (odes == "RK1" || odes == "RK4")
   {
      evolver.reset(new NonlinearEvolver(*mass_matrix, *res, -1.0));
   }
   else
   {
      evolver.reset(new ImplicitNonlinearEvolver(*mass_matrix, *res, -1.0));
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
#ifndef MFEM_USE_PUMI
   if (smesh == nullptr)
   { // read in the serial mesh
      smesh.reset(new Mesh(options["mesh"]["file"].get<string>().c_str(), 1, 1));
   }
#endif

#ifdef MFEM_USE_MPI
   comm = MPI_COMM_WORLD; // TODO: how to pass communicator as an argument?
   MPI_Comm_rank(comm, &rank);
#ifdef MFEM_USE_PUMI // if using pumi mesh
   if (smesh != nullptr)
   {
      throw MachException("AbstractSolver::constructMesh(smesh)\n"
                          "\tdo not provide smesh when using PUMI!");
   }
   // problem with using these in loadMdsMesh
   std::cout << options["model-file"].get<string>().c_str() << std::endl;
   const char *model_file = options["model-file"].get<string>().c_str();
   const char *mesh_file = options["mesh"]["file"].get<string>().c_str();
   PCU_Comm_Init();
#ifdef MFEM_USE_SIMMETRIX
   Sim_readLicenseFile(0);
   gmi_sim_start();
   gmi_register_sim();
#endif
   gmi_register_mesh();
   pumi_mesh = apf::loadMdsMesh(options["model-file"].get<string>().c_str(),
                                options["mesh"]["file"].get<string>().c_str());
   int mesh_dim = pumi_mesh->getDimension();
   int nEle = pumi_mesh->count(mesh_dim);
   int ref_levels = (int)floor(log(10000. / nEle) / log(2.) / mesh_dim);
   // Perform Uniform refinement
   // if (ref_levels > 1)
   // {
   //    ma::Input* uniInput = ma::configureUniformRefine(pumi_mesh, ref_levels);
   //    ma::adapt(uniInput);
   // }
   pumi_mesh->verify();
   mesh.reset(new MeshType(comm, pumi_mesh));
   PCU_Comm_Free();
#ifdef MFEM_USE_SIMMETRIX
   gmi_sim_stop();
   Sim_unregisterAllKeys();
#endif
#else
   mesh.reset(new MeshType(comm, *smesh));
#endif //MFEM_USE_PUMI
#else
   mesh.reset(new MeshType(*smesh));
#endif //MFEM_USE_MPI
}

void AbstractSolver::setInitialCondition(
    void (*u_init)(const Vector &, Vector &))
{
   // TODO: Need to verify that this is ok for scalar fields
   VectorFunctionCoefficient u0(num_state, u_init);
   u->ProjectCoefficient(u0);
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
   GridFunType r(fes.get());
   double res_norm;
#ifdef MFEM_USE_MPI
   HypreParVector *U = u->GetTrueDofs();
   HypreParVector *R = r.GetTrueDofs();
   res->Mult(*U, *R);   
   double loc_norm = (*R)*(*R);
   MPI_Allreduce(&loc_norm, &res_norm, 1, MPI_DOUBLE, MPI_SUM, comm);
#else
   res->Mult(*u, r);
   res_norm = r*r;
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
   u->SaveVTK(sol_ofs, "Solution", refine);
   sol_ofs.close();
}

void AbstractSolver::printResidual(const std::string &file_name,
                                        int refine)
{
   GridFunType r(fes.get());
#ifdef MFEM_USE_MPI
   HypreParVector *U = u->GetTrueDofs();
   res->Mult(*U, r);
#else
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
   double t1, t2;
   if (0==rank)
   {
      t1 = MPI_Wtime();
   }
#ifdef MFEM_USE_PETSC   
   // Get the PetscSolver option 
   double abstol = options["petscsolver"]["abstol"].get<double>();
   double reltol = options["petscsolver"]["reltol"].get<double>();
   int maxiter = options["petscsolver"]["maxiter"].get<int>();
   int ptl = options["petscsolver"]["printlevel"].get<int>();

   solver.reset(new mfem::PetscLinearSolver(fes->GetComm(), "solver_", 0));
   prec.reset(new mfem::PetscPreconditioner(fes->GetComm(), "prec_"));
   dynamic_cast<mfem::PetscLinearSolver *>(solver.get())->SetPreconditioner(*prec);

   dynamic_cast<mfem::PetscSolver *>(solver.get())->SetAbsTol(abstol);
   dynamic_cast<mfem::PetscSolver *>(solver.get())->SetRelTol(reltol);
   dynamic_cast<mfem::PetscSolver *>(solver.get())->SetMaxIter(maxiter);
   dynamic_cast<mfem::PetscSolver *>(solver.get())->SetPrintLevel(ptl);
   std::cout << "Petsc Solver set.\n";
   //Get the newton solver options
   double nabstol = options["newton"]["abstol"].get<double>();
   double nreltol = options["newton"]["reltol"].get<double>();
   int nmaxiter = options["newton"]["maxiter"].get<int>();
   int nptl = options["newton"]["printlevel"].get<int>();
   newton_solver.reset(new mfem::NewtonSolver(fes->GetComm()));
   newton_solver->iterative_mode = true;
   newton_solver->SetSolver(*solver);
   newton_solver->SetOperator(*res);
   newton_solver->SetAbsTol(nabstol);
   newton_solver->SetRelTol(nreltol);
   newton_solver->SetMaxIter(nmaxiter);
   newton_solver->SetPrintLevel(nptl);
   std::cout << "Newton solver is set.\n";
   // Solve the nonlinear problem with r.h.s at 0
   mfem::Vector b;
   mfem::Vector u_true;
   u->GetTrueDofs(u_true);
   newton_solver->Mult(b, u_true);
   MFEM_VERIFY(newton_solver->GetConverged(), "Newton solver did not converge.");
   u->SetFromTrueDofs(u_true);
   if (0==rank)
   {
      t2 = MPI_Wtime();
      cout << "Time for solving nonlinear system is " << (t2 - t1) << endl;
   }
#else
   // Hypre solver section
   //prec.reset( new HypreBoomerAMG() );
   //prec->SetPrintLevel(0);
   std::cout << "ILU preconditioner is not available in Hypre. Running HypreGMRES"
               << " without preconditioner.\n";
   
   double tol = options["lin-solver"]["tol"].get<double>();
   int maxiter = options["lin-solver"]["maxiter"].get<int>();
   int ptl = options["lin-solver"]["printlevel"].get<int>();
   solver.reset( new HypreGMRES(fes->GetComm()) );
   dynamic_cast<mfem::HypreGMRES*> (solver.get())->SetTol(tol);
   dynamic_cast<mfem::HypreGMRES*> (solver.get())->SetMaxIter(maxiter);
   dynamic_cast<mfem::HypreGMRES*> (solver.get())->SetPrintLevel(ptl);

   //solver->SetPreconditioner(*prec);
   double nabstol = options["newton"]["abstol"].get<double>();
   double nreltol = options["newton"]["reltol"].get<double>();
   int nmaxiter = options["newton"]["maxiter"].get<int>();
   int nptl = options["newton"]["printlevel"].get<int>();
   newton_solver.reset(new mfem::NewtonSolver(fes->GetComm()));
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
   newton_solver->Mult(b,  *u);
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
      HypreParVector *U = u->GetTrueDofs();
      ode_solver->Step(*U, t, dt_real);
      *u = *U;
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

   std::unique_ptr<GridFunType> res_plus;
   std::unique_ptr<GridFunType> res_minus;
   res_plus.reset(new GridFunType(fes.get()));
   res_minus.reset(new GridFunType(fes.get()));

   res->Mult(*u_plus, *res_plus);
   res->Mult(*u_minus, *res_minus);

   res_plus->Add(-1.0, *res_minus);
   res_plus->Set(1 / (2 * delta), *res_plus);

   // result from GetGradient(x)
   std::unique_ptr<GridFunType> jac_v;
   jac_v.reset(new GridFunType(fes.get()));
   mfem::Operator &jac = res->GetGradient(*u);
   jac.Mult(*perturbation_vec, *jac_v);
   // check the difference norm
   jac_v->Add(-1.0, *res_plus);
   std::cout << "The difference norm is " << jac_v->Norml2() << '\n';
}

} // namespace mach
