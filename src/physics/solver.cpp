#include <fstream>
#include <iostream>
#include <time.h>

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
   *out << setw(3) << options << endl;

   // Construct Mesh
   constructMesh(move(smesh));
   for (int l = 0; l < options["mesh"]["refine"].get<int>(); l++)
   {
      mesh->UniformRefinement();
   }
   int dim = mesh->Dimension();
   *out << "Number of elements: " << mesh->GetNE() << '\n';
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
   // for (int l = 0; l < options["mesh"]["refine"].get<int>(); l++)
   // {
   //    mesh->UniformRefinement();
   // }

   // Define the SBP elements and finite-element space; eventually, we will want
   // to have a case or if statement here for both CSBP and DSBP, and (?) standard FEM.
   // and here it is for first two
   if (options["space-dis"]["GD"].get<bool>() == true || 
       options["space-dis"]["basis-type"].get<string>() == "dsbp")
   {
      fec.reset(new DSBPCollection(options["space-dis"]["degree"].get<int>(),
                                   dim));
   }
   else if (options["space-dis"]["basis-type"].get<string>() == "csbp")
   {
      fec.reset(new SBPCollection(options["space-dis"]["degree"].get<int>(),
                                  dim));
   }
}

void AbstractSolver::initDerived()
{
   // define the number of states, the fes, and the state grid function
   num_state = this->getNumState(); // <--- this is a virtual fun
   *out << "Num states = " << num_state << endl;
   if (options["space-dis"]["GD"].get<bool>() == true ||
       options["space-dis"]["basis-type"].get<string>() == "dsbp")
   {
      int gd_degree = options["space-dis"]["GD-degree"].get<int>();
      mesh->ElementToElementTable();
      fes.reset(new GalerkinDifference(mesh.get(), fec.get(), num_state,
                                       Ordering::byVDIM, gd_degree));
      fes_normal.reset(new SpaceType(mesh.get(), fec.get(), num_state,
                              Ordering::byVDIM));
      uc.reset(new CentGridFunction(fes.get()));
      u.reset(new GridFunType(fes.get()));
   }
   else
   {
      fes.reset(new SpaceType(mesh.get(), fec.get(), num_state,
                              Ordering::byVDIM));
      u.reset(new GridFunType(fes.get()));
   }

#ifdef MFEM_USE_MPI
   *out << "Number of finite element unknowns: " << fes->GlobalTrueVSize() << endl;
#else
   *out << "Number of finite element unknowns: "
        << fes->GetTrueVSize() << endl;
#endif

   // set up the mass matrix
   mass.reset(new BilinearFormType(fes_normal.get()));
   mass->AddDomainIntegrator(new DiagMassIntegrator(num_state));
   mass->Assemble();
   mass->Finalize();

   // set nonlinear mass matrix form
   nonlinear_mass.reset(new NonlinearFormType(fes.get()));

   // set up the spatial semi-linear form
   double alpha = 1.0;
   res.reset(new NonlinearFormType(fes.get()));
   std::cout << "In rank " << rank << ": fes Vsize " << fes->GetVSize() << ". fes TrueVsize " << fes->GetTrueVSize();
   std::cout << ". fes ndofs is " << fes->GetNDofs() << ". res size " << res->Width() << ". u size " << u->Size();
   std::cout << ". uc size is " << uc->Size() << '\n';

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
   MatrixType *cp = dynamic_cast<GalerkinDifference*>(fes.get())->GetCP();
   MatrixType *p = RAP(*cp, *mass_matrix, *cp);
   mass_matrix_gd.reset(new MatrixType(*p));
#endif
   const string odes = options["time-dis"]["ode-solver"].get<string>();
   if (odes == "RK1" || odes == "RK4")
   {
      evolver.reset(new NonlinearEvolver(*mass_matrix, *res, -1.0));
   }
   else
   {
      evolver.reset(new ImplicitNonlinearMassEvolver(*nonlinear_mass, *res, -1.0));
      //evolver.reset(new ImplicitNonlinearEvolver(*mass_matrix, *res, -1.0));
      //evolver.reset(new ImplicitNonlinearEvolver(*mass_matrix_gd, *res, -1.0));
   }

   // add the output functional QoIs
   auto &fun = options["outputs"];
   using json_iter = nlohmann::json::iterator;
   int num_bndry_outputs = 0;
   for (json_iter it = fun.begin(); it != fun.end(); ++it) {
      if (it->is_array()) ++num_bndry_outputs;
   }
   output_bndry_marker.resize(num_bndry_outputs);
   addOutputs(); // virtual function
}

AbstractSolver::~AbstractSolver()
{
   *out << "Deleting Abstract Solver..." << endl;
}

void AbstractSolver::constructMesh(unique_ptr<Mesh> smesh)
{
   cout << "In construct Mesh:\n";
#ifndef MFEM_USE_PUMI
   if (smesh == nullptr)
   { // read in the serial mesh
      smesh.reset(new Mesh(options["mesh"]["file"].get<string>().c_str(), 1, 1));
   }
#endif
#ifdef MFEM_USE_MPI
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
   std::cout << "model file " << model_file << '\n';
   std::cout << "mesh file " << mesh_file << '\n';
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
   // mesh.reset(new MeshType("annulus_fine.mesh", 1, 0, 1));
   // std::cout << "Mesh is constructed.\n";
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

   ofstream initial("initial_condition.vtk");
   initial.precision(14);
   mesh->PrintVTK(initial, 0);
   u->SaveVTK(initial, "initial", 0);
   initial.close();
   
   uc->ProjectCoefficient(u0);
   GridFunType u_test(fes.get());
   fes->GetProlongationMatrix()->Mult(*uc, u_test);
   ofstream projection("initial_projection.vtk");
   projection.precision(14);
   mesh->PrintVTK(projection, 0);
   u_test.SaveVTK(projection, "projection", 0);
   projection.close();


   
   u_test -= *u;
   cout << "After projection, the difference norm is " << u_test.Norml2() << '\n';
   ofstream sol_ofs("projection_error.vtk");
   sol_ofs.precision(14);
   mesh->PrintVTK(sol_ofs, 0);
   u_test.SaveVTK(sol_ofs, "project_error", 0);
   sol_ofs.close();

   // ofstream u_write("u_init.txt");
   // ofstream uc_write("uc_init.txt");
   // u->Print(u_write, 1);
   // uc->Print(uc_write, 1);
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

void AbstractSolver::printError(const std::string &file_name,
                                  int refine,
                                  void (*u_exact)(const Vector &, Vector &))
{
   VectorFunctionCoefficient exsol(num_state, u_exact);
   GridFunType u_q(fes.get());
   fes->GetProlongationMatrix()->Mult(*uc, *u);

   double loc_norm = 0.0;
   const FiniteElement *fe;
   ElementTransformation *T;
   DenseMatrix vals, exact_vals;
   Vector loc_errs;

   const int num_el = fes->GetNE();
   for (int i = 0; i < fes->GetNE(); i++)
   {
      fe = fes->GetFE(i);
      const IntegrationRule *ir = &(fe->GetNodes());
      T = fes->GetElementTransformation(i);
      u->GetVectorValues(*T, *ir, vals);
      exsol.Eval(exact_vals, *T, *ir);
      vals -= exact_vals;
      const int num_q = ir->GetNPoints();
      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         for(int k = 0; k < num_state; k++)
         {
            u_q(i*num_q*num_state + j*num_state + k) = vals(k, j); 
         }
      }
   }

   ofstream sol_ofs(file_name + ".vtk");
   sol_ofs.precision(14);
   if (refine == -1)
   {
      refine = options["space-dis"]["degree"].get<int>() + 1;
   }
   mesh->PrintVTK(sol_ofs, refine);
   u_q.SaveVTK(sol_ofs, "loc_error", refine);
   sol_ofs.close();
}

double AbstractSolver::calcInnerProduct(const GridFunType &x, const GridFunType &y)
{
   double loc_prod = 0.0;
   const FiniteElement *fe;
   ElementTransformation *T;
   DenseMatrix x_vals, y_vals;
   // calculate the L2 inner product for component index `entry`
   for (int i = 0; i < fes->GetNE(); i++)
   {
      fe = fes->GetFE(i);
      const IntegrationRule *ir = &(fe->GetNodes());
      T = fes->GetElementTransformation(i);
      x.GetVectorValues(*T, *ir, x_vals);
      y.GetVectorValues(*T, *ir, y_vals);
      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         T->SetIntPoint(&ip);
         double node_prod = 0.0;
         for (int n = 0; n < num_state; ++n)
         {
            node_prod += x_vals(n,j)*y_vals(n,j);
         }
         loc_prod += ip.weight * T->Weight() * node_prod;
      }
   }
   double prod;
#ifdef MFEM_USE_MPI
   MPI_Allreduce(&loc_prod, &prod, 1, MPI_DOUBLE, MPI_SUM, comm);
#else
   prod = loc_prod;
#endif
   return prod;
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
      fes->GetProlongationMatrix()->Mult(*uc, *u);
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
            //cout << "loc_norm is " << loc_errs(j) <<'\n';
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
   res->Mult(*uc, r);
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
   mesh->PrintVTK(sol_ofs, 0);
   fes->GetProlongationMatrix()->Mult(*uc, *u);
   u->SaveVTK(sol_ofs, "Solution", 0);
   sol_ofs.close();
}

void AbstractSolver::printAdjoint(const std::string &file_name,
                                  int refine)
{
   // TODO: These mfem functions do not appear to be parallelized
   ofstream adj_ofs(file_name + ".vtk");
   adj_ofs.precision(14);
   if (refine == -1)
   {
      refine = options["space-dis"]["degree"].get<int>() + 1;
   }
   mesh->PrintVTK(adj_ofs, refine);
   adj->SaveVTK(adj_ofs, "Adjoint", refine);
   adj_ofs.close();
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

void AbstractSolver::solveForAdjoint(const std::string &fun)
{
   if (options["steady"].get<bool>() == true)
   {
      solveSteadyAdjoint(fun);
   }
   else 
   {
      solveUnsteadyAdjoint(fun);
   }
}

void AbstractSolver::solveSteady()
{
#ifdef MFEM_USE_MPI
   double t1, t2;
   if (0==rank)
   {
      t1 = MPI_Wtime();
   }
#ifdef MFEM_USE_PETSC
   // Get the PetscSolver option
   *out << "Petsc solver with lu preconditioner.\n";
   // Currently need to use serial GMRES solver
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
   *out << "Petsc Solver set.\n";
   //Get the newton solver options
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
   mfem::Vector u_true;
   u->GetTrueDofs(u_true);
   newton_solver->Mult(b, *uc);
   MFEM_VERIFY(newton_solver->GetConverged(), "Newton solver did not converge.");
   u->SetFromTrueDofs(u_true);

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
   *out << "HypreGMRES Solver with euclid preconditioner.\n";
   prec.reset(new HypreEuclid(fes->GetComm()));
   double reltol = options["lin-solver"]["reltol"].get<double>();
   int maxiter = options["lin-solver"]["maxiter"].get<int>();
   int ptl = options["lin-solver"]["printlevel"].get<int>();
   int fill = options["lin-solver"]["filllevel"].get<int>();
   //HYPRE_EuclidSetLevel(dynamic_cast<HypreEuclid*>(prec.get())->GetPrec(), fill);
   solver.reset( new HypreGMRES(fes->GetComm()) );
   dynamic_cast<mfem::HypreGMRES*> (solver.get())->SetTol(reltol);
   dynamic_cast<mfem::HypreGMRES*> (solver.get())->SetMaxIter(maxiter);
   dynamic_cast<mfem::HypreGMRES*> (solver.get())->SetPrintLevel(ptl);
   dynamic_cast<mfem::HypreGMRES*> (solver.get())->SetPreconditioner(*dynamic_cast<HypreSolver*>(prec.get()));
   double nabstol = options["newton"]["abstol"].get<double>();
   double nreltol = options["newton"]["reltol"].get<double>();
   int nmaxiter = options["newton"]["maxiter"].get<int>();
   int nptl = options["newton"]["printlevel"].get<int>();
   newton_solver.reset(new mfem::NewtonSolver(fes->GetComm()));
   //double eta = 1e-1;
   //newton_solver.reset(new InexactNewton(fes->GetComm(), eta));
   newton_solver->iterative_mode = true;
   newton_solver->SetSolver(*solver);
   newton_solver->SetOperator(*res);
   newton_solver->SetPrintLevel(nptl);
   newton_solver->SetRelTol(nreltol);
   newton_solver->SetAbsTol(nabstol);
   newton_solver->SetMaxIter(nmaxiter);

   // Solve the nonlinear problem with r.h.s at 0
   mfem::Vector b;
   //mfem::Vector u_true;
   HypreParVector *u_true = u->GetTrueDofs();
   //HypreParVector b(*u_true);
   //u->GetTrueDofs(u_true);
   newton_solver->Mult(b, *u_true);
   MFEM_VERIFY(newton_solver->GetConverged(), "Newton solver did not converge.");
   //u->SetFromTrueDofs(u_true);
   u->SetFromTrueDofs(*u_true);
#endif
   if (0==rank)
   {
      t2 = MPI_Wtime();
      *out << "Time for solving nonlinear system is " << (t2 - t1) << endl;
   }
#else
   // serial
   cout << "Solve the gd problem in serial.\n";
   solver.reset(new UMFPackSolver());
   dynamic_cast<UMFPackSolver *>(solver.get())->Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
   dynamic_cast<UMFPackSolver *>(solver.get())->SetPrintLevel(1);
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
   // mfem::Vector u_true;
   // u->GetTrueDofs(u_true);
   // newton_solver->Mult(b, *u);
   // MFEM_VERIFY(newton_solver->GetConverged(), "Newton solver did not converge.");
   // u->SetFromTrueDofs(u_true);
   clock_t start_t = clock();
   newton_solver->Mult(b, *uc);
   clock_t end_t = clock();
   double total_t = (double)(end_t - start_t) / CLOCKS_PER_SEC;
   cout << "Time for solve the nonlinear prroblem: " << total_t << "s.\n";
   MFEM_VERIFY(newton_solver->GetConverged(), "Newton solver did not converge.");
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
   int precision = 14;
   // {
   //    ofstream omesh("initial.mesh");
   //    omesh.precision(precision);
   //    mesh->Print(omesh);
   //    ofstream osol("initial-sol.gf");
   //    osol.precision(precision);
   //    u->Save(osol);
   // }

   bool done = false;
   double t_final = options["time-dis"]["t-final"].get<double>();
   *out << "t_final is " << t_final << '\n';
   double dt = options["time-dis"]["dt"].get<double>();
   bool calc_dt = options["time-dis"]["const-cfl"].get<bool>();
   double entropy;
   ofstream entropylog;
   entropylog.open("entropylog.txt", fstream::app | fstream::trunc);
   for (int ti = 0; !done;)
   {
      if (calc_dt)
      {
         dt = calcStepSize(options["time-dis"]["cfl"].get<double>());
      }
      double dt_real = min(dt, t_final - t);
      updateNonlinearMass(ti, dt_real, 1.0);
      //dynamic_cast<mach::ImplicitNonlinearMassEvolver*>(evolver.get())->SetParameters(dt_real, *uc);
      //dynamic_cast<mach::ImplicitNonlinearMassEvolver*>(evolver.get())->checkJacobian(pert);
      // if (ti % 10 == 0)
      // {
      //    *out << "iter " << ti << ": time = " << t << ": dt = " << dt_real
      //         << " (" << round(100 * t / t_final) << "% complete)" << endl;
      // }
      *out << "iter " << ti << ": time = " << t << ": dt = " << dt_real
              << " (" << round(100 * t / t_final) << "% complete)" << endl;
      // ofstream c_write("u_"+std::to_string(ti)+".txt");
      // u->Print(c_write);
      // c_write.close();
#ifdef MFEM_USE_MPI
      HypreParVector *U = u->GetTrueDofs();
      ode_solver->Step(*U, t, dt_real);
      *u = *U;
#else
      ode_solver->Step(*uc, t, dt_real);
#endif
      entropy = calcOutput("entropy");
      entropylog << t << ' ' << entropy << '\n';
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
   printSolution("final_solution");

   // Save the final solution. This output can be viewed later using GLVis:
   // glvis -m unitGridTestMesh.msh -g adv-final.gf".
   // {
   //    ofstream osol("final.gf");
   //    osol.precision(precision);
   //    u->Save(osol);
   // }
   // // write the solution to vtk file
   // if (options["space-dis"]["basis-type"].get<string>() == "csbp")
   // {
   //    ofstream sol_ofs("final_cg.vtk");
   //    sol_ofs.precision(14);
   //    mesh->PrintVTK(sol_ofs, options["space-dis"]["degree"].get<int>() + 1);
   //    u->SaveVTK(sol_ofs, "Solution", options["space-dis"]["degree"].get<int>() + 1);
   //    sol_ofs.close();
   //    printSolution("final");
   // }
   // else if (options["space-dis"]["basis-type"].get<string>() == "dsbp")
   // {
   //    ofstream sol_ofs("final_dg.vtk");
   //    sol_ofs.precision(14);
   //    mesh->PrintVTK(sol_ofs, options["space-dis"]["degree"].get<int>() + 1);
   //    u->SaveVTK(sol_ofs, "Solution", options["space-dis"]["degree"].get<int>() + 1);
   //    sol_ofs.close();
   //    printSolution("final");
   // }
   // TODO: These mfem functions do not appear to be parallelized
}

void AbstractSolver::solveSteadyAdjoint(const std::string &fun)
{
// #ifdef MFEM_USE_MPI
//    double time_beg, time_end;
//    if (0==rank)
//    {
//       time_beg = MPI_Wtime();
//    }
// #endif
//    // Step 0: allocate the adjoint variable
//    adj.reset(new GridFunType(fes.get()));

//    // Step 1: get the right-hand side vector, dJdu, and make an appropriate
//    // alias to it, the state, and the adjoint
//    std::unique_ptr<GridFunType> dJdu(new GridFunType(fes.get()));
// #ifdef MFEM_USE_MPI
//    HypreParVector *state = u->GetTrueDofs();
//    HypreParVector *dJ = dJdu->GetTrueDofs();
//    HypreParVector *adjoint = adj->GetTrueDofs();
// #else
//    GridFunType *state = u.get();
//    GridFunType *dJ = dJdu.get();
//    GridFunType *adjoint = adj.get();
// #endif
//    output.at(fun).Mult(*state, *dJ);

//    // Step 2: get the Jacobian and transpose it
//    Operator *jac = &res->GetGradient(*state);
//    TransposeOperator jac_trans = TransposeOperator(jac);

//    // Step 3: Solve the adjoint problem
//    *out << "Solving adjoint problem:\n"
//         << "\tsolver: HypreGMRES\n"
//         << "\tprec. : Euclid ILU" << endl;
//    prec.reset(new HypreEuclid(fes->GetComm()));
//    double tol = options["adj-solver"]["tol"].get<double>();
//    int maxiter = options["adj-solver"]["maxiter"].get<int>();
//    int ptl = options["adj-solver"]["printlevel"].get<int>();
//    solver.reset(new HypreGMRES(fes->GetComm()));
//    solver->SetOperator(jac_trans);
//    dynamic_cast<mfem::HypreGMRES *>(solver.get())->SetTol(tol);
//    dynamic_cast<mfem::HypreGMRES *>(solver.get())->SetMaxIter(maxiter);
//    dynamic_cast<mfem::HypreGMRES *>(solver.get())->SetPrintLevel(ptl);
//    dynamic_cast<mfem::HypreGMRES *>(solver.get())->SetPreconditioner(*dynamic_cast<HypreSolver *>(prec.get()));
//    solver->Mult(*dJ, *adjoint);
// #ifdef MFEM_USE_MPI
//    adj->SetFromTrueDofs(*adjoint);
//    if (0==rank)
//    {
//       time_end = MPI_Wtime();
//       *out << "Time for solving adjoint is " << (time_end - time_beg) << endl;
//    }
// #endif
}

void AbstractSolver::solveUnsteadyAdjoint(const std::string &fun)
{
   throw MachException("AbstractSolver::solveUnsteadyAdjoint(fun)\n"
                       "\tnot implemented yet!");
}

double AbstractSolver::calcOutput(const std::string &fun)
{
   try
   {
      if (output.find(fun) == output.end())
      {
         cout << "Did not find " << fun << " in output map?" << endl;
      }
      return output.at(fun).GetEnergy(*uc);
   }
   catch (const std::out_of_range &exception)
   {
      std::cerr << exception.what() << endl;
   }
}

void AbstractSolver::checkJacobian(
    void (*pert_fun)(const mfem::Vector &, mfem::Vector &))
{
   // this is a specific version for gd_serial_mfem
   // and dont accept incoming changes
   // initialize some variables
   const double delta = 1e-5;
   CentGridFunction u_plus(*uc);
   CentGridFunction u_minus(*uc);
   CentGridFunction pert_vec(fes.get());
   VectorFunctionCoefficient up(num_state, pert_fun);
   pert_vec.ProjectCoefficient(up);

   // perturb in the positive and negative pert_vec directions
   u_plus.Add(delta, pert_vec);
   u_minus.Add(-delta, pert_vec);

   // Get the product using a 2nd-order finite-difference approximation
   CentGridFunction res_plus(fes.get());
   CentGridFunction res_minus(fes.get());
#ifdef MFEM_USE_MPI 
   HypreParVector *u_p = u_plus.GetTrueDofs();
   HypreParVector *u_m = u_minus.GetTrueDofs();
   HypreParVector *res_p = res_plus.GetTrueDofs();
   HypreParVector *res_m = res_minus.GetTrueDofs();
#else 
   CentGridFunction *u_p = &u_plus;
   CentGridFunction *u_m = &u_minus;
   CentGridFunction *res_p = &res_plus;
   CentGridFunction *res_m = &res_minus;
#endif
   res->Mult(*u_p, *res_p);
   res->Mult(*u_m, *res_m);
#ifdef MFEM_USE_MPI
   res_plus.SetFromTrueDofs(*res_p);
   res_minus.SetFromTrueDofs(*res_m);
#endif
   // res_plus = 1/(2*delta)*(res_plus - res_minus)
   subtract(1/(2*delta), res_plus, res_minus, res_plus);

   // Get the product directly using Jacobian from GetGradient
   CentGridFunction jac_v(fes.get());
#ifdef MFEM_USE_MPI
   HypreParVector *u_true = u->GetTrueDofs();
   HypreParVector *pert = pert_vec.GetTrueDofs();
   HypreParVector *prod = jac_v.GetTrueDofs();
#else
   CentGridFunction *u_true = uc.get();
   CentGridFunction *pert = &pert_vec;
   CentGridFunction *prod = &jac_v;
#endif
   mfem::Operator &jac = res->GetGradient(*u_true);
   jac.Mult(*pert, *prod);
#ifdef MFEM_USE_MPI 
   jac_v.SetFromTrueDofs(*prod);
#endif 

   // check the difference norm
   jac_v -= res_plus;
   double error = calcInnerProduct(jac_v, jac_v);
   *out << "The Jacobian product error norm is " << sqrt(error) << endl;
}

} // namespace mach
