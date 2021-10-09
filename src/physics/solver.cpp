#include <fstream>
#include <iostream>
#include <time.h>
#include <iomanip>
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
   else if (options["time-dis"]["ode-solver"].get<string>() == "RRK")
   {
      ode_solver.reset(new RRKImplicitMidpointSolver);
   }
   else
   {
      throw MachException("Unknown ODE solver type " +
                          options["time-dis"]["ode-solver"].get<string>());
      // TODO: parallel exit
   }

   // Define the SBP elements and finite-element space; eventually, we will want
   // to have a case or if statement here for both CSBP and DSBP, and (?) standard FEM.
   // and here it is for first two
   if (options["space-dis"]["basis-type"].get<string>() == "dg")
   {
      fec.reset(new DG_FECollection(options["space-dis"]["degree"].get<int>(), dim, BasisType::GaussLobatto));
   }
   else if ( options["space-dis"]["basis-type"].get<string>() == "dsbp")
   {
      fec.reset(new DSBPCollection(options["space-dis"]["degree"].get<int>(), dim));
   }
   else if (options["space-dis"]["basis-type"].get<string>() == "csbp")
   {
      fec.reset(new SBPCollection(options["space-dis"]["degree"].get<int>(), dim ));
   }

}
void AbstractSolver::masslumpCheck( void (*u_init)(const Vector &, Vector &))
{
   num_state = 1;
   *out << "Num states = " << num_state << endl;
   if (options["space-dis"]["GD"].get<bool>() == true)
   {
      int gd_degree = options["space-dis"]["GD-degree"].get<int>();
      mesh->ElementToElementTable();
      fes.reset(new GalerkinDifference(mesh.get(), fec.get(), num_state,
                                       Ordering::byVDIM, gd_degree));
      fes_normal.reset(new SpaceType(mesh.get(), fec.get(), num_state,
                              Ordering::byVDIM));
      uc.reset(new CentGridFunction(fes.get()));
      u.reset(new GridFunType(fes_normal.get()));
   }
   else
   {
      fes.reset(new SpaceType(mesh.get(), fec.get(), num_state,
                              Ordering::byVDIM));
      u.reset(new GridFunType(fes.get()));
   }

   double alpha = 1.0;
   // set up the mass matrix
   mass.reset(new BilinearFormType(fes_normal.get()));
   if (options["space-dis"]["basis-type"].get<string>() == "dg")
   {
      mass->AddDomainIntegrator(new MassIntegrator);
   }
   else if (options["space-dis"]["basis-type"].get<string>() == "dsbp")
   {
      mass->AddDomainIntegrator(new DiagMassIntegrator(num_state));
   }
   mass->Assemble();
   mass->Finalize();
   mass_matrix.reset(new MatrixType(mass->SpMat()));
   MatrixType *cp = dynamic_cast<GalerkinDifference*>(fes.get())->GetCP();
   MatrixType *p = RAP(*cp, *mass_matrix, *cp);
   mass_matrix_gd.reset(new MatrixType(*p));
   // mass lumping
   const bool lump = options["mass-matrix"]["lump"].get<bool>();
   Vector diag(mass_matrix_gd->Height());
   if (lump)
   {
      double *cols;
      int num_in_row;
      diag = 0.0;
      for (int i = 0; i < mass_matrix_gd->Height(); i++)
      {
         cols = mass_matrix_gd->GetRowEntries(i);
         num_in_row = mass_matrix_gd->RowSize(i);
         for (int j = 0; j < num_in_row; j++)
         {
            diag(i) += cols[j];
         }
      }
      mass_matrix_gd.reset(new MatrixType(diag));
   }
   setInitialCondition(u_init);
   double integration = (*uc) * diag;
   // double exact = 2 * M_PI; // 0th order
   // double exact = 26.0/3.0; // 1st order
   // double exact = 5.0*M_PI; // 2nd order
   // double exact = 484.0/15.0; // 3rd order
   // double exact = 91.0 *M_PI / 4.0; // 4th order
   // double exact = 4.0; // nonliear
   cout << "Integration error is " <<  std::setprecision(14) << 4.0 - integration << endl;
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
      u.reset(new GridFunType(fes_normal.get()));
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

   double alpha = 1.0;
   // set up the mass matrix
   mass.reset(new BilinearFormType(fes_normal.get()));
   mass->AddDomainIntegrator(new DiagMassIntegrator(num_state));
   mass->Assemble();
   mass->Finalize();
   // set nonlinear mass matrix form
   nonlinear_mass.reset(new NonlinearFormType(fes.get()));
   addMassIntegrators(alpha);
   // set up the spatial semi-linear form
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

   // add the output functional QoIs 
   auto &fun = options["outputs"];
   using json_iter = nlohmann::json::iterator;
   int num_bndry_outputs = 0;
   for (json_iter it = fun.begin(); it != fun.end(); ++it) {
      if (it->is_array()) ++num_bndry_outputs;
   }
   output_bndry_marker.resize(num_bndry_outputs);
   addOutputs(); // virtual function

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


   // mass lumping
   const bool lump = options["mass-matrix"]["lump"].get<bool>();
   if (lump)
   {
      double *cols;
      int num_in_row;
      Vector diag(mass_matrix_gd->Height());
      diag = 0.0;
      for (int i = 0; i < mass_matrix_gd->Height(); i++)
      {
         cols = mass_matrix_gd->GetRowEntries(i);
         num_in_row = mass_matrix_gd->RowSize(i);
         for (int j = 0; j < num_in_row; j++)
         {
            diag(i) += cols[j];
         }
      }
      mass_matrix_gd.reset(new MatrixType(diag));
   }

#endif
   const string odes = options["time-dis"]["ode-solver"].get<string>();
   if (odes == "RK1" || odes == "RK4")
   {
      evolver.reset(new NonlinearEvolver(*mass_matrix_gd, *res, -1.0));
   }
   else if (odes == "MIDPOINT")
   {
      //evolver.reset(new ImplicitNonlinearMassEvolver(*nonlinear_mass, *res, -1.0));
      //evolver.reset(new ImplicitNonlinearEvolver(*mass_matrix, *res, -1.0));
      evolver.reset(new ImplicitNonlinearEvolver(*mass_matrix_gd, *res, this, -1.0));
   }
   else if (odes == "RRK")
   {
      evolver.reset(
          new ImplicitNonlinearMassEvolver(*nonlinear_mass, *res,
                                           output.at("entropy"), -1.0));
   }
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
   // ofstream sol_ofs("unsteady_vortex_mesh.vtk");
   // sol_ofs.precision(14);
   // smesh->PrintVTK(sol_ofs,0);
   // sol_ofs.close();
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


void AbstractSolver::setInverseInitialCondition(
   void (*u_init)(const Vector &, Vector &))
{
   // Apply the initial condition at SBP nodes
   VectorFunctionCoefficient u0(num_state, u_init);
   u->ProjectCoefficient(u0);

   // apply the quadrature weights
   GridFunType u_weighted(fes_normal.get());
   CentGridFunction temp(fes.get());
   Array<int> vdofs;
   Vector el_x, el_y;
   const FiniteElement *fe;
   const SBPFiniteElement *sbp;
   ElementTransformation *T;
   Vector u_j(num_state);
   int num_nodes;
   for (int i = 0; i < fes->GetNE(); i++)
   {
      fe = fes->GetFE(i);
      fes->GetElementVDofs(i, vdofs);
      T = fes->GetElementTransformation(i);
      u->GetSubVector(vdofs, el_x);
      sbp = dynamic_cast<const SBPFiniteElement*>(fe);
      const IntegrationRule &ir = sbp->GetNodes();
      num_nodes = sbp->GetDof();
      el_y.SetSize(num_state*num_nodes);
      DenseMatrix u_matrix(el_x.GetData(), num_nodes, num_state);
      DenseMatrix res(el_y.GetData(), num_nodes, num_state);
      el_y = 0.0;
      for (int j = 0; j < num_nodes; j++)
      {
         u_matrix.GetRow(j, u_j);
         const IntegrationPoint &ip = fe->GetNodes().IntPoint(j);
         T->SetIntPoint(&ip);
         double weight = T->Weight() * ip.weight;
         for (int n = 0; n < num_state; n++)
         {
            res(j,n) += weight * u_j(n);
         }
      }
      u_weighted.SetSubVector(vdofs, el_y);
   }

   //compute temp = P^t H u_{sbp}
   dynamic_cast<GalerkinDifference*>(fes.get())->GetCP()->MultTranspose(u_weighted, temp);

   // assemble the mass matrix M
   MatrixType *mass_matrix = &(mass->SpMat());
   MatrixType *cp = dynamic_cast<GalerkinDifference*>(fes.get())->GetCP();
   MatrixType *p = RAP(*cp, *mass_matrix, *cp);

   // Solver for M u_c = P^t H u_{sbp}
   mfem::CG(*p, temp, *uc, 1, 1000, 1e-13, 1e-24);
   GridFunType u_test(fes.get());
   fes->GetProlongationMatrix()->Mult(*uc, u_test);
   ofstream projection("initial_projection.vtk");
   projection.precision(14);
   mesh->PrintVTK(projection, 0);
   u_test.SaveVTK(projection, "projection", 0);
   projection.close();

   // check the projection error
   u_test -= *u;
   cout << "The initial condition projection error norm is " << u_test.Norml2() << '\n';
}


void AbstractSolver::setMinL2ErrorInitialCondition(
   void (*u_init)(const Vector &, Vector &))
{
   // apply the initial condition to quadrature points
   VectorFunctionCoefficient u0(num_state, u_init);
   u->ProjectCoefficient(u0);
   ofstream initial("initial_condition.vtk");
   initial.precision(14);
   mesh->PrintVTK(initial,0);
   u->SaveVTK(initial,"initial",0);
   initial.close();
   cout << "After apply initial condition.\n";

   // Compute the projected coefficient
   MatrixType *cp = dynamic_cast<GalerkinDifference*>(fes.get())->GetCP();
   MatrixType *p = RAP(*cp, *mass_matrix, *cp);
   cout << "p size is " << p->Height() << ' ' << p->Width() << endl;
   Vector hu(u->Size()),pthu(num_state * fes->GetNE());
   cout << "mass_matrix size is " << mass_matrix->Height() << ' ' << mass_matrix->Width() << endl;
   cout << "u size is " << u->Size() << endl;
   mass_matrix->Mult(*u,hu);
   cout << "hu size is " << hu.Size() << endl;
   cp->MultTranspose(hu,pthu);
   cout << "Get A and b.\n";
   mfem::CG(*p,pthu,*uc,1,100,1e-24,1e-48);
   
   GridFunType u_test(fes.get());
   fes->GetProlongationMatrix()->Mult(*uc, u_test);
   ofstream projection("initial_projection.vtk");
   projection.precision(14);
   mesh->PrintVTK(projection, 0);
   u_test.SaveVTK(projection, "projection", 0);
   projection.close();
   ofstream up_write("up_init.txt");
   up_write<<setprecision(15);
   u_test.Print(up_write,1);
   up_write.close();

   u_test -= *u;
   cout << "After projection, the difference norm is " << u_test.Norml2() << '\n';
   ofstream sol_ofs("projection_error.vtk");
   sol_ofs.precision(14);
   mesh->PrintVTK(sol_ofs, 0);
   u_test.SaveVTK(sol_ofs, "project_error", 0);
   sol_ofs.close();

   ofstream u_write("u_init.txt");
   ofstream uc_write("uc_init.txt");
   u_write << setprecision(15);
   uc_write << setprecision(15);
   u->Print(u_write, 1);
   uc->Print(uc_write, 1);
   u_write.close();
   uc_write.close();
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
   // cout << "check nodal values\n";
   // mfem::Array<int> vdofs;
   // int num_dofs;
   // for (int i = 0; i < fes_normal->GetNE(); i++)
   // {
   //    const FiniteElement *fe = fes->GetFE(i);
   //    num_dofs = fe->GetDof();
   //    fes_normal->GetElementVDofs(i, vdofs);
   //    for(int j = 0; j < num_dofs; j++)
   //    {
   //       for(int k = 0; k < num_state; k++)
   //       {
   //          cout << (*u)(vdofs[k*num_dofs + j]) << ' ';
   //          cout << u_test(vdofs[k*num_dofs + j]) << "   ";
   //       }
   //       cout << std::endl;
   //    }
   // }
   
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


double AbstractSolver::calcSodShockMaxError(
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
         loc_errs.SetSize(vals.Height());
         //vals.Norm1(loc_errs);
         for (int j = 0; j < ir->GetNPoints(); j++)
         {
            const IntegrationPoint &ip = ir->IntPoint(j);
            T->SetIntPoint(&ip);
            vals.GetColumn(j, loc_errs);
            for (int k = 0; k < num_state; k++)
            {
               //  loc_norm += ip.weight * T->Weight() * abs(loc_errs(k));
               //  loc_norm  = max(loc_norm, ip.weight * T->Weight() * abs(loc_errs(k)));
               loc_norm = max(loc_norm, abs(loc_errs(k)));
            }
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
            //loc_norm = max(loc_norm, ip.weight * T->Weight() * loc_errs(j));
            loc_norm = max(loc_norm, abs(loc_errs(j)));
            //loc_norm += ip.weight * T->Weight() * abs(loc_errs(j));
            //cout << "loc_norm is " << loc_errs(j) <<'\n';
         }
      }
   }
   double norm = loc_norm;
   return norm;
}


double AbstractSolver::calcSodShockL1Error(
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
         loc_errs.SetSize(vals.Height());
         //vals.Norm1(loc_errs);
         for (int j = 0; j < ir->GetNPoints(); j++)
         {
            const IntegrationPoint &ip = ir->IntPoint(j);
            T->SetIntPoint(&ip);
            vals.GetColumn(j, loc_errs);
            for (int k = 0; k < num_state; k++)
            {
               loc_norm += ip.weight * T->Weight() * abs(loc_errs(k));  
            }
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
            loc_norm += ip.weight * T->Weight() * abs(loc_errs(j));
         }
      }
   }
   double norm = loc_norm;
   return norm;
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
   // uc->SaveVTK(sol_ofs, "GD Solution",0);
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

void AbstractSolver::printResidual(const std::string &file_name)
{
//    GridFunType r(fes.get());
// #ifdef MFEM_USE_MPI
//    // HypreParVector *U = u->GetTrueDofs();
//    // res->Mult(*U, r);
//    cout << "Print residual is called, in MPI, uc is feed into Mult, size: " << uc->Size() << '\n';
//    res->Mult(*uc, r);
// #else
//    cout << "Print residual is called, not in MPI, uc is feed into Mult, size: " << uc->Size() << '\n';
//    res->Mult(*u, r);
// #endif
//    // TODO: These mfem functions do not appear to be parallelized
//    ofstream res_ofs(file_name + ".vtk");
//    res_ofs.precision(14);
//    if (refine == -1)
//    {
//       refine = options["space-dis"]["degree"].get<int>() + 1;
//    }
//    mesh->PrintVTK(res_ofs, refine);
//    r.SaveVTK(res_ofs, "Residual", refine);
//    res_ofs.close();
   mfem::Vector test(uc->Size());
   res->Mult(*uc, test);
   ofstream write_center(file_name+"_res_coord.txt");
   ofstream write_state(file_name+"_res.txt");
   write_state.precision(14);
   write_center.precision(14);
   // print the state
   mfem::Vector cent(1);
   int geom = mesh->GetElement(0)->GetGeometryType();
   ElementTransformation *eltransf;
   for (int i = 0; i < fes->GetNE(); i++)
   {
      eltransf = mesh->GetElementTransformation(i);
      eltransf->Transform(Geometries.GetCenter(geom), cent);
      write_center << cent(0) << std::endl;
      for (int j = 0; j < num_state; j++)
      {
         write_state << test( i * num_state + j) << ' ';
      }
      write_state << std::endl;
   }
   write_state.close();
   write_center.close();
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
   cout << "Check the inner product.\n";
   mfem::Vector test(uc->Size());
   res->Mult(*uc, test);
   // printResidual("initial_cent");
   double inner = (*uc) * test;
   cout << "The inner product is " << test.Norml2() << '\n'; 
   // check the jacobian
   bool done = false;
   double t_final = options["time-dis"]["t-final"].get<double>();
   *out << "t_final is " << t_final << '\n';
   double dt = options["time-dis"]["dt"].get<double>();
   bool calc_dt = options["time-dis"]["const-cfl"].get<bool>();
   remove("entropylog.txt");
   double entropy;
   ofstream entropylog;
   entropylog.open("entropylog.txt", fstream::app);
   entropylog << setprecision(17);
   clock_t start_t = clock();
   for (int ti = 0; !done;)
   {
      entropy = calcOutput("entropy");
      entropylog << t << ' ' << entropy << '\n';
      if (calc_dt)
      {
         dt = calcStepSize(options["time-dis"]["cfl"].get<double>());
      }
      double dt_real = min(dt, t_final - t);
      // TODO: !!!!! The following does not generalize beyond midpoint !!!!!
      //updateNonlinearMass(ti, 0.5*dt_real, 1.0);
      // if (0 == ti)
      // {
      //    //dynamic_cast<mach::ImplicitNonlinearMassEvolver *>(evolver.get())->checkJacobian(pert, *uc);
      //    MatrixType *jac1 = dynamic_cast<MatrixType *>(&res->GetGradient(*uc));
      //    MatrixType *jac2 = dynamic_cast<MatrixType *>(&nonlinear_mass->GetGradient(*uc));
      //    ofstream jac_save("jac.txt");
      //    ofstream mass_save("mass.txt");
      //    jac1->PrintMatlab(jac_save);
      //    jac2->PrintMatlab(mass_save);
      //    jac_save.close();
      //    mass_save.close();
      // }
      //updateNonlinearMass(ti, dt_real/2, 1.0);
      if (ti % 20 == 0)
      {
         *out << "iter " << ti << ": time = " << t << ": dt = " << dt_real
              << " (" << round(100 * t / t_final) << "% complete)" << endl;
      }
#ifdef MFEM_USE_MPI
      HypreParVector *U = u->GetTrueDofs();
      ode_solver->Step(*U, t, dt_real);
      *u = *U;
#else
      ode_solver->Step(*uc, t, dt_real);
#endif
      ti++;
      done = (t >= t_final - 1e-8 * dt);
   }
   *out << "Time steps are done, final time t = " << t << endl;
   entropy = calcOutput("entropy");
   entropylog << t << ' ' << entropy << '\n';
   clock_t end_t = clock();
   double total_t = (double)(end_t - start_t) / CLOCKS_PER_SEC;
   cout << "Wall time for solving unsteady vortex problem: " << total_t << '\n';
   entropylog.close();
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
   ofstream jac_save("jac.txt");
   jac.PrintMatlab(jac_save);
   jac_save.close();
   jac.Mult(*pert, *prod);
#ifdef MFEM_USE_MPI 
   jac_v.SetFromTrueDofs(*prod);
#endif 

   // check the difference norm
   jac_v -= res_plus;
   //double error = calcInnerProduct(jac_v, jac_v);
   double error = jac_v *jac_v;
   *out << "The Jacobian product error norm is " << sqrt(error) << endl;
}

} // namespace mach
