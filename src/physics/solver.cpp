#include <fstream>
#include <iostream>

#include "default_options.hpp"
#include "solver.hpp"
#include "sbp_fe.hpp"
#include "evolver.hpp"
#include "diag_mass_integ.hpp"
#include "solver.hpp"
#include "material_library.hpp"


using namespace std;
using namespace mfem;

/// anonymous namespace for this function so it's only available in this file
#ifdef MFEM_USE_PUMI
namespace
{
/// function to figure out if tet element is next to a model surface
bool isBoundaryTet(apf::Mesh2* m, apf::MeshEntity* e)
{
   apf::MeshEntity* dfs[12];
   int nfs = m->getDownward(e, 2, dfs);
   for (int i = 0; i < nfs; i++)
   {
      int mtype = m->getModelType(m->toModel(dfs[i]));
      if (mtype == 2)
         return true;
   }
   return false;
}
} // anonymous namespace
#endif

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

	materials = material_library;

   constructMesh(move(smesh));
   int dim = mesh->Dimension();
   *out << "problem space dimension = " << dim << endl;

   // Define the ODE solver used for time integration (possibly not used)
   ode_solver = NULL;
   *out << "ode-solver type = "
        << options["time-dis"]["ode-solver"].template get<string>() << endl;
   if (options["time-dis"]["ode-solver"].template get<string>() == "RK1")
   {
      ode_solver.reset(new ForwardEulerSolver);
   }
   else if (options["time-dis"]["ode-solver"].template get<string>() == "RK4")
   {
      ode_solver.reset(new RK4Solver);
   }
   else if (options["time-dis"]["ode-solver"].template get<string>() == "MIDPOINT")
   {
      ode_solver.reset(new ImplicitMidpointSolver);
   }
   else
   {
      throw MachException("Unknown ODE solver type " +
                          options["time-dis"]["ode-solver"].template get<string>());
      // TODO: parallel exit
   }

   // Refine the mesh here, or have a separate member function?
   for (int l = 0; l < options["mesh"]["refine"].template get<int>(); l++)
   {
      mesh->UniformRefinement();
   }

   int fe_order = options["space-dis"]["degree"].template get<int>();
   std::string basis_type = options["space-dis"]["basis-type"].template get<string>();
   bool galerkin_diff = options["space-dis"].value("GD", false);
   // Define the SBP elements and finite-element space; eventually, we will want
   // to have a case or if statement here for both CSBP and DSBP, and (?) standard FEM.
   // and here it is for first two
   if (basis_type == "csbp")
   {
      fec.reset(new SBPCollection(fe_order, dim));
   }
   else if (basis_type == "dsbp" || galerkin_diff)
   {
      fec.reset(new DSBPCollection(fe_order, dim));
   }
   else if (basis_type == "nedelec")
   {
      fec.reset(new ND_FECollection(fe_order, dim));
      // mesh->ReorientTetMesh();
   }
   else if (basis_type == "H1")
   {
      fec.reset(new H1_FECollection(fe_order, dim));
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
   *out << "Number of finite element unknowns: " << fes->GlobalTrueVSize() << endl;
#else
   *out << "Number of finite element unknowns: "
        << fes->GetTrueVSize() << endl;
#endif

   double alpha = 1.0;

   /// construct coefficients before nonlinear/bilinear forms
   constructCoefficients();

   // set up the mass matrix
   mass.reset(new BilinearFormType(fes.get()));
   addMassVolumeIntegrators();
   mass->Assemble();
   mass->Finalize();

   /// TODO: look at partial assembly
   stiff.reset(new BilinearFormType(fes.get()));
   addStiffVolumeIntegrators(alpha);
   addStiffBoundaryIntegrators(alpha);
   addStiffInterfaceIntegrators(alpha);
   stiff->Assemble();
   stiff->Finalize();

   load.reset(new LinearFormType(fes.get()));
   addLoadVolumeIntegrators(alpha);
   addLoadBoundaryIntegrators(alpha);
   addLoadInterfaceIntegrators(alpha);
   load->Assemble();

   /// TODO: look at partial assembly
   // set up the spatial semi-linear form
   res.reset(new NonlinearFormType(fes.get()));
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
   mass_matrix.reset(mass->ParallelAssemble());
   stiffness_matrix.reset(stiff->ParallelAssemble());
#else
   mass_matrix.reset(new MatrixType(mass->SpMat()));
   stiffness_matrix.reset(new MatrixType(stiff->SpMat()));
#endif

   /// check to see if the nonlinear residual has any domain integrators added
   // int num_dnfi = res->GetDNFI()->Size();
   // bool nonlinear = num_dnfi > 0 ? true : false;

   // const string odes = options["time-dis"]["ode-solver"].get<string>();
   // if (odes == "RK1" || odes == "RK4")
   // {
   //    if (nonlinear)
   //       evolver.reset(new NonlinearEvolver(*mass_matrix, *res, -1.0));
   //    else
   //       evolver.reset(new LinearEvolver(*mass_matrix, *stiffness_matrix, *out));
   // }
   // else
   // {
   //    if (nonlinear)
   //       evolver.reset(new ImplicitNonlinearEvolver(*mass_matrix, *res, -1.0));
   //    else
   //    {
   //       /// TODO: revisit this -> evolvers shouldn't need options file
   //       std::string opt_file_name = "options";
   //       evolver.reset(new ImplicitLinearEvolver(opt_file_name, *mass_matrix, *stiffness_matrix, *load, *out));
   //    }
   // }

   constructLinearSolver(options["lin-solver"]);
   constructNewtonSolver();

   constructEvolver();

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
   std::string mesh_file = options["mesh"]["file"].template get<string>();
   std::string mesh_ext;
   size_t i = mesh_file.rfind('.', mesh_file.length());
   if (i != string::npos) {
      mesh_ext = (mesh_file.substr(i+1, mesh_file.length() - i));
   }
   else
   {
      throw MachException("AbstractSolver::constructMesh(smesh)\n"
                        "\tMesh file has no extension!\n");
   }

   /// if serial mesh passed in, use that
   if (smesh != nullptr)
   {
#ifdef MFEM_USE_MPI
      comm = MPI_COMM_WORLD; // TODO: how to pass communicator as an argument?
      MPI_Comm_rank(comm, &rank);
      mesh.reset(new MeshType(comm, *smesh));
#else
      mesh.reset(new MeshType(*smesh));
#endif
   }
   /// native MFEM mesh
   else if (mesh_ext == "mesh")
   {
      // // read in the serial mesh
      // if (smesh == nullptr)
      // {
         smesh.reset(new Mesh(mesh_file.c_str(), 1, 1));
      // }

#ifdef MFEM_USE_MPI
      comm = MPI_COMM_WORLD; // TODO: how to pass communicator as an argument?
      MPI_Comm_rank(comm, &rank);
      mesh.reset(new MeshType(comm, *smesh));
#else
      mesh.reset(new MeshType(*smesh));
#endif
   }
   /// PUMI mesh
   else if (mesh_ext == "smb")
   {
      constructPumiMesh();
   }
}

void AbstractSolver::constructPumiMesh()
{
#ifndef MFEM_USE_PUMI // if using pumi mesh
      throw MachException("AbstractSolver::constructPumiMesh()\n"
                          "\nMFEM was not built with PUMI!\n"
                          "\trecompile MFEM with PUMI\n");
#else
   comm = MPI_COMM_WORLD; // TODO: how to pass communicator as an argument?
   MPI_Comm_rank(comm, &rank);   // problem with using these in loadMdsMesh
   *out << options["model-file"].template get<string>().c_str() << std::endl;
   const char *model_file = options["model-file"].template get<string>().c_str();
   const char *mesh_file = options["mesh"]["file"].template get<string>().c_str();
   PCU_Comm_Init();
#ifdef MFEM_USE_SIMMETRIX
   Sim_readLicenseFile(0);
   gmi_sim_start();
   gmi_register_sim();
#endif
#ifdef MFEM_USE_EGADS
   gmi_egads_start();
   gmi_register_egads();
#endif
   gmi_register_mesh();
   pumi_mesh = apf::loadMdsMesh(options["model-file"].template get<string>().c_str(),
                                options["mesh"]["file"].template get<string>().c_str());
   int mesh_dim = pumi_mesh->getDimension();
   int nEle = pumi_mesh->count(mesh_dim);
   int ref_levels = (int)floor(log(10000. / nEle) / log(2.) / mesh_dim);
   // Perform Uniform refinement
   // if (ref_levels > 1)
   // {
   //    ma::Input* uniInput = ma::configureUniformRefine(pumi_mesh, ref_levels);
   //    ma::adapt(uniInput);
   // }

   /// TODO: change this to use options
   /// If it is higher order change shape
   // int order = options["space-dis"]["degree"].template get<int>();
   // if (order > 1)
   // {
   //     crv::BezierCurver bc(pumi_mesh, order, 2);
   //     bc.run();
   // }

   pumi_mesh->verify();

   apf::Numbering* aux_num = apf::createNumbering(pumi_mesh, "aux_numbering",
                                                  pumi_mesh->getShape(), 1);

   apf::MeshIterator* it = pumi_mesh->begin(0);
   apf::MeshEntity* v;
   int count = 0;
   while ((v = pumi_mesh->iterate(it)))
   {
     apf::number(aux_num, v, 0, 0, count++);
   }
   pumi_mesh->end(it);

   mesh.reset(new ParPumiMesh(comm, pumi_mesh));

   it = pumi_mesh->begin(pumi_mesh->getDimension());
   count = 0;
   while ((v = pumi_mesh->iterate(it)))
   {
     if (count > 10) break;
     printf("at element %d =========\n", count);
     if (isBoundaryTet(pumi_mesh, v))
       printf("tet is connected to the boundary\n");
     else
       printf("tet is NOT connected to the boundary\n");
     apf::MeshEntity* dvs[12];
     int nd = pumi_mesh->getDownward(v, 0, dvs);
     for (int i = 0; i < nd; i++) {
       int id = apf::getNumber(aux_num, dvs[i], 0, 0);
       printf("%d ", id);
     }
     printf("\n");
     Array<int> mfem_vs;
     mesh->GetElementVertices(count, mfem_vs);
     for (int i = 0; i < mfem_vs.Size(); i++) {
       printf("%d ", mfem_vs[i]);
     }
     printf("\n");
     printf("=========\n");
     count++;
   }

   /// Add attributes based on reverse classification
   // Boundary faces
   int dim = mesh->Dimension();
   apf::MeshIterator* itr = pumi_mesh->begin(dim-1);
   apf::MeshEntity* ent ;
   int ent_cnt = 0;
   while ((ent = pumi_mesh->iterate(itr)))
   {
      apf::ModelEntity *me = pumi_mesh->toModel(ent);
      if (pumi_mesh->getModelType(me) == (dim-1))
      {
         //Get tag from model by  reverse classification
         int tag = pumi_mesh->getModelTag(me);
         (mesh->GetBdrElement(ent_cnt))->SetAttribute(tag);
         ent_cnt++;
      }
   }
   pumi_mesh->end(itr);  
   
   // Volume faces
   itr = pumi_mesh->begin(dim);
   ent_cnt = 0;
   while ((ent = pumi_mesh->iterate(itr)))
   {
       apf::ModelEntity *me = pumi_mesh->toModel(ent);
       int tag = pumi_mesh->getModelTag(me);
       mesh->SetAttribute(ent_cnt, tag);
       ent_cnt++;
   }
   pumi_mesh->end(itr);
   
   // Apply the attributes
   mesh->SetAttributes();

   /// TODO: where should we destroy the mesh?
   // pumi_mesh->destroyNative();
   // apf::destroyMesh(pumi_mesh);
   
   PCU_Comm_Free();

#ifdef MFEM_USE_SIMMETRIX
   gmi_sim_stop();
   Sim_unregisterAllKeys();
#endif // MFEM_USE_SIMMETRIX

#ifdef MFEM_USE_EGADS
   gmi_egads_stop();
#endif // MFEM_USE_EGADS

#endif // MFEM_USE_PUMI
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

void AbstractSolver::setInitialCondition(
    double (*u_init)(const Vector &))
{
   FunctionCoefficient u0(u_init);
	u->ProjectCoefficient(u0);
}

void AbstractSolver::setInitialCondition(const Vector &uic)
{
   // TODO: Need to verify that this is ok for scalar fields
   VectorConstantCoefficient u0(uic);
   u->ProjectCoefficient(u0);
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
   return calcL2Error(u.get(), u_exact, entry);
}

double AbstractSolver::calcL2Error(GridFunType *field,
    void (*u_exact)(const Vector &, Vector &), int entry)
{
   // TODO: need to generalize to parallel
   VectorFunctionCoefficient exsol(num_state, u_exact);
   FiniteElementSpace *fe_space = field->FESpace();

   const char* name = fe_space->FEColl()->Name();

   double loc_norm = 0.0;
   const FiniteElement *fe;
   ElementTransformation *T;
   DenseMatrix vals, exact_vals;
   Vector loc_errs;

   if (entry < 0)
   {
      // sum up the L2 error over all states
      for (int i = 0; i < fe_space->GetNE(); i++)
      {
         fe = fe_space->GetFE(i);
         const IntegrationRule *ir;
         if (!strncmp(name, "SBP", 3) || !strncmp(name, "DSBP", 4))
         {        
            ir = &(fe->GetNodes());
         }
         else
         {
            int intorder = 2*fe->GetOrder() + 1;
            ir = &(IntRules.Get(fe->GetGeomType(), intorder));
         }
         T = fe_space->GetElementTransformation(i);
         field->GetVectorValues(*T, *ir, vals);
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
      for (int i = 0; i < fe_space->GetNE(); i++)
      {
         fe = fe_space->GetFE(i);
         const IntegrationRule *ir;
         if (!strncmp(name, "SBP", 3) || !strncmp(name, "DSBP", 4))
         {        
            ir = &(fe->GetNodes());
         }
         else
         {
            int intorder = 2*fe->GetOrder() + 1;
            ir = &(IntRules.Get(fe->GetGeomType(), intorder));
         }
         T = fe_space->GetElementTransformation(i);
         field->GetVectorValues(*T, *ir, vals);
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

void AbstractSolver::printFields(const std::string &file_name,
                                 std::vector<GridFunType*> fields,
                                 std::vector<std::string> names,
                                 int refine)
{
   if (fields.size() != names.size())
   {
      throw MachException(
         "Must supply a name for each grid function to print!");
   }
   // TODO: These mfem functions do not appear to be parallelized
   ofstream sol_ofs(file_name + ".vtk");
   sol_ofs.precision(14);
   if (refine == -1)
   {
      refine = options["space-dis"]["degree"].get<int>() + 1;
   }
   mesh->PrintVTK(sol_ofs, refine);
   for (int i = 0; i < fields.size(); ++i)
   {
      fields[i]->SaveVTK(sol_ofs, names[i], refine);
   }
   sol_ofs.close();
}

std::vector<GridFunType*> AbstractSolver::getFields()
{
   return {u.get()};
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

void AbstractSolver::addMassVolumeIntegrators()
{
   const char* name = fes->FEColl()->Name();
   if (!strncmp(name, "SBP", 3) || !strncmp(name, "DSBP", 4))
   {
      *out << "Adding SBP mass integrator...\n";
      // *out << "num states: " << num_state << "\n";
      mass->AddDomainIntegrator(new DiagMassIntegrator(num_state));
   }
   else
   {
      mass->AddDomainIntegrator(new MassIntegrator());
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
   *out << "Petsc solver with lu preconditioner.\n";
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
   newton_solver.reset(new mfem::NewtonSolver(fes->GetComm()));
   newton_solver->iterative_mode = true;
   newton_solver->SetSolver(*solver);
   newton_solver->SetOperator(*res);
   newton_solver->SetAbsTol(nabstol);
   newton_solver->SetRelTol(nreltol);
   newton_solver->SetMaxIter(nmaxiter);
   newton_solver->SetPrintLevel(nptl);
   *out << "Newton solver is set.\n";
   // Solve the nonlinear problem with r.h.s at 0
   mfem::Vector b;
   mfem::Vector u_true;
   u->GetTrueDofs(u_true);
   newton_solver->Mult(b, u_true);
   MFEM_VERIFY(newton_solver->GetConverged(), "Newton solver did not converge.");
   u->SetFromTrueDofs(u_true);
#else
   // Hypre solver section
   *out << "HypreGMRES Solver with euclid preconditioner.\n";
   // prec.reset(new HypreEuclid(fes->GetComm()));
   // double reltol = options["lin-solver"]["reltol"].get<double>();
   // int maxiter = options["lin-solver"]["maxiter"].get<int>();
   // int ptl = options["lin-solver"]["printlevel"].get<int>();
   // solver.reset( new HypreGMRES(fes->GetComm()) );
   // dynamic_cast<mfem::HypreGMRES*> (solver.get())->SetTol(reltol);
   // dynamic_cast<mfem::HypreGMRES*> (solver.get())->SetMaxIter(maxiter);
   // dynamic_cast<mfem::HypreGMRES*> (solver.get())->SetPrintLevel(ptl);
   // dynamic_cast<mfem::HypreGMRES*> (solver.get())->SetPreconditioner(*dynamic_cast<HypreSolver*>(prec.get()));
   // double nabstol = options["newton"]["abstol"].get<double>();
   // double nreltol = options["newton"]["reltol"].get<double>();
   // int nmaxiter = options["newton"]["maxiter"].get<int>();
   // int nptl = options["newton"]["printlevel"].get<int>();
   // newton_solver.reset(new mfem::NewtonSolver(fes->GetComm()));
   // //double eta = 1e-1;
   // //newton_solver.reset(new InexactNewton(fes->GetComm(), eta));
   // newton_solver->iterative_mode = true;
   // newton_solver->SetSolver(*solver);
   // newton_solver->SetOperator(*res);
   // newton_solver->SetPrintLevel(nptl);
   // newton_solver->SetRelTol(nreltol);
   // newton_solver->SetAbsTol(nabstol);
   // newton_solver->SetMaxIter(nmaxiter);

   constructLinearSolver(options["lin-solver"]);
   constructNewtonSolver();

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
   double t_final = options["time-dis"]["t-final"].template get<double>();
   *out << "t_final is " << t_final << '\n';
   double dt = options["time-dis"]["dt"].get<double>();
   bool calc_dt = options["time-dis"]["const-cfl"].get<bool>();
   for (int ti = 0; !done;)
   {
      if (calc_dt)
      {
         dt = calcStepSize(options["time-dis"]["cfl"].template get<double>());
      }
      double dt_real = min(dt, t_final - t);
      if (ti % 10 == 0)
      {
         *out << "iter " << ti << ": time = " << t << ": dt = " << dt_real
              << " (" << round(100 * t / t_final) << "% complete)" << endl;
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
   if (options["space-dis"]["basis-type"].template get<string>() == "csbp")
   {
      ofstream sol_ofs("final_cg.vtk");
      sol_ofs.precision(14);
      mesh->PrintVTK(sol_ofs, options["space-dis"]["degree"].template get<int>() + 1);
      u->SaveVTK(sol_ofs, "Solution", options["space-dis"]["degree"].template get<int>() + 1);
      sol_ofs.close();
      printSolution("final");
   }
   else if (options["space-dis"]["basis-type"].template get<string>() == "dsbp")
   {
      ofstream sol_ofs("final_dg.vtk");
      sol_ofs.precision(14);
      mesh->PrintVTK(sol_ofs, options["space-dis"]["degree"].template get<int>() + 1);
      u->SaveVTK(sol_ofs, "Solution", options["space-dis"]["degree"].template get<int>() + 1);
      sol_ofs.close();
      printSolution("final");
   }
   // TODO: These mfem functions do not appear to be parallelized
}

void AbstractSolver::solveSteadyAdjoint(const std::string &fun)
{
   double time_beg, time_end;
   if (0==rank)
   {
      time_beg = MPI_Wtime();
   }

   // Step 0: allocate the adjoint variable
   adj.reset(new GridFunType(fes.get()));

   // Step 1: get the right-hand side vector, dJdu, and make an appropriate
   // alias to it, the state, and the adjoint
   std::unique_ptr<GridFunType> dJdu(new GridFunType(fes.get()));
#ifdef MFEM_USE_MPI
   HypreParVector *state = u->GetTrueDofs();
   HypreParVector *dJ = dJdu->GetTrueDofs();
   HypreParVector *adjoint = adj->GetTrueDofs();
#else
   GridFunType *state = u.get();
   GridFunType *dJ = dJdu.get();
   GridFunType *adjoint = adj.get();
#endif
   output.at(fun).Mult(*state, *dJ);

   // Step 2: get the Jacobian and transpose it
   Operator *jac = &res->GetGradient(*state);
   TransposeOperator jac_trans = TransposeOperator(jac);

   // Step 3: Solve the adjoint problem
   *out << "Solving adjoint problem:\n"
        << "\tsolver: HypreGMRES\n"
        << "\tprec. : Euclid ILU" << endl;
   constructLinearSolver(options["adj-solver"]);
   solver->SetOperator(jac_trans);
   solver->Mult(*dJ, *adjoint);
#ifdef MFEM_USE_MPI
   adj->SetFromTrueDofs(*adjoint);
#endif
   if (0==rank)
   {
      time_end = MPI_Wtime();
      *out << "Time for solving adjoint is " << (time_end - time_beg) << endl;
   }
}

void AbstractSolver::constructLinearSolver(nlohmann::json &_options)
{
   std::string prec_type = _options["pctype"].get<std::string>();
   std::string solver_type = _options["type"].get<std::string>();

   if (prec_type == "hypreeuclid")
   {
#ifdef MFEM_USE_MPI
      prec.reset(new HypreEuclid(fes->GetComm()));
#else
      throw MachException("Hypre preconditioners require building MFEM with "
               "MPI!\n");
#endif
   }
   else if (prec_type == "hypreams")
   {
#ifdef MFEM_USE_MPI
      prec.reset(new HypreAMS(fes.get()));
      dynamic_cast<mfem::HypreAMS *>(prec.get())->SetSingularProblem();
#else
      throw MachException("Hypre preconditioners require building MFEM with "
               "MPI!\n");
#endif
   }
   else
   {
      throw MachException("Unsupported preconditioner type!\n"
               "\tavilable options are: HypreEuclid, HypreAMS.\n");
   }

   if (solver_type == "hypregmres")
   {
#ifdef MFEM_USE_MPI
      solver.reset(new HypreGMRES(fes->GetComm()));
#else
      throw MachException("Hypre solvers require building MFEM with MPI!\n");
#endif
   }
   else if (solver_type == "gmressolver")
   {
#ifdef MFEM_USE_MPI
      solver.reset(new GMRESSolver(fes->GetComm()));
#else
      solver.reset(new GMRESSolver());
#endif
   }
   else if (solver_type == "hyprepcg")
   {
#ifdef MFEM_USE_MPI
      solver.reset(new HyprePCG(fes->GetComm()));
#else
      throw MachException("Hypre solvers require building MFEM with MPI!\n");
#endif
   }
   else if (solver_type == "cgsolver")
   {
#ifdef MFEM_USE_MPI
      solver.reset(new CGSolver(fes->GetComm()));
#else
      solver.reset(new CGSolver());
#endif
   }
   else
   {
      throw MachException("Unsupported preconditioner type!\n"
               "\tavilable options are: HypreGMRES, GMRESSolver,\n"
               "\tHyprePCG, CGSolver");
   }

   setIterSolverOptions(_options);
}


void AbstractSolver::constructNewtonSolver()
{
   double abstol = options["newton"]["abstol"].get<double>();
   double reltol = options["newton"]["reltol"].get<double>();
   int maxiter = options["newton"]["maxiter"].get<int>();
   int ptl = options["newton"]["printlevel"].get<int>();

   newton_solver.reset(new mfem::NewtonSolver(fes->GetComm()));
   newton_solver->iterative_mode = true;
   newton_solver->SetSolver(*solver);
   newton_solver->SetOperator(*res);
   newton_solver->SetPrintLevel(ptl);
   newton_solver->SetRelTol(reltol);
   newton_solver->SetAbsTol(abstol);
   newton_solver->SetMaxIter(maxiter);
}

void AbstractSolver::setIterSolverOptions(nlohmann::json &_options)
{
   std::string solver_type = _options["type"].get<std::string>();

   double reltol = _options["reltol"].get<double>();
   int maxiter = _options["maxiter"].get<int>();
   int ptl = _options["printlevel"].get<int>();

   if (solver_type == "hypregmres")
   {
#ifndef MFEM_USE_MPI
      throw MachException("Hypre solvers require building MFEM with MPI!\n");
#endif
      dynamic_cast<mfem::HypreGMRES*> (solver.get())->SetTol(reltol);
      dynamic_cast<mfem::HypreGMRES*> (solver.get())->SetMaxIter(maxiter);
      dynamic_cast<mfem::HypreGMRES*> (solver.get())->SetPrintLevel(ptl);
      dynamic_cast<mfem::HypreGMRES*> (solver.get())->SetPreconditioner(
                                    *dynamic_cast<HypreSolver*>(prec.get()));

      /// set GMRES restart value
      int kdim = _options.value("kdim", -1);
      if (kdim != -1)
         dynamic_cast<mfem::HypreGMRES*> (solver.get())->SetKDim(kdim);
   }
   else if (solver_type == "gmressolver")
   {
      dynamic_cast<mfem::GMRESSolver*> (solver.get())->SetRelTol(reltol);
      dynamic_cast<mfem::GMRESSolver*> (solver.get())->SetMaxIter(maxiter);
      dynamic_cast<mfem::GMRESSolver*> (solver.get())->SetPrintLevel(ptl);
      dynamic_cast<mfem::GMRESSolver*> (solver.get())->SetPreconditioner(
                                    *dynamic_cast<mfem::Solver*>(prec.get()));
   }
   else if (solver_type == "hyprepcg")
   {
#ifndef MFEM_USE_MPI
      throw MachException("Hypre solvers require building MFEM with MPI!\n");
#endif
      dynamic_cast<mfem::HyprePCG*> (solver.get())->SetTol(reltol);
      dynamic_cast<mfem::HyprePCG*> (solver.get())->SetMaxIter(maxiter);
      dynamic_cast<mfem::HyprePCG*> (solver.get())->SetPrintLevel(ptl);
      dynamic_cast<mfem::HyprePCG*> (solver.get())->SetPreconditioner(
                                    *dynamic_cast<HypreSolver*>(prec.get()));
   }
   else if (solver_type == "cgsolver")
   {
      dynamic_cast<mfem::CGSolver*> (solver.get())->SetRelTol(reltol);
      dynamic_cast<mfem::CGSolver*> (solver.get())->SetMaxIter(maxiter);
      dynamic_cast<mfem::CGSolver*> (solver.get())->SetPrintLevel(ptl);
      dynamic_cast<mfem::CGSolver*> (solver.get())->SetPreconditioner(
                                    *dynamic_cast<mfem::Solver*>(prec.get()));
   }
   else
   {
      throw MachException("Unsupported preconditioner type!\n"
               "\tavilable options are: HypreGMRES, GMRESSolver,\n"
               "\tHyprePCG, CGSolver");
   }
}

void AbstractSolver::constructEvolver()
{
   /// check to see if the nonlinear residual has any domain integrators added
   int num_dnfi = res->GetDNFI()->Size();
   bool nonlinear = num_dnfi > 0 ? true : false;

   const string odes = options["time-dis"]["ode-solver"].get<string>();
   if (odes == "RK1" || odes == "RK4")
   {
      if (nonlinear)
         evolver.reset(new NonlinearEvolver(mass.get(), res.get()));
      else
         evolver.reset(new LinearEvolver(mass.get(), stiff.get()));
   }
   else
   {
      if (nonlinear)
         evolver.reset(new ImplicitNonlinearEvolver(mass.get(), res.get()));
      else
      {
         evolver.reset(new ImplicitLinearEvolver(mass.get(), stiff.get()));
      }
   }

   // evolver.reset(new MachEvolver(mass.get(), res.get(), stiff.get(),
   //                               load.get(), *out, 0.0,
   //                               TimeDependentOperator::Type::IMPLICIT));
   evolver->SetNewtonSolver(newton_solver.get());
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
      return output.at(fun).GetEnergy(*u);
   }
   catch (const std::out_of_range &exception)
   {
      std::cerr << exception.what() << endl;
      return -1.0;
   }
}

void AbstractSolver::checkJacobian(
    void (*pert_fun)(const mfem::Vector &, mfem::Vector &))
{
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
   res->Mult(*u_p, *res_p);
   res->Mult(*u_m, *res_m);
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
