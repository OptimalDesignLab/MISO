#include "solver.hpp"

#include <fstream>
#include <iostream>

#ifdef MFEM_USE_PUMI

#include "apfMDS.h"
#include "PCU.h"
#include "apfConvert.h"
#include "crv.h"
#include "gmi_mesh.h"
#include "gmi_null.h"

#ifdef MFEM_USE_SIMMETRIX
#include "SimUtil.h"
#include "gmi_sim.h"
#endif // MFEM_USE_SIMMETRIX

#ifdef MFEM_USE_EGADS
#include "gmi_egads.h"
#endif // MFEM_USE_EGADS

#endif // MFEM_USE_PUMI

#include "utils.hpp"
#include "mfem_extensions.hpp"
#include "default_options.hpp"
#include "solver.hpp"
#include "sbp_fe.hpp"
#include "evolver.hpp"
#include "diag_mass_integ.hpp"
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

namespace mach
{
/// specifies how to delete a PUMI mesh so that the PUMI mesh can be stored in
/// a unique_ptr and safely deleted
/*struct pumiDeleter
{
   void operator()(apf::Mesh2* mesh) const
   {
      mesh->destroyNative();
      apf::destroyMesh(mesh);
      PCU_Comm_Free();
#ifdef MFEM_USE_SIMMETRIX
      gmi_sim_stop();
      Sim_unregisterAllKeys();
#endif // MFEM_USE_SIMMETRIX

#ifdef MFEM_USE_EGADS
      gmi_egads_stop();
#endif // MFEM_USE_EGADS
   }
};
*/
} // namespace mach
#endif

namespace mach
{

adept::Stack AbstractSolver::diff_stack;

AbstractSolver::AbstractSolver(const nlohmann::json &file_options,
                               unique_ptr<Mesh> smesh,
                               MPI_Comm comm)
{
   initBase(file_options, move(smesh), comm);
}

// Note: the following constructor is protected
AbstractSolver::AbstractSolver(const string &opt_file_name,
                               MPI_Comm incomm)
{
   // Some of the following code would normally happen in initBase, but this is 
   // a parred down version of the AbstractSolver that does not need most of 
   // the functionality (e.g. multiphysics code)
   // TODO: Do we need a separate super class for this case?

   // Set the options; the defaults are overwritten by the values in the file
   // using the merge_patch method
   nlohmann::json file_options;
   ifstream options_file(opt_file_name);
   options_file >> file_options;
   options = default_options;
   options.merge_patch(file_options);
   *out << setw(3) << options << endl;

   // comm = incomm;
   MPI_Comm_dup(incomm, &comm);
   MPI_Comm_rank(comm, &rank);
   out = getOutStream(rank);
}

void AbstractSolver::initBase(const nlohmann::json &file_options,
                              std::unique_ptr<Mesh> smesh,
                              MPI_Comm incomm)
{
   // Set the options; the defaults are overwritten by the values in the file
   // using the merge_patch method
   // comm = incomm;
   MPI_Comm_dup(incomm, &comm);
   MPI_Comm_rank(comm, &rank);
   out = getOutStream(rank);
   options = default_options;
   options.merge_patch(file_options);
   if(options["print-options"])
   {
      *out << setw(3) << options << endl;
   }

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
   else if (options["time-dis"]["ode-solver"].get<string>() == "RRK")
   {
      ode_solver.reset(new RRKImplicitMidpointSolver(out));
   }
   else if (options["time-dis"]["ode-solver"].template get<string>() == "PTC")
   {
      ode_solver.reset(new PseudoTransientSolver(out));
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
   *out << "Number of finite element unknowns: " << fes->GlobalTrueVSize() << endl;

   double alpha = 1.0;

   // construct coefficients before nonlinear/bilinear forms
   constructCoefficients();

   // need to set this before adding boundary integrators
   auto &bcs = options["bcs"];
   bndry_marker.resize(bcs.size());

   // construct/initialize the forms needed by derived class
   constructForms();

   if (nonlinear_mass)
   {
      addNonlinearMassIntegrators(alpha);
   }

   if (mass)
   {
      addMassIntegrators(alpha);
      mass->Assemble(0);
      mass->Finalize();
   }

   if (res)
   {
      /// TODO: look at partial assembly
      addResVolumeIntegrators(alpha);
      addResBoundaryIntegrators(alpha);
      addResInterfaceIntegrators(alpha);
   }

   if (stiff)
   {
      /// TODO: look at partial assembly
      addStiffVolumeIntegrators(alpha);
      addStiffBoundaryIntegrators(alpha);
      addStiffInterfaceIntegrators(alpha);
      stiff->Assemble(0);
      stiff->Finalize();
   }

   if (load)
   {
      /// TODO: make this work for a grid function as well as a linear form
      //load.reset(new LinearFormType(fes.get()));
      addLoadVolumeIntegrators(alpha);
      addLoadBoundaryIntegrators(alpha);
      addLoadInterfaceIntegrators(alpha);
      load->Assemble();
   }

   if (ent)
   {
      addEntVolumeIntegrators();
   }

   // This just lists the boundary markers for debugging purposes
   for (unsigned k = 0; k < bndry_marker.size(); ++k)
   {
      *out << "boundary_marker[" << k << "]: ";
      for (int i = 0; i < bndry_marker[k].Size(); ++i)
      {
         *out << bndry_marker[k][i] << " ";
      }
      *out << endl;
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

   constructEvolver();
}

AbstractSolver::~AbstractSolver()
{
   MPI_Comm_free(&comm);
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

   // if serial mesh passed in, use that
   if (smesh != nullptr)
   {
      mesh.reset(new MeshType(comm, *smesh));
   }
   // native MFEM mesh
   else if (mesh_ext == "mesh")
   {
      // read in the serial mesh
      smesh.reset(new Mesh(mesh_file.c_str(), 1, 1));
      mesh.reset(new MeshType(comm, *smesh));
   }
   // PUMI mesh
   else if (mesh_ext == "smb")
   {
      constructPumiMesh();
   }
   mesh->EnsureNodes();
}

void AbstractSolver::constructPumiMesh()
{
#ifdef MFEM_USE_PUMI // if using pumi mesh
   *out << options["mesh"]["model-file"].get<string>().c_str() << std::endl;
   std::string model_file = options["mesh"]["model-file"].get<string>();
   std::string mesh_file = options["mesh"]["file"].get<string>();
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
   pumi_mesh.reset(apf::loadMdsMesh(model_file.c_str(), mesh_file.c_str()));
   // int mesh_dim = pumi_mesh->getDimension();
   // int nEle = pumi_mesh->count(mesh_dim);
   // int ref_levels = (int)floor(log(10000. / nEle) / log(2.) / mesh_dim);
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

   apf::Numbering* aux_num = apf::createNumbering(pumi_mesh.get(), "aux_numbering",
                                                  pumi_mesh->getShape(), 1);

   apf::MeshIterator* it = pumi_mesh->begin(0);
   apf::MeshEntity* v;
   int count = 0;
   while ((v = pumi_mesh->iterate(it)))
   {
     apf::number(aux_num, v, 0, 0, count++);
   }
   pumi_mesh->end(it);

   mesh.reset(new ParPumiMesh(comm, pumi_mesh.get()));

   it = pumi_mesh->begin(pumi_mesh->getDimension());
   count = 0;
   while ((v = pumi_mesh->iterate(it)))
   {
     if (count > 10) break;
     printf("at element %d =========\n", count);
     if (isBoundaryTet(pumi_mesh.get(), v))
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
#else
   throw MachException("AbstractSolver::constructPumiMesh()\n"
                       "\tMFEM was not built with PUMI!\n"
                       "\trecompile MFEM with PUMI\n");
#endif // MFEM_USE_PUMI
}

void AbstractSolver::setInitialCondition(
   ParGridFunction &state,
   const std::function<double(const mfem::Vector &)> &u_init)
{
   FunctionCoefficient u0(u_init);
   state.ProjectCoefficient(u0);
}

void AbstractSolver::setInitialCondition(
   ParGridFunction &state,
   const std::function<void(const mfem::Vector &, mfem::Vector &)> &u_init)
{
   VectorFunctionCoefficient u0(num_state, u_init);
   state.ProjectCoefficient(u0);
}

void AbstractSolver::setInitialCondition(ParGridFunction &state,
                                         const double u_init)
{
   ConstantCoefficient u0(u_init);
   state.ProjectCoefficient(u0);
}

void AbstractSolver::setInitialCondition(ParGridFunction &state,
                                         const Vector &u_init)
{
   VectorConstantCoefficient u0(u_init);
   state.ProjectCoefficient(u0);
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
   MPI_Allreduce(&loc_prod, &prod, 1, MPI_DOUBLE, MPI_SUM, comm);
   return prod;
}

double AbstractSolver::calcL2Error(
   const std::function<double(const mfem::Vector &)> &u_exact)
{
   return calcL2Error(u.get(), u_exact);
}

double AbstractSolver::calcL2Error(
   const std::function<void(const mfem::Vector &, mfem::Vector &)> &u_exact,
   int entry)
{
   return calcL2Error(u.get(), u_exact, entry);
}

double AbstractSolver::calcL2Error(
   GridFunType *field,
   const std::function<double(const mfem::Vector &)> &u_exact)
{
   // TODO: need to generalize to parallel
   FunctionCoefficient exsol(u_exact);
   FiniteElementSpace *fe_space = field->FESpace();

   const char* name = fe_space->FEColl()->Name();

   double loc_norm = 0.0;
   const FiniteElement *fe;
   ElementTransformation *T;
   Vector shape;
   Array<int> vdofs;

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
      int fdof = fe->GetDof();
      T = fe_space->GetElementTransformation(i);
      shape.SetSize(fdof);
      fes->GetElementVDofs(i, vdofs);
      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         fe->CalcShape(ip, shape);
         double a = 0;
         for (int k = 0; k < fdof; k++)
            if (vdofs[k] >= 0)
            {
               a += (*field)(vdofs[k]) * shape(k);
            }
            else
            {
               a -= (*field)(-1-vdofs[k]) * shape(k);
            }
         T->SetIntPoint(&ip);
         a -= exsol.Eval(*T, ip);
         loc_norm += ip.weight * T->Weight() * a * a;
      }
   }

   double norm;
   MPI_Allreduce(&loc_norm, &norm, 1, MPI_DOUBLE, MPI_SUM, comm);
   if (norm < 0.0) // This was copied from mfem...should not happen for us
   {
      return -sqrt(-norm);
   }
   return sqrt(norm);
}

double AbstractSolver::calcL2Error(
   GridFunType *field,
   const std::function<void(const mfem::Vector &, mfem::Vector &)> &u_exact,
   int entry)
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
   MPI_Allreduce(&loc_norm, &norm, 1, MPI_DOUBLE, MPI_SUM, comm);
   if (norm < 0.0) // This was copied from mfem...should not happen for us
   {
      return -sqrt(-norm);
   }
   return sqrt(norm);
}

double AbstractSolver::calcResidualNorm() const
{
   GridFunType r(fes.get());
   double res_norm;
   HypreParVector *u_true = u->GetTrueDofs();
   HypreParVector *r_true = r.GetTrueDofs();
   res->Mult(*u_true, *r_true); 
   double loc_norm = (*r_true)*(*r_true);
   MPI_Allreduce(&loc_norm, &res_norm, 1, MPI_DOUBLE, MPI_SUM, comm);
   res_norm = sqrt(res_norm);
   return res_norm;
}

std::unique_ptr<ParGridFunction> AbstractSolver::getNewStateVector(
   double *data)
{
   if (data == nullptr)
   {
      return std::unique_ptr<ParGridFunction>(
         new ParGridFunction(fes.get()));
   }
   else
   {
      return std::unique_ptr<ParGridFunction>(
         new ParGridFunction(fes.get(), data));
   }

}

/// TODO: this won't work for every solver, need to create a new operator like
/// the one used in the evolvers to evaluate the residual
void AbstractSolver::calcResidual(const ParGridFunction &state,
                                  ParGridFunction &residual)
{
   auto *u_true = state.GetTrueDofs();
   res->Mult(*u_true, residual);
}

double AbstractSolver::calcStepSize(int iter, double t, double t_final,
                                    double dt_old) const
{
   double dt = options["time-dis"]["dt"].get<double>();
   dt = min(dt, t_final - t);
   return dt;
}

bool AbstractSolver::iterationExit(int iter, double t, double t_final,
                                   double dt)
{
   if (t >= t_final - 1e-14 * dt) return true;
   return false;
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
   HypreParVector *u_true = u->GetTrueDofs();
   res->Mult(*u_true, r);
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
   for (unsigned i = 0; i < fields.size(); ++i)
   {
      fields[i]->SaveVTK(sol_ofs, names[i], refine);
   }
   sol_ofs.close();
}

std::vector<GridFunType*> AbstractSolver::getFields()
{
   return {u.get()};
}

void AbstractSolver::solveForState(ParGridFunction &state)
{
   if (options["steady"].get<bool>() == true)
   {
      solveSteady(state);
   }
   else 
   {
      solveUnsteady(state);
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

void AbstractSolver::setMeshNodalCoordinates(mfem::Vector &coords)
{
   /// Added `EnsureNodes` to `constructMesh`
   // if (mesh->GetNodes() == nullptr)
   // {
   //    // TODO: this approach will have the mesh allocate a `Nodes`
   //    // gridfunction that we immediately replace (wasteful)
   //    mesh->EnsureNodes();
   // }
   // auto mesh_gf = static_cast<ParGridFunction*>(mesh->GetNodes());
   auto mesh_gf = mesh->GetNodes();
   // mesh_gf->MakeRef(coords, 0);
   mesh_gf->MakeRef(coords, 0);
}

void AbstractSolver::addMassIntegrators(double alpha)
{
   const char* name = fes->FEColl()->Name();
   if (!strncmp(name, "SBP", 3) || !strncmp(name, "DSBP", 4))
   {
      mass->AddDomainIntegrator(new DiagMassIntegrator(num_state));
   }
   else
   {
      mass->AddDomainIntegrator(new MassIntegrator());
   }
}

void AbstractSolver::setEssentialBoundaries()
{
   auto &bcs = options["bcs"];

   if (bcs.find("essential") != bcs.end())
   {
      auto tmp = bcs["essential"].get<vector<int>>();
      ess_bdr.SetSize(tmp.size(), 0);
      ess_bdr.Assign(tmp.data());
   }
   /// otherwise mark all attributes as nonessential
   else
   {
      *out << "No essential BCs" << endl;
      if (mesh->bdr_attributes) // some meshes may not have boundary attributes
      {
         *out << "mesh with boundary attributes" << endl;
         ess_bdr.SetSize(mesh->bdr_attributes.Max());
         ess_bdr = 0;
      }
   }
   Array<int> ess_tdof_list;
   fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   res->SetEssentialTrueDofs(ess_tdof_list);
}

void AbstractSolver::solveSteady(ParGridFunction &state)
{
   *out << "AbstractSolver::solveSteady() is deprecated!!!!!!!!!!!!!!" << endl;
   *out << "calling AbstractSolver::solveUnsteady() instead" << endl;
   solveUnsteady(state);
   return;

// #ifdef MFEM_USE_MPI
//    double t1, t2;
//    if (0==rank)
//    {
//       t1 = MPI_Wtime();
//    }
// #ifdef MFEM_USE_PETSC
//    // Get the PetscSolver option
//    *out << "Petsc solver with lu preconditioner.\n";
//    double abstol = options["petscsolver"]["abstol"].get<double>();
//    double reltol = options["petscsolver"]["reltol"].get<double>();
//    int maxiter = options["petscsolver"]["maxiter"].get<int>();
//    int ptl = options["petscsolver"]["printlevel"].get<int>();

//    solver.reset(new mfem::PetscLinearSolver(comm, "solver_", 0));
//    prec.reset(new mfem::PetscPreconditioner(comm, "prec_"));
//    dynamic_cast<mfem::PetscLinearSolver *>(solver.get())->SetPreconditioner(*prec);

//    dynamic_cast<mfem::PetscSolver *>(solver.get())->SetAbsTol(abstol);
//    dynamic_cast<mfem::PetscSolver *>(solver.get())->SetRelTol(reltol);
//    dynamic_cast<mfem::PetscSolver *>(solver.get())->SetMaxIter(maxiter);
//    dynamic_cast<mfem::PetscSolver *>(solver.get())->SetPrintLevel(ptl);
//    *out << "Petsc Solver set.\n";
//    //Get the newton solver options
//    double nabstol = options["newton"]["abstol"].get<double>();
//    double nreltol = options["newton"]["reltol"].get<double>();
//    int nmaxiter = options["newton"]["maxiter"].get<int>();
//    int nptl = options["newton"]["printlevel"].get<int>();
//    newton_solver.reset(new mfem::NewtonSolver(comm));
//    newton_solver->iterative_mode = true;
//    newton_solver->SetSolver(*solver);
//    newton_solver->SetOperator(*res);
//    newton_solver->SetAbsTol(nabstol);
//    newton_solver->SetRelTol(nreltol);
//    newton_solver->SetMaxIter(nmaxiter);
//    newton_solver->SetPrintLevel(nptl);
//    *out << "Newton solver is set.\n";
//    // Solve the nonlinear problem with r.h.s at 0
//    mfem::Vector b;
//    mfem::Vector u_true;
//    u->GetTrueDofs(u_true);
//    newton_solver->Mult(b, u_true);
//    MFEM_VERIFY(newton_solver->GetConverged(), "Newton solver did not converge.");
//    u->SetFromTrueDofs(u_true);
// #else
//    // Hypre solver section
//    if (newton_solver == nullptr)
//       constructNewtonSolver();

//    // Solve the nonlinear problem with r.h.s at 0
//    mfem::Vector b;
//    //mfem::Vector u_true;
//    HypreParVector *u_true = u->GetTrueDofs();
//    //HypreParVector b(*u_true);
//    //u->GetTrueDofs(u_true);
//    newton_solver->Mult(b, *u_true);
//    MFEM_VERIFY(newton_solver->GetConverged(), "Newton solver did not converge.");
//    //u->SetFromTrueDofs(u_true);
//    u->SetFromTrueDofs(*u_true);
// #endif
//    if (0==rank)
//    {
//       t2 = MPI_Wtime();
//       *out << "Time for solving nonlinear system is " << (t2 - t1) << endl;
//    }
// #else
//    solver.reset(new UMFPackSolver());
//    dynamic_cast<UMFPackSolver*>(solver.get())->Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
//    dynamic_cast<UMFPackSolver*>(solver.get())->SetPrintLevel(2);
//    //Get the newton solver options
//    double nabstol = options["newton"]["abstol"].get<double>();
//    double nreltol = options["newton"]["reltol"].get<double>();
//    int nmaxiter = options["newton"]["maxiter"].get<int>();
//    int nptl = options["newton"]["printlevel"].get<int>();
//    newton_solver.reset(new mfem::NewtonSolver());
//    newton_solver->iterative_mode = true;
//    newton_solver->SetSolver(*solver);
//    newton_solver->SetOperator(*res);
//    newton_solver->SetAbsTol(nabstol);
//    newton_solver->SetRelTol(nreltol);
//    newton_solver->SetMaxIter(nmaxiter);
//    newton_solver->SetPrintLevel(nptl);
//    std::cout << "Newton solver is set.\n";
//    // Solve the nonlinear problem with r.h.s at 0
//    mfem::Vector b;
//    mfem::Vector u_true;
//    u->GetTrueDofs(u_true);
//    newton_solver->Mult(b, u_true);
//    MFEM_VERIFY(newton_solver->GetConverged(), "Newton solver did not converge.");
//    u->SetFromTrueDofs(u_true);
// #endif // MFEM_USE_MPI
}

void AbstractSolver::solveUnsteady(ParGridFunction &state)
{
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
      state.Save(osol);
   }

   printSolution("init");

   double t_final = options["time-dis"]["t-final"].template get<double>();
   *out << "t_final is " << t_final << '\n';

   int ti;
   bool done = false;
   double dt = 0.0;
   initialHook();
   for (ti = 0; ti < options["time-dis"]["max-iter"].get<int>(); ++ti)
   {
      dt = calcStepSize(ti, t, t_final, dt);
      *out << "iter " << ti << ": time = " << t << ": dt = " << dt;
      if (!options["time-dis"]["steady"].get<bool>())
         *out << " (" << round(100 * t / t_final) << "% complete)";
      *out << endl;
      iterationHook(ti, t, dt);
      HypreParVector *u_true = state.GetTrueDofs();
      ode_solver->Step(*u_true, t, dt);
      *u = *u_true;
      if (iterationExit(ti, t, t_final, dt)) break;
   }
   terminalHook(ti, t);

   // Save the final solution. This output can be viewed later using GLVis:
   // glvis -m unitGridTestMesh.msh -g adv-final.gf".
   {
      ofstream osol("final.gf");
      osol.precision(precision);
      state.Save(osol);
   }
   // write the solution to vtk file
   if (options["space-dis"]["basis-type"].template get<string>() == "csbp")
   {
      ofstream sol_ofs("final_cg.vtk");
      sol_ofs.precision(14);
      mesh->PrintVTK(sol_ofs, options["space-dis"]["degree"].template get<int>() + 1);
      state.SaveVTK(sol_ofs, "Solution", options["space-dis"]["degree"].template get<int>() + 1);
      sol_ofs.close();
      printSolution("final");
   }
   else if (options["space-dis"]["basis-type"].template get<string>() == "dsbp")
   {
      ofstream sol_ofs("final_dg.vtk");
      sol_ofs.precision(14);
      mesh->PrintVTK(sol_ofs, options["space-dis"]["degree"].template get<int>() + 1);
      state.SaveVTK(sol_ofs, "Solution", options["space-dis"]["degree"].template get<int>() + 1);
      sol_ofs.close();
      printSolution("final");
   }
   // TODO: These mfem functions do not appear to be parallelized
}

void AbstractSolver::solveSteadyAdjoint(const std::string &fun)
{
   double time_beg, time_end;
   time_beg = MPI_Wtime();

   // Step 0: allocate the adjoint variable
   adj.reset(new GridFunType(fes.get()));

   // Step 1: get the right-hand side vector, dJdu, and make an appropriate
   // alias to it, the state, and the adjoint
   std::unique_ptr<GridFunType> dJdu(new GridFunType(fes.get()));

   HypreParVector *u_true = u->GetTrueDofs();
   HypreParVector *dJdu_true = dJdu->GetTrueDofs();
   HypreParVector *adj_true = adj->GetTrueDofs();
   output.at(fun).Mult(*u_true, *dJdu_true);

   // Step 2: get the Jacobian and transpose it
   Operator *jac = &res->GetGradient(*u_true);
   const Operator *jac_trans =
       dynamic_cast<const HypreParMatrix *>(jac)->Transpose();
   MFEM_VERIFY(jac_trans, "Jacobian must be a HypreParMatrix!");

   // Step 3: Solve the adjoint problem
   *out << "Solving adjoint problem:\n"
        << "\tsolver: HypreGMRES\n"
        << "\tprec. : Euclid ILU" << endl;
   constructLinearSolver(options["adj-solver"]);
   solver->SetOperator(*jac_trans);
   solver->Mult(*dJdu_true, *adj_true);

   // check that adjoint residual is small
   std::unique_ptr<GridFunType> adj_res(new GridFunType(fes.get()));
   double res_norm = 0;
   HypreParVector *adj_res_true = adj_res->GetTrueDofs();
   jac_trans->Mult(*adj_true, *adj_res_true);
   *adj_res_true -= *dJdu_true;
   double loc_norm = (*adj_res_true)*(*adj_res_true);
   MPI_Allreduce(&loc_norm, &res_norm, 1, MPI_DOUBLE, MPI_SUM, comm);
   res_norm = sqrt(res_norm);
   *out << "Adjoint residual norm = " << res_norm << endl;
   adj->SetFromTrueDofs(*adj_true);
 
   time_end = MPI_Wtime();
   *out << "Time for solving adjoint is " << (time_end - time_beg) << endl;
}

void AbstractSolver::constructLinearSolver(nlohmann::json &_options)
{
   std::string prec_type = _options["pctype"].get<std::string>();
   std::string solver_type = _options["type"].get<std::string>();

   if (prec_type == "hypreeuclid")
   {
      prec.reset(new HypreEuclid(comm));
      // TODO: need to add HYPRE_EuclidSetLevel to odl branch of mfem
      *out << "!!!!!!! Euclid Fill level is not set "
           << "(see AbstractSolver::constructLinearSolver() for details)" << endl;
      //int fill = options["lin-solver"]["filllevel"].get<int>();
      //HYPRE_EuclidSetLevel(dynamic_cast<HypreEuclid*>(prec.get())->GetPrec(), fill);
   }
   else if (prec_type == "hypreams")
   {
      prec.reset(new HypreAMS(fes.get()));
      dynamic_cast<mfem::HypreAMS *>(prec.get())->SetPrintLevel(0);
      dynamic_cast<mfem::HypreAMS *>(prec.get())->SetSingularProblem();
   }
   else if (prec_type == "hypreboomeramg")
   {
      prec.reset(new HypreBoomerAMG());
      dynamic_cast<mfem::HypreBoomerAMG *>(prec.get())->SetPrintLevel(0);
   }
   else if (prec_type == "blockilu")
   {
      prec.reset(new BlockILU(getNumState()));
   }
   else
   {
      throw MachException("Unsupported preconditioner type!\n"
         "\tavilable options are: HypreEuclid, HypreAMS, HypreBoomerAMG.\n");
   }

   if (solver_type == "hypregmres")
   {
      solver.reset(new HypreGMRES(comm));
   }
   else if (solver_type == "gmressolver")
   {
      solver.reset(new GMRESSolver(comm));
   }
   else if (solver_type == "hyprepcg")
   {
      solver.reset(new HyprePCG(comm));
   }
   else if (solver_type == "cgsolver")
   {
      solver.reset(new CGSolver(comm));
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
   if (solver == nullptr)
      constructLinearSolver(options["lin-solver"]);

   double nabstol = options["newton"]["abstol"].get<double>();
   double nreltol = options["newton"]["reltol"].get<double>();
   int nmaxiter = options["newton"]["maxiter"].get<int>();
   int nptl = options["newton"]["printlevel"].get<int>();
   newton_solver.reset(new mfem::NewtonSolver(comm));
   //double eta = 1e-1;
   //newton_solver.reset(new InexactNewton(comm, eta));
   newton_solver->iterative_mode = true;
   newton_solver->SetSolver(*solver);
   // newton_solver->SetOperator(*res);
   newton_solver->SetPrintLevel(nptl);
   newton_solver->SetRelTol(nreltol);
   newton_solver->SetAbsTol(nabstol);
   newton_solver->SetMaxIter(nmaxiter);
}

void AbstractSolver::setIterSolverOptions(nlohmann::json &_options)
{
   std::string solver_type = _options["type"].get<std::string>();

   double reltol = _options["reltol"].get<double>();
   int maxiter = _options["maxiter"].get<int>();
   int ptl = _options["printlevel"].get<int>();

   if (solver_type == "hypregmres")
   {
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
   evolver.reset(new MachEvolver(ess_bdr, nonlinear_mass.get(), mass.get(),
                                 res.get(), stiff.get(), load.get(), ent.get(),
                                 *out, 0.0,
                                 TimeDependentOperator::Type::IMPLICIT));
   if (newton_solver == nullptr)
      constructNewtonSolver();
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
         *out << "Did not find " << fun << " in output map?" << endl;
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
   HypreParVector *u_p = u_plus.GetTrueDofs();
   HypreParVector *u_m = u_minus.GetTrueDofs();
   HypreParVector *res_p = res_plus.GetTrueDofs();
   HypreParVector *res_m = res_minus.GetTrueDofs();
   res->Mult(*u_p, *res_p);
   res->Mult(*u_m, *res_m);
   res_plus.SetFromTrueDofs(*res_p);
   res_minus.SetFromTrueDofs(*res_m);
   // res_plus = 1/(2*delta)*(res_plus - res_minus)
   subtract(1/(2*delta), res_plus, res_minus, res_plus);

   // Get the product directly using Jacobian from GetGradient
   GridFunType jac_v(fes.get());
   HypreParVector *u_true = u->GetTrueDofs();
   HypreParVector *pert = pert_vec.GetTrueDofs();
   HypreParVector *prod = jac_v.GetTrueDofs();
   mfem::Operator &jac = res->GetGradient(*u_true);
   jac.Mult(*pert, *prod);
   jac_v.SetFromTrueDofs(*prod);

   // check the difference norm
   jac_v -= res_plus;
   double error = calcInnerProduct(jac_v, jac_v);
   *out << "The Jacobian product error norm is " << sqrt(error) << endl;
}

} // namespace mach
