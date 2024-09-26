#include "solver.hpp"

#include <fstream>
#include <iostream>
#include <cmath>
#include <memory>
#include <utility>

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
#endif  // MFEM_USE_SIMMETRIX

#ifdef MFEM_USE_EGADS
#include "gmi_egads.h"
#endif  // MFEM_USE_EGADS

#endif  // MFEM_USE_PUMI

#include "utils.hpp"
#include "mfem_extensions.hpp"
#include "default_options.hpp"
#include "sbp_fe.hpp"
#include "evolver.hpp"
#include "diag_mass_integ.hpp"
#include "material_library.hpp"
#include "miso_input.hpp"
#include "miso_integrator.hpp"
#include "miso_load.hpp"

using namespace std;
using namespace mfem;

namespace miso
{
AbstractSolver::AbstractSolver(const nlohmann::json &file_options,
                               unique_ptr<Mesh> smesh,
                               MPI_Comm comm)
 : diff_stack(getDiffStack())
{
   initBase(file_options, move(smesh), comm);
}

// Note: the following constructor is protected
// AbstractSolver::AbstractSolver(const string &opt_file_name, MPI_Comm incomm)
//  : out(getOutStream(rank))
// {
//    // Some of the following code would normally happen in initBase, but this
//    is
//    // a parred down version of the AbstractSolver that does not need most of
//    // the functionality (e.g. multiphysics code)
//    // TODO: Do we need a separate super class for this case?

//    // Set the options; the defaults are overwritten by the values in the file
//    // using the merge_patch method
//    nlohmann::json file_options;
//    ifstream options_file(opt_file_name);
//    options_file >> file_options;
//    options = default_options;
//    options.merge_patch(file_options);

//    bool silent = options.value("silent", false);
//    out = getOutStream(rank, silent);
//    *out << setw(3) << options << endl;

//    // comm = incomm;
//    MPI_Comm_dup(incomm, &comm);
//    MPI_Comm_rank(comm, &rank);
// }

void AbstractSolver::initBase(const nlohmann::json &file_options,
                              std::unique_ptr<Mesh> smesh,
                              MPI_Comm incomm)
{
   // Set the options; the defaults are overwritten by the values in the file
   // using the merge_patch method
   // comm = incomm;
   MPI_Comm_dup(incomm, &comm);
   MPI_Comm_rank(comm, &rank);
   options = default_options;
   options.merge_patch(file_options);

   bool silent = options.value("silent", false);
   out = getOutStream(rank, silent);

   if (options["print-options"])
   {
      *out << setw(3) << options << endl;
   }

   materials = material_library;

   constructMesh(move(smesh));
   // mesh->EnsureNodes();
   // mesh_fes = static_cast<SpaceType*>(mesh->GetNodes()->FESpace());
   // /// before internal boundaries are removed
   // ess_bdr.SetSize(mesh->bdr_attributes.Max());
   // ess_bdr = 1;
   // /// get all dofs on model surfaces
   // mesh_fes->GetEssentialTrueDofs(ess_bdr, mesh_fes_surface_dofs);
   int dim = mesh->Dimension();
   *out << "problem space dimension = " << dim << endl;
   // Define the ODE solver used for time integration (possibly not used)
   ode_solver = nullptr;
   *out << "ode-solver type = "
        << options["time-dis"]["ode-solver"].template get<string>() << endl;
   if (options["time-dis"]["ode-solver"].template get<string>() == "RK1")
   {
      ode_solver = std::make_unique<ForwardEulerSolver>();
   }
   else if (options["time-dis"]["ode-solver"].template get<string>() == "RK4")
   {
      ode_solver = std::make_unique<RK4Solver>();
   }
   else if (options["time-dis"]["ode-solver"].template get<string>() ==
            "MIDPOINT")
   {
      ode_solver = std::make_unique<ImplicitMidpointSolver>();
   }
   else if (options["time-dis"]["ode-solver"].get<string>() == "RRK")
   {
      ode_solver = std::make_unique<RRKImplicitMidpointSolver>(out);
   }
   else if (options["time-dis"]["ode-solver"].template get<string>() == "PTC")
   {
      ode_solver = std::make_unique<PseudoTransientSolver>(out);
   }
   else
   {
      throw MISOException(
          "Unknown ODE solver type " +
          options["time-dis"]["ode-solver"].template get<string>());
      // TODO: parallel exit
   }

   // Refine the mesh here, or have a separate member function?
   for (int l = 0; l < options["mesh"]["refine"].template get<int>(); l++)
   {
      mesh->UniformRefinement();
   }
}

void AbstractSolver::initDerived()
{
   int dim = mesh->Dimension();
   int fe_order = options["space-dis"]["degree"].template get<int>();
   std::string basis_type =
       options["space-dis"]["basis-type"].template get<string>();
   bool galerkin_diff = options["space-dis"].value("GD", false);
   // Define the SBP elements and finite-element space; eventually, we will want
   // to have a case or if statement here for both CSBP and DSBP, and (?)
   // standard FEM. and here it is for first two
   if (basis_type == "csbp")
   {
      fec = std::make_unique<SBPCollection>(fe_order, dim);
   }
   else if (basis_type == "dsbp" || galerkin_diff)
   {
      fec = std::make_unique<DSBPCollection>(fe_order, dim);
   }
   else if (basis_type == "nedelec")
   {
      fec = std::make_unique<ND_FECollection>(fe_order, dim);
   }
   else if (basis_type == "H1")
   {
      fec = std::make_unique<H1_FECollection>(fe_order, dim);
   }

   // define the number of states, the fes, and the state grid function
   num_state = this->getNumState();  // <--- this is a virtual fun
   *out << "Num states = " << num_state << endl;
   fes = std::make_unique<SpaceType>(
       mesh.get(), fec.get(), num_state, Ordering::byVDIM);
   /// we'll stop using `u` eventually
   /// start creating your own state vector with `getNewField`
   u = std::make_unique<GridFunType>(fes.get());
   *out << "Number of finite element unknowns: " << fes->GlobalTrueVSize()
        << endl;

   /// initialize scratch work vectors
   scratch = std::make_unique<ParGridFunction>(fes.get());
   scratch_tv = std::make_unique<HypreParVector>(fes.get());

   double alpha = 1.0;

   setUpExternalFields();

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
      addLoadVolumeIntegrators(alpha);
      addLoadBoundaryIntegrators(alpha);
      addLoadInterfaceIntegrators(alpha);
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

   setEssentialBoundaries();

   // // add the output functional QoIs
   // auto &fun = options["outputs"];
   // using json_iter = nlohmann::json::iterator;
   // int num_bndry_outputs = 0;
   // for (json_iter it = fun.begin(); it != fun.end(); ++it) {
   //    if (it->is_array()) ++num_bndry_outputs;
   // }
   // output_bndry_marker.resize(num_bndry_outputs);

   prec = constructPreconditioner(options["lin-prec"]);
   solver = constructLinearSolver(options["lin-solver"], *prec);
   newton_solver = constructNonlinearSolver(options["nonlin-solver"], *solver);
   constructEvolver();
}

AbstractSolver::~AbstractSolver()
{
   *out << "Deleting Abstract Solver..." << endl;

#ifdef MFEM_USE_PUMI
   if (pumi_mesh)
   {
      if (!PCU_previously_initialized)
      {
         PCU_Comm_Free();
      }
   }
#ifdef MFEM_USE_SIMMETRIX
   gmi_sim_stop();
   Sim_unregisterAllKeys();
#endif  // MFEM_USE_SIMMETRIX

#ifdef MFEM_USE_EGADS
   gmi_egads_stop();
#endif  // MFEM_USE_EGADS
#endif

   MPI_Comm_free(&comm);
}

void AbstractSolver::constructMesh(unique_ptr<Mesh> smesh)
{
   std::string mesh_file = options["mesh"]["file"].template get<string>();
   std::string mesh_ext;
   size_t i = mesh_file.rfind('.', mesh_file.length());
   if (i != string::npos)
   {
      mesh_ext = (mesh_file.substr(i + 1, mesh_file.length() - i));
   }
   else
   {
      throw MISOException(
          "AbstractSolver::constructMesh(smesh)\n"
          "\tMesh file has no extension!\n");
   }

   // if serial mesh passed in, use that
   if (smesh != nullptr)
   {
      mesh = std::make_unique<MeshType>(comm, *smesh);
   }
   // native MFEM mesh
   else if (mesh_ext == "mesh")
   {
      // read in the serial mesh
      smesh = std::make_unique<Mesh>(mesh_file.c_str(), 1, 1);
      mesh = std::make_unique<MeshType>(comm, *smesh);
   }
   // PUMI mesh
   else if (mesh_ext == "smb")
   {
      constructPumiMesh();
   }
   mesh->EnsureNodes();

   removeInternalBoundaries();
   *out << "bdr_attr: ";
   mesh->bdr_attributes.Print(*out);
}

void AbstractSolver::constructPumiMesh()
{
#ifdef MFEM_USE_PUMI  // if using pumi mesh
   *out << options["mesh"]["model-file"].get<string>().c_str() << std::endl;
   std::string model_file = options["mesh"]["model-file"].get<string>();
   std::string mesh_file = options["mesh"]["file"].get<string>();
   if (PCU_Comm_Initialized())
   {
      PCU_previously_initialized = true;
   }
   if (!PCU_previously_initialized)
   {
      PCU_Comm_Init();
   }
   PCU_Switch_Comm(comm);
#ifdef MFEM_USE_SIMMETRIX
   Sim_readLicenseFile(0);
   gmi_sim_start();
   gmi_register_sim();
#endif
#ifdef MFEM_USE_EGADS
   gmi_register_egads();
   gmi_egads_start();
#endif
   gmi_register_mesh();
   pumi_mesh.reset(apf::loadMdsMesh(model_file.c_str(), mesh_file.c_str()));
   // int mesh_dim = pumi_mesh->getDimension();
   // int nEle = pumi_mesh->count(mesh_dim);
   // int ref_levels = (int)floor(log(10000. / nEle) / log(2.) / mesh_dim);
   // Perform Uniform refinement
   // if (ref_levels > 1)
   // {
   //    ma::Input* uniInput = ma::configureUniformRefine(pumi_mesh,
   //    ref_levels); ma::adapt(uniInput);
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

   apf::Numbering *aux_num = apf::createNumbering(
       pumi_mesh.get(), "aux_numbering", pumi_mesh->getShape(), 1);

   apf::MeshIterator *it = pumi_mesh->begin(0);
   apf::MeshEntity *v = nullptr;
   int count = 0;
   while ((v = pumi_mesh->iterate(it)) != nullptr)
   {
      apf::number(aux_num, v, 0, 0, count++);
   }
   pumi_mesh->end(it);

   mesh = std::make_unique<ParPumiMesh>(comm, pumi_mesh.get());

   // it = pumi_mesh->begin(pumi_mesh->getDimension());
   // count = 0;
   // while ((v = pumi_mesh->iterate(it)))
   // {
   // //   if (count > 10) break;
   // // //   printf("at element %d =========\n", count);
   // //   if (isBoundaryTet(pumi_mesh.get(), v))
   // //    //  printf("tet is connected to the boundary\n");
   // //   else
   //    //  printf("tet is NOT connected to the boundary\n");
   // //   apf::MeshEntity* dvs[12];
   // //   int nd = pumi_mesh->getDownward(v, 0, dvs);
   // //   for (int i = 0; i < nd; i++) {
   // //    //  int id = apf::getNumber(aux_num, dvs[i], 0, 0);
   // //    //  printf("%d ", id);
   // //   }
   // //   printf("\n");
   // //   Array<int> mfem_vs;
   // //   mesh->GetElementVertices(count, mfem_vs);
   // //   for (int i = 0; i < mfem_vs.Size(); i++) {
   // //    //  printf("%d ", mfem_vs[i]);
   // //   }
   // // //   printf("\n");
   // //   printf("=========\n");
   // //   count++;
   // }

   /// Add attributes based on reverse classification
   // Boundary faces
   int dim = mesh->Dimension();
   apf::MeshIterator *itr = pumi_mesh->begin(dim - 1);
   apf::MeshEntity *ent = nullptr;
   int ent_cnt = 0;
   while ((ent = pumi_mesh->iterate(itr)) != nullptr)
   {
      apf::ModelEntity *me = pumi_mesh->toModel(ent);
      if (pumi_mesh->getModelType(me) == (dim - 1))
      {
         // Get tag from model by  reverse classification
         int tag = pumi_mesh->getModelTag(me);
         (mesh->GetBdrElement(ent_cnt))->SetAttribute(tag);
         ent_cnt++;
      }
   }
   pumi_mesh->end(itr);

   // Volume faces
   itr = pumi_mesh->begin(dim);
   ent_cnt = 0;
   while ((ent = pumi_mesh->iterate(itr)) != nullptr)
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
   throw MISOException(
       "AbstractSolver::constructPumiMesh()\n"
       "\tMFEM was not built with PUMI!\n"
       "\trecompile MFEM with PUMI\n");
#endif  // MFEM_USE_PUMI
}

void AbstractSolver::setInitialCondition(
    ParGridFunction &state,
    const std::function<double(const mfem::Vector &)> &u_init)
{
   // state = 0.0;
   FunctionCoefficient u0(u_init);
   state.ProjectCoefficient(u0);
   // state.ProjectBdrCoefficient(u0, ess_bdr);
   // std::cout << "ess_bdr: "; ess_bdr.Print();
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

void AbstractSolver::setFieldValue(HypreParVector &field, const double u_init)
{
   ConstantCoefficient u0(u_init);
   scratch->ProjectCoefficient(u0);
   scratch->GetTrueDofs(field);
}

void AbstractSolver::setFieldValue(double *field_buffer, const double u_init)
{
   auto field = bufferToHypreParVector(field_buffer, *fes);
   setFieldValue(field, u_init);
}

void AbstractSolver::setFieldValue(
    HypreParVector &field,
    const std::function<double(const Vector &)> &u_init)
{
   FunctionCoefficient u0(u_init);
   scratch->ProjectCoefficient(u0);
   scratch->GetTrueDofs(field);
}

void AbstractSolver::setFieldValue(
    double *field_buffer,
    const std::function<double(const mfem::Vector &)> &u_init)
{
   auto field = bufferToHypreParVector(field_buffer, *fes);
   setFieldValue(field, u_init);
}

void AbstractSolver::setFieldValue(HypreParVector &field, const Vector &u_init)
{
   VectorConstantCoefficient u0(u_init);
   scratch->ProjectCoefficient(u0);
   scratch->GetTrueDofs(field);
}

void AbstractSolver::setFieldValue(double *field_buffer,
                                   const mfem::Vector &u_init)
{
   auto field = bufferToHypreParVector(field_buffer, *fes);
   setFieldValue(field, u_init);
}

void AbstractSolver::setFieldValue(
    HypreParVector &field,
    const std::function<void(const Vector &, Vector &)> &u_init)
{
   VectorFunctionCoefficient u0(num_state, u_init);
   scratch->ProjectCoefficient(u0);
   scratch->GetTrueDofs(field);
}

void AbstractSolver::setFieldValue(
    double *field_buffer,
    const std::function<void(const mfem::Vector &, mfem::Vector &)> &u_init)
{
   auto field = bufferToHypreParVector(field_buffer, *fes);
   setFieldValue(field, u_init);
}

double AbstractSolver::calcInnerProduct(const GridFunType &x,
                                        const GridFunType &y) const
{
   ParFiniteElementSpace *fe_space = x.ParFESpace();
   const char *name = fe_space->FEColl()->Name();

   double loc_prod = 0.0;
   const FiniteElement *fe = nullptr;
   ElementTransformation *T = nullptr;
   DenseMatrix x_vals;
   DenseMatrix y_vals;
   // calculate the L2 inner product for component index `entry`
   for (int i = 0; i < fe_space->GetNE(); i++)
   {
      fe = fe_space->GetFE(i);
      const IntegrationRule *ir = nullptr;
      if ((strncmp(name, "SBP", 3) == 0) || (strncmp(name, "DSBP", 4) == 0))
      {
         ir = &(fe->GetNodes());
      }
      else
      {
         int intorder = 2 * fe->GetOrder() + 1;
         ir = &(IntRules.Get(fe->GetGeomType(), intorder));
      }
      T = fe_space->GetElementTransformation(i);
      x.GetVectorValues(*T, *ir, x_vals);
      y.GetVectorValues(*T, *ir, y_vals);
      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         T->SetIntPoint(&ip);
         double node_prod = 0.0;
         for (int n = 0; n < num_state; ++n)
         {
            node_prod += x_vals(n, j) * y_vals(n, j);
         }
         loc_prod += ip.weight * T->Weight() * node_prod;
      }
   }
   double prod = NAN;
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

   const char *name = fe_space->FEColl()->Name();

   double loc_norm = 0.0;
   const FiniteElement *fe = nullptr;
   ElementTransformation *T = nullptr;
   Vector shape;
   Array<int> vdofs;

   // sum up the L2 error over all states
   for (int i = 0; i < fe_space->GetNE(); i++)
   {
      fe = fe_space->GetFE(i);
      const IntegrationRule *ir = nullptr;
      if ((strncmp(name, "SBP", 3) == 0) || (strncmp(name, "DSBP", 4) == 0))
      {
         ir = &(fe->GetNodes());
      }
      else
      {
         int intorder = 2 * fe->GetOrder() + 1;
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
         {
            if (vdofs[k] >= 0)
            {
               a += (*field)(vdofs[k]) * shape(k);
            }
            else
            {
               a -= (*field)(-1 - vdofs[k]) * shape(k);
            }
         }
         T->SetIntPoint(&ip);
         a -= exsol.Eval(*T, ip);
         loc_norm += ip.weight * T->Weight() * a * a;
      }
   }

   double norm = NAN;
   MPI_Allreduce(&loc_norm, &norm, 1, MPI_DOUBLE, MPI_SUM, comm);
   if (norm < 0.0)  // This was copied from mfem...should not happen for us
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

   const char *name = fe_space->FEColl()->Name();

   double loc_norm = 0.0;
   const FiniteElement *fe = nullptr;
   ElementTransformation *T = nullptr;
   DenseMatrix vals;
   DenseMatrix exact_vals;
   Vector loc_errs;

   if (entry < 0)
   {
      // sum up the L2 error over all states
      for (int i = 0; i < fe_space->GetNE(); i++)
      {
         fe = fe_space->GetFE(i);
         const IntegrationRule *ir = nullptr;
         if ((strncmp(name, "SBP", 3) == 0) || (strncmp(name, "DSBP", 4) == 0))
         {
            ir = &(fe->GetNodes());
         }
         else
         {
            int intorder = 2 * fe->GetOrder() + 1;
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
         const IntegrationRule *ir = nullptr;
         if ((strncmp(name, "SBP", 3) == 0) || (strncmp(name, "DSBP", 4) == 0))
         {
            ir = &(fe->GetNodes());
         }
         else
         {
            int intorder = 2 * fe->GetOrder() + 1;
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
   double norm = NAN;
   MPI_Allreduce(&loc_norm, &norm, 1, MPI_DOUBLE, MPI_SUM, comm);
   if (norm < 0.0)  // This was copied from mfem...should not happen for us
   {
      return -sqrt(-norm);
   }
   return sqrt(norm);
}

double AbstractSolver::calcL2Error(
    mfem::HypreParVector &field,
    const std::function<double(const mfem::Vector &)> &u_exact)
{
   *scratch = field;
   return calcL2Error(scratch.get(), u_exact);
}

double AbstractSolver::calcL2Error(
    mfem::HypreParVector &field,
    const std::function<void(const mfem::Vector &, mfem::Vector &)> &u_exact,
    int entry)
{
   *scratch = field;
   return calcL2Error(scratch.get(), u_exact, entry);
}

double AbstractSolver::calcResidualNorm() const
{
   HypreParVector u_true(fes.get());
   u->GetTrueVector().SetDataAndSize(u_true.GetData(), u_true.Size());
   u->SetTrueVector();

   return calcResidualNorm(*u);
}
double AbstractSolver::calcResidualNorm(const ParGridFunction &state) const
{
   MISOInputs inputs{{"state", state.GetTrueVector()}};

   calcResidual(inputs, *scratch_tv);
   return std::sqrt(InnerProduct(comm, *scratch_tv, *scratch_tv));
}

// std::unique_ptr<ParGridFunction> AbstractSolver::getNewField(
//    double *data)
// {
//    if (data == nullptr)
//    {
//       auto gf = std::unique_ptr<ParGridFunction>(
//          new ParGridFunction(fes.get()));

//       *gf = 0.0;
//       return gf;
//    }
//    else
//    {
//       auto gf = std::unique_ptr<ParGridFunction>(
//          new ParGridFunction(fes.get(), data));

//       return gf;
//    }
// }

std::unique_ptr<HypreParVector> AbstractSolver::getNewField(double *data)
{
   if (data == nullptr)
   {
      auto field = std::make_unique<HypreParVector>(fes.get());

      *field = 0.0;
      return field;
   }
   else
   {
      auto field = std::make_unique<HypreParVector>(fes->GetComm(),
                                                    fes->GlobalTrueVSize(),
                                                    data,
                                                    fes->GetTrueDofOffsets());
      return field;
   }
}

/// NOTE: the load vectors must have previously been assembled
void AbstractSolver::calcResidual(const ParGridFunction &state,
                                  ParGridFunction &residual) const
{
   auto *u_true = state.GetTrueDofs();
   auto *r_true = residual.GetTrueDofs();
   res->Mult(*u_true, *r_true);
   if (load)
   {
      // auto load_lf = dynamic_cast<ParLinearForm*>(load.get());
      // if (load_lf)
      // {
      //    auto *l_true = load_lf->ParallelAssemble();
      //    *r_true += *l_true;
      //    delete l_true;
      // }
      // auto load_gf = dynamic_cast<ParGridFunction*>(load.get());
      // if (load_gf)
      // {
      //    auto *l_true = load_gf->GetTrueDofs();
      //    *r_true += *l_true;
      //    delete l_true;
      // }
   }
   residual.SetFromTrueDofs(*r_true);
   delete u_true;
   delete r_true;
}

void AbstractSolver::calcResidual(const MISOInputs &inputs,
                                  double *res_buffer) const
{
   auto residual = bufferToHypreParVector(res_buffer, *fes);
   calcResidual(inputs, residual);
}

void AbstractSolver::calcResidual(const MISOInputs &inputs,
                                  HypreParVector &residual) const
{
   // setInputs(*res, inputs); // once I've added MISONonlinearForm

   /// this approach would require communication twice, once to set the tdofs
   ///   and then again inside of res->Mult
   // auto &state_gf = res_fields.at("state");
   // state_gf.GetTrueVector().SetDataAndSize(inputs.at("state").getField(),
   //                                         res.Size());
   // state_gf.SetFromTrueVector();
   // auto &state = state_gf.GetTrueVector();

   // this only communicates once inside of res->Mult to distribute state
   // create HypreParVector that contains the data from the state input
   // auto state = bufferToHypreParVector(inputs.at("state").getField(), *fes);
   mfem::Vector state;
   setVectorFromInputs(inputs, "state", state, false, true);

   res->Mult(state, residual);

   if (load)
   {
      miso::setInputs(*load, inputs);
      miso::addLoad(*load, residual);
   }
}

void AbstractSolver::linearize(const MISOInputs &inputs)
{
   setInputs(res_integrators, inputs);
   if (load)
   {
      miso::setInputs(*load, inputs);
   }

   /// something like this...
   // state_jac = evolver->GetGradient();
}

double AbstractSolver::vectorJacobianProduct(double *res_bar_buffer,
                                             const std::string &wrt)
{
   auto res_bar = bufferToHypreParVector(res_bar_buffer, *fes);
   return vectorJacobianProduct(res_bar, wrt);
}

double AbstractSolver::vectorJacobianProduct(const HypreParVector &res_bar,
                                             const std::string &wrt)
{
   double wrt_bar = 0.0;
   if (res_scalar_sens.count(wrt) != 0)
   {
      auto &state = res_fields.at("state");
      wrt_bar += res_scalar_sens.at(wrt).GetEnergy(state);
   }
   if (load)
   {
      wrt_bar += miso::vectorJacobianProduct(*load, res_bar, wrt);
   }
   return wrt_bar;
}

void AbstractSolver::vectorJacobianProduct(double *res_bar_buffer,
                                           const std::string &wrt,
                                           double *wrt_bar_buffer)
{
   auto res_bar = bufferToHypreParVector(res_bar_buffer, *fes);

   auto &wrt_fes = *res_fields.at(wrt).ParFESpace();
   auto wrt_bar = bufferToHypreParVector(wrt_bar_buffer, wrt_fes);

   vectorJacobianProduct(res_bar, wrt, wrt_bar);
}

void AbstractSolver::vectorJacobianProduct(const HypreParVector &res_bar,
                                           const std::string &wrt,
                                           HypreParVector &wrt_bar)
{
   res_fields.at("res_bar") = res_bar;

   if (wrt == "state")
   {
      // throw std::runtime_error("vectorJacobianProduct not supported for "
      //                          "state derivative!\n");
      std::cerr << "WARNING: vectorJacobianProduct not supported for ";
      std::cerr << "state derivative!\n";
   }
   else
   {
      res_sens.at(wrt).Assemble();

      /// ParallelAssemble overwrites its argument, so to accumulate into
      /// wrt_bar we need a second vector stored in ext_tv to assemble into,
      /// then we add it to wrt_bar
      ext_tvs.emplace(wrt + "_bar", res_fields.at(wrt).ParFESpace());
      auto &temp_wrt_bar = ext_tvs.at(wrt + "_bar");
      res_sens.at(wrt).ParallelAssemble(temp_wrt_bar);
      wrt_bar += temp_wrt_bar;
   }
}

double AbstractSolver::calcStepSize(int iter,
                                    double t,
                                    double t_final,
                                    double dt_old,
                                    const ParGridFunction &state) const
{
   auto dt = options["time-dis"]["dt"].get<double>();
   dt = min(dt, t_final - t);
   return dt;
}

bool AbstractSolver::iterationExit(int iter,
                                   double t,
                                   double t_final,
                                   double dt,
                                   const ParGridFunction &state) const
{
   return t >= t_final - 1e-14 * dt;
}

void AbstractSolver::printSolution(const std::string &file_name, int refine)
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

void AbstractSolver::printAdjoint(const std::string &file_name, int refine)
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

void AbstractSolver::printResidual(const std::string &file_name, int refine)
{
   GridFunType r(fes.get());
   calcResidual(*u, r);
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
                                 std::vector<ParGridFunction *> fields,
                                 std::vector<std::string> names,
                                 int refine,
                                 int cycle)
{
   if (fields.size() != names.size())
   {
      throw MISOException(
          "Must supply a name for each grid function to print!");
   }
   ParaViewDataCollection paraview_dc(file_name, mesh.get());
   paraview_dc.SetPrefixPath("ParaView");
   if (refine == -1)
   {
      refine = options["space-dis"]["degree"].get<int>() + 1;
   }
   paraview_dc.SetLevelsOfDetail(refine);
   paraview_dc.SetCycle(cycle);
   paraview_dc.SetDataFormat(VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(true);
   paraview_dc.SetTime((double)cycle);  // set the time
   for (unsigned i = 0; i < fields.size(); ++i)
   {
      paraview_dc.RegisterField(names[i], fields[i]);
   }
   paraview_dc.Save();

   // // TODO: These mfem functions do not appear to be parallelized
   // ofstream sol_ofs(file_name + ".vtk");
   // sol_ofs.precision(14);
   // if (refine == -1)
   // {
   //    refine = options["space-dis"]["degree"].get<int>() + 1;
   // }
   // mesh->PrintVTK(sol_ofs, refine);
   // for (unsigned i = 0; i < fields.size(); ++i)
   // {
   //    fields[i]->SaveVTK(sol_ofs, names[i], refine);
   // }
   // sol_ofs.close();
}

std::vector<GridFunType *> AbstractSolver::getFields() { return {u.get()}; }

void AbstractSolver::getField(const std::string &name, double *field_buffer)
{
   auto *field_fes = res_fields.at(name).ParFESpace();
   HypreParVector field(field_fes->GetComm(),
                        field_fes->GlobalTrueVSize(),
                        field_buffer,
                        field_fes->GetTrueDofOffsets());
   getField(name, field);
}

void AbstractSolver::getField(const std::string &name,
                              mfem::HypreParVector &field)
{
   res_fields.at(name).GetTrueDofs(field);
}

std::unique_ptr<HypreParVector> AbstractSolver::getField(
    const std::string &name)
{
   auto &field_gf = res_fields.at(name);
   auto *field_fes = field_gf.ParFESpace();

   auto field = std::make_unique<HypreParVector>(field_fes);
   field_gf.GetTrueDofs(*field);
   return field;
}

void AbstractSolver::setField(const std::string &name,
                              const mfem::HypreParVector &field)
{
   res_fields.at(name).SetFromTrueDofs(field);
}

void AbstractSolver::solveForState(ParGridFunction &state)
{
   HypreParVector state_true(fes.get());
   state.GetTrueVector().SetDataAndSize(state_true.GetData(),
                                        state_true.Size());
   state.SetTrueVector();

   if (options["steady"].get<bool>())
   {
      solveSteady(state);
   }
   else
   {
      solveUnsteady(state);
   }
}

void AbstractSolver::solveForState(const MISOInputs &inputs,
                                   double *state_buffer)
{
   HypreParVector state(fes->GetComm(),
                        fes->GlobalTrueVSize(),
                        state_buffer,
                        fes->GetTrueDofOffsets());

   solveForState(inputs, state);
}

void AbstractSolver::solveForState(const MISOInputs &inputs,
                                   mfem::HypreParVector &state)
{
   // miso::setInputs(*res, inputs);
   if (load)
   {
      miso::setInputs(*load, inputs);
   }

   auto &state_gf = res_fields.at("state");
   state_gf.GetTrueVector().SetDataAndSize(state.GetData(), state.Size());
   state_gf.SetFromTrueVector();

   solveUnsteady(state_gf);
   // this is not necessary if state_gf is unchanged after timestepping
   // unnecessary communication
   state_gf.GetTrueDofs(state);
}

void AbstractSolver::solveForAdjoint(const std::string &fun)
{
   if (options["steady"].get<bool>())
   {
      solveSteadyAdjoint(fun);
   }
   else
   {
      solveUnsteadyAdjoint(fun);
   }
}

void AbstractSolver::setMeshCoordinates(mfem::Vector &coords)
{
   /// Added `EnsureNodes` to `constructMesh`
   // if (mesh->GetNodes() == nullptr)
   // {
   //    // TODO: this approach will have the mesh allocate a `Nodes`
   //    // gridfunction that we immediately replace (wasteful)
   //    mesh->EnsureNodes();
   // }
   // auto mesh_gf = static_cast<ParGridFunction*>(mesh->GetNodes());
   auto *mesh_gf = mesh->GetNodes();
   mfem::Vector diff(coords);
   diff -= *mesh_gf;

   std::cout << "\n-------------------------------\n";
   std::cout << "l2 norm of mesh diff: " << diff.Norml2() << "\n";
   std::cout << "-------------------------------\n";

   std::cout << "input coords: \n";
   coords.Print();
   std::cout << "mesh gf: \n";
   mesh_gf->Print();
   // mesh_gf->MakeRef(coords, 0);
   mesh_gf->MakeRef(coords, 0);
}

void AbstractSolver::removeInternalBoundaries()
{
   auto &prob_opts = options["problem-opts"];
   if (prob_opts.contains("keep-bndrys"))
   {
      /// any string keeps all boundaries, use "all" for convention
      if (prob_opts["keep-bndrys"].type() == nlohmann::json::value_t::string)
      {
         return;
      }
      else if (prob_opts["keep-bndrys"].type() ==
               nlohmann::json::value_t::array)
      {
         const auto &tmp = prob_opts["keep-bndrys"].get<vector<int>>();
         Array<int> keep(tmp.size());
         for (size_t i = 0; i < tmp.size(); ++i)
         {
            keep[i] = tmp[i];
         }
         *out << "keep: ";
         keep.Print(*out);
         mesh->RemoveInternalBoundaries(keep);
         return;
      }
      else
      {
         throw MISOException("Unrecognized entry for keep-bndrys!\n");
      }
   }
   else if (prob_opts.contains("keep-bndrys-adj-to"))
   {
      const auto &tmp = prob_opts["keep-bndrys-adj-to"].get<vector<int>>();
      Array<int> regions(tmp.size());
      for (size_t i = 0; i < tmp.size(); ++i)
      {
         regions[i] = tmp[i];
      }
      *out << "regions: ";
      regions.Print(*out);
      mesh->RemoveInternalBoundariesNotAdjacentTo(regions);
      return;
   }
   else
   {
      mesh->RemoveInternalBoundaries();
      return;
   }
}

/// This approach will only work for fields on the same mesh
void AbstractSolver::setUpExternalFields()
{
   res_fields.emplace("state", fes.get());
   res_fields.emplace("residual", fes.get());
   // give the solver ownership over the mesh coords grid function, and store
   // it in `res_fields` under key "mesh_coords"
   {
      auto &mesh_gf = *dynamic_cast<ParGridFunction *>(mesh->GetNodes());
      ParFiniteElementSpace *mesh_fespace = nullptr;
      auto *mesh_fec = mesh_gf.OwnFEC();
      // if the mesh GF owns its FESpace and FECollection
      if (mesh_fec != nullptr)
      {
         // steal them and tell the mesh GF it no longer owns them
         mesh_fespace = mesh_gf.ParFESpace();
         mesh_gf.MakeOwner(nullptr);
      }
      else
      {
         // else copy the FESpace and get its FECollection
         mesh_fespace = new ParFiniteElementSpace(*mesh_gf.ParFESpace());
         mesh_fec =
             const_cast<FiniteElementCollection *>(mesh_fespace->FEColl());
      }
      // create a new GF for the mesh nodes with the new/stolen FESpace
      res_fields.emplace("mesh_coords", mesh_fespace);

      // tell the new GF it owns the FESpace and FECollection
      res_fields.at("mesh_coords").MakeOwner(mesh_fec);
      // get the values of the new GF to those of the mesh's old nodes
      res_fields.at("mesh_coords") = mesh_gf;
      // tell the mesh to use this GF for its Nodes
      // (and that it doesn't own it)
      mesh->NewNodes(res_fields.at("mesh_coords"), false);
   }
   /// allocate field for adjoint/res_bar for sensitivity integrators to use
   res_fields.emplace("res_bar", fes.get());

   if (options.contains("external-fields"))
   {
      int dim = mesh->Dimension();
      auto &external_fields = options["external-fields"];
      for (auto &f : external_fields.items())
      {
         auto field = f.value();
         std::string name = std::string(f.key());

         /// this approach will only work for fields on the same mesh
         auto order = field["degree"].get<int>();
         auto basis = field["basis-type"].get<std::string>();
         auto num_states = field["num-states"].get<int>();

         FiniteElementCollection *fecoll = nullptr;
         if (basis == "H1")
         {
            fecoll = new H1_FECollection(order, dim, num_state);
         }
         else if (basis == "nedelec")
         {
            fecoll = new ND_FECollection(order, dim, num_state);
         }
         else
         {
            throw MISOException("Unrecognized basis type: " + basis +
                                "!\n"
                                "Known types are:\n"
                                "\tH1\n"
                                "\tnedelec\n");
         }

         auto *fespace =
             new ParFiniteElementSpace(mesh.get(), fecoll, num_states);

         /// constructs a grid function with an empty data array that must
         /// later be set by `setResidualInput`
         res_fields.emplace(name, fespace);
         res_fields.at(name).MakeOwner(fecoll);
      }
   }
}

void AbstractSolver::addMassIntegrators(double alpha)
{
   const char *name = fes->FEColl()->Name();
   if ((strncmp(name, "SBP", 3) == 0) || (strncmp(name, "DSBP", 4) == 0))
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
   // std::cout << "bdr_attributes: "; mesh->bdr_attributes.Print();
   ess_bdr.SetSize(mesh->bdr_attributes.Max());
   // ess_bdr.SetSize(13);
   // *out << "ess_bdr size: " << ess_bdr.Size() << "\n";

   if (bcs.find("essential") != bcs.end())
   {
      ess_bdr = 0;
      try
      {
         auto tmp = bcs["essential"].get<std::vector<int>>();
         for (auto &bdr : tmp)
         {
            ess_bdr[bdr - 1] = 1;
         }
      }
      catch (nlohmann::json::type_error &e)
      {
         auto all = bcs["essential"].get<std::string>();
         if (all == "all")
         {
            ess_bdr = 1;
         }
      }

      // *out << "ess_bdr: "; ess_bdr.Print(*out);
   }
   /// otherwise mark all attributes as nonessential
   else
   {
      *out << "No essential BCs" << endl;
      if (mesh->bdr_attributes !=
          nullptr)  // some meshes may not have boundary attributes
      {
         *out << "mesh with boundary attributes" << endl;
         ess_bdr = 0;
      }
   }
   Array<int> ess_tdof_list;
   fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   if (ess_tdof_list != nullptr)
   {
      res->SetEssentialTrueDofs(ess_tdof_list);
   }
}

void AbstractSolver::solveSteady(ParGridFunction &state)
{
   *out << "AbstractSolver::solveSteady() is deprecated!!!!!!!!!!!!!!" << endl;
   *out << "calling AbstractSolver::solveUnsteady() instead" << endl;
   solveUnsteady(state);

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
   //    dynamic_cast<mfem::PetscLinearSolver
   //    *>(solver.get())->SetPreconditioner(*prec);

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
   //    MFEM_VERIFY(newton_solver->GetConverged(), "Newton solver did not
   //    converge."); u->SetFromTrueDofs(u_true);
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
   //    MFEM_VERIFY(newton_solver->GetConverged(), "Newton solver did not
   //    converge.");
   //    //u->SetFromTrueDofs(u_true);
   //    u->SetFromTrueDofs(*u_true);
   // #endif
   //    if (0==rank)
   //    {
   //       t2 = MPI_Wtime();
   //       *out << "Time for solving nonlinear system is " << (t2 - t1) <<
   //       endl;
   //    }
   // #else
   //    solver.reset(new UMFPackSolver());
   //    dynamic_cast<UMFPackSolver*>(solver.get())->Control[UMFPACK_ORDERING] =
   //    UMFPACK_ORDERING_METIS;
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
   //    MFEM_VERIFY(newton_solver->GetConverged(), "Newton solver did not
   //    converge."); u->SetFromTrueDofs(u_true);
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

   /// TODO: put this in options
   // bool paraview = !options["time-dis"]["steady"].get<bool>();
   bool paraview = true;
   std::unique_ptr<ParaViewDataCollection> pd;
   if (paraview)
   {
      pd = std::make_unique<ParaViewDataCollection>("time_hist", mesh.get());
      pd->SetPrefixPath("ParaView");
      pd->RegisterField("state", &state);
      pd->SetLevelsOfDetail(options["space-dis"]["degree"].get<int>() + 1);
      pd->SetDataFormat(VTKFormat::BINARY);
      pd->SetHighOrderOutput(true);
      pd->SetCycle(0);
      pd->SetTime(t);
      pd->Save();
   }

   std::cout.precision(16);
   std::cout << "res norm: " << calcResidualNorm(state) << "\n";

   auto &residual = res_fields.at("residual");
   calcResidual(state, residual);
   // printFields("init", {&residual, &state}, {"Residual", "Solution"});

   auto t_final = options["time-dis"]["t-final"].template get<double>();
   *out << "t_final is " << t_final << '\n';
   int ti = 0;
   double dt = 0.0;
   initialHook(state);
   for (ti = 0; ti < options["time-dis"]["max-iter"].get<int>(); ++ti)
   {
      dt = calcStepSize(ti, t, t_final, dt, state);
      *out << "iter " << ti << ": time = " << t << ": dt = " << dt;
      if (!options["time-dis"]["steady"].get<bool>())
      {
         *out << " (" << round(100 * t / t_final) << "% complete)";
      }
      *out << endl;
      iterationHook(ti, t, dt, state);
      auto &u_true = state.GetTrueVector();
      ode_solver->Step(u_true, t, dt);
      state.SetFromTrueDofs(u_true);

      if (paraview)
      {
         pd->SetCycle(ti);
         pd->SetTime(t);
         pd->Save();
      }
      // std::cout << "res norm: " << calcResidualNorm(state) << "\n";

      if (iterationExit(ti, t, t_final, dt, state))
      {
         break;
      }
   }
   {
      ofstream osol("final_before_TH.gf");
      osol.precision(std::numeric_limits<long double>::digits10 + 1);
      state.Save(osol);
   }
   terminalHook(ti, t, state);

   // Save the final solution. This output can be viewed later using GLVis:
   // glvis -m unitGridTestMesh.msh -g adv-final.gf".
   {
      ofstream osol("final.gf");
      osol.precision(std::numeric_limits<long double>::digits10 + 1);
      state.Save(osol);
   }
   // write the solution to vtk file
   if (options["space-dis"]["basis-type"].template get<string>() == "csbp")
   {
      ofstream sol_ofs("final_cg.vtk");
      sol_ofs.precision(14);
      mesh->PrintVTK(sol_ofs,
                     options["space-dis"]["degree"].template get<int>() + 1);
      state.SaveVTK(sol_ofs,
                    "Solution",
                    options["space-dis"]["degree"].template get<int>() + 1);
      sol_ofs.close();
      printField("final", state, "Solution");
   }
   else if (options["space-dis"]["basis-type"].template get<string>() == "dsbp")
   {
      ofstream sol_ofs("final_dg.vtk");
      sol_ofs.precision(14);
      mesh->PrintVTK(sol_ofs,
                     options["space-dis"]["degree"].template get<int>() + 1);
      state.SaveVTK(sol_ofs,
                    "Solution",
                    options["space-dis"]["degree"].template get<int>() + 1);
      sol_ofs.close();
      printField("final", state, "Solution");
   }
   // TODO: These mfem functions do not appear to be parallelized
}

void AbstractSolver::solveSteadyAdjoint(const std::string &fun)
{
   // double time_beg = NAN;
   // double time_end = NAN;
   // time_beg = MPI_Wtime();

   // // Step 0: allocate the adjoint variable
   // adj.reset(new GridFunType(fes.get()));
   // *adj = 0.0;

   // // Step 1: get the right-hand side vector, dJdu, and make an appropriate
   // // alias to it, the state, and the adjoint
   // std::unique_ptr<GridFunType> dJdu(new GridFunType(fes.get()));

   // HypreParVector *u_true = u->GetTrueDofs();
   // HypreParVector *dJdu_true = dJdu->GetTrueDofs();
   // HypreParVector *adj_true = adj->GetTrueDofs();
   // output.at(fun).Mult(*u_true, *dJdu_true);

   // // Step 2: get the Jacobian and transpose it
   // Operator *jac = &res->GetGradient(*u_true);
   // const Operator *jac_trans =
   //     dynamic_cast<const HypreParMatrix *>(jac)->Transpose();
   // MFEM_VERIFY(jac_trans, "Jacobian must be a HypreParMatrix!");

   // // Step 3: Solve the adjoint problem
   // *out << "Solving adjoint problem" << endl;
   // unique_ptr<Solver> adj_prec =
   // constructPreconditioner(options["adj-prec"]); unique_ptr<Solver>
   // adj_solver =
   //     constructLinearSolver(options["adj-solver"], *adj_prec);
   // adj_solver->SetOperator(*jac_trans);
   // adj_solver->Mult(*dJdu_true, *adj_true);

   // // check that adjoint residual is small
   // std::unique_ptr<GridFunType> adj_res(new GridFunType(fes.get()));
   // double res_norm = 0;
   // HypreParVector *adj_res_true = adj_res->GetTrueDofs();
   // jac_trans->Mult(*adj_true, *adj_res_true);
   // *adj_res_true -= *dJdu_true;
   // double loc_norm = (*adj_res_true) * (*adj_res_true);
   // MPI_Allreduce(&loc_norm, &res_norm, 1, MPI_DOUBLE, MPI_SUM, comm);
   // res_norm = sqrt(res_norm);
   // *out << "Adjoint residual norm = " << res_norm << endl;
   // adj->SetFromTrueDofs(*adj_true);

   // time_end = MPI_Wtime();
   // *out << "Time for solving adjoint is " << (time_end - time_beg) << endl;
}

unique_ptr<Solver> AbstractSolver::constructLinearSolver(
    nlohmann::json &_options,
    mfem::Solver &_prec)
{
   std::string solver_type = _options["type"].get<std::string>();
   auto reltol = _options["reltol"].get<double>();
   int maxiter = _options["maxiter"].get<int>();
   int ptl = _options["printlevel"].get<int>();
   int kdim = _options.value("kdim", -1);

   unique_ptr<Solver> lin_solver;
   if (solver_type == "hypregmres")
   {
      lin_solver = std::make_unique<HypreGMRES>(comm);
      auto *gmres = dynamic_cast<HypreGMRES *>(lin_solver.get());
      gmres->SetTol(reltol);
      gmres->SetMaxIter(maxiter);
      gmres->SetPrintLevel(ptl);
      gmres->SetPreconditioner(dynamic_cast<HypreSolver &>(_prec));
      if (kdim != -1)
      {
         gmres->SetKDim(kdim);  // set GMRES subspace size
      }
   }
   else if (solver_type == "gmres")
   {
      lin_solver = std::make_unique<GMRESSolver>(comm);
      auto *gmres = dynamic_cast<GMRESSolver *>(lin_solver.get());
      gmres->SetRelTol(reltol);
      gmres->SetMaxIter(maxiter);
      gmres->SetPrintLevel(ptl);
      gmres->SetPreconditioner(dynamic_cast<Solver &>(_prec));
      if (kdim != -1)
      {
         gmres->SetKDim(kdim);  // set GMRES subspace size
      }
   }
   else if (solver_type == "hyprefgmres")
   {
      lin_solver = std::make_unique<HypreFGMRES>(comm);
      auto *fgmres = dynamic_cast<HypreFGMRES *>(lin_solver.get());
      fgmres->SetTol(reltol);
      fgmres->SetMaxIter(maxiter);
      fgmres->SetPrintLevel(ptl);
      fgmres->SetPreconditioner(dynamic_cast<HypreSolver &>(_prec));
      if (kdim != -1)
      {
         fgmres->SetKDim(kdim);  // set FGMRES subspace size
      }
   }
   else if (solver_type == "hyprepcg")
   {
      lin_solver = std::make_unique<HyprePCG>(comm);
      auto *pcg = dynamic_cast<HyprePCG *>(lin_solver.get());
      pcg->SetTol(reltol);
      pcg->SetMaxIter(maxiter);
      pcg->SetPrintLevel(ptl);
      pcg->SetPreconditioner(dynamic_cast<HypreSolver &>(_prec));
   }
   else if (solver_type == "pcg")
   {
      lin_solver = std::make_unique<CGSolver>(comm);
      auto *pcg = dynamic_cast<CGSolver *>(lin_solver.get());
      pcg->SetRelTol(reltol);
      pcg->SetMaxIter(maxiter);
      pcg->SetPrintLevel(ptl);
      pcg->SetPreconditioner(dynamic_cast<Solver &>(_prec));
   }
   else if (solver_type == "minres")
   {
      lin_solver = std::make_unique<MINRESSolver>(comm);
      auto *minres = dynamic_cast<MINRESSolver *>(lin_solver.get());
      minres->SetRelTol(reltol);
      minres->SetMaxIter(maxiter);
      minres->SetPrintLevel(ptl);
      minres->SetPreconditioner(dynamic_cast<Solver &>(_prec));
   }
   else
   {
      throw MISOException(
          "Unsupported iterative solver type!\n"
          "\tavilable options are: hypregmres, gmres, hyprefgmres,\n"
          "\thyprepcg, pcg, minres");
   }
   return lin_solver;
}

unique_ptr<Solver> AbstractSolver::constructPreconditioner(
    nlohmann::json &_options)
{
   std::string prec_type = _options["type"].get<std::string>();
   unique_ptr<Solver> precond;
   if (prec_type == "hypreeuclid")
   {
      precond = std::make_unique<HypreEuclid>(comm);
      // TODO: need to add HYPRE_EuclidSetLevel to odl branch of mfem
      *out << "WARNING! Euclid fill level is hard-coded"
           << "(see AbstractSolver::constructLinearSolver() for details)"
           << endl;
      // int fill = options["lin-solver"]["filllevel"].get<int>();
      // HYPRE_EuclidSetLevel(dynamic_cast<HypreEuclid*>(precond.get())->GetPrec(),
      // fill);
   }
   else if (prec_type == "hypreilu")
   {
      precond = std::make_unique<HypreILU>();
      auto *ilu = dynamic_cast<HypreILU *>(precond.get());
      HYPRE_ILUSetType(*ilu, _options["ilu-type"].get<int>());
      HYPRE_ILUSetLevelOfFill(*ilu, _options["lev-fill"].get<int>());
      HYPRE_ILUSetLocalReordering(*ilu, _options["ilu-reorder"].get<int>());
      HYPRE_ILUSetPrintLevel(*ilu, _options["printlevel"].get<int>());
      // cout << "Just after Hypre options" << endl;
      // Just listing the options below in case we need them in the future
      // HYPRE_ILUSetSchurMaxIter(ilu, schur_max_iter);
      // HYPRE_ILUSetNSHDropThreshold(ilu, nsh_thres); needs type = 20,21
      // HYPRE_ILUSetDropThreshold(ilu, drop_thres);
      // HYPRE_ILUSetMaxNnzPerRow(ilu, nz_max);
   }
   else if (prec_type == "hypreams")
   {
      precond = std::make_unique<HypreAMS>(fes.get());
      auto *ams = dynamic_cast<HypreAMS *>(precond.get());
      ams->SetPrintLevel(_options["printlevel"].get<int>());
      ams->SetSingularProblem();
   }
   else if (prec_type == "hypreboomeramg")
   {
      precond = std::make_unique<HypreBoomerAMG>();
      auto *amg = dynamic_cast<HypreBoomerAMG *>(precond.get());
      amg->SetPrintLevel(_options["printlevel"].get<int>());
   }
   else if (prec_type == "blockilu")
   {
      precond = std::make_unique<BlockILU>(getNumState());
   }
   else
   {
      throw MISOException(
          "Unsupported preconditioner type!\n"
          "\tavilable options are: HypreEuclid, HypreILU, HypreAMS,"
          " HypreBoomerAMG.\n");
   }
   return precond;
}

unique_ptr<NewtonSolver> AbstractSolver::constructNonlinearSolver(
    nlohmann::json &_options,
    mfem::Solver &_lin_solver)
{
   std::string solver_type = _options["type"].get<std::string>();
   auto abstol = _options["abstol"].get<double>();
   auto reltol = _options["reltol"].get<double>();
   int maxiter = _options["maxiter"].get<int>();
   int ptl = _options["printlevel"].get<int>();
   unique_ptr<NewtonSolver> nonlin_solver;
   if (solver_type == "newton")
   {
      nonlin_solver = std::make_unique<NewtonSolver>(comm);
   }
   else if (solver_type == "inexactnewton")
   {
      nonlin_solver = std::make_unique<NewtonSolver>(comm);
      auto *newton = dynamic_cast<NewtonSolver *>(nonlin_solver.get());

      /// use defaults from SetAdaptiveLinRtol unless specified
      int type = _options.value("inexacttype", 2);
      double rtol0 = _options.value("rtol0", 0.5);
      double rtol_max = _options.value("rtol_max", 0.9);
      double alpha = _options.value("alpha", (0.5) * ((1.0) + sqrt((5.0))));
      double gamma = _options.value("gamma", 1.0);
      newton->SetAdaptiveLinRtol(type, rtol0, rtol_max, alpha, gamma);
   }
   else
   {
      throw MISOException(
          "Unsupported nonlinear solver type!\n"
          "\tavilable options are: newton, inexactnewton\n");
   }

   nonlin_solver->iterative_mode = true;
   nonlin_solver->SetSolver(dynamic_cast<Solver &>(_lin_solver));
   nonlin_solver->SetPrintLevel(ptl);
   nonlin_solver->SetRelTol(reltol);
   nonlin_solver->SetAbsTol(abstol);
   nonlin_solver->SetMaxIter(maxiter);

   return nonlin_solver;
}

void AbstractSolver::constructEvolver()
{
   bool newton_abort = options["nonlin-solver"]["abort"].get<bool>();
   evolver =
       std::make_unique<MISOEvolver>(ess_bdr,
                                     nonlinear_mass.get(),
                                     mass.get(),
                                     res.get(),
                                     stiff.get(),
                                     load.get(),
                                     ent.get(),
                                     *out,
                                     0.0,
                                     TimeDependentOperator::Type::IMPLICIT,
                                     newton_abort);
   evolver->SetNewtonSolver(newton_solver.get());
}

void AbstractSolver::solveUnsteadyAdjoint(const std::string &fun)
{
   throw MISOException(
       "AbstractSolver::solveUnsteadyAdjoint(fun)\n"
       "\tnot implemented yet!");
}

void AbstractSolver::createOutput(const std::string &fun)
{
   nlohmann::json options;
   createOutput(fun, options);
}

void AbstractSolver::createOutput(const std::string &fun,
                                  const nlohmann::json &options)
{
   if (outputs.count(fun) == 0)
   {
      addOutput(fun, options);
   }
   else
   {
      throw MISOException("Output with name " + fun + " already created!\n");
   }
}

void AbstractSolver::setOutputOptions(const std::string &fun,
                                      const nlohmann::json &options)
{
   try
   {
      auto output = outputs.find(fun);
      if (output == outputs.end())
      {
         throw MISOException("Did not find " + fun + " in output map?");
      }
      miso::setOptions(output->second, options);
   }
   catch (const std::out_of_range &exception)
   {
      std::cerr << exception.what() << endl;
   }
}

double AbstractSolver::calcOutput(const ParGridFunction &state,
                                  const std::string &fun)
{
   HypreParVector state_true(fes.get());
   state.GetTrueDofs(state_true);

   MISOInputs inputs{{"state", state_true}};
   return calcOutput(fun, inputs);
}

double AbstractSolver::calcOutput(const std::string &fun,
                                  const MISOInputs &inputs)
{
   try
   {
      auto output = outputs.find(fun);
      if (output == outputs.end())
      {
         throw MISOException("Did not find " + fun + " in output map?");
      }
      miso::setInputs(output->second, inputs);
      return miso::calcOutput(output->second, inputs);
   }
   catch (const std::out_of_range &exception)
   {
      std::cerr << exception.what() << endl;
      return std::nan("");
   }
}

void AbstractSolver::calcOutputPartial(const std::string &of,
                                       const std::string &wrt,
                                       const MISOInputs &inputs,
                                       double &partial)
{
   try
   {
      auto output = outputs.find(of);
      if (output == outputs.end())
      {
         throw MISOException("Did not find " + of + " in output map?");
      }
      double part = miso::calcOutputPartial(output->second, wrt, inputs);
      partial += part;
   }
   catch (const std::out_of_range &exception)
   {
      std::cerr << exception.what() << endl;
      partial = std::nan("");
   }
}

void AbstractSolver::calcOutputPartial(const std::string &of,
                                       const std::string &wrt,
                                       const MISOInputs &inputs,
                                       double *partial_buffer)
{
   /// get FESpace for field we're taking partial with respect to
   auto *wrt_fes = res_fields.at(wrt).ParFESpace();
   HypreParVector partial(wrt_fes->GetComm(),
                          wrt_fes->GlobalTrueVSize(),
                          partial_buffer,
                          wrt_fes->GetTrueDofOffsets());
   calcOutputPartial(of, wrt, inputs, partial);
}

void AbstractSolver::calcOutputPartial(const std::string &of,
                                       const std::string &wrt,
                                       const MISOInputs &inputs,
                                       HypreParVector &partial)
{
   try
   {
      auto output = outputs.find(of);
      if (output == outputs.end())
      {
         throw MISOException("Did not find " + of + " in output map?");
      }
      miso::calcOutputPartial(output->second, wrt, inputs, partial);
   }
   catch (const std::out_of_range &exception)
   {
      std::cerr << exception.what() << endl;
   }
}

void AbstractSolver::checkJacobian(
    const ParGridFunction &state,
    std::function<double(const Vector &)> pert_fun)
{
   // initialize some variables
   const double delta = 1e-5;

   GridFunType u_plus(state);
   GridFunType u_minus(state);
   GridFunType pert_vec(fes.get());
   FunctionCoefficient up(std::move(pert_fun));
   pert_vec.ProjectCoefficient(up);

   Array<int> ess_tdof_list;
   fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   pert_vec.SetSubVector(ess_tdof_list, 0.0);

   // perturb in the positive and negative pert_vec directions
   u_plus.Add(delta, pert_vec);
   u_minus.Add(-delta, pert_vec);

   // Get the product using a 2nd-order finite-difference approximation
   GridFunType res_plus(fes.get());
   GridFunType res_minus(fes.get());

   calcResidual(u_plus, res_plus);
   calcResidual(u_minus, res_minus);
   // res_plus = 1/(2*delta)*(res_plus - res_minus)
   subtract(1 / (2 * delta), res_plus, res_minus, res_plus);

   // Get the product directly using Jacobian from GetGradient
   GridFunType jac_v(fes.get());
   HypreParVector *u_true = state.GetTrueDofs();
   HypreParVector *pert = pert_vec.GetTrueDofs();
   HypreParVector *prod = jac_v.GetTrueDofs();
   mfem::Operator &jac = res->GetGradient(*u_true);
   jac.Mult(*pert, *prod);
   jac_v.SetFromTrueDofs(*prod);

   // check the difference norm
   jac_v -= res_plus;
   double error = calcInnerProduct(jac_v, jac_v);
   *out << "The Jacobian product error norm is " << sqrt(error) << endl;
   // for (int i = 0; i < jac_v.Size(); ++i)
   // {
   //    if (jac_v(i) > 1e-6)
   //       *out << "jac_v(" << i << "): " << jac_v(i) << "\n";
   // }

   // ParGridFunction real_res(fes.get());
   // ParGridFunction zero_res(fes.get());
   // ParGridFunction jac_temp(fes.get());
   // ParGridFunction zero_state(fes.get());
   // zero_state = 0.0;

   // calcResidual(state, real_res);
   // calcResidual(zero_state, zero_res);

   // Array<int> ess_tdof_list;
   // fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   // zero_res.SetSubVector(ess_tdof_list, 100.0);

   // auto *real_res_true = real_res.GetTrueDofs();
   // auto *zero_res_true = zero_res.GetTrueDofs();

   // auto *u_true = state.GetTrueDofs();
   // mfem::Operator &jac = res->GetGradient(*u_true);
   // auto *jac_temp_true = jac_temp.GetTrueDofs();
   // jac.Mult(*u_true, *jac_temp_true);

   // jac_temp.SetFromTrueDofs(*jac_temp_true);

   // printFields("real_res", {&real_res, &jac_temp, &zero_res}, {"res",
   // "jac_temp", "zero_res"});

   // real_res -= jac_temp;
   // real_res += zero_res;

   // // *real_res_true -= *jac_temp_true;
   // // *real_res_true += *zero_res_true;

   // printField("res_diff", real_res, "res");

   // double norm = calcInnerProduct(real_res, real_res);
   // // double norm = *real_res_true * *real_res_true;
   // *out << "The norm is " << sqrt(norm) << endl;
}

void AbstractSolver::checkJacobian(void (*pert_fun)(const mfem::Vector &,
                                                    mfem::Vector &))
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
   subtract(1 / (2 * delta), res_plus, res_minus, res_plus);

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

}  // namespace miso
