#include "mfem.hpp"

#ifdef MFEM_USE_PUMI
#include "apf.h"
#include "apfMDS.h"
#include "apfMesh.h"
#include "apfNumbering.h"
#include "PCU.h"
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

#include "finite_element_state.hpp"
#include "material_library.hpp"
#include "sbp_fe.hpp"
#include "utils.hpp"

#include "pde_solver.hpp"

namespace
{
template <typename T>
T createFiniteElementVector(mfem::ParMesh &mesh,
                            const nlohmann::json &space_options,
                            const int num_states,
                            const std::string &name)
{
   const int dim = mesh.Dimension();
   const auto order = space_options["degree"].get<int>();
   const auto basis_type = space_options["basis-type"].get<std::string>();
   const bool galerkin_diff = space_options.value("GD", false);
   // Define the SBP elements and finite-element space; eventually, we will want
   // to have a case or if statement here for both CSBP and DSBP, and (?)
   // standard FEM. and here it is for first two
   std::unique_ptr<mfem::FiniteElementCollection> fec;
   if (basis_type == "csbp")
   {
      fec = std::make_unique<mfem::SBPCollection>(order, dim);
   }
   else if (basis_type == "dsbp" || galerkin_diff)
   {
      fec = std::make_unique<mfem::DSBPCollection>(order, dim);
   }
   else if (basis_type == "nedelec" || basis_type == "nd")
   {
      fec = std::make_unique<mfem::ND_FECollection>(order, dim);
   }
   else if (basis_type == "H1")
   {
      fec = std::make_unique<mfem::H1_FECollection>(order, dim);
   }

   T vec(mesh,
         {.order = order,
          .num_states = num_states,
          .coll = std::move(fec),
          .ordering = mfem::Ordering::byVDIM,
          .name = name});
   return vec;
}

mach::FiniteElementState createState(mfem::ParMesh &mesh,
                                     const nlohmann::json &space_options,
                                     const int num_states,
                                     const std::string &name)
{
   return createFiniteElementVector<mach::FiniteElementState>(
       mesh, space_options, num_states, name);
}

mach::FiniteElementDual createDual(mfem::ParMesh &mesh,
                                   const nlohmann::json &space_options,
                                   const int num_states,
                                   const std::string &name)
{
   return createFiniteElementVector<mach::FiniteElementDual>(
       mesh, space_options, num_states, name);
}

}  // namespace

namespace mach
{

int PDESolver::getFieldSize(const std::string &name) const
{
   auto size = AbstractSolver2::getFieldSize(name);
   if (size != 0)
   {
      return size;
   }
   auto field = fields.find(name);
   if (field != fields.end())
   {
      return field->second.space().GetTrueVSize();
   }
   return 0;
}

PDESolver::PDESolver(MPI_Comm incomm,
                     const nlohmann::json &solver_options,
                     const int num_states,
                     std::unique_ptr<mfem::Mesh> smesh)
 : AbstractSolver2(incomm, solver_options),
   mesh_(constructMesh(comm, options["mesh"], std::move(smesh))),
   materials(material_library)
{
   fields.emplace(
       "state", createState(*mesh_, options["space-dis"], num_states, "state"));
   fields.emplace(
       "adjoint",
       createState(*mesh_, options["space-dis"], num_states, "adjoint"));
   duals.emplace(
       "residual",
       createDual(*mesh_, options["space-dis"], num_states, "residual"));

   setUpExternalFields();
}

std::unique_ptr<mfem::ParMesh> PDESolver::constructMesh(
    MPI_Comm comm,
    const nlohmann::json &mesh_options,
    std::unique_ptr<mfem::Mesh> smesh)
{
   auto mesh_file = mesh_options["file"].get<std::string>();
   std::string mesh_ext;
   std::size_t i = mesh_file.rfind('.', mesh_file.length());
   if (i != std::string::npos)
   {
      mesh_ext = (mesh_file.substr(i + 1, mesh_file.length() - i));
   }
   else
   {
      throw MachException(
          "AbstractSolver::constructMesh(smesh)\n"
          "\tMesh file has no extension!\n");
   }

   std::unique_ptr<mfem::ParMesh> mesh;
   // if serial mesh passed in, use that
   if (smesh != nullptr)
   {
      mesh = std::make_unique<mfem::ParMesh>(comm, *smesh);
   }
   // native MFEM mesh
   else if (mesh_ext == "mesh")
   {
      // read in the serial mesh
      smesh = std::make_unique<mfem::Mesh>(mesh_file.c_str(), 1, 1);
      mesh = std::make_unique<mfem::ParMesh>(comm, *smesh);
   }
   // // PUMI mesh
   // else if (mesh_ext == "smb")
   // {
   //    mesh = constructPumiMesh(comm, mesh_options);
   // }
   mesh->EnsureNodes();

   mesh->RemoveInternalBoundaries();
   // std::cout << "bdr_attr: ";
   // mesh->bdr_attributes.Print(std::cout);
   return mesh;
}

/*
std::unique_ptr<mfem::ParMesh> PDESolver::constructPumiMesh(
    MPI_Comm comm,
    const nlohmann::json &mesh_options)
{
#ifdef MFEM_USE_PUMI  // if using pumi mesh
   auto model_file = mesh_options["model-file"].get<std::string>();
   auto mesh_file = mesh_options["file"].get<std::string>();
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

   /// TODO: change this to use options
   /// If it is higher order change shape
   // int order = options["space-dis"]["degree"].template get<int>();
   // if (order > 1)
   // {
   //     crv::BezierCurver bc(pumi_mesh, order, 2);
   //     bc.run();
   // }

   pumi_mesh->verify();

   auto *aux_num = apf::createNumbering(
       pumi_mesh.get(), "aux_numbering", pumi_mesh->getShape(), 1);

   auto *itr = pumi_mesh->begin(0);
   apf::MeshEntity *v = nullptr;
   int count = 0;
   while ((v = pumi_mesh->iterate(itr)) != nullptr)
   {
      apf::number(aux_num, v, 0, 0, count++);
   }
   pumi_mesh->end(itr);

   auto mesh = std::make_unique<mfem::ParPumiMesh>(comm, pumi_mesh.get());

   /// Add attributes based on reverse classification
   // Boundary faces
   int dim = mesh->Dimension();
   itr = pumi_mesh->begin(dim - 1);
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
   return mesh;
#else
   throw MachException(
       "AbstractSolver::constructPumiMesh()\n"
       "\tMFEM was not built with PUMI!\n"
       "\trecompile MFEM with PUMI\n");
#endif  // MFEM_USE_PUMI
}
*/

void PDESolver::setUpExternalFields()
{
   // give the solver ownership over the mesh coords grid function, and store
   // it in `fields` with name "mesh_coords"
   {
      auto &mesh_gf = *dynamic_cast<mfem::ParGridFunction *>(mesh_->GetNodes());
      auto *mesh_fespace = mesh_gf.ParFESpace();

      /// create new state vector copying the mesh's fe space
      fields.emplace(
          std::piecewise_construct,
          std::forward_as_tuple("mesh_coords"),
          std::forward_as_tuple(*mesh_, *mesh_fespace, "mesh_coords"));
      FiniteElementState &mesh_coords = fields.at("mesh_coords");
      /// set the values of the new GF to those of the mesh's old nodes
      mesh_coords.gridFunc() = mesh_gf;
      // mesh_coords.setTrueVec();  // distribute coords
      /// tell the mesh to use this GF for its Nodes
      /// (and that it doesn't own it)
      mesh_->NewNodes(mesh_coords.gridFunc(), false);
   }

   if (options.contains("external-fields"))
   {
      auto &external_fields = options["external-fields"];
      for (auto &f : external_fields.items())
      {
         std::string name(f.key());
         auto field = f.value();

         /// this approach will only work for fields on the same mesh
         auto num_states = field["num-states"].get<int>();
         fields.emplace(name, createState(*mesh_, field, num_states, name));
      }
   }
}

void PDESolver::setState_(std::any function,
                          const std::string &name,
                          mfem::Vector &state)
{
   AbstractSolver2::setState_(function, name, state);

   auto *coeff_func =
       std::any_cast<std::function<double(const mfem::Vector &)>>(&function);
   if (coeff_func != nullptr)
   {
      fields.at(name).project(*coeff_func, state);
      return;
   }
   auto *coeff = std::any_cast<mfem::Coefficient *>(&function);
   if (coeff != nullptr)
   {
      fields.at(name).project(**coeff, state);
      return;
   }
   auto *vec_coeff_func =
       std::any_cast<std::function<void(const mfem::Vector &, mfem::Vector &)>>(
           &function);
   if (vec_coeff_func != nullptr)
   {
      fields.at(name).project(*vec_coeff_func, state);
      return;
   }
   auto *vec_coeff = std::any_cast<mfem::VectorCoefficient *>(&function);
   if (vec_coeff != nullptr)
   {
      fields.at(name).project(**vec_coeff, state);
      return;
   }
}

double PDESolver::calcStateError_(std::any ex_sol,
                                  const std::string &name,
                                  const mfem::Vector &state)
{
   auto err = AbstractSolver2::calcStateError_(ex_sol, name, state);
   if (!std::isnan(err))
   {
      return err;
   }

   auto *coeff_func =
       std::any_cast<std::function<double(const mfem::Vector &)>>(&ex_sol);
   if (coeff_func != nullptr)
   {
      auto &field = fields.at(name);
      field.distributeSharedDofs(state);
      return calcLpError(field, *coeff_func, 2);
   }
   auto *coeff = std::any_cast<mfem::Coefficient *>(&ex_sol);
   if (coeff != nullptr)
   {
      auto &field = fields.at(name);
      field.distributeSharedDofs(state);
      return calcLpError(field, **coeff, 2);
   }
   auto *vec_coeff_func =
       std::any_cast<std::function<void(const mfem::Vector &, mfem::Vector &)>>(
           &ex_sol);
   if (vec_coeff_func != nullptr)
   {
      auto &field = fields.at(name);
      field.distributeSharedDofs(state);
      return calcLpError(field, *vec_coeff_func, 2);
   }
   auto *vec_coeff = std::any_cast<mfem::VectorCoefficient *>(&ex_sol);
   if (vec_coeff != nullptr)
   {
      auto &field = fields.at(name);
      field.distributeSharedDofs(state);
      return calcLpError(field, **vec_coeff, 2);
   }
   return NAN;
}

}  // namespace mach
