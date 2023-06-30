#include <cstddef>
#include "finite_element_dual.hpp"
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

// namespace
// {
// /// TODO: Get rid of this function and instead opt for FiniteElementVector
// /// constructor
// template <typename T>
// T createFiniteElementVector(mfem::ParMesh &mesh,
//                             const nlohmann::json &space_options,
//                             const int num_states,
//                             const std::string &name)
// {
//    const int dim = mesh.Dimension();
//    const auto order = space_options["degree"].get<int>();
//    const auto basis_type = space_options["basis-type"].get<std::string>();
//    const bool galerkin_diff = space_options.value("GD", false);
//    // Define the SBP elements and finite-element space; eventually, we will
//    want
//    // to have a case or if statement here for both CSBP and DSBP, and (?)
//    // standard FEM. and here it is for first two
//    std::unique_ptr<mfem::FiniteElementCollection> fec;
//    if (basis_type == "csbp")
//    {
//       fec = std::make_unique<mfem::SBPCollection>(order, dim);
//    }
//    else if (basis_type == "dsbp" || galerkin_diff)
//    {
//       fec = std::make_unique<mfem::DSBPCollection>(order, dim);
//    }
//    else if (basis_type == "nedelec" || basis_type == "nd" || basis_type ==
//    "ND")
//    {
//       fec = std::make_unique<mfem::ND_FECollection>(order, dim);
//    }
//    else if (basis_type == "H1")
//    {
//       fec = std::make_unique<mfem::H1_FECollection>(order, dim);
//    }

//    T vec(mesh,
//          {.order = order,
//           .num_states = num_states,
//           .coll = std::move(fec),
//           .ordering = mfem::Ordering::byVDIM,
//           .name = name});
//    return vec;
// }

// mach::FiniteElementState createState(mfem::ParMesh &mesh,
//                                      const nlohmann::json &space_options,
//                                      const int num_states,
//                                      const std::string &name)
// {
//    return createFiniteElementVector<mach::FiniteElementState>(
//        mesh, space_options, num_states, name);
// }

// mach::FiniteElementDual createDual(mfem::ParMesh &mesh,
//                                    const nlohmann::json &space_options,
//                                    const int num_states,
//                                    const std::string &name)
// {
//    return createFiniteElementVector<mach::FiniteElementDual>(
//        mesh, space_options, num_states, name);
// }

// }  // namespace

namespace mach
{
#ifdef MFEM_USE_PUMI
int MachMesh::pumi_mesh_count = 0;
bool MachMesh::PCU_previously_initialized = false;

MachMesh::MachMesh()
{
   /// If Something else has initialized PCU
   if (PCU_Comm_Initialized() && pumi_mesh_count == 0)
   {
      PCU_previously_initialized = true;
   }
   /// If nothing else has initialized PCU and we haven't yet either
   if (!PCU_previously_initialized && pumi_mesh_count == 0)
   {
      /// Initialize PCU
      PCU_Comm_Init();

      /// Prep GMI interfaces
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
      gmi_register_null();
   }
   ++pumi_mesh_count;
}

MachMesh::MachMesh(MachMesh &&other) noexcept
 : mesh(std::move(other.mesh)), pumi_mesh(std::move(other.pumi_mesh))
{
   ++pumi_mesh_count;
}

MachMesh &MachMesh::operator=(MachMesh &&other) noexcept
{
   if (this != &other)
   {
      mesh = std::move(other.mesh);
      pumi_mesh = std::move(other.pumi_mesh);
   }
   return *this;
}

MachMesh::~MachMesh()
{
   --pumi_mesh_count;
   /// If we started PCU and we're the last one using it, close it
   if (!PCU_previously_initialized && pumi_mesh_count == 0)
   {
#ifdef MFEM_USE_EGADS
      gmi_egads_stop();
#endif
#ifdef HAVE_SIMMETRIX
      gmi_sim_stop();
      Sim_unregisterAllKeys();
      SimModel_stop();
      MS_exit();
#endif
      PCU_Comm_Free();
   }
}
#endif

MachMesh constructMesh(MPI_Comm comm,
                       const nlohmann::json &mesh_options,
                       std::unique_ptr<mfem::Mesh> smesh,
                       bool keep_boundaries)
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

   MachMesh mesh;
   // if serial mesh passed in, use that
   if (smesh != nullptr)
   {
      mesh.mesh = std::make_unique<mfem::ParMesh>(comm, *smesh);
   }
   // native MFEM mesh
   else if (mesh_ext == "mesh")
   {
      // read in the serial mesh
      smesh = std::make_unique<mfem::Mesh>(mesh_file.c_str(), 1, 1);
      mesh.mesh = std::make_unique<mfem::ParMesh>(comm, *smesh);
   }
   // PUMI mesh
   else if (mesh_ext == "smb" || mesh_ext == "ugrid")
   {
      mesh = constructPumiMesh(comm, mesh_options);
   }
   else
   {
      throw MachException("Unrecognized mesh file extension!\n");
   }
   // auto *nodes = mesh.mesh->GetNodes();
   // if (nodes == nullptr)
   // {
   //    mesh.mesh->SetCurvature(1, false, 3, mfem::Ordering::byVDIM);
   // }
   mesh.mesh->EnsureNodes();

   // if (!keep_boundaries)
   // {
   //    mesh.mesh->RemoveInternalBoundaries();
   // }

   // std::ofstream file("pde_solver_mesh.mesh");
   // mesh.mesh->Print(file);
   return mesh;
}

MachMesh constructPumiMesh(MPI_Comm comm, const nlohmann::json &mesh_options)
{
#ifdef MFEM_USE_PUMI  // if using pumi mesh
   auto model_file = mesh_options["model-file"].get<std::string>();
   auto mesh_file = mesh_options["file"].get<std::string>();
   std::string mesh_ext;
   std::size_t i = mesh_file.rfind('.', mesh_file.length());
   if (i != std::string::npos)
   {
      mesh_ext = (mesh_file.substr(i + 1, mesh_file.length() - i));
   }
   /// Switch PUMI MPI Comm to the mesh's comm
   PCU_Switch_Comm(comm);

   MachMesh mesh;
   if (mesh_ext == "ugrid")
   {
      gmi_model *g = gmi_load(model_file.c_str());  // will this leak?
      mesh.pumi_mesh = std::unique_ptr<apf::Mesh2, pumiDeleter>(
          apf::loadMdsFromUgrid(g, mesh_file.c_str()));
   }
   else if (mesh_ext == "smb")
   {
      mesh.pumi_mesh = std::unique_ptr<apf::Mesh2, pumiDeleter>(
          apf::loadMdsMesh(model_file.c_str(), mesh_file.c_str()));
   }
   auto &pumi_mesh = mesh.pumi_mesh;

   /// TODO: change this to use options
   /// If it is higher order change shape
   // int order = options["space-dis"]["degree"].template get<int>();
   // if (order > 1)
   // {
   //     crv::BezierCurver bc(pumi_mesh, order, 2);
   //     bc.run();
   // }

   // pumi_mesh->verify();

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

   mesh.mesh = std::make_unique<mfem::ParPumiMesh>(comm, pumi_mesh.get());

   /// Add attributes based on reverse classification
   // Boundary faces
   int dim = mesh.mesh->Dimension();
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
         (mesh.mesh->GetBdrElement(ent_cnt))->SetAttribute(tag);
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
      mesh.mesh->SetAttribute(ent_cnt, tag);
      ent_cnt++;
   }
   pumi_mesh->end(itr);

   // Apply the attributes
   mesh.mesh->SetAttributes();
   return mesh;
#else
   throw MachException(
       "AbstractSolver::constructPumiMesh()\n"
       "\tMFEM was not built with PUMI!\n"
       "\trecompile MFEM with PUMI\n");
#endif  // MFEM_USE_PUMI
}

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

void PDESolver::getMeshCoordinates(mfem::Vector &mesh_coords) const
{
   fields.at("mesh_coords").setTrueVec(mesh_coords);
}

PDESolver::PDESolver(MPI_Comm incomm,
                     const nlohmann::json &solver_options,
                     const int num_states,
                     std::unique_ptr<mfem::Mesh> smesh)
 : AbstractSolver2(incomm, solver_options),
   mesh_(constructMesh(comm, options["mesh"], std::move(smesh))),
   materials(material_library)
{
   /// loop over all components specified in options and add their specified
   /// materials to the solver's known material library
   if (solver_options.contains("components"))
   {
      for (const auto &component : solver_options["components"])
      {
         const auto &material = component["material"];
         if (material.is_string())
         {
            continue;
         }
         else
         {
            const auto &material_name = material["name"].get<std::string>();
            materials[material_name].merge_patch(material);
         }
      }
   }

   fields.emplace(
       "state",
       FiniteElementState(mesh(), options["space-dis"], num_states, "state"));
   fields.emplace(
       "dirichlet_bc",
       FiniteElementState(
           mesh(), options["space-dis"], num_states, "dirichlet_bc"));
   fields.emplace(
       "adjoint",
       FiniteElementState(mesh(), options["space-dis"], num_states, "adjoint"));
   duals.emplace(
       "residual",
       FiniteElementDual(mesh(), options["space-dis"], num_states, "residual"));

   // // The pm demag constraint field
   // fields.emplace(
   //     "pm_demag_field",
   //     FiniteElementState(mesh(), options["space-dis"], num_states,
   //     "pm_demag_field"));

   setUpExternalFields();
}

PDESolver::PDESolver(
    MPI_Comm incomm,
    const nlohmann::json &solver_options,
    const std::function<int(const nlohmann::json &, int)> &num_states,
    std::unique_ptr<mfem::Mesh> smesh)
 : AbstractSolver2(incomm, solver_options),
   mesh_(constructMesh(comm, options["mesh"], std::move(smesh))),
   materials(material_library)
{
   int ns = num_states(solver_options, mesh().SpaceDimension());
   fields.emplace(
       "state", FiniteElementState(mesh(), options["space-dis"], ns, "state"));
   fields.emplace(
       "dirichlet_bc",
       FiniteElementState(mesh(), options["space-dis"], ns, "dirichlet_bc"));
   fields.emplace(
       "adjoint",
       FiniteElementState(mesh(), options["space-dis"], ns, "adjoint"));
   duals.emplace(
       "residual",
       FiniteElementDual(mesh(), options["space-dis"], ns, "residual"));

   // // The pm demag constraint field
   // fields.emplace(
   //     "pm_demag_field",
   //     FiniteElementState(mesh(), options["space-dis"], ns,
   //     "pm_demag_field"));

   setUpExternalFields();
}

void PDESolver::setUpExternalFields()
{
   // give the solver ownership over the mesh coords grid function, and store
   // it in `fields` with name "mesh_coords"
   {
      auto &mesh_gf = *dynamic_cast<mfem::ParGridFunction *>(mesh().GetNodes());
      auto *mesh_fespace = mesh_gf.ParFESpace();

      /// create new state vector copying the mesh's fe space
      fields.emplace(
          std::piecewise_construct,
          std::forward_as_tuple("mesh_coords"),
          std::forward_as_tuple(mesh(), *mesh_fespace, "mesh_coords"));
      FiniteElementState &mesh_coords = fields.at("mesh_coords");
      /// set the values of the new GF to those of the mesh's old nodes
      mesh_coords.gridFunc() = mesh_gf;
      // mesh_coords.setTrueVec();  // distribute coords
      /// tell the mesh to use this GF for its Nodes
      /// (and that it doesn't own it)
      mesh().NewNodes(mesh_coords.gridFunc(), false);
   }

   if (options.contains("external-fields"))
   {
      auto &external_fields = options["external-fields"];
      for (auto &[name, field] : external_fields.items())
      {
         // std::string name(f.key());
         // auto field = f.value();

         /// this approach will only work for fields on the same mesh
         auto num_states = field["num-states"].get<int>();
         fields.emplace(name,
                        FiniteElementState(mesh(), field, num_states, name));
      }
   }
}

void PDESolver::setState_(std::any function,
                          const std::string &name,
                          mfem::Vector &state)
{
   AbstractSolver2::setState_(function, name, state);

   useAny(
       function,
       [&](std::function<double(const mfem::Vector &)> &fun)
       { fields.at(name).project(fun, state); },
       [&](mfem::Coefficient *coeff)
       { fields.at(name).project(*coeff, state); },
       [&](std::function<void(const mfem::Vector &, mfem::Vector &)> &vec_fun)
       { fields.at(name).project(vec_fun, state); },
       [&](mfem::VectorCoefficient *vec_coeff)
       { fields.at(name).project(*vec_coeff, state); });

   if (name == "state")
   {
      PDESolver::setState_(function, "dirichlet_bc", state);
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

   return useAny(
       ex_sol,
       [&](std::function<double(const mfem::Vector &)> &fun)
       {
          auto &field = fields.at(name);
          field.distributeSharedDofs(state);
          return calcLpError(field, fun, 2);
       },
       [&](mfem::Coefficient *coeff)
       {
          auto &field = fields.at(name);
          field.distributeSharedDofs(state);
          return calcLpError(field, *coeff, 2);
       },
       [&](std::function<void(const mfem::Vector &, mfem::Vector &)> &fun)
       {
          auto &field = fields.at(name);
          field.distributeSharedDofs(state);
          return calcLpError(field, fun, 2);
       },
       [&](mfem::VectorCoefficient *coeff)
       {
          auto &field = fields.at(name);
          field.distributeSharedDofs(state);
          return calcLpError(field, *coeff, 2);
       });
}

void PDESolver::initialHook(const mfem::Vector &state)
{
   AbstractSolver2::initialHook(state);
   int inverted_elems = mesh().CheckElementOrientation(false);
   if (inverted_elems > 0)
   {
      mesh().PrintVTU("inverted_mesh", mfem::VTKFormat::BINARY, true);
      throw MachException("Mesh contains inverted elements!\n");
   }
   else
   {
      std::cout << "No inverted elements!\n";
   }

   getState().distributeSharedDofs(state);
   derivedPDEInitialHook(state);
}

void PDESolver::iterationHook(int iter,
                              double t,
                              double dt,
                              const mfem::Vector &state)
{
   AbstractSolver2::iterationHook(iter, t, dt, state);
   derivedPDEIterationHook(iter, t, dt, state);
}

void PDESolver::terminalHook(int iter,
                             double t_final,
                             const mfem::Vector &state)
{
   AbstractSolver2::terminalHook(iter, t_final, state);
   derivedPDETerminalHook(iter, t_final, state);
}

}  // namespace mach
