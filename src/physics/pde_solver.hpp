#ifndef MACH_PDE_SOLVER
#define MACH_PDE_SOLVER

#include <memory>
#include <vector>
#include <functional>

#include "mfem.hpp"
#include "nlohmann/json.hpp"
#ifdef MFEM_USE_PUMI
#include "apf.h"
#include "apfMesh2.h"

namespace mach
{
struct pumiDeleter
{
   void operator()(apf::Mesh2 *mesh) const
   {
      if (mesh != nullptr)
      {
         mesh->destroyNative();
         apf::destroyMesh(mesh);
      }
   }
};

}  // namespace mach
#endif

#include "abstract_solver.hpp"
#include "finite_element_state.hpp"
#include "finite_element_dual.hpp"

namespace mach
{
struct MachMesh
{
   std::unique_ptr<mfem::ParMesh> mesh = nullptr;
#ifdef MFEM_USE_PUMI
   std::unique_ptr<apf::Mesh2, pumiDeleter> pumi_mesh = nullptr;
   static int pumi_mesh_count;
   static bool PCU_previously_initialized;

   MachMesh();

   MachMesh(const MachMesh &) = delete;
   MachMesh &operator=(const MachMesh &) = delete;

   MachMesh(MachMesh &&) noexcept;
   MachMesh &operator=(MachMesh &&) noexcept;

   ~MachMesh();
#endif
};

MachMesh constructMesh(MPI_Comm comm,
                       const nlohmann::json &mesh_options,
                       std::unique_ptr<mfem::Mesh> smesh = nullptr,
                       bool keep_boundaries = false);

MachMesh constructPumiMesh(MPI_Comm comm, const nlohmann::json &mesh_options);

class PDESolver : public AbstractSolver2
{
public:
   int getNumStates() const { return fes().GetVDim(); }
   int getFieldSize(const std::string &name) const override;

   FiniteElementState &getState() { return state(); }
   const FiniteElementState &getState() const { return state(); }

   FiniteElementDual &getResVec() { return res_vec(); }
   const FiniteElementDual &getResVec() const { return res_vec(); }

   /// Construct a `PDESolver`
   /// \param[in] incomm - MPI communicator to associate with the solver
   /// \param[in] solver_options - options used to define the solver
   /// \param[in] num_states - number of states at each degree of freedom
   /// \param[in] smesh - serial mesh for the solver (optional)
   PDESolver(MPI_Comm incomm,
             const nlohmann::json &solver_options,
             const int num_states,
             std::unique_ptr<mfem::Mesh> smesh = nullptr);

   /// Construct a `PDESolver` whose num of states is determined by a function
   /// \param[in] incomm - MPI communicator to associate with the solver
   /// \param[in] solver_options - options used to define the solver
   /// \param[in] num_states - a function that returns the number of states
   /// \param[in] smesh - serial mesh for the solver (optional)
   /// \note This version is needed when the number of states depends on the
   /// mesh, but the mesh is not available until after it has been loaded.  In
   /// this case, the `num_states` function can be used after the mesh is
   /// available to determine the number of states.
   PDESolver(MPI_Comm incomm,
             const nlohmann::json &solver_options,
             std::function<int(const nlohmann::json &, int)> num_states,
             std::unique_ptr<mfem::Mesh> smesh = nullptr);

protected:
   /// object defining the mfem computational mesh
   MachMesh mesh_;

   /// Reference to solver state vector
   mfem::ParMesh &mesh() { return *mesh_.mesh; }
   const mfem::ParMesh &mesh() const { return *mesh_.mesh; }

   /// solver material properties
   nlohmann::json materials;

   /// Members associated with fields
   /// Map of all state vectors used by the solver
   std::map<std::string, FiniteElementState> fields;
   /// Map of dual vectors used by the solver
   std::map<std::string, FiniteElementDual> duals;

   /// Reference to solver state vector
   FiniteElementState &state() { return fields.at("state"); }
   const FiniteElementState &state() const { return fields.at("state"); }
   /// Reference to solver adjoint vector
   FiniteElementState &adjoint() { return fields.at("adjoint"); }
   const FiniteElementState &adjoint() const { return fields.at("adjoint"); }
   /// Reference to solver residual dual vec
   FiniteElementDual &res_vec() { return duals.at("residual"); }
   const FiniteElementDual &res_vec() const { return duals.at("residual"); }

   /// Reference to the state vectors finite element space
   mfem::ParFiniteElementSpace &fes() { return state().space(); }
   const mfem::ParFiniteElementSpace &fes() const { return state().space(); }

   void setUpExternalFields();

   void setState_(std::any function,
                  const std::string &name,
                  mfem::Vector &state) override;

   double calcStateError_(std::any ex_sol,
                          const std::string &name,
                          const mfem::Vector &state) override;
};

}  // namespace mach

#endif
