#ifndef MACH_PDE_SOLVER
#define MACH_PDE_SOLVER

#include <memory>
#include <vector>

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
      mesh->destroyNative();
      apf::destroyMesh(mesh);
   }
};

}  // namespace mach
#endif

#include "abstract_solver.hpp"
#include "finite_element_state.hpp"
#include "finite_element_dual.hpp"

namespace mach
{
class PDESolver : public AbstractSolver2
{
public:
   FiniteElementState &getState() { return state(); }
   const FiniteElementState &getState() const { return state(); }

   PDESolver(MPI_Comm incomm,
             const nlohmann::json &solver_options,
             const int num_states,
             std::unique_ptr<mfem::Mesh> smesh = nullptr);

protected:
   /// Members associated with the mesh
   /// object defining the mfem computational mesh
   std::unique_ptr<mfem::ParMesh> mesh_ = nullptr;
   // #ifdef MFEM_USE_PUMI
   //    /// pumi mesh object
   //    std::unique_ptr<apf::Mesh2, pumiDeleter> pumi_mesh;
   //    bool PCU_previously_initialized = false;
   // #endif

   /// Constructs the mesh member based on c preprocesor defs
   /// \param[in] smesh - if provided, defines the mesh for the problem
   std::unique_ptr<mfem::ParMesh> constructMesh(
       MPI_Comm comm,
       const nlohmann::json &mesh_options,
       std::unique_ptr<mfem::Mesh> smesh);

   /*
   /// Construct PUMI Mesh
   std::unique_ptr<mfem::ParMesh> constructPumiMesh(
       MPI_Comm comm,
       const nlohmann::json &mesh_options);
   */

   /// Members associated with fields
   /// Vector of all state vectors used by the solver
   std::vector<FiniteElementState> fields;
   /// Vector of dual vectors used by the solver
   std::vector<FiniteElementDual> duals;

   /// Reference to solver state vector
   FiniteElementState &state() { return fields[0]; }
   const FiniteElementState &state() const { return fields[0]; }
   /// Reference to solver adjoint vector
   FiniteElementState &adjoint() { return fields[1]; }
   const FiniteElementState &adjoint() const { return fields[1]; }
   /// Reference to solver residual dual vec
   FiniteElementDual &res_vec() { return duals[0]; }
   const FiniteElementDual &res_vec() const { return duals[0]; }

   /// Reference to the state vectors finite element space
   mfem::ParFiniteElementSpace &fes() { return state().space(); }
   const mfem::ParFiniteElementSpace &fes() const { return state().space(); }

   void setUpExternalFields();

   /// ParaView object for saving fields
   mfem::ParaViewDataCollection vis;

   /// Time-stepping overrides
   void initialHook(const mfem::Vector &state) override;

   void iterationHook(int iter,
                      double t,
                      double dt,
                      const mfem::Vector &state) override;

   void terminalHook(int iter,
                     double t_final,
                     const mfem::Vector &state) override;
};

}  // namespace mach

#endif
