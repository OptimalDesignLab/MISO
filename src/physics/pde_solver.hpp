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
   PDESolver(MPI_Comm incomm,
             const nlohmann::json &solver_options,
             const int num_states,
             std::unique_ptr<mfem::Mesh> smesh = nullptr);

protected:
   /// Members associated with the mesh
   /// object defining the mfem computational mesh
   std::unique_ptr<mfem::ParMesh> mesh;
#ifdef MFEM_USE_PUMI
   /// pumi mesh object
   std::unique_ptr<apf::Mesh2, pumiDeleter> pumi_mesh;
   bool PCU_previously_initialized = false;
#endif

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
   /// Reference to solver state vector
   FiniteElementState &state;
   /// Vector of dual vectors used by the solver
   std::vector<FiniteElementDual> duals;
   /// Reference to solver residual dual vec
   FiniteElementDual &res_vec;

   void setUpExternalFields();
};

}  // namespace mach

#endif
