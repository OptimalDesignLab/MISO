#ifndef MACH_MESH_WARPER
#define MACH_MESH_WARPER

#include <map>
#include <string>

#include "mfem.hpp"
#include "mpi.h"
#include "nlohmann/json.hpp"

#include "abstract_solver.hpp"
#include "finite_element_dual.hpp"
#include "finite_element_state.hpp"
#include "pde_solver.hpp"

namespace mach
{
class MeshWarper : public AbstractSolver2
{
public:
   int getSurfaceCoordsSize() const;
   void getInitialSurfaceCoords(mfem::Vector &surface_coords) const;

   int getVolumeCoordsSize() const;
   void getInitialVolumeCoords(mfem::Vector &volume_coords) const;

   /// Get for surface coord, get its index into the volume coords array
   void getSurfCoordIndices(mfem::Array<int> &indices) const;

   MeshWarper(MPI_Comm incomm,
              const nlohmann::json &solver_options,
              std::unique_ptr<mfem::Mesh> smesh = nullptr);

private:
   /// object defining the mfem computational mesh
   MachMesh mesh_;

   /// Reference to solver state vector
   mfem::ParMesh &mesh() { return *mesh_.mesh; }
   const mfem::ParMesh &mesh() const { return *mesh_.mesh; }

   /// Members associated with fields
   /// Map of all state vectors used by the solver
   std::map<std::string, FiniteElementState> fields;
   /// Map of dual vectors used by the solver
   std::map<std::string, FiniteElementDual> duals;

   /// Reference to solver state vector
   FiniteElementState &state() { return fields.at("state"); }
   const FiniteElementState &state() const { return fields.at("state"); }

   /// Reference to the state vectors finite element space
   mfem::ParFiniteElementSpace &fes() { return state().space(); }
   const mfem::ParFiniteElementSpace &fes() const { return state().space(); }

   mfem::Array<int> surface_indices;
   mfem::Vector surf_coords;
   mfem::Vector vol_coords;
};

}  // namespace mach

#endif
