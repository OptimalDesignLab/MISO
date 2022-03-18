#include "catch.hpp"
#include "nlohmann/json.hpp"
#include "mfem.hpp"

#include "mesh_warper.hpp"

auto warp_options = R"(
{
   "space-dis": {
      "basis-type": "h1",
      "degree": 1
   },
   "lin-solver": {
      "type": "pcg",
      "printlevel": 1,
      "maxiter": 100,
      "abstol": 1e-14,
      "reltol": 1e-14
   },
   "nonlin-solver": {
      "type": "newton",
      "printlevel": 3,
      "maxiter": 1,
      "reltol": 1e-10,
      "abstol": 1e-9
   },
   "bcs": {
      "essential": "all"
   }
})"_json;

TEST_CASE("")
{
   auto comm = MPI_COMM_WORLD;

   int nxyz = 2;
   auto smesh = std::make_unique<mfem::Mesh>(
      mfem::Mesh::MakeCartesian3D(nxyz, nxyz, nxyz,
                                  mfem::Element::TETRAHEDRON));

   mach::MeshWarper warper(comm, warp_options, std::move(smesh));

   auto surf_mesh_size = warper.getSurfaceCoordsSize();
   mfem::Vector surf_coords(surf_mesh_size);
   warper.getInitialSurfaceCoords(surf_coords);

   auto vol_mesh_size = warper.getVolumeCoordsSize();
   mfem::Vector vol_coords(vol_mesh_size);
   warper.getInitialVolumeCoords(vol_coords);

   for (int i = 0; i < surf_mesh_size; i+=3)
   {
      surf_coords(i + 0) += 1.0; 
      surf_coords(i + 1) += 1.0; 
      surf_coords(i + 2) += 0.0; 
   }

   mfem::Array<int> surf_indices;
   warper.getSurfCoordIndices(surf_indices);

   for (int i = 0; i < surf_mesh_size; ++i)
   {
      vol_coords(surf_indices[i]) = surf_coords(i);
   }

   warper.solveForState(vol_coords);
   vol_coords.Print(mfem::out, 3);
}
