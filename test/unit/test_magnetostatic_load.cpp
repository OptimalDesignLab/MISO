#include <iostream>
#include <random>

#include "adept.h"
#include "catch.hpp"
#include "nlohmann/json.hpp"
#include "mfem.hpp"

#include "utils.hpp"
#include "miso_load.hpp"
#include "magnetostatic.hpp"
#include "material_library.hpp"

using namespace miso;

/// Generate mesh 
/// \param[in] nxy - number of nodes in the x and y directions
/// \param[in] nz - number of nodes in the z direction
mfem::Mesh buildMesh(int nxy, int nz);

static std::default_random_engine gen;
static std::uniform_real_distribution<double> uniform_rand(-1.0,1.0);

auto options = R"(
{
   "space-dis": {
      "degree": 2,
      "basis-type": "ND"
   },
   "current": {
      "group1": {
         "ring": [1]
      }
   },
   "magnets": {
      "Nd2Fe14B": {
         "ccw": [1]
      }
   }
})"_json;

TEST_CASE("MagnetostaticLoad Value Test")
{
   mfem::Mesh smesh = buildMesh(3, 3);
   mfem::ParMesh mesh(MPI_COMM_WORLD, smesh);
   mesh.EnsureNodes();

   const auto dim = mesh.Dimension();

   // create nu coeff
   mfem::ConstantCoefficient nu(1.0);

   adept::Stack diff_stack;
   std::map<std::string, FiniteElementState> fields;
   fields.emplace(std::piecewise_construct,
                  std::forward_as_tuple("state"),
                  std::forward_as_tuple(mesh, options["space-dis"]));

   auto &mesh_gf = *dynamic_cast<mfem::ParGridFunction *>(mesh.GetNodes());
   auto *mesh_fespace = mesh_gf.ParFESpace();
   /// create new state vector copying the mesh's fe space
   fields.emplace(
         std::piecewise_construct,
         std::forward_as_tuple("mesh_coords"),
         std::forward_as_tuple(mesh, *mesh_fespace, "mesh_coords"));
   FiniteElementState &mesh_coords = fields.at("mesh_coords");
   /// set the values of the new GF to those of the mesh's old nodes
   mesh_coords.gridFunc() = mesh_gf;
   /// tell the mesh to use this GF for its Nodes
   /// (and that it doesn't own it)
   mesh.NewNodes(mesh_coords.gridFunc(), false);

   nlohmann::json materials(material_library);

   auto &fes = fields.at("state").space();
   MagnetostaticLoad load(diff_stack, fes, fields, options, materials, nu);

   mfem::HypreParVector tv(&fes);

   auto inputs = MISOInputs({
      {"current_density:group1", 1.0}
   });

   setInputs(load, inputs);
   tv = 0.0;
   addLoad(load, tv);

   auto norm = mfem::ParNormlp(tv, 2.0, MPI_COMM_WORLD);
   std::cout << "\nnorm: " << norm << "\n\n";

   // REQUIRE(norm == Approx(1.8696677851).margin(1e-10));

   inputs.at("current_density:group1") = 2.0;
   setInputs(load, inputs);
   tv = 0.0;
   addLoad(load, tv);

   norm = mfem::ParNormlp(tv, 2.0, MPI_COMM_WORLD);
   std::cout << "\nnorm: " << norm << "\n\n";

   // REQUIRE(norm == Approx(1.9505429368).margin(1e-10));

   inputs.at("current_density:group1") = 0.0;
   setInputs(load, inputs);
   tv = 0.0;
   addLoad(load, tv);

   norm = ParNormlp(tv, 2.0, MPI_COMM_WORLD);
   std::cout << "\nnorm: " << norm << "\n\n";

   // REQUIRE(norm == Approx(1.8280057201).margin(1e-10));

}

mfem::Mesh buildMesh(int nxy, int nz)
{
   /// generate a simple tet mesh
   auto mesh = mfem::Mesh::MakeCartesian3D(nxy, nxy, nz,
                                     mfem::Element::TETRAHEDRON, 1.0,
                                     1.0, (double)nz / (double)nxy, true);

   // assign attributes to top and bottom sides
   for (int i = 0; i < mesh.GetNE(); ++i)
   {
      auto *elem = mesh.GetElement(i);
      elem->SetAttribute(1);

      // Array<int> verts;
      // elem->GetVertices(verts);

      // bool below = true;
      // for (int i = 0; i < 4; ++i)
      // {
      //    auto vtx = mesh.GetVertex(verts[i]);
      //    if (vtx[1] <= 0.5)
      //    {
      //       below = below & true;
      //    }
      //    else
      //    {
      //       below = below & false;
      //    }
      // }
      // if (below)
      // {
      //    elem->SetAttribute(1);
      // }
      // else
      // {
      //    elem->SetAttribute(2);
      // }
   }
   return mesh;
}
