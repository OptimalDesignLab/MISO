#include <iostream>
#include <random>

#include "adept.h"
#include "catch.hpp"
#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "utils.hpp"
#include "mach_load.hpp"
#include "magnetic_load.hpp"
#include "material_library.hpp"

static std::default_random_engine gen;
static std::uniform_real_distribution<double> uniform_rand(-1.0,1.0);

using namespace mach;

/// Generate mesh 
/// \param[in] nxy - number of nodes in the x and y directions
/// \param[in] nz - number of nodes in the z direction
mfem::Mesh buildMesh(int nxy, int nz);

TEST_CASE("MagneticLoad Value Test")
{
   auto smesh = buildMesh(3, 3);
   mfem::ParMesh mesh(MPI_COMM_WORLD, smesh);
   mesh.EnsureNodes();

   auto p = 2;
   const auto dim = mesh.Dimension();

   // get the finite-element space for the state
   mfem::ND_FECollection fec(p, dim);
   mfem::ParFiniteElementSpace fes(&mesh, &fec);

   adept::Stack diff_stack;
   std::map<std::string, FiniteElementState> fields;
   {
      auto &mesh_gf = *dynamic_cast<mfem::ParGridFunction *>(mesh.GetNodes());
      auto &mesh_fespace = *mesh_gf.ParFESpace();

      /// create new state vector copying the mesh's fe space
      fields.emplace(
            std::piecewise_construct,
            std::forward_as_tuple("mesh_coords"),
            std::forward_as_tuple(mesh, mesh_fespace, "mesh_coords"));
      auto &mesh_coords = fields.at("mesh_coords");
      /// set the values of the new GF to those of the mesh's old nodes
      mesh_coords.gridFunc() = mesh_gf;
      /// tell the mesh to use this GF for its Nodes
      /// (and that it doesn't own it)
      mesh.NewNodes(mesh_coords.gridFunc(), false);
   }

   auto options = R"({
      "magnets": {
         "testmat": {
            "north": [1],
            "south": [2]
         }
      }
   })"_json;
   mfem::ConstantCoefficient nu(1.0); ///(M_PI*4e-7));

   MagneticLoad load_0(diff_stack, fes, fields, options, material_library, nu);

   MagneticLoad load(std::move(load_0));

   MachInputs inputs;
   setInputs(load, inputs);

   mfem::Vector tv(getSize(load));
   tv = 0.0;
   addLoad(load, tv);

   auto norm = mfem::ParNormlp(tv, 2.0, MPI_COMM_WORLD);
   REQUIRE(norm == Approx(1.8280057201).margin(1e-10));

   setInputs(load, inputs);
   tv = 0.0;
   addLoad(load, tv);

   norm = ParNormlp(tv, 2.0, MPI_COMM_WORLD);
   REQUIRE(norm == Approx(1.8280057201).margin(1e-10));
}

TEST_CASE("MagneticLoad vectorJacobianProduct wrt mesh_coords")
{
   auto smesh = buildMesh(4, 4);
   mfem::ParMesh mesh(MPI_COMM_WORLD, smesh);
   mesh.EnsureNodes();

   auto p = 2;
   const auto dim = mesh.Dimension();

   // get the finite-element space for the state
   mfem::ND_FECollection fec(p, dim);
   mfem::ParFiniteElementSpace fes(&mesh, &fec);

   adept::Stack diff_stack;
   std::map<std::string, FiniteElementState> fields;
   
   auto &mesh_gf = *dynamic_cast<mfem::ParGridFunction *>(mesh.GetNodes());
   auto &mesh_fespace = *mesh_gf.ParFESpace();

   /// create new state vector copying the mesh's fe space
   fields.emplace(
         std::piecewise_construct,
         std::forward_as_tuple("mesh_coords"),
         std::forward_as_tuple(mesh, mesh_fespace, "mesh_coords"));
   auto &mesh_coords = fields.at("mesh_coords");
   /// set the values of the new GF to those of the mesh's old nodes
   mesh_coords.gridFunc() = mesh_gf;
   /// tell the mesh to use this GF for its Nodes
   /// (and that it doesn't own it)
   mesh.NewNodes(mesh_coords.gridFunc(), false);

   auto options = R"({
      "magnets": {
         "testmat": {
            "north": [1],
            "south": [2]
         }
      }
   })"_json;
   mfem::ConstantCoefficient nu(1.0); ///(M_PI*4e-7));

   MagneticLoad load(diff_stack, fes, fields, options, material_library, nu);

   mfem::Vector mesh_coords_tv(mesh_coords.space().GetTrueVSize());
   mesh_coords.setTrueVec(mesh_coords_tv);

   auto inputs = MachInputs({
      {"mesh_coords", mesh_coords_tv}
   });
   setInputs(load, inputs);

   mfem::Vector load_bar(getSize(load));
   for (int i = 0; i < load_bar.Size(); ++i)
   {
      load_bar(i) = uniform_rand(gen);
   }

   mfem::Vector wrt_bar(mesh_coords.space().GetTrueVSize());
   wrt_bar = 0.0;
   vectorJacobianProduct(load, load_bar, "mesh_coords", wrt_bar);

   // initialize the vector that we use to perturb the mesh nodes
   mfem::Vector v_tv(mesh_coords.space().GetTrueVSize());
   for (int i = 0; i < v_tv.Size(); ++i)
   {
      v_tv(i) = uniform_rand(gen);
   }

   auto dJdx_v_local = wrt_bar * v_tv;
   double dJdx_v;
   MPI_Allreduce(&dJdx_v_local,
                 &dJdx_v,
                 1,
                 MPI_DOUBLE,
                 MPI_SUM,
                 MPI_COMM_WORLD);

   // now compute the finite-difference approximation...
   auto delta = 1e-5;
   double dJdx_v_fd_local = 0.0;
   mfem::Vector load_vec(getSize(load));

   add(mesh_coords_tv, delta, v_tv, mesh_coords_tv);
   mesh_coords.distributeSharedDofs(mesh_coords_tv); // update mesh nodes
   inputs.at("mesh_coords") = mesh_coords_tv;
   setInputs(load, inputs);

   load_vec = 0.0;
   addLoad(load, load_vec);
   dJdx_v_fd_local += load_bar * load_vec;

   add(mesh_coords_tv, -2*delta, v_tv, mesh_coords_tv);
   mesh_coords.distributeSharedDofs(mesh_coords_tv); // update mesh nodes
   inputs.at("mesh_coords") = mesh_coords_tv;
   setInputs(load, inputs);

   load_vec = 0.0;
   addLoad(load, load_vec);
   dJdx_v_fd_local -= load_bar * load_vec;

   dJdx_v_fd_local /= 2*delta;

   double dJdx_v_fd;
   MPI_Allreduce(&dJdx_v_fd_local,
                 &dJdx_v_fd,
                 1,
                 MPI_DOUBLE,
                 MPI_SUM,
                 MPI_COMM_WORLD);

   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   if (rank == 0)
   {
      std::cout << "dJdx_v: " << dJdx_v << "\n";
      std::cout << "dJdx_v_fd: " << dJdx_v_fd << "\n";
   }

   REQUIRE(dJdx_v == Approx(dJdx_v_fd).margin(1e-8));
}

mfem::Mesh buildMesh(int nxy, int nz)
{
   /// generate a simple hex mesh
   auto mesh = mfem::Mesh::MakeCartesian3D(nxy, nxy, nz,
                                           mfem::Element::TETRAHEDRON, 1.0,
                                           1.0, (double)nz / (double)nxy, true);

   // assign attributes to top and bottom sides
   for (int i = 0; i < mesh.GetNE(); ++i)
   {
      auto *elem = mesh.GetElement(i);

      mfem::Array<int> verts;
      elem->GetVertices(verts);

      bool below = true;
      for (int i = 0; i < 4; ++i)
      {
         auto vtx = mesh.GetVertex(verts[i]);
         if (vtx[1] <= 0.5)
         {
            below = below & true;
         }
         else
         {
            below = below & false;
         }
      }
      if (below)
      {
         elem->SetAttribute(1);
      }
      else
      {
         elem->SetAttribute(2);
      }
   }
   return mesh;
}
