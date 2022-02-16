#include <iostream>

#include "adept.h"
#include "catch.hpp"
#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "coefficient.hpp"
#include "mach_load.hpp"
#include "current_load.hpp"
#include "utils.hpp"

static std::default_random_engine gen;
static std::uniform_real_distribution<double> uniform_rand(-1.0,1.0);

using namespace mach;

/// Generate mesh 
/// \param[in] nxy - number of nodes in the x and y directions
/// \param[in] nz - number of nodes in the z direction
mfem::Mesh buildMesh(int nxy, int nz);

TEST_CASE("CurrentLoad setInputs")
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
   auto options = R"({
      "current": {
         "test": {
            "box1": [1],
            "box2": [2]
         }
      }
   })"_json;
   CurrentLoad load(diff_stack, fes, fields, options);

   MachInputs inputs{{"current_density:test", 1.0}};

   setInputs(load, inputs);

   mfem::Vector tv(getSize(load));
   tv = 0.0;
   addLoad(load, tv);

   auto norm = mfem::ParNormlp(tv, 2.0, MPI_COMM_WORLD);
   std::cout << "norm: " << norm << "\n";

   REQUIRE(norm == Approx(0.3186887196).margin(1e-10));

   inputs.at("current_density:test") = 2.0;
   setInputs(load, inputs);

   tv = 0.0;
   addLoad(load, tv);

   norm = mfem::ParNormlp(tv, 2.0, MPI_COMM_WORLD);
   std::cout << "norm: " << norm << "\n";

   REQUIRE(norm == Approx(0.6373774392).margin(1e-10));
}

TEST_CASE("CurrentLoad vectorJacobianProduct wrt current_density")
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
   auto options = R"({
      "current": {
         "test1": {
            "box1": [1]
         },
         "test2": {
            "box2": [2]
         }
      }
   })"_json;
   CurrentLoad load(diff_stack, fes, fields, options);

   auto current_density = 1e6;
   MachInputs inputs{{"current_density:test1", current_density}};

   mfem::Vector load_bar(getSize(load));
   for (int i = 0; i < load_bar.Size(); ++i)
   {
      load_bar(i) = uniform_rand(gen);
   }
   setInputs(load, inputs);
   double wrt_bar_local = vectorJacobianProduct(load, load_bar, "current_density:test1");
   double wrt_bar;
   MPI_Allreduce(&wrt_bar_local,
                 &wrt_bar,
                 1,
                 MPI_DOUBLE,
                 MPI_SUM,
                 MPI_COMM_WORLD);   

   /// somewhat large step size since the magnitude of current density is large
   auto delta = 1e-2;
   double wrt_bar_fd_local = 0.0;
   mfem::Vector tv(getSize(load));

   inputs.at("current_density:test1") = current_density + delta;
   setInputs(load, inputs);
   tv = 0.0;
   addLoad(load, tv);
   wrt_bar_fd_local += load_bar * tv;

   inputs.at("current_density:test1") = current_density - delta;
   setInputs(load, inputs);
   tv = 0.0;
   addLoad(load, tv);
   wrt_bar_fd_local -= load_bar * tv;

   wrt_bar_fd_local /= 2*delta;
   double wrt_bar_fd;
   MPI_Allreduce(&wrt_bar_fd_local,
                 &wrt_bar_fd,
                 1,
                 MPI_DOUBLE,
                 MPI_SUM,
                 MPI_COMM_WORLD); 

   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   if (rank == 0)
   {
      std::cout << "wrt_bar: " << wrt_bar << "\n";
      std::cout << "wrt_bar_fd: " << wrt_bar_fd << "\n";
   }
   REQUIRE(wrt_bar == Approx(wrt_bar_fd));
}

TEST_CASE("CurrentLoad vectorJacobianProduct wrt mesh_coords")
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
      "current": {
         "test": {
            "z": [1, 2]
         }
      }
   })"_json;
   CurrentLoad load(diff_stack, fes, fields, options);

   mfem::Vector mesh_coords_tv(mesh_coords.space().GetTrueVSize());
   mesh_coords.setTrueVec(mesh_coords_tv);

   auto current_density = 1e6;
   auto inputs = MachInputs({
      {"current_density:test", current_density},
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
   // generate a simple tet mesh
   auto mesh = mfem::Mesh::MakeCartesian3D(nxy, nxy, nz,
                                           mfem::Element::HEXAHEDRON,
                                           1.0, 1.0, (double)nz / (double)nxy,
                                           true);

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
