#include <fstream>
#include <iostream>

#include "catch.hpp"
#include "nlohmann/json.hpp"
#include "mfem.hpp"

#include "thermal.hpp"

std::unique_ptr<mfem::Mesh> buildMesh(int nxy);

TEST_CASE("Thermal Solver Thermal Contact Resistance Regression Test")
{

auto options = R"(
{
   "silent": false,
   "print-options": false,
   "problem": "box",
   "space-dis": {
      "basis-type": "dg",
      "degree": 1
   },
   "lin-solver": {
      "type": "gmres",
      "printlevel": 1,
      "maxiter": 100,
      "abstol": 1e-14,
      "reltol": 1e-14
   },
   "lin-prec": {
      "type": "hypreboomeramg",
      "printlevel": 0
   },
   "nonlin-solver": {
      "type": "newton",
      "printlevel": 1,
      "maxiter": 1,
      "reltol": 1e-10,
      "abstol": 1e-10
   },
   "components": {
      "box1": {
         "attrs": [1, 2],
         "material": {
            "name": "test",
            "kappa": 1.0
         }
      }
   },
   "interfaces": {
      "tcr": {
         "attrs": [1],
         "kind": "thermal_contact_resistance",
         "name": "test",
         "h_c": 1.0
      }
   },
   "bcs": {
      "weak-essential": [1, 3]
   }
})"_json;

for (int order = 1; order <= 1; ++order)
{
   options["space-dis"]["degree"] = order;
   int nxy = 4;
   for (int ref = 1; ref <= 1; ++ref)
   {  
      nxy *= 1;
      DYNAMIC_SECTION("...for order " << order << " and mesh sizing nxy = " << nxy)
      {
         // construct the solver, set the initial condition, and solve
         std::unique_ptr<mfem::Mesh> smesh = buildMesh(nxy);

         mach::ThermalSolver solver(MPI_COMM_WORLD, options, std::move(smesh));
         mfem::Vector state_tv(solver.getStateSize());

         auto &state = solver.getState();

         /// Set initial/boundary conditions
         // solver.setState(aex`act, state_tv);

         solver.setState([](const mfem::Vector &p)
         {
            auto x = p(0);

            return x;
         },
         state_tv);


         solver.solveForState(state_tv);
      }
   }
}

}

std::unique_ptr<mfem::Mesh> buildMesh(int nxy)
{
   // generate a simple tet mesh
   std::unique_ptr<mfem::Mesh> mesh(
      new mfem::Mesh(mfem::Mesh::MakeCartesian2D(nxy, nxy,
                                     mfem::Element::TRIANGLE)));

   // assign attributes to top and bottom sides
   for (int i = 0; i < mesh->GetNE(); ++i)
   {
      auto *elem = mesh->GetElement(i);

      mfem::Array<int> verts;
      elem->GetVertices(verts);

      bool below = true;
      for (int i = 0; i < verts.Size(); ++i)
      {
         auto *vtx = mesh->GetVertex(verts[i]);
         if (vtx[1] <= 0.5)
         {
            below = below;
         }
         else
         {
            below = false;
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

   // assign attributes to middle edge
   for (int i = 0; i < mesh->GetNumFaces(); i++)
   {
      auto *face = const_cast<mfem::Element *>(mesh->GetFace(i));

      mfem::Array<int> verts;
      face->GetVertices(verts);

      bool on_edge = true;
      for (int i = 0; i < verts.Size(); ++i)
      {
         auto *vtx = mesh->GetVertex(verts[i]);
         if (abs(vtx[1] - 0.5) < 1e-6)
         {
            on_edge = on_edge;
         }
         else
         {
            on_edge = false;
         }
      }
      if (on_edge)
      {
         face->SetAttribute(2);
      }
      else
      {
         face->SetAttribute(0);
      }

      std::cout << "face attr: " << face->GetAttribute() << "\n";

   }

   // assign attributes to middle edge
   for (int i = 0; i < mesh->GetNumFaces(); i++)
   {
      auto *face = mesh->GetFace(i);

      mfem::Array<int> verts;
      face->GetVertices(verts);

      bool on_edge = true;
      for (int i = 0; i < verts.Size(); ++i)
      {
         auto *vtx = mesh->GetVertex(verts[i]);
         if (abs(vtx[1] - 0.5) < 1e-6)
         {
            on_edge = on_edge;
         }
         else
         {
            on_edge = false;
         }
      }
      if (on_edge)
      {
         face->SetAttribute(1);
      }
      else
      {
         face->SetAttribute(-1);
      }

      std::cout << "face attr: " << face->GetAttribute() << "\n";

   }

   mesh->SetAttributes();

   // assign attributes to middle edge
   for (int i = 0; i < mesh->GetNumFaces(); i++)
   {
      auto *face = mesh->GetFace(i);

      std::cout << "face attr: " << face->GetAttribute() << "\n";
   }

   return mesh;
}