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
      "thermal_contact_resistance": {
         "test": {
            "attrs": [5],
            "h": 1.0
         }
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

      bool left = true;
      for (int j = 0; j < verts.Size(); ++j)
      {
         auto *vtx = mesh->GetVertex(verts[j]);
         if (vtx[0] <= 0.5)
         {
            continue;
         }
         else
         {
            left = false;
         }
      }
      if (left)
      {
         elem->SetAttribute(1);
      }
      else
      {
         elem->SetAttribute(2);
      }
   }

   // assign boundary element attributes to left and right sides
   for (int i = 0; i < mesh->GetNBE(); ++i)
   {
      auto *elem = mesh->GetBdrElement(i);

      mfem::Array<int> verts;
      elem->GetVertices(verts);

      bool left = true;
      bool right = true;
      bool top = true;
      bool bottom = true;
      for (int j = 0; j < verts.Size(); ++j)
      {
         auto *vtx = mesh->GetVertex(verts[j]);
         left = left && abs(vtx[0] - 0.0) < 1e-12;
         right = right && abs(vtx[0] - 2.0) < 1e-12;
         top = top && abs(vtx[1] - 1.0) < 1e-12;
         bottom = bottom && abs(vtx[1] - 0.0) < 1e-12;
      }
      if (left)
      {
         elem->SetAttribute(1);
      }
      else if (right)
      {
         elem->SetAttribute(2);
      }
      else if (top)
      {
         elem->SetAttribute(3);
      }
      else if (bottom)
      {
         elem->SetAttribute(4);
      }
   }

   // add internal boundary elements
   for (int i = 0; i < mesh->GetNumFaces(); ++i)
   {
      int e1, e2;
      mesh->GetFaceElements(i, &e1, &e2);
      if (e1 >= 0 && e2 >= 0 && mesh->GetAttribute(e1) != mesh->GetAttribute(e2))
      {
         // This is the internal face between attributes.
         auto *new_elem = mesh->GetFace(i)->Duplicate(mesh.get());
         new_elem->SetAttribute(5);
         mesh->AddBdrElement(new_elem);
      }
   }

   mesh->FinalizeTopology(); // Finalize to build relevant tables
   mesh->Finalize();
   mesh->SetAttributes();

   return mesh;
}