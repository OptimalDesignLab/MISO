#include <iostream>

#include "catch.hpp"
#include "nlohmann/json.hpp"
#include "mfem.hpp"

#include "solver.hpp"
#include "test_mach_inputs.hpp"

auto options = R"(
{
   "silent": false,
   "print-options": false,
   "space-dis": {
      "basis-type": "H1",
      "degree": 1
   },
   "time-dis": {
      "steady": true,
      "steady-abstol": 1e-12,
      "steady-reltol": 1e-10,
      "ode-solver": "PTC",
      "t-final": 100,
      "dt": 1e12,
      "max-iter": 10
   },
   "lin-solver": {
      "type": "hypregmres",
      "printlevel": 0,
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
      "printlevel": 3,
      "maxiter": 50,
      "reltol": 1e-10,
      "abstol": 1e-12
   },
   "external-fields": {
      "test_field": {
         "basis-type": "H1",
         "degree": 1,
         "num-states": 1
      }
   }
})"_json;

using namespace mach;
using namespace mfem;

/// Generate mesh 
/// \param[in] nxy - number of nodes in the x and y directions
/// \param[in] nz - number of nodes in the z direction
std::unique_ptr<Mesh> buildMesh(int nxy,
                                int nz);

TEST_CASE("MachInputs Scalar Input Test",
          "[MachInputs]")
{
   // construct the solver, set the initial condition, and solve
   std::unique_ptr<Mesh> mesh = buildMesh(4, 4);
   auto solver = createSolver<TestMachInputSolver>(options,
                                                   move(mesh));
   auto state = solver->getNewField();
   solver->setFieldValue(*state, 0.0);

   auto test_field = solver->getNewField();
   solver->setFieldValue(*test_field, 0.0);

   auto inputs = MachInputs({
      {"test_val", 2.0},
      {"test_field", test_field->GetData()},
      {"state", state->GetData()}
   });

   solver->createOutput("testMachInput");
   auto fun = solver->calcOutput("testMachInput", inputs);
   std::cout << "fun: " << fun << "\n";
   REQUIRE(fun == Approx(2.0).margin(1e-10));

   inputs.at("test_val") = 1.0;
   fun = solver->calcOutput("testMachInput", inputs);
   std::cout << "fun: " << fun << "\n";
   REQUIRE(fun == Approx(1.0).margin(1e-10));
}

TEST_CASE("MachInputs Field Input Test",
          "[MachInputs]")
{
   // construct the solver, set the initial condition, and solve
   std::unique_ptr<Mesh> mesh = buildMesh(4, 4);
   auto solver = createSolver<TestMachInputSolver>(options,
                                                   move(mesh));
   auto state = solver->getNewField();
   solver->setFieldValue(*state, 0.0);

   auto test_field = solver->getNewField();
   solver->setFieldValue(*test_field, 0.0);

   auto inputs = MachInputs({
      {"test_val", 0.0},
      {"test_field", test_field->GetData()},
      {"state", state->GetData()}
   });

   solver->createOutput("testMachInput");
   auto fun = solver->calcOutput("testMachInput", inputs);
   std::cout << "fun: " << fun << "\n";
   REQUIRE(fun == Approx(0.0).margin(1e-10));

   solver->setFieldValue(*test_field, -1.0);
   fun = solver->calcOutput("testMachInput", inputs);
   std::cout << "fun: " << fun << "\n";
   REQUIRE(fun == Approx(-1.0).margin(1e-10));
}

std::unique_ptr<Mesh> buildMesh(int nxy, int nz)
{
   // generate a simple tet mesh
   std::unique_ptr<Mesh> mesh(new Mesh(nxy, nxy, nz,
                                    Element::TETRAHEDRON, true, 1.0,
                                    1.0, (double)nz / (double)nxy, true));

   // assign attributes to top and bottom sides
   for (int i = 0; i < mesh->GetNE(); ++i)
   {
      Element *elem = mesh->GetElement(i);

      Array<int> verts;
      elem->GetVertices(verts);

      bool below = true;
      for (int i = 0; i < 4; ++i)
      {
         auto vtx = mesh->GetVertex(verts[i]);
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
