#include <fstream>
#include <iostream>

#include "catch.hpp"
#include "json.hpp"
#include "mfem.hpp"

#include "mesh_movement.hpp"
#include "mach_egads.hpp"

using namespace std;
using namespace mfem;
using namespace mach;

// Provide the options explicitly for regression tests
auto options = R"(
{
   "silent": false,
   "print-options": false,
   "mesh": {
      "file": "data/cyl.smb",
      "model-file": "data/cyl.egads"
   },
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
   "components": {
   },
   "problem-opts": {
      "uniform-stiff": {
         "lambda": 1,
         "mu": 1
      }
   },
   "outputs": {
   }
})"_json;

TEST_CASE("Mesh Movement EGADS Cylinder Test",
          "[Mesh-Movement-EGADS-Cyl]")
{
   SECTION("...for mesh sizing nxy = ")
   {
      // construct the solver, set the initial condition, and solve
      auto solver = createSolver<LEAnalogySolver>(options);
      auto coord_field = solver->getNewField();

      std::string old_model("data/cyl.egads");
      std::string new_model("data/cyl2.egads");
      std::string tess("data/cyl.eto");
      auto surface_displacement = solver->getNewField();

      mapSurfaceMesh(old_model, new_model, tess, *surface_displacement);

      *surface_displacement += solver->getMeshCoordinates();

      solver->setInitialCondition(*coord_field, *surface_displacement);
      solver->solveForState(*coord_field);
      // solver->printSolution("final");

      // Compute error and check against appropriate target
      // mfem::VectorFunctionCoefficient dispEx(3, boxDisplacement);
      // double l2_error = fields[0]->ComputeL2Error(dispEx);
      // std::cout << "\n\nl2 error in field: " << l2_error << "\n\n\n";
      // REQUIRE(l2_error == Approx(target_error[ref - 1]).margin(1e-10));

      auto &mesh_coords = solver->getMeshCoordinates();
      mesh_coords.SetData(coord_field->GetData());
      solver->printMesh("moved_egads_cyl_mesh");
   }
}
