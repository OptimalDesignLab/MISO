#include <fstream>
#include <iostream>
#include <set>

#include "catch.hpp"
#include "json.hpp"
#include "mfem.hpp"

#include "magnetostatic.hpp"
#include "thermal.hpp"

using namespace mfem;
using namespace mach;

// Provide the options explicitly for regression tests
auto em_options = R"(
{
   "mesh": {
      "file": "data/coulomb1984.smb",
      "model-file": "data/coulomb1984.egads",
      "refine": 0
   },
   "space-dis": {
      "basis-type": "nedelec",
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
      "type": "hypreams",
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
      "ring": {
         "material": "copperwire",
         "attrs": [1, 2],
         "linear": true
      }
   },
   "bcs": {
      "essential": [1, 3]
   },
   "problem-opts": {
      "fill-factor": 1.0,
      "current-density": 1.0,
      "current": {
         "ring": [1, 2]
      }
   }
})"_json;

/// exact force is 0.078 N
TEST_CASE("Force Regression Test Coulomb 1984 Paper")
{
   auto em_solver = createSolver<MagnetostaticSolver>(em_options);
   auto em_state = em_solver->getNewField();
   *em_state = 0.0; // initialize zero field

   auto current_density = 3e6;
   MachInputs inputs {
      {"current-density", current_density},
      {"fill-factor", 1.0},
      {"state", em_state->GetData()}
   };
   em_solver->solveForState(inputs, *em_state);

   nlohmann::json force_options = {
      {"attributes", {1}},
      {"axis", {0, 0, 1}}
   };
   em_solver->createOutput("force", force_options);

   double force = em_solver->calcOutput("force", inputs);
   REQUIRE(force == Approx(-0.0791988853).margin(1e-10));

   force_options["attributes"] = {2};
   em_solver->setOutputOptions("force", force_options);

   force = em_solver->calcOutput("force", inputs);
   REQUIRE(force == Approx(0.0781336686).margin(1e-10));

   nlohmann::json torque_options = {
      {"attributes", {1}},
      {"axis", {0, 0, 1}},
      {"about", {0.0, 0.0, 0.0}}
   };
   em_solver->createOutput("torque", torque_options);

   auto &v = em_solver->getField("vtorque");
   em_solver->printField("v", v, "v", 0);

   double torque = em_solver->calcOutput("torque", inputs);
   REQUIRE(torque == Approx(0.0000104977).margin(1e-10));

}
