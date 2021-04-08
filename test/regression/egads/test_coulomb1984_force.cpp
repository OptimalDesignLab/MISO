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

   auto force_options = R"(
   {
      "attrs": [1]
   })"_json;
   em_solver->createOutput("force", force_options);
   auto &v = em_solver->getField("v");
   v = 0.0;

   double ring1_data[] = {0.0, 0.0, 1.0};
   VectorConstantCoefficient v_ring1(Vector{ring1_data, 3});
   v.ProjectCoefficient(v_ring1, 1);

   double force = em_solver->calcOutput("force", inputs);

   /// exact solution is 0.078
   REQUIRE(force == Approx(-0.0791988853).margin(1e-10));
}
