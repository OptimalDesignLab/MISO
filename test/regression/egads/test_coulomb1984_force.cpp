#include "catch.hpp"
#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "magnetostatic.hpp"

using namespace miso;

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
      "type": "gmres",
      "printlevel": 1,
      "maxiter": 200,
      "kdim": 200,
      "abstol": 1e-14,
      "reltol": 1e-14
   },
   "lin-prec": {
      "type": "hypreams",
      "printlevel": 0
   },
   "nonlin-solver": {
      "type": "relaxednewton",
      "printlevel": 1,
      "maxiter": 1,
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
   "current": {
      "test": {
         "ring": [1, 2]
      }
   }
})"_json;

/// exact force is 0.078 N
TEST_CASE("Force Regression Test Coulomb 1984 Paper")
{
   MagnetostaticSolver em_solver(MPI_COMM_WORLD, em_options, nullptr);
   mfem::Vector em_state(em_solver.getStateSize());
   em_state = 0.0; // initialize zero field

   auto current_density = 3e6;
   MISOInputs inputs {
      {"current_density:test", current_density},
      {"state", em_state}
   };
   em_solver.solveForState(inputs, em_state);

   em_solver.createOutput("energy");
   double energy = em_solver.calcOutput("energy", inputs);
   REQUIRE(energy == Approx(0.0142746123).margin(1e-10));

   nlohmann::json force_options = {
      {"attributes", {1}},
      {"axis", {0, 0, 1}}
   };
   em_solver.createOutput("force", force_options);

   double force = em_solver.calcOutput("force", inputs);
   REQUIRE(force == Approx(-0.0791988853).margin(1e-10));

   force_options["attributes"] = {2};
   em_solver.setOutputOptions("force", force_options);

   force = em_solver.calcOutput("force", inputs);
   REQUIRE(force == Approx(0.0781336686).margin(1e-10));

   nlohmann::json torque_options = {
      {"attributes", {1}},
      {"axis", {0, 0, 1}},
      {"about", {0.0, 0.0, 0.0}}
   };
   em_solver.createOutput("torque", torque_options);

   double torque = em_solver.calcOutput("torque", inputs);
   REQUIRE(torque == Approx(0.0000104977).margin(1e-10));
}
