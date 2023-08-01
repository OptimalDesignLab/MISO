#include <fstream>
#include <iostream>

#include "catch.hpp"
#include "json.hpp"
#include "mfem.hpp"

#include "magnetostatic.hpp"
#include "thermal.hpp"

using namespace std;
using namespace mfem;
using namespace miso;

// Provide the options explicitly for regression tests
auto em_options = R"(
{
   "silent": false,
   "print-options": false,
   "mesh": {
      "file": "data/wire.smb",
      "model-file": "data/wire.egads"
   },
   "space-dis": {
      "basis-type": "nedelec",
      "degree": 2
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
      "attr1": {
         "material": "copperwire",
         "attr": 1,
         "linear": true
      }
   },
   "bcs": {
      "essential": [1, 2, 3, 4]
   },
   "problem-opts": {
      "fill-factor": 1.0,
      "current_density": 1.2732395447351627e7,
      "current": {
         "z": [1]
      }
   },
   "outputs": {
      "co-energy": {}
   }
})"_json;

auto therm_options = R"(
{
   "print-options": false,
   "mesh": {
      "file": "data/wire.smb",
      "model-file": "data/wire.egads"
   },
   "space-dis": {
      "basis-type": "H1",
      "degree": 1
   },
   "steady": false,
   "time-dis": {
      "ode-solver": "MIDPOINT",
      "const-cfl": true,
      "cfl": 1.0,
      "dt": 0.01,
      "t-final": 10.5
   },
   "lin-prec": {
      "type": "hypreboomeramg"
   },
   "lin-solver": {
      "reltol": 1e-14,
      "abstol": 0.0,
      "printlevel": 0,
      "maxiter": 500
   },
   "nonlin-solver": {
      "type": "newton",
      "printlevel": 3,
      "maxiter": 50,
      "reltol": 1e-10,
      "abstol": 1e-12
   },
   "components": {
      "attr1": {
         "material": "copperwire",
         "attr": 1,
         "linear": true
      }
   },
   "problem-opts": {
      "fill-factor": 1.0,
      "current_density": 1.2732395447351627e7,
      "frequency": 0,
      "current": {
         "z": [1]
      },
      "rho-agg": 10,
      "init-temp": 300,
      "outflux-type": "test"
   },
   "bcs": {
      "outflux": [0, 0, 0, 0]
   },
   "outputs": {
      "temp-agg": {}
   },
   "external-fields": {
      "mvp": {
         "basis-type": "nedelec",
         "degree": 1,
         "num-states": 1
      }
   }
})"_json;

TEST_CASE("Joule Wire Solver Regression Test",
          "[Joule-Wire]")
{
    auto em_solver = createSolver<MagnetostaticSolver>(em_options);
    auto em_state = em_solver->getNewField();
    *em_state = 0.0; // initialize zero field
    em_solver->solveForState(*em_state);
    em_solver->printSolution("em_sol");

    auto therm_solver = createSolver<ThermalSolver>(therm_options);
    auto therm_state = therm_solver->getNewField();

    therm_solver->setResidualInput("mvp", *em_state);
    *therm_state = 300; // initialize termperature field at constant 300 K
    therm_solver->solveForState(*therm_state);

    /// the final temperature in the wire after 10.5s should be ~308.69 K
   auto diff = therm_solver->getNewField();
   *diff = 308.69037774418;
   *diff -= *therm_state;
   REQUIRE(diff->Norml2() == Approx(0.0).margin(1e-10));
}
