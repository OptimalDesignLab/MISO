#include <fstream>
#include <iostream>

#include "catch.hpp"
#include "json.hpp"
#include "mfem.hpp"

#include "magnetostatic.hpp"
#include "thermal.hpp"

using namespace std;
using namespace mfem;
using namespace mach;

// Provide the options explicitly for regression tests
auto em_options = R"(
{
   "silent": false,
   "print-options": false,
   "mesh": {
      "file": "data/team13.smb",
      "model-file": "data/team13.egads"
   },
   "space-dis": {
      "basis-type": "nedelec",
      "degree": 1
   },
   "time-dis": {
      "steady": true,
      "steady-abstol": 5e-3,
      "steady-reltol": 5e-6,
      "ode-solver": "PTC",
      "dt": 1e12,
      "res-exp": 2.0,
      "max-iter": 20
   },
   "lin-solver": {
      "type": "hypregmres",
      "printlevel": 3,
      "maxiter": 250,
      "abstol": 1e-8,
      "reltol": 1e-10
   },
   "lin-prec": {
      "type": "hypreams",
      "printlevel": 0
   },
   "nonlin-solver": {
      "type": "relaxed-newton",
      "printlevel": 3,
      "maxiter": 50,
      "reltol": 1e-8,
      "abstol": 1e-6
   },
   "components": {
      "farfields": {
         "material": "air",
         "linear": true,
         "attrs": [1, 2]
      },
      "channel": {
         "attrs": [3, 5],
         "material": "team13",
         "linear": false
      },
      "center": {
         "attr": 6,
         "material": "team13",
         "linear": false
      },
      "airgap": {
         "attrs": [7, 8],
         "material": "air",
         "linear": true
      },
      "windings": {
         "material": "copperwire",
         "linear": true,
         "attrs": [4]
      }
   },
   "problem-opts": {
      "fill-factor": 1.0,
      "current-density": 400000,
      "current": {
         "team13": [4]
      }
   },
   "bcs": {
      "essential": [32, 33]
   },
   "outputs": {
      "co-energy": {}
   }
})"_json;

// area = 0.1 * 0.025
// current = 1000 or 3000 AT

TEST_CASE("TEAM 13 Nonlinear Magnetostatic Benchmark Regression Test",
          "[Magnetostatic-TEAM13]")
{
   auto em_solver = createSolver<MagnetostaticSolver>(em_options);
   auto em_state = em_solver->getNewField();

   em_solver->setInitialCondition(*em_state,
                                  [](const mfem::Vector &x, mfem::Vector &A)
   {
      A = 0.0;
   });

   // int rank;
   // auto comm = MPI_COMM_WORLD;
   // MPI_Comm_rank(comm, &rank);
   // HypreParVector *u_true = em_state->GetTrueDofs();
   // std::cout.precision(16);
   // std::cout << "res norm: " << em_solver->calcResidualNorm(*em_state) << "\n";
   // // std::cout << "u0 norm: " << std::sqrt(em_solver->calcInnerProduct(*em_state, *em_state)) << "\n";
   // std::cout << "u0 norm: " << std::sqrt(InnerProduct(comm, *u_true, *u_true)) << "\n";

   em_solver->solveForState(*em_state);
   em_solver->printSolution("em_sol");
}
