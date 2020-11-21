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
      "file": "data/ser_team13.smb",
      "model-file": "data/team13.egads"
   },
   "space-dis": {
      "basis-type": "nedelec",
      "degree": 1
   },
   "time-dis": {
      "steady": true,
      "steady-abstol": 1e-8,
      "steady-reltol": 1e-8,
      "ode-solver": "PTC",
      "t-final": 100,
      "dt": 1e12,
      "max-iter": 2
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
      "abstol": 1e-8
   },
   "components": {
      "farfields": {
         "material": "air",
         "linear": true,
         "attrs": [1, 2]
      },
      "channel": {
         "attrs": [6, 7],
         "material": "team13",
         "linear": false
      },
      "center": {
         "attr": 5,
         "material": "team13",
         "linear": false
      },
      "airgap": {
         "attrs": [3, 8],
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
      "current-density": 1200000,
      "current": {
         "team13": [4]
      }
   },
   "outputs": {
      "co-energy": {}
   }
})"_json;

// area = 0.1 * 0.0025
// current = 1000 or 3000 AT

TEST_CASE("TEAM 13 Nonlinear Magnetostatic Benchmark Regression Test",
          "[Magnetostatic-TEAM13]")
{
   auto em_solver = createSolver<MagnetostaticSolver>(em_options);
   auto em_state = em_solver->getNewField();

   auto uinit = [](const mfem::Vector &x, mfem::Vector &A)
   {
      A = 0.0;
   };
   em_solver->setInitialCondition(*em_state, uinit);

   int rank;
   auto comm = MPI_COMM_WORLD;
   MPI_Comm_rank(comm, &rank);
   HypreParVector *u_true = em_state->GetTrueDofs();
   std::cout.precision(16);
   std::cout << "res norm: " << em_solver->calcResidualNorm(*em_state) << "\n";
   // std::cout << "u0 norm: " << std::sqrt(em_solver->calcInnerProduct(*em_state, *em_state)) << "\n";
   std::cout << "u0 norm: " << std::sqrt(InnerProduct(comm, *u_true, *u_true)) << "\n";

   em_solver->solveForState(*em_state);
   em_solver->printSolution("em_sol");
}
