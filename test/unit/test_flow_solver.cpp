#include "catch.hpp"
#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "flow_solver.hpp"

TEST_CASE("Testing FlowSolver on the steady isentropic vortex")
{
   const bool verbose = true; // set to true for some output 
   std::ostream *out = verbose ? mach::getOutStream(0) : mach::getOutStream(1);
   using namespace mfem;
   using namespace mach;
  
   // Provide the options explicitly for regression tests
   auto options = R"(
   {
      "flow-param": {
         "entvar": false,
         "mach": 0.5
      },
      "space-dis": {
         "degree": 1,
         "lps-coeff": 1.0,
         "basis-type": "csbp",
         "flux-fun": "IR"
      },
      "time-dis": {
         "steady": true,
         "steady-abstol": 1e-12,
         "steady-restol": 1e-10,
         "ode-solver": "PTC",
         "t-final": 100,
         "dt": 1e12,
         "cfl": 1.0,
         "res-exp": 2.0
      },
      "bcs": {
         "vortex": [1, 1, 1, 0],
         "slip-wall": [0, 0, 0, 1]
      },
      "nonlin-solver": {
         "printlevel": 0,
         "maxiter": 50,
         "reltol": 1e-1,
         "abstol": 1e-12
      },
      "lin-solver": {
         "type": "hyprefgmres",
         "printlevel": 0,
         "filllevel": 3,
         "maxiter": 100,
         "reltol": 1e-2,
         "abstol": 1e-12
      },
      "saveresults": false,
      "outputs":
      { 
         "drag": [0, 0, 0, 1]
      }
   })"_json;

   for (int nx = 1; nx <= 4; ++nx)
   {
      DYNAMIC_SECTION("...for mesh sizing nx = " << nx)
      {
         // Build the vortex mesh
         int mesh_degree = options["space-dis"]["degree"].get<int>() + 1;
         auto mesh = buildQuarterAnnulusMesh(mesh_degree, nx, nx);

         // Create solver and solve for the state 
         FlowSolver solver(MPI_COMM_WORLD, options, std::move(mesh));
         MachInputs inputs;
         //solver.solveForState(inputs, state_tv);
         //auto &state = solver.getState();
      }
   }
}
