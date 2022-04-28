#include <iostream>

#include "catch.hpp"
#include "nlohmann/json.hpp"
#include "mfem.hpp"

#include "mach_input.hpp"
#include "thermal.hpp"

// Provide the options explicitly for regression tests
auto options = R"(
{
   "space-dis": {
      "basis-type": "h1",
      "degree": 3
   },
   "lin-solver": {
      "type": "pcg",
      "printlevel": 1,
      "maxiter": 100,
      "abstol": 1e-14,
      "reltol": 1e-14
   },
   "lin-prec": {
      "printlevel": 0
   },
   "nonlin-solver": {
      "type": "newton",
      "printlevel": 2,
      "maxiter": 1,
      "reltol": 1e-10,
      "abstol": 1e-10
   },
   "components": {
      "box": {
         "material": "box1",
         "attr": 1
      }
   },
   "bcs": {
      "convection": [2, 3, 4, 5]
   }
})"_json;

using namespace mach;

TEST_CASE("ThermalSolver Box Regression Test")
{
   // define the target state solution error
   std::vector<std::vector<double>> target_error = {
      // nxy = 2, nxy = 4, nyx = 8, nyx = 16, nxy = 32
      {0.0,     0.0,     0.0,      0.0,      0.0}, // p = 1
      {0.0,     0.0,     0.0,      0.0,      0.0}, // p = 2
      {0.0,     0.0,     0.0,      0.0,      0.0}, // p = 3
      {0.0,     0.0,     0.0,      0.0,      0.0}  // p = 4
   };

   /// number of elements in Z direction
   auto nz = 2;

   for (int order = 1; order <= 2; ++order)
   {
      options["space-dis"]["degree"] = order;
      int nxy = 1;
      for (int ref = 1; ref <= 2; ++ref)
      {  
         nxy *= 2;
         DYNAMIC_SECTION("...for order " << order << " and mesh sizing nxy = " << nxy)
         {
            // construct the solver, set the initial condition, and solve
            auto smesh = std::unique_ptr<mfem::Mesh>(
                  new mfem::Mesh(
                     mfem::Mesh::MakeCartesian3D(
                        nxy, nxy, nz,
                        mfem::Element::TETRAHEDRON,
                        1.0, 1.0, (double)nz / (double)nxy, true)));
                     // mfem::Mesh::MakeCartesian2D(
                     //    nxy, nxy, mfem::Element::TRIANGLE)));

            ThermalSolver solver(MPI_COMM_WORLD, options, std::move(smesh));
            mfem::Vector state(solver.getStateSize());

            /// Set initial conditions
            solver.setState([](const mfem::Vector &x)
            {
               return sin(x(0));
            }, state);

            MachInputs inputs{
               {"h", 1.0},
               {"fluid_temp", 1.0}
            };
            solver.solveForState(inputs, state);

            /// Compute state error and check against target error
            double error = solver.calcStateError([](const mfem::Vector &x)
            {
               return 1.0;
            }, state);

            std::cout.precision(10);
            std::cout << "error: " << error << "\n";
            REQUIRE(error == Approx(target_error[order-1][ref - 1]).margin(1e-10));

            // /// Calculate the magnetic energy and check against target energy
            // solver.createOutput("energy");
            // MachInputs inputs{{"state", state}};
            // double energy = solver.calcOutput("energy", inputs);
            // std::cout << "energy: " << energy << "\n";
            // REQUIRE(energy == Approx(target_energy[order-1][ref - 1]).margin(1e-10));
         }
      }
   }
}
