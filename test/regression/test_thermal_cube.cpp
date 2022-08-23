#include <iostream>

#include "catch.hpp"
#include "nlohmann/json.hpp"
#include "mfem.hpp"

#include "mach_input.hpp"
#include "thermal.hpp"

using namespace mach;

TEST_CASE("ThermalSolver Box Regression Test")
{
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
            "attrs": [1],
            "material": {
               "name": "box1",
               "kappa": 1
            }
         }
      },
      "bcs": {
         "convection": [2, 3, 4, 5]
      }
   })"_json;

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
         }
      }
   }
}

TEST_CASE("ThermalSolver Box Regression Test with load")
{
   auto options = R"(
   {
      "space-dis": {
         "basis-type": "h1",
         "degree": 1
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
         "printlevel": 1,
         "maxiter": 5,
         "reltol": 1e-10,
         "abstol": 1e-10
      },
      "components": {
         "box": {
            "attrs": [1],
            "material": {
               "name": "box1",
               "kappa": 1
            }
         }
      },
      "bcs": {
         "essential": [3, 5]
      }
   })"_json;

   // define the target state solution error
   std::vector<std::vector<double>> target_error = {
      // nxy = 2, nxy = 4, nyx = 8, nyx = 16, nxy = 32
      {0.0080687153, 0.0014263608, 0.0002521474, 0.0, 0.0}, // p = 1
      {0.0, 0.0, 0.0, 0.0, 0.0}, // p = 2
   };

   /// number of elements in Z direction
   auto nz = 2;

   for (int order = 1; order <= 2; ++order)
   {
      options["space-dis"]["degree"] = order;
      int nxy = 2;
      for (int ref = 1; ref <= 3; ++ref)
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

            auto &fes = solver.getState().space();
            mfem::ParLinearForm load(&fes);
            mfem::FunctionCoefficient force([](const mfem::Vector &p)
            {
               return -2;
            });
            load.AddDomainIntegrator(new mfem::DomainLFIntegrator(force));
            load.Assemble();
            mfem::Vector load_tv(fes.GetTrueVSize());
            load.ParallelAssemble(load_tv);

            mfem::Array<int> ess_bdr(fes.GetParMesh()->bdr_attributes.Max());
            getMFEMBoundaryArray(options["bcs"]["essential"], ess_bdr);

            mfem::Array<int> ess_tdof_list;
            fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
            load_tv.SetSubVector(ess_tdof_list, 0.0);

            /// Set initial conditions
            solver.setState([](const mfem::Vector &p)
            {
               auto x = p(0);
               auto y = p(1);
               auto tol = 1e-10;
               if (fabs(x - 1.0) < tol || fabs(y - 1.0) < tol|| fabs(x) < tol || fabs(y) < tol )
               {
                  return pow(x, 2);
               }
               return 0.0;
            }, state);

            MachInputs inputs{
               {"thermal_load", load_tv}
            };
            solver.solveForState(inputs, state);

            /// Compute state error and check against target error
            double error = solver.calcStateError([](const mfem::Vector &p)
            {
               auto x = p(0);
               return pow(x, 2);
            }, state);

            std::cout.precision(10);
            std::cout << "error: " << error << "\n";
            REQUIRE(error == Approx(target_error[order-1][ref - 1]).margin(1e-10));
         }
      }
   }
}
