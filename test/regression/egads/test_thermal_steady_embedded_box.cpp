#include <fstream>
#include <iostream>

#include "catch.hpp"
#include "json.hpp"
#include "mfem.hpp"

#include "magnetostatic.hpp"
#include "thermal.hpp"

using namespace mfem;
using namespace mach;

// Provide the options explicitly for regression tests
auto current_options = R"(
{
   "print-options": false,
   "mesh": {
      "file": "data/box_pumi.smb",
      "model-file": "data/box_pumi.egads"
   },
   "space-dis": {
      "basis-type": "H1",
      "degree": 1
   },
   "time-dis": {
      "steady": true,
      "steady-abstol": 1e-8,
      "steady-reltol": 1e-8,
      "ode-solver": "PTC",
      "dt": 1e3,
      "max-iter": 5
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
      "abstol": 1e-10
   },
   "components": {
      "box": {
         "material": "copperwire",
         "attr": 1
      },
      "sphere": {
         "material": "copperwire",
         "attrs": [2, 3]
      }
   },
   "bcs": {
      "essential": [1, 3, 6, 7, 10, 11, 12, 13]
   },
   "problem-opts": {
      "keep-bndrys": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
      "fill-factor": 1.0,
      "current-density": 1.2732395447351627e7,
      "frequency": 0,
      "current": {
         "z": [1]
      },
      "rho-agg": 10,
      "init-temp": 0.0
   }
})"_json;

      // "keep-bndrys-adj-to": [2, 3],

/// NOTE: This test doesn't actually solve anything right now, since the
/// initial condition is the exact solution the residual is zero and the newton
/// solver doesn't run.
TEST_CASE("Thermal Solver Current Steady Regression Test",
          "[Thermal Steady Embedded Box]")
{
   auto temp0 = current_options["problem-opts"]["init-temp"].get<double>();
   double target_error[] {
      0.0133079447, 0.0
   };

   /// set correct current density source
   auto kappa = 4.01e2;
   auto sigma = 58.14e6;
   auto current_density = std::sqrt(4*kappa*sigma);
   current_options["problem-opts"]["current-density"] = current_density;

   for (int order = 1; order <= 2; ++order)
   {
      current_options["space-dis"]["degree"] = order;
 
      DYNAMIC_SECTION("...for order " << order)
      {
         // construct the solver, set the initial condition, and solve
         auto solver = createSolver<ThermalSolver>(current_options, nullptr);
         auto state = solver->getNewField();

         auto u0 = [temp0](const Vector &x)
         {
            return -x(0)*x(0) - x(1)*x(1) + temp0;
         };

         solver->setFieldValue(*state, u0);

         solver->solveForState(*state);
         double l2_error = solver->calcL2Error(*state, u0);
         std::cout << "l2_error: " << l2_error << "\n";
         REQUIRE(l2_error == Approx(target_error[order-1]).margin(1e-10));
      }
   }
}

// Provide the options explicitly for regression tests
auto mag_options = R"(
{
   "print-options": false,
   "mesh": {
      "file": "data/box_pumi.smb",
      "model-file": "data/box_pumi.egads"
   },
   "space-dis": {
      "basis-type": "H1",
      "degree": 1
   },
   "time-dis": {
      "steady": true,
      "steady-abstol": 1e-8,
      "steady-reltol": 1e-8,
      "ode-solver": "PTC",
      "dt": 1e3,
      "max-iter": 5
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
      "abstol": 1e-10
   },
   "components": {
      "box": {
         "material": "hiperco50",
         "attr": 1
      },
      "sphere": {
         "material": "hiperco50",
         "attrs": [2, 3]
      }
   },
   "bcs": {
      "essential": [1, 3, 6, 7, 10, 11, 12, 13]
   },
   "problem-opts": {
      "keep-bndrys-adj-to": [2, 3],
      "fill-factor": 1.0,
      "current-density": 0.0,
      "frequency": 91.64428855781742,
      "init-temp": 0.0
   },
   "external-fields": {
      "mvp": {
         "basis-type": "nedelec",
         "degree": 1,
         "num-states": 1
      }
   }
})"_json;

// Provide the options explicitly for regression tests
auto em_opts = R"(
{
   "print-options": false,
   "mesh": {
      "file": "data/box_pumi.smb",
      "model-file": "data/box_pumi.egads"
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
      "dt": 1e3,
      "max-iter": 5
   },
   "lin-prec": {
      "type": "hypreams"
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
      "abstol": 1e-10
   },
   "components": {
      "box": {
         "material": "hiperco50",
         "attr": 1,
         "linear": false
      },
      "sphere": {
         "material": "hiperco50",
         "attrs": [2, 3],
         "linear": false
      }
   },
   "bcs": {
      "essential": [1, 3]
   },
   "problem-opts": {
      "fill-factor": 1.0,
      "current-density": 0.0,
      "frequency": 91.64428855781742,
      "init-temp": 0.0
   }
})"_json;

/// NOTE: This test doesn't actually solve anything right now, since the
/// initial condition is the exact solution the residual is zero and the newton
/// solver doesn't run.
TEST_CASE("Thermal Solver Mag-Field Steady Regression Test",
          "[Thermal Steady Embedded Box]")
{
        
   auto temp0 = mag_options["problem-opts"]["init-temp"].get<double>();
   double target_error[] {
      0.0133079447, 0.0
   };

   mag_options["problem-opts"]["current-density"] = 0.0;

   auto em_solver = createSolver<MagnetostaticSolver>(em_opts, nullptr);
   auto em_state = em_solver->getNewField();

   auto initState = [](const mfem::Vector &x, mfem::Vector &A)
   {
      A(0) = -0.5*x(1);
      A(1) = 0.5*x(0);
      A(2) = 0.0;
   };
   // VectorFunctionCoefficient internalState(3, initState);
   // em_state->ProjectCoefficient(internalState);
   em_solver->setFieldValue(*em_state, initState);

   for (int order = 1; order <= 1; ++order)
   {
      mag_options["space-dis"]["degree"] = order;
 
      DYNAMIC_SECTION("...for order " << order)
      {
         // construct the solver, set the initial condition, and solve
         auto solver = createSolver<ThermalSolver>(mag_options, nullptr);
         auto state = solver->getNewField();

         auto u0 = [temp0](const Vector &x)
         {
            return -x(0)*x(0) - x(1)*x(1) + temp0;
         };

         solver->setFieldValue(*state, u0);
         // solver->setResidualInput("mvp", *em_state);
         MachInputs inputs = {
            {"mvp", *em_state->GetData()}
         };
         solver->solveForState(inputs, *state);
         double l2_error = solver->calcL2Error(*state, u0);
         std::cout << "l2_error: " << l2_error << "\n";
         REQUIRE(l2_error == Approx(target_error[order-1]).margin(1e-10));
      }
   }
}
