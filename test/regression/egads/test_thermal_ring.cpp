// #include <fstream>
// #include <iostream>
// #include <random>

// #include "catch.hpp"
// #include "nlohmann/json.hpp"
// #include "mfem.hpp"

// #include "thermal.hpp"

using namespace mfem;
using namespace miso;

// // Provide the options explicitly for regression tests
// auto current_options = R"(
// {
//    "print-options": false,
//    "mesh": {
//       "file": "data/ring_pumi.smb",
//       "model-file": "data/ring_pumi.egads"
//    },
//    "space-dis": {
//       "basis-type": "H1",
//       "degree": 1
//    },
//    "time-dis": {
//       "steady": true,
//       "steady-abstol": 1e-8,
//       "steady-reltol": 1e-8,
//       "ode-solver": "PTC",
//       "dt": 1e3,
//       "max-iter": 10
//    },
//    "lin-prec": {
//       "type": "hypreboomeramg"
//    },
//    "lin-solver": {
//       "type": "hypregmres",
//       "reltol": 1e-10,
//       "abstol": 1e-8,
//       "printlevel": 3,
//       "maxiter": 500
//    },
//    "nonlin-solver": {
//       "type": "newton",
//       "printlevel": 3,
//       "maxiter": 50,
//       "reltol": 1e-10,
//       "abstol": 1e-8
//    },
//    "components": {
//       "ring": {
//          "material": "copperwire",
//          "attr": 3
//       },
//       "sphere": {
//          "material": "air",
//          "attrs": [1, 2]
//       }
//    },
//    "bcs": {
//       "essential": [1, 3, 4, 5]
//    },
//    "problem-opts": {
//       "keep-bndrys": [4, 5],
//       "fill-factor": 1.0,
//       "current_density": 100000,
//       "current": {
//          "z": [3] 
//       },
//       "rho-agg": 10,
//       "init-temp": 100.0
//    }
// })"_json;

//    // "bcs": {
//    //    "essential": [1, 3]
//    // },
//       // "convection-coeff": 30,
//       // "keep-bndrys": [4, 5],
//       // "convection": [3, 4, 5]

// // define the random-number generator; uniform between 0 and 1
// static std::default_random_engine gen(std::random_device{}());
// static std::uniform_real_distribution<double> uniform_rand(0.0, 1.0);

// double randBaselinePert(const Vector &x)
// {
//     const double scale = 0.01;
//     return 1.0 + scale * uniform_rand(gen);
// }

// double randState(const Vector &x)
// {
//     return 2.0 * uniform_rand(gen) - 1.0;
// }

// TEST_CASE("Thermal Solver Convection Ring Regression Test",
//           "[Thermal Steady Convection Ring]")
// {
//    auto temp0 = current_options["problem-opts"]["init-temp"].get<double>();
//    double target_error[] {
//       0.0133079447, 0.0
//    };

//    for (int order = 1; order <= 1; ++order)
//    {
//       current_options["space-dis"]["degree"] = order;
 
//       DYNAMIC_SECTION("...for order " << order)
//       {
//          // construct the solver, set the initial condition, and solve
//          auto solver = createSolver<ThermalSolver>(current_options, nullptr);
//          auto state = solver->getNewField();

//          solver->setFieldValue(*state, temp0);

//          // FunctionCoefficient rand_s(randState);
//          // state->ProjectCoefficient(rand_s);

//          solver->solveForState(*state);
//          // solver->checkJacobian(*state, randBaselinePert);

//          // double l2_error = solver->calcL2Error(state.get(), u0);
//          // std::cout << "l2_error: " << l2_error << "\n";
//          // REQUIRE(l2_error == Approx(target_error[order-1]).margin(1e-10));
//       }
//    }
// }
