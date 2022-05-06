// #include "catch.hpp"
// #include "mfem.hpp"
// #include "thermal.hpp"

// #include <fstream>
// #include <iostream>

// using namespace std;
// using namespace mfem;
// using namespace mach;

// // Provide the options explicitly for regression tests
// auto options = R"(
// {
//    "print-options": false,
//    "space-dis": {
//       "basis-type": "H1",
//       "degree": 1
//    },
//    "time-dis": {
//       "steady": true,
//       "steady-abstol": 1e-8,
//       "steady-reltol": 1e-8,
//       "ode-solver": "PTC",
//       "dt": 1e12,
//       "max-iter": 5
//    },
//    "lin-prec": {
//       "type": "hypreboomeramg"
//    },
//    "lin-solver": {
//       "reltol": 1e-14,
//       "abstol": 0.0,
//       "printlevel": 0,
//       "maxiter": 500
//    },
//    "nonlin-solver": {
//       "type": "newton",
//       "printlevel": 3,
//       "maxiter": 50,
//       "reltol": 1e-10,
//       "abstol": 1e-10
//    },
//    "components": {
//       "box": {
//          "material": "copperwire",
//          "attr": 1
//       }
//    },
//    "bcs": {
//       "essential": [2, 3, 4, 5]
//    },
//    "problem-opts": {
//       "fill-factor": 1.0,
//       "current_density": 1.2732395447351627e7,
//       "frequency": 0,
//       "current": {
//          "z": [1]
//       },
//       "rho-agg": 10,
//       "init-temp": 0.0
//    }
// })"_json;

// /// Generate mesh 
// /// \param[in] nxy - number of nodes in the x and y directions
// /// \param[in] nz - number of nodes in the z direction
// std::unique_ptr<Mesh> buildMesh(int nxy,
//                                 int nz);

// /// NOTE: This test doesn't actually solve anything right now, since the
// /// initial condition is the exact solution the residual is zero and the newton
// /// solver doesn't run.
// TEST_CASE("Thermal Cube Solver Steady Regression Test",
//           "[Steady Thermal Box]")
// {
//    auto temp0 = options["problem-opts"]["init-temp"].get<double>();
//    double target_error[] {
//       0.0841109552, 0.0148688567, 0.0026284673, 0.0004646518
//    };

//    /// set correct current density source
//    auto kappa = 2.49;
//    auto sigma = 58.14e6;
//    auto current_density = std::sqrt(4*kappa*sigma);
//    options["problem-opts"]["current_density"] = current_density;

//    /// number of elements in Z direction
//    auto nz = 2;

//    for (int order = 1; order <= 1; ++order)
//    {
//       options["space-dis"]["degree"] = order;
//       int nxy = 1;
//       for (int ref = 1; ref <= 4; ++ref)
//       {  
//          nxy *= 2;
//          DYNAMIC_SECTION("...for order " << order << " and mesh sizing nxy = " << nxy)
//          {
//             // construct the solver, set the initial condition, and solve
//             unique_ptr<Mesh> smesh = buildMesh(nxy, nz);
//             auto solver = createSolver<ThermalSolver>(options, move(smesh));
//             auto state = solver->getNewField();

//             auto u0 = [temp0](const Vector &x)
//             {
//                return -x(0)*x(0) - x(1)*x(1) + temp0;
//             };

//             solver->setFieldValue(*state, u0);

//             MachInputs inputs;
//             solver->solveForState(inputs, *state);
//             double l2_error = solver->calcL2Error(*state, u0);
//             std::cout << "l2_error: " << l2_error << "\n";
//             REQUIRE(l2_error == Approx(target_error[ref-1]).margin(1e-10));
//          }
//       }
//    }
// }

// unique_ptr<Mesh> buildMesh(int nxy, int nz)
// {
//    // generate a simple tet mesh
//    std::unique_ptr<Mesh> mesh(
//       new Mesh(Mesh::MakeCartesian3D(nxy, nxy, nz,
//                                      Element::TETRAHEDRON,
//                                      1.0, 1.0, (double)nz / (double)nxy, true)));
//    return mesh;
// }
