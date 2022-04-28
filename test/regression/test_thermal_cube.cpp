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
   /// number of elements in Z direction
   auto nz = 2;

   for (int order = 1; order <= 1; ++order)
   {
      options["space-dis"]["degree"] = order;
      int nxy = 1;
      for (int ref = 1; ref <= 1; ++ref)
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
            // REQUIRE(error == Approx(target_error[order-1][ref - 1]).margin(1e-10));

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
//    "mesh": {
//       "file": "initial.mesh",
//       "num-edge-x": 20,
//       "num-edge-y": 5,
//       "num-edge-z": 5
//    },
//    "space-dis": {
//       "basis-type": "H1",
//       "degree": 1,
//       "GD": false
//    },
//    "steady": false,
//    "time-dis": {
//       "ode-solver": "MIDPOINT",
//       "const-cfl": true,
//       "cfl": 1.0,
//       "dt": 0.01,
//       "t-final": 0.2
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
//    "adj-solver":{
//       "reltol": 1e-8,
//       "abstol": 0.0,
//       "printlevel": 0,
//       "maxiter": 500
//    },
//    "nonlin-solver":{
//       "printlevel": 0
//    },
//    "components": {
//       "stator": {
//          "material": "regtestmat1",
//          "attr": 1,
//          "max-temp": 0.5
//       },
//       "rotor": {
//          "material": "regtestmat1",
//          "attr": 2,
//          "max-temp": 0.5
//       }
//    },
//    "bcs": {
//       "outflux": [0, 0, 1, 0, 1, 0]
//    },
//    "outputs": {
//       "temp-agg": "temp-agg"
//    },
//    "problem-opts": {
//       "outflux-type": "test",
//       "rho-agg": 10,
//       "max-temp": 0.1,
//       "init-temp": 300,
//       "current_density": 1,
//       "frequency": 1500
//    }
// })"_json;


// static double temp_0;

// static double t_final;

// static double InitialTemperature(const Vector &x);

// static double ExactSolution(const Vector &x);

// TEST_CASE("Thermal Cube Solver Regression Test", "[thermal]")
// {
//    temp_0 = options["problem-opts"]["init-temp"].get<double>();
//    t_final = options["time-dis"]["t-final"].get<double>();
//    double target_error[4] {
//       0.0548041517, 0.0137142199, 0.0060951886, 0.0034275387
//    };

//    for (int h = 1; h <= 4; ++h)
//    {
//       DYNAMIC_SECTION("...for mesh sizing h = " << h)
//       {
//          // generate a simple tet mesh
//          int num_edge_x = 2*h;
//          int num_edge_y = 2;
//          int num_edge_z = 2;

//          std::unique_ptr<Mesh> mesh(
//             new Mesh(Mesh::MakeCartesian3D(num_edge_x, num_edge_y, num_edge_z,
//                                            Element::HEXAHEDRON,
//                                            1.0, 1.0, 1.0, true)));

//          std::cout << "Number of Boundary Attributes: "<< mesh->bdr_attributes.Size() <<std::endl;
//          // assign attributes to top and bottom sides
//          for (int i = 0; i < mesh->GetNE(); ++i)
//          {
//             Element *elem = mesh->GetElement(i);

//             Array<int> verts;
//             elem->GetVertices(verts);

//             bool below = true;
//             for (int i = 0; i < 4; ++i)
//             {
//                auto vtx = mesh->GetVertex(verts[i]);
//                if (vtx[0] <= 0.5)
//                {
//                   below = below & true;
//                }
//                else
//                {
//                   below = below & false;
//                }
//             }
//             if (below)
//             {
//                elem->SetAttribute(1);
//             }
//             else
//             {
//                elem->SetAttribute(2);
//             }
//          }
//          mesh->SetAttributes();

//          auto solver = createSolver<ThermalSolver>(options, move(mesh));
//          solver->setInitialCondition(InitialTemperature);
//          solver->solveForState();
//          solver->printSolution("thermal_final", 0);
//          double l2_error = solver->calcL2Error(ExactSolution);
//          REQUIRE(l2_error == Approx(target_error[h-1]).margin(1e-10));
//       }
//    }
// }

// double InitialTemperature(const Vector &x)
// {
//    return sin(M_PI*x(0)/2) - x(0)*x(0)/2;
// }

// double ExactSolution(const Vector &x)
// {
//    return sin(M_PI*x(0)/2)*exp(-M_PI*M_PI*t_final/4) - x(0)*x(0)/2 - 0.2;
// }