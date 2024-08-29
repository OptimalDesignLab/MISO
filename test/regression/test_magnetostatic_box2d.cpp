#include <fstream>
#include <iostream>

#include "catch.hpp"
#include "nlohmann/json.hpp"
#include "mfem.hpp"

#include "magnetostatic.hpp"

using namespace std;
using namespace mfem;
using namespace miso;

// Provide the options explicitly for regression tests
auto options = R"(
{
   "silent": false,
   "print-options": false,
   "problem": "box",
   "space-dis": {
      "basis-type": "h1",
      "degree": 1,
      "sipg-penalty": 1e10
   },
   "time-dis": {
      "steady": true,
      "steady-abstol": 1e-10,
      "steady-reltol": 1e-10,
      "ode-solver": "PTC",
      "t-final": 100,
      "dt": 1,
      "max-iter": 5
   },
   "lin-solver": {
      "type": "gmres",
      "printlevel": 1,
      "maxiter": 100,
      "abstol": 1e-14,
      "reltol": 1e-14
   },
   "lin-prec": {
      "type": "hypreboomeramg",
      "printlevel": 0
   },
   "nonlin-solver": {
      "type": "newton",
      "printlevel": 1,
      "maxiter": 1,
      "reltol": 1e-10,
      "abstol": 1e-9
   },
   "components": {
      "box1": {
         "attrs": [1],
         "material": {
            "name": "box1",
            "mu_r": 795774.7154594767
         }
      },
      "box2": {
         "attrs": [2],
         "material": {
            "name": "box2",
            "mu_r": 795774.7154594767
         }
      }
   },
   "current": {
      "box": {
         "box1": [1],
         "box2": [2]
      }
   },
   "bcs": {
      "weak-essential": [1, 2, 3, 4]
   }
})"_json;

            // "mu_r": 795774.7154594767

/// \brief Exact solution for magnetic vector potential
/// \param[in] x - coordinate of the point at which the state is needed
///return z component of magnetic vector potential
double aexact(const Vector &x);

/// Generate mesh 
/// \param[in] nxy - number of nodes in the x and y directions
std::unique_ptr<Mesh> buildMesh(int nxy);

TEST_CASE("Magnetostatic Box Solver Regression Test",
          "[Magnetostatic-Box]")
{
   // define the target state solution error
   std::vector<std::vector<double>> target_error = {
      //     nxy = 2, nxy = 4, nyx = 8, nyx = 16, nxy = 32
      {0.0306325207,     0.0,     0.0,      0.0,      0.0}, // p = 1
      {0.004603664996,    0.0,     0.0,      0.0,      0.0}, // p = 2
      {0.0,               0.0,     0.0,      0.0,      0.0}, // p = 3
      {0.0,               0.0,     0.0,      0.0,      0.0}  // p = 4
   };

   // // define the target computed energy
   // std::vector<std::vector<double>> target_energy = {
   //    {0.0456124231, 0.0, 0.0, 0.0},
   //    {0.05807012599, 0.0, 0.0, 0.0},
   //    {0.05629189119, 0.0, 0.0, 0.0},
   //    {0.05625, 0.0, 0.0, 0.0}
   // };

   for (int order = 1; order <= 4; ++order)
   {
      options["space-dis"]["degree"] = order;
      int nxy = 1;
      for (int ref = 1; ref <= 1; ++ref)
      {  
         nxy *= 2;
         DYNAMIC_SECTION("...for order " << order << " and mesh sizing nxy = " << nxy)
         {
            // construct the solver, set the initial condition, and solve
            unique_ptr<Mesh> smesh = buildMesh(nxy);

            MagnetostaticSolver solver(MPI_COMM_WORLD, options, std::move(smesh));
            mfem::Vector state_tv(solver.getStateSize());

            auto &state = solver.getState();

            /// Set initial/boundary conditions
            solver.setState(aexact, state_tv);

            /// Log initial condition
            ParaViewLogger logger_init("2d_magnetostatic_init", &state.mesh());
            logger_init.registerField("h1_state", state.gridFunc());
            logger_init.saveState(state_tv, "h1_state", 0, 0.0, 0);

            solver.solveForState(state_tv);

            // state.distributeSharedDofs(state_tv);
            ParaViewLogger logger("2d_magnetostatic", &state.mesh());
            logger.registerField("h1_state", state.gridFunc());
            logger.saveState(state_tv, "h1_state", 0, 0.0, 0);

            /// Compute state error and check against target error
            double error = solver.calcStateError(aexact, state_tv);
            std::cout.precision(10);
            std::cout << "error: " << error << "\n";
            REQUIRE(error == Approx(target_error[order-1][ref - 1]).margin(1e-10));

            // /// Calculate the magnetic energy and check against target energy
            // solver.createOutput("energy");
            // MachInputs inputs{{"state", state_tv}};
            // double energy = solver.calcOutput("energy", inputs);
            // std::cout << "energy: " << energy << "\n";
            // // REQUIRE(energy == Approx(target_energy[order-1][ref - 1]).margin(1e-10));

            // solver.solveForState({{"current_density:box", 2.0}}, state_tv);
            // solver.solveForState({{"current_density:box", 3.0}}, state_tv);
            // solver.solveForState({{"current_density:box", 4.0}}, state_tv);
         }
      }
   }
}

// TEST_CASE("2D Magnetostatic Solver MMS solution error wrt mesh size")
// {
//    const bool verbose = true; // set to true for some output 
//    std::ostream *out = verbose ? miso::getOutStream(0) : miso::getOutStream(1);
//    using namespace mfem;
//    using namespace mach;

//    std::vector<int> orders = {1, 2, 3, 4};
//    std::vector<int> mesh_nxy = {2, 4, 8, 16, 32, 64};

//    std::vector<std::vector<double>> sol_error(orders.size());
//    for (int i = 0; i < orders.size(); ++i)
//    {
//       sol_error[i].resize(mesh_nxy.size());
//    }

//    for (const auto &p : orders)
//    {
//       for (int i = 0; i < mesh_nxy.size(); ++i)
//       {
//          auto options = R"(
//          {
//             "print-options": false,
//             "space-dis": {
//                "basis-type": "H1",
//                "degree": 1,
//                "sipg-penalty": 100.0
//             },
//             "lin-solver": {
//                "type": "gmres",
//                "reltol": 1e-12,
//                "abstol": 0.0,
//                "printlevel": -1,
//                "maxiter": 500,
//                "kdim": 500
//             },
//             "nonlin-solver": {
//                "type": "relaxednewton",
//                "linesearch": {
//                   "type": "backtracking",
//                   "maxiter": 3
//                },
//                "maxiter": 1,
//                "printlevel": 1
//             },
//             "components": {
//                "box1": {
//                   "attrs": [1],
//                   "material": {
//                      "name": "box1",
//                      "mu_r": 795774.7154594767
//                   }
//                },
//                "box2": {
//                   "attrs": [2],
//                   "material": {
//                      "name": "box2",
//                      "mu_r": 795774.7154594767
//                   }
//                }
//             },
//             "current": {
//                "box": {
//                   "box1": [1],
//                   "box2": [2]
//                }
//             },
//             "bcs": {
//                "weak-essential": [1, 2, 3, 4]
//             }
//          })"_json;

//          options["space-dis"]["degree"] = p;
//          options["space-dis"]["sipg-penalty"] = pow(p+1, 2);

//          int nxy = mesh_nxy[i];
//          unique_ptr<Mesh> smesh = buildMesh(nxy);

//          // Create solver and solve for the state 
//          MagnetostaticSolver solver(MPI_COMM_WORLD, options, std::move(smesh));
//          mfem::Vector state_tv(solver.getStateSize());

//          solver.setState(aexact, state_tv);

//          solver.solveForState(state_tv);

//          // Check that solution is accurate
//          auto error = solver.calcStateError(aexact, state_tv);

//          sol_error[p-1][i] = error;
//       }
//    }

//    for (const auto &p_err : sol_error)
//    {
//       for (const auto &err : p_err)
//       {
//          std::cout << err << ", ";
//       }
//       std::cout << "\n";
//    }

//    for (const auto &nxy : mesh_nxy)
//    {
//       std::cout << nxy << ", ";
//    }
//    std::cout << "\n";

// }

// TEST_CASE("2D Magnetostatic Solver MMS residual")
// {
//    const bool verbose = true; // set to true for some output 
//    std::ostream *out = verbose ? miso::getOutStream(0) : miso::getOutStream(1);
//    using namespace mfem;
//    using namespace mach;

//    auto options = R"(
//    {
//       "print-options": false,
//       "space-dis": {
//          "basis-type": "H1",
//          "degree": 1,
//          "sipg-penalty": 100.0
//       },
//       "lin-solver": {
//          "type": "gmres",
//          "reltol": 1e-12,
//          "abstol": 0.0,
//          "printlevel": -1,
//          "maxiter": 500,
//          "kdim": 500
//       },
//       "nonlin-solver": {
//          "type": "relaxednewton",
//          "linesearch": {
//             "type": "backtracking",
//             "maxiter": 3
//          },
//          "maxiter": 1,
//          "printlevel": 1
//       },
//       "components": {
//          "box1": {
//             "attrs": [1],
//             "material": {
//                "name": "box1",
//                "mu_r": 795774.7154594767
//             }
//          },
//          "box2": {
//             "attrs": [2],
//             "material": {
//                "name": "box2",
//                "mu_r": 795774.7154594767
//             }
//          }
//       },
//       "current": {
//          "box": {
//             "box1": [1],
//             "box2": [2]
//          }
//       },
//       "bcs": {
//          "weak-essential": [1, 2, 3, 4]
//       }
//    })"_json;

//    options["space-dis"]["degree"] = 3;
//    options["space-dis"]["sipg-penalty"] = 1e2;

//    int nxy = 2;
//    unique_ptr<Mesh> smesh = buildMesh(nxy);

//    // Create solver and solve for the state 
//    MagnetostaticSolver solver(MPI_COMM_WORLD, options, std::move(smesh));
//    mfem::Vector state_tv(solver.getStateSize());
//    solver.setState(aexact, state_tv);

//    auto error = solver.calcStateError(aexact, state_tv);
//    std::cout << "exact sol error: " << error << "\n";

//    mfem::Vector residual_vec(solver.getStateSize());
//    solver.calcResidual(state_tv, residual_vec);

//    auto res_norm = solver.calcResidualNorm(state_tv);
//    std::cout << "exact sol res_norm: " << res_norm << "\n";

//    auto &state = solver.getState();
//    auto &res = solver.getResVec();
//    res.distributeSharedDofs(residual_vec);

//    miso::ParaViewLogger paraview("magnetostatic_box2d", &res.mesh());
//    paraview.registerField("state", state.gridFunc());
//    paraview.registerField(
//        "residual",
//        dynamic_cast<mfem::ParGridFunction &>(res.localVec()));

//    paraview.saveState(state_tv, "state", 0, 0, 0);
//    paraview.saveState(residual_vec, "residual", 0, 0, 0);

//    solver.solveForState(state_tv);
//    solver.calcResidual(state_tv, residual_vec);

//    paraview.saveState(state_tv, "state", 1, 1, 0);
//    paraview.saveState(residual_vec, "residual", 1, 1, 0);

//    error = solver.calcStateError(aexact, state_tv);
//    std::cout << "exact sol error: " << error << "\n";

//    res_norm = solver.calcResidualNorm(state_tv);
//    std::cout << "exact sol res_norm: " << res_norm << "\n";

// }

double aexact(const Vector &x)
{
   double y = x(1) - 0.5;
   if ( y <= 0.0)
   {
      return y*y*y;
   }
   else 
   {
      return -y*y*y;
   }
}

unique_ptr<Mesh> buildMesh(int nxy)
{
   // generate a simple tet mesh
   std::unique_ptr<Mesh> mesh(
      new Mesh(Mesh::MakeCartesian2D(nxy, nxy,
                                     Element::TRIANGLE)));

   // assign attributes to top and bottom sides
   for (int i = 0; i < mesh->GetNE(); ++i)
   {
      Element *elem = mesh->GetElement(i);

      Array<int> verts;
      elem->GetVertices(verts);

      bool below = true;
      for (int i = 0; i < verts.Size(); ++i)
      {
         auto *vtx = mesh->GetVertex(verts[i]);
         // std::cout << "mesh vtx: " << vtx[0] << ", " << vtx[1] << "\n";
         if (vtx[1] <= 0.5)
         {
            below = below;
         }
         else
         {
            below = false;
         }
      }
      if (below)
      {
         elem->SetAttribute(1);
      }
      else
      {
         elem->SetAttribute(2);
      }
   }
   mesh->SetAttributes();

   return mesh;
}
