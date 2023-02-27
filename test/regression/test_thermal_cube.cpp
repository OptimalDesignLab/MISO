#include <iostream>

#include "catch.hpp"
#include "nlohmann/json.hpp"
#include "mfem.hpp"

#include "mach_input.hpp"
#include "thermal.hpp"

///TODO: Once install mach again, replace the below lines with relative rather than absolute path
#include "../../src/physics/electromagnetics/magnetostatic.hpp"
// #include "../../src/physics/electromagnetics/electromag_outputs.hpp"

using namespace mach;
using namespace mfem;

/// Generate mesh - from test_magnetostatic_box
/// \param[in] nxy - number of nodes in the x and y directions
/// \param[in] nz - number of nodes in the z direction
std::unique_ptr<mfem::Mesh> buildMesh(int nxy,
                                int nz);

/*
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
            ///TODO: Remove once done debugging
            // std::cout << "intial_state=np.array([";
            // for (int j = 0; j < state.Size(); j++) {std::cout << state.Elem(j) << ", ";}
            // std::cout << "])\n";
            // std::cout << "thermal_load=np.array([";
            // for (int j = 0; j < load_tv.Size(); j++) {std::cout << load_tv.Elem(j) << ", ";}
            // std::cout << "])\n";
            
            solver.solveForState(inputs, state);

            ///TODO: Remove once done debugging
            // std::cout << "solved_state=np.array([";
            // for (int j = 0; j < state.Size(); j++) {std::cout << state.Elem(j) << ", ";}
            // std::cout << "])\n";

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
*/

// Adding new simple thermal test case to try to test the thermal load produced by losses
///TODO: Come back to after MMS Loss
TEST_CASE("ThermalSolver Square Box Regression Test - Thermal Load from Losses")
{
   std::cout << "TEST_CASE(\"ThermalSolver Box Regression Test - Thermal Load from Losses\")...............................\n";

   // ThermalSolver Box Regression Test as a starting point, adding in load
   auto acdcloss_options = R"(
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
         "printlevel": 2,
         "maxiter": 1,
         "reltol": 1e-10,
         "abstol": 1e-10
      },
      "components": {
         "windings": {
            "attrs": [1],
            "material": {
               "name": "copperwire",
               "conductivity":{
                  "model" : "linear",
                  "sigma_T_ref": 5.6497e7,
                  "T_ref": 293.15,
                  "alpha_resistivity": 3.8e-3
               }           
            }
         }
      },
      "current": {
         "phase" : {
            "z": [1]
         }
      },
      "bcs": {
         "convection": [1,3]
      }
   })"_json;
   auto coreloss_options = R"(
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
         "printlevel": 2,
         "maxiter": 1,
         "reltol": 1e-10,
         "abstol": 1e-10
      },
      "components": {
         "stator": {
            "attrs": [1],
            "material": {
               "name": "hiperco50",
               "core_loss" : {
                  "model": "CAL2",
                  "T0": 293.15,
                  "kh_T0": [5.97783049251564E-02, -6.58569751792524E-02, 3.52052785575931E-02, -6.54762513683037E-03],
                  "ke_T0": [3.83147202762929E-05, -4.19965038193089E-05, 2.09788988466414E-05, -3.88567697029196E-06],
                  "T1": 423.15,
                  "kh_T1": [5.78728253280150E-02, -7.94684973286488E-02, 5.09165213772802E-02, -1.11117379956941E-02],
                  "ke_T1": [3.20525407302126E-05, -1.43502199723297E-05, -3.74786590271071E-06, 2.68517704958978E-06]
               }
            }
         }
      },
      "bcs": {
         "convective": [1,3]
      },
      "UseCAL2forCoreLoss": true
   })"_json;

   // Set the options for the losses being considered
   auto options = acdcloss_options;
   // auto options = coreloss_options;

   ///TODO: Adjust the target errors once get a sense of how the previous test target errors were construed
   // define the target state solution error
   std::vector<std::vector<double>> target_error = {
      // nxy = 2, nxy = 4, nyx = 8, nyx = 16, nxy = 32
      {0.0,     0.0,     0.0,      0.0,      0.0}, // p = 1
      {0.0,     0.0,     0.0,      0.0,      0.0}, // p = 2
      {0.0,     0.0,     0.0,      0.0,      0.0}, // p = 3
      {0.0,     0.0,     0.0,      0.0,      0.0}  // p = 4
   };

   /// number of elements in Z direction (not needed for square)
   auto nz = 2;

   ///TODO: Change back to: (int order = 1; order <= 2; ++order)
   for (int order = 1; order <= 2; ++order)
   {
      options["space-dis"]["degree"] = order;
      int nxy = 1;
      ///TODO: Change back to: (int ref = 1; ref <= 2; ++ref)
      for (int ref = 1; ref <= 2; ++ref)
      {  
         nxy *= 2;
         DYNAMIC_SECTION("...for order " << order << " and mesh sizing nxy = " << nxy)
         {
            // construct the solver, set the initial condition, and solve
            // Using a 2D square mesh
            auto smesh = std::unique_ptr<mfem::Mesh>(
                  new mfem::Mesh(
                     // mfem::Mesh::MakeCartesian3D(
                     //    nxy, nxy, nz,
                     //    mfem::Element::TETRAHEDRON,
                     //    1.0, 1.0, (double)nz / (double)nxy, true)));
                     mfem::Mesh::MakeCartesian2D(
                        nxy, nxy, mfem::Element::TRIANGLE, false, 1.0, 1.0)));

            // Set up the thermal solver
            ThermalSolver solver(MPI_COMM_WORLD, options, std::move(smesh));
            mfem::Vector temperature_state_vector(solver.getStateSize()); // set the temperature state vector

            /// Set initial conditions (set the temperature field)
            solver.setState([](const mfem::Vector &x)
            {
               return 293.15; // prescibed temperature field
            }, temperature_state_vector);

            // Set up the magnetostatic solver. Will need it to compute the heat sources
            // Have to use buildMesh here because setting the third argument to smesh or std::move(smesh) resulted in build or run errors
            auto EMmesh = std::unique_ptr<mfem::Mesh>(
                  new mfem::Mesh(
                     // mfem::Mesh::MakeCartesian3D(
                     //    nxy, nxy, nz,
                     //    mfem::Element::TETRAHEDRON,
                     //    1.0, 1.0, (double)nz / (double)nxy, true)));
                     mfem::Mesh::MakeCartesian2D(
                        nxy, nxy, mfem::Element::TRIANGLE, false, 1.0, 1.0)));
            MagnetostaticSolver MagSolver(MPI_COMM_WORLD, options, std::move(EMmesh));
            
            ///TODO: Solve the EM problem first
            // auto &state = MagSolver.getState();
            // mfem::Vector mag_state_tv(MagSolver.getStateSize());
            // // MagSolver.setState(aexact, mag_state_tv);
            // MagSolver.solveForState(mag_state_tv);
            // ///TODO: Remove once done debugging
            // std::cout << "solved_mag_state=np.array([";
            // for (int j = 0; j < mag_state_tv.Size(); j++) {std::cout << mag_state_tv.Elem(j) << ", ";}
            // std::cout << "])\n";

            ///TODO: Resume here. Build up code, run, and try to get peak flux that can be used for AC (and pseudo-coupling)

            /// Assemble the load_tv
            mfem::Vector load_tv(MagSolver.getStateSize());
            MachInputs MagSolverMachInputs{
               {"temperature", temperature_state_vector},
               {"wire_length", 1.0},
               {"rms_current", 22.0}, //sqrt(1.0/sqrt(2.0))},
               {"strand_radius", 0.0007995}, //sqrt(1/M_PI)},
               {"strands_in_hand", 1.0},
               {"frequency", 1000.0},
               {"max_flux_magnitude:stator", 2.0}
            }; 
            // Call upon magnetostatic file to get the load_vec
            MagSolver.createOutput("heat_source", options);
            MagSolver.calcOutput("heat_source", MagSolverMachInputs, load_tv);
            MagSolver.createOutput("dc_loss", options);
            double DC_loss=MagSolver.calcOutput("dc_loss", MagSolverMachInputs);
            std::cout << "mach output DC_loss=" << DC_loss << "\n";

            ///TODO: Remove once done debugging
            std::cout << "load_tv.Size() = " << load_tv.Size() << "\n";
            std::cout << "load_tv.Min() = " << load_tv.Min() << "\n";
            std::cout << "load_tv.Max() = " << load_tv.Max() << "\n";
            std::cout << "load_tv.Sum() = " << load_tv.Sum() << "\n";

            // Set the inputs for the convection BC integrator and add in the thermal load coming from the EMHeatSourceOutput
            MachInputs inputs{
               {"h", 1.0},
               {"fluid_temp", 293.15},
               {"thermal_load", load_tv}
            };

            // // Now that losses are computed, set the state
            // solver.setState([](const mfem::Vector &p)
            // {
            //    auto x = p(0);
            //    auto y = p(1);
            //    // auto tol = 1e-10;
            //    // if (fabs(x - 1.0) < tol || fabs(y - 1.0) < tol|| fabs(x) < tol || fabs(y) < tol )
            //    // {
            //    //    return pow(x, 2);
            //    // }
            //    // return 0.0;
            //    return 0.5*6.03321*std::pow(x,2);
            // }, temperature_state_vector);

            ///TODO: Remove once done debugging
            std::cout << "intial_state=np.array([";
            for (int j = 0; j < temperature_state_vector.Size(); j++) {std::cout << temperature_state_vector.Elem(j) << ", ";}
            std::cout << "])\n";
            std::cout << "thermal_load=np.array([";
            for (int j = 0; j < load_tv.Size(); j++) {std::cout << load_tv.Elem(j) << ", ";}
            std::cout << "])\n";

            // Solve for the state (temperatures)
            solver.solveForState(inputs, temperature_state_vector);
            std::cout << "solved_state=np.array([";
            for (int j = 0; j < temperature_state_vector.Size(); j++) {std::cout << temperature_state_vector.Elem(j) << ", ";}
            std::cout << "])\n";

            // Compute the thermal output for the thermal flux (should be equal to 2*area in this case)
            MachInputs ThermalOutputMachInputs{
               {"state", temperature_state_vector}
            }; 
            std::vector<int> attrs = {1, 2, 3, 4};
            solver.createOutput("thermal_flux", {{"attributes", attrs}});
            double ThermalFlux = solver.calcOutput("thermal_flux", ThermalOutputMachInputs);
            std::cout << "From mach output, the ThermalFlux = " << ThermalFlux << "\n";


            ///TODO: Determine if this is right

            // /// Compute state error and check against target error
            // double error = solver.calcStateError([](const mfem::Vector &x)
            // {
            //    return 20.0;
            // }, state);

            // std::cout.precision(10);
            // std::cout << "error: " << error << "\n";
            ///TODO: Once content with test, bring back the assertion
            // REQUIRE(error == Approx(target_error[order-1][ref - 1]).margin(1e-10));
         }
      }
   }
}


/// NOTE: This test is good, just don't need the output currently
/*
TEST_CASE("MMS Loss -> Test BoundaryNormalIntegrator")
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
         "printlevel": 2,
         "maxiter": 1,
         "reltol": 1e-10,
         "abstol": 1e-10
      },
      "components": {
         "windings": {
            "attrs": [1],
            "material": {
               "name": "copperwire",
               "conductivity":{
                  "model" : "linear",
                  "sigma_T_ref": 5.6497e7,
                  "T_ref": 293.15,
                  "alpha_resistivity": 3.8e-3
               },
               "kappa": 1           
            }
         }
      },
      "current": {
         "phase" : {
            "z": [1]
         }
      },
      "bcs": {
         "essential": [1, 2, 3, 4]
      }
   })"_json;

   ///TODO: Adjust the target errors once get a sense of how the previous test target errors were construed
   // define the target state solution error
   std::vector<std::vector<double>> target_error = {
      // nxy = 2, nxy = 4, nyx = 8, nyx = 16, nxy = 32
      {0.0,     0.0,     0.0,      0.0,      0.0}, // p = 1
      {0.0,     0.0,     0.0,      0.0,      0.0}, // p = 2
      {0.0,     0.0,     0.0,      0.0,      0.0}, // p = 3
      {0.0,     0.0,     0.0,      0.0,      0.0}  // p = 4
   };

   for (int order = 1; order <= 2; ++order)
   {
      options["space-dis"]["degree"] = order;
      int nxy = 1;
      double x_length = 1.0;
      double y_length = 1.0;
      ///TODO: Change back to: (int ref = 1; ref <= 2; ++ref)
      for (int ref = 1; ref <= 2; ++ref)
      {  
         nxy *= 2;
         DYNAMIC_SECTION("...for order " << order << " and mesh sizing nxy = " << nxy)
         {
            // construct the solver, set the initial condition, and solve
            // Using a 2D square mesh
            auto smesh = std::unique_ptr<mfem::Mesh>(
                  new mfem::Mesh(
                     // mfem::Mesh::MakeCartesian3D(
                     //    nxy, nxy, nz,
                     //    mfem::Element::TETRAHEDRON,
                     //    1.0, 1.0, (double)nz / (double)nxy, true)));
                     mfem::Mesh::MakeCartesian2D(
                        nxy, nxy, mfem::Element::TRIANGLE, false, x_length, y_length)));

            // Set up the thermal solver
            ThermalSolver solver(MPI_COMM_WORLD, options, std::move(smesh));
            mfem::Vector init_temperature_field_for_losses(solver.getStateSize()); // the temperatures that will be used to compute the losses/heat sources
            mfem::Vector temperature_state_vector(solver.getStateSize()); // set the temperature state vector

            /// Set initial conditions (set the temperature field)
            // Set initial temperature distribution to allow for the pseudo DC loss forcing function to be 2 in conjunction with the manufactured mach inputs
            solver.setState([](const mfem::Vector &p)
            {
               // auto x = p(0);
               // auto y = p(1);
               // auto tol = 1e-10;
               // if (fabs(x - 1.0) < tol || fabs(y - 1.0) < tol|| fabs(x) < tol || fabs(y) < tol )
               // {
               //    return pow(x, 2);
               // }
               // Want sigma to be 1: sigma=sigma_T_ref/(1+alpha_T_ref*(T-T_ref))=1
               // Using root finder: T=14867404446.9177
               return 14867404446.9177;
            }, init_temperature_field_for_losses);

            // Set up the magnetostatic solver. Will need it to compute the heat sources
            // Have to use buildMesh here because setting the third argument to smesh or std::move(smesh) resulted in build or run errors
            auto EMmesh = std::unique_ptr<mfem::Mesh>(
                  new mfem::Mesh(
                     // mfem::Mesh::MakeCartesian3D(
                     //    nxy, nxy, nz,
                     //    mfem::Element::TETRAHEDRON,
                     //    1.0, 1.0, (double)nz / (double)nxy, true)));
                     mfem::Mesh::MakeCartesian2D(
                        nxy, nxy, mfem::Element::TRIANGLE, false, x_length, y_length)));
            MagnetostaticSolver MagSolver(MPI_COMM_WORLD, options, std::move(EMmesh));
            
            /// Assemble the load_tv
            ///TODO: Set Mach Inputs manufactured to produce a value of loss in DCLFDI of 2
            mfem::Vector load_tv(MagSolver.getStateSize());
            MachInputs MagSolverMachInputs{
               {"temperature", init_temperature_field_for_losses},
               {"wire_length", 2.0},
               {"rms_current", sqrt(1.0/sqrt(2.0))},
               {"strand_radius", sqrt(1/M_PI)},
               {"strands_in_hand", 1.0},
            }; 
            // Call upon magnetostatic file to get the load_vec
            MagSolver.createOutput("heat_source", options);
            MagSolver.calcOutput("heat_source", MagSolverMachInputs, load_tv);
            // MagSolver.createOutput("dc_loss", options);
            // double DC_loss=MagSolver.calcOutput("dc_loss", MagSolverMachInputs);
            // std::cout << "DC_loss=" << DC_loss << "\n";

            ///TODO: Remove once done debugging
            std::cout << "load_tv.Size() = " << load_tv.Size() << "\n";
            std::cout << "load_tv.Min() = " << load_tv.Min() << "\n";
            std::cout << "load_tv.Max() = " << load_tv.Max() << "\n";
            std::cout << "load_tv.Sum() = " << load_tv.Sum() << "\n";

            // Set the inputs for the thermal solver
            MachInputs inputs{
               {"thermal_load", load_tv}
            };

            // Set the temperature field back to the prescribed temperature distribution
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
               return pow(x, 2);
            }, temperature_state_vector);
            auto initial_state = temperature_state_vector; // save this for comparison and error determining (temperature_state_vector gets overwritten with solution)

            ///TODO: Remove once done debugging
            std::cout << "intial_state=np.array([";
            for (int j = 0; j < temperature_state_vector.Size(); j++) {std::cout << temperature_state_vector.Elem(j) << ", ";}
            std::cout << "])\n";
            std::cout << "thermal_load=np.array([";
            for (int j = 0; j < load_tv.Size(); j++) {std::cout << load_tv.Elem(j) << ", ";}
            std::cout << "])\n";

            // Use thermal solver to solve for the temperature distribution
            // It should be the prescibed temperature distribution of T=x(0)^2

            // Solve for the state (temperatures)
            solver.solveForState(inputs, temperature_state_vector);
            std::cout << "solved_state=np.array([";
            for (int j = 0; j < temperature_state_vector.Size(); j++) {std::cout << temperature_state_vector.Elem(j) << ", ";}
            std::cout << "])\n";

            // Check to ensure it lines up with prescibed temperature field
            // std::cout << "error (solved_state-initial_state)=np.array([";
            // for (int j = 0; j < temperature_state_vector.Size(); j++) {std::cout << temperature_state_vector.Elem(j)-initial_state.Elem(j) << ", ";}
            // std::cout << "])\n";

            // Compute the thermal output for the thermal flux (should be equal to 2*area in this case)
            MachInputs ThermalOutputMachInputs{
               {"state", temperature_state_vector}
            }; 
            std::vector<int> attrs = {1, 2, 3, 4};
            solver.createOutput("thermal_flux", {{"attributes", attrs}});
            double ThermalFlux = solver.calcOutput("thermal_flux", ThermalOutputMachInputs);
            std::cout << "From mach output, the ThermalFlux = " << ThermalFlux << "\n";

            /// Compute state error and check against target error
            // double error = solver.calcStateError([](const mfem::Vector &p)
            // {
            //    auto x = p(0);
            //    return pow(x, 2);
            // }, temperature_state_vector);

            // REQUIRE(ThermalFlux == Approx(2*x_length*y_length).margin(1e-4));
            // REQUIRE(error == Approx(target_error[order-1][ref - 1]).margin(1e-10));
         }
      }
   }
}
*/

// From test_magnetostatic_box
std::unique_ptr<Mesh> buildMesh(int nxy, int nz)
{
   // generate a simple tet mesh
   std::unique_ptr<mfem::Mesh> mesh(
      // new Mesh(mfem::Mesh::MakeCartesian3D(nxy, nxy, nz,
      //                                mfem::Element::TETRAHEDRON,
      //                                1.0, 1.0, (double)nz / (double)nxy, true)));
      new Mesh(Mesh::MakeCartesian2D(nxy, nxy,
                                     Element::TRIANGLE, true,
                                     1.0, 1.0, true)));

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
