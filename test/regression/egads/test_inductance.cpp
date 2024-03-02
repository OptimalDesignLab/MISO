#include <fstream>
#include <iostream>

#include "catch.hpp"
#include "nlohmann/json.hpp"
#include "mfem.hpp"

#include "magnetostatic.hpp"

TEST_CASE("MagnetostaticSolver Flux Linkage Regression Test")
{

auto options = R"(
{
   "paraview": {
      "log": true,
      "fields": ["state", "flux_magnitude"],
      "directory": "toroid"
   },
   "silent": false,
   "print-options": false,
   "mesh": {
      "file": "data/toroidal_inductor/toroidal_inductor.smb",
      "model-file": "data/toroidal_inductor/toroidal_inductor.egads"
   },
   "space-dis": {
      "degree": 1,
      "basis-type": "h1"
   },
   "lin-solver": {
      "type": "gmres",
      "printlevel": 1,
      "maxiter": 500,
      "kdim": 500,
      "abstol": 1e-10,
      "reltol": 1e-10
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
      "abstol": 1e-10
   },
   "components": {
      "core": {
         "attrs": [1],
         "material": {
            "name": "core",
            "reluctivity": {
               "model": "linear",
               "mu_r": 1000
            }
         }
      },
      "windings": {
         "attrs": [2, 3],
         "material": {
            "name": "copper"
         }
      },
      "air": {
         "attrs": [4, 5],
         "material": "air"
      }
   },
   "current": {
      "inductor": {
         "z": [2],
         "-z": [3]
      }
   },
   "bcs": {
      "weak-essential": [9, 10]
   }
})"_json;

for (int order = 1; order <= 1; ++order)
{
   // options["space-dis"]["degree"] = order;
   for (int ref = 1; ref <= 1; ++ref)
   {  
      DYNAMIC_SECTION("...for order " << order)
      {
         // construct the solver, set the initial condition, and solve
         mach::MagnetostaticSolver solver(MPI_COMM_WORLD, options);

         solver.createOutput("flux_linkage:inductor_z", {{"attributes", {2}}});
         solver.createOutput("flux_linkage:inductor_minus_z", {{"attributes", {3}}});
         solver.createOutput("flux_magnitude");

         mfem::Vector state_tv(solver.getStateSize());
         state_tv = 0.0;
         mfem::Vector flux_tv(solver.getFieldSize("flux_magnitude"));
         flux_tv = 0.0;

         auto current = 5;
         int n_turns = 300;
         auto strand_radius = 0.005;
         auto strand_area = M_PI * pow(strand_radius, 2);
         auto true_winding_area = 0.0942478;
         // auto true_winding_area = 0.000310518; //0.0942478;
         auto winding_area = strand_area * n_turns;
         auto fill_factor = winding_area / true_winding_area;

         auto current_density = current / strand_area * fill_factor;
         std::cout << "current_density: " << current_density << "\n";
         std::cout << "fill_factor: " << fill_factor << "\n";
         mach::MachInputs inputs {
            {"state", state_tv},
            {"current_density:inductor", current_density},
         };
         solver.solveForState(inputs, state_tv);

         // auto flux_linkage_plus = solver.calcOutput("flux_linkage:inductor_z", inputs);
         auto flux_linkage_minus = solver.calcOutput("flux_linkage:inductor_minus_z", inputs);
         auto flux_linkage_plus = 0.0;
         // auto flux_linkage_minus = 0.0;

         solver.calcOutput("flux_magnitude", inputs, flux_tv);
         solver.solveForState(inputs, state_tv);


         std::cout << "flux_linkage_plus: " << flux_linkage_plus << "\n";
         std::cout << "flux_linkage_minus: " << flux_linkage_minus << "\n";

         auto total_flux_linkage = flux_linkage_plus + flux_linkage_minus;
         auto inductance = n_turns * total_flux_linkage / current;

         std::cout << "total_flux_linkage: " << total_flux_linkage << "\n";
         std::cout << "current: " << current << "\n";
         std::cout << "inductance: " << inductance << "\n";

         auto core_avg_radius = (0.2 + 0.3) / 2;
         auto flux = 1000 * 4 * M_PI * 1e-7 * n_turns * current / (2 * M_PI * core_avg_radius);
         std::cout << "flux: " << flux << "\n";

         auto annulus_area = M_PI * (pow(0.3, 2) - pow(0.2, 2));
         inductance = 1000 * 4 * M_PI * 1e-7 * pow(n_turns, 2) * (0.1) / (2*M_PI*0.2);
         std::cout << "inductance: " << inductance << "\n";
      }
   }
}

}

TEST_CASE("MagnetostaticSolver Flux Linkage Regression Test 2")
{

auto options = R"(
{
   "paraview": {
      "log": true,
      "fields": ["state", "flux_magnitude"],
      "directory": "rect"
   },
   "silent": false,
   "print-options": false,
   "mesh": {
      "file": "data/rect_inductor/rect_inductor.smb",
      "model-file": "data/rect_inductor/rect_inductor.egads"
   },
   "space-dis": {
      "degree": 2,
      "basis-type": "h1"
   },
   "lin-solver": {
      "type": "gmres",
      "printlevel": 1,
      "maxiter": 500,
      "kdim": 500,
      "abstol": 1e-10,
      "reltol": 1e-10
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
      "abstol": 1e-10
   },
   "components": {
      "core": {
         "attrs": [3],
         "material": {
            "name": "core",
            "reluctivity": {
               "model": "linear",
               "mu_r": 1000
            }
         }
      },
      "windings": {
         "attrs": [2, 4],
         "material": {
            "name": "copper"
         }
      },
      "air": {
         "attrs": [1],
         "material": "air"
      }
   },
   "current": {
      "inductor": {
         "-z": [2],
         "z": [4]
      }
   },
   "bcs": {
      "weak-essential": [1, 2]
   }
})"_json;

for (int order = 1; order <= 1; ++order)
{
   // options["space-dis"]["degree"] = order;
   for (int ref = 1; ref <= 1; ++ref)
   {  
      DYNAMIC_SECTION("...for order " << order)
      {
         // construct the solver, set the initial condition, and solve
         mach::MagnetostaticSolver solver(MPI_COMM_WORLD, options);

         solver.createOutput("flux_linkage:inductor_z", {{"attributes", {2}}});
         solver.createOutput("flux_linkage:inductor_minus_z", {{"attributes", {4}}});
         solver.createOutput("flux_magnitude");

         mfem::Vector state_tv(solver.getStateSize());
         state_tv = 0.0;
         mfem::Vector flux_tv(solver.getFieldSize("flux_magnitude"));
         flux_tv = 0.0;

         auto current = 100;
         int n_turns = 1000;
         auto strand_radius = 0.005;
         auto strand_area = M_PI * pow(strand_radius, 2);
         auto true_winding_area = 0.1;
         auto winding_area = strand_area * n_turns;
         auto fill_factor = winding_area / true_winding_area;

         auto current_density = current / strand_area * fill_factor;
         std::cout << "current_density: " << current_density << "\n";
         std::cout << "fill_factor: " << fill_factor << "\n";
         mach::MachInputs inputs {
            {"state", state_tv},
            {"current_density:inductor", current_density},
         };
         solver.solveForState(inputs, state_tv);

         auto flux_linkage_plus = solver.calcOutput("flux_linkage:inductor_z", inputs);
         auto flux_linkage_minus = solver.calcOutput("flux_linkage:inductor_minus_z", inputs);


         solver.calcOutput("flux_magnitude", inputs, flux_tv);
         solver.solveForState(inputs, state_tv);


         std::cout << "flux_linkage_plus: " << flux_linkage_plus << "\n";
         std::cout << "flux_linkage_minus: " << flux_linkage_minus << "\n";

         auto total_flux_linkage = flux_linkage_plus + flux_linkage_minus;
         auto inductance = total_flux_linkage / current;

         std::cout << "total_flux_linkage: " << total_flux_linkage << "\n";
         std::cout << "current: " << current << "\n";
         std::cout << "inductance: " << inductance << "\n";

         auto flux = 1000 * 4 * M_PI * 1e-7 * n_turns * current / (1);

         auto core_area = 0.2;
         inductance = 4 * M_PI * 1e-7 * pow(n_turns, 2) * core_area;
         std::cout << "inductance: " << inductance << "\n";
      }
   }
}

}

TEST_CASE("MagnetostaticSolver EI Inductor Core Regression Test")
{

auto options = R"(
{
   "paraview": {
      "log": true,
      "fields": ["state", "flux_magnitude"],
      "directory": "ei"
   },
   "silent": false,
   "print-options": false,
   "mesh": {
      "file": "data/ei_inductor/ei_inductor.smb",
      "model-file": "data/ei_inductor/ei_inductor.egads"
   },
   "space-dis": {
      "degree": 2,
      "basis-type": "h1"
   },
   "lin-solver": {
      "type": "gmres",
      "printlevel": 1,
      "maxiter": 500,
      "kdim": 500,
      "abstol": 1e-10,
      "reltol": 1e-10
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
      "abstol": 1e-10
   },
   "components": {
      "core": {
         "attrs": [2, 4],
         "material": {
            "name": "core",
            "reluctivity": {
               "model": "linear",
               "mu_r": 2500
            }
         }
      },
      "windings": {
         "attrs": [3, 5],
         "material": {
            "name": "copper"
         }
      },
      "air": {
         "attrs": [1],
         "material": "air"
      }
   },
   "current": {
      "inductor": {
         "z": [3],
         "-z": [5]
      }
   },
   "bcs": {
      "weak-essential": [1, 2, 3, 4]
   }
})"_json;

// construct the solver, set the initial condition, and solve
mach::MagnetostaticSolver solver(MPI_COMM_WORLD, options);

solver.createOutput("flux_linkage:inductor_z", {{"attributes", {3}}});
solver.createOutput("flux_linkage:inductor_minus_z", {{"attributes", {5}}});
solver.createOutput("flux_magnitude");
// solver.createOutput("energy", {{"attributes", {2, 3, 4, 5}}});
solver.createOutput("energy");

mfem::Vector state_tv(solver.getStateSize());
state_tv = 0.0;
mfem::Vector flux_tv(solver.getFieldSize("flux_magnitude"));
flux_tv = 0.0;

auto current = -1.0;
int n_turns = 66;
// auto strand_radius = 0.001024 / 2;
auto strand_area = M_PI * pow(0.001024/2, 2);
auto winding_area = 0.00635 * 0.0127;
// auto winding_area = strand_area * n_turns;
// auto fill_factor = winding_area / true_winding_area;
auto fill_factor = n_turns * strand_area / winding_area;
// auto fill_factor = 0.66;

auto winding_area_in2 = 0.25 * 0.5;
auto strand_area_in2 = M_PI * pow(0.0403/2, 2);
auto fill_factor_in = n_turns * strand_area_in2 / winding_area_in2;

auto current_density = (current / strand_area) * fill_factor;
std::cout << "current_density: " << current_density << "\n";
std::cout << "fill_factor: " << fill_factor << "\n";
std::cout << "fill_factor_in: " << fill_factor_in << "\n";
mach::MachInputs inputs {
   {"state", state_tv},
   {"current_density:inductor", current_density},
};
solver.solveForState(inputs, state_tv);

auto flux_linkage_plus = solver.calcOutput("flux_linkage:inductor_z", inputs);
auto flux_linkage_minus = solver.calcOutput("flux_linkage:inductor_minus_z", inputs);

solver.calcOutput("flux_magnitude", inputs, flux_tv);
solver.solveForState(inputs, state_tv);

std::cout << "flux_linkage_plus: " << flux_linkage_plus << "\n";
std::cout << "flux_linkage_minus: " << flux_linkage_minus << "\n";

const auto inductor_depth = 0.0254;

auto total_flux_linkage = flux_linkage_plus + flux_linkage_minus;
auto inductance = n_turns * inductor_depth * total_flux_linkage / current;

std::cout << "total_flux_linkage: " << total_flux_linkage << "\n";
std::cout << "current: " << current << "\n";
std::cout << "inductance: " << inductance << "\n";

auto energy = solver.calcOutput("energy", inputs) * inductor_depth;
std::cout << "energy: " << energy << "\n";

}
