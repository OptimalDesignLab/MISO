#include <fstream>
#include <iostream>

#include "catch.hpp"
#include "json.hpp"
#include "mfem.hpp"

#include "magnetostatic.hpp"
#include "thermal.hpp"

using namespace mfem;
using namespace mach;

auto em_options = R"(
{
   "silent": false,
   "print-options": false,
   "mesh": {
      "file": "data/motor.smb",
      "model-file": "data/motor.egads"
   },
   "space-dis": {
      "basis-type": "nedelec",
      "degree": 1
   },
   "time-dis": {
      "steady": true,
      "steady-abstol": 0.0,
      "steady-reltol": 0.0,
      "ode-solver": "PTC",
      "t-final": 100,
      "dt": 1e14,
      "max-iter": 8
   },
   "lin-solver": {
      "type": "hypregmres",
      "printlevel": 2,
      "maxiter": 250,
      "abstol": 0.0,
      "reltol": 1e-8
   },
   "lin-prec": {
      "type": "hypreams",
      "printlevel": 0
   },
   "nonlin-solver": {
      "type": "relaxed-newton",
      "printlevel": 3,
      "maxiter": 5,
      "reltol": 1e-4,
      "abstol": 5e-1,
      "abort": false
   },
   "components": {
      "farfields": {
         "material": "air",
         "linear": true,
         "attrs": [1, 2]
      },
      "stator": {
         "attr": 3,
         "material": "hiperco50",
         "linear": false
      },
      "rotor": {
         "attr": 4,
         "material": "hiperco50",
         "linear": false
      },
      "airgap": {
         "attr": 5,
         "material": "air",
         "linear": true
      },
      "heatsink": {
         "attr": 6,
         "material": "2024-T3",
         "linear": true
      },
      "magnets": {
         "material": "Nd2Fe14B",
         "linear": true,
         "attrs": [7, 11, 15, 19, 23, 27, 31, 35, 39, 43,
                   8, 12, 16, 20, 24, 28, 32, 36, 40, 44,
                   9, 13, 17, 21, 25, 29, 33, 37, 41, 45,
                   10, 14, 18, 22, 26, 30, 34, 38, 42, 46]
      },
      "windings": {
         "material": "copperwire",
         "linear": true,
         "attrs": [47, 48, 49, 50,
                   51, 52, 53, 54,
                   55, 56, 57, 58,
                   59, 60, 61, 62,
                   63, 64, 65, 66,
                   67, 68, 69, 70,
                   71, 72, 73, 74,
                   75, 76, 77, 78,
                   79, 80, 81, 82,
                   83, 84, 85, 86,
                   87, 88, 89, 90,
                   91, 92, 93, 94,
                   95, 96, 97, 98,
                   99, 100, 101, 102,
                   103, 104, 105, 106,
                   107, 108, 109, 110,
                   111, 112, 113, 114,
                   115, 116, 117, 118,
                   119, 120, 121, 122,
                   123, 124, 125, 126,
                   127, 128, 129, 130,
                   131, 132, 133, 134,
                   135, 136, 137, 138, 139, 140,
                   141, 142, 143, 144, 145, 146]
      }
   },
   "problem-opts": {
      "fill-factor": 0.6,
      "current-density": 10.7e6,
      "current" : {
         "Phase-A": [47, 48, 49, 50,
                     51, 52, 53, 54,
                     67, 68, 69, 70,
                     75, 76, 77, 78,
                     91, 92, 93, 94,
                     95, 96, 97, 98,
                     111, 112, 113, 114,
                     119, 120, 121, 122],
         "Phase-B": [55, 56, 57, 58,
                     71, 72, 73, 74,
                     87, 88, 89, 90,
                     99, 100, 101, 102,
                     115, 116, 117, 118,
                     131, 132, 133, 134,
                     135, 136, 137, 138, 139, 140,
                     141, 142, 143, 144, 145, 146],
         "Phase-C": [59, 60, 61, 62,
                     63, 64, 65, 66,
                     79, 80, 81, 82,
                     83, 84, 85, 86,
                     103, 104, 105, 106,
                     107, 108, 109, 110,
                     123, 124, 125, 126,
                     127, 128, 129, 130]
      },
      "magnets": {
         "south": [7, 11, 15, 19, 23, 27, 31, 35, 39, 43],
         "cw": [8, 12, 16, 20, 24, 28, 32, 36, 40, 44],
         "north": [9, 13, 17, 21, 25, 29, 33, 37, 41, 45],
         "ccw": [10, 14, 18, 22, 26, 30, 34, 38, 42, 46]
      }
   },
   "bcs": {
      "essential": [1, 3]
   },
   "outputs": {
      "co-energy": {}
   },
   "external-fields": {
   }
})"_json;

auto thermal_options = R"(
{
   "print-options": false,
   "mesh": {
      "file": "data/motor.smb",
      "model-file": "data/motor.egads"
   },
   "space-dis": {
      "basis-type": "H1",
      "degree": 2
   },
   "time-dis": {
      "steady": true,
      "steady-abstol": 1e-8,
      "steady-reltol": 1e-8,
      "ode-solver": "PTC",
      "dt": 1e12,
      "max-iter": 5
   },
   "lin-prec": {
      "type": "hypreboomeramg"
   },
   "lin-solver": {
      "type": "hyprepcg",
      "reltol": 1e-10,
      "abstol": 1e-8,
      "printlevel": 2,
      "maxiter": 100
   },
   "nonlin-solver": {
      "type": "newton",
      "printlevel": 3,
      "maxiter": 10,
      "reltol": 1e-8,
      "abstol": 1e-8
   },
   "components": {
      "farfields": {
         "material": "air",
         "linear": true,
         "attrs": [1, 2]
      },
      "stator": {
         "attr": 3,
         "material": "hiperco50",
         "linear": false
      },
      "rotor": {
         "attr": 4,
         "material": "hiperco50",
         "linear": false
      },
      "airgap": {
         "attr": 5,
         "material": "air",
         "linear": true
      },
      "heatsink": {
         "attr": 6,
         "material": "2024-T3",
         "linear": true
      },
      "magnets": {
         "material": "Nd2Fe14B",
         "linear": true,
         "attrs": [7, 11, 15, 19, 23, 27, 31, 35, 39, 43,
                   8, 12, 16, 20, 24, 28, 32, 36, 40, 44,
                   9, 13, 17, 21, 25, 29, 33, 37, 41, 45,
                   10, 14, 18, 22, 26, 30, 34, 38, 42, 46]
      },
      "windings": {
         "material": "copperwire",
         "linear": true,
         "attrs": [47, 48, 49, 50,
                   51, 52, 53, 54,
                   55, 56, 57, 58,
                   59, 60, 61, 62,
                   63, 64, 65, 66,
                   67, 68, 69, 70,
                   71, 72, 73, 74,
                   75, 76, 77, 78,
                   79, 80, 81, 82,
                   83, 84, 85, 86,
                   87, 88, 89, 90,
                   91, 92, 93, 94,
                   95, 96, 97, 98,
                   99, 100, 101, 102,
                   103, 104, 105, 106,
                   107, 108, 109, 110,
                   111, 112, 113, 114,
                   115, 116, 117, 118,
                   119, 120, 121, 122,
                   123, 124, 125, 126,
                   127, 128, 129, 130,
                   131, 132, 133, 134,
                   135, 136, 137, 138, 139, 140,
                   141, 142, 143, 144, 145, 146]
      }
   },
   "problem-opts": {
      "keep-bndrys-adj-to": [1, 2],
      "rho-agg": 10,
      "init-temp": 300,
      "fill-factor": 0.6,
      "current-density": 10.7e6,
      "frequency": 1000,
      "current" : {
         "Phase-A": [47, 48, 49, 50,
                     51, 52, 53, 54,
                     67, 68, 69, 70,
                     75, 76, 77, 78,
                     91, 92, 93, 94,
                     95, 96, 97, 98,
                     111, 112, 113, 114,
                     119, 120, 121, 122],
         "Phase-B": [55, 56, 57, 58,
                     71, 72, 73, 74,
                     87, 88, 89, 90,
                     99, 100, 101, 102,
                     115, 116, 117, 118,
                     131, 132, 133, 134,
                     135, 136, 137, 138, 139, 140,
                     141, 142, 143, 144, 145, 146],
         "Phase-C": [59, 60, 61, 62,
                     63, 64, 65, 66,
                     79, 80, 81, 82,
                     83, 84, 85, 86,
                     103, 104, 105, 106,
                     107, 108, 109, 110,
                     123, 124, 125, 126,
                     127, 128, 129, 130]
      },
      "magnets": {
         "south": [7, 11, 15, 19, 23, 27, 31, 35, 39, 43],
         "cw": [8, 12, 16, 20, 24, 28, 32, 36, 40, 44],
         "north": [9, 13, 17, 21, 25, 29, 33, 37, 41, 45],
         "ccw": [10, 14, 18, 22, 26, 30, 34, 38, 42, 46]
      }
   },
   "bcs": {
      "essential": [1, 3, 27, 28, 38, 39, 76, 77, 91, 92]
   },
   "outflux-type": "test",
   "outputs": {
      "temp-agg": {}
   },
   "external-fields": {
      "mvp": {
         "basis-type": "nedelec",
         "degree": 1,
         "num-states": 1
      }
   }
})"_json;
      // "keep-bndrys": [19, 20, 42, 43, 72, 73, 87, 88],
      // "essential": [1, 3, 19, 20, 42, 43, 72, 73, 87, 88]
      // "essential": [1, 3, 27, 28, 38, 39, 76, 77, 91, 92] // model with heatsink
// ,
//    "external-fields": {
//       "mvp": {
//          "basis-type": "nedelec",
//          "degree": 1,
//          "num-states": 1
//       }
//    }

TEST_CASE("Coupled Motor Solve",
          "[SciTech-2021]")
{
   auto em_solver = createSolver<MagnetostaticSolver>(em_options);
   auto em_state = em_solver->getNewField();
   em_solver->setInitialCondition(*em_state,
                                  [](const mfem::Vector &x, mfem::Vector &A)
   {
      A = 0.0;
   });

   auto therm_solver = createSolver<ThermalSolver>(thermal_options);
   auto therm_state = therm_solver->getNewField();
   therm_solver->setInitialCondition(*therm_state,
                                     [](const mfem::Vector &x)
   {
      if (x.Norml2() < 0.06)
         return 351.25;
      else if (x.Norml2() < 0.1)
         // return 401.15;
         return 389.51;
      else
         return 303.15;
   });
   therm_solver->setResidualInput("mvp", *em_state);

   em_solver->solveForState(*em_state);
   therm_solver->solveForState(*therm_state);

}
