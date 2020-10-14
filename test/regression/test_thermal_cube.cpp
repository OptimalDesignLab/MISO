#include "catch.hpp"
#include "mfem.hpp"
#include "thermal.hpp"

#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mach;

// Provide the options explicitly for regression tests
auto options = R"(
{
   "print-options": false,
    "mesh": {
       "file": "initial.mesh",
       "num-edge-x": 20,
       "num-edge-y": 5,
       "num-edge-z": 5
    },
    "space-dis": {
       "basis-type": "H1",
       "degree": 1,
       "GD": false
    },
    "steady": false,
    "time-dis": {
        "ode-solver": "MIDPOINT",
        "const-cfl": true,
        "cfl": 1.0,
        "dt": 0.01,
        "t-final": 0.2
    },
    "lin-solver": {
       "reltol": 1e-8,
       "abstol": 0.0,
       "printlevel": 0,
       "maxiter": 500
    },
    "adj-solver":{
       "reltol": 1e-8,
       "abstol": 0.0,
       "printlevel": 0,
       "maxiter": 500
    },
    "nonlin-solver":{
       "printlevel": 0
    },
    "motor-opts" : {
       "current": 1,
       "frequency": 1500
    },
    "components": {
       "stator": {
          "material": "regtestmat1",
          "attr": 1,
          "max-temp": 0.5
       },
       "rotor": {
          "material": "regtestmat1",
          "attr": 2,
          "max-temp": 0.5
       }
    },
    "bcs": {
        "outflux": [0, 0, 1, 0, 1, 0]
    },
    "outflux-type": "test",
    "outputs": {
        "temp-agg": "temp-agg"
    },
    "rho-agg": 10,
    "max-temp": 0.1,
    "init-temp": 300,
    "material-lib-path": "../../src/material_options.json"
})"_json;


static double temp_0;

static double t_final;

static double InitialTemperature(const Vector &x);

static double ExactSolution(const Vector &x);

TEST_CASE("Thermal Cube Solver Regression Test", "[thermal]")
{
   temp_0 = options["init-temp"].get<double>();
   t_final = options["time-dis"]["t-final"].get<double>();
   double target_error[4] {
      0.0548041517, 0.0137142199, 0.0060951886, 0.0034275387
   };

   for (int h = 1; h <= 4; ++h)
   {
      DYNAMIC_SECTION("...for mesh sizing h = " << h)
      {
         // generate a simple tet mesh
         int num_edge_x = 2*h;
         int num_edge_y = 2;
         int num_edge_z = 2;

         std::unique_ptr<Mesh> mesh(new Mesh(num_edge_x, num_edge_y, num_edge_z,
                                    Element::HEXAHEDRON, true /* gen. edges */, 1.0,
                                    1.0, 1.0, true));

         mesh->ReorientTetMesh();
         std::cout << "Number of Boundary Attributes: "<< mesh->bdr_attributes.Size() <<std::endl;
         // assign attributes to top and bottom sides
         for (int i = 0; i < mesh->GetNE(); ++i)
         {
            Element *elem = mesh->GetElement(i);

            Array<int> verts;
            elem->GetVertices(verts);

            bool below = true;
            for (int i = 0; i < 4; ++i)
            {
               auto vtx = mesh->GetVertex(verts[i]);
               if (vtx[0] <= 0.5)
               {
                  below = below & true;
               }
               else
               {
                  below = below & false;
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

         auto solver = createSolver<ThermalSolver>(options, move(mesh));
         solver->setInitialCondition(InitialTemperature);
         solver->solveForState();
         solver->printSolution("thermal_final", 0);
         double l2_error = solver->calcL2Error(ExactSolution);
         REQUIRE(l2_error == Approx(target_error[h-1]).margin(1e-10));
      }
   }
}

double InitialTemperature(const Vector &x)
{
   return sin(M_PI*x(0)/2) - x(0)*x(0)/2;
}

double ExactSolution(const Vector &x)
{
   return sin(M_PI*x(0)/2)*exp(-M_PI*M_PI*t_final/4) - x(0)*x(0)/2 - 0.2;
}