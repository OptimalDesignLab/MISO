#include "catch.hpp"
#include "mfem.hpp"
#include "thermal.hpp"

#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mach;

static double temp_0;

static double t_final;

static double InitialTemperature(const Vector &x);

static double ExactSolution(const Vector &x);

TEST_CASE("Thermal Cube Solver Regression Test", "[thermal]")
{
   const char *options_file = "test_thermal_cube_options.json";

   string opt_file_name(options_file);
   nlohmann::json options;
   nlohmann::json file_options;
   ifstream opts(opt_file_name);
   opts >> file_options;
   options.merge_patch(file_options);

   temp_0 = options["init-temp"].get<double>();
   t_final = options["time-dis"]["t-final"].get<double>();

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

         auto solver = createSolver<ThermalSolver>(opt_file_name, move(mesh));
         solver->setInitialCondition(InitialTemperature);
         solver->solveForState();
         solver->printSolution("thermal_final", 0);
         double l2_error = solver->calcL2Error(ExactSolution);

         double target;
         switch(h)
         {
            case 1: 
               target = 0.0548041517;
               break;
            case 2: 
               target = 0.0137142199;
               break;
            case 3: 
               target = 0.0060951886;
               break;
            case 4: 
               target = 0.0034275387;
               break;
         }
         REQUIRE(l2_error == Approx(target).margin(1e-10));
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