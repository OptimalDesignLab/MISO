#include <fstream>
#include <iostream>

#include "catch.hpp"
#include "json.hpp"
#include "mfem.hpp"

#include "magnetostatic.hpp"
#include "thermal.hpp"

using namespace std;
using namespace mfem;
using namespace mach;

// Provide the options explicitly for regression tests
auto em_options = R"(
{
   "silent": false,
   "print-options": false,
   "problem": "box",
   "space-dis": {
      "basis-type": "nedelec",
      "degree": 1
   },
   "time-dis": {
      "steady": true,
      "steady-abstol": 1e-10,
      "steady-reltol": 1e-10,
      "ode-solver": "PTC",
      "t-final": 100,
      "dt": 1,
      "max-iter": 10
   },
   "lin-solver": {
      "type": "hypregmres",
      "printlevel": 0,
      "maxiter": 100,
      "abstol": 1e-14,
      "reltol": 1e-14
   },
   "lin-prec": {
      "type": "hypreams",
      "printlevel": 0
   },
   "nonlin-solver": {
      "type": "newton",
      "printlevel": 3,
      "maxiter": 5,
      "reltol": 1e-10,
      "abstol": 1e-9
   },
   "components": {
      "attr1": {
         "material": "box1",
         "attr": 1,
         "linear": true
      },
      "attr2": {
         "material": "box2",
         "attr": 2,
         "linear": true
      }
   },
   "problem-opts": {
      "fill-factor": 1.0,
      "current_density": 1.0,
      "current": {
         "box1": [1],
         "box2": [2]
      }
   },
   "outputs": {
      "co-energy": {}
   },
   "bcs": {
      "essential": [1, 2, 3, 4, 5, 6]
   }
})"_json;

auto therm_options = R"(
{
   "print-options": false,
   "space-dis": {
      "basis-type": "H1",
      "degree": 1
   },
   "steady": false,
   "time-dis": {
      "ode-solver": "MIDPOINT",
      "const-cfl": true,
      "cfl": 1.0,
      "dt": 0.01,
      "t-final": 1.0
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
   "nonlin-solver":{
      "printlevel": 0
   },
   "motor-opts" : {
      "current": 1,
      "frequency": 1500
   },
   "components": {
      "attr1": {
         "material": "box1",
         "attr": 1,
         "linear": true,
         "max-temp": 0.5
      },
      "attr2": {
         "material": "box2",
         "attr": 2,
         "linear": true,
         "max-temp": 0.5
      }
   },
   "problem-opts": {
      "outflux-type": "test",
      "fill-factor": 1.0,
      "current_density": 1.0,
      "current": {
         "box1": [1],
         "box2": [2]
      },
      "rho-agg": 10,
      "max-temp": 0.1,
      "init-temp": 300
   },
   "bcs": {
      "outflux": [0, 0, 1, 0, 1, 0]
   },
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

double temp_0;

double t_final;

/// \brief Exact solution for magnetic vector potential
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] A - magnetic vector potential
void aexact(const Vector &x, Vector& A);

/// \brief Exact solution for magnetic flux density
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] B - magnetic flux density
void bexact(const Vector &x, Vector& B);

double initialTemp(const Vector &x);

double exactSolution(const Vector &x);

/// Generate mesh 
/// \param[in] nxy - number of nodes in the x and y directions
/// \param[in] nz - number of nodes in the z direction
std::unique_ptr<Mesh> buildMesh(int nxy,
                                int nz);

TEST_CASE("Joule Box Solver Regression Test",
          "[Joule-Box]")
{
   temp_0 = therm_options["problem-opts"]["init-temp"].get<double>();
   t_final = therm_options["time-dis"]["t-final"].get<double>();
   double target_error[4] {
      0.0548041517, 0.0137142199, 0.0060951886, 0.0034275387
   };

   /// number of elements in Z direction
   auto nz = 2;

   for (int order = 1; order <= 1; ++order)
   {
      em_options["space-dis"]["degree"] = order;
      therm_options["space-dis"]["degree"] = order;
      int nxy = 1;
      for (int ref = 1; ref <= 1; ++ref)
      {  
         nxy *= 4;
         DYNAMIC_SECTION("...for order " << order
                         << " and mesh sizing nxy = " << nxy)
         {
            // construct the solver, set the initial condition, and solve
            unique_ptr<Mesh> em_mesh = buildMesh(nxy, nz);
            unique_ptr<Mesh> therm_mesh = buildMesh(nxy, nz);


            auto em_solver = createSolver<MagnetostaticSolver>(em_options, 
                                                               move(em_mesh));
            auto em_state = em_solver->getNewField();
            em_solver->setFieldValue(*em_state, aexact);
            MachInputs em_inputs;
            em_solver->solveForState(em_inputs, *em_state);

            auto therm_solver = createSolver<ThermalSolver>(therm_options, 
                                                            move(therm_mesh));
            auto therm_state = therm_solver->getNewField();
            // therm_solver->setResidualInput("mvp", *em_state);
            therm_solver->setFieldValue(*therm_state, initialTemp);

            MachInputs therm_inputs = {
               {"mvp", em_state->GetData()}
            };
            therm_solver->solveForState(therm_inputs, *therm_state);

            // solver->printSolution("thermal_final", 0);
            // double l2_error = solver->calcL2Error(exactSolution);
            // REQUIRE(l2_error == Approx(target_error[h-1]).margin(1e-10));
         }
      }
   }
}

void aexact(const Vector &x, Vector& A)
{
   A.SetSize(3);
   A = 0.0;
   double y = x(1) - .5;
   if ( x(1) <= .5)
   {
      A(2) = y*y*y; 
      // A(2) = y*y; 
   }
   else 
   {
      A(2) = -y*y*y;
      // A(2) = -y*y;
   }
}

void bexact(const Vector &x, Vector& B)
{
   B.SetSize(3);
   B = 0.0;
   double y = x(1) - .5;
   if ( x(1) <= .5)
   {
      B(0) = 3*y*y; 
      // B(0) = 2*y; 
   }
   else 
   {
      B(0) = -3*y*y;
      // B(0) = -2*y;
   }	
}

double initialTemp(const Vector &x)
{
   return sin(M_PI*x(0)/2) - x(0)*x(0)/2;
}

double exactSolution(const Vector &x)
{
   return sin(M_PI*x(0)/2)*exp(-M_PI*M_PI*t_final/4) - x(0)*x(0)/2 - 0.2;
}

unique_ptr<Mesh> buildMesh(int nxy, int nz)
{
   // generate a simple tet mesh
   std::unique_ptr<Mesh> mesh(new Mesh(nxy, nxy, nz,
                              Element::TETRAHEDRON, true /* gen. edges */, 1.0,
                              1.0, (double)nz / (double)nxy, true));

   mesh->ReorientTetMesh();

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
         if (vtx[1] <= 0.5)
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
   return mesh;
}