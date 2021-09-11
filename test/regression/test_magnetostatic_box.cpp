#include <fstream>
#include <iostream>

#include "catch.hpp"
#include "json.hpp"
#include "mfem.hpp"

#include "magnetostatic.hpp"

using namespace std;
using namespace mfem;
using namespace mach;

// Provide the options explicitly for regression tests
auto options = R"(
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
      "max-iter": 5
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
      "maxiter": 15,
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

/// \brief Exact solution for magnetic vector potential
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] A - magnetic vector potential
void aexact(const Vector &x, Vector& A);

/// \brief Exact solution for magnetic flux density
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] B - magnetic flux density
void bexact(const Vector &x, Vector& B);

/// Generate mesh 
/// \param[in] nxy - number of nodes in the x and y directions
/// \param[in] nz - number of nodes in the z direction
std::unique_ptr<Mesh> buildMesh(int nxy,
                                int nz);

TEST_CASE("Magnetostatic Box Solver Regression Test",
          "[Magnetostatic-Box]")
{
   // define the appropriate exact solution error
   std::vector<std::vector<double>> target_error = {
      // nxy = 2,    nxy = 4,      nyx = 8,      nyx = 16,     nxy = 32
      {0.1909560005, 0.0714431496, 0.0254173455, 0.0089731459, 0.0031667063}, // p = 1
      {0.0503432775, 0.0103611278, 0.0022325345, 0.0005048188, 0.0001186963}, // p = 2
      {0.0056063669, 0.0006908125, 8.56636e-05,  1.0664e-05,   1.33025e-06},  // p = 3
      {0.0,          0.0,          0.0,          0.0,          0.0}           // p = 4
   };

   /// TODO:
   std::vector<std::vector<double>> target_coenergy = {
      {0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0}
   };

   /// number of elements in Z direction
   auto nz = 2;

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
            unique_ptr<Mesh> smesh = buildMesh(nxy, nz);
            auto solver = createSolver<MagnetostaticSolver>(options, move(smesh));

            auto state = solver->getNewField();
            solver->setFieldValue(*state, aexact);
            MachInputs inputs {
               {"state", state->GetData()}
            };
            solver->solveForState(inputs, *state);

            auto fields = solver->getFields();

            // Compute error and check against appropriate target
            mfem::VectorFunctionCoefficient bEx(3, bexact);
            double l2_error = fields[1]->ComputeL2Error(bEx);
            std::cout << "\n\nl2 error in B: " << l2_error << "\n\n\n";
            REQUIRE(l2_error == Approx(target_error[order-1][ref - 1]).margin(1e-10));

            // // Compute co-energy and check against target
            // double coenergy = solver->calcOutput("co-energy");
            // REQUIRE(coenergy == Approx(target_coenergy[nxy-1]).margin(1e-10));
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

unique_ptr<Mesh> buildMesh(int nxy, int nz)
{
   // generate a simple tet mesh
   std::unique_ptr<Mesh> mesh(new Mesh(nxy, nxy, nz,
                              Element::TETRAHEDRON, true /* gen. edges */, 1.0,
                              1.0, (double)nz / (double)nxy, true));

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
