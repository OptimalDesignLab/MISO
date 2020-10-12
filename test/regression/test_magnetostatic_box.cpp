/// Solve the steady isentropic vortex problem on a quarter annulus
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
   "print-options": false,
   "problem": "box",
   "space-dis": {
      "basis-type": "nedelec",
      "degree": 1
   },
   "time-dis": {
      "steady": true,
      "steady-abstol": 1e-12,
      "steady-restol": 1e-10,
      "ode-solver": "PTC",
      "t-final": 100,
      "dt": 1e12
   },
   "lin-solver": {
      "type": "hypregmres",
      "printlevel": 1,
      "maxiter": 100,
      "abstol": 1e-12,
      "reltol": 1e-12
   },
   "lin-prec": {
      "type": "hypreams",
      "printlevel": 0
   },
   "nonlin-solver": {
      "type": "newton",
      "printlevel": 1,
      "maxiter": 50,
      "reltol": 1e-10,
      "abstol": 1e-12
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
      "current-density": 1.0,
      "current": {
         "box1": [1],
         "box2": [2]
      },
      "box": true
   },
   "outputs": {
      "co-energy": {}
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
   std::vector<double> target_error = {0.0690131081,
                                       0.0224304871, 
                                       0.0107753424, 
                                       0.0064387612};

   std::vector<double> target_coenergy = {-0.7355357753, 
                                          -0.717524391,
                                          -0.7152446356,
                                          -0.7146853447};


   /// number of elements in Z direction
   auto nz = 2;

   for (int nxy = 4; nxy <= 8; ++nxy)
   {
      DYNAMIC_SECTION("...for mesh sizing nxy = " << nxy)
      {
         // construct the solver, set the initial condition, and solve
         unique_ptr<Mesh> smesh = buildMesh(nxy, nz);
         auto solver = createSolver<MagnetostaticSolver>(options, move(smesh));
         solver->setInitialCondition(aexact);
         solver->solveForState();
         solver->printSolution("test_mag_box");
         auto fields = solver->getFields();


         // Compute error and check against appropriate target
         mfem::VectorFunctionCoefficient bEx(3, bexact);
         double l2_error = fields[1]->ComputeL2Error(bEx);
         std::cout << "l2 error in B: " << l2_error << "\n";
         REQUIRE(l2_error == Approx(target_error[nxy - 1]).margin(1e-10));

         // // Compute co-energy and check against target
         // double coenergy = solver->calcOutput("co-energy");
         // REQUIRE(coenergy == Approx(target_coenergy[nxy-1]).margin(1e-10));
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
