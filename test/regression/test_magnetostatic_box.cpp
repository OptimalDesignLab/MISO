#include <fstream>
#include <iostream>

#include "catch.hpp"
#include "nlohmann/json.hpp"
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
      "type": "minres",
      "printlevel": 1,
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
      "printlevel": 1,
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
   "current": {
      "box": {
         "box1": [1],
         "box2": [2]
      }
   },
   "bcs": {
      "essential": "all"
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
   // define the target state solution error
   std::vector<std::vector<double>> target_error = {
      //     nxy = 2, nxy = 4, nyx = 8, nyx = 16, nxy = 32
      {0.04538234599,     0.0,     0.0,      0.0,      0.0}, // p = 1
      {0.01203088004,     0.0,     0.0,      0.0,      0.0}, // p = 2
      {0.00257648892,     0.0,     0.0,      0.0,      0.0}, // p = 3
      {0.0,               0.0,     0.0,      0.0,      0.0}  // p = 4
   };

   // define the target computed energy
   std::vector<std::vector<double>> target_energy = {
      {0.0456124231, 0.0, 0.0, 0.0},
      {0.05807012599, 0.0, 0.0, 0.0},
      {0.05629189119, 0.0, 0.0, 0.0},
      {0.05625, 0.0, 0.0, 0.0}
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
            MagnetostaticSolver solver(MPI_COMM_WORLD, options, std::move(smesh));
            auto &state = solver.getState();
            mfem::Vector state_tv(solver.getStateSize());

            /// Set initial/boundary conditions
            solver.setState(aexact, state_tv);

            /// Log initial condition
            ParaViewLogger logger_init("2d_magnetostatic_initND", &state.mesh());
            logger_init.registerField("state", state.gridFunc());
            logger_init.saveState(state_tv, "state", 0, 0.0, 0);

            solver.solveForState(state_tv);

            ParaViewLogger logger("2d_magnetostaticND", &state.mesh());
            logger.registerField("state", state.gridFunc());
            logger.saveState(state_tv, "state", 0, 0.0, 0);

            /// Compute state error and check against target error
            double error = solver.calcStateError(aexact, state_tv);
            std::cout.precision(10);
            std::cout << "error: " << error << "\n";
            REQUIRE(error == Approx(target_error[order-1][ref - 1]).margin(1e-10));

            /// Calculate the magnetic energy and check against target energy
            solver.createOutput("energy");
            MachInputs inputs{{"state", state_tv}};
            double energy = solver.calcOutput("energy", inputs);
            std::cout << "energy: " << energy << "\n";
            REQUIRE(energy == Approx(target_energy[order-1][ref - 1]).margin(1e-10));

            // solver.solveForState({{"current_density:box", 2.0}}, state_tv);
            // solver.solveForState({{"current_density:box", 3.0}}, state_tv);
            // solver.solveForState({{"current_density:box", 4.0}}, state_tv);
         }
      }
   }
}

void aexact(const Vector &x, Vector& A)
{
   int dim = x.Size();
   A.SetSize(dim);

   A = 0.0;
   double y = x(1) - 0.5;
   if ( x(1) <= .5)
   {
      A(2) = y*y*y; 
   }
   else 
   {
      A(2) = -y*y*y;
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
   std::unique_ptr<Mesh> mesh(
      new Mesh(Mesh::MakeCartesian3D(nxy, nxy, nz,
                                     Element::TETRAHEDRON,
                                     1.0, 1.0, (double)nz / (double)nxy, true)));
      // new Mesh(Mesh::MakeCartesian2D(nxy, nxy,
      //                                Element::TRIANGLE, true,
      //                                1.0, 1.0, true)));

   // assign attributes to top and bottom sides
   for (int i = 0; i < mesh->GetNE(); ++i)
   {
      Element *elem = mesh->GetElement(i);

      Array<int> verts;
      elem->GetVertices(verts);

      bool below = true;
      for (int i = 0; i < verts.Size(); ++i)
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
