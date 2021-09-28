/// Solve the steady isentropic vortex problem on a quarter annulus
#include <fstream>
#include <iostream>

#include "catch.hpp"
#include "nlohmann/json.hpp"
#include "mfem.hpp"

#include "mesh_movement.hpp"

using namespace std;
using namespace mfem;
using namespace mach;

// Provide the options explicitly for regression tests
auto options = R"(
{
   "silent": false,
   "print-options": false,
   "space-dis": {
      "basis-type": "H1",
      "degree": 1
   },
   "time-dis": {
      "steady": true,
      "steady-abstol": 1e-12,
      "steady-reltol": 1e-10,
      "ode-solver": "PTC",
      "t-final": 100,
      "dt": 1e12,
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
      "type": "hypreboomeramg",
      "printlevel": 0
   },
   "nonlin-solver": {
      "type": "newton",
      "printlevel": 3,
      "maxiter": 50,
      "reltol": 1e-10,
      "abstol": 1e-12
   },
   "components": {
   },
   "problem-opts": {
      "uniform-stiff": {
         "lambda": 1,
         "mu": 1
      }
   },
   "outputs": {
   }
})"_json;

/// \brief Mapping function for coordinate field
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] X - new coordinate
void boxDisplacement(const Vector &x, Vector& X);

/// Generate mesh 
/// \param[in] nxy - number of nodes in the x and y directions
/// \param[in] nz - number of nodes in the z direction
std::unique_ptr<Mesh> buildBoxMesh(int nxy,
                                   int nz);

TEST_CASE("Mesh Movement Box Regression Test",
          "[Mesh-Movement-Box]")
{
   // define the appropriate exact solution error
   std::vector<double> target_error = {0.0, 0.0, 0.0, 0.0, 0.0};

   /// number of elements in Z direction
   auto nz = 8;

   int nxy = 1;
   for (int ref = 1; ref <= 2; ++ref)
   {  
      nxy *= 2;
      DYNAMIC_SECTION("...for mesh sizing nxy = " << nxy)
      {
         // construct the solver, set the initial condition, and solve
         unique_ptr<Mesh> smesh = buildBoxMesh(nxy, nz);
         auto solver = createSolver<LEAnalogySolver>(options, move(smesh));
         auto coord_field = solver->getNewField();
         solver->setFieldValue(*coord_field, boxDisplacement);
         MachInputs inputs;
         solver->solveForState(inputs, *coord_field);
         // solver->printField("sol", *coord_field, "coords", 0);

         // Compute error and check against appropriate target
         double l2_error = solver->calcL2Error(*coord_field, boxDisplacement);
         std::cout << "\n\nl2 error in field: " << l2_error << "\n\n\n";
         REQUIRE(l2_error == Approx(target_error[ref - 1]).margin(1e-10));

         // auto res_norm = solver->calcResidualNorm(*coord_field);
         // std::cout << "\n\n res_norm: " << res_norm << "\n";
         // auto &mesh_coords = solver->getMeshCoordinates();
         // mesh_coords.SetData(coord_field->GetData());
         // solver->printMesh("moved_box_mesh");
      }
   }
}

void boxDisplacement(const Vector &x, Vector& X)
{
   X.SetSize(x.Size());
   X = x;
   X *= 2; // new field is 2x
}

unique_ptr<Mesh> buildBoxMesh(int nxy, int nz)
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
