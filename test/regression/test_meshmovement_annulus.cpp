/// Solve the steady isentropic vortex problem on a quarter annulus
#include <fstream>
#include <iostream>

#include "catch.hpp"
#include "json.hpp"
#include "mfem.hpp"

#include "mesh_movement.hpp"

using namespace std;
using namespace mfem;
using namespace mach;

// Provide the options explicitly for regression tests
auto options = R"(
{
   "silent": false,
   "print-options": true,
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
      "printlevel": -1,
      "maxiter": 50,
      "reltol": 1e-10,
      "abstol": 1e-12
   },
   "problem-opts": {
      "uniform-stiff": {
         "lambda": 1,
         "mu": 1
      }
   }
})"_json;

/// \brief Mapping function for coordinate field for extending the annulus
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] X - new coordinate
void annulusExtension(const Vector &x, Vector& X);

/// \brief Mapping function for coordinate field for contracting the annulus
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] X - new coordinate
void annulusContraction(const Vector &x, Vector& X);

/// Generate quarter annulus mesh 
/// \param[in] degree - polynomial degree of the mapping
/// \param[in] num_rad - number of nodes in the radial direction
/// \param[in] num_ang - number of nodes in the angular direction
/// \param[in] num_z - number of nodes in the z direction
std::unique_ptr<Mesh> buildQuarterAnnulusMesh(int degree,
                                              int num_rad,
                                              int num_ang,
                                              int num_z);

TEST_CASE("Mesh Movement Annulus Extension Regression Test",
          "[Mesh-Movement-Annulus-Extension]")
{
   // define the appropriate exact solution error
   std::vector<std::vector<double>> target_error = {
      {0.88841942, 0.39106207, 0.2952895, 0.27851959},
      {0.3637634, 0.37588127, 0.38201795, 0.38372452}
   };

   /// number of elements in Z direction
   auto nz = 2;

   for (int order = 1; order <= 1; ++order)
   {
      // order = 2;
      options["space-dis"]["degree"] = order;
      int nxy = 1;
      for (int ref = 1; ref <= 4; ++ref)
      {  
         nxy *= 2;
         DYNAMIC_SECTION("...for order " << order << " and mesh sizing nxy = " << nxy)
         {
            // construct the solver, set the initial condition, and solve
            unique_ptr<Mesh> smesh = buildQuarterAnnulusMesh(order, nxy, nxy, nz);
            auto solver = createSolver<LEAnalogySolver>(options, move(smesh));
            solver->setInitialCondition(annulusExtension);
            solver->solveForState();
            auto fields = solver->getFields();

            // Compute error and check against appropriate target
            mfem::VectorFunctionCoefficient dispEx(3, annulusExtension);
            double l2_error = fields[0]->ComputeL2Error(dispEx);
            // std::cout.precision(8);
            // std::cout << "l2 error in field: " << l2_error << "\n";
            REQUIRE(l2_error == Approx(target_error[order - 1][ref - 1]).margin(1e-10));

            // auto &mesh_coords = solver->getMeshCoordinates();
            // mesh_coords += *fields[0];
            // std::stringstream filename("moved_annulus_mesh_extend");
            // filename << "p";
            // filename << order;
            // filename << "nx";
            // filename << nxy;
            // std::cout << "filename: " << filename.str();
            // solver->printMesh(filename.str());
         }
      }
   }
}

TEST_CASE("Mesh Movement Annulus Contraction Regression Test",
          "[Mesh-Movement-Annulus-Contraction]")
{
   // define the appropriate exact solution error
   std::vector<std::vector<double>> target_error = {
      {0.13187374, 0.044586049, 0.024658222, 0.02083205},
      {0.027446909, 0.027539583, 0.027979098, 0.028106704}
   };

   /// number of elements in Z direction
   auto nz = 2;

   for (int order = 1; order <= 2; ++order)
   {
      // order = 2;
      options["space-dis"]["degree"] = order;
      int nxy = 1;
      for (int ref = 1; ref <= 4; ++ref)
      {  
         nxy *= 2;
         DYNAMIC_SECTION("...for order " << order << " and mesh sizing nxy = " << nxy)
         {
            // construct the solver, set the initial condition, and solve
            unique_ptr<Mesh> smesh = buildQuarterAnnulusMesh(order, nxy, nxy, nz);
            auto solver = createSolver<LEAnalogySolver>(options, move(smesh));
            solver->setInitialCondition(annulusContraction);
            solver->solveForState();
            auto fields = solver->getFields();

            // Compute error and check against appropriate target
            mfem::VectorFunctionCoefficient dispEx(3, annulusContraction);
            double l2_error = fields[0]->ComputeL2Error(dispEx);
            REQUIRE(l2_error == Approx(target_error[order - 1][ref - 1]).margin(1e-10));

            // {
            //    std::string filename("annulus_mesh_contract");
            //    filename += "p";
            //    filename += std::to_string(order);
            //    filename += "nx";
            //    filename += std::to_string(nxy);
            //    solver->printMesh(filename);
            // }

            // auto &mesh_coords = solver->getMeshCoordinates();
            // mesh_coords += *fields[0];
            // {
            //    std::string filename("moved_annulus_mesh_contract");
            //    filename += "p";
            //    filename += std::to_string(order);
            //    filename += "nx";
            //    filename += std::to_string(nxy);
            //    solver->printMesh(filename);
            // }
         }
      }
   }
}

void annulusExtension(const Vector &x, Vector& X)
{
   X.SetSize(x.Size());

   // This lambda function transforms from (x,y,z) space to (r,\theta,z) space
   auto rtz_fun = [](const Vector &xyz, Vector &rtz)
   {
      rtz(0) = std::sqrt(xyz(0)*xyz(0) + xyz(1)*xyz(1)) - 1;
      rtz(1) = std::atan2(xyz(1), xyz(0));
      rtz(2) = xyz(2);
   };
   Vector rtz(x.Size());
   rtz_fun(x, rtz);
   rtz(1) *= 2.0;

   // This lambda function transforms from (r,\theta,z) space to (x,y,z) space
   auto xyz_fun = [](const Vector &rtz, Vector &xyz)
   {
      xyz(0) = (rtz(0) + 1.0)*cos(rtz(1)); // need + 1.0 to shift r away from origin
      xyz(1) = (rtz(0) + 1.0)*sin(rtz(1));
      xyz(2) = rtz(2);
   };
   xyz_fun(rtz, X); // new field has had theta doubled (semi-annulus now)
   X -= x; // get displacement
}

void annulusContraction(const Vector &x, Vector& X)
{
   X.SetSize(x.Size());

   // This lambda function transforms from (x,y,z) space to (r,\theta,z) space
   auto rtz_fun = [](const Vector &xyz, Vector &rtz)
   {
      rtz(0) = std::sqrt(xyz(0)*xyz(0) + xyz(1)*xyz(1)) - 1;
      rtz(1) = std::atan2(xyz(1), xyz(0));
      rtz(2) = xyz(2);
   };
   Vector rtz(x.Size());
   rtz_fun(x, rtz);
   rtz(1) *= 0.75;

   // This lambda function transforms from (r,\theta,z) space to (x,y,z) space
   auto xyz_fun = [](const Vector &rtz, Vector &xyz)
   {
      xyz(0) = (rtz(0) + 1.0)*cos(rtz(1)); // need + 1.0 to shift r away from origin
      xyz(1) = (rtz(0) + 1.0)*sin(rtz(1));
      xyz(2) = rtz(2);
   };
   xyz_fun(rtz, X); // new field has had theta doubled (semi-annulus now)
   X -= x; // get displacement
}

unique_ptr<Mesh> buildQuarterAnnulusMesh(int degree,
                                         int num_rad,
                                         int num_ang,
                                         int num_z)
{
   auto mesh_ptr = unique_ptr<Mesh>(new Mesh(num_rad, num_ang, num_z,
                                             Element::TETRAHEDRON, true,
                                             2.0, M_PI*0.5, 1.0, true));
   // strategy:
   // 1) generate a fes for Lagrange elements of desired degree
   // 2) create a Grid Function using a VectorFunctionCoefficient
   // 4) use mesh_ptr->NewNodes(nodes, true) to set the mesh nodes
   
   // Problem: fes does not own fec, which is generated in this function's scope
   // Solution: the grid function can own both the fec and fes
   H1_FECollection *fec = new H1_FECollection(degree, 3 /* = dim */);
   FiniteElementSpace *fes = new FiniteElementSpace(mesh_ptr.get(), fec, 3,
                                                    Ordering::byVDIM);

   // This lambda function transforms from (r,\theta) space to (x,y) space
   auto xyz_fun = [](const Vector& rtz, Vector &xyz)
   {
      xyz(0) = (rtz(0) + 1.0)*cos(rtz(1)); // need + 1.0 to shift r away from origin
      xyz(1) = (rtz(0) + 1.0)*sin(rtz(1));
      xyz(2) = rtz(2);
   };
   VectorFunctionCoefficient xyz_coeff(3, xyz_fun);
   GridFunction *xyz = new GridFunction(fes);
   xyz->MakeOwner(fec);
   xyz->ProjectCoefficient(xyz_coeff);

   mesh_ptr->NewNodes(*xyz, true);
   return mesh_ptr;
}
