/// Solve the Navier-Stokes MMS verification problem
#include <fstream>
#include <iostream>

#include "adept.h"
#include "catch.hpp"
#include "mfem.hpp"

#include "navier_stokes.hpp"

using namespace std;
using namespace mfem;
using namespace mach;

// Provide the options explicitly for regression tests
auto options = R"(
{
   "print-options": false,
   "flow-param": {
      "mu": 1.0,
      "Re": 10.0,
      "Pr": 0.75,
      "viscous-mms": true
   },
   "space-dis": {
      "degree": 1,
      "lps-coeff": 1.0,
      "basis-type": "csbp"
   },
   "time-dis": {
      "steady": true,
      "steady-abstol": 1e-12,
      "steady-restol": 1e-10,
      "ode-solver": "PTC",
      "dt": 1e12,
      "cfl": 1.0,
      "res-exp": 2.0
   },
   "nonlin-solver": {
      "printlevel": 1,
      "maxiter": 50,
      "reltol": 1e-1,
      "abstol": 1e-12
   },
   "lin-solver": {
      "type": "hyprefgmres",
      "printlevel": 0,
      "filllevel": 3,
      "maxiter": 100,
      "reltol": 1e-2,
      "abstol": 1e-12
   },
   "bcs": {
      "no-slip-adiabatic": [1, 0, 1, 0],
      "viscous-mms": [0, 1, 0, 1]
   },
   "outputs":
   {
      "drag": [1, 0, 1, 0]
   }
})"_json;

/// Generate smoothly perturbed mesh 
/// \param[in] degree - polynomial degree of the mapping
/// \param[in] num_x - number of nodes in the x direction
/// \param[in] num_y - number of nodes in the y direction
Mesh buildCurvilinearMesh(int degree, int num_x, int num_y);

/// \brief Defines the exact solution for the manufactured solution
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - state variables stored as a 4-vector
void uexact(const Vector &x, Vector& u);

TEST_CASE( "Navier-Stokes MMS regression test", "[NS-MMS]")
{
   int nx = 2;

   const double target_error[4] = {
       0.0737783882613367, // p = 1
       0.0301870954654436, // p = 2
       0.0110927229078876, // p = 3
       0.00373341426303304 // p = 4
   };
   const double target_drag_error[4] = {
       0.0265315876403271, // p = 1
       0.0102832210446619, // p = 2
       0.00980301225032143,// p = 3
       0.00435654251310824 // p = 4
   };

   // Just test p=1 and p=2 to keep test time limited
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for solution degree p = " << p)
      {
         options["space-dis"]["degree"] = p;
         // std::cout << setw(3) << options << std::endl;
         
         // construct the mesh
         unique_ptr<Mesh> smesh(new Mesh(buildCurvilinearMesh(p+1, nx, nx)));
         //std::cout << "Number of elements " << smesh->GetNE() << '\n';

         // construct the solver, set the initial condition, and solve
         auto solver = createSolver<NavierStokesSolver<2>>(options,
                                                           move(smesh));
         solver->setInitialCondition(uexact);
         solver->solveForState();

         // get L2 density error, and compare against target error
         double l2_error = solver->calcL2Error(uexact, 0);
         REQUIRE(l2_error == Approx(target_error[p-1]).margin(1e-10));

         // get the drag error and compare against target error
         // Note: the "exact" value of drag is set to 1.6, but this has
         // not been verified (unlike the L2 error).
         auto drag_opts = R"({ 
            "boundaries": [1, 0, 1, 0]
         })"_json;
         solver->createOutput("drag", drag_opts);
         double drag_error = solver->calcOutput("drag") - 1.6;
         REQUIRE(drag_error == Approx(target_drag_error[p-1]).margin(1e-10));
      }
   }
}

Mesh buildCurvilinearMesh(int degree, int num_x, int num_y)
{
   Mesh mesh = Mesh::MakeCartesian2D(num_x, num_y, Element::TRIANGLE, 
                                     true /* gen. edges */, 1.0, 1.0, true);
   // strategy:
   // 1) generate a fes for Lagrange elements of desired degree
   // 2) create a Grid Function using a VectorFunctionCoefficient
   // 4) use mesh_ptr->NewNodes(nodes, true) to set the mesh nodes
   H1_FECollection *fec = new H1_FECollection(degree, 2 /* = dim */);
   FiniteElementSpace *fes = new FiniteElementSpace(&mesh, fec, 2,
                                                    Ordering::byVDIM);

   auto xy_fun = [](const Vector& xi, Vector &x)
   {
      x(0) = xi(0) + (1.0/40.0)*sin(2.0*M_PI*xi(0))*sin(2.0*M_PI*xi(1));
      x(1) = xi(1) + (1.0/40.0)*sin(2.0*M_PI*xi(1))*sin(2.0*M_PI*xi(0));
   };
   VectorFunctionCoefficient xy_coeff(2, xy_fun);
   GridFunction *xy = new GridFunction(fes);
   xy->MakeOwner(fec);
   xy->ProjectCoefficient(xy_coeff);
   mesh.NewNodes(*xy, true);
   return mesh;
}

void uexact(const Vector &x, Vector& q)
{
   const double rho0 = 1.0;
   const double rhop = 0.05;
   const double U0 = 0.5;
   const double Up = 0.05;
   const double T0 = 1.0;
   const double Tp = 0.05;
   q.SetSize(4);
   q(0) = rho0 + rhop*pow(sin(M_PI*x(0)),2)*sin(M_PI*x(1));
   q(1) = 4.0*U0*x(1)*(1.0 - x(1)) + Up*sin(2 * M_PI * x(1)) * pow(sin(M_PI * x(0)),2);
   q(2) = -Up*pow(sin(2 * M_PI * x(0)),2) * sin(M_PI * x(1));
   double T = T0 + Tp*(pow(x(0), 4) - 2 * pow(x(0), 3) + pow(x(0), 2) 
                     + pow(x(1), 4) - 2 * pow(x(1), 3) + pow(x(1), 2));
   double p = q(0)*T; // T is nondimensionalized by 1/(R*a_infty^2)
   q(3) = p/euler::gami + 0.5*q(0)*(q(1)*q(1) + q(2)*q(2));
   q(1) *= q(0);
   q(2) *= q(0);
}