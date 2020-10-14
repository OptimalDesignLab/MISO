/// Solve the steady isentropic vortex problem on a quarter annulus
#include <fstream>
#include <iostream>

#include "adept.h"
#include "catch.hpp"
#include "mfem.hpp"

#include "euler.hpp"

using namespace std;
using namespace mfem;
using namespace mach;

// Provide the options explicitly for regression tests
auto options = R"(
{
   "print-options": false,
   "flow-param": {
      "mach": 1.0,
      "aoa": 0.0
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
      "t-final": 100,
      "dt": 1e12,
      "cfl": 1.0,
      "res-exp": 2.0
   },
   "bcs": {
      "vortex": [1, 1, 1, 0],
      "slip-wall": [0, 0, 0, 1]
   },
   "nonlin-solver": {
      "printlevel": 0,
      "maxiter": 50,
      "reltol": 1e-1,
      "abstol": 1e-12
   },
   "petscsolver": {
      "ksptype": "gmres",
      "pctype": "lu",
      "abstol": 1e-15,
      "reltol": 1e-15,
      "maxiter": 100,
      "printlevel": 0
   },
   "lin-solver": {
      "printlevel": 0,
      "filllevel": 3,
      "maxiter": 100,
      "reltol": 1e-2,
      "abstol": 1e-12
   },
   "saveresults":false,
   "outputs":
   { 
      "drag": [0, 0, 0, 1]
   }
})"_json;

/// NOTE: Not sure how I'll iterate over order, etc.
bool entvarg;

/// \brief Exact conservative variables for the steady isentropic vortex
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] q - conservative variables stored as a 4-vector
void qexact(const Vector &x, Vector& q);

/// \brief Exact entropy variables for the steady isentropic vortex
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] w - entropy variables stored as a 4-vector
void wexact(const Vector &x, Vector& w);

/// Generate quarter annulus mesh 
/// \param[in] degree - polynomial degree of the mapping
/// \param[in] num_rad - number of nodes in the radial direction
/// \param[in] num_ang - number of nodes in the angular direction
std::unique_ptr<Mesh> buildQuarterAnnulusMesh(int degree, int num_rad,
                                              int num_ang);

TEMPLATE_TEST_CASE_SIG("Steady Vortex Solver Regression Test",
                       "[Euler-Vortex]", ((bool entvar), entvar), false, true)
{
   // define the appropriate exact solution based on entvar
   auto uexact = !entvar ? qexact : wexact;
   int mesh_degree = options["space-dis"]["degree"].get<int>() + 1;
   std::vector<double> target_error;
   std::vector<double> target_drag_error;
   if (entvar)
   {
      target_error = {0.0901571779707352, 0.0311496716090168,
                      0.0160317976193675, 0.0098390277746438};
      target_drag_error = {0.0149959859366922, 0.00323047181284186,
                           0.00103944525942667, 0.000475003178886935};
      // target_error = {0.0690131081, 0.0224304871, 0.0107753424, 0.0064387612};
   }
   else
   {
      target_error = {0.0901571779707358, 0.0311496716080777,
                      0.0160317976193642, 0.00983902777464837};
      target_drag_error = {0.0149959859366923, 0.00323047181325709,
                           0.00103944525936905, 0.000475003178874278};
      // target_error = {0.0700148195, 0.0260625842, 0.0129909277, 0.0079317615};
   }

   for (int nx = 1; nx <= 4; ++nx)
   {
      DYNAMIC_SECTION("...for mesh sizing nx = " << nx)
      {
         // construct the solver, set the initial condition, and solve
         unique_ptr<Mesh> smesh = buildQuarterAnnulusMesh(mesh_degree, nx, nx);
         auto solver = createSolver<EulerSolver<2,entvar>>(options,
                                                           move(smesh));
         solver->setInitialCondition(uexact);
         solver->solveForState();

         // Compute error and check against appropriate target:
         // Using calcConservativeVarsL2Error, we should have the same error 
         // for both entvar=true and entvar=false
         double l2_error = (static_cast<EulerSolver<2, entvar>&>(*solver)
                            .calcConservativeVarsL2Error(uexact, 0));
         //double l2_error = solver->calcL2Error(uexact, 0);
         double drag_error = fabs(solver->calcOutput("drag") - (-1 /mach::euler::gamma));
         REQUIRE(l2_error == Approx(target_error[nx - 1]).margin(1e-10));
      }
   }
}

// exact solution for conservative variables
void qexact(const Vector &x, Vector& q)
{
   q.SetSize(4);
   double ri = 1.0;
   double Mai = 0.5; //0.95 
   double rhoi = 2.0;
   double prsi = 1.0/euler::gamma;
   double rinv = ri/sqrt(x(0)*x(0) + x(1)*x(1));
   double rho = rhoi*pow(1.0 + 0.5*euler::gami*Mai*Mai*(1.0 - rinv*rinv),
                         1.0/euler::gami);
   double Ma = sqrt((2.0/euler::gami)*( ( pow(rhoi/rho, euler::gami) ) * 
                    (1.0 + 0.5*euler::gami*Mai*Mai) - 1.0 ) );
   double theta;
   if (x(0) > 1e-15)
   {
      theta = atan(x(1)/x(0));
   }
   else
   {
      theta = M_PI/2.0;
   }
   double press = prsi* pow( (1.0 + 0.5*euler::gami*Mai*Mai) / 
                 (1.0 + 0.5*euler::gami*Ma*Ma), euler::gamma/euler::gami);
   double a = sqrt(euler::gamma*press/rho);

   q(0) = rho;
   q(1) = -rho*a*Ma*sin(theta);
   q(2) = rho*a*Ma*cos(theta);
   q(3) = press/euler::gami + 0.5*rho*a*a*Ma*Ma;
}

void wexact(const Vector &x, Vector& w)
{
   w.SetSize(4);
   Vector q(4);
   qexact(x, q);
   calcEntropyVars<double, 2>(q.GetData(), w.GetData());
}

unique_ptr<Mesh> buildQuarterAnnulusMesh(int degree, int num_rad, int num_ang)
{
   auto mesh_ptr = unique_ptr<Mesh>(new Mesh(num_rad, num_ang,
                                             Element::TRIANGLE, true /* gen. edges */,
                                             2.0, M_PI*0.5, true));
   // strategy:
   // 1) generate a fes for Lagrange elements of desired degree
   // 2) create a Grid Function using a VectorFunctionCoefficient
   // 4) use mesh_ptr->NewNodes(nodes, true) to set the mesh nodes
   
   // Problem: fes does not own fec, which is generated in this function's scope
   // Solution: the grid function can own both the fec and fes
   H1_FECollection *fec = new H1_FECollection(degree, 2 /* = dim */);
   FiniteElementSpace *fes = new FiniteElementSpace(mesh_ptr.get(), fec, 2,
                                                    Ordering::byVDIM);

   // This lambda function transforms from (r,\theta) space to (x,y) space
   auto xy_fun = [](const Vector& rt, Vector &xy)
   {
      xy(0) = (rt(0) + 1.0)*cos(rt(1)); // need + 1.0 to shift r away from origin
      xy(1) = (rt(0) + 1.0)*sin(rt(1));
   };
   VectorFunctionCoefficient xy_coeff(2, xy_fun);
   GridFunction *xy = new GridFunction(fes);
   xy->MakeOwner(fec);
   xy->ProjectCoefficient(xy_coeff);

   mesh_ptr->NewNodes(*xy, true);
   return mesh_ptr;
}
