/// Solve the steady isentropic vortex problem on a quarter annulus
#include <fstream>
#include <iostream>

#include "catch.hpp"

#include "miso.hpp"

using namespace std;
using namespace mfem;
using namespace miso;

// Provide the options explicitly for regression tests
auto options = R"(
{
   "print-options": false,
   "flow-param": {
      "miso": 1.0,
      "aoa": 0.0
   },
   "space-dis": {
      "degree": 1,
      "lps-coeff": 1.0,
      "basis-type": "csbp"
   },
   "time-dis": {
      "type": "PTC",
      "steady": true,
      "steady-abstol": 1e-12,
      "steady-restol": 1e-10,
      "t-final": 100,
      "dt": 1e12,
      "cfl": 1.0,
      "res-exp": 2.0
   },
   "bcs": {
      "vortex": [1, 2, 3],
      "slip-wall": [4]
   },
   "nonlin-solver": {
      "printlevel": 1,
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
      "type": "hyprefgmres",
      "printlevel": 0,
      "filllevel": 3,
      "maxiter": 100,
      "reltol": 1e-2,
      "abstol": 1e-12
   },
   "lin-prec": {
      "type": "hypreilu",
      "ilu-reorder": 1,
      "ilu-type": 0,
      "lev-fill": 1,
      "printlevel": 0
   },
   "saveresults":false,
   "outputs": {
      "drag": {
         "boundaries": [4]
      }
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

TEMPLATE_TEST_CASE_SIG("Steady Vortex Solver Regression Test",
                       "[Euler-Vortex]", ((bool entvar), entvar), true, false)
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
         auto mesh = buildQuarterAnnulusMesh(mesh_degree, nx, nx);
         options["flow-param"]["entropy-state"] = entvar;
         FlowSolver<2, entvar> solver(MPI_COMM_WORLD, options, std::move(mesh));
         Vector state_tv(solver.getStateSize());
         solver.setState(uexact, state_tv);
         MISOInputs inputs({{"state", state_tv}});
         solver.solveForState(inputs, state_tv);

         // Compute error and check against appropriate target:
         // Using calcConservativeVarsL2Error, we should have the same error 
         // for both entvar=true and entvar=false
         double l2_error = solver.calcConservativeVarsL2Error(uexact, 0);
         REQUIRE(l2_error == Approx(target_error[nx - 1]).margin(1e-10));

         // Compute drag and check against target 
         solver.createOutput("drag", options["outputs"].at("drag"));
         double drag = solver.calcOutput("drag", inputs);
         double drag_error = fabs(drag - (-1 /miso::euler::gamma));
         REQUIRE(drag_error == Approx(target_drag_error[nx-1]).margin(1e-10));
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
   calcEntropyVars<double, 2, false>(q.GetData(), w.GetData());
}