#include "catch.hpp"
#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "flow_solver.hpp"
#include "euler_fluxes.hpp"

/// Steady isentropic vortex exact solution for conservative variables
/// \param[in] x - spatial location at which exact solution is sought
/// \param[out] q - conservative variables at `x`.
void vortexExact(const mfem::Vector &x, mfem::Vector& q)
{
   using namespace mach;
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

/// Steady isentropic vortex exact solution for entropy variables
/// \param[in] x - spatial location at which exact solution is sought
/// \param[out] w - entropy variables at `x`.
void vortexExactEntVars(const mfem::Vector &x, mfem::Vector& w)
{
   using namespace mach;
   w.SetSize(4);
   mfem::Vector q(4);
   vortexExact(x, q);
   calcEntropyVars<double, 2>(q.GetData(), w.GetData());
}

TEMPLATE_TEST_CASE_SIG("Testing FlowSolver on the steady isentropic vortex",
                       "[FlowSolver]", ((bool entvar), entvar), true, false)
{
   const bool verbose = true; // set to true for some output 
   std::ostream *out = verbose ? mach::getOutStream(0) : mach::getOutStream(1);
   using namespace mfem;
   using namespace mach;
   auto uexact = !entvar ? vortexExact : vortexExactEntVars;
  
   // Provide the options explicitly for regression tests
   auto options = R"(
   {
      "flow-param": {
         "entropy-state": false,
         "mach": 0.5
      },
      "space-dis": {
         "degree": 1,
         "lps-coeff": 1.0,
         "basis-type": "csbp",
         "flux-fun": "IR"
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
         "vortex": [1, 1, 1, 0],
         "slip-wall": [0, 0, 0, 1]
      },
      "nonlin-solver": {
         "printlevel": 0,
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
      "saveresults": false,
      "outputs":
      { 
         "drag": { "boundaries": [0, 0, 0, 1] }
      }
   })"_json;
   if (entvar)
   {
      options.at("flow-param").at("entropy-state") = true;
   }

   // Both the conservative and entropy-based states should give the same error
   // for the CSBP scheme (the error is the density error)
   std::vector target_error = {0.0901571779707352,
                               0.0311496716090168,
                               0.0160317976193675,
                               0.0098390277746438};
   std::vector target_drag_error = {0.0149959859366922,
                                    0.00323047181284186,
                                    0.00103944525942667,
                                    0.000475003178886935};

   for (int nx = 1; nx <= 4; ++nx)
   {
      DYNAMIC_SECTION("...for mesh sizing nx = " << nx)
      {
         // Build the vortex mesh
         int mesh_degree = options["space-dis"]["degree"].get<int>() + 1;
         auto mesh = buildQuarterAnnulusMesh(mesh_degree, nx, nx);

         // Create solver and set initial guess to exact
         FlowSolver solver(MPI_COMM_WORLD, options, std::move(mesh));
         mfem::Vector state_tv(solver.getStateSize());
         solver.setState(uexact, state_tv);

         // write the initial state for debugging 
         auto &state = solver.getState();
         mach::ParaViewLogger paraview("test_flow_solver",
            state.gridFunc().ParFESpace()->GetParMesh());
         paraview.registerField("state", state.gridFunc());
         paraview.saveState(state_tv, "state", 0, 1.0, 0);

         // Solve for the state
         MachInputs inputs;
         solver.solveForState(inputs, state_tv);
         state.distributeSharedDofs(state_tv);

         double l2_error = solver.calcConservativeVarsL2Error(uexact, 0);
         std::cout << "l2 error = " << l2_error << std::endl;
         REQUIRE(l2_error == Approx(target_error[nx - 1]).margin(1e-10));

         solver.createOutput("drag", options["outputs"]["drag"]);
         double drag_error = fabs(solver.calcOutput("drag") - (-1 /mach::euler::gamma));
         REQUIRE(drag_error == Approx(target_drag_error[nx-1]).margin(1e-10));
      }
   }
}
