#include "catch.hpp"
#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "flow_solver.hpp"
#include "euler_fluxes.hpp"

/// Unsteady isentropic exact solution for conservative variables
/// \param[in] x - spatial location at which exact solution is sought
/// \param[out] q - conservative variables at `x`.
void vortexExact(const mfem::Vector &xy, mfem::Vector& q)
{
   using namespace mach;

   double t = 0.0;
   double x = xy(0) - 5.0 - t; // x0 = 5.0
   double y = xy(1) - 5.0; // y0 = 5.0
   const double Mach = 0.5;
   const double epsilon = 1.0;
   double expf = exp(0.5*(1 - x*x - y*y));

   q.SetSize(4);
   q(0) = pow(1 - euler::gami*pow(epsilon*expf*Mach/M_PI,2.0)/8.0,
              -1.0/euler::gami);
   q(1) = 1.0 - epsilon*y*expf/(2*M_PI);
   q(2) = epsilon*x*expf/(2*M_PI);
   double press = pow(q(0), euler::gamma)/(euler::gamma * Mach * Mach);
   q(3) = press/euler::gami + 0.5*q(0)*(q(1)*q(1) + q(2)*q(2));
   q(1) *= q(0);
   q(2) *= q(1);
}

/// Unsteady isentropic exact solution for entropy variables
/// \param[in] x - spatial location at which exact solution is sought
/// \param[out] w - entropy variables at `x`.
void vortexExactEntVars(const mfem::Vector &x, mfem::Vector& w)
{
   w.SetSize(4);
   mfem::Vector q(4);
   vortexExact(x, q);
   mach::calcEntropyVars<double, 2, true>(q.GetData(), w.GetData());
}

// TEMPLATE_TEST_CASE_SIG("Testing FlowSolver on unsteady isentropic vortex",
//                        "[Euler-Vortex]", ((bool entvar), entvar), true, false)
// {
TEST_CASE("Testing FlowSolver on unsteady isentropic vortex", "[FlowSolver]")
{
   const bool entvar = false;

   const bool verbose = true; // set to true for some output 
   std::ostream *out = verbose ? mach::getOutStream(0) : mach::getOutStream(1);
   using namespace mfem;
   using namespace mach;
   auto uexact = !entvar ? vortexExact : vortexExactEntVars;

   // Provide the options explicitly for regression tests
   auto options = R"(
   {
      "silent" : false,
      "paraview": {
         "each-timestep": true,
         "directory": "isentropic-vortex"
      },
      "flow-param": {
         "entropy-state": false,
         "mach": 0.5
      },
      "space-dis": {
         "degree": 1,
         "lps-coeff": 0.0,
         "basis-type": "csbp",
         "flux-fun": "IR"
      },
      "time-dis": {
         "type": "RRK",
         "steady": false,
         "t-final": 0.2,
         "dt": 0.1,
         "cfl": 1.0,
         "entropy-log": true
      },
      "nonlin-solver": {
         "printlevel": 0,
         "maxiter": 50,
         "reltol": 1e-8,
         "abstol": 1e-10
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
         "entropy": {}
      }
   })"_json;
   if constexpr(entvar)
   {
      options["flow-param"].at("entropy-state") = true;
   }

   // Build a periodic uniform mesh over the square domain
   const int num = 5;
   Mesh mesh = Mesh::MakeCartesian2D(
       num, num, Element::TRIANGLE, true /* gen. edges */, 10.0, 10.0, true);
   std::vector<Vector> translations;
   translations.push_back(Vector({10.0, 0.0}));
   translations.push_back(Vector({0.0, 10.0}));
   std::vector<int> v2v = mesh.CreatePeriodicVertexMapping(translations);

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for polynomial degree p = " << p)
      {
         options["space-dis"].at("degree") = p;

         // Create solver and set initial guess to exact
         std::unique_ptr<Mesh> smesh =
             std::make_unique<Mesh>(Mesh::MakePeriodic(mesh, v2v));
         FlowSolver<2, entvar> solver(
             MPI_COMM_WORLD, options, std::move(smesh));
         mfem::Vector state_tv(solver.getStateSize());
         solver.setState(uexact, state_tv);

         // write the initial state for debugging 
         auto &state = solver.getState();
         mach::ParaViewLogger paraview("test_flow_solver",
            state.gridFunc().ParFESpace()->GetParMesh());
         paraview.registerField("state", state.gridFunc());
         paraview.saveState(state_tv, "state", 0, 1.0, 0);

         // get the initial entropy 
         solver.createOutput("entropy", options["outputs"].at("entropy"));
         MachInputs inputs({{"state", state_tv}});
         double entropy0 = solver.calcOutput("entropy", inputs);

         // Solve for the state; inputs are not used at present...
         solver.solveForState(inputs, state_tv);
         state.distributeSharedDofs(state_tv);

         // get the final entropy 
         double entropy = solver.calcOutput("entropy", inputs);
         REQUIRE(entropy == Approx(entropy0).margin(1e-10));
         //std::cout << "entropy change = " << entropy0 - entropy << std::endl;
      }
   }
}

/// Steady isentropic exact solution for conservative variables
/// \param[in] x - spatial location at which exact solution is sought
/// \param[out] q - conservative variables at `x`.
void steadyVortexExact(const mfem::Vector &x, mfem::Vector& q)
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

/// Steady isentropic exact solution for entropy variables
/// \param[in] x - spatial location at which exact solution is sought
/// \param[out] w - entropy variables at `x`.
void steadyVortexExactEntVars(const mfem::Vector &x, mfem::Vector& w)
{
   w.SetSize(4);
   mfem::Vector q(4);
   steadyVortexExact(x, q);
   mach::calcEntropyVars<double, 2, false>(q.GetData(), w.GetData());
}

TEMPLATE_TEST_CASE_SIG("Testing FlowSolver on steady isentropic vortex",
                       "[Euler-Vortex]", ((bool entvar), entvar), true, false)
{
   const bool verbose = true; // set to true for some output 
   std::ostream *out = verbose ? mach::getOutStream(0) : mach::getOutStream(1);
   using namespace mfem;
   using namespace mach;
   auto uexact = !entvar ? steadyVortexExact : steadyVortexExactEntVars;

   // Provide the options explicitly for regression tests
   auto options = R"(
   {
      "silent" : false,
      "flow-param": {
         "entropy-state": false,
         "mach": 1.0
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
         "vortex": [1, 2, 3],
         "slip-wall": [4]
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
         "drag": {
            "boundaries": [4]
         }
      }
   })"_json;
   if (entvar)
   {
      options["flow-param"].at("entropy-state") = true;
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
         FlowSolver<2,entvar> solver(MPI_COMM_WORLD, options, std::move(mesh));
         mfem::Vector state_tv(solver.getStateSize());
         solver.setState(uexact, state_tv);

         // write the initial state for debugging 
         auto &state = solver.getState();
         mach::ParaViewLogger paraview("test_flow_solver", &state.mesh());
         paraview.registerField("state", state.gridFunc());
         paraview.saveState(state_tv, "state", 0, 1.0, 0);

         // Solve for the state
         MachInputs inputs;
         solver.solveForState(inputs, state_tv);
         state.distributeSharedDofs(state_tv);

         double l2_error = solver.calcConservativeVarsL2Error(uexact, 0);
         std::cout << "l2 error = " << l2_error << std::endl;
         REQUIRE(l2_error == Approx(target_error[nx - 1]).margin(1e-10));
 
         inputs = MachInputs({
            {"time", M_PI}, {"state", state.gridFunc()}
         });
         solver.createOutput("drag", options["outputs"].at("drag"));
         double drag_error = fabs(solver.calcOutput("drag", inputs) - (-1 /mach::euler::gamma));
         REQUIRE(drag_error == Approx(target_drag_error[nx-1]).margin(1e-10));
      }
   }
}