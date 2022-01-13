#include <random>

#include "catch.hpp"
#include "mfem.hpp"
#include "adept.h"

#include "flow_control_residual.hpp"
#include "sbp_fe.hpp"
#include "euler_fluxes.hpp"
#include "mach_input.hpp"

using namespace std;
using namespace mfem;
using namespace mach;

auto options = R"(
{
   "flow-param": {
      "mach": 0.5,
      "aoa": 0.0
   },
   "space-dis": {
      "degree": 1,
      "lps-coeff": 1.0,
      "basis-type": "csbp",
      "flux-fun": "Euler"
   },
   "bcs": {
      "far-field": [0, 1, 1, 1],
      "slip-wall": [1, 0, 0, 0]
   }
})"_json;

TEST_CASE("ControlResidual construction and evaluation", "[ControlResidual]")
{
   // construct the residual
   ControlResidual res(options);
   int num_var = getSize(res);
   REQUIRE(num_var == 2);

   // evaluate the residual at an arbitrary state
   Vector x(num_var);
   std::default_random_engine gen(std::random_device{}());
   std::uniform_real_distribution<double> uniform_rand(0.0, 1.0);
   for (int i = 0; i < num_var; ++i)
   {
      x(i) = uniform_rand(gen);
   }
   double time = uniform_rand(gen);
   auto inputs = MachInputs({{"state", x}, {"time", time}});
   Vector res_vec(num_var);
   evaluate(res, inputs, res_vec);
   REQUIRE( res_vec(0) == Approx(-x(1)).margin(1e-14) );
   REQUIRE( res_vec(1) == Approx(x(0)).margin(1e-14) );

   // evaluate the Jacobian; perform matrix-vector product on random vector
   Operator &Jac = getJacobian(res, inputs, "state");
   Vector v(num_var), Jac_v(num_var);
   for (int i = 0; i < num_var; ++i)
   {
      v(i) = uniform_rand(gen);
   }
   Jac.Mult(v, Jac_v);
   REQUIRE( Jac_v(0) == Approx(-v(1)).margin(1e-14) );
   REQUIRE( Jac_v(1) == Approx(v(0)).margin(1e-14) );

   // check the entropy 
   double entropy = calcEntropy(res, inputs);
   REQUIRE(entropy == Approx(x(0) * x(0) + x(1) * x(1)).margin(1e-14));

   // check entropy change
   v = 0.0;
   inputs = MachInputs(
       {{"state", x}, {"time", time}, {"state_dot", v}, {"dt", 0.0}});
   double entropy_change = calcEntropyChange(res, inputs);
   REQUIRE(entropy_change == Approx(0.0).margin(1e-14));
}