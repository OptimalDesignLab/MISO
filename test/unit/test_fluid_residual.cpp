#include <random>

#include "catch.hpp"
#include "mfem.hpp"
#include "adept.h"

#include "fluid_residual.hpp"
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

TEST_CASE("FluidResidual construction and evaluation", "[FluidResidual]")
{
   const int dim = 2; // templating is hard here because mesh constructors
   int num_state = dim + 2;
   adept::Stack diff_stack;

   // generate a 8 element mesh and build the finite-element space
   int num_edge = 2;
   Mesh smesh(Mesh::MakeCartesian2D(num_edge, num_edge, Element::TRIANGLE,
                                    true /* gen. edges */, 1.0, 1.0, true));
   ParMesh mesh(MPI_COMM_WORLD, smesh);
   int p = options["space-dis"]["degree"].get<int>();
   SBPCollection fec(p, dim);
   ParFiniteElementSpace fespace(&mesh, &fec, num_state, Ordering::byVDIM);

   // construct the residual
   FluidResidual res(options, fespace, diff_stack);
   int num_var = getSize(res);
   REQUIRE(num_var == 132);

   // evaluate the residual using a constant state
   Vector q(num_var);
   double mach = options["flow-param"]["mach"].get<double>();
   double aoa = options["flow-param"]["aoa"].get<double>();
   for (int i = 0; i < num_var/num_state; ++i)
   {
      getFreeStreamQ<double, dim>(mach, aoa, 0, 1, q.GetData()+num_state*i);
   }
   auto inputs = MachInputs({{"state", q}});
   Vector res_vec(num_var);
   evaluate(res, inputs, res_vec);

   // the res_vec should be zero, since we are differentiating a constant flux
   REQUIRE( res_vec.Norml2() == Approx(0.0).margin(1e-14) );

   // check the entropy calculation; grabs the first 4 vars from q.GetData() to
   // compute the entropy, and then scales by domain size (which is 1 unit sqrd)
   double total_ent = entropy<double, 2, false>(q.GetData());
   REQUIRE( calcEntropy(res, inputs) == Approx(total_ent) );

}