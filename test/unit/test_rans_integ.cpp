#include <random>

#include "catch.hpp"
#include "mfem.hpp"
#include "euler_integ.hpp"
#include "rans_fluxes.hpp"
#include "rans_integ.hpp"
#include "euler_test_data.hpp"

TEMPLATE_TEST_CASE_SIG("SA Inviscid Test", "[sa_inviscid_test]",
                       ((int dim), dim), 1, 2, 3)
{
   using namespace euler_data;
   mfem::Vector qL(dim + 3);
   mfem::Vector qR(dim + 3);
   mfem::Vector flux1(dim + 2);
   mfem::Vector flux2(dim + 3);
   mfem::Vector nrm(dim);
   qL(0) = rho; qR(0) = rho*1.1;
   qL(dim + 1) = rhoe; qR(dim + 1) = rhoe*1.1;
   for (int di = 0; di < dim; ++di)
   {
      qL(di + 1) = rhou[di];
      qR(di + 1) = rhou[di]*1.1;
      nrm(di) = dir[di];
   }
   qL(dim + 2) = 5.0; qR(dim + 2) = 5.5;
   adept::Stack stack;
   mach::IsmailRoeIntegrator<dim> irinteg(stack);
   mach::SAInviscidIntegrator<dim> sainteg(stack);

    SECTION("Check if SA integrator matches the output for the conservative variables")
    {
        // calculate the flux 
        irinteg.calcFlux(nrm, qL, qR, flux);
        sainteg.calcFlux(nrm, qL, qR, flux);

        // check if euler variables are the same as before
        for(int n = 0; n < dim+2; n++)
        {
            REQUIRE(flux1(n) - flux2(n) == Approx(0.0));
        }
    }
}