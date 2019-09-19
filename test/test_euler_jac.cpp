
#include "catch.hpp"
#include "mfem.hpp"
#include "euler_fluxes.hpp"
#include "euler.hpp"


TEMPLATE_TEST_CASE_SIG( "Euler flux functions, etc, produce correct values", "[euler]",
                        ((int dim), dim), 1, 2, 3 )
{
   using namespace std;
   /// Used for floating point checks when the benchmark value is zero
   const double abs_tol = std::numeric_limits<double>::epsilon()*100;

   // Define a random (but physical) state for the following tests
   const double rho = 0.9856566615165173;
   const double rhoe = 2.061597236955558;
   const double rhou[3] = {0.09595562550099601, -0.030658751626551423, -0.13471469906596886};

   // Define a second ("right") state for the tests
   const double rho2 = 0.8575252486261279;
   const double rhoe2 = 2.266357718749846;
   const double rhou2[3] = {0.020099729730903737, -0.2702434209304979, -0.004256150573245826};

   // Define a random direction vector
   const double dir[3] = {0.6541305612927484, -0.0016604759052086802, -0.21763228465741322};

   const double press_check[3] = {0.8227706007961364, 0.8225798733170867, 0.8188974449720051};
   const double spect_check[3] = {0.7708202616595441, 0.7707922224516813, 0.8369733021138251};
   
   // copy the data into mfem vectors for convenience
   mfem::Vector q(dim+2);
   mfem::Vector qR(dim+2);
   mfem::Vector flux(dim+2);
   mfem::Vector nrm(dim);
   mfem::Vector nrm_r(dim);
   mfem::Vector nrm_l(dim);
   q(0) = rho;
   q(dim+1) = rhoe;
   qR(0) = rho2;
   qR(dim+1) = rhoe2;
   for (int di = 0; di < dim; ++di)
   {
      q(di+1) = rhou[di];
      qR(di+1) = rhou2[di];
      nrm(di) = dir[di];
   }

   adept::Stack diff_stack;
   mach::EntStableLPSIntegrator<dim> ob(diff_stack);
   //just trying out dir for now
   SECTION( "Jacobian of Spectral radius of flux Jacobian is correct" )
   {
	  double delta = 1e-06;
	  // create Jacobian matrix for AD approach
	  mfem::DenseMatrix Jac_ad(1, dim);

	  // create vector to multiply Jacobian by
	  double v_dat[dim];
	  mfem::Vector v(dim);
	  for (int di = 0; di < dim; di++)
     {
        v_dat[di] = 1;
	     v(di) = delta*v_dat[di];
        // +v perturbation
        nrm_r(di) = nrm(di) + v(di);
        // -v perturbation
        nrm_l(di) = nrm(di) - v(di);
     }

	  // create vectors to store matrix-vector products
	  mfem::Vector Jac_v_ad(1);
	  mfem::Vector Jac_v_fd(1);
     mfem::Vector d_v_prod(dim);

	  // get derivative information from AD functions
	  ob.calcSpectralRadiusJacDir(&dir, &q, &Jac_ad);

	  Jac_ad.Mult(v, Jac_v_ad);
   
	  // FD approximation
	  Jac_v_fd = (calcSpectralRadius(nrm_r, q) -
				     calcSpectralRadius(nrm_l, q))/
				     (2*delta);

	  REQUIRE( Jac_v_ad(0) == Approx(Jac_v_fd(0)) );
   }

}
