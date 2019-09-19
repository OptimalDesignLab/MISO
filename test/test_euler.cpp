#include "catch.hpp"
#include "mfem.hpp"
#include "euler_fluxes.hpp"
#include "euler.hpp"

//#include "euler_test_data.hpp"

TEMPLATE_TEST_CASE_SIG( "Euler flux jacobian", "[euler_flux_jac]",
                        ((int dim),dim), 1, 2, 3 )
{
   #include "euler_test_data.hpp"
   mfem::Vector q(dim+2);
   mfem::Vector flux(dim+2);
   mfem::Vector nrm(dim);
   q(0) = rho;
   q(dim+1) = rhoe;
   for (int di = 0; di < dim; ++di)
   {
      q(di+1) = rhou[di];
      nrm(di) = dir[di];
   }

   SECTION(" Euler flux jacobian w.r.t state is correct.")
   {
      //Create the perturbation vector
      mfem::Vector v(dim+2);
      for(int i=0; i<dim+2;i++)
      {
         v[i] = 1e-5 * vec_pert[i];
      }

      adept::Stack stack;
      mach::EulerIntegrator<dim> eulerinteg(stack);

      // Create some intermediate variables
      mfem::Vector q_plus(q), q_minus(q);
      mfem::Vector flux_plus(dim+2), flux_minus(dim+2);
      mfem::Vector jac_v(dim+2);
      mfem::DenseMatrix flux_jac(dim+2);

      // calculate the flux jacobian
      eulerinteg.calcFluxJacState(nrm, q, flux_jac);
      flux_jac.Mult(v, jac_v);

      // calculate the plus and minus fluxes
      q_plus.Add(1.0, v);
      q_minus.Add(-1.0, v);
      eulerinteg.calcFlux(nrm, q_plus, flux_plus);
      eulerinteg.calcFlux(nrm, q_minus, flux_minus);

      // compare the difference
      mfem::Vector jac_v_fd(flux_plus);
      jac_v_fd -= flux_minus;
      jac_v_fd /= 2.0;
      mfem::Vector diff(jac_v);
      diff -= jac_v_fd;
      // REQUIRE( jac_v[1] == Approx(jac_v_fd[1]) );
      // REQUIRE( jac_v[2] == Approx(jac_v_fd[2]) );
      // REQUIRE( jac_v[3] == Approx(jac_v_fd[3]) );
      REQUIRE( diff.Norml2() == Approx(0.0).margin(abs_tol) ); 
   }

   SECTION(" Euler flux jacobian w.r.t direction is correct")
   {
      // Create the perturbation vector
      mfem::Vector v(dim);
      for(int i=0; i<dim;i++)
      {
         v[i] = 1e-5*vec_pert[i];
      }

      adept::Stack stack;
      mach::EulerIntegrator<dim> eulerinteg(stack);

      // Create the intermediate variables
      mfem::Vector nrm_plus(nrm), nrm_minus(nrm);
      mfem::Vector flux_plus(dim+2), flux_minus(dim+2);
      mfem::Vector jac_v(dim+2);
      mfem::DenseMatrix flux_jac(dim+2,dim);

      eulerinteg.calcFluxJacDir(nrm, q, flux_jac);
      flux_jac.Mult(v, jac_v);

      nrm_plus.Add(1.0,v);
      nrm_minus.Add(-1.0,v);
      eulerinteg.calcFlux(nrm_plus, q, flux_plus);
      eulerinteg.calcFlux(nrm_minus, q, flux_minus);

      // compare the difference
      mfem::Vector jac_v_fd(flux_plus);
      jac_v_fd -= flux_minus;
      jac_v_fd /= 2.0;
      mfem::Vector diff(jac_v);
      diff -= jac_v_fd;
      // REQUIRE( jac_v[1] == Approx(jac_v_fd[1]) );
      // REQUIRE( jac_v[2] == Approx(jac_v_fd[2]) );
      // REQUIRE( jac_v[3] == Approx(jac_v_fd[3]) );
      REQUIRE( diff.Norml2() == Approx(0.0).margin(abs_tol) ); 
   }

}

TEMPLATE_TEST_CASE_SIG( "Ismail Jacobian", "[Ismail]",
                        ((int dim), dim), 1, 2, 3 )
{
#if 0
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
    
#endif
   #include "euler_test_data.hpp"

   // copy the data into mfem vectors for convenience 
   mfem::Vector qL(dim+2);
   mfem::Vector qR(dim+2);
   mfem::Vector qL_r(dim+2);
   mfem::Vector qR_r(dim+2);
   mfem::Vector qL_l(dim+2);
   mfem::Vector qR_l(dim+2);
   mfem::Vector flux(dim+2);
   mfem::Vector flux_r(dim+2);
   mfem::Vector flux_l(dim+2);
   mfem::Vector v(dim+2);     
   mfem::Vector Jac_v(dim+2);  
   mfem::Vector Jac_v_fd(dim+2);  
   mfem::DenseMatrix jacL(dim+2,2*(dim+2));
   mfem::DenseMatrix jacR(dim+2,2*(dim+2));
   qL(0) = rho;
   qL(dim+1) = rhoe;
   qR(0) = rho2;
   qR(dim+1) = rhoe2;
   for (int di = 0; di < dim; ++di)
   {
      qL(di+1) = rhou[di];
      qR(di+1) = rhou2[di];
   }
   for(int di=0; di < dim+2; ++di)
   {
      v(di) =    di*1e-7;
      // +ve perturbation 
      qR_r(di) = qR(di) + v(di); 
      qL_r(di) = qL(di) + v(di);
      // -ve perturbation
      qR_l(di) = qR(di) - v(di);
      qL_l(di) = qL(di) - v(di);
   }

   adept::Stack diff_stack;
   mach::IsmailRoeIntegrator<dim> ob(diff_stack);
   for (int di = 0; di < dim; ++di)
   {
       DYNAMIC_SECTION( "Ismail-Roe flux jacobian is correct w.r.t left state ")
       {    
         ob.calcFlux(di, qL_r, qR,flux_r);
         ob.calcFlux(di, qL_l, qR,flux_l);
         ob.calcFluxJacStates(di,qL_l,qR,jacL,jacR);
         jacL.Mult(v,Jac_v);
         for (int i = 0; i < dim+2; ++i)
         {
            Jac_v_fd(i) = (flux_r(i)-flux_l(i))/(2.0);
         }
         REQUIRE(Jac_v_fd.Norml2() == Approx(Jac_v.Norml2()));
       }
       DYNAMIC_SECTION( "Ismail-Roe flux jacobian is correct w.r.t right state ")
       {    
         ob.calcFlux(di, qL, qR_r,flux_r);
         ob.calcFlux(di, qL, qR_l,flux_l);
         ob.calcFluxJacStates(di,qL,qR_l,jacL,jacR);
         jacR.Mult(v,Jac_v);
         for (int i = 0; i < dim+2; ++i)
         {
            Jac_v_fd(i) = (flux_r(i)-flux_l(i))/(2.0);
         }
         REQUIRE(Jac_v_fd.Norml2() == Approx(Jac_v.Norml2()));
        }
   }
}


