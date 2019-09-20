#include "catch.hpp"
#include "mfem.hpp"
#include "euler_fluxes.hpp"
#include "euler.hpp"

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
   #include "euler_test_data.hpp"
   // copy the data into mfem vectors for convenience 
   mfem::Vector qL(dim+2);
   mfem::Vector qR(dim+2);
   mfem::Vector flux(dim+2);
   mfem::Vector flux_plus(dim+2);
   mfem::Vector flux_minus(dim+2);
   mfem::Vector v(dim+2);     
   mfem::Vector jac_v(dim+2);  
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
   // create perturbation vector
   for(int di=0; di < dim+2; ++di)
   {
      v(di) = 1e-07* vec_pert[di];
   }
   // perturbed vectors
   mfem::Vector qL_plus(qL), qL_minus(qL) ;
   mfem::Vector qR_plus(qR), qR_minus(qR);
   adept::Stack diff_stack;
   mach::IsmailRoeIntegrator<dim> ismailinteg(diff_stack);
   // +ve perturbation 
   qL_plus.Add(1.0, v);
   qR_plus.Add(1.0, v);
   // -ve perturbation
   qL_minus.Add(-1.0, v);
   qR_minus.Add(-1.0, v);
   for (int di = 0; di < dim; ++di)
   {
       DYNAMIC_SECTION( "Ismail-Roe flux jacismailintegian is correct w.r.t left state ")
       {  
         // get perturbed states flux vector
         ismailinteg.calcFlux(di, qL_plus, qR,flux_plus);
         ismailinteg.calcFlux(di, qL_minus, qR,flux_minus);
         // compute the jacobian
         ismailinteg.calcFluxJacStates(di,qL_minus,qR,jacL,jacR);
         jacL.Mult(v,jac_v);
         // finite difference jacobian
         mfem::Vector jac_v_fd(flux_plus);
         jac_v_fd -= flux_minus;
         jac_v_fd /= 2.0;
         // difference vector
         mfem::Vector diff(jac_v);
         diff -= jac_v_fd;
         // this gets passed at little higher perturbation too
         //REQUIRE(jac_v.Norml2() == Approx(jac_v_fd.Norml2()));
         REQUIRE(diff.Norml2() == Approx(0.0).margin(abs_tol) );
       }
       DYNAMIC_SECTION( "Ismail-Roe flux jacismailintegian is correct w.r.t right state ")
       {   
         // get perturbed states flux vector 
         ismailinteg.calcFlux(di, qL, qR_plus,flux_plus);
         ismailinteg.calcFlux(di, qL, qR_minus,flux_minus);
         // compute the jacobian
         ismailinteg.calcFluxJacStates(di,qL,qR_minus,jacL,jacR);
         jacR.Mult(v,jac_v);
         // finite difference jacobian
         mfem::Vector jac_v_fd(flux_plus);
         jac_v_fd -= flux_minus;
         jac_v_fd /= 2.0;
         // difference vector
         mfem::Vector diff(jac_v);
         diff -= jac_v_fd;
         //REQUIRE(jac_v.Norml2() == Approx(jac_v_fd.Norml2()));
         REQUIRE(diff.Norml2() == Approx(0.0).margin(abs_tol) );
        }
   }
}

