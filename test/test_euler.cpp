#include "catch.hpp"
#include "mfem.hpp"
#include "euler_fluxes.hpp"
#include "euler.hpp"

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


