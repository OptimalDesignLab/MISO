#include "catch.hpp"
#include "mfem.hpp"
#include "euler_fluxes.hpp"
#include "euler.hpp"

TEMPLATE_TEST_CASE_SIG( "Ismail Jacobian", "[Ismail]",
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


