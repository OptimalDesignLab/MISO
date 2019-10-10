#include <random>

#include "catch.hpp"
#include "mfem.hpp"
#include "euler_fluxes.hpp"
#include "euler.hpp"
#include "euler_test_data.hpp"

TEMPLATE_TEST_CASE_SIG( "Euler flux jacobian", "[euler_flux_jac]",
                        ((int dim),dim), 1, 2, 3 )
{
   using namespace euler_data;
   double delta = 1e-5;
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
   adept::Stack stack;
   mach::EulerIntegrator<dim> eulerinteg(stack);

   SECTION(" Euler flux jacobian w.r.t state is correct.")
   {
      //Create the perturbation vector
      mfem::Vector v(dim+2);
      for(int i=0; i<dim+2;i++)
      {
         v[i] = vec_pert[i];
      }
      // Create some intermediate variables
      mfem::Vector q_plus(q), q_minus(q);
      mfem::Vector flux_plus(dim+2), flux_minus(dim+2);
      mfem::Vector jac_v(dim+2);
      mfem::DenseMatrix flux_jac(dim+2);

      // calculate the flux jacobian
      eulerinteg.calcFluxJacState(nrm, q, flux_jac);
      flux_jac.Mult(v, jac_v);

      // calculate the plus and minus fluxes
      q_plus.Add(delta, v);
      q_minus.Add(-delta, v);
      eulerinteg.calcFlux(nrm, q_plus, flux_plus);
      eulerinteg.calcFlux(nrm, q_minus, flux_minus);

      // compare the difference
      mfem::Vector jac_v_fd(flux_plus);
      jac_v_fd -= flux_minus;
      jac_v_fd /= (2.0*delta);
      mfem::Vector diff(jac_v);
      diff -= jac_v_fd;
      for (int i = 0; i < dim+2; ++i)
      {
         REQUIRE( jac_v[i] == Approx(jac_v_fd[i]) );
      }
   }

   SECTION(" Euler flux jacobian w.r.t direction is correct")
   {
      // Create the perturbation vector
      mfem::Vector v(dim);
      for(int i=0; i<dim;i++)
      {
         v[i] = vec_pert[i];
      }
      // Create the intermediate variables
      mfem::Vector nrm_plus(nrm), nrm_minus(nrm);
      mfem::Vector flux_plus(dim+2), flux_minus(dim+2);
      mfem::Vector jac_v(dim+2);
      mfem::DenseMatrix flux_jac(dim+2,dim);

      eulerinteg.calcFluxJacDir(nrm, q, flux_jac);
      flux_jac.Mult(v, jac_v);

      nrm_plus.Add(delta,v);
      nrm_minus.Add(-delta,v);
      eulerinteg.calcFlux(nrm_plus, q, flux_plus);
      eulerinteg.calcFlux(nrm_minus, q, flux_minus);

      // compare the difference
      mfem::Vector jac_v_fd(flux_plus);
      jac_v_fd -= flux_minus;
      jac_v_fd /= (2.0*delta);
      mfem::Vector diff(jac_v);
      diff -= jac_v_fd;
      for (int i = 0; i < dim; ++i)
      {
         REQUIRE( jac_v[i] == Approx(jac_v_fd[i]) );
      }
   }
}

TEMPLATE_TEST_CASE_SIG("Ismail-Roe Jacobian", "[Ismail]",
                       ((int dim), dim), 1, 2, 3)
{
   using namespace euler_data;
   // copy the data into mfem vectors for convenience
   mfem::Vector qL(dim + 2);
   mfem::Vector qR(dim + 2);
   mfem::Vector flux(dim + 2);
   mfem::Vector flux_plus(dim + 2);
   mfem::Vector flux_minus(dim + 2);
   mfem::Vector v(dim + 2);
   mfem::Vector jac_v(dim + 2);
   mfem::DenseMatrix jacL(dim + 2, 2 * (dim + 2));
   mfem::DenseMatrix jacR(dim + 2, 2 * (dim + 2));
   double delta = 1e-5;
   qL(0) = rho;
   qL(dim + 1) = rhoe;
   qR(0) = rho2;
   qR(dim + 1) = rhoe2;
   for (int di = 0; di < dim; ++di)
   {
      qL(di + 1) = rhou[di];
      qR(di + 1) = rhou2[di];
   }
   // create perturbation vector
   for (int di = 0; di < dim + 2; ++di)
   {
      v(di) = vec_pert[di];
   }
   // perturbed vectors
   mfem::Vector qL_plus(qL), qL_minus(qL);
   mfem::Vector qR_plus(qR), qR_minus(qR);
   adept::Stack diff_stack;
   mach::IsmailRoeIntegrator<dim> ismailinteg(diff_stack);
   // +ve perturbation
   qL_plus.Add(delta, v);
   qR_plus.Add(delta, v);
   // -ve perturbation
   qL_minus.Add(-delta, v);
   qR_minus.Add(-delta, v);
   for (int di = 0; di < dim; ++di)
   {
      DYNAMIC_SECTION("Ismail-Roe flux jacismailintegian is correct w.r.t left state ")
      {
         // get perturbed states flux vector
         ismailinteg.calcFlux(di, qL_plus, qR, flux_plus);
         ismailinteg.calcFlux(di, qL_minus, qR, flux_minus);
         // compute the jacobian
         ismailinteg.calcFluxJacStates(di, qL, qR, jacL, jacR);
         jacL.Mult(v, jac_v);
         // finite difference jacobian
         mfem::Vector jac_v_fd(flux_plus);
         jac_v_fd -= flux_minus;
         jac_v_fd /= 2.0 * delta;
         // compare each component of the matrix-vector products
         for (int i = 0; i < dim + 2; ++i)
         {
            REQUIRE(jac_v[i] == Approx(jac_v_fd[i]));
         }
      }
      DYNAMIC_SECTION("Ismail-Roe flux jacismailintegian is correct w.r.t right state ")
      {
         // get perturbed states flux vector
         ismailinteg.calcFlux(di, qL, qR_plus, flux_plus);
         ismailinteg.calcFlux(di, qL, qR_minus, flux_minus);
         // compute the jacobian
         ismailinteg.calcFluxJacStates(di, qL, qR, jacL, jacR);
         jacR.Mult(v, jac_v);
         // finite difference jacobian
         mfem::Vector jac_v_fd(flux_plus);
         jac_v_fd -= flux_minus;
         jac_v_fd /= 2.0 * delta;
         // compare each component of the matrix-vector products
         for (int i = 0; i < dim + 2; ++i)
         {
            REQUIRE(jac_v[i] == Approx(jac_v_fd[i]));
         }
      }
   }
}

TEMPLATE_TEST_CASE_SIG( "Spectral Radius", "[Spectral]",
                        ((int dim), dim), 1, 2, 3 )
{
   using namespace euler_data;
   // copy the data into mfem vectors for convenience
   double delta = 1e-5;
   mfem::Vector q(dim+2);
   mfem::Vector nrm(dim);
   q(0) = rho;
   q(dim+1) = rhoe;
   for (int di = 0; di < dim; ++di)
   {
      q(di+1) = rhou[di];
      nrm(di) = dir[di];
   }
   mfem::Vector q_plus(q);
   mfem::Vector q_minus(q);
   mfem::Vector nrm_plus(nrm);
   mfem::Vector nrm_minus(nrm);
   adept::Stack diff_stack;
   mach::EntStableLPSIntegrator<dim> lps_integ(diff_stack);

   SECTION( "Jacobian of Spectral radius w.r.t dir is correct" )
   {
      // create the perturbation vector
      mfem::Vector v(dim);
	   for (int i = 0; i < dim; i++)
      {
	      v(i) = vec_pert[i];
      }
      nrm_plus.Add(delta, v);
      nrm_minus.Add(-delta, v);

	   // get derivative information from AD functions and form product
	   mfem::DenseMatrix Jac_ad(1, dim);
	   mfem::Vector Jac_v_ad(1);
	   lps_integ.spectralRadiusJacDir(nrm, q, Jac_ad);
	   Jac_ad.Mult(v, Jac_v_ad);
   
	   // FD approximation
      mfem::Vector Jac_v_fd(1);
	   Jac_v_fd(0) = (lps_integ.spectralRadius(nrm_plus, q) -
		 		         lps_integ.spectralRadius(nrm_minus, q))/
				         (2*delta);

      // compare
      REQUIRE(Jac_v_ad(0) == Approx(Jac_v_fd(0)));
   }

   SECTION( "Jacobian of Spectral radius w.r.t state is correct" )
   {
      // create the perturbation vector
      mfem::Vector v(dim+2);
	   for (int i = 0; i < dim+2; i++)
      {
	      v(i) = vec_pert[i];
      }
      q_plus.Add(delta, v);
      q_minus.Add(-delta, v);

	   // get derivative information from AD functions and form product
	   mfem::DenseMatrix Jac_ad(1, dim+2);
	   mfem::Vector Jac_v_ad(1);
	   lps_integ.spectralRadiusJacState(nrm, q, Jac_ad);
	   Jac_ad.Mult(v, Jac_v_ad);
   
	   // FD approximation
	   mfem::Vector Jac_v_fd(1);
	   Jac_v_fd(0) = (lps_integ.spectralRadius(nrm, q_plus) -
		 		         lps_integ.spectralRadius(nrm, q_minus))/
				         (2*delta);

      // compare
      REQUIRE(Jac_v_ad(0) == Approx(Jac_v_fd(0)));
   }
}

// TODO: add dim = 1, 3 once 3d sbp operators implemented
TEMPLATE_TEST_CASE_SIG( "Slip Wall Flux", "[Slip Wall]",
                        ((int dim), dim), 2 )
{
   using namespace euler_data;
   // copy the data into mfem vectors for convenience
   double delta = 1e-5;
   mfem::Vector nrm(dim);
   for (int di = 0; di < dim; ++di)
   {
      nrm(di) = dir[di];
   }
   mfem::Vector q(dim+2);
   q(0) = rho;
   q(dim+1) = rhoe;
   for (int di = 0; di < dim; ++di)
   {
      q(di+1) = rhou[di];
   }
   
   // dummy const vector x for calcFlux - unused
   const mfem::Vector x(nrm);

   /// finite element or SBP operators
   std::unique_ptr<mfem::FiniteElementCollection> fec;
   adept::Stack diff_stack;

   const int max_degree = 4;
   for (int p = 1; p <= max_degree; ++p)
   {
      // Define the SBP elements and finite-element space
      fec.reset(new mfem::SBPCollection(p, dim));
      mach::SlipWallBC<dim> slip_wall(diff_stack, fec.get());

      DYNAMIC_SECTION( "Jacobian of slip wall flux w.r.t state is correct" )
      {
         // create the perturbation vector
         mfem::Vector v(dim+2);
         for (int i = 0; i < dim+2; i++)
         {
            v(i) = vec_pert[i];
         }

         // get derivative information from AD functions and form product
         mfem::DenseMatrix jac_ad(dim+2, dim+2);
         mfem::Vector jac_v_ad(dim+2);
         slip_wall.calcFluxJacState(x, nrm, q, jac_ad);
         jac_ad.Mult(v, jac_v_ad);
      
         // FD approximation
         mfem::Vector q_plus(q);
         mfem::Vector q_minus(q);
         q_plus.Add(delta, v);
         q_minus.Add(-delta, v);

         mfem::Vector flux_plus(dim+2);
         mfem::Vector flux_minus(dim+2);
         slip_wall.calcFlux(x, nrm, q_plus, flux_plus);
         slip_wall.calcFlux(x, nrm, q_minus, flux_minus);

         // finite difference jacobian
         mfem::Vector jac_v_fd(dim+2);
         subtract(flux_plus, flux_minus, jac_v_fd);
         jac_v_fd /= 2*delta;

         // compare
         for (int i = 0; i < dim+2; ++i)
         {
            REQUIRE( jac_v_ad(i) == Approx(jac_v_fd(i)).margin(1e-12) );
         }
      }

      DYNAMIC_SECTION( "Jacobian of slip wall flux w.r.t dir is correct" )
      {
         // create the perturbation vector
         mfem::Vector v(dim);
         for (int i = 0; i < dim; i++)
         {
            v(i) = vec_pert[i];
         }

         // get derivative information from AD functions and form product
         mfem::DenseMatrix jac_ad(dim+2, dim);
         mfem::Vector jac_v_ad(dim+2);
         slip_wall.calcFluxJacDir(x, nrm, q, jac_ad);
         jac_ad.Mult(v, jac_v_ad);
      
         // FD approximation
         mfem::Vector nrm_plus(nrm);
         mfem::Vector nrm_minus(nrm);
         nrm_plus.Add(delta, v);
         nrm_minus.Add(-delta, v);
         
         mfem::Vector flux_plus(dim+2);
         mfem::Vector flux_minus(dim+2);
         slip_wall.calcFlux(x, nrm_plus, q, flux_plus);
         slip_wall.calcFlux(x, nrm_minus, q, flux_minus);

         // finite difference jacobian
         mfem::Vector jac_v_fd(dim+2);
         subtract(flux_plus, flux_minus, jac_v_fd);
         jac_v_fd /= 2*delta;

         // compare
         for (int i = 0; i < dim; ++i)
         {
            REQUIRE( jac_v_ad(i) == Approx(jac_v_fd(i)).margin(1e-12) );
         }
      }
   }
}

TEMPLATE_TEST_CASE_SIG( "Entropy variables Jacobian", "[lps integrator]",
                        ((int dim), dim), 1, 2, 3 )
{
   using namespace euler_data;
   // copy the data into mfem vectors for convenience
   mfem::Vector q(dim + 2);
   mfem::Vector w(dim + 2);
   mfem::Vector w_plus(dim + 2);
   mfem::Vector w_minus(dim + 2);
   mfem::Vector v(dim + 2);
   mfem::Vector dwdu_v(dim + 2);
   mfem::DenseMatrix dwdu(dim + 2);
   double delta = 1e-5;
   q(0) = rho;
   q(dim + 1) = rhoe;
   for (int di = 0; di < dim; ++di)
   {
      q(di + 1) = rhou[di];
   }
   // create perturbation vector
   for (int di = 0; di < dim + 2; ++di)
   {
      v(di) = vec_pert[di];
   }
   // perturbed vectors
   mfem::Vector q_plus(q), q_minus(q);
   adept::Stack diff_stack;
   mach::EntStableLPSIntegrator<dim> lpsinteg(diff_stack);
   // +ve perturbation
   q_plus.Add(delta, v);
    // -ve perturbation
   q_minus.Add(-delta, v);  
   SECTION( "Entropy variables Jacobian is correct" )
   {
      // get perturbed states entropy variables vector
      lpsinteg.convertVars(q_plus, w_plus);
      lpsinteg.convertVars(q_minus, w_minus);
      // compute the jacobian
      lpsinteg.convertVarsJacState(q, dwdu);
      dwdu.Mult(v, dwdu_v);
      // finite difference jacobian
      mfem::Vector dwdu_v_fd(w_plus);
      dwdu_v_fd -= w_minus;
      dwdu_v_fd /= 2.0 * delta;
      // compare each component of the matrix-vector products
      for (int i = 0; i < dim + 2; ++i)
      {
        REQUIRE(dwdu_v[i] == Approx(dwdu_v_fd[i]));
      }
   }

   SECTION( "Apply scaling jacobian w.r.t state is correct" )
   {
      // Create the adjJ matrix
      mfem::DenseMatrix adjJ(dim);
      for(int i = 0; i < dim; i++)
      {
         for(int j = 0; j < dim; j++)
         {
            adjJ(i,j) = adjJ_data[j*3+i];
         }
      }
      // Create the perturbation vector
      mfem::Vector v(dim+2);
      mfem::Vector vec(dim+2);
      for(int i=0; i<dim+2;i++)
      {
         v[i] = vec_pert[i];
         //vec[i] = 1.0*(i+1);
         //vec[i] = vec_pert[4-i];
      }
      vec = 0.0;
      vec(0) = 1.0;

      // calculate the jacobian w.r.t q
      mfem::DenseMatrix mat_vec_jac(dim+2);
      mfem::Vector mat_vec_jac_v(dim+2);
      lpsinteg.applyScalingJacState(adjJ, q, vec, mat_vec_jac);
      //mat_vec_jac.Mult(v, mat_vec_jac_v);
      std::cout << "\n\nCheck the jac:\n";
      mat_vec_jac.Print();

      // Calculate the jacobian w.r.t. q using finite difference
      // mfem::Vector q_plus(q), q_minus(q);
      // mfem::Vector mat_vec_plus(dim+2), mat_vec_minus(dim+2);
      std::cout << "\n\nCheck the data again.\n";
      std::cout << "adjJ: ";
      for(int i = 0; i< dim; i ++)
      {
         for(int j = 0; j<dim; j++)
         {
            std::cout << adjJ(j,i) << ' ';
         }
      }
         std::cout << "\nq: ";
      for(int i = 0; i<q.Size(); i ++)
      {
         std::cout << q(i) << ' ';
      }
      std::cout << "\nvec: ";
      for(int i = 0; i<vec.Size(); i ++)
      {
         std::cout << vec(i) << ' ';
      }

      for(int i = 0; i < dim+2; i++)
      {
         mfem::Vector q_plus(q), q_minus(q);
         mfem::Vector mat_vec_plus(dim+2), mat_vec_minus(dim+2);
         q_plus(i) += delta;
         q_minus(i) -= delta;

         lpsinteg.applyScaling(adjJ, q_plus, vec, mat_vec_plus);
         lpsinteg.applyScaling(adjJ, q_minus, vec, mat_vec_minus);

         mfem::Vector mat_vec_diff(dim+2);
         mat_vec_diff = 0.0;
         subtract(mat_vec_plus, mat_vec_minus, mat_vec_diff);
         //mat_vec_plus -= mat_vec_minus;
         std::cout << "\ndelta is " << delta;
         mat_vec_diff /= 2.0*delta;

         std::cout << "\nCheck the column of jac: ";
         for(int j = 0; j < dim+2; j++)
         {
            std::cout << mat_vec_diff(j) << ' ';
         }
         std::cout << '\n';
      }
      
      // q_plus.Add(delta, v);
      // q_minus.Add(-delta, v);

      // std::cout << "\n\nCheck the data again.\n";
      // std::cout << "adjJ: ";
      // for(int i = 0; i< dim; i ++)
      // {
      //    for(int j = 0; j<dim; j++)
      //    {
      //       std::cout << adjJ(j,i) << ' ';
      //    }
      // }
      //    std::cout << "\nq: ";
      // for(int i = 0; i<q.Size(); i ++)
      // {
      //    std::cout << q(i) << ' ';
      // }
      // std::cout << "\nvec: ";
      // for(int i = 0; i<vec.Size(); i ++)
      // {
      //    std::cout << vec(i) << ' ';
      // }
      // lpsinteg.applyScaling(adjJ, q_plus, vec, mat_vec_plus);
      // lpsinteg.applyScaling(adjJ, q_minus, vec, mat_vec_minus);

      // mfem::Vector mat_vec_diff(mat_vec_plus);
      // mat_vec_diff -= mat_vec_minus;
      // mat_vec_diff /= (2.0 * delta);
      // //compare the element wise result
      // std::cout << "\nmat_vec_jac_v is: ";
      // for(int i = 0; i < dim+2;i ++)
      // {
      //    std::cout << mat_vec_jac_v(i) << ' ';
      // }
      // std::cout << "\nmat_vec_diff is: ";
      // for(int i = 0; i < dim+2;i ++)
      // {
      //    std::cout << mat_vec_diff(i) << ' ';
      // }
      // std::cout << '\n';
      // for(int i = 0; i < dim+2; i++)
      // {
      //    REQUIRE( mat_vec_diff(i) == Approx( mat_vec_jac_v(i) ));
      // }
   }
}

TEST_CASE("EulerIntegrator::AssembleElementGrad", "[EulerIntegrator]")
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 2;  // templating is hard here because mesh constructors
   int num_state = dim + 2;
   static adept::Stack diff_stack;
   double delta = 1e-5;

   // generate a 2 element mesh
   int num_edge = 1;
   std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, Element::TRIANGLE,
                              true /* gen. edges */, 1.0, 1.0, true));
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         std::unique_ptr<FiniteElementCollection> fec(
            new SBPCollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
            mesh.get(), fec.get(), num_state, Ordering::byVDIM));
                         
         NonlinearForm res(fes.get());
         res.AddDomainIntegrator(new mach::EulerIntegrator<2>(diff_stack));

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselinePert<2>);
         q.ProjectCoefficient(pert);

         // initialize the vector that the Jacobian multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(num_state, randState);
         v.ProjectCoefficient(v_rand);

         // evaluate the Jacobian and compute its product with v
         Operator& Jac = res.GetGradient(q);
         GridFunction jac_v(fes.get());
         Jac.Mult(v, jac_v);

         // now compute the finite-difference approximation...
         GridFunction q_pert(q), r(fes.get()), jac_v_fd(fes.get());
         q_pert.Add(-delta, v);
         res.Mult(q_pert, r);
         q_pert.Add(2*delta, v);
         res.Mult(q_pert, jac_v_fd);
         jac_v_fd -= r;
         jac_v_fd /= (2*delta);

         for (int i = 0; i < jac_v.Size(); ++i)
         {
            REQUIRE( jac_v(i) == Approx(jac_v_fd(i)) );
         }
      }
   }
}