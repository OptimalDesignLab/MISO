#include <random>

#include "catch.hpp"
#include "mfem.hpp"
#include "euler_fluxes.hpp"
#include "euler_integ.hpp"
#include "euler_test_data.hpp"

TEMPLATE_TEST_CASE_SIG("Euler flux jacobian", "[euler_flux_jac]",
                       ((int dim), dim), 1, 2, 3)
{
   using namespace euler_data;
   double delta = 1e-5;
   mfem::Vector q(dim + 2);
   mfem::Vector flux(dim + 2);
   mfem::Vector nrm(dim);
   q(0) = rho;
   q(dim + 1) = rhoe;
   for (int di = 0; di < dim; ++di)
   {
      q(di + 1) = rhou[di];
      nrm(di) = dir[di];
   }
   adept::Stack stack;
   mach::EulerIntegrator<dim> eulerinteg(stack);

   SECTION(" Euler flux jacobian w.r.t state is correct.")
   {
      //Create the perturbation vector
      mfem::Vector v(dim + 2);
      for (int i = 0; i < dim + 2; i++)
      {
         v[i] = vec_pert[i];
      }
      // Create some intermediate variables
      mfem::Vector q_plus(q), q_minus(q);
      mfem::Vector flux_plus(dim + 2), flux_minus(dim + 2);
      mfem::Vector jac_v(dim + 2);
      mfem::DenseMatrix flux_jac(dim + 2);

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
      jac_v_fd /= (2.0 * delta);
      mfem::Vector diff(jac_v);
      diff -= jac_v_fd;
      for (int i = 0; i < dim + 2; ++i)
      {
         REQUIRE(jac_v[i] == Approx(jac_v_fd[i]));
      }
   }

   SECTION(" Euler flux jacobian w.r.t direction is correct")
   {
      // Create the perturbation vector
      mfem::Vector v(dim);
      for (int i = 0; i < dim; i++)
      {
         v[i] = vec_pert[i];
      }
      // Create the intermediate variables
      mfem::Vector nrm_plus(nrm), nrm_minus(nrm);
      mfem::Vector flux_plus(dim + 2), flux_minus(dim + 2);
      mfem::Vector jac_v(dim + 2);
      mfem::DenseMatrix flux_jac(dim + 2, dim);

      eulerinteg.calcFluxJacDir(nrm, q, flux_jac);
      flux_jac.Mult(v, jac_v);

      nrm_plus.Add(delta, v);
      nrm_minus.Add(-delta, v);
      eulerinteg.calcFlux(nrm_plus, q, flux_plus);
      eulerinteg.calcFlux(nrm_minus, q, flux_minus);

      // compare the difference
      mfem::Vector jac_v_fd(flux_plus);
      jac_v_fd -= flux_minus;
      jac_v_fd /= (2.0 * delta);
      mfem::Vector diff(jac_v);
      diff -= jac_v_fd;
      for (int i = 0; i < dim; ++i)
      {
         REQUIRE(jac_v[i] == Approx(jac_v_fd[i]));
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

TEMPLATE_TEST_CASE_SIG("Ismail-Roe based on ent-vars Jacobian", "[Ismail-ent]",
                       ((int dim), dim), 1, 2, 3)
{
   using namespace euler_data;
   // copy the data into mfem vectors for convenience
   mfem::Vector qL(dim + 2);
   mfem::Vector qR(dim + 2);
   mfem::Vector wL(dim + 2);
   mfem::Vector wR(dim + 2);
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
   mach::calcEntropyVars<double, dim>(qL.GetData(), wL.GetData());
   mach::calcEntropyVars<double, dim>(qR.GetData(), wR.GetData());
   // create perturbation vector
   for (int di = 0; di < dim + 2; ++di)
   {
      v(di) = vec_pert[di];
   }
   // perturbed vectors
   mfem::Vector wL_plus(wL), wL_minus(wL);
   mfem::Vector wR_plus(wR), wR_minus(wR);
   adept::Stack diff_stack;
   mach::IsmailRoeIntegrator<dim,true> ismailinteg(diff_stack);
   // +ve perturbation
   wL_plus.Add(delta, v);
   wR_plus.Add(delta, v);
   // -ve perturbation
   wL_minus.Add(-delta, v);
   wR_minus.Add(-delta, v);
   for (int di = 0; di < dim; ++di)
   {
      DYNAMIC_SECTION("Ismail-Roe flux jacismailintegian is correct w.r.t left state ")
      {
         // get perturbed states flux vector
         ismailinteg.calcFlux(di, wL_plus, wR, flux_plus);
         ismailinteg.calcFlux(di, wL_minus, wR, flux_minus);
         // compute the jacobian
         ismailinteg.calcFluxJacStates(di, wL, wR, jacL, jacR);
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
         ismailinteg.calcFlux(di, wL, wR_plus, flux_plus);
         ismailinteg.calcFlux(di, wL, wR_minus, flux_minus);
         // compute the jacobian
         ismailinteg.calcFluxJacStates(di, wL, wR, jacL, jacR);
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

TEMPLATE_TEST_CASE_SIG("ApplyLPSScaling", "[LPSScaling]",
                       ((int dim), dim), 1, 2, 3)
{
   using namespace euler_data;
   double delta = 1e-5;
   int num_states = dim + 2;

   // construct state vec
   mfem::Vector q(num_states);
   q(0) = rho;
   q(dim + 1) = rhoe;
   for (int di = 0; di < dim; ++di)
   {
      q(di + 1) = rhou[di];
   }

   // Create the adjJ matrix, the AD stack, and the integrator
   mfem::DenseMatrix adjJ(adjJ_data, dim, dim);
   adept::Stack diff_stack;
   mach::EntStableLPSIntegrator<dim> lpsinteg(diff_stack);

   SECTION("Apply scaling jacobian w.r.t AdjJ is correct")
   {
      // random vector used in scaling product
      mfem::Vector vec(vec_pert, num_states);

      // calculate the jacobian w.r.t AdjJ
      mfem::DenseMatrix mat_vec_jac(num_states, dim * dim);
      lpsinteg.applyScalingJacAdjJ(adjJ, q, vec, mat_vec_jac);

      // matrix perturbation reshaped into vector
      mfem::Vector v_vec(vec_pert, dim * dim);
      mfem::Vector mat_vec_jac_v(num_states);
      mat_vec_jac.Mult(v_vec, mat_vec_jac_v);

      // perturb the transformation Jacobian adjugate by v_mat
      mfem::DenseMatrix v_mat(vec_pert, dim, dim);
      mfem::DenseMatrix adjJ_plus(adjJ), adjJ_minus(adjJ);
      adjJ_plus.Add(delta, v_mat);
      adjJ_minus.Add(-delta, v_mat);

      // calculate the jabobian with finite differences
      mfem::Vector mat_vec_plus(num_states), mat_vec_minus(num_states);
      lpsinteg.applyScaling(adjJ_plus, q, vec, mat_vec_plus);
      lpsinteg.applyScaling(adjJ_minus, q, vec, mat_vec_minus);
      mfem::Vector mat_vec_jac_v_fd(num_states);
      subtract(mat_vec_plus, mat_vec_minus, mat_vec_jac_v_fd);
      mat_vec_jac_v_fd /= 2.0 * delta;

      // compare
      for (int i = 0; i < num_states; ++i)
      {
         REQUIRE(mat_vec_jac_v(i) == Approx(mat_vec_jac_v_fd(i)));
      }
   }

   SECTION("Apply scaling jacobian w.r.t state is correct")
   {
      // random vector used in scaling product
      mfem::Vector vec(vec_pert, num_states);

      // calculate the jacobian w.r.t q
      mfem::DenseMatrix mat_vec_jac(num_states);
      mfem::Vector mat_vec_jac_v(num_states);
      lpsinteg.applyScalingJacState(adjJ, q, vec, mat_vec_jac);

      // loop over each state variable and check column of mat_vec_jac...
      for (int i = 0; i < num_states; i++)
      {
         mfem::Vector q_plus(q), q_minus(q);
         mfem::Vector mat_vec_plus(num_states), mat_vec_minus(num_states);
         q_plus(i) += delta;
         q_minus(i) -= delta;

         // get finite-difference approximation of ith column
         lpsinteg.applyScaling(adjJ, q_plus, vec, mat_vec_plus);
         lpsinteg.applyScaling(adjJ, q_minus, vec, mat_vec_minus);
         mfem::Vector mat_vec_fd(num_states);
         mat_vec_fd = 0.0;
         subtract(mat_vec_plus, mat_vec_minus, mat_vec_fd);
         mat_vec_fd /= 2.0 * delta;

         // compare with explicit Jacobian
         for (int j = 0; j < num_states; j++)
         {
            REQUIRE(mat_vec_jac(j, i) == Approx(mat_vec_fd(j)));
         }
      }
   }

   SECTION("Apply scaling jacobian w.r.t vec is correct")
   {
      // random vector used in scaling product
      mfem::Vector vec(vec_pert, num_states);

      // calculate the jacobian w.r.t q
      mfem::DenseMatrix mat_vec_jac(num_states);
      mfem::Vector mat_vec_jac_v(num_states);
      lpsinteg.applyScalingJacV(adjJ, q, mat_vec_jac);

      // loop over each state variable and check column of mat_vec_jac...
      for (int i = 0; i < num_states; i++)
      {
         mfem::Vector vec_plus(vec), vec_minus(vec);
         mfem::Vector mat_vec_plus(num_states), mat_vec_minus(num_states);
         vec_plus(i) += delta;
         vec_minus(i) -= delta;

         // get finite-difference approximation of ith column
         lpsinteg.applyScaling(adjJ, q, vec_plus, mat_vec_plus);
         lpsinteg.applyScaling(adjJ, q, vec_minus, mat_vec_minus);
         mfem::Vector mat_vec_fd(num_states);
         mat_vec_fd = 0.0;
         subtract(mat_vec_plus, mat_vec_minus, mat_vec_fd);
         mat_vec_fd /= 2.0 * delta;

         // compare with explicit Jacobian
         for (int j = 0; j < num_states; j++)
         {
            REQUIRE(mat_vec_jac(j, i) == Approx(mat_vec_fd(j)));
         }
      }
   }
}

TEST_CASE("Isentropic BC flux", "[IsentropricVortexBC]")
{
   const int dim = 2;
   using namespace euler_data;
   double delta = 1e-5;
   mfem::Vector nrm(dim);
   for (int di = 0; di < dim; ++di)
   {
      nrm(di) = dir[di];
   }
   mfem::Vector q(dim + 2);
   q(0) = rho;
   q(dim + 1) = rhoe;
   for (int di = 0; di < dim; ++di)
   {
      q(di + 1) = rhou[di];
   }
   // dummy const vector x for calcFlux - unused
   const mfem::Vector x(nrm);

   /// finite element or SBP operators
   std::unique_ptr<mfem::FiniteElementCollection> fec;
   adept::Stack diff_stack;
   //diff_stack.deactivate();
   const int max_degree = 4;
   for (int p = 1; p <= max_degree; p++)
   {
      fec.reset(new mfem::SBPCollection(1, dim));
      mach::IsentropicVortexBC<dim> isentropic_vortex(diff_stack, fec.get());

      DYNAMIC_SECTION("Jacobian of Isentropic Vortex BC flux w.r.t state is correct")
      {
         // create the perturbation vector
         mfem::Vector v(dim + 2);
         for (int i = 0; i < dim + 2; i++)
         {
            v(i) = vec_pert[i];
         }

         // get derivative information from AD functions and form product
         mfem::DenseMatrix jac_ad(dim + 2, dim + 2);
         mfem::Vector jac_v_ad(dim + 2);
         isentropic_vortex.calcFluxJacState(x, nrm, q, jac_ad);
         jac_ad.Mult(v, jac_v_ad);

         // FD approximation
         mfem::Vector q_plus(q);
         mfem::Vector q_minus(q);
         q_plus.Add(delta, v);
         q_minus.Add(-delta, v);

         mfem::Vector flux_plus(dim + 2);
         mfem::Vector flux_minus(dim + 2);
         isentropic_vortex.calcFlux(x, nrm, q_plus, flux_plus);
         isentropic_vortex.calcFlux(x, nrm, q_minus, flux_minus);

         // finite difference jacobian
         mfem::Vector jac_v_fd(dim + 2);
         subtract(flux_plus, flux_minus, jac_v_fd);
         jac_v_fd /= 2 * delta;

         // compare
         for (int i = 0; i < dim + 2; ++i)
         {
            REQUIRE(jac_v_ad(i) == Approx(jac_v_fd(i)).margin(1e-12));
         }
      }

      DYNAMIC_SECTION("Jacobian of Isentropic Vortex BC flux w.r.t dir is correct")
      {
         // create the perturbation vector
         mfem::Vector v(dim);
         for (int i = 0; i < dim; i++)
         {
            v(i) = vec_pert[i];
         }

         // get derivative information from AD functions and form product
         mfem::DenseMatrix jac_ad(dim + 2, dim);
         mfem::Vector jac_v_ad(dim + 2);
         isentropic_vortex.calcFluxJacDir(x, nrm, q, jac_ad);
         jac_ad.Mult(v, jac_v_ad);

         // FD approximation
         mfem::Vector nrm_plus(nrm);
         mfem::Vector nrm_minus(nrm);
         nrm_plus.Add(delta, v);
         nrm_minus.Add(-delta, v);

         mfem::Vector flux_plus(dim + 2);
         mfem::Vector flux_minus(dim + 2);
         isentropic_vortex.calcFlux(x, nrm_plus, q, flux_plus);
         isentropic_vortex.calcFlux(x, nrm_minus, q, flux_minus);

         // finite difference jacobian
         mfem::Vector jac_v_fd(dim + 2);
         subtract(flux_plus, flux_minus, jac_v_fd);
         jac_v_fd /= 2 * delta;

         // compare
         for (int i = 0; i < dim; ++i)
         {
            REQUIRE(jac_v_ad(i) == Approx(jac_v_fd(i)).margin(1e-12));
         }
      }
   }

}

// TODO: add dim = 1, 3 once 3d sbp operators implemented
TEMPLATE_TEST_CASE_SIG("Slip Wall Flux", "[Slip Wall]",
                       ((int dim), dim), 2)
{
   using namespace euler_data;
   // copy the data into mfem vectors for convenience
   double delta = 1e-5;
   mfem::Vector nrm(dim);
   for (int di = 0; di < dim; ++di)
   {
      nrm(di) = dir[di];
   }
   mfem::Vector q(dim + 2);
   q(0) = rho;
   q(dim + 1) = rhoe;
   for (int di = 0; di < dim; ++di)
   {
      q(di + 1) = rhou[di];
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

      DYNAMIC_SECTION("Jacobian of slip wall flux w.r.t state is correct")
      {
         // create the perturbation vector
         mfem::Vector v(dim + 2);
         for (int i = 0; i < dim + 2; i++)
         {
            v(i) = vec_pert[i];
         }

         // get derivative information from AD functions and form product
         mfem::DenseMatrix jac_ad(dim + 2, dim + 2);
         mfem::Vector jac_v_ad(dim + 2);
         slip_wall.calcFluxJacState(x, nrm, q, jac_ad);
         jac_ad.Mult(v, jac_v_ad);

         // FD approximation
         mfem::Vector q_plus(q);
         mfem::Vector q_minus(q);
         q_plus.Add(delta, v);
         q_minus.Add(-delta, v);

         mfem::Vector flux_plus(dim + 2);
         mfem::Vector flux_minus(dim + 2);
         slip_wall.calcFlux(x, nrm, q_plus, flux_plus);
         slip_wall.calcFlux(x, nrm, q_minus, flux_minus);

         // finite difference jacobian
         mfem::Vector jac_v_fd(dim + 2);
         subtract(flux_plus, flux_minus, jac_v_fd);
         jac_v_fd /= 2 * delta;

         // compare
         for (int i = 0; i < dim + 2; ++i)
         {
            REQUIRE(jac_v_ad(i) == Approx(jac_v_fd(i)).margin(1e-12));
         }
      }

      DYNAMIC_SECTION("Jacobian of slip wall flux w.r.t dir is correct")
      {
         // create the perturbation vector
         mfem::Vector v(dim);
         for (int i = 0; i < dim; i++)
         {
            v(i) = vec_pert[i];
         }

         // get derivative information from AD functions and form product
         mfem::DenseMatrix jac_ad(dim + 2, dim);
         mfem::Vector jac_v_ad(dim + 2);
         slip_wall.calcFluxJacDir(x, nrm, q, jac_ad);
         jac_ad.Mult(v, jac_v_ad);

         // FD approximation
         mfem::Vector nrm_plus(nrm);
         mfem::Vector nrm_minus(nrm);
         nrm_plus.Add(delta, v);
         nrm_minus.Add(-delta, v);

         mfem::Vector flux_plus(dim + 2);
         mfem::Vector flux_minus(dim + 2);
         slip_wall.calcFlux(x, nrm_plus, q, flux_plus);
         slip_wall.calcFlux(x, nrm_minus, q, flux_minus);

         // finite difference jacobian
         mfem::Vector jac_v_fd(dim + 2);
         subtract(flux_plus, flux_minus, jac_v_fd);
         jac_v_fd /= 2 * delta;

         // compare
         for (int i = 0; i < dim; ++i)
         {
            REQUIRE(jac_v_ad(i) == Approx(jac_v_fd(i)).margin(1e-12));
         }
      }
   }
}

TEMPLATE_TEST_CASE_SIG("Entropy variables Jacobian", "[lps integrator]",
                       ((int dim), dim), 1, 2, 3)
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
   SECTION("Entropy variables Jacobian is correct")
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
}

TEST_CASE("EulerIntegrator::AssembleElementGrad", "[EulerIntegrator]")
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 2; // templating is hard here because mesh constructors
   int num_state = dim + 2;
   adept::Stack diff_stack;
   double delta = 1e-5;

   // generate a 2 element mesh
   int num_edge = 1;
   std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, Element::TRIANGLE,
                                       true /* gen. edges */, 1.0, 1.0, true));
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
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
         Operator &Jac = res.GetGradient(q);
         GridFunction jac_v(fes.get());
         Jac.Mult(v, jac_v);

         // now compute the finite-difference approximation...
         GridFunction q_pert(q), r(fes.get()), jac_v_fd(fes.get());
         q_pert.Add(-delta, v);
         res.Mult(q_pert, r);
         q_pert.Add(2 * delta, v);
         res.Mult(q_pert, jac_v_fd);
         jac_v_fd -= r;
         jac_v_fd /= (2 * delta);

         for (int i = 0; i < jac_v.Size(); ++i)
         {
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)));
         }
      }
   }
}

TEST_CASE("SlipWallBC::AssembleFaceGrad", "[SlipWallBC]")
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 2; // templating is hard here because mesh constructors
   int num_state = dim + 2;
   adept::Stack diff_stack;
   double delta = 1e-5;

   // generate a 2 element mesh
   int num_edge = 1;
   std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, Element::TRIANGLE,
                                       true /* gen. edges */, 1.0, 1.0, true));
   for (int p = 1; p <= 1; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
             new SBPCollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             mesh.get(), fec.get(), num_state, Ordering::byVDIM));

         NonlinearForm res(fes.get());
         res.AddBdrFaceIntegrator(new mach::SlipWallBC<dim>(diff_stack,
                                                            fec.get()));

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselinePert<dim>);
         q.ProjectCoefficient(pert);

         // initialize the vector that the Jacobian multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(num_state, randState);
         v.ProjectCoefficient(v_rand);

         // evaluate the Jacobian and compute its product with v
         Operator &Jac = res.GetGradient(q);
         GridFunction jac_v(fes.get());
         Jac.Mult(v, jac_v);

         // now compute the finite-difference approximation...
         GridFunction q_pert(q), r(fes.get()), jac_v_fd(fes.get());
         q_pert.Add(-delta, v);
         res.Mult(q_pert, r);
         q_pert.Add(2 * delta, v);
         res.Mult(q_pert, jac_v_fd);
         jac_v_fd -= r;
         jac_v_fd /= (2 * delta);

         for (int i = 0; i < jac_v.Size(); ++i)
         {
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)));
         }
      }
   }
}

TEST_CASE("DyadicFluxIntegrator::AssembleElementGrad", "[DyadicIntegrator]")
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 2; // templating is hard here because mesh constructors
   int num_state = dim + 2;
   adept::Stack diff_stack;
   double delta = 1e-5;

   // generate a 2 element mesh
   int num_edge = 1;
   std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, Element::TRIANGLE,
                                       true /* gen. edges */, 1.0, 1.0, true));
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
             new SBPCollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             mesh.get(), fec.get(), num_state, Ordering::byVDIM));

         NonlinearForm res(fes.get());
         res.AddDomainIntegrator(new mach::IsmailRoeIntegrator<2>(diff_stack));

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselinePert<2>);
         q.ProjectCoefficient(pert);

         // initialize the vector that the Jacobian multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(num_state, randState);
         v.ProjectCoefficient(v_rand);

         // evaluate the Jacobian and compute its product with v
         Operator &Jac = res.GetGradient(q);
         GridFunction jac_v(fes.get());
         Jac.Mult(v, jac_v);

         // now compute the finite-difference approximation...
         GridFunction q_pert(q), r(fes.get()), jac_v_fd(fes.get());
         q_pert.Add(-delta, v);
         res.Mult(q_pert, r);
         q_pert.Add(2 * delta, v);
         res.Mult(q_pert, jac_v_fd);
         jac_v_fd -= r;
         jac_v_fd /= (2 * delta);

         for (int i = 0; i < jac_v.Size(); ++i)
         {
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)));
         }
      }
   }
}

// TODO: add dim = 1, 3 once 3d sbp operators implemented
TEST_CASE("InviscidFaceIntegrtor::AssembleFaceGrad", "[InterfaceIntegrator]")
{
   using namespace euler_data;
   using namespace mfem;
   const int dim = 2;
   double delta = 1e-5;
   int num_state = dim + 2;
   adept::Stack diff_stack;

   // generate a 2 element mesh
   int num_edge = 1;
   std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, Element::TRIANGLE,
                                       true /* gen. edges */, 1.0, 1.0, true));

   const int max_degree = 4;
   for (int p = 0; p < max_degree; p++)
   {
      DYNAMIC_SECTION("Jacobian of Interface flux w.r.t state is correct" << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
             new SBPCollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             mesh.get(), fec.get(), num_state, Ordering::byVDIM));

         NonlinearForm res(fes.get());
         res.AddInteriorFaceIntegrator(new mach::InterfaceIntegrator<dim>(diff_stack, fec.get()));

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselinePert<dim>);
         q.ProjectCoefficient(pert);

         // initialize the vector that the Jacobian multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(num_state, randState);
         v.ProjectCoefficient(v_rand);

         // evaluate the Jacobian and compute its product with v
         Operator &Jac = res.GetGradient(q);
         GridFunction jac_v(fes.get());
         Jac.Mult(v, jac_v);

         // now compute the finite-difference approximation...
         GridFunction q_pert(q), r(fes.get()), jac_v_fd(fes.get());
         q_pert.Add(-delta, v);
         res.Mult(q_pert, r);
         q_pert.Add(2 * delta, v);
         res.Mult(q_pert, jac_v_fd);
         jac_v_fd -= r;
         jac_v_fd /= (2 * delta);

         for (int i = 0; i < jac_v.Size(); ++i)
         {
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)));
         }
      }
   } // loop different order of elements
}

TEST_CASE("EntStableLPSIntegrator::AssembleElementGrad", "[LPSIntegrator]")
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 2; // templating is hard here because mesh constructors
   int num_state = dim + 2;
   adept::Stack diff_stack;
   double delta = 1e-5;

   // generate a 2 element mesh
   int num_edge = 1;
   std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, Element::TRIANGLE,
                                       true /* gen. edges */, 1.0, 1.0, true));
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
             new SBPCollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             mesh.get(), fec.get(), num_state, Ordering::byVDIM));

         NonlinearForm res(fes.get());
         res.AddDomainIntegrator(new mach::EntStableLPSIntegrator<2>(diff_stack));

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselinePert<2>);
         q.ProjectCoefficient(pert);

         // initialize the vector that the Jacobian multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(num_state, randState);
         v.ProjectCoefficient(v_rand);

         // evaluate the Jacobian and compute its product with v
         Operator &Jac = res.GetGradient(q);
         GridFunction jac_v(fes.get());
         Jac.Mult(v, jac_v);

         // now compute the finite-difference approximation...
         GridFunction q_pert(q), r(fes.get()), jac_v_fd(fes.get());
         q_pert.Add(-delta, v);
         res.Mult(q_pert, r);
         q_pert.Add(2 * delta, v);
         res.Mult(q_pert, jac_v_fd);
         jac_v_fd -= r;
         jac_v_fd /= (2 * delta);

         for (int i = 0; i < jac_v.Size(); ++i)
         {
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)));
         }
      }
   }
}

// append from test_euler_flux
// TEST_CASE( "Log-average is correct", "[log-avg]")
// {
//    using namespace euler_data;
//    REQUIRE( mach::logavg(rho, rho) == Approx(rho) );
//    REQUIRE( mach::logavg(rho, 2.0*rho) == Approx(1.422001977589051) );
// }

// TEMPLATE_TEST_CASE_SIG("Euler flux functions, etc, produce correct values",
//                        "[euler]", ((int dim), dim), 1, 2, 3)
// {
//    using namespace euler_data;
//    // copy the data into mfem vectors for convenience 
//    mfem::Vector q(dim+2);
//    mfem::Vector qR(dim+2);
//    mfem::Vector flux(dim+2);
//    mfem::Vector nrm(dim);
//    mfem::Vector work(dim+2);
//    q(0) = rho;
//    q(dim+1) = rhoe;
//    qR(0) = rho2;
//    qR(dim+1) = rhoe2;
//    for (int di = 0; di < dim; ++di)
//    {
//       q(di+1) = rhou[di];
//       qR(di+1) = rhou2[di];
//       nrm(di) = dir[di];
//    }

//    SECTION( "Pressure function is correct" )
//    {
//       REQUIRE( mach::pressure<double,dim>(q.GetData()) == 
//                Approx(press_check[dim-1]) );
//    }

//    SECTION( "Spectral radius of flux Jacobian is correct" )
//    {
//       REQUIRE(mach::calcSpectralRadius<double, dim>(nrm.GetData(), q.GetData()) == Approx(spect_check[dim - 1]));
//    }

//    SECTION( "Euler flux is correct" )
//    {
//       // Load the data to test the Euler flux; the pointer arithmetic in the 
//       // following constructor is to find the appropriate offset
//       int offset = div((dim+1)*(dim+2),2).quot - 3;
//       mfem::Vector flux_vec(flux_check + offset, dim + 2);
//       mach::calcEulerFlux<double,dim>(nrm.GetData(), q.GetData(), flux.GetData());
//       for (int i = 0; i < dim+2; ++i)
//       {
//          REQUIRE( flux(i) == Approx(flux_vec(i)) );
//       }
//    }  
   
//    // load the data to test the IR flux function into an mfem DenseMatrix
//    // TODO: I could not find an elegant way to do this
//    mfem::DenseMatrix fluxIR_check;
//    if (dim == 1)
//    {
//       fluxIR_check.Reset(fluxIR_1D_check, dim+2, dim);
//    }
//    else if (dim == 2)
//    {
//       fluxIR_check.Reset(fluxIR_2D_check, dim+2, dim);
//    }
//    else 
//    {
//       fluxIR_check.Reset(fluxIR_3D_check, dim+2, dim);
//    }
//    for (int di = 0; di < dim; ++di)
//    {
//       DYNAMIC_SECTION( "Ismail-Roe flux is correct in direction " << di )
//       {
//          mach::calcIsmailRoeFlux<double,dim>(di, q.GetData(), qR.GetData(),
//                                              flux.GetData());
//          for (int i = 0; i < dim+2; ++i)
//          {
//             REQUIRE( flux(i) == Approx(fluxIR_check(i,di)) );
//          }
//       }
//    }

//    SECTION( "Ismail-Roe face flux is correct in given direction" )
//    {
//       // get flux from function
//       mach::calcIsmailRoeFaceFlux<double, dim>(nrm.GetData(), q.GetData(),
//                                                qR.GetData(), flux.GetData());
//       for (int i = 0; i < dim+2; ++i)
//       {
//          // get true flux by scaling fluxIR_check data
//          double fluxIR = 0.0;
//          for (int di = 0; di < dim; ++di)
//          {
//             fluxIR += fluxIR_check(i,di)*nrm(di);
//          }
//          REQUIRE( flux(i) == Approx(fluxIR) );
//       }
//    }

//    SECTION( "Entropy variables are correct" )
//    {
//       // Load the data to test the entropy variables; the pointer arithmetic in
//       // the following constructor is to find the appropriate offset
//       int offset = div((dim+1)*(dim+2),2).quot - 3;
//       mfem::Vector entvar_vec(entvar_check + offset, dim + 2);
//       mach::calcEntropyVars<double,dim>(q.GetData(), qR.GetData());
//       for (int i = 0; i < dim+2; ++i)
//       {
//          REQUIRE( qR(i) == Approx(entvar_vec(i)) );
//       }
//    }

//    SECTION( "dQ/dW * vec product is correct" )
//    {
//       // Load the data to test the dQ/dW * vec product; the pointer arithmetic
//       // in the following constructor is to find the appropriate offset
//       int offset = div((dim+1)*(dim+2),2).quot - 3;
//       mfem::Vector dqdw_prod(dqdw_prod_check + offset, dim+2);
//       mach::calcdQdWProduct<double, dim>(q.GetData(), qR.GetData(),
//                                          flux.GetData());
//       for (int i = 0; i < dim+2; ++i)
//       {
//          REQUIRE( flux(i) == Approx(dqdw_prod(i)) );
//       }
//    }

//    SECTION( "Boundary flux is consistent" )
//    {
//       // Load the data to test the boundary flux; this only tests that the
//       // boundary flux agrees with the Euler flux when given the same states.
//       // The pointer arithmetic in the following constructor is to find the
//       // appropriate offset
//       int offset = div((dim+1)*(dim+2),2).quot - 3;
//       mfem::Vector flux_vec(flux_check + offset, dim + 2);
//       mach::calcBoundaryFlux<double,dim>(nrm.GetData(), q.GetData(),
//                                          q.GetData(), work.GetData(),
//                                          flux.GetData());
//       for (int i = 0; i < dim+2; ++i)
//       {
//          REQUIRE( flux(i) == Approx(flux_vec(i)) );
//       }
//    }

//    SECTION( "projectStateOntoWall removes normal component" )
//    {
//       // In this test case, the wall normal is set proportional to the momentum,
//       // so the momentum should come back as zero
//       mach::projectStateOntoWall<double,dim>(q.GetData()+1, q.GetData(),
//                                              flux.GetData());
//       REQUIRE( flux(0) == Approx(q(0)) );
//       for (int i = 0; i < dim; ++i)
//       {
//          REQUIRE( flux(i+1) == Approx(0.0).margin(abs_tol) );
//       }
//       REQUIRE( flux(dim+1) == Approx(q(dim+1)) );      
//    }

//    SECTION( "calcSlipWallFlux is correct" )
//    {
//       // As above with projectStateOntoWall, the wall normal is set
//       // proportional to the momentum, so the flux will be zero, except for the
//       // term flux[1:dim] = pressure*dir[0:dim-1]
//       mfem::Vector x(dim);
//       mach::calcSlipWallFlux<double,dim>(x.GetData(), q.GetData()+1,
//                                          q.GetData(), flux.GetData());
//       mach::projectStateOntoWall<double,dim>(q.GetData()+1, q.GetData(),
//                                              work.GetData());
//       double press = mach::pressure<double,dim>(work.GetData());
//       REQUIRE( flux(0) == Approx(0.0).margin(abs_tol) );
//       for (int i = 0; i < dim; ++i)
//       {
//          REQUIRE( flux(i+1) == Approx(press*q(i+1)) );
//       }
//       REQUIRE( flux(dim+1) == Approx(0.0).margin(abs_tol) );
//    }

// }

// TEST_CASE( "calcBoundaryFlux is correct", "[bndry-flux]")
// {
//    using namespace euler_data;
//    // copy the data into mfem vectors
//    mfem::Vector q(4);
//    mfem::Vector flux(4);
//    mfem::Vector qbnd(4);
//    mfem::Vector work(4);
//    mfem::Vector nrm(2);
//    q(0) = rho;
//    q(3) = rhoe;
//    qbnd(0) = rho2;
//    qbnd(3) = rhoe2;
//    for (int di = 0; di < 2; ++di)
//    {
//       q(di+1) = rhou[di];
//       qbnd(di+1) = rhou2[di];
//       nrm(di) = dir[di];
//    }
//    mach::calcBoundaryFlux<double,2>(nrm.GetData(), qbnd.GetData(), q.GetData(),
//                                     work.GetData(), flux.GetData());
//    for (int i = 0; i < 4; ++i)
//    {
//       REQUIRE( flux(i) == Approx(flux_bnd_check[i]) );
//    }
// }

// TEST_CASE( "calcIsentropicVortexFlux is correct", "[vortex-flux]")
// {
//    using namespace euler_data;
//    // copy the data into mfem vectors for convenience 
//    mfem::Vector q(4);
//    mfem::Vector flux(4);
//    mfem::Vector flux2(4);
//    mfem::Vector x(2);
//    mfem::Vector nrm(2);
//    // set location where we want to evaluate the flux
//    x(0) = cos(M_PI*0.25);
//    x(1) = sin(M_PI*0.25);
//    for (int di = 0; di < 2; ++di)
//    {
//       nrm(di) = dir[di];
//    }
//    mach::calcIsentropicVortexState<double>(x.GetData(), q.GetData());
//    mach::calcIsentropicVortexFlux<double>(x.GetData(), nrm.GetData(),
//                                           q.GetData(), flux.GetData());
//    mach::calcEulerFlux<double,2>(nrm.GetData(), q.GetData(), flux2.GetData());
//    for (int i = 0; i < 4; ++i)
//    {
//       REQUIRE( flux(i) == Approx(flux2(i)) );
//    }
// }
