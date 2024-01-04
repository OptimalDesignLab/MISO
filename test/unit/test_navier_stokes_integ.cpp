#include <random>
#include "catch.hpp"
#include "mfem.hpp"
#include "euler_test_data.hpp"
#include "navier_stokes_integ.hpp"
#include "navier_stokes_fluxes.hpp"

using namespace std;

TEMPLATE_TEST_CASE_SIG("ESViscousIntegrator::applyScalingJacState", "[ESViscousIntegrator]",
                       ((int dim), dim), 1, 2, 3)
{
   using namespace euler_data;
   double delta = 1e-5;
   int num_states = dim + 2;
   double Re_num = 1;
   double Pr_num = 1;
   double mu = -1;
   // construct state vec
   mfem::Vector q(num_states);
   mfem::Vector x(num_states - 2);
   q(0) = rho;
   q(dim + 1) = rhoe;
   for (int di = 0; di < dim; ++di)
   {
      q(di + 1) = rhou[di];
      x(di) = 0;
   }
   // spatial derivatives of entropy variables
   mfem::DenseMatrix delw(delw_data, dim + 2, dim);
   // Create the AD stack, and the integrator
   adept::Stack diff_stack;
   mach::ESViscousIntegrator<dim> esviscousinteg(diff_stack, Re_num, Pr_num, mu);
   // calculate the jacobian w.r.t q
   for (int di = 0; di < dim; ++di)
   {
      DYNAMIC_SECTION("Apply scaling jacobian w.r.t state is correct for di = " << di)
      {
         mfem::DenseMatrix mat_vec_jac(num_states);
         esviscousinteg.applyScalingJacState(di, x, q, delw, mat_vec_jac);
         //mat_vec_jac.Print();
         // loop over each state variable and check column of mat_vec_jac...
         for (int i = 0; i < num_states; ++i)
         {
            mfem::Vector q_plus(q), q_minus(q);
            mfem::Vector mat_vec_plus(num_states), mat_vec_minus(num_states);
            q_plus(i) += delta;
            q_minus(i) -= delta;
            esviscousinteg.applyScaling(di, x, q_plus, delw, mat_vec_plus);
            esviscousinteg.applyScaling(di, x, q_minus, delw, mat_vec_minus);
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
}

TEMPLATE_TEST_CASE_SIG("ESViscousIntegrator::applyScalingJacDw", "[ESViscousIntegrator]",
                       ((int dim), dim), 1, 2, 3)
{
   using namespace euler_data;
   double delta = 1e-5;
   int num_states = dim + 2;
   double Re_num = 1;
   double Pr_num = 1;
   double mu = -1;
   // construct state vec
   mfem::Vector q(num_states);
   mfem::Vector v(num_states);
   mfem::Vector x(num_states - 2);
   q(0) = rho;
   q(dim + 1) = rhoe;
   for (int di = 0; di < dim; ++di)
   {
      q(di + 1) = rhou[di];
      x(di) = 0;
   }
   for (int di = 0; di < dim + 2; ++di)
   {
      v(di) = vec_pert[di];
   }
   // spatial derivatives of entropy variables
   mfem::DenseMatrix delw(delw_data, dim + 2, dim);
   // Create the AD stack, and the integrator
   adept::Stack diff_stack;
   mach::ESViscousIntegrator<dim> esviscousinteg(diff_stack, Re_num, Pr_num, mu);
   // calculate the jacobian w.r.t q
   for (int di = 0; di < dim; ++di)
   {
      DYNAMIC_SECTION("Apply scaling jacobian w.r.t Dw is correct for di = " << di)
      {
         std::vector<mfem::DenseMatrix> mat_vec_jac(dim);
         for (int d = 0; d < dim; ++d)
         {
            mat_vec_jac[d].SetSize(num_states);
         }
         // get jacobian from adept
         esviscousinteg.applyScalingJacDw(di, x, q, delw, mat_vec_jac);
         // matrix perturbation reshaped into vector
         mfem::Vector v_vec(vec_pert, num_states);
         for (int d = 0; d < dim; ++d)
         {
            mfem::Vector mat_vec_jac_v(num_states);
            mat_vec_jac[d].Mult(v_vec, mat_vec_jac_v);
            // perturb one column of delw everytime
            mfem::DenseMatrix delw_plus(delw), delw_minus(delw);
            for (int s = 0; s < num_states; ++s)
            {
               delw_plus.GetColumn(d)[s] += v(s) * delta;
               delw_minus.GetColumn(d)[s] -= v(s) * delta;
            }
            // get finite-difference jacobian
            mfem::Vector mat_vec_plus(num_states), mat_vec_minus(num_states);
            esviscousinteg.applyScaling(di, x, q, delw_plus, mat_vec_plus);
            esviscousinteg.applyScaling(di, x, q, delw_minus, mat_vec_minus);
            mfem::Vector mat_vec_jac_v_fd(num_states);
            subtract(mat_vec_plus, mat_vec_minus, mat_vec_jac_v_fd);
            mat_vec_jac_v_fd /= 2.0 * delta;
            // compare with explicit Jacobian
            for (int s = 0; s < num_states; s++)
            {
               REQUIRE(mat_vec_jac_v(s) == Approx(mat_vec_jac_v_fd(s)));
            }
         } // d loop
      }    // section
   }       // di loop
} // test case

TEMPLATE_TEST_CASE_SIG("Noslip Jacobian", "[NoSlipAdiabaticWallBC]",
                       ((int dim), dim), 2, 3)
{
   using namespace euler_data;
   double delta = 1e-5;
   int num_states = dim + 2;
   double Re_num = 1;
   double Pr_num = 1;
   double mu = -1;
   double jac = 1;
   // construct state vec
   mfem::Vector q(num_states);
   mfem::Vector q_ref(num_states);
   mfem::Vector nrm(dim);
   for (int di = 0; di < dim; ++di)
   {
      nrm(di) = dir[di];
   }
   q(0) = rho;
   q(dim + 1) = rhoe;
   q_ref(0) = rhou[0];
   q_ref(dim + 1) = rhou[dim - 1];
   for (int di = 0; di < dim; ++di)
   {
      q(di + 1) = rhou[di];
      q_ref(di + 1) = 0.0;
   }
   // random delw matrix
   // spatial derivatives of entropy variables
   mfem::DenseMatrix delw(delw_data, dim + 2, dim);
   // dummy const vector x for calcFlux - unused
   const mfem::Vector x(nrm);
   /// finite element or SBP operators
   std::unique_ptr<mfem::FiniteElementCollection> fec;
   // Create the AD stack, and the integrator
   adept::Stack diff_stack;

   for (int p = 0; p <= 1; ++p)
   {
      fec.reset(new mfem::SBPCollection(p, dim));
      mach::NoSlipAdiabaticWallBC<dim> noslipadiabatic(diff_stack, fec.get(),
                                                      Re_num, Pr_num, q_ref, mu);

      DYNAMIC_SECTION("jacobian of no slip adiabatic wall w.r.t state failed for degree p = " << p)
      {
         mfem::DenseMatrix mat_vec_jac(num_states);
         noslipadiabatic.calcFluxJacState(x, nrm, jac, q, delw, mat_vec_jac);
         //mat_vec_jac.Print();
         // loop over each state variable and check column of mat_vec_jac...
         for (int i = 0; i < num_states; ++i)
         {
            mfem::Vector q_plus(q), q_minus(q);
            mfem::Vector mat_vec_plus(num_states), mat_vec_minus(num_states);
            q_plus(i) += delta;
            q_minus(i) -= delta;
            noslipadiabatic.calcFlux(x, nrm, jac, q_plus, delw, mat_vec_plus);
            noslipadiabatic.calcFlux(x, nrm, jac, q_minus, delw, mat_vec_minus);
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
      DYNAMIC_SECTION("jacobian of no slip dual flux w.r.t state is incorrect for degree p = " << p)
      {
         std::vector<mfem::DenseMatrix> mat_vec_jac(dim);
         for (int d = 0; d < dim; ++d)
         {
            mat_vec_jac[d].SetSize(num_states);
         }
         noslipadiabatic.calcFluxDvJacState(x, nrm, q, mat_vec_jac);
         //mat_vec_jac.Print();
         // loop over each state variable and check column of mat_vec_jac...
         for (int i = 0; i < num_states; ++i)
         {
            mfem::Vector q_plus(q), q_minus(q);
            mfem::DenseMatrix flux_mat_plus(num_states, dim);
            mfem::DenseMatrix flux_mat_minus(num_states, dim);
            q_plus(i) += delta;
            q_minus(i) -= delta;
            noslipadiabatic.calcFluxDv(x, nrm, q_plus, flux_mat_plus);
            noslipadiabatic.calcFluxDv(x, nrm, q_minus, flux_mat_minus);
            mfem::DenseMatrix flux_mat_fd(num_states, dim);
            flux_mat_fd = 0.0;
            mfem::Add(flux_mat_plus, flux_mat_minus, -1.0, flux_mat_fd);
            flux_mat_fd *= 1.0/(2.0 * delta);
            // compare with explicit Jacobian
            for (int d = 0; d < dim; ++d)
            {
               for (int j = 0; j < num_states; j++)
               {
                  REQUIRE(mat_vec_jac[d](j, i) == Approx(flux_mat_fd(j, d)));
               }
            }
         }
      }      
   }

}

TEMPLATE_TEST_CASE_SIG("Noslip Jacobian w.r.t Dw ", "[NoSlipAdiabaticWallBC]",
                       ((int dim), dim), 2, 3)
{
   using namespace euler_data;
   double delta = 1e-5;
   int num_states = dim + 2;
   double Re_num = 1;
   double Pr_num = 1;
   double mu = -1;
   double jac = 1;
   // construct state vec
   mfem::Vector q(num_states);
   mfem::Vector q_ref(num_states);
   mfem::Vector nrm(dim);
   mfem::Vector v(num_states);
   for (int di = 0; di < dim; ++di)
   {
      nrm(di) = dir[di];
   }
   q(0) = rho;
   q(dim + 1) = rhoe;
   q_ref(0) = rhou[0];
   q_ref(dim + 1) = rhou[dim - 1];
   for (int di = 0; di < dim; ++di)
   {
      q(di + 1) = rhou[di];
      q_ref(di + 1) = 0.0;
   }
   for (int di = 0; di < dim + 2; ++di)
   {
      v(di) = vec_pert[di];
   }
   // spatial derivatives of entropy variables
   mfem::DenseMatrix delw(delw_data, dim + 2, dim);
   // dummy const vector x for calcFlux - unused
   const mfem::Vector x(nrm);
   /// finite element or SBP operators
   std::unique_ptr<mfem::FiniteElementCollection> fec;
   // Create the AD stack, and the integrator
   adept::Stack diff_stack;
   for (int p = 0; p <= 1; ++p)
   {
      fec.reset(new mfem::SBPCollection(p, dim));
      mach::NoSlipAdiabaticWallBC<dim> noslipadiabatic(diff_stack, fec.get(),
                                                      Re_num, Pr_num, q_ref, mu);

      DYNAMIC_SECTION("jacobian of no slip adiabatic wall w.r.t Dw failed for degree p = " << p)
      {
         std::vector<mfem::DenseMatrix> mat_vec_jac(dim);
         for (int d = 0; d < dim; ++d)
         {
            mat_vec_jac[d].SetSize(num_states);
         }
         noslipadiabatic.calcFluxJacDw(x, nrm, jac, q, delw, mat_vec_jac);
         // loop over each state variable and check column of mat_vec_jac...
         // matrix perturbation reshaped into vector
         mfem::Vector v_vec(vec_pert, num_states);
         for (int d = 0; d < dim; ++d)
         {
            mfem::Vector mat_vec_jac_v(num_states);
            mat_vec_jac[d].Mult(v_vec, mat_vec_jac_v);
            // perturb one column of delw everytime
            mfem::DenseMatrix delw_plus(delw), delw_minus(delw);
            for (int s = 0; s < num_states; ++s)
            {
               delw_plus.GetColumn(d)[s] += v(s) * delta;
               delw_minus.GetColumn(d)[s] -= v(s) * delta;
            }
            mfem::Vector mat_vec_plus(num_states), mat_vec_minus(num_states);
            noslipadiabatic.calcFlux(x, nrm, jac, q, delw_plus, mat_vec_plus);
            noslipadiabatic.calcFlux(x, nrm, jac, q, delw_minus, mat_vec_minus);
            mfem::Vector mat_vec_fd(num_states);
            mat_vec_fd = 0.0;
            subtract(mat_vec_plus, mat_vec_minus, mat_vec_fd);
            mat_vec_fd /= 2.0 * delta;
            // compare with explicit Jacobian
            for (int s = 0; s < num_states; ++s)
            {
               REQUIRE(mat_vec_jac_v(s) == Approx(mat_vec_fd(s)));
            }
         } // d loop
      }    // section      
   }
} // test case

TEMPLATE_TEST_CASE_SIG("Slip wall Jacobian states", "[ViscousSlipWallBC]",
                       ((int dim), dim), 2, 3)
{
   using namespace euler_data;
   double delta = 1e-5;
   int num_states = dim + 2;
   double Re_num = 1;
   double Pr_num = 1;
   double mu = 1;
   double jac = 1;
   // construct state vec
   mfem::Vector q(num_states);
   mfem::Vector q_ref(num_states);
   mfem::Vector nrm(dim);
   for (int di = 0; di < dim; ++di)
   {
      nrm(di) = dir[di];
   }
   q(0) = rho;
   q(dim + 1) = rhoe;
   q_ref(0) = rhou[0];
   q_ref(dim + 1) = rhou[dim - 1];
   for (int di = 0; di < dim; ++di)
   {
      q(di + 1) = rhou[di];
      q_ref(di + 1) = 0.0;
   }
   // random delw matrix
   // spatial derivatives of entropy variables
   mfem::DenseMatrix delw(delw_data, dim + 2, dim);
   // dummy const vector x for calcFlux - unused
   const mfem::Vector x(nrm);
   /// finite element or SBP operators
   std::unique_ptr<mfem::FiniteElementCollection> fec;
   // Create the AD stack, and the integrator
   adept::Stack diff_stack;
   for (int p = 0; p <= 1; p++)
   {
      fec.reset(new mfem::SBPCollection(p, dim));
      mach::ViscousSlipWallBC<dim> viscousslipwall(diff_stack, fec.get(), Re_num,
                                                   Pr_num, mu);

      DYNAMIC_SECTION("jacobian of Viscous Slip Wall BC w.r.t state failed for degree p = " << p)
      {
         mfem::DenseMatrix mat_vec_jac(num_states);
         viscousslipwall.calcFluxJacState(x, nrm, jac, q, delw, mat_vec_jac);
         //mat_vec_jac.Print();
         // loop over each state variable and check column of mat_vec_jac...
         for (int i = 0; i < num_states; ++i)
         {
            mfem::Vector q_plus(q), q_minus(q);
            mfem::Vector mat_vec_plus(num_states), mat_vec_minus(num_states);
            q_plus(i) += delta;
            q_minus(i) -= delta;
            viscousslipwall.calcFlux(x, nrm, jac, q_plus, delw, mat_vec_plus);
            viscousslipwall.calcFlux(x, nrm, jac, q_minus, delw, mat_vec_minus);
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
}

TEMPLATE_TEST_CASE_SIG("Viscous inflow Jacobian", "[ViscousInflowBC]",
                       ((int dim), dim), 2, 3)
{
   using namespace euler_data;
   double delta = 1e-5;
   int num_states = dim + 2;
   double Re_num = 1;
   double Pr_num = 1;
   double mu = 1;
   double jac = 1;
   // construct state vec
   mfem::Vector q(num_states);
   mfem::Vector q_in(num_states);
   mfem::Vector nrm(dim);
   for (int di = 0; di < dim; ++di)
   {
      nrm(di) = dir[di];
   }
   q(0) = rho;
   q(dim + 1) = rhoe;
   q_in(0) = rho2;        //1.2;
   q_in(dim + 1) = rhoe2; //1;
   for (int di = 0; di < dim; ++di)
   {
      q(di + 1) = rhou[di];
      q_in(di + 1) = rhou2[di]; //1;
   }
   // random delw matrix
   // spatial derivatives of entropy variables
   mfem::DenseMatrix delw(delw_data, dim + 2, dim);
   // dummy const vector x for calcFlux - unused
   const mfem::Vector x(nrm);
   /// finite element or SBP operators
   std::unique_ptr<mfem::FiniteElementCollection> fec;
   // Create the AD stack, and the integrator
   adept::Stack diff_stack;
   for (int p = 0; p <= 1; ++p)
   {
      fec.reset(new mfem::SBPCollection(p, dim));
      mach::ViscousInflowBC<dim> viscousinflow(diff_stack, fec.get(), Re_num,
                                             Pr_num, q_in, mu);

      DYNAMIC_SECTION("jacobian of viscous inflow bc w.r.t state failed for degree p = " << p)
      {
         mfem::DenseMatrix mat_vec_jac(num_states);
         viscousinflow.calcFluxJacState(x, nrm, jac, q, delw, mat_vec_jac);
         // loop over each state variable and check column of mat_vec_jac...
         for (int i = 0; i < num_states; ++i)
         {
            mfem::Vector q_plus(q), q_minus(q);
            mfem::Vector mat_vec_plus(num_states), mat_vec_minus(num_states);
            q_plus(i) += delta;
            q_minus(i) -= delta;
            viscousinflow.calcFlux(x, nrm, jac, q_plus, delw, mat_vec_plus);
            viscousinflow.calcFlux(x, nrm, jac, q_minus, delw, mat_vec_minus);
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
}

TEMPLATE_TEST_CASE_SIG("Viscous outflow Jacobian", "[ViscousOutflowBC]",
                       ((int dim), dim), 2, 3)
{
   using namespace euler_data;
   double delta = 1e-5;
   int num_states = dim + 2;
   double Re_num = 1;
   double Pr_num = 1;
   double mu = 1;
   double jac = 1;
   // construct state vec
   mfem::Vector q(num_states);
   mfem::Vector q_out(num_states);
   mfem::Vector nrm(dim);
   for (int di = 0; di < dim; ++di)
   {
      nrm(di) = dir[di];
   }
   q(0) = rho;
   q(dim + 1) = rhoe;
   q_out(0) = rho2;        //1.2;
   q_out(dim + 1) = rhoe2; //1;
   for (int di = 0; di < dim; ++di)
   {
      q(di + 1) = rhou[di];
      q_out(di + 1) = rhou2[di]; //1;
   }
   // random delw matrix
   // spatial derivatives of entropy variables
   mfem::DenseMatrix delw(delw_data, dim + 2, dim);
   // dummy const vector x for calcFlux - unused
   const mfem::Vector x(nrm);
   /// finite element or SBP operators
   std::unique_ptr<mfem::FiniteElementCollection> fec;
   // Create the AD stack, and the integrator
   adept::Stack diff_stack;
   for (int p = 0; p <= 1; ++p)
   {
      fec.reset(new mfem::SBPCollection(p, dim));
      mach::ViscousOutflowBC<dim> viscousoutflow(diff_stack, fec.get(), Re_num,
                                                Pr_num, q_out, mu);

      DYNAMIC_SECTION("jacobian of viscous outflow bc w.r.t state failed for degree p = " << p)
      {
         mfem::DenseMatrix mat_vec_jac(num_states);
         viscousoutflow.calcFluxJacState(x, nrm, jac, q, delw, mat_vec_jac);
         // loop over each state variable and check column of mat_vec_jac...
         for (int i = 0; i < num_states; ++i)
         {
            mfem::Vector q_plus(q), q_minus(q);
            mfem::Vector mat_vec_plus(num_states), mat_vec_minus(num_states);
            q_plus(i) += delta;
            q_minus(i) -= delta;
            viscousoutflow.calcFlux(x, nrm, jac, q_plus, delw, mat_vec_plus);
            viscousoutflow.calcFlux(x, nrm, jac, q_minus, delw, mat_vec_minus);
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
}

TEMPLATE_TEST_CASE_SIG("Viscous farfield Jacobian", "[ViscousFarFieldBC]",
                       ((int dim), dim), 2, 3)
{
   using namespace euler_data;
   double delta = 1e-5;
   int num_states = dim + 2;
   double Re_num = 1;
   double Pr_num = 1;
   double mu = 1;
   double jac = 1;
   // construct state vec
   mfem::Vector q(num_states);
   mfem::Vector qfs(num_states);
   mfem::Vector nrm(dim);
   for (int di = 0; di < dim; ++di)
   {
      nrm(di) = dir[di];
   }
   q(0) = rho;
   q(dim + 1) = rhoe;
   qfs(0) = rho2;        //1.2;
   qfs(dim + 1) = rhoe2; //1;
   for (int di = 0; di < dim; ++di)
   {
      q(di + 1) = rhou[di];
      qfs(di + 1) = rhou2[di]; //1;
   }
   // random delw matrix
   // spatial derivatives of entropy variables
   mfem::DenseMatrix delw(delw_data, dim + 2, dim);
   // dummy const vector x for calcFlux - unused
   const mfem::Vector x(nrm);
   /// finite element or SBP operators
   std::unique_ptr<mfem::FiniteElementCollection> fec;
   // Create the AD stack, and the integrator
   adept::Stack diff_stack;
   for (int p = 0; p <= 1; ++p)
   {
      fec.reset(new mfem::SBPCollection(p, dim));
      mach::ViscousFarFieldBC<dim> viscousfarfield(diff_stack, fec.get(), Re_num,
                                                   Pr_num, qfs, mu);

      DYNAMIC_SECTION("jacobian of viscous farfield bc w.r.t state failed for degree p = " << p)
      {
         mfem::DenseMatrix mat_vec_jac(num_states);
         viscousfarfield.calcFluxJacState(x, nrm, jac, q, delw, mat_vec_jac);
         // loop over each state variable and check column of mat_vec_jac...
         for (int i = 0; i < num_states; ++i)
         {
            mfem::Vector q_plus(q), q_minus(q);
            mfem::Vector mat_vec_plus(num_states), mat_vec_minus(num_states);
            q_plus(i) += delta;
            q_minus(i) -= delta;
            viscousfarfield.calcFlux(x, nrm, jac, q_plus, delw, mat_vec_plus);
            viscousfarfield.calcFlux(x, nrm, jac, q_minus, delw, mat_vec_minus);
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
}

TEMPLATE_TEST_CASE_SIG("Viscous Exact BC Jacobian", "[VisExactBC]",
                       ((int dim), dim), 2)
{
   // Simple function to act as the exact solution
   auto exact = [](const mfem::Vector &x, mfem::Vector &q) {
      q(0) = 0.9856566615165173;
      q(1) = 0.09595562550099601;
      q(2) = -0.030658751626551423;
      q(3) = 2.061597236955558;
   };

   using namespace euler_data;
   double delta = 1e-5;
   int num_states = dim + 2;
   double Re_num = 1;
   double Pr_num = 1;
   double mu = 1;
   double jac = 1;
   // construct state vec
   mfem::Vector q(num_states);
   mfem::Vector nrm(dim);
   for (int di = 0; di < dim; ++di)
   {
      nrm(di) = dir[di];
   }
   q(0) = rho;
   q(dim + 1) = rhoe;
   for (int di = 0; di < dim; ++di)
   {
      q(di + 1) = rhou[di];
   }
   // random delw matrix
   // spatial derivatives of entropy variables
   mfem::DenseMatrix delw(delw_data, dim + 2, dim);
   // dummy const vector x for calcFlux - unused
   const mfem::Vector x(nrm);
   /// finite element or SBP operators
   std::unique_ptr<mfem::FiniteElementCollection> fec;
   // Create the AD stack, and the integrator
   adept::Stack diff_stack;
   fec.reset(new mfem::SBPCollection(1, dim));
   mach::ViscousExactBC<dim> viscousexact(diff_stack, fec.get(), Re_num,
                                          Pr_num, exact, mu);

   SECTION("jacobian of viscous farfield bc w.r.t state is correct")
   {
      mfem::DenseMatrix mat_vec_jac(num_states);
      viscousexact.calcFluxJacState(x, nrm, jac, q, delw, mat_vec_jac);
      // loop over each state variable and check column of mat_vec_jac...
      for (int i = 0; i < num_states; ++i)
      {
         mfem::Vector q_plus(q), q_minus(q);
         mfem::Vector mat_vec_plus(num_states), mat_vec_minus(num_states);
         q_plus(i) += delta;
         q_minus(i) -= delta;
         viscousexact.calcFlux(x, nrm, jac, q_plus, delw, mat_vec_plus);
         viscousexact.calcFlux(x, nrm, jac, q_minus, delw, mat_vec_minus);
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

// TODO: add dim = 1, 3 once 3d sbp operators implemented
TEMPLATE_TEST_CASE_SIG("Surface force gradients", "[Surface Force]",
                       ((int dim), dim), 2, 3)
{
   using namespace euler_data;
   // copy the data into mfem vectors for convenience
   double delta = 1e-5;
   double jac = 2.0; // mapping Jacobian determinant
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
   // random delw matrix
   // spatial derivatives of entropy variables
   mfem::DenseMatrix delw(delw_data, dim + 2, dim);

   mfem::Vector drag_dir(dim);
   drag_dir = 0.0;
   double aoa_fs = 5.0 * M_PI / 180;
   double Re = 1.0;
   double Pr = 0.72;
   mfem::Vector q_ref(q);
   if (dim == 1)
   {
      drag_dir(0) = 1.0;
   }
   else if (dim == 2)
   {
      drag_dir(0) = cos(aoa_fs);
      drag_dir(1) = sin(aoa_fs);
   }
   else
   {
      drag_dir(0) = cos(aoa_fs);
      drag_dir(1) = sin(aoa_fs);
      drag_dir(2) = 0.0;
   }

   // dummy const vector x for calcBndryFun and calcBndryFunJacState - unused
   const mfem::Vector x(nrm);

   // create the perturbation vector
   mfem::Vector v(dim + 2);
   for (int i = 0; i < dim + 2; i++)
   {
      v(i) = vec_pert[i];
   }

   // Define the SBP elements and finite-element space, and integrator
   std::unique_ptr<mfem::FiniteElementCollection> fec;
   adept::Stack diff_stack;

   for (int p = 0; p <= 1; ++p)
   {
      fec.reset(new mfem::SBPCollection(p, dim));
      mach::SurfaceForce<dim> force(diff_stack, fec.get(), dim + 2, Re, Pr,
                                    q_ref, drag_dir);

      DYNAMIC_SECTION("Gradient of surface force w.r.t q ... for degree p = " << p)
      {
         // get derivative information from AD functions
         mfem::Vector dJdu_ad(dim + 2);
         force.calcBndryFunJacState(x, nrm, jac, q, delw, dJdu_ad);
         double dJdu_dot_v_ad = mfem::InnerProduct(dJdu_ad, v);

         // FD approximation
         mfem::Vector q_plus(q);
         mfem::Vector q_minus(q);
         q_plus.Add(delta, v);
         q_minus.Add(-delta, v);

         double dJdu_dot_v_fd = force.calcBndryFun(x, nrm, jac, q_plus, delw);
         dJdu_dot_v_fd -= force.calcBndryFun(x, nrm, jac, q_minus, delw);
         dJdu_dot_v_fd /= 2 * delta;

         // compare
         REQUIRE(dJdu_dot_v_ad == Approx(dJdu_dot_v_fd).margin(1e-10));
      }

      DYNAMIC_SECTION("Gradient of surface force w.r.t Dw ... degree p = " << p)
      {
         // get derivative information from AD functions
         mfem::DenseMatrix dJdDw_ad(dim + 2, dim);
         force.calcBndryFunJacDw(x, nrm, jac, q, delw, dJdDw_ad);
         mfem::Vector mat_vec_ad(dim);
         dJdDw_ad.MultTranspose(v, mat_vec_ad);

         // loop over each dimension and check against mat_vec_ad
         for (int d = 0; d < dim; ++d)
         {
            // perturb one column of delw everytime
            mfem::DenseMatrix delw_plus(delw), delw_minus(delw);
            for (int s = 0; s < dim + 2; ++s)
            {
               delw_plus.GetColumn(d)[s] += v(s) * delta;
               delw_minus.GetColumn(d)[s] -= v(s) * delta;
            }
            double mat_vec_fd = force.calcBndryFun(x, nrm, jac, q, delw_plus);
            mat_vec_fd -= force.calcBndryFun(x, nrm, jac, q, delw_minus);
            mat_vec_fd /= 2.0 * delta;

            // compare
            REQUIRE(mat_vec_ad(d) == Approx(mat_vec_fd).margin(1e-10));
         } // d loop
      }    // section
   }
}