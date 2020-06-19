#include <random>

#include "catch.hpp"
#include "mfem.hpp"
#include "euler_fluxes.hpp"
#include "euler_integ.hpp"
#include "euler_test_data.hpp"

TEST_CASE("EulerIntegrator::AssembleElementGrad", "[EulerIntegrator]")
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 2; // templating is hard here because mesh constructors
   int num_state = dim + 2;
   adept::Stack diff_stack;
   double delta = 1e-5;

   // generate a 2 element mesh
   int num_edge = 2;
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
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)).margin(1e-10));
         }
      }

      DYNAMIC_SECTION("(DSBP)...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
             new DSBPCollection(p, dim));
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
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)).margin(1e-10));
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
   int num_edge = 2;
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
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)).margin(1e-10));
         }
      }

      DYNAMIC_SECTION("(DSBP)...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
             new DSBPCollection(p, dim));
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
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)).margin(1e-10));
         }
      }
   }
}

TEST_CASE("PressureForce::AssembleVector", "[PressureForce]")
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 2; // templating is hard here because mesh constructors
   int num_state = dim + 2;
   adept::Stack diff_stack;
   double delta = 1e-5;

   mfem::Vector drag_dir(dim);
   double aoa_fs = 5.0*M_PI/180;
   drag_dir(0) = cos(aoa_fs);
   drag_dir(1) = sin(aoa_fs);

   // generate a 2 element mesh
   int num_edge = 2;
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

         NonlinearForm drag(fes.get());
         drag.AddBdrFaceIntegrator(
            new mach::PressureForce<dim>(diff_stack, fec.get(), drag_dir));

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselinePert<dim>);
         q.ProjectCoefficient(pert);

         // initialize the vector that dJdu multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(num_state, randState);
         v.ProjectCoefficient(v_rand);

         // evaluate dJdu and compute its product with v
         GridFunction dJdu(fes.get());
         drag.Mult(q, dJdu);
         double dJdu_dot_v = InnerProduct(dJdu, v);

         // now compute the finite-difference approximation...
         GridFunction q_pert(q);
         q_pert.Add(-delta, v);
         double dJdu_dot_v_fd = -drag.GetEnergy(q_pert);
         q_pert.Add(2 * delta, v);
         dJdu_dot_v_fd += drag.GetEnergy(q_pert);
         dJdu_dot_v_fd /= (2 * delta);
         REQUIRE(dJdu_dot_v == Approx(dJdu_dot_v_fd).margin(1e-10));
      }

      DYNAMIC_SECTION("(DSBP)...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
             new DSBPCollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             mesh.get(), fec.get(), num_state, Ordering::byVDIM));

         NonlinearForm drag(fes.get());
         drag.AddBdrFaceIntegrator(
            new mach::PressureForce<dim>(diff_stack, fec.get(), drag_dir));

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselinePert<dim>);
         q.ProjectCoefficient(pert);

         // initialize the vector that dJdu multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(num_state, randState);
         v.ProjectCoefficient(v_rand);

         // evaluate dJdu and compute its product with v
         GridFunction dJdu(fes.get());
         drag.Mult(q, dJdu);
         double dJdu_dot_v = InnerProduct(dJdu, v);

         // now compute the finite-difference approximation...
         GridFunction q_pert(q);
         q_pert.Add(-delta, v);
         double dJdu_dot_v_fd = -drag.GetEnergy(q_pert);
         q_pert.Add(2 * delta, v);
         dJdu_dot_v_fd += drag.GetEnergy(q_pert);
         dJdu_dot_v_fd /= (2 * delta);
         REQUIRE(dJdu_dot_v == Approx(dJdu_dot_v_fd).margin(1e-10));
      }
   }
}

TEMPLATE_TEST_CASE_SIG("DyadicFluxIntegrator::AssembleElementGrad",
                       "[DyadicIntegrator]",
                       ((bool entvar), entvar), false, true)
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 2; // templating is hard here because mesh constructors
   int num_state = dim + 2;
   adept::Stack diff_stack;
   double delta = 1e-5;

   // generate a 2 element mesh
   int num_edge = 2;
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
         res.AddDomainIntegrator(
            new mach::IsmailRoeIntegrator<2,entvar>(diff_stack));

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselinePert<2,entvar>);
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
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)).margin(1e-10));
         }
      }

      DYNAMIC_SECTION("(DSBP)...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
             new DSBPCollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             mesh.get(), fec.get(), num_state, Ordering::byVDIM));

         NonlinearForm res(fes.get());
         res.AddDomainIntegrator(new mach::IsmailRoeIntegrator<2, entvar>(diff_stack));

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselinePert<2, entvar>);
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
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)).margin(1e-10));
         }
      }
   }
}

// JEH: This is redundant, as far as I can tell.
// TODO: add dim = 1, 3 once 3d sbp operators implemented
// TEST_CASE("InviscidFaceIntegrator::AssembleFaceGrad", "[InterfaceIntegrator]")
// {
//    using namespace euler_data;
//    using namespace mfem;
//    const int dim = 2;
//    double delta = 1e-5;
//    int num_state = dim + 2;
//    adept::Stack diff_stack;
//    double diss_coeff = 1.0;

//    // generate a 2 element mesh
//    int num_edge = 2;
//    std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, Element::TRIANGLE,
//                                        true /* gen. edges */, 1.0, 1.0, true));

//    const int max_degree = 4;
//    for (int p = 0; p <= max_degree; p++)
//    {
//       DYNAMIC_SECTION("Jacobian of Interface flux w.r.t state is correct (DSBP)" << p)
//       {
//          std::unique_ptr<FiniteElementCollection> fec(
//              new DSBPCollection(p, dim));
//          std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
//              mesh.get(), fec.get(), num_state, Ordering::byVDIM));

//          NonlinearForm res(fes.get());
//          res.AddInteriorFaceIntegrator(new mach::InterfaceIntegrator<dim>(
//              diff_stack, diss_coeff, fec.get()));

//          // initialize state; here we randomly perturb a constant state
//          GridFunction q(fes.get());
//          VectorFunctionCoefficient pert(num_state, randBaselinePert<dim>);
//          q.ProjectCoefficient(pert);

//          // initialize the vector that the Jacobian multiplies
//          GridFunction v(fes.get());
//          VectorFunctionCoefficient v_rand(num_state, randState);
//          v.ProjectCoefficient(v_rand);

//          // evaluate the Jacobian and compute its product with v
//          Operator &Jac = res.GetGradient(q);
//          GridFunction jac_v(fes.get());
//          Jac.Mult(v, jac_v);

//          // now compute the finite-difference approximation...
//          GridFunction q_pert(q), r(fes.get()), jac_v_fd(fes.get());
//          q_pert.Add(-delta, v);
//          res.Mult(q_pert, r);
//          q_pert.Add(2 * delta, v);
//          res.Mult(q_pert, jac_v_fd);
//          jac_v_fd -= r;
//          jac_v_fd /= (2 * delta);

//          for (int i = 0; i < jac_v.Size(); ++i)
//          {
//             REQUIRE(jac_v(i) == Approx(jac_v_fd(i)));
//          }
//       }
//    } // loop different order of elements
// }

TEMPLATE_TEST_CASE_SIG("EntStableLPSIntegrator::AssembleElementGrad using entvar",
                       "[LPSIntegrator]",
                       ((bool entvar), entvar), false, true)
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 2; // templating is hard here because mesh constructors
   int num_state = dim + 2;
   adept::Stack diff_stack;
   double delta = 1e-5;

   // generate a 2 element mesh
   int num_edge = 2;
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
         res.AddDomainIntegrator(
             new mach::EntStableLPSIntegrator<2, entvar>(diff_stack));

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselinePert<2, entvar>);
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
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)).margin(1e-10));
         }
      }

      DYNAMIC_SECTION("(DSBP)...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
             new DSBPCollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             mesh.get(), fec.get(), num_state, Ordering::byVDIM));

         NonlinearForm res(fes.get());
         res.AddDomainIntegrator(new mach::EntStableLPSIntegrator<2, entvar>(diff_stack));

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselinePert<2, entvar>);
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
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)).margin(1e-10));
         }
      }
   }
}

TEMPLATE_TEST_CASE_SIG("MassIntegrator::AssembleElementGrad",
                       "[MassIntegrator]", ((bool entvar), entvar), false, true)
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 2; // templating is hard here because mesh constructors
   int num_state = dim + 2;
   adept::Stack diff_stack;
   double delta = 1e-5;

   // generate an 8 element mesh
   int num_edge = 2;
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

         // initialize state; we randomly perturb a constant state
         GridFunction u(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselinePert<2, entvar>);
         u.ProjectCoefficient(pert);

         // initialize the vector that the Jacobian multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(num_state, randState);
         v.ProjectCoefficient(v_rand);

         NonlinearForm res(fes.get());
         res.AddDomainIntegrator(
             new mach::MassIntegrator<2, entvar>(diff_stack));

         // evaluate the Jacobian and compute its product with v
         Operator &Jac = res.GetGradient(u);
         GridFunction jac_v(fes.get());
         Jac.Mult(v, jac_v);

         // now compute the finite-difference approximation...
         GridFunction u_pert(u), r(fes.get()), jac_v_fd(fes.get());
         u_pert.Add(-delta, v);
         res.Mult(u_pert, r);
         u_pert.Add(2 * delta, v);
         res.Mult(u_pert, jac_v_fd);
         jac_v_fd -= r;
         jac_v_fd /= (2 * delta);

         for (int i = 0; i < jac_v.Size(); ++i)
         {
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)).margin(1e-10));
         }
      }

      DYNAMIC_SECTION("(DSBP)...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
             new DSBPCollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             mesh.get(), fec.get(), num_state, Ordering::byVDIM));

         // initialize state and k = du/dt; here we randomly perturb a constant state
         GridFunction u(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselinePert<2, entvar>);
         u.ProjectCoefficient(pert);

         // initialize the vector that the Jacobian multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(num_state, randState);
         v.ProjectCoefficient(v_rand);

         NonlinearForm res(fes.get());
         res.AddDomainIntegrator(
             new mach::MassIntegrator<2, entvar>(diff_stack));

         // evaluate the Jacobian and compute its product with v
         Operator &Jac = res.GetGradient(u);
         GridFunction jac_v(fes.get());
         Jac.Mult(v, jac_v);

         // now compute the finite-difference approximation...
         GridFunction u_pert(u), r(fes.get()), jac_v_fd(fes.get());
         u_pert.Add(-delta, v);
         res.Mult(u_pert, r);
         u_pert.Add(2 * delta, v);
         res.Mult(u_pert, jac_v_fd);
         jac_v_fd -= r;
         jac_v_fd /= (2 * delta);

         for (int i = 0; i < jac_v.Size(); ++i)
         {
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)).margin(1e-10));
         }
      }
   }
}

// TODO: add dim = 1, 3 once 3d sbp operators implemented
TEMPLATE_TEST_CASE_SIG("InviscidFaceIntegrator::AssembleFaceGrad", 
                        "[InterfaceIntegrator]",
                       ((bool entvar), entvar), false, true)
{
   using namespace euler_data;
   using namespace mfem;
   const int dim = 2;
   double delta = 1e-5;
   int num_state = dim + 2;
   adept::Stack diff_stack;
   double diss_coeff = 1.0;

   // generate a 2 element mesh
   int num_edge = 2;
   std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, Element::TRIANGLE,
                                       true /* gen. edges */, 1.0, 1.0, true));

   const int max_degree = 4;
   for (int p = 0; p <= max_degree; p++)
   {
      DYNAMIC_SECTION("Jacobian of Interface flux w.r.t state is correct (DSBP)" << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
             new DSBPCollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             mesh.get(), fec.get(), num_state, Ordering::byVDIM));

         NonlinearForm res(fes.get());
         res.AddInteriorFaceIntegrator(
             new mach::InterfaceIntegrator<dim, entvar>(diff_stack, diss_coeff,
                                                        fec.get()));

         // initialize state; here we randomly perturb a constant state
         GridFunction w(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselinePert<dim, entvar>);
         w.ProjectCoefficient(pert);

         // initialize the vector that the Jacobian multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(num_state, randState);
         v.ProjectCoefficient(v_rand);

         // evaluate the Jacobian and compute its product with v
         Operator &Jac = res.GetGradient(w);
         GridFunction jac_v(fes.get());
         Jac.Mult(v, jac_v);

         // now compute the finite-difference approximation...
         GridFunction w_pert(w), r(fes.get()), jac_v_fd(fes.get());
         w_pert.Add(-delta, v);
         res.Mult(w_pert, r);
         w_pert.Add(2 * delta, v);
         res.Mult(w_pert, jac_v_fd);
         jac_v_fd -= r;
         jac_v_fd /= (2 * delta);

         for (int i = 0; i < jac_v.Size(); ++i)
         {
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)));
            //std::cout << "jac_v(" << i << ") = " << jac_v(i) << std::endl;
            //std::cout << "jac_v_fd = " << jac_v_fd(i) << std::endl;
         }
      }
   } // loop different order of elements
}

TEMPLATE_TEST_CASE_SIG("Isentropic BC flux using entvar", "[IsentropricVortexBC]",
                       ((bool entvar), entvar), false, true)
{
   const int dim = 2;
   using namespace euler_data;
   double delta = 1e-5;
   mfem::Vector nrm(dim);
   for (int di = 0; di < dim; ++di)
   {
      nrm(di) = dir[di];
   }
   mfem::Vector q(dim + 2), w(dim + 2);
   q(0) = rho;
   q(dim + 1) = rhoe;
   for (int di = 0; di < dim; ++di)
   {
      q(di + 1) = rhou[di];
   }
   mach::calcEntropyVars<double, dim>(q.GetData(), w.GetData());

   // dummy const vector x for calcFlux - unused
   const mfem::Vector x(nrm);
   /// finite element or SBP operators
   std::unique_ptr<mfem::FiniteElementCollection> fec;
   adept::Stack diff_stack;
   //diff_stack.deactivate();
   const int max_degree = 4;
   for (int p = 1; p <= max_degree; p++)
   {
      

      DYNAMIC_SECTION("Jacobian of Isentropic Vortex BC flux w.r.t state is correct")
      {
         fec.reset(new mfem::SBPCollection(1, dim));
         mach::IsentropicVortexBC<dim, entvar> isentropic_vortex(diff_stack, fec.get());
         // create the perturbation vector
         mfem::Vector v(dim + 2);
         for (int i = 0; i < dim + 2; i++)
         {
            v(i) = vec_pert[i];
         }

         // get derivative information from AD functions and form product
         mfem::DenseMatrix jac_ad(dim + 2, dim + 2);
         mfem::Vector jac_v_ad(dim + 2);
         isentropic_vortex.calcFluxJacState(x, nrm, w, jac_ad);
         jac_ad.Mult(v, jac_v_ad);

         // FD approximation
         mfem::Vector w_plus(w);
         mfem::Vector w_minus(w);
         w_plus.Add(delta, v);
         w_minus.Add(-delta, v);

         mfem::Vector flux_plus(dim + 2);
         mfem::Vector flux_minus(dim + 2);
         isentropic_vortex.calcFlux(x, nrm, w_plus, flux_plus);
         isentropic_vortex.calcFlux(x, nrm, w_minus, flux_minus);

         // finite difference jacobian
         mfem::Vector jac_v_fd(dim + 2);
         subtract(flux_plus, flux_minus, jac_v_fd);
         jac_v_fd /= 2 * delta;

         // compare
         for (int i = 0; i < dim + 2; ++i)
         {
            REQUIRE(jac_v_ad(i) == Approx(jac_v_fd(i)).margin(1e-10));
         }
      }

      DYNAMIC_SECTION("Jacobian of Isentropic Vortex BC flux w.r.t state is correct(DSBP)")
      {
         fec.reset(new mfem::DSBPCollection(1, dim));
         mach::IsentropicVortexBC<dim, entvar> isentropic_vortex(diff_stack, fec.get());
         // create the perturbation vector
         mfem::Vector v(dim + 2);
         for (int i = 0; i < dim + 2; i++)
         {
            v(i) = vec_pert[i];
         }

         // get derivative information from AD functions and form product
         mfem::DenseMatrix jac_ad(dim + 2, dim + 2);
         mfem::Vector jac_v_ad(dim + 2);
         isentropic_vortex.calcFluxJacState(x, nrm, w, jac_ad);
         jac_ad.Mult(v, jac_v_ad);

         // FD approximation
         mfem::Vector w_plus(w);
         mfem::Vector w_minus(w);
         w_plus.Add(delta, v);
         w_minus.Add(-delta, v);

         mfem::Vector flux_plus(dim + 2);
         mfem::Vector flux_minus(dim + 2);
         isentropic_vortex.calcFlux(x, nrm, w_plus, flux_plus);
         isentropic_vortex.calcFlux(x, nrm, w_minus, flux_minus);

         // finite difference jacobian
         mfem::Vector jac_v_fd(dim + 2);
         subtract(flux_plus, flux_minus, jac_v_fd);
         jac_v_fd /= 2 * delta;

         // compare
         for (int i = 0; i < dim + 2; ++i)
         {
            REQUIRE(jac_v_ad(i) == Approx(jac_v_fd(i)).margin(1e-10));
         }
      }

      DYNAMIC_SECTION("Jacobian of Isentropic Vortex BC flux w.r.t dir is correct")
      {
         fec.reset(new mfem::SBPCollection(1, dim));
         mach::IsentropicVortexBC<dim, entvar> isentropic_vortex(diff_stack, fec.get());
         // create the perturbation vector
         mfem::Vector v(dim);
         for (int i = 0; i < dim; i++)
         {
            v(i) = vec_pert[i];
         }

         // get derivative information from AD functions and form product
         mfem::DenseMatrix jac_ad(dim + 2, dim);
         mfem::Vector jac_v_ad(dim + 2);
         isentropic_vortex.calcFluxJacDir(x, nrm, w, jac_ad);
         jac_ad.Mult(v, jac_v_ad);

         // FD approximation
         mfem::Vector nrm_plus(nrm);
         mfem::Vector nrm_minus(nrm);
         nrm_plus.Add(delta, v);
         nrm_minus.Add(-delta, v);

         mfem::Vector flux_plus(dim + 2);
         mfem::Vector flux_minus(dim + 2);
         isentropic_vortex.calcFlux(x, nrm_plus, w, flux_plus);
         isentropic_vortex.calcFlux(x, nrm_minus, w, flux_minus);

         // finite difference jacobian
         mfem::Vector jac_v_fd(dim + 2);
         subtract(flux_plus, flux_minus, jac_v_fd);
         jac_v_fd /= 2 * delta;

         // compare
         for (int i = 0; i < dim; ++i)
         {
            REQUIRE(jac_v_ad(i) == Approx(jac_v_fd(i)).margin(1e-10));
         }
      }

      DYNAMIC_SECTION("Jacobian of Isentropic Vortex BC flux w.r.t dir is correct(DSBP)")
      {
         fec.reset(new mfem::DSBPCollection(1, dim));
         mach::IsentropicVortexBC<dim, entvar> isentropic_vortex(diff_stack, fec.get());
         // create the perturbation vector
         mfem::Vector v(dim);
         for (int i = 0; i < dim; i++)
         {
            v(i) = vec_pert[i];
         }

         // get derivative information from AD functions and form product
         mfem::DenseMatrix jac_ad(dim + 2, dim);
         mfem::Vector jac_v_ad(dim + 2);
         isentropic_vortex.calcFluxJacDir(x, nrm, w, jac_ad);
         jac_ad.Mult(v, jac_v_ad);

         // FD approximation
         mfem::Vector nrm_plus(nrm);
         mfem::Vector nrm_minus(nrm);
         nrm_plus.Add(delta, v);
         nrm_minus.Add(-delta, v);

         mfem::Vector flux_plus(dim + 2);
         mfem::Vector flux_minus(dim + 2);
         isentropic_vortex.calcFlux(x, nrm_plus, w, flux_plus);
         isentropic_vortex.calcFlux(x, nrm_minus, w, flux_minus);

         // finite difference jacobian
         mfem::Vector jac_v_fd(dim + 2);
         subtract(flux_plus, flux_minus, jac_v_fd);
         jac_v_fd /= 2 * delta;

         // compare
         for (int i = 0; i < dim; ++i)
         {
            REQUIRE(jac_v_ad(i) == Approx(jac_v_fd(i)).margin(1e-10));
         }
      }
   }

}

// test of slip wall bc using entropy varibles
TEMPLATE_TEST_CASE_SIG("Slip Wall Flux using entvar", "[Slip Wall]",
                       ((bool entvar), entvar), false, true)
{
   using namespace euler_data;
   // copy the data into mfem vectors for convenience
   const int dim = 2;
   double delta = 1e-5;
   mfem::Vector nrm(dim);
   for (int di = 0; di < dim; ++di)
   {
      nrm(di) = dir[di];
   }
   mfem::Vector q(dim + 2), w(dim + 2);
   q(0) = rho;
   q(dim + 1) = rhoe;
   for (int di = 0; di < dim; ++di)
   {
      q(di + 1) = rhou[di];
   }
   mach::calcEntropyVars<double, dim>(q.GetData(), w.GetData());

   // dummy const vector x for calcFlux - unused
   const mfem::Vector x(nrm);

   /// finite element or SBP operators
   std::unique_ptr<mfem::FiniteElementCollection> fec;
   adept::Stack diff_stack;

   const int max_degree = 4;
   for (int p = 1; p <= max_degree; ++p)
   {
      DYNAMIC_SECTION("Jacobian of slip wall flux w.r.t state is correct")
      {
         // Define the SBP elements and finite-element space
         fec.reset(new mfem::SBPCollection(p, dim));
         mach::SlipWallBC<dim, entvar> slip_wall(diff_stack, fec.get());
         // create the perturbation vector
         mfem::Vector v(dim + 2);
         for (int i = 0; i < dim + 2; i++)
         {
            v(i) = vec_pert[i];
         }

         // get derivative information from AD functions and form product
         mfem::DenseMatrix jac_ad(dim + 2, dim + 2);
         mfem::Vector jac_v_ad(dim + 2);
         slip_wall.calcFluxJacState(x, nrm, w, jac_ad);
         jac_ad.Mult(v, jac_v_ad);

         // FD approximation
         mfem::Vector w_plus(w);
         mfem::Vector w_minus(w);
         w_plus.Add(delta, v);
         w_minus.Add(-delta, v);

         mfem::Vector flux_plus(dim + 2);
         mfem::Vector flux_minus(dim + 2);
         slip_wall.calcFlux(x, nrm, w_plus, flux_plus);
         slip_wall.calcFlux(x, nrm, w_minus, flux_minus);

         // finite difference jacobian
         mfem::Vector jac_v_fd(dim + 2);
         subtract(flux_plus, flux_minus, jac_v_fd);
         jac_v_fd /= 2 * delta;

         // compare
         for (int i = 0; i < dim + 2; ++i)
         {
            REQUIRE(jac_v_ad(i) == Approx(jac_v_fd(i)).margin(1e-10));
         }
      }

      DYNAMIC_SECTION("Jacobian of slip wall flux w.r.t state is correct(DSBP)")
      {
         // Define the SBP elements and finite-element space
         fec.reset(new mfem::DSBPCollection(p, dim));
         mach::SlipWallBC<dim, entvar> slip_wall(diff_stack, fec.get());
         // create the perturbation vector
         mfem::Vector v(dim + 2);
         for (int i = 0; i < dim + 2; i++)
         {
            v(i) = vec_pert[i];
         }

         // get derivative information from AD functions and form product
         mfem::DenseMatrix jac_ad(dim + 2, dim + 2);
         mfem::Vector jac_v_ad(dim + 2);
         slip_wall.calcFluxJacState(x, nrm, w, jac_ad);
         jac_ad.Mult(v, jac_v_ad);

         // FD approximation
         mfem::Vector w_plus(w);
         mfem::Vector w_minus(w);
         w_plus.Add(delta, v);
         w_minus.Add(-delta, v);

         mfem::Vector flux_plus(dim + 2);
         mfem::Vector flux_minus(dim + 2);
         slip_wall.calcFlux(x, nrm, w_plus, flux_plus);
         slip_wall.calcFlux(x, nrm, w_minus, flux_minus);

         // finite difference jacobian
         mfem::Vector jac_v_fd(dim + 2);
         subtract(flux_plus, flux_minus, jac_v_fd);
         jac_v_fd /= 2 * delta;

         // compare
         for (int i = 0; i < dim + 2; ++i)
         {
            REQUIRE(jac_v_ad(i) == Approx(jac_v_fd(i)).margin(1e-10));
         }
      }

      DYNAMIC_SECTION("Jacobian of slip wall flux w.r.t dir is correct")
      {
         // Define the SBP elements and finite-element space
         fec.reset(new mfem::SBPCollection(p, dim));
         mach::SlipWallBC<dim, entvar> slip_wall(diff_stack, fec.get());
         // create the perturbation vector
         mfem::Vector v(dim);
         for (int i = 0; i < dim; i++)
         {
            v(i) = vec_pert[i];
         }

         // get derivative information from AD functions and form product
         mfem::DenseMatrix jac_ad(dim + 2, dim);
         mfem::Vector jac_v_ad(dim + 2);
         slip_wall.calcFluxJacDir(x, nrm, w, jac_ad);
         jac_ad.Mult(v, jac_v_ad);

         // FD approximation
         mfem::Vector nrm_plus(nrm);
         mfem::Vector nrm_minus(nrm);
         nrm_plus.Add(delta, v);
         nrm_minus.Add(-delta, v);

         mfem::Vector flux_plus(dim + 2);
         mfem::Vector flux_minus(dim + 2);
         slip_wall.calcFlux(x, nrm_plus, w, flux_plus);
         slip_wall.calcFlux(x, nrm_minus, w, flux_minus);

         // finite difference jacobian
         mfem::Vector jac_v_fd(dim + 2);
         subtract(flux_plus, flux_minus, jac_v_fd);
         jac_v_fd /= 2 * delta;

         // compare
         for (int i = 0; i < dim; ++i)
         {
            REQUIRE(jac_v_ad(i) == Approx(jac_v_fd(i)).margin(1e-10));
         }
      }

      DYNAMIC_SECTION("Jacobian of slip wall flux w.r.t dir is correct(DSBP)")
      {
         // Define the SBP elements and finite-element space
         fec.reset(new mfem::DSBPCollection(p, dim));
         mach::SlipWallBC<dim, entvar> slip_wall(diff_stack, fec.get());
         // create the perturbation vector
         mfem::Vector v(dim);
         for (int i = 0; i < dim; i++)
         {
            v(i) = vec_pert[i];
         }

         // get derivative information from AD functions and form product
         mfem::DenseMatrix jac_ad(dim + 2, dim);
         mfem::Vector jac_v_ad(dim + 2);
         slip_wall.calcFluxJacDir(x, nrm, w, jac_ad);
         jac_ad.Mult(v, jac_v_ad);

         // FD approximation
         mfem::Vector nrm_plus(nrm);
         mfem::Vector nrm_minus(nrm);
         nrm_plus.Add(delta, v);
         nrm_minus.Add(-delta, v);

         mfem::Vector flux_plus(dim + 2);
         mfem::Vector flux_minus(dim + 2);
         slip_wall.calcFlux(x, nrm_plus, w, flux_plus);
         slip_wall.calcFlux(x, nrm_minus, w, flux_minus);

         // finite difference jacobian
         mfem::Vector jac_v_fd(dim + 2);
         subtract(flux_plus, flux_minus, jac_v_fd);
         jac_v_fd /= 2 * delta;

         // compare
         for (int i = 0; i < dim; ++i)
         {
            REQUIRE(jac_v_ad(i) == Approx(jac_v_fd(i)).margin(1e-10));
         }
      }
   }
}