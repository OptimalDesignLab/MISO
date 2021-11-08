#include <random>

#include "catch.hpp"
#include "mfem.hpp"
#include "euler_fluxes.hpp"
#include "euler_integ_dg.hpp"
#include "euler_test_data.hpp"

TEST_CASE("EulerDGIntegrator::AssembleElementGrad", "[EulerDGIntegrator]")
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 2; // templating is hard here because mesh constructors
   int num_state = dim + 2;
   adept::Stack diff_stack;
   double delta = 1e-5;

   // generate a 2 element mesh
   int num_edge = 2;
   Mesh mesh(Mesh::MakeCartesian2D(num_edge, num_edge, Element::TRIANGLE,
                                   true /* gen. edges */, 1.0, 1.0, true));
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
             new DG_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             &mesh, fec.get(), num_state, Ordering::byVDIM));

         NonlinearForm res(fes.get());
         res.AddDomainIntegrator(new mach::EulerDGIntegrator<2>(diff_stack));

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselineVectorPert<2>);
         q.ProjectCoefficient(pert);

         // initialize the vector that the Jacobian multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(num_state, randVectorState);
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

      DYNAMIC_SECTION("(DG)...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
             new DG_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             &mesh, fec.get(), num_state, Ordering::byVDIM));

         NonlinearForm res(fes.get());
         res.AddDomainIntegrator(new mach::EulerDGIntegrator<2>(diff_stack));

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselineVectorPert<2>);
         q.ProjectCoefficient(pert);

         // initialize the vector that the Jacobian multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(num_state, randVectorState);
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

TEST_CASE("DGSlipWallBC::AssembleFaceGrad", "[DGSlipWallBC]")
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 2; // templating is hard here because mesh constructors
   int num_state = dim + 2;
   adept::Stack diff_stack;
   double delta = 1e-5;

   // generate a 2 element mesh
   int num_edge = 2;
   Mesh mesh(Mesh::MakeCartesian2D(num_edge, num_edge, Element::TRIANGLE,
                                   true /* gen. edges */, 1.0, 1.0, true));
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
             new DG_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             &mesh, fec.get(), num_state, Ordering::byVDIM));

         NonlinearForm res(fes.get());
         res.AddBdrFaceIntegrator(new mach::DGSlipWallBC<dim>(diff_stack,
                                                            fec.get()));

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselineVectorPert<dim>);
         q.ProjectCoefficient(pert);

         // initialize the vector that the Jacobian multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(num_state, randVectorState);
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

      DYNAMIC_SECTION("(DG)...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
             new DG_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             &mesh, fec.get(), num_state, Ordering::byVDIM));

         NonlinearForm res(fes.get());
         res.AddBdrFaceIntegrator(new mach::DGSlipWallBC<dim>(diff_stack,
                                                            fec.get()));

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselineVectorPert<dim>);
         q.ProjectCoefficient(pert);

         // initialize the vector that the Jacobian multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(num_state, randVectorState);
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

TEST_CASE("DGPressureForce::AssembleVector", "[DGPressureForce]")
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
   Mesh mesh(Mesh::MakeCartesian2D(num_edge, num_edge, Element::TRIANGLE,
                                   true /* gen. edges */, 1.0, 1.0, true));
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
             new DG_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             &mesh, fec.get(), num_state, Ordering::byVDIM));

         NonlinearForm drag(fes.get());
         drag.AddBdrFaceIntegrator(
            new mach::DGPressureForce<dim>(diff_stack, fec.get(), drag_dir));

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselineVectorPert<dim>);
         q.ProjectCoefficient(pert);

         // initialize the vector that dJdu multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(num_state, randVectorState);
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

      DYNAMIC_SECTION("(DG)...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
             new DG_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             &mesh, fec.get(), num_state, Ordering::byVDIM));

         NonlinearForm drag(fes.get());
         drag.AddBdrFaceIntegrator(
            new mach::DGPressureForce<dim>(diff_stack, fec.get(), drag_dir));

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselineVectorPert<dim>);
         q.ProjectCoefficient(pert);

         // initialize the vector that dJdu multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(num_state, randVectorState);
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

// TODO: add dim = 1, 3 once 3d sbp operators implemented
TEMPLATE_TEST_CASE_SIG("DGInviscidFaceIntegrator::AssembleFaceGrad", 
                        "[DGInterfaceIntegrator]",
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
   Mesh mesh(Mesh::MakeCartesian2D(num_edge, num_edge, Element::TRIANGLE,
                                   true /* gen. edges */, 1.0, 1.0, true));

   const int max_degree = 4;
   for (int p = 0; p <= max_degree; p++)
   {
      DYNAMIC_SECTION("Jacobian of Interface flux w.r.t state is correct (DSBP)" << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
             new DG_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             &mesh, fec.get(), num_state, Ordering::byVDIM));

         NonlinearForm res(fes.get());
         res.AddInteriorFaceIntegrator(
             new mach::DGInterfaceIntegrator<dim, entvar>(diff_stack, diss_coeff,
                                                        fec.get()));

         // initialize state; here we randomly perturb a constant state
         GridFunction w(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselineVectorPert<dim, entvar>);
         w.ProjectCoefficient(pert);

         // initialize the vector that the Jacobian multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(num_state, randVectorState);
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