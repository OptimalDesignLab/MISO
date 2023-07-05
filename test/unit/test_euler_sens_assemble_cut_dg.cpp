#include <random>

#include "catch.hpp"
#include "mfem.hpp"
#include "euler_fluxes.hpp"
#include "euler_integ_dg_cut_sens.hpp"
#include "euler_test_data.hpp"
#include "inviscid_integ_dg_cut_sens.hpp"

TEST_CASE("CutEulerDGSensitivityIntegrator::AssembleElementVector", "[CutEulerDGSensitivityIntegrator]")
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 2;  // templating is hard here because mesh constructors
   int num_state = dim + 2;
   adept::Stack diff_stack;
   double delta = 1e-5;

   // generate a 2 element mesh
   int num_edge = 10;
   Mesh mesh = Mesh::MakeCartesian2D(
       num_edge, num_edge, Element::QUADRILATERAL, true, 10.0, 10.0, true);
   mesh.reset(new MeshType(comm, *smesh));
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
             new DG_FECollection(p, dim));
         std::unique_ptr<GDSpaceType> fes_gd_p;
         fes_gd_p.reset(new GDSpaceType(mesh.get(),
                                        fec.get(),
                                        embeddedElements_p,
                                        cutElements_p,
                                        num_state,
                                        Ordering::byVDIM,
                                        p));

         NonlinearFormType res(fes_gd_p.get());
    //      res.AddDomainIntegrator(new mach::EulerDGIntegrator<2>(diff_stack));

    //      // initialize state; here we randomly perturb a constant state
    //      GridFunction q(fes.get());
    //      VectorFunctionCoefficient pert(num_state, randBaselineVectorPert<2>);
    //      q.ProjectCoefficient(pert);

    //      // initialize the vector that the Jacobian multiplies
    //      GridFunction v(fes.get());
    //      VectorFunctionCoefficient v_rand(num_state, randVectorState);
    //      v.ProjectCoefficient(v_rand);

    //      // evaluate the Jacobian and compute its product with v
    //      Operator &Jac = res.GetGradient(q);
    //      GridFunction jac_v(fes.get());
    //      Jac.Mult(v, jac_v);

    //      // now compute the finite-difference approximation...
    //      GridFunction q_pert(q), r(fes.get()), jac_v_fd(fes.get());
    //      q_pert.Add(-delta, v);
    //      res.Mult(q_pert, r);
    //      q_pert.Add(2 * delta, v);
    //      res.Mult(q_pert, jac_v_fd);
    //      jac_v_fd -= r;
    //      jac_v_fd /= (2 * delta);

    //      for (int i = 0; i < jac_v.Size(); ++i)
    //      {
    //         REQUIRE(jac_v(i) == Approx(jac_v_fd(i)).margin(1e-10));
    //      }
    //   }

    //   DYNAMIC_SECTION("(DG)...for degree p = " << p)
    //   {
    //      std::unique_ptr<FiniteElementCollection> fec(
    //          new DG_FECollection(p, dim));
    //      std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
    //          &mesh, fec.get(), num_state, Ordering::byVDIM));

    //      NonlinearForm res(fes.get());
    //      res.AddDomainIntegrator(new mach::EulerDGIntegrator<2>(diff_stack));

    //      // initialize state; here we randomly perturb a constant state
    //      GridFunction q(fes.get());
    //      VectorFunctionCoefficient pert(num_state, randBaselineVectorPert<2>);
    //      q.ProjectCoefficient(pert);

    //      // initialize the vector that the Jacobian multiplies
    //      GridFunction v(fes.get());
    //      VectorFunctionCoefficient v_rand(num_state, randVectorState);
    //      v.ProjectCoefficient(v_rand);

    //      // evaluate the Jacobian and compute its product with v
    //      Operator &Jac = res.GetGradient(q);
    //      GridFunction jac_v(fes.get());
    //      Jac.Mult(v, jac_v);

    //      // now compute the finite-difference approximation...
    //      GridFunction q_pert(q), r(fes.get()), jac_v_fd(fes.get());
    //      q_pert.Add(-delta, v);
    //      res.Mult(q_pert, r);
    //      q_pert.Add(2 * delta, v);
    //      res.Mult(q_pert, jac_v_fd);
    //      jac_v_fd -= r;
    //      jac_v_fd /= (2 * delta);

    //      for (int i = 0; i < jac_v.Size(); ++i)
    //      {
    //         REQUIRE(jac_v(i) == Approx(jac_v_fd(i)).margin(1e-10));
    //      }
      }
   }
}
#if 0
TEST_CASE("DGPressureForce::AssembleVector", "[DGPressureForce]")
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 2;  // templating is hard here because mesh constructors
   int num_state = dim + 2;
   adept::Stack diff_stack;
   double delta = 1e-5;

   mfem::Vector drag_dir(dim);
   double aoa_fs = 5.0 * M_PI / 180;
   drag_dir(0) = cos(aoa_fs);
   drag_dir(1) = sin(aoa_fs);

   // generate a 2 element mesh
   int num_edge = 2;
   Mesh mesh(Mesh::MakeCartesian2D(num_edge,
                                   num_edge,
                                   Element::TRIANGLE,
                                   true /* gen. edges */,
                                   1.0,
                                   1.0,
                                   true));
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
#endif