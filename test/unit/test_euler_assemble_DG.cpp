#include <random>

#include "catch.hpp"
#include "mfem.hpp"
#include "euler_fluxes.hpp"
#include "euler_integ_DG.hpp"
#include "euler_test_data.hpp"

TEST_CASE("EulerDomainIntegrator::AssembleElementGrad", "[EulerDomainIntegrator]")
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
             new mfem::DG_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             mesh.get(), fec.get(), num_state, Ordering::byVDIM));

         NonlinearForm res(fes.get());
         res.AddDomainIntegrator(new mach::EulerDomainIntegrator<2>(diff_stack, dim + 2));

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

TEST_CASE("Vortex flux::AssembleFaceGrad", "[Vortex fluxBC]")
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 2; // templating is hard here because mesh constructors
   int num_state = dim + 2;
   adept::Stack diff_stack;
   double delta = 1e-5;
   mfem::Vector qf(dim + 2);
   qf(0) = rho;
   qf(dim + 1) = rhoe;
   for (int di = 0; di < dim; ++di)
   {
      qf(di + 1) = rhou[di];
   }
   // generate a 2 element mesh
   int num_edge = 2;
   std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, Element::TRIANGLE,
                                       true /* gen. edges */, 1.0, 1.0, true));
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
             new mfem::DG_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             mesh.get(), fec.get(), num_state, Ordering::byVDIM));

         NonlinearForm res(fes.get());
         res.AddBdrFaceIntegrator(new mach::EulerBoundaryIntegrator<dim, 2>(diff_stack,
                                                                            fec.get(), dim + 2, qf));

         /// initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselinePert<dim>);
         q.ProjectCoefficient(pert);

         // initialize the vector that the Jacobian multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(num_state, randState);
         v.ProjectCoefficient(v_rand);

         /// evaluate the Jacobian and compute its product with v
         Operator &Jac = res.GetGradient(q);
         GridFunction jac_v(fes.get());
         Jac.Mult(v, jac_v);

         /// now compute the finite-difference approximation...
         GridFunction q_pert(q), r(fes.get()), jac_v_fd(fes.get());
         q_pert.Add(-delta, v);
         res.Mult(q_pert, r);
         q_pert.Add(2 * delta, v);
         res.Mult(q_pert, jac_v_fd);
         jac_v_fd -= r;
         jac_v_fd /= (2 * delta);

         for (int i = 0; i < jac_v.Size(); ++i)
         {
            //cout << jac_v_fd(i) << endl;
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)).margin(1e-10));
         }
      }
   }
}

TEST_CASE("RoeFlux::AssembleFaceGrad", "[Roe flux face]")
{
   using namespace mfem;
   using namespace euler_data;
   const int dim = 2; /// templating is hard here because mesh constructors
   int num_state = dim + 2;
   adept::Stack diff_stack;
   double delta = 1e-5;
   mfem::Vector qf(dim + 2);
   qf(0) = rho;
   qf(dim + 1) = rhoe;
   for (int di = 0; di < dim; ++di)
   {
      qf(di + 1) = rhou[di];
   }
   /// generate a 2 element mesh
   int num_edge = 2;
   std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, Element::TRIANGLE,
                                       true /* gen. edges */, 1.0, 1.0, true));
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
             new mfem::DG_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             mesh.get(), fec.get(), num_state, Ordering::byVDIM));

         NonlinearForm res(fes.get());
         res.AddInteriorFaceIntegrator(new mach::EulerFaceIntegrator<dim>(diff_stack,
                                                                          fec.get(), 1.0, num_state));

         /// initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselinePert<dim>);
         q.ProjectCoefficient(pert);

         // initialize the vector that the Jacobian multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(num_state, randState);
         v.ProjectCoefficient(v_rand);

         /// evaluate the Jacobian and compute its product with v
         Operator &Jac = res.GetGradient(q);
         GridFunction jac_v(fes.get());
         Jac.Mult(v, jac_v);

         /// now compute the finite-difference approximation...
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
