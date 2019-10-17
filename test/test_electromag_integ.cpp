#include <random>

#include "catch.hpp"
#include "mfem.hpp"

#include "electromag_integ.hpp"
#include "electromag_test_data.hpp"

TEST_CASE("CurlCurlNLFIntegrator::AssembleElementGrad - linear", "Works for linear")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3;  // templating is hard here because mesh constructors
   // static adept::Stack diff_stack;
   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 1;
   std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, num_edge,
                              Element::TETRAHEDRON, true /* gen. edges */, 1.0,
                              1.0, 1.0, true));

   for (int p = 1; p <= 1; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         int ndof = p*(p + 2)*(p + 3)/2;
         std::unique_ptr<FiniteElementCollection> fec(
            new ND_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
            mesh.get(), fec.get()));

         NonlinearForm res(fes.get());

         std::unique_ptr<mach::StateCoefficient> nu(
            new LinearCoefficient(1.0));

         res.AddDomainIntegrator(new mach::CurlCurlNLFIntegrator(nu.get()));

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         // I think this should be 1? Only have one DOF per "node"
         VectorFunctionCoefficient pert(1, randBaselinePert);
         q.ProjectCoefficient(pert);

         // initialize the vector that the Jacobian multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(1, randState);
         v.ProjectCoefficient(v_rand);

         // evaluate the Jacobian and compute its product with v
         Operator& Jac = res.GetGradient(q);
         GridFunction jac_v(fes.get());
         Jac.Mult(v, jac_v);

         /// Bilinear Form
         BilinearForm blf(fes.get());
         blf.AddDomainIntegrator(new CurlCurlIntegrator(*nu.get()));

         blf.Assemble();
         GridFunction blf_v(fes.get());
         blf.Mult(v, blf_v);

         for (int i = 0; i < jac_v.Size(); ++i)
         {
            REQUIRE( jac_v(i) == Approx(blf_v(i)) );
         }
      }
   }
}

TEST_CASE("CurlCurlNLFIntegrator::AssembleElementGrad", "[CurlCurlNLFIntegrator]")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3;  // templating is hard here because mesh constructors
   // static adept::Stack diff_stack;
   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 1;
   std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, num_edge,
                              Element::TETRAHEDRON, true /* gen. edges */, 1.0,
                              1.0, 1.0, true));

   for (int p = 1; p <= 1; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         std::unique_ptr<FiniteElementCollection> fec(
            new ND_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
            mesh.get(), fec.get()));

         NonlinearForm res(fes.get());

         std::unique_ptr<mach::StateCoefficient> nu(
            new LinearCoefficient());

         res.AddDomainIntegrator(new mach::CurlCurlNLFIntegrator(nu.get()));

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(3, randBaselinePert);
         q.ProjectCoefficient(pert);

         // initialize the vector that the Jacobian multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(3, randState);
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

TEST_CASE("CurlCurlNLFIntegrator::AssembleElementGrad - Nonlinear", "[CurlCurlNLFIntegrator]")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3;  // templating is hard here because mesh constructors
   // static adept::Stack diff_stack;
   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 1;
   std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, num_edge,
                              Element::TETRAHEDRON, true /* gen. edges */, 1.0,
                              1.0, 1.0, true));

   for (int p = 1; p <= 1; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         std::unique_ptr<FiniteElementCollection> fec(
            new ND_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
            mesh.get(), fec.get()));

         // initialize state; here we randomly perturb a constant state
         GridFunction a(fes.get());
         VectorFunctionCoefficient pert(3, randBaselinePert);
         a.ProjectCoefficient(pert);

         // create H(div) finite element space and grid function
         std::unique_ptr<FiniteElementCollection> fec_b(
            new RT_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes_b(new FiniteElementSpace(
            mesh.get(), fec_b.get()));
         GridFunction b(fes_b.get());

         // compute B = curl(A)
         DiscreteLinearOperator curl(fes.get(), fes_b.get());
         curl.AddDomainInterpolator(new CurlInterpolator);
         curl.Assemble();
         curl.Finalize();
         curl.Mult(a, b);

         std::unique_ptr<mach::StateCoefficient> nu(
            new NonLinearCoefficient(&b));

         NonlinearForm res(fes.get());
         res.AddDomainIntegrator(new mach::CurlCurlNLFIntegrator(nu.get()));

         // initialize the vector that the Jacobian multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(3, randState);
         v.ProjectCoefficient(v_rand);

         // evaluate the Jacobian and compute its product with v
         Operator& Jac = res.GetGradient(a);
         GridFunction jac_v(fes.get());
         Jac.Mult(v, jac_v);

         // now compute the finite-difference approximation...
         GridFunction r(fes.get()), jac_v_fd(fes.get());
         a.Add(-delta, v);
         curl.Mult(a, b);
         res.Mult(a, r);
         a.Add(2*delta, v);
         curl.Mult(a, b);
         res.Mult(a, jac_v_fd);
         jac_v_fd -= r;
         jac_v_fd /= (2*delta);

         for (int i = 0; i < jac_v.Size(); ++i)
         {
            REQUIRE( jac_v(i) == Approx(jac_v_fd(i)) );
         }
      }
   }
}