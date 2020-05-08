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
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
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

         std::unique_ptr<mfem::Coefficient> nu_linear(
            new ConstantCoefficient(1.0));
         /// Bilinear Form
         BilinearForm blf(fes.get());
         blf.AddDomainIntegrator(new CurlCurlIntegrator(*nu_linear.get()));

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

   for (int p = 1; p <= 4; ++p)
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
            REQUIRE( jac_v(i) == Approx(jac_v_fd(i)).margin(1e-10) );
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

   for (int p = 1; p <= 4; ++p)
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

         std::unique_ptr<mach::StateCoefficient> nu(
            new NonLinearCoefficient());

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
         res.Mult(a, r);
         a.Add(2*delta, v);
         res.Mult(a, jac_v_fd);
         jac_v_fd -= r;
         jac_v_fd /= (2*delta);

         for (int i = 0; i < jac_v.Size(); ++i)
         {
            REQUIRE( jac_v(i) == Approx(jac_v_fd(i)).margin(1e-10) );
         }
      }
   }
}

TEST_CASE("MagneticCoenergyIntegrator::AssembleElementVector", "[MagneticCoenergyIntegrator]")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3; // templating is hard here because mesh constructors
   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 1;
   std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, num_edge, Element::TETRAHEDRON,
                                       true /* gen. edges */, 1.0, 1.0, 1.0, true));
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
            new ND_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
            mesh.get(), fec.get()));

         NonlinearForm functional(fes.get());

         const double scale = 0.01;

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(3, randState);
         q.ProjectCoefficient(pert);

         std::unique_ptr<mach::StateCoefficient> nu(
            // new LinearCoefficient());
            new NonLinearCoefficient());

         functional.AddDomainIntegrator(
            new mach::MagneticCoenergyIntegrator(nu.get()));

         // initialize the vector that dJdu multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(3, randState);
         v.ProjectCoefficient(v_rand);

         // evaluate dJdu and compute its product with v
         GridFunction dJdu(fes.get());
         functional.Mult(q, dJdu);
         double dJdu_dot_v = InnerProduct(dJdu, v);

         // now compute the finite-difference approximation...
         GridFunction q_pert(q);
         q_pert.Add(-delta, v);
         double dJdu_dot_v_fd = -functional.GetEnergy(q_pert);
         q_pert.Add(2 * delta, v);
         dJdu_dot_v_fd += functional.GetEnergy(q_pert);
         dJdu_dot_v_fd /= (2 * delta);
         // std::cout << "dJdu_dot_v = " << dJdu_dot_v << std::endl;
         // std::cout << "dJdu_dot_v_fd = " << dJdu_dot_v_fd << std::endl;
         REQUIRE(dJdu_dot_v == Approx(dJdu_dot_v_fd));
      }
   }
}

TEST_CASE("BNormIntegrator::AssembleElementVector", "[BNormIntegrator]")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3; // templating is hard here because mesh constructors
   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 1;
   std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, num_edge, Element::TETRAHEDRON,
                                       true /* gen. edges */, 1.0, 1.0, 1.0, true));
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
            new ND_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
            mesh.get(), fec.get()));

         NonlinearForm functional(fes.get());

         const double scale = 0.01;

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(3, randState);
         q.ProjectCoefficient(pert);

         functional.AddDomainIntegrator(
            new mach::BNormIntegrator());

         // initialize the vector that dJdu multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(3, randState);
         v.ProjectCoefficient(v_rand);

         // evaluate dJdu and compute its product with v
         GridFunction dJdu(fes.get());
         functional.Mult(q, dJdu);
         double dJdu_dot_v = InnerProduct(dJdu, v);

         // now compute the finite-difference approximation...
         GridFunction q_pert(q);
         q_pert.Add(-delta, v);
         double dJdu_dot_v_fd = -functional.GetEnergy(q_pert);
         q_pert.Add(2 * delta, v);
         dJdu_dot_v_fd += functional.GetEnergy(q_pert);
         dJdu_dot_v_fd /= (2 * delta);
         // std::cout << "dJdu_dot_v = " << dJdu_dot_v << std::endl;
         // std::cout << "dJdu_dot_v_fd = " << dJdu_dot_v_fd << std::endl;
         REQUIRE(dJdu_dot_v == Approx(dJdu_dot_v_fd));
      }
   }
}

TEST_CASE("BNormdJdX::AssembleRHSElementVect", "[BNormdJdX]")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3; // templating is hard here because mesh constructors
   double delta = 1e-5;

   // generate a 2 element mesh
   int num_edge = 1;
   std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, num_edge, Element::TETRAHEDRON,
                                       true /* gen. edges */, 1.0, 1.0, 1.0, true));
   mesh->EnsureNodes();
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         // get the finite-element space for the state
         std::unique_ptr<FiniteElementCollection> fec(
             new ND_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             mesh.get(), fec.get()));

         // extract mesh nodes and get their finite-element space
         GridFunction *x_nodes = mesh->GetNodes();
         FiniteElementSpace *mesh_fes = x_nodes->FESpace();

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(3, randState);
         q.ProjectCoefficient(pert);

         // initialize the vector that dJdx multiplies
         GridFunction v(mesh_fes);
         VectorFunctionCoefficient v_rand(dim, randState);
         v.ProjectCoefficient(v_rand);

         // evaluate dJdx and compute its product with v
         // GridFunction dJdx(*x_nodes);
         LinearForm dJdx(mesh_fes);
         dJdx.AddDomainIntegrator(
            new mach::BNormdJdx(q));
         dJdx.Assemble();
         double dJdx_dot_v = dJdx * v;

         // now compute the finite-difference approximation...
         NonlinearForm functional(fes.get());
         functional.AddDomainIntegrator(
            new mach::BNormIntegrator());

         double delta = 1e-5;
         GridFunction x_pert(*x_nodes);
         x_pert.Add(-delta, v);
         mesh->SetNodes(x_pert);
         fes->Update();
         double dJdx_dot_v_fd = -functional.GetEnergy(q);
         x_pert.Add(2 * delta, v);
         mesh->SetNodes(x_pert);
         fes->Update();
         dJdx_dot_v_fd += functional.GetEnergy(q);
         dJdx_dot_v_fd /= (2 * delta);
         // std::cout << "dJdx_dot_v = " << dJdx_dot_v << std::endl;
         // std::cout << "dJdx_dot_v_fd = " << dJdx_dot_v_fd << std::endl;
         REQUIRE(dJdx_dot_v == Approx(dJdx_dot_v_fd));
      }
   }
}

TEST_CASE("nuBNormIntegrator::AssembleElementVector", "[nuBNormIntegrator]")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3; // templating is hard here because mesh constructors
   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 1;
   std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, num_edge, Element::TETRAHEDRON,
                                       true /* gen. edges */, 1.0, 1.0, 1.0, true));
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
            new ND_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
            mesh.get(), fec.get()));

         NonlinearForm functional(fes.get());

         const double scale = 0.01;

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(3, randState);
         q.ProjectCoefficient(pert);

         std::unique_ptr<mach::StateCoefficient> nu(
            new NonLinearCoefficient());

         functional.AddDomainIntegrator(
            new mach::nuBNormIntegrator(nu.get()));

         // initialize the vector that dJdu multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(3, randState);
         v.ProjectCoefficient(v_rand);

         // evaluate dJdu and compute its product with v
         GridFunction dJdu(fes.get());
         functional.Mult(q, dJdu);
         double dJdu_dot_v = InnerProduct(dJdu, v);

         // now compute the finite-difference approximation...
         GridFunction q_pert(q);
         q_pert.Add(-delta, v);
         double dJdu_dot_v_fd = -functional.GetEnergy(q_pert);
         q_pert.Add(2 * delta, v);
         dJdu_dot_v_fd += functional.GetEnergy(q_pert);
         dJdu_dot_v_fd /= (2 * delta);
         // std::cout << "dJdu_dot_v = " << dJdu_dot_v << std::endl;
         // std::cout << "dJdu_dot_v_fd = " << dJdu_dot_v_fd << std::endl;
         REQUIRE(dJdu_dot_v == Approx(dJdu_dot_v_fd));
      }
   }
}

TEST_CASE("nuBNormdJdX::AssembleRHSElementVect", "[nuBNormdJdX]")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3; // templating is hard here because mesh constructors
   double delta = 1e-5;

   // generate a 2 element mesh
   int num_edge = 1;
   std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, num_edge, Element::TETRAHEDRON,
                                       true /* gen. edges */, 1.0, 1.0, 1.0, true));
   mesh->EnsureNodes();
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         // get the finite-element space for the state
         std::unique_ptr<FiniteElementCollection> fec(
             new ND_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             mesh.get(), fec.get()));

         // extract mesh nodes and get their finite-element space
         GridFunction *x_nodes = mesh->GetNodes();
         FiniteElementSpace *mesh_fes = x_nodes->FESpace();

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(3, randState);
         q.ProjectCoefficient(pert);

         // initialize the vector that dJdx multiplies
         GridFunction v(mesh_fes);
         VectorFunctionCoefficient v_rand(dim, randState);
         v.ProjectCoefficient(v_rand);

         std::unique_ptr<mach::StateCoefficient> nu(
            new NonLinearCoefficient());

         // evaluate dJdx and compute its product with v
         // GridFunction dJdx(*x_nodes);
         LinearForm dJdx(mesh_fes);
         dJdx.AddDomainIntegrator(
            new mach::nuBNormdJdx(q, nu.get()));
         dJdx.Assemble();
         double dJdx_dot_v = dJdx * v;

         // now compute the finite-difference approximation...
         NonlinearForm functional(fes.get());
         functional.AddDomainIntegrator(
            new mach::nuBNormIntegrator(nu.get()));

         double delta = 1e-5;
         GridFunction x_pert(*x_nodes);
         x_pert.Add(-delta, v);
         mesh->SetNodes(x_pert);
         fes->Update();
         double dJdx_dot_v_fd = -functional.GetEnergy(q);
         x_pert.Add(2 * delta, v);
         mesh->SetNodes(x_pert);
         fes->Update();
         dJdx_dot_v_fd += functional.GetEnergy(q);
         dJdx_dot_v_fd /= (2 * delta);
         std::cout << "dJdx_dot_v = " << dJdx_dot_v << std::endl;
         std::cout << "dJdx_dot_v_fd = " << dJdx_dot_v_fd << std::endl;
         REQUIRE(dJdx_dot_v == Approx(dJdx_dot_v_fd));
      }
   }
}

/// commenting out because I removed support for nonlinear magnets for now
// TEST_CASE("MagnetizationIntegrator::AssembleElementGrad - Nonlinear", "[MagnetizationIntegrator]")
// {
//    using namespace mfem;
//    using namespace electromag_data;

//    const int dim = 3;  // templating is hard here because mesh constructors
//    // static adept::Stack diff_stack;
//    double delta = 1e-5;

//    // generate a 6 element mesh
//    int num_edge = 1;
//    std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, num_edge,
//                               Element::TETRAHEDRON, true /* gen. edges */, 1.0,
//                               1.0, 1.0, true));

//    for (int p = 1; p <= 4; ++p)
//    {
//       DYNAMIC_SECTION( "...for degree p = " << p )
//       {
//          std::unique_ptr<FiniteElementCollection> fec(
//             new ND_FECollection(p, dim));
//          std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
//             mesh.get(), fec.get()));

//          // initialize state; here we randomly perturb a constant state
//          GridFunction a(fes.get());
//          VectorFunctionCoefficient pert(3, randBaselinePert);
//          a.ProjectCoefficient(pert);

//          std::unique_ptr<mach::StateCoefficient> nu(
//             new NonLinearCoefficient());

//          std::unique_ptr<mfem::VectorCoefficient> mag(
//             new mfem::VectorFunctionCoefficient(3, mag_func));

//          NonlinearForm res(fes.get());
//          res.AddDomainIntegrator(new mach::MagnetizationIntegrator(nu.get(),
//                                                                    mag.get()));

//          // initialize the vector that the Jacobian multiplies
//          GridFunction v(fes.get());
//          VectorFunctionCoefficient v_rand(3, randState);
//          v.ProjectCoefficient(v_rand);

//          // evaluate the Jacobian and compute its product with v
//          Operator& Jac = res.GetGradient(a);
//          GridFunction jac_v(fes.get());
//          Jac.Mult(v, jac_v);

//          // now compute the finite-difference approximation...
//          GridFunction r(fes.get()), jac_v_fd(fes.get());
//          a.Add(-delta, v);
//          res.Mult(a, r);
//          a.Add(2*delta, v);
//          res.Mult(a, jac_v_fd);
//          jac_v_fd -= r;
//          jac_v_fd /= (2*delta);

//          for (int i = 0; i < jac_v.Size(); ++i)
//          {
//             REQUIRE( jac_v(i) == Approx(jac_v_fd(i)) );
//          }
//       }
//    }
// }