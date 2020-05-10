#include <random>

#include "catch.hpp"
#include "mfem.hpp"

#include "electromag_integ.hpp"
#include "electromag_test_data.hpp"

TEST_CASE("CurlCurlNLFIntegrator::AssembleElementGrad - linear",
          "[CurlCurlNLFIntegrator]")
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

TEST_CASE("CurlCurlNLFIntegrator::AssembleElementGrad",
          "[CurlCurlNLFIntegrator]")
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

TEST_CASE("CurlCurlNLFIntegrator::AssembleElementGrad - Nonlinear",
          "[CurlCurlNLFIntegrator]")
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

TEST_CASE("CurlCurlNLFIntegrator::AssembleRHSElementVect",
          "[CurlCurlNLFIntegrator]")
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
   mesh->EnsureNodes();
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         // get the finite-element space for the state and adjoint
         std::unique_ptr<FiniteElementCollection> fec(
            new ND_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
            mesh.get(), fec.get()));

         std::unique_ptr<mach::StateCoefficient> nu(
            new NonLinearCoefficient());

         // initialize state and adjoint; here we randomly perturb a constant state
         GridFunction state(fes.get()), adjoint(fes.get());
         VectorFunctionCoefficient pert(3, randState);
         state.ProjectCoefficient(pert);
         adjoint.ProjectCoefficient(pert);

         // we use res for finite-difference approximation
         NonlinearForm res(fes.get());
         res.AddDomainIntegrator(new mach::CurlCurlNLFIntegrator(nu.get(), &state, &adjoint));

         // extract mesh nodes and get their finite-element space
         GridFunction *x_nodes = mesh->GetNodes();
         FiniteElementSpace *mesh_fes = x_nodes->FESpace();

         // build the nonlinear form for d(psi^T R)/dx 
         LinearForm dfdx(mesh_fes);
         dfdx.AddDomainIntegrator(
            new mach::CurlCurlNLFIntegrator(nu.get(),
               &state, &adjoint));
         dfdx.Assemble();

         // initialize the vector that we use to perturb the mesh nodes
         GridFunction v(mesh_fes);
         VectorFunctionCoefficient v_rand(3, randState);
         v.ProjectCoefficient(v_rand);

         // contract dfdx with v
         double dfdx_v = dfdx * v;

         // now compute the finite-difference approximation...
         GridFunction x_pert(*x_nodes);
         GridFunction r(fes.get());
         x_pert.Add(delta, v);
         mesh->SetNodes(x_pert);
         fes->Update();
         res.Mult(state, r);
         // double dfdx_v_fd = res.GetEnergy(state);
         double dfdx_v_fd = adjoint * r;
         x_pert.Add(-2 * delta, v);
         mesh->SetNodes(x_pert);
         fes->Update();
         res.Mult(state, r);
         // dfdx_v_fd -= res.GetEnergy(state);
         dfdx_v_fd -= adjoint * r;
         dfdx_v_fd /= (2 * delta);
         mesh->SetNodes(*x_nodes); // remember to reset the mesh nodes
         fes->Update();

         // std::cout << "Order: " << p << "\n";
         // std::cout << "dfdx_v = " << dfdx_v << "\n";
         // std::cout << "dfdx_v_fd = " << dfdx_v_fd << "\n";

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-10));
      }
   }
}

TEST_CASE("VectorFECurldJdXIntegerator::AssembleRHSElementVect",
          "[VectorFECurldJdXIntegerator]")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3;  // templating is hard here because mesh constructors
   // static adept::Stack diff_stack;
   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 2;
   std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, num_edge,
                              Element::TETRAHEDRON, true /* gen. edges */, 1.0,
                              1.0, 1.0, true));
   mesh->ReorientTetMesh();
   mesh->EnsureNodes();
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         // get the finite-element space for the adjoint
         std::unique_ptr<FiniteElementCollection> nd_fec(
            new ND_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> nd_fes(new FiniteElementSpace(
            mesh.get(), nd_fec.get()));

         // get the finite-element space for the magnetization grid function
         std::unique_ptr<FiniteElementCollection> rt_fec(
            new RT_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> rt_fes(new FiniteElementSpace(
            mesh.get(), rt_fec.get()));

         /// We use a linear coefficient since the magnets are linear
         std::unique_ptr<mach::StateCoefficient> nu(
            new LinearCoefficient());

         // initialize magnetization source and adjoint; here we randomly perturb a constant state
         GridFunction M(rt_fes.get()), adjoint(nd_fes.get());
         VectorFunctionCoefficient pert(3, randState);
         M.ProjectCoefficient(pert);
         adjoint.ProjectCoefficient(pert);

         // we use res for finite-difference approximation
         // MixedBilinearForm res(rt_fes.get(), nd_fes.get());
         // res.AddDomainIntegrator(new VectorFECurlIntegrator(*nu));

         // res.Assemble();
         // res.Finalize();

         // extract mesh nodes and get their finite-element space
         GridFunction *x_nodes = mesh->GetNodes();
         FiniteElementSpace *mesh_fes = x_nodes->FESpace();

         // build the nonlinear form for d(psi^T R)/dx 
         LinearForm dfdx(mesh_fes);
         dfdx.AddDomainIntegrator(
            new mach::VectorFECurldJdXIntegerator(nu.get(), &M, &adjoint));
         dfdx.Assemble();

         // initialize the vector that we use to perturb the mesh nodes
         GridFunction v(mesh_fes);
         VectorFunctionCoefficient v_rand(3, randState);
         v.ProjectCoefficient(v_rand);

         // contract dfdx with v
         double dfdx_v = dfdx * v;

         // now compute the finite-difference approximation...
         GridFunction x_pert(*x_nodes);
         GridFunction r(nd_fes.get());
         GridFunction r2(nd_fes.get());
         r = 0.0;
         x_pert.Add(delta, v);
         mesh->SetNodes(x_pert);
         rt_fes->Update();
         nd_fes->Update();
         MixedBilinearForm res1(rt_fes.get(), nd_fes.get());
         res1.AddDomainIntegrator(new VectorFECurlIntegrator(*nu));
         res1.Update();
         res1.Assemble();
         res1.Finalize();
         res1.AddMult(M, r);
         double dfdx_v_fd = adjoint * r;
         x_pert.Add(-2 * delta, v);
         mesh->SetNodes(x_pert);
         rt_fes->Update();
         nd_fes->Update();
         MixedBilinearForm res2(rt_fes.get(), nd_fes.get());
         res2.AddDomainIntegrator(new VectorFECurlIntegrator(*nu));
         res2.Update();
         res2.Assemble();
         res2.Finalize();
         r2 = 0.0;
         res2.AddMult(M, r2);
         dfdx_v_fd -= adjoint * r2;
         dfdx_v_fd /= (2 * delta);
         mesh->SetNodes(*x_nodes); // remember to reset the mesh nodes

         std::cout << "Order: " << p << "\n";
         std::cout << "dfdx_v = " << dfdx_v << "\n";
         std::cout << "dfdx_v_fd = " << dfdx_v_fd << "\n";
         std::cout << dfdx_v_fd / dfdx_v << "\n";

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-10));
      }
   }
}

TEST_CASE("MagneticCoenergyIntegrator::AssembleElementVector",
          "[MagneticCoenergyIntegrator]")
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
            new mach::MagneticCoenergyIntegrator(q, nu.get()));

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

         REQUIRE(dJdu_dot_v == Approx(dJdu_dot_v_fd));
      }
   }
}

TEST_CASE("MagneticCoenergyIntegrator::AssembleElementRHSVect",
          "[MagneticCoenergyIntegrator]")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3; // templating is hard here because mesh constructors
   double delta = 1e-5;

   // generate a 2 element mesh
   int num_edge = 1;
   std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, num_edge,
                                       Element::TETRAHEDRON,
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
            new mach::MagneticCoenergyIntegrator(q, nu.get()));
         dJdx.Assemble();
         double dJdx_dot_v = dJdx * v;

         // now compute the finite-difference approximation...
         NonlinearForm functional(fes.get());
         functional.AddDomainIntegrator(
            new mach::MagneticCoenergyIntegrator(q, nu.get()));

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
         mesh->SetNodes(*x_nodes); // remember to reset the mesh nodes
         fes->Update();

         REQUIRE(dJdx_dot_v == Approx(dJdx_dot_v_fd));
      }
   }
}

TEST_CASE("BNormIntegrator::AssembleElementVector",
          "[BNormIntegrator]")
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

         REQUIRE(dJdu_dot_v == Approx(dJdu_dot_v_fd));
      }
   }
}

TEST_CASE("BNormdJdX::AssembleRHSElementVect",
          "[BNormdJdX]")
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
         mesh->SetNodes(*x_nodes); // remember to reset the mesh nodes
         fes->Update();

         REQUIRE(dJdx_dot_v == Approx(dJdx_dot_v_fd));
      }
   }
}

TEST_CASE("nuBNormIntegrator::AssembleElementVector",
          "[nuBNormIntegrator]")
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

         REQUIRE(dJdu_dot_v == Approx(dJdu_dot_v_fd));
      }
   }
}

TEST_CASE("nuBNormdJdX::AssembleRHSElementVect",
          "[nuBNormdJdX]")
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
         mesh->SetNodes(*x_nodes); // remember to reset the mesh nodes
         fes->Update();

         REQUIRE(dJdx_dot_v == Approx(dJdx_dot_v_fd));
      }
   }
}

TEST_CASE("nuFuncIntegrator::AssembleRHSElementVect",
          "[nuFuncIntegrator]")
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
            new mach::nuFuncIntegrator(&q, nu.get()));
         dJdx.Assemble();
         double dJdx_dot_v = dJdx * v;

         // now compute the finite-difference approximation...
         NonlinearForm functional(fes.get());
         functional.AddDomainIntegrator(
            new mach::nuFuncIntegrator(nu.get()));

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
         mesh->SetNodes(*x_nodes); // remember to reset the mesh nodes
         fes->Update();

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