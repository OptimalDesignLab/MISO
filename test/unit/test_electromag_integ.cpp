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

   int num_edge = 2;
   std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, num_edge,
                              Element::TETRAHEDRON, true /* gen. edges */, 1.0,
                              1.0, 1.0, true));
   mesh->ReorientTetMesh();
   mesh->EnsureNodes();

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
         VectorFunctionCoefficient pert(1, randBaselineVectorPert);
         q.ProjectCoefficient(pert);

         // initialize the vector that the Jacobian multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(1, randVectorState);
         v.ProjectCoefficient(v_rand);

         // evaluate the Jacobian and compute its product with v
         Operator& Jac = res.GetGradient(q);
         GridFunction jac_v(fes.get());
         Jac.Mult(v, jac_v);

         GridFunction res_v(fes.get());
         res.Mult(v, res_v);

         std::unique_ptr<mfem::Coefficient> nu_linear(
            new ConstantCoefficient(1.0));
         /// Bilinear Form
         BilinearForm blf(fes.get());
         blf.AddDomainIntegrator(new CurlCurlIntegrator(*nu_linear));

         blf.Assemble();
         GridFunction blf_v(fes.get());
         blf.Mult(v, blf_v);

         for (int i = 0; i < jac_v.Size(); ++i)
         {
            REQUIRE( res_v(i) == Approx(blf_v(i)) );
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
   int num_edge = 2;
   std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, num_edge,
                              Element::TETRAHEDRON, true /* gen. edges */, 1.0,
                              1.0, 1.0, true));
   mesh->ReorientTetMesh();
   mesh->EnsureNodes();

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

         // GridFunction A(fes.get());
         // VectorFunctionCoefficient a0(3, [](const Vector &x, Vector &a)
         // {
         //    a(0) = -0.05*x(1);
         //    a(1) = 0.05*x(0);
         //    a(2) = 0.0;
         // });
         // A.ProjectCoefficient(a0);


         GridFunction q(fes.get());
         // res.Mult(A, q);
         // Operator& tJac = res.GetGradient(A);


         // initialize state; here we randomly perturb a constant state
         VectorFunctionCoefficient pert(3, randBaselineVectorPert);
         q.ProjectCoefficient(pert);

         // initialize the vector that the Jacobian multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(3, randVectorState);
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
            REQUIRE( jac_v(i) == Approx(jac_v_fd(i)).margin(1e-8) );
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
   int num_edge = 2;
   std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, num_edge,
                              Element::TETRAHEDRON, true /* gen. edges */, 1.0,
                              1.0, 1.0, true));
   mesh->ReorientTetMesh();
   mesh->EnsureNodes();

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
         VectorFunctionCoefficient pert(3, randBaselineVectorPert);
         a.ProjectCoefficient(pert);

         std::unique_ptr<mach::StateCoefficient> nu(
            new NonLinearCoefficient());

         NonlinearForm res(fes.get());
         res.AddDomainIntegrator(new mach::CurlCurlNLFIntegrator(nu.get()));

         // initialize the vector that the Jacobian multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(3, randVectorState);
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
            REQUIRE( jac_v(i) == Approx(jac_v_fd(i)).margin(1e-6) );
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
   mesh->ReorientTetMesh();
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
         VectorFunctionCoefficient pert(3, randVectorState);
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
         VectorFunctionCoefficient v_rand(3, randVectorState);
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

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-8));
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
   auto mesh = getMesh(num_edge, num_edge);
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
         VectorFunctionCoefficient pert(3, randVectorState);
         VectorFunctionCoefficient mag(3, vectorFunc, vectorFuncRevDiff);

         // /// Costruct coefficient
         // mach::VectorMeshDependentCoefficient mag(dim);
         // std::unique_ptr<mfem::VectorCoefficient> coeff1(
         //    new VectorFunctionCoefficient(dim, vectorFunc, vectorFuncRevDiff));
         // std::unique_ptr<mfem::VectorCoefficient> coeff2(
         //    new VectorFunctionCoefficient(dim, vectorFunc2, vectorFunc2RevDiff));
         // mag.addCoefficient(1, move(coeff1));
         // mag.addCoefficient(2, move(coeff2));

         M.ProjectCoefficient(mag);
         // M.ProjectCoefficient(pert);
         adjoint.ProjectCoefficient(pert);

         // we use res for finite-difference approximation
         MixedBilinearForm res(rt_fes.get(), nd_fes.get());
         res.AddDomainIntegrator(new VectorFECurlIntegrator(*nu));

         // extract mesh nodes and get their finite-element space
         GridFunction *x_nodes = mesh->GetNodes();
         FiniteElementSpace *mesh_fes = x_nodes->FESpace();

         // build the nonlinear form for d(psi^T R)/dx 
         LinearForm dfdx(mesh_fes);
         dfdx.AddDomainIntegrator(
            new mach::VectorFECurldJdXIntegerator(nu.get(), &M, &adjoint, &mag));
         dfdx.Assemble();

         // initialize the vector that we use to perturb the mesh nodes
         GridFunction v(mesh_fes);
         VectorFunctionCoefficient v_rand(3, randVectorState);
         v.ProjectCoefficient(v_rand);

         // contract dfdx with v
         double dfdx_v = dfdx * v;

         // now compute the finite-difference approximation...
         GridFunction x_pert(*x_nodes);
         GridFunction r(nd_fes.get());
         x_pert.Add(delta, v);
         mesh->SetNodes(x_pert);
         rt_fes->Update();
         nd_fes->Update();
         M.Update();
         M.ProjectCoefficient(mag);
         res.Update();
         res.Assemble();
         res.Finalize();
         res.Mult(M, r);
         double dfdx_v_fd = adjoint * r;
         x_pert.Add(-2 * delta, v);
         mesh->SetNodes(x_pert);
         rt_fes->Update();
         nd_fes->Update();
         M.Update();
         M.ProjectCoefficient(mag);
         res.Update();
         res.Assemble();
         res.Finalize();
         res.Mult(M, r);
         dfdx_v_fd -= adjoint * r;
         dfdx_v_fd /= (2 * delta);
         mesh->SetNodes(*x_nodes); // remember to reset the mesh nodes

         // std::cout << "dfdx_v " << dfdx_v << "\n";
         // std::cout << "dfdx_v_fd " << dfdx_v_fd << "\n";

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-8));
      }
   }
}

TEST_CASE("VectorFEMassdJdXIntegerator::AssembleRHSElementVect",
          "[VectorFEMassdJdXIntegerator]")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3;  // templating is hard here because mesh constructors
   // static adept::Stack diff_stack;
   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 2;
   auto mesh = getMesh(num_edge, num_edge);
   mesh->ReorientTetMesh();
   mesh->EnsureNodes();
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         // get the finite-element space for the adjoint
         std::unique_ptr<FiniteElementCollection> fec(
            new ND_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
            mesh.get(), fec.get()));

         // initialize magnetization source and adjoint; here we randomly perturb a constant state
         GridFunction J(fes.get()), adjoint(fes.get());
         VectorFunctionCoefficient pert(3, randVectorState);
         VectorFunctionCoefficient current(3, vectorFunc, vectorFuncRevDiff);

         /// Costruct coefficient
         // mach::VectorMeshDependentCoefficient current(dim);
         // std::unique_ptr<mfem::VectorCoefficient> coeff1(
         //    new VectorFunctionCoefficient(dim, vectorFunc, vectorFuncRevDiff));
         // std::unique_ptr<mfem::VectorCoefficient> coeff2(
         //    new VectorFunctionCoefficient(dim, vectorFunc2, vectorFunc2RevDiff));
         // current.addCoefficient(1, move(coeff1));
         // current.addCoefficient(2, move(coeff2));

         J.ProjectCoefficient(current);
         // J.ProjectCoefficient(pert);
         adjoint.ProjectCoefficient(pert);

         // we use res for finite-difference approximation
         BilinearForm res(fes.get());
         res.AddDomainIntegrator(new VectorFEMassIntegrator);

         // extract mesh nodes and get their finite-element space
         GridFunction *x_nodes = mesh->GetNodes();
         FiniteElementSpace *mesh_fes = x_nodes->FESpace();

         // build the nonlinear form for d(psi^T R)/dx 
         LinearForm dfdx(mesh_fes);
         dfdx.AddDomainIntegrator(
            new mach::VectorFEMassdJdXIntegerator(&J, &adjoint, &current));
         dfdx.Assemble();

         // initialize the vector that we use to perturb the mesh nodes
         GridFunction v(mesh_fes);
         VectorFunctionCoefficient v_rand(3, randVectorState);
         v.ProjectCoefficient(v_rand);

         // contract dfdx with v
         double dfdx_v = dfdx * v;

         // now compute the finite-difference approximation...
         GridFunction x_pert(*x_nodes);
         GridFunction r(fes.get());
         x_pert.Add(delta, v);
         mesh->SetNodes(x_pert);
         fes->Update();
         J.Update();
         J.ProjectCoefficient(current);
         res.Update();
         res.Assemble();
         res.Finalize();
         res.Mult(J, r);
         double dfdx_v_fd = adjoint * r;
         x_pert.Add(-2 * delta, v);
         mesh->SetNodes(x_pert);
         fes->Update();
         J.Update();
         J.ProjectCoefficient(current);
         res.Update();
         res.Assemble();
         res.Finalize();
         res.Mult(J, r);
         dfdx_v_fd -= adjoint * r;
         dfdx_v_fd /= (2 * delta);
         mesh->SetNodes(*x_nodes); // remember to reset the mesh nodes

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-8));
      }
   }
}

TEST_CASE("VectorFEWeakDivergencedJdXIntegrator::AssembleRHSElementVect",
          "[VectorFEWeakDivergencedJdXIntegrator]")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3;
   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 2;
   auto mesh = getMesh(num_edge, num_edge);
   mesh->ReorientTetMesh();
   mesh->EnsureNodes();
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         // get the finite-element space for the current grid function
         std::unique_ptr<FiniteElementCollection> nd_fec(
            new ND_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> nd_fes(new FiniteElementSpace(
            mesh.get(), nd_fec.get()));
            
         // get the finite-element space for the adjoint
         std::unique_ptr<FiniteElementCollection> h1_fec(
            new H1_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> h1_fes(new FiniteElementSpace(
            mesh.get(), h1_fec.get()));

         // initialize current source and adjoint
         // here we randomly perturb a constant state
         GridFunction c(nd_fes.get()), adjoint(h1_fes.get());
         VectorFunctionCoefficient pert(3, randVectorState);
         FunctionCoefficient adj_pert(randState);
         VectorFunctionCoefficient current(3, vectorFunc, vectorFuncRevDiff);

         /// Costruct coefficient
         // mach::VectorMeshDependentCoefficient current(dim);
         // std::unique_ptr<mfem::VectorCoefficient> coeff1(
         //    new VectorFunctionCoefficient(dim, vectorFunc, vectorFuncRevDiff));
         // std::unique_ptr<mfem::VectorCoefficient> coeff2(
         //    new VectorFunctionCoefficient(dim, vectorFunc2, vectorFunc2RevDiff));
         // current.addCoefficient(1, move(coeff1));
         // current.addCoefficient(2, move(coeff2));

         c.ProjectCoefficient(current);
         // c.ProjectCoefficient(pert);
         adjoint.ProjectCoefficient(adj_pert);

         // we use res for finite-difference approximation
         MixedBilinearForm res(nd_fes.get(), h1_fes.get());
         res.AddDomainIntegrator(new VectorFEWeakDivergenceIntegrator);

         // extract mesh nodes and get their finite-element space
         GridFunction *x_nodes = mesh->GetNodes();
         FiniteElementSpace *mesh_fes = x_nodes->FESpace();

         // build the nonlinear form for d(psi^T R)/dx 
         LinearForm dfdx(mesh_fes);
         dfdx.AddDomainIntegrator(
            new mach::VectorFEWeakDivergencedJdXIntegrator(&c, &adjoint, &current));
         dfdx.Assemble();

         // initialize the vector that we use to perturb the mesh nodes
         GridFunction v(mesh_fes);
         VectorFunctionCoefficient v_rand(3, randVectorState);
         v.ProjectCoefficient(v_rand);

         // contract dfdx with v
         double dfdx_v = dfdx * v;

         // now compute the finite-difference approximation...
         GridFunction x_pert(*x_nodes);
         GridFunction r(h1_fes.get());
         x_pert.Add(delta, v);
         mesh->SetNodes(x_pert);
         nd_fes->Update();
         h1_fes->Update();
         c.Update();
         c.ProjectCoefficient(current);
         res.Update();
         res.Assemble();
         res.Finalize();
         res.Mult(c, r);
         double dfdx_v_fd = adjoint * r;
         x_pert.Add(-2 * delta, v);
         mesh->SetNodes(x_pert);
         nd_fes->Update();
         h1_fes->Update();
         c.Update();
         c.ProjectCoefficient(current);
         res.Update();
         res.Assemble();
         res.Finalize();
         res.Mult(c, r);
         dfdx_v_fd -= adjoint * r;
         dfdx_v_fd /= (2 * delta);
         mesh->SetNodes(*x_nodes); // remember to reset the mesh nodes

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-8));
      }
   }
}

TEST_CASE("VectorFEDomainLFMeshSensInteg::AssembleRHSElementVect"
          "[VectorFEDomainLFMeshSensInteg]")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3;  // templating is hard here because mesh constructors
   // static adept::Stack diff_stack;
   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 2;
   auto mesh = getMesh(num_edge, num_edge);
   mesh->ReorientTetMesh();
   mesh->EnsureNodes();
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         // get the finite-element space for the adjoint
         std::unique_ptr<FiniteElementCollection> fec(
            new ND_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
            mesh.get(), fec.get()));

         // initialize adjoint; here we randomly perturb a constant state
         GridFunction adjoint(fes.get());
         VectorFunctionCoefficient pert(3, randVectorState);

         /// Costruct coefficient
         mach::VectorMeshDependentCoefficient current(dim);
         std::unique_ptr<mfem::VectorCoefficient> coeff1(
            new VectorFunctionCoefficient(dim, vectorFunc, vectorFuncRevDiff));
         std::unique_ptr<mfem::VectorCoefficient> coeff2(
            new VectorFunctionCoefficient(dim, vectorFunc2, vectorFunc2RevDiff));
         current.addCoefficient(1, move(coeff1));
         current.addCoefficient(2, move(coeff2));

         adjoint.ProjectCoefficient(pert);

         // we use res for finite-difference approximation
         LinearForm res(fes.get());
         res.AddDomainIntegrator(new VectorFEDomainLFIntegrator(current));

         // extract mesh nodes and get their finite-element space
         GridFunction *x_nodes = mesh->GetNodes();
         FiniteElementSpace *mesh_fes = x_nodes->FESpace();

         // build the nonlinear form for d(psi^T R)/dx 
         LinearForm dfdx(mesh_fes);
         dfdx.AddDomainIntegrator(
            new mach::VectorFEDomainLFMeshSensInteg(&adjoint, current));
         dfdx.Assemble();

         // initialize the vector that we use to perturb the mesh nodes
         GridFunction v(mesh_fes);
         VectorFunctionCoefficient v_rand(3, randVectorState);
         v.ProjectCoefficient(v_rand);

         // contract dfdx with v
         double dfdx_v = dfdx * v;

         // now compute the finite-difference approximation...
         GridFunction x_pert(*x_nodes);
         x_pert.Add(delta, v);
         mesh->SetNodes(x_pert);
         fes->Update();
         res.Update();
         res.Assemble();
         double dfdx_v_fd = adjoint * res;
         x_pert.Add(-2 * delta, v);
         mesh->SetNodes(x_pert);
         fes->Update();
         res.Update();
         res.Assemble();
         dfdx_v_fd -= adjoint * res;
         dfdx_v_fd /= (2 * delta);
         mesh->SetNodes(*x_nodes); // remember to reset the mesh nodes

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-8));
      }
   }
}

// TEST_CASE("GridFuncMeshSensIntegrator::AssembleRHSElementVect",
//           "[GridFuncMeshSensIntegrator]")
// {
//    using namespace mfem;
//    using namespace electromag_data;

//    const int dim = 3;  // templating is hard here because mesh constructors
//    // static adept::Stack diff_stack;
//    double delta = 1e-5;

//    // generate a 6 element mesh
//    int num_edge = 2;
//    std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, num_edge,
//                               Element::TETRAHEDRON, true /* gen. edges */, 1.0,
//                               1.0, 1.0, true));
//    mesh->ReorientTetMesh();
//    mesh->EnsureNodes();
//    for (int p = 1; p <= 4; ++p)
//    {
//       DYNAMIC_SECTION("...for degree p = " << p)
//       {
//          // get the finite-element space for the current grid function
//          std::unique_ptr<FiniteElementCollection> nd_fec(
//             new ND_FECollection(p, dim));
//          std::unique_ptr<FiniteElementSpace> nd_fes(new FiniteElementSpace(
//             mesh.get(), nd_fec.get()));


//          // initialize current source and adjoint
//          // here we randomly perturb a constant state
//          GridFunction c(nd_fes.get()), adjoint(nd_fes.get());
//          VectorFunctionCoefficient pert(3, randVectorState);
//          VectorFunctionCoefficient current(3, vectorFunc, vectorFuncRevDiff);
//          adjoint.ProjectCoefficient(pert);

//          // extract mesh nodes and get their finite-element space
//          GridFunction *x_nodes = mesh->GetNodes();
//          FiniteElementSpace *mesh_fes = x_nodes->FESpace();

//          // build the linear form for d(psi^T R)/dx 
//          LinearForm dfdx(mesh_fes);
//          dfdx.AddDomainIntegrator(
//             new mach::GridFuncMeshSensIntegrator(&adjoint, &current));
//          dfdx.Assemble();

//          GridFunction dFdX_gf(mesh_fes);
//          dFdX_gf.ProjectCoefficientRevDiff(adjoint, current);

//          // initialize the vector that we use to perturb the mesh nodes
//          GridFunction v(mesh_fes);
//          VectorFunctionCoefficient v_rand(3, randVectorState);
//          v.ProjectCoefficient(v_rand);

//          // contract dfdx with v
//          double dfdx_v = dfdx * v;

//          double dFdX_gf_v = dFdX_gf * v;
//          std::cout << "dFdX_gf_v " << dFdX_gf_v << "\n";


//          // now compute the finite-difference approximation...
//          GridFunction x_pert(*x_nodes);
//          x_pert.Add(delta, v);
//          mesh->SetNodes(x_pert);
//          nd_fes->Update();
//          c.Update();
//          c.ProjectCoefficient(current);
//          double dfdx_v_fd = adjoint * c;
//          x_pert.Add(-2 * delta, v);
//          mesh->SetNodes(x_pert);
//          nd_fes->Update();
//          c.Update();
//          c.ProjectCoefficient(current);
//          dfdx_v_fd -= adjoint * c;
//          dfdx_v_fd /= (2 * delta);
//          mesh->SetNodes(*x_nodes); // remember to reset the mesh nodes

//          REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-8));
//       }
//    }
// }

TEST_CASE("MagneticEnergyIntegrator::GetEnergy",
          "[MagneticEnergyIntegrator]")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3; // templating is hard here because mesh constructors
   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 2;
   std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, num_edge,
                                       Element::TETRAHEDRON,
                                       true /* gen. edges */, 2.0, 3.0, 1.0,
                                       true));
   mesh->ReorientTetMesh();
   mesh->EnsureNodes();

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {

         auto nonlin_energy = [](double B) { return 1.0/3.0 * (sqrt(B+1) * (B-2) + 2); };

         std::unique_ptr<FiniteElementCollection> fec(
            new ND_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
            mesh.get(), fec.get()));


         // initialize state; here we randomly perturb a constant state
         GridFunction A(fes.get());
         VectorFunctionCoefficient pert(3, [](const Vector &x, Vector &a)
         {
            a(0) = 1.5*x(1);
            a(1) = -0.5*x(0);
            a(2) = 0.0;
         });
         A.ProjectCoefficient(pert);

         std::unique_ptr<mach::StateCoefficient> nu(
            new NonLinearCoefficient());

         NonlinearForm functional(fes.get());
         functional.AddDomainIntegrator(
            new mach::MagneticEnergyIntegrator(*nu));

         const double fun = functional.GetEnergy(A);
         const double b_mag = 2.0;
         const double energy = nonlin_energy(b_mag);
         const double vol = 6.0;
         // std::cout << "fun: " << fun << " energy * vol: " << energy*vol << "\n";
         REQUIRE(fun == Approx(energy * vol));
      }
   }
}

TEST_CASE("MagneticEnergyIntegrator::AssembleElementVector")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3;
   double delta = 1e-5;

   int num_edge = 2;
   Mesh mesh(num_edge, num_edge, num_edge, Element::TETRAHEDRON,
             true /* gen. edges */, 2.0, 3.0, 1.0, true);
   mesh.ReorientTetMesh();
   mesh.EnsureNodes();

   NonLinearCoefficient nu;

   /// construct elements
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         ND_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize state; here we randomly perturb a constant state
         GridFunction A(&fes);
         VectorFunctionCoefficient pert(3, randVectorState);
         A.ProjectCoefficient(pert);

         NonlinearForm functional(&fes);
         functional.AddDomainIntegrator(
            new mach::MagneticEnergyIntegrator(nu));

         // initialize the vector that dJdu multiplies
         GridFunction p(&fes);
         p.ProjectCoefficient(pert);

         // evaluate dJdu and compute its product with v
         GridFunction dJdu(&fes);
         functional.Mult(A, dJdu);
         double dJdu_dot_p = InnerProduct(dJdu, p);

         // now compute the finite-difference approximation...
         GridFunction q_pert(A);
         q_pert.Add(-delta, p);
         double dJdu_dot_p_fd = -functional.GetEnergy(q_pert);
         q_pert.Add(2 * delta, p);
         dJdu_dot_p_fd += functional.GetEnergy(q_pert);
         dJdu_dot_p_fd /= (2 * delta);

         REQUIRE(dJdu_dot_p == Approx(dJdu_dot_p_fd));
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
   int num_edge = 2;
   std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, num_edge, Element::TETRAHEDRON,
                                       true /* gen. edges */, 1.0, 1.0, 1.0, true));
   mesh->ReorientTetMesh();
   mesh->EnsureNodes();

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
            new ND_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
            mesh.get(), fec.get()));

         NonlinearForm functional(fes.get());

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(3, randVectorState);
         q.ProjectCoefficient(pert);

         std::unique_ptr<mach::StateCoefficient> nu(
            // new LinearCoefficient());
            new NonLinearCoefficient());

         functional.AddDomainIntegrator(
            new mach::MagneticCoenergyIntegrator(q, nu.get()));

         // initialize the vector that dJdu multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(3, randVectorState);
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
   int num_edge = 2;
   std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, num_edge,
                                       Element::TETRAHEDRON,
                                       true /* gen. edges */, 1.0, 1.0, 1.0, true));
   mesh->ReorientTetMesh();
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
         VectorFunctionCoefficient pert(3, randVectorState);
         q.ProjectCoefficient(pert);

         // initialize the vector that dJdx multiplies
         GridFunction v(mesh_fes);
         VectorFunctionCoefficient v_rand(dim, randVectorState);
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

TEST_CASE("BNormIntegrator::GetElementEnergy",
          "[BNormIntegrator]")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3; // templating is hard here because mesh constructors
   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 2;
   std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, num_edge, Element::TETRAHEDRON,
                                       true /* gen. edges */, 2.0, 2.0, 1.0, true));
   mesh->ReorientTetMesh();
   mesh->EnsureNodes();

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
            new ND_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
            mesh.get(), fec.get()));

         // initialize state; here we randomly perturb a constant state
         GridFunction A(fes.get());
         VectorFunctionCoefficient A_fun(3, [](const Vector &x, Vector &a)
         {
            a(0) = 1.5*x(1);
            a(1) = -0.5*x(0);
            a(2) = 0.0;
         });
         A.ProjectCoefficient(A_fun);

         NonlinearForm functional(fes.get());
         functional.AddDomainIntegrator(
            new mach::BNormIntegrator());


         const double fun = functional.GetEnergy(A);
         const double b_mag = 2.0;
         const double vol = 4.0;
         REQUIRE(fun == Approx(b_mag * vol));
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
   int num_edge = 2;
   std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, num_edge, Element::TETRAHEDRON,
                                       true /* gen. edges */, 1.0, 1.0, 1.0, true));
   mesh->ReorientTetMesh();
   mesh->EnsureNodes();

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
         VectorFunctionCoefficient pert(3, randVectorState);
         q.ProjectCoefficient(pert);

         functional.AddDomainIntegrator(
            new mach::BNormIntegrator());

         // initialize the vector that dJdu multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(3, randVectorState);
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
   int num_edge = 3;
   std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, num_edge, Element::TETRAHEDRON,
                                       true /* gen. edges */, 1.0, 1.0, 1.0, true));
   mesh->ReorientTetMesh();
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
         GridFunction A(fes.get());
         VectorFunctionCoefficient pert(3, randVectorState);
         A.ProjectCoefficient(pert);

         // initialize the vector that dJdx multiplies
         GridFunction v(mesh_fes);
         VectorFunctionCoefficient v_rand(dim, randVectorState);
         v.ProjectCoefficient(v_rand);

         // evaluate dJdx and compute its product with v
         // GridFunction dJdx(*x_nodes);
         LinearForm dJdx(mesh_fes);
         dJdx.AddDomainIntegrator(
            new mach::BNormdJdx(A));
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
         double dJdx_dot_v_fd = -functional.GetEnergy(A);
         x_pert.Add(2 * delta, v);
         mesh->SetNodes(x_pert);
         fes->Update();
         dJdx_dot_v_fd += functional.GetEnergy(A);
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
   int num_edge = 2;
   std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, num_edge, Element::TETRAHEDRON,
                                       true /* gen. edges */, 1.0, 1.0, 1.0, true));
   mesh->ReorientTetMesh();
   mesh->EnsureNodes();

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
         VectorFunctionCoefficient pert(3, randVectorState);
         q.ProjectCoefficient(pert);

         std::unique_ptr<mach::StateCoefficient> nu(
            new NonLinearCoefficient());

         functional.AddDomainIntegrator(
            new mach::nuBNormIntegrator(nu.get()));

         // initialize the vector that dJdu multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(3, randVectorState);
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
   int num_edge = 2;
   std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, num_edge, Element::TETRAHEDRON,
                                       true /* gen. edges */, 1.0, 1.0, 1.0, true));
   mesh->ReorientTetMesh();
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
         VectorFunctionCoefficient pert(3, randVectorState);
         q.ProjectCoefficient(pert);

         // initialize the vector that dJdx multiplies
         GridFunction v(mesh_fes);
         VectorFunctionCoefficient v_rand(dim, randVectorState);
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
   int num_edge = 2;
   std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, num_edge, Element::TETRAHEDRON,
                                       true /* gen. edges */, 1.0, 1.0, 1.0, true));
   mesh->ReorientTetMesh();
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
         VectorFunctionCoefficient pert(3, randVectorState);
         q.ProjectCoefficient(pert);

         // initialize the vector that dJdx multiplies
         GridFunction v(mesh_fes);
         VectorFunctionCoefficient v_rand(dim, randVectorState);
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

TEST_CASE("DCLossFunctionalIntegrator::GetEnergy",
          "[DCLossFunctionalIntegrator]")
{
   using namespace mfem;

   const int dim = 3; // templating is hard here because mesh constructors


   int num_edge = 1;
   for (int i = 0; i < 2; ++i)
   {
      num_edge *= 2;
      DYNAMIC_SECTION("...for num edges = " << num_edge)
      {
         Mesh mesh(num_edge, num_edge, num_edge, Element::TETRAHEDRON,
                  true /* gen. edges */, 1.0, 1.0, 1.0, true);
         mesh.ReorientTetMesh();
         mesh.EnsureNodes();

         const auto p = 2;

         std::unique_ptr<FiniteElementCollection> fec(
            new ND_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
            &mesh, fec.get()));

         NonlinearForm functional(fes.get());

         // initialize state
         GridFunction A(fes.get());

         // auto sigma_val = 58.14e6;
         auto sigma_val = 1.0;
         ConstantCoefficient sigma(sigma_val); // conductivity

         double current_vec_data[] = {0.0, 0.0, 1.0};
         Vector current_vec(current_vec_data, 3);
         VectorConstantCoefficient current(current_vec); // current density coefficient
         auto current_density = 1.0; // current density magnitude

         auto fill_factor = 0.7853981633974483; // the density of strands in the bundle

         functional.AddDomainIntegrator(
            new mach::DCLossFunctionalIntegrator(sigma, current, current_density, fill_factor));

         const auto R_dc = 1.0 / (sigma_val * fill_factor); // length / (sigma * area)
         const auto loss = std::pow(current_density * fill_factor, 2) * R_dc;
         
         const double loss_fe = functional.GetEnergy(A);
         // std::cout << "functional loss: " << loss_fe << "\n";
         // std::cout << "analytical loss: " << loss << "\n";
         const double loss_ratio = loss_fe / loss;
         REQUIRE(loss_ratio == Approx(1.0).epsilon(1e-1));
      }
   }
}

namespace mach
{
/// Compute the finite-difference approximation of the derivative of the
/// magnetic energy with respect to B
/// \param[in] trans - element transformation for where to evaluate `nu`
/// \param[in] ip - integration point for where to evaluate `nu`
/// \param[in] nu - material dependent model describing reluctivity
/// \param[in] B - upper bound for integration
/// \return the derivative of the magnetic energy with respect to B
double calcMagneticEnergyFD(
   mfem::ElementTransformation &trans,
   const mfem::IntegrationPoint &ip,
   StateCoefficient &nu,
   double B)
{
   double delta = 1e-5;

   double fd_val;
   fd_val = calcMagneticEnergy(trans, ip, nu, B + delta);
   fd_val -= calcMagneticEnergy(trans, ip, nu, B - delta);
   return fd_val / (2*delta);
}

/// Compute the finite-difference approximation of the second derivative of the
/// magnetic energy with respect to B
/// \param[in] trans - element transformation for where to evaluate `nu`
/// \param[in] ip - integration point for where to evaluate `nu`
/// \param[in] nu - material dependent model describing reluctivity
/// \param[in] B - upper bound for integration
/// \return the second derivative of the magnetic energy with respect to B
double calcMagneticEnergyDoubleFD(
   mfem::ElementTransformation &trans,
   const mfem::IntegrationPoint &ip,
   StateCoefficient &nu,
   double B)
{
   double delta = 1e-5;

   double fd_val;
   fd_val = calcMagneticEnergyDot(trans, ip, nu, B + delta);
   fd_val -= calcMagneticEnergyDot(trans, ip, nu, B - delta);
   return fd_val / (2*delta);
}

}

TEST_CASE("calcMagneticEnergy")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3; // templating is hard here because mesh constructors
   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 1;
   std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, num_edge, Element::TETRAHEDRON,
                                       true /* gen. edges */, 1.0, 1.0, 1.0, true));
   mesh->ReorientTetMesh();
   mesh->EnsureNodes();

   const double lin_nu_val = 0.42;
   std::unique_ptr<mach::StateCoefficient> lin_nu(new LinearCoefficient(lin_nu_val));
   std::unique_ptr<mach::StateCoefficient> nu(new NonLinearCoefficient());

   /// construct elements
   int p = 1;
   ND_FECollection fec(p, dim);
   FiniteElementSpace fes(mesh.get(), &fec);

   const double B_mag = 1.3;

   auto nonlin_energy = [](double B) { return 1.0/3.0 * (sqrt(B+1) * (B-2) + 2); };

   for (int j = 0; j < fes.GetNE(); j++)
   {
      const FiniteElement &el = *fes.GetFE(j);

      IsoparametricTransformation trans;
      mesh->GetElementTransformation(j, &trans);

      int order = trans.OrderW() + 2 * el.GetOrder();
      const IntegrationRule *ir = &IntRules.Get(el.GetGeomType(), order);

      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         trans.SetIntPoint(&ip);

         double lin_en = mach::calcMagneticEnergy(trans, ip, *lin_nu, B_mag);
         REQUIRE(lin_en == Approx(0.5*pow(B_mag, 2)*lin_nu_val).epsilon(1e-8));

         double en = mach::calcMagneticEnergy(trans, ip, *nu, B_mag);
         REQUIRE(en == Approx(nonlin_energy(B_mag)).epsilon(1e-8));
         // std::cout << "lin_en: " << lin_en << " en: " << en << "\n";
      }
   }
}

TEST_CASE("calcMagneticEnergyDot")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3; // templating is hard here because mesh constructors
   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 1;
   std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, num_edge, Element::TETRAHEDRON,
                                       true /* gen. edges */, 1.0, 1.0, 1.0, true));
   mesh->ReorientTetMesh();
   mesh->EnsureNodes();

   // std::unique_ptr<mach::StateCoefficient> nu(new LinearCoefficient(0.75));
   std::unique_ptr<mach::StateCoefficient> nu(new NonLinearCoefficient());

   /// construct elements
   int p = 1;
   ND_FECollection fec(p, dim);
   FiniteElementSpace fes(mesh.get(), &fec);

   const double B_mag = 0.5;

   for (int j = 0; j < fes.GetNE(); j++)
   {
      const FiniteElement &el = *fes.GetFE(j);

      IsoparametricTransformation trans;
      mesh->GetElementTransformation(j, &trans);

      int order = trans.OrderW() + 2 * el.GetOrder();
      const IntegrationRule *ir = &IntRules.Get(el.GetGeomType(), order);

      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         trans.SetIntPoint(&ip);

         double dWdB = mach::calcMagneticEnergyDot(trans, ip, *nu, B_mag);
         double dWdB_fd = mach::calcMagneticEnergyFD(trans, ip, *nu, B_mag);
         REQUIRE(dWdB == Approx(dWdB_fd).epsilon(1e-10));
      }
   }
}

TEST_CASE("calcMagneticEnergyDoubleDot")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3; // templating is hard here because mesh constructors
   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 1;
   std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, num_edge, Element::TETRAHEDRON,
                                       true /* gen. edges */, 1.0, 1.0, 1.0, true));
   mesh->ReorientTetMesh();
   mesh->EnsureNodes();

   // std::unique_ptr<mach::StateCoefficient> nu(new LinearCoefficient(0.75));
   std::unique_ptr<mach::StateCoefficient> nu(new NonLinearCoefficient());

   /// construct elements
   int p = 1;
   ND_FECollection fec(p, dim);
   FiniteElementSpace fes(mesh.get(), &fec);

   const double B_mag = 0.25;

   for (int j = 0; j < fes.GetNE(); j++)
   {
      const FiniteElement &el = *fes.GetFE(j);

      IsoparametricTransformation trans;
      mesh->GetElementTransformation(j, &trans);

      int order = trans.OrderW() + 2 * el.GetOrder();
      const IntegrationRule *ir = &IntRules.Get(el.GetGeomType(), order);

      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         trans.SetIntPoint(&ip);

         double d2WdB2 = mach::calcMagneticEnergyDoubleDot(trans, ip, *nu, B_mag);
         double d2WdB2_fd = mach::calcMagneticEnergyDoubleFD(trans, ip, *nu, B_mag);
         REQUIRE(d2WdB2 == Approx(d2WdB2_fd).epsilon(1e-10));
      }
   }
}

TEST_CASE("ForceIntegrator::GetElementEnergy")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3; // templating is hard here because mesh constructors
   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 2;
   Mesh mesh(num_edge, num_edge, num_edge, Element::TETRAHEDRON,
             true /* gen. edges */, 2.0, 3.0, 1.0, true);
   mesh.ReorientTetMesh();
   mesh.EnsureNodes();

   NonLinearCoefficient nu;

   /// construct elements
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         ND_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // extract mesh nodes and get their finite-element space
         auto &x_nodes = *mesh.GetNodes();
         auto &mesh_fes = *x_nodes.FESpace();

         // create v displacement field
         GridFunction v(&mesh_fes);
         VectorFunctionCoefficient pert(3, randVectorState);
         v.ProjectCoefficient(pert);

         // initialize state; here we randomly perturb a constant state
         GridFunction A(&fes);
         A.ProjectCoefficient(pert);

         NonlinearForm functional(&fes);
         functional.AddDomainIntegrator(
            new mach::ForceIntegrator(nu, v));

         auto force = functional.GetEnergy(A);

         NonlinearForm energy(&fes);
         energy.AddDomainIntegrator(
            new mach::MagneticEnergyIntegrator(nu));
         
         add(x_nodes, -delta, v, x_nodes);
         auto dWds = -energy.GetEnergy(A);
         add(x_nodes, 2*delta, v, x_nodes);
         dWds += energy.GetEnergy(A);
         dWds /= 2*delta;

         std::cout << "-dWds: " << -dWds << " Force: " << force << "\n";
         REQUIRE(force == Approx(-dWds));
      }
   }
}

TEST_CASE("ForceIntegrator::AssembleElementVector")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3;
   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 2;
   Mesh mesh(num_edge, num_edge, num_edge, Element::TETRAHEDRON,
             true /* gen. edges */, 2.0, 3.0, 1.0, true);
   mesh.ReorientTetMesh();
   mesh.EnsureNodes();

   NonLinearCoefficient nu;

   /// construct elements
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         ND_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // extract mesh nodes and get their finite-element space
         auto &x_nodes = *mesh.GetNodes();
         auto &mesh_fes = *x_nodes.FESpace();

         // create v displacement field
         GridFunction v(&mesh_fes);
         VectorFunctionCoefficient pert(3, randVectorState);
         v.ProjectCoefficient(pert);

         // initialize state; here we randomly perturb a constant state
         GridFunction A(&fes);
         A.ProjectCoefficient(pert);

         NonlinearForm functional(&fes);
         functional.AddDomainIntegrator(
            new mach::ForceIntegrator(nu, v));

         // initialize the vector that dJdu multiplies
         GridFunction p(&fes);
         p.ProjectCoefficient(pert);

         // evaluate dJdu and compute its product with v
         GridFunction dJdu(&fes);
         functional.Mult(A, dJdu);
         double dJdu_dot_p = InnerProduct(dJdu, p);

         // now compute the finite-difference approximation...
         GridFunction q_pert(A);
         q_pert.Add(-delta, p);
         double dJdu_dot_p_fd = -functional.GetEnergy(q_pert);
         q_pert.Add(2 * delta, p);
         dJdu_dot_p_fd += functional.GetEnergy(q_pert);
         dJdu_dot_p_fd /= (2 * delta);

         REQUIRE(dJdu_dot_p == Approx(dJdu_dot_p_fd));
      }
   }
}

TEST_CASE("dBdsIntegrator::AssembleElementVector")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3;
   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 2;
   Mesh mesh(num_edge, num_edge, num_edge, Element::TETRAHEDRON,
             true /* gen. edges */, 2.0, 3.0, 1.0, true);
   mesh.ReorientTetMesh();
   mesh.EnsureNodes();

   NonLinearCoefficient nu;

   /// construct elements
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         ND_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // extract mesh nodes and get their finite-element space
         auto &x_nodes = *mesh.GetNodes();
         auto &mesh_fes = *x_nodes.FESpace();

         // create v displacement field
         GridFunction v(&mesh_fes);
         VectorFunctionCoefficient pert(3, randVectorState);
         v.ProjectCoefficient(pert);

         // initialize state; here we randomly perturb a constant state
         GridFunction A(&fes);
         A.ProjectCoefficient(pert);

         NonlinearForm functional(&fes);
         functional.AddDomainIntegrator(
            new mach::dBdsIntegrator(nu, v));

         // initialize the vector that dJdu multiplies
         GridFunction p(&fes);
         p.ProjectCoefficient(pert);

         // evaluate dJdu and compute its product with v
         GridFunction dJdu(&fes);
         functional.Mult(A, dJdu);
         double dJdu_dot_p = InnerProduct(dJdu, p);

         // now compute the finite-difference approximation...
         GridFunction q_pert(A);
         q_pert.Add(-delta, p);
         double dJdu_dot_p_fd = -functional.GetEnergy(q_pert);
         q_pert.Add(2 * delta, p);
         dJdu_dot_p_fd += functional.GetEnergy(q_pert);
         dJdu_dot_p_fd /= (2 * delta);

         REQUIRE(dJdu_dot_p == Approx(dJdu_dot_p_fd));
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
//          VectorFunctionCoefficient pert(3, randBaselineVectorPert);
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
//          VectorFunctionCoefficient v_rand(3, randVectorState);
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