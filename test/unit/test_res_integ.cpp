#include <random>

#include "catch.hpp"
#include "mfem.hpp"
#include "res_integ.hpp"
#include "euler_test_data.hpp"

TEMPLATE_TEST_CASE_SIG("DomainResIntegrator::AssembleElementVector",
                       "[DomainResIntegrator]",
                       ((bool entvar), entvar), false, true)
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 3; // templating is hard here because mesh constructors
   double delta = 1e-5;

   // generate a 8 element mesh
   int num_edge = 2;
   std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, num_edge, Element::TETRAHEDRON,
                                       true /* gen. edges */, 1.0, 1.0, 1.0, true));
   mesh->EnsureNodes();
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         // get the finite-element space for the state and adjoint
         std::unique_ptr<FiniteElementCollection> fec(
             new H1_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             mesh.get(), fec.get()));

         // we use res for finite-difference approximation
         std::unique_ptr<Coefficient> Q(new ConstantCoefficient(1));
         LinearForm res(fes.get());
         res.AddDomainIntegrator(
            new DomainLFIntegrator(*Q));

         // initialize state and adjoint; here we randomly perturb a constant state
         GridFunction state(fes.get()), adjoint(fes.get());
         FunctionCoefficient pert(randState);
         state.ProjectCoefficient(pert);
         adjoint.ProjectCoefficient(pert);

         // extract mesh nodes and get their finite-element space
         GridFunction *x_nodes = mesh->GetNodes();
         FiniteElementSpace *mesh_fes = x_nodes->FESpace();

         // build the nonlinear form for d(psi^T R)/dx 
         NonlinearForm dfdx_form(mesh_fes);
         dfdx_form.AddDomainIntegrator(
            new mach::DomainResIntegrator(*Q,
               &state, &adjoint));

         // initialize the vector that we use to perturb the mesh nodes
         GridFunction v(mesh_fes);
         VectorFunctionCoefficient v_rand(dim, randState);
         v.ProjectCoefficient(v_rand);

         // evaluate df/dx and contract with v
         GridFunction dfdx(*x_nodes);
         dfdx_form.Mult(*x_nodes, dfdx);
         double dfdx_v = dfdx * v;

         // now compute the finite-difference approximation...
         GridFunction x_pert(*x_nodes);
         GridFunction r(fes.get());
         x_pert.Add(delta, v);
         mesh->SetNodes(x_pert);
         res.Assemble();
         double dfdx_v_fd = adjoint * res;
         x_pert.Add(-2 * delta, v);
         mesh->SetNodes(x_pert);
         res.Assemble();
         dfdx_v_fd -= adjoint * res;
         dfdx_v_fd /= (2 * delta);
         mesh->SetNodes(*x_nodes); // remember to reset the mesh nodes

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-10));
      }
   }
}

TEMPLATE_TEST_CASE_SIG("MassResIntegrator::AssembleElementVector",
                       "[MassResIntegrator]",
                       ((bool entvar), entvar), false, true)
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 3; // templating is hard here because mesh constructors
   double delta = 1e-5;

   // generate a 8 element mesh
   int num_edge = 2;
   std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, num_edge, Element::TETRAHEDRON,
                                       true /* gen. edges */, 1.0, 1.0, 1.0, true));
   mesh->EnsureNodes();
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         // get the finite-element space for the state and adjoint
         std::unique_ptr<FiniteElementCollection> fec(
             new H1_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             mesh.get(), fec.get()));

         // we use res for finite-difference approximation
         std::unique_ptr<Coefficient> Q(new ConstantCoefficient(1));
         BilinearForm res(fes.get());
         res.AddDomainIntegrator(
            new MassIntegrator(*Q));

         // initialize state and adjoint; here we randomly perturb a constant state
         GridFunction state(fes.get()), adjoint(fes.get());
         FunctionCoefficient pert(randState);
         state.ProjectCoefficient(pert);
         adjoint.ProjectCoefficient(pert);

         // extract mesh nodes and get their finite-element space
         GridFunction *x_nodes = mesh->GetNodes();
         FiniteElementSpace *mesh_fes = x_nodes->FESpace();

         // build the nonlinear form for d(psi^T R)/dx 
         NonlinearForm dfdx_form(mesh_fes);
         dfdx_form.AddDomainIntegrator(
            new mach::MassResIntegrator(*Q,
               &state, &adjoint));

         // initialize the vector that we use to perturb the mesh nodes
         GridFunction v(mesh_fes);
         VectorFunctionCoefficient v_rand(dim, randState);
         v.ProjectCoefficient(v_rand);

         // evaluate df/dx and contract with v
         GridFunction dfdx(*x_nodes);
         dfdx_form.Mult(*x_nodes, dfdx);
         double dfdx_v = dfdx * v;

         // now compute the finite-difference approximation...
         Vector resid(res.Size());
         GridFunction x_pert(*x_nodes);
         GridFunction r(fes.get());
         x_pert.Add(delta, v);
         mesh->SetNodes(x_pert);
         res.Assemble();
         res.Mult(state, resid);
         double dfdx_v_fd = adjoint * resid;
         x_pert.Add(-2*delta, v);
         mesh->SetNodes(x_pert);
         res.Update();
         res.Assemble();
         res.Mult(state, resid);
         dfdx_v_fd -= adjoint * resid;
         dfdx_v_fd /= (2*delta);
         mesh->SetNodes(*x_nodes); // remember to reset the mesh nodes

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-10));
      }
   }
}

TEMPLATE_TEST_CASE_SIG("DiffusionResIntegrator::AssembleElementVector",
                       "[DiffusionResIntegrator]",
                       ((bool entvar), entvar), false, true)
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 3; // templating is hard here because mesh constructors
   double delta = 1e-5;

   // generate a 8 element mesh
   int num_edge = 2;
   std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, num_edge, Element::TETRAHEDRON,
                                       true /* gen. edges */, 1.0, 1.0, 1.0, true));
   mesh->EnsureNodes();
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         // get the finite-element space for the state and adjoint
         std::unique_ptr<FiniteElementCollection> fec(
             new H1_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             mesh.get(), fec.get()));

         // we use res for finite-difference approximation
         std::unique_ptr<Coefficient> Q(new ConstantCoefficient(1));
         BilinearForm res(fes.get());
         res.AddDomainIntegrator(
            new DiffusionIntegrator(*Q));

         // initialize state and adjoint; here we randomly perturb a constant state
         GridFunction state(fes.get()), adjoint(fes.get());
         FunctionCoefficient pert(randState);
         state.ProjectCoefficient(pert);
         adjoint.ProjectCoefficient(pert);

         // extract mesh nodes and get their finite-element space
         GridFunction *x_nodes = mesh->GetNodes();
         FiniteElementSpace *mesh_fes = x_nodes->FESpace();

         // build the nonlinear form for d(psi^T R)/dx 
         NonlinearForm dfdx_form(mesh_fes);
         dfdx_form.AddDomainIntegrator(
            new mach::DiffusionResIntegrator(*Q,
               &state, &adjoint));

         // initialize the vector that we use to perturb the mesh nodes
         GridFunction v(mesh_fes);
         VectorFunctionCoefficient v_rand(dim, randState);
         v.ProjectCoefficient(v_rand);

         // evaluate df/dx and contract with v
         GridFunction dfdx(*x_nodes);
         dfdx_form.Mult(*x_nodes, dfdx);
         double dfdx_v = dfdx * v;

         // now compute the finite-difference approximation...
         Vector resid(res.Size());
         GridFunction x_pert(*x_nodes);
         GridFunction r(fes.get());
         x_pert.Add(delta, v);
         mesh->SetNodes(x_pert);
         res.Assemble();
         res.Mult(state, resid);
         double dfdx_v_fd = adjoint * resid;
         x_pert.Add(-2*delta, v);
         mesh->SetNodes(x_pert);
         res.Update();
         res.Assemble();
         res.Mult(state, resid);
         dfdx_v_fd -= adjoint * resid;
         dfdx_v_fd /= (2*delta);
         mesh->SetNodes(*x_nodes); // remember to reset the mesh nodes

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-10));
      }
   }
}

TEMPLATE_TEST_CASE_SIG("BoundaryNormalResIntegrator::AssembleFaceVector",
                       "[BoundaryNormalResIntegrator]",
                       ((bool entvar), entvar), false, true)
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 3; // templating is hard here because mesh constructors
   double delta = 1e-5;

   // generate a 8 element mesh
   int num_edge = 2;
   std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, num_edge, Element::TETRAHEDRON,
                                       true /* gen. edges */, 1.0, 1.0, 1.0, true));
   mesh->EnsureNodes();
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         // get the finite-element space for the state and adjoint
         std::unique_ptr<FiniteElementCollection> fec(
             new H1_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             mesh.get(), fec.get()));

         // we use res for finite-difference approximation
         Vector V(dim); Array<int> attr;
         attr.SetSize(mesh->bdr_attributes.Size(), 0);
         attr[0] = 1;
         V = 1.0;
         std::unique_ptr<VectorCoefficient> Q(new VectorConstantCoefficient(V));
         LinearForm res(fes.get());
         res.AddBoundaryIntegrator(
            new BoundaryNormalLFIntegrator(*Q), attr);

         // initialize state and adjoint; here we randomly perturb a constant state
         GridFunction state(fes.get()), adjoint(fes.get());
         FunctionCoefficient pert(randState);
         state.ProjectCoefficient(pert);
         adjoint.ProjectCoefficient(pert);

         // extract mesh nodes and get their finite-element space
         GridFunction *x_nodes = mesh->GetNodes();
         FiniteElementSpace *mesh_fes = x_nodes->FESpace();

         // build the nonlinear form for d(psi^T R)/dx 
         LinearForm dfdx_form(mesh_fes);
         dfdx_form.AddBdrFaceIntegrator(
            new mach::BoundaryNormalResIntegrator(*Q,
               &state, &adjoint), attr);

         // initialize the vector that we use to perturb the mesh nodes
         GridFunction v(mesh_fes);
         VectorFunctionCoefficient v_rand(dim, randState);
         v.ProjectCoefficient(v_rand);

         // evaluate df/dx and contract with v
         //GridFunction dfdx(*x_nodes);
         //dfdx_form.Mult(*x_nodes, dfdx);
         dfdx_form.Assemble();
         double dfdx_v = dfdx_form * v;

         // now compute the finite-difference approximation...
         GridFunction x_pert(*x_nodes);
         GridFunction r(fes.get());
         x_pert.Add(delta, v);
         mesh->SetNodes(x_pert);
         res.Assemble();
         double dfdx_v_fd = adjoint * res;
         x_pert.Add(-2 * delta, v);
         mesh->SetNodes(x_pert);
         res.Assemble();
         dfdx_v_fd -= adjoint * res;
         dfdx_v_fd /= (2 * delta);
         mesh->SetNodes(*x_nodes); // remember to reset the mesh nodes

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-10));
      }
   }
}