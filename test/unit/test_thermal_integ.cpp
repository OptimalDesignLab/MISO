#include <random>

#include "catch.hpp"
#include "mfem.hpp"

#include "electromag_test_data.hpp"

#include "thermal_integ.hpp"

TEST_CASE("TestBCIntegratorMeshRevSens::AssembleRHSElementVect")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge,
                                     num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         H1_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize state; here we randomly perturb a constant state
         GridFunction state(&fes);
         GridFunction adjoint(&fes);
         FunctionCoefficient pert(randState);
         state.ProjectCoefficient(pert);
         adjoint.ProjectCoefficient(pert);

         NonlinearForm res(&fes);
         auto *integ = new mach::TestBCIntegrator;
         res.AddBdrFaceIntegrator(integ);

         // extract mesh nodes and get their finite-element space
         auto &x_nodes = *mesh.GetNodes();
         auto &mesh_fes = *x_nodes.FESpace();

         // initialize the vector that we use to perturb the mesh nodes
         GridFunction v(&mesh_fes);
         VectorFunctionCoefficient v_pert(dim, randVectorState);
         v.ProjectCoefficient(v_pert);

         // evaluate d(psi^T R)/dx and contract with v
         LinearForm dfdx(&mesh_fes);
         // dfdx.AddBdrFaceIntegrator(
         dfdx.AddBoundaryIntegrator(
            new mach::TestBCIntegratorMeshRevSens(state, adjoint, *integ));
         dfdx.Assemble();
         double dfdx_v = dfdx * v;

         // now compute the finite-difference approximation...
         GridFunction x_pert(x_nodes);
         GridFunction r(&fes);
         x_pert.Add(delta, v);
         mesh.SetNodes(x_pert);
         fes.Update();
         res.Mult(state, r);
         double dfdx_v_fd = adjoint * r;
         x_pert.Add(-2 * delta, v);
         mesh.SetNodes(x_pert);
         fes.Update();
         res.Mult(state, r);
         dfdx_v_fd -= adjoint * r;
         dfdx_v_fd /= (2 * delta);
         mesh.SetNodes(x_nodes); // remember to reset the mesh nodes
         fes.Update();

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-8));
      }
   }
}

TEST_CASE("L2ProjectionIntegrator::AssembleElementGrad")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge,
                                     num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   mfem::ConstantCoefficient one(1.0);
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         L2_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize state; here we randomly perturb a constant state
         GridFunction state(&fes);
         FunctionCoefficient pert(randState);
         state.ProjectCoefficient(pert);

         NonlinearForm res(&fes);
         res.AddDomainIntegrator(new mach::L2ProjectionIntegrator(one));

         // initialize the vector that the Jacobian multiplies
         GridFunction v(&fes);
         v.ProjectCoefficient(pert);

         // evaluate the Jacobian and compute its product with v
         Operator& jac = res.GetGradient(state);
         GridFunction jac_v(&fes);
         jac.Mult(v, jac_v);

         // now compute the finite-difference approximation...
         GridFunction r(&fes), jac_v_fd(&fes);
         state.Add(-delta, v);
         res.Mult(state, r);
         state.Add(2*delta, v);
         res.Mult(state, jac_v_fd);
         jac_v_fd -= r;
         jac_v_fd /= (2*delta);

         for (int i = 0; i < jac_v.Size(); ++i)
         {
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)).margin(1e-6));
         }
      }
   }
}

TEST_CASE("L2ProjectionIntegratorMeshRevSens::AssembleRHSElementVect")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge,
                                     num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   mfem::ConstantCoefficient one(1.0);
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         L2_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize state; here we randomly perturb a constant state
         GridFunction state(&fes);
         GridFunction adjoint(&fes);
         FunctionCoefficient pert(randState);
         state.ProjectCoefficient(pert);
         adjoint.ProjectCoefficient(pert);

         NonlinearForm res(&fes);
         auto *integ = new mach::L2ProjectionIntegrator(one);
         res.AddDomainIntegrator(integ);

         // extract mesh nodes and get their finite-element space
         auto &x_nodes = *mesh.GetNodes();
         auto &mesh_fes = *x_nodes.FESpace();

         // initialize the vector that we use to perturb the mesh nodes
         GridFunction v(&mesh_fes);
         VectorFunctionCoefficient v_pert(dim, randVectorState);
         v.ProjectCoefficient(v_pert);

         // evaluate d(psi^T R)/dx and contract with v
         LinearForm dfdx(&mesh_fes);
         dfdx.AddDomainIntegrator(
            new mach::L2ProjectionIntegratorMeshRevSens(state, adjoint, *integ));
         dfdx.Assemble();
         double dfdx_v = dfdx * v;

         // now compute the finite-difference approximation...
         GridFunction x_pert(x_nodes);
         GridFunction r(&fes);
         x_pert.Add(delta, v);
         mesh.SetNodes(x_pert);
         fes.Update();
         res.Mult(state, r);
         double dfdx_v_fd = adjoint * r;
         x_pert.Add(-2 * delta, v);
         mesh.SetNodes(x_pert);
         fes.Update();
         res.Mult(state, r);
         dfdx_v_fd -= adjoint * r;
         dfdx_v_fd /= (2 * delta);
         mesh.SetNodes(x_nodes); // remember to reset the mesh nodes
         fes.Update();

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-8));
      }
   }
}

TEST_CASE("ThermalContactResistanceIntegrator::AssembleFaceGrad")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge,
                                     num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         L2_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize state; here we randomly perturb a constant state
         GridFunction state(&fes);
         FunctionCoefficient pert(randState);
         state.ProjectCoefficient(pert);

         NonlinearForm res(&fes);
         res.AddInteriorFaceIntegrator(new mach::ThermalContactResistanceIntegrator);

         // initialize the vector that the Jacobian multiplies
         GridFunction v(&fes);
         v.ProjectCoefficient(pert);

         // evaluate the Jacobian and compute its product with v
         Operator& jac = res.GetGradient(state);
         GridFunction jac_v(&fes);
         jac.Mult(v, jac_v);

         // now compute the finite-difference approximation...
         GridFunction r(&fes), jac_v_fd(&fes);
         state.Add(-delta, v);
         res.Mult(state, r);
         state.Add(2*delta, v);
         res.Mult(state, jac_v_fd);
         jac_v_fd -= r;
         jac_v_fd /= (2*delta);

         for (int i = 0; i < jac_v.Size(); ++i)
         {
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)).margin(1e-6));
         }
      }
   }
}

TEST_CASE("ThermalContactResistanceIntegratorMeshRevSens::AssembleRHSElementVect")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge,
                                     num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         L2_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize state; here we randomly perturb a constant state
         GridFunction state(&fes);
         GridFunction adjoint(&fes);
         FunctionCoefficient pert(randState);
         state.ProjectCoefficient(pert);
         adjoint.ProjectCoefficient(pert);

         NonlinearForm res(&fes);
         auto *integ = new mach::ThermalContactResistanceIntegrator;
         res.AddInteriorFaceIntegrator(integ);

         // extract mesh nodes and get their finite-element space
         auto &x_nodes = *mesh.GetNodes();
         auto &mesh_fes = *x_nodes.FESpace();

         // initialize the vector that we use to perturb the mesh nodes
         GridFunction v(&mesh_fes);
         VectorFunctionCoefficient v_pert(dim, randVectorState);
         v.ProjectCoefficient(v_pert);

         // evaluate d(psi^T R)/dx and contract with v
         LinearForm dfdx(&mesh_fes);
         dfdx.AddInteriorFaceIntegrator(
            new mach::ThermalContactResistanceIntegratorMeshRevSens(mesh_fes, state, adjoint, *integ));
         dfdx.Assemble();
         double dfdx_v = dfdx * v;

         // now compute the finite-difference approximation...
         GridFunction x_pert(x_nodes);
         GridFunction r(&fes);
         x_pert.Add(delta, v);
         mesh.SetNodes(x_pert);
         fes.Update();
         res.Mult(state, r);
         double dfdx_v_fd = adjoint * r;
         x_pert.Add(-2 * delta, v);
         mesh.SetNodes(x_pert);
         fes.Update();
         res.Mult(state, r);
         dfdx_v_fd -= adjoint * r;
         dfdx_v_fd /= (2 * delta);
         mesh.SetNodes(x_nodes); // remember to reset the mesh nodes
         fes.Update();

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-8));
      }
   }
}

TEST_CASE("ConvectionBCIntegrator::AssembleFaceGrad")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge,
                                     num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         H1_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize state; here we randomly perturb a constant state
         GridFunction state(&fes);
         FunctionCoefficient pert(randState);
         state.ProjectCoefficient(pert);

         NonlinearForm res(&fes);
         res.AddBdrFaceIntegrator(new mach::ConvectionBCIntegrator);

         // initialize the vector that the Jacobian multiplies
         GridFunction v(&fes);
         v.ProjectCoefficient(pert);

         // evaluate the Jacobian and compute its product with v
         Operator& jac = res.GetGradient(state);
         GridFunction jac_v(&fes);
         jac.Mult(v, jac_v);

         // now compute the finite-difference approximation...
         GridFunction r(&fes), jac_v_fd(&fes);
         state.Add(-delta, v);
         res.Mult(state, r);
         state.Add(2*delta, v);
         res.Mult(state, jac_v_fd);
         jac_v_fd -= r;
         jac_v_fd /= (2*delta);

         for (int i = 0; i < jac_v.Size(); ++i)
         {
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)).margin(1e-6));
         }
      }
   }
}

TEST_CASE("ConvectionBCIntegratorMeshRevSens::AssembleRHSElementVect")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge,
                                     num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   NonLinearCoefficient nu;
   // LinearCoefficient nu;
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         H1_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize state; here we randomly perturb a constant state
         GridFunction state(&fes);
         GridFunction adjoint(&fes);
         FunctionCoefficient pert(randState);
         state.ProjectCoefficient(pert);
         adjoint.ProjectCoefficient(pert);

         NonlinearForm res(&fes);
         auto *integ = new mach::ConvectionBCIntegrator;
         res.AddBdrFaceIntegrator(integ);

         // extract mesh nodes and get their finite-element space
         auto &x_nodes = *mesh.GetNodes();
         auto &mesh_fes = *x_nodes.FESpace();

         // initialize the vector that we use to perturb the mesh nodes
         GridFunction v(&mesh_fes);
         VectorFunctionCoefficient v_pert(dim, randVectorState);
         v.ProjectCoefficient(v_pert);

         // evaluate d(psi^T R)/dx and contract with v
         LinearForm dfdx(&mesh_fes);
         // dfdx.AddBdrFaceIntegrator(
         dfdx.AddBoundaryIntegrator(
            new mach::ConvectionBCIntegratorMeshRevSens(state, adjoint, *integ));
         dfdx.Assemble();
         double dfdx_v = dfdx * v;

         // now compute the finite-difference approximation...
         GridFunction x_pert(x_nodes);
         GridFunction r(&fes);
         x_pert.Add(delta, v);
         mesh.SetNodes(x_pert);
         fes.Update();
         res.Mult(state, r);
         double dfdx_v_fd = adjoint * r;
         x_pert.Add(-2 * delta, v);
         mesh.SetNodes(x_pert);
         fes.Update();
         res.Mult(state, r);
         dfdx_v_fd -= adjoint * r;
         dfdx_v_fd /= (2 * delta);
         mesh.SetNodes(x_nodes); // remember to reset the mesh nodes
         fes.Update();

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-8));
      }
   }
}

TEST_CASE("ConvectionBCIntegratorHRevSens::AssembleRHSElementVect")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge,
                                     num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         H1_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize state; here we randomly perturb a constant state
         GridFunction state(&fes);
         GridFunction adjoint(&fes);
         FunctionCoefficient pert(randState);
         state.ProjectCoefficient(pert);
         adjoint.ProjectCoefficient(pert);

         NonlinearForm res(&fes);
         auto *integ = new mach::ConvectionBCIntegrator;
         setInputs(*integ, {
            {"h", 10.0}
         });
         res.AddBdrFaceIntegrator(integ);

         // evaluate d(psi^T R)/dx and contract with v
         NonlinearForm dfdx(&fes);
         dfdx.AddBdrFaceIntegrator(
         // dfdx.AddBoundaryIntegrator(
            new mach::ConvectionBCIntegratorHRevSens(adjoint, *integ));

         // random perturbation
         double v = 0.3042434;
         double dfdx_v = dfdx.GetEnergy(state) * v;

         // now compute the finite-difference approximation...
         GridFunction r(&fes);
         setInputs(*integ, {
            {"h", 10 + delta * v}
         });
         res.Mult(state, r);
         double dfdx_v_fd = adjoint * r;

         setInputs(*integ, {
            {"h", 10 - delta * v}
         });
         res.Mult(state, r);
         dfdx_v_fd -= adjoint * r;
         dfdx_v_fd /= (2 * delta);

         std::cout << "dfdx_v: " << dfdx_v << " dfdx_v_fd: " << dfdx_v_fd << "\n";
         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-8));
      }
   }
}

TEST_CASE("ConvectionBCIntegratorFluidTempRevSens::AssembleRHSElementVect")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge,
                                     num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         H1_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize state; here we randomly perturb a constant state
         GridFunction state(&fes);
         GridFunction adjoint(&fes);
         FunctionCoefficient pert(randState);
         state.ProjectCoefficient(pert);
         adjoint.ProjectCoefficient(pert);

         NonlinearForm res(&fes);
         auto *integ = new mach::ConvectionBCIntegrator;
         setInputs(*integ, {
            {"fluid_temp", 100.0}
         });
         res.AddBdrFaceIntegrator(integ);

         // evaluate d(psi^T R)/dx and contract with v
         NonlinearForm dfdx(&fes);
         dfdx.AddBdrFaceIntegrator(
         // dfdx.AddBoundaryIntegrator(
            new mach::ConvectionBCIntegratorFluidTempRevSens(adjoint, *integ));

         // random perturbation
         double v = 0.3042434;
         double dfdx_v = dfdx.GetEnergy(state) * v;

         // now compute the finite-difference approximation...
         GridFunction r(&fes);
         setInputs(*integ, {
            {"fluid_temp", 100 + delta * v}
         });
         res.Mult(state, r);
         double dfdx_v_fd = adjoint * r;

         setInputs(*integ, {
            {"fluid_temp", 100 - delta * v}
         });
         res.Mult(state, r);
         dfdx_v_fd -= adjoint * r;
         dfdx_v_fd /= (2 * delta);

         std::cout << "dfdx_v: " << dfdx_v << " dfdx_v_fd: " << dfdx_v_fd << "\n";
         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-8));
      }
   }
}

TEST_CASE("OutfluxBCIntegrator::AssembleFaceGrad")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge,
                                     num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         H1_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize state; here we randomly perturb a constant state
         GridFunction state(&fes);
         FunctionCoefficient pert(randState);
         state.ProjectCoefficient(pert);

         NonlinearForm res(&fes);
         res.AddBdrFaceIntegrator(new mach::OutfluxBCIntegrator);

         // initialize the vector that the Jacobian multiplies
         GridFunction v(&fes);
         v.ProjectCoefficient(pert);

         // evaluate the Jacobian and compute its product with v
         Operator& jac = res.GetGradient(state);
         GridFunction jac_v(&fes);
         jac.Mult(v, jac_v);

         // now compute the finite-difference approximation...
         GridFunction r(&fes), jac_v_fd(&fes);
         state.Add(-delta, v);
         res.Mult(state, r);
         state.Add(2*delta, v);
         res.Mult(state, jac_v_fd);
         jac_v_fd -= r;
         jac_v_fd /= (2*delta);

         for (int i = 0; i < jac_v.Size(); ++i)
         {
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)).margin(1e-6));
         }
      }
   }
}

TEST_CASE("OutfluxBCIntegratorMeshRevSens::AssembleRHSElementVect")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge,
                                     num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   NonLinearCoefficient nu;
   // LinearCoefficient nu;
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         H1_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize state; here we randomly perturb a constant state
         GridFunction state(&fes);
         GridFunction adjoint(&fes);
         FunctionCoefficient pert(randState);
         state.ProjectCoefficient(pert);
         adjoint.ProjectCoefficient(pert);

         NonlinearForm res(&fes);
         auto *integ = new mach::OutfluxBCIntegrator;
         res.AddBdrFaceIntegrator(integ);

         // extract mesh nodes and get their finite-element space
         auto &x_nodes = *mesh.GetNodes();
         auto &mesh_fes = *x_nodes.FESpace();

         // initialize the vector that we use to perturb the mesh nodes
         GridFunction v(&mesh_fes);
         VectorFunctionCoefficient v_pert(dim, randVectorState);
         v.ProjectCoefficient(v_pert);

         // evaluate d(psi^T R)/dx and contract with v
         LinearForm dfdx(&mesh_fes);
         // dfdx.AddBdrFaceIntegrator(
         dfdx.AddBoundaryIntegrator(
            new mach::OutfluxBCIntegratorMeshRevSens(state, adjoint, *integ));
         dfdx.Assemble();
         double dfdx_v = dfdx * v;

         // now compute the finite-difference approximation...
         GridFunction x_pert(x_nodes);
         GridFunction r(&fes);
         x_pert.Add(delta, v);
         mesh.SetNodes(x_pert);
         fes.Update();
         res.Mult(state, r);
         double dfdx_v_fd = adjoint * r;
         x_pert.Add(-2 * delta, v);
         mesh.SetNodes(x_pert);
         fes.Update();
         res.Mult(state, r);
         dfdx_v_fd -= adjoint * r;
         dfdx_v_fd /= (2 * delta);
         mesh.SetNodes(x_nodes); // remember to reset the mesh nodes
         fes.Update();

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-8));
      }
   }
}
