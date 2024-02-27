#include <random>

#include "catch.hpp"
#include "mfem.hpp"

#include "electromag_test_data.hpp"

#include "thermal_integ.hpp"

using namespace mfem;
using namespace miso;
using namespace electromag_data;

TEST_CASE("TestBCIntegratorMeshRevSens::AssembleRHSElementVect")
{
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
         auto *integ = new miso::TestBCIntegrator;
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
            new miso::TestBCIntegratorMeshRevSens(state, adjoint, *integ));
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
         res.AddDomainIntegrator(new miso::L2ProjectionIntegrator(one));

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
         auto *integ = new miso::L2ProjectionIntegrator(one);
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
            new miso::L2ProjectionIntegratorMeshRevSens(state, adjoint, *integ));
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
         res.AddInteriorFaceIntegrator(new miso::ThermalContactResistanceIntegrator);

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
         auto *integ = new miso::ThermalContactResistanceIntegrator;
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
            new miso::ThermalContactResistanceIntegratorMeshRevSens(mesh_fes, state, adjoint, *integ));
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

TEST_CASE("ThermalContactResistanceIntegratorHRevSens::GetFaceEnergy")
{
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
         auto *integ = new miso::ThermalContactResistanceIntegrator;
         setInputs(*integ, {
            {"h", 10.0}
         });
         res.AddInteriorFaceIntegrator(integ);

         // evaluate d(psi^T R)/dx and contract with v
         NonlinearForm dfdx(&fes);
         dfdx.AddInteriorFaceIntegrator(
            new miso::ThermalContactResistanceIntegratorHRevSens(state, adjoint, *integ));

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

TEST_CASE("InternalConvectionInterfaceIntegrator::AssembleFaceGrad")
{
   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge,
                                     num_edge,
                                     Element::TRIANGLE);
   
   // assign attributes to left and right sides
   for (int i = 0; i < mesh.GetNE(); ++i)
   {
      auto *elem = mesh.GetElement(i);

      Array<int> verts;
      elem->GetVertices(verts);

      bool left = true;
      for (int i = 0; i < verts.Size(); ++i)
      {
         auto *vtx = mesh.GetVertex(verts[i]);
         if (vtx[0] <= 0.5)
         {
            left = left;
         }
         else
         {
            left = false;
         }
      }
      if (left)
      {
         elem->SetAttribute(1);
      }
      else
      {
         elem->SetAttribute(2);
      }
   }

   // add internal boundary elements
   for (int i = 0; i < mesh.GetNumFaces(); ++i)
   {
      int e1, e2;
      mesh.GetFaceElements(i, &e1, &e2);
      if (e1 >= 0 && e2 >= 0 && mesh.GetAttribute(e1) != mesh.GetAttribute(e2))
      {
         // This is the internal face between attributes.
         auto *new_elem = mesh.GetFace(i)->Duplicate(&mesh);
         new_elem->SetAttribute(5);
         mesh.AddBdrElement(new_elem);
      }
   }
   mesh.FinalizeTopology(); // Finalize to build relevant tables
   mesh.Finalize();
   mesh.SetAttributes(); 
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
         mfem::Array<int> bdr_attr(mesh.bdr_attributes.Max());
         bdr_attr = 0;
         bdr_attr[4] = 1;
         res.AddInternalBoundaryFaceIntegrator(
            new miso::InternalConvectionInterfaceIntegrator(1.0, 1.0),
            bdr_attr);

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

         // std::cout << "jac_v: ";
         // jac_v.Print(mfem::out, jac_v.Size());

         // std::cout << "jac_v_fd: ";
         jac_v_fd.Print(mfem::out, jac_v_fd.Size());
         for (int i = 0; i < jac_v.Size(); ++i)
         {
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)).margin(1e-6));
         }
      }
   }
}

TEST_CASE("InternalConvectionInterfaceIntegratorMeshRevSens::AssembleRHSElementVect")
{
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
         auto *integ = new miso::InternalConvectionInterfaceIntegrator;
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
            new miso::InternalConvectionInterfaceIntegratorMeshRevSens(mesh_fes, state, adjoint, *integ));
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

TEST_CASE("InternalConvectionInterfaceIntegratorHRevSens::GetFaceEnergy")
{
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
         auto *integ = new miso::InternalConvectionInterfaceIntegrator;
         setInputs(*integ, {
            {"h", 10.0}
         });
         res.AddInteriorFaceIntegrator(integ);

         // evaluate d(psi^T R)/dx and contract with v
         NonlinearForm dfdx(&fes);
         dfdx.AddInteriorFaceIntegrator(
            new miso::InternalConvectionInterfaceIntegratorHRevSens(state, adjoint, *integ));

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

TEST_CASE("InternalConvectionInterfaceIntegratorFluidTempRevSens::GetFaceEnergy")
{
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
         auto *integ = new miso::InternalConvectionInterfaceIntegrator;
         setInputs(*integ, {
            {"fluid_temp", 10.0}
         });
         res.AddInteriorFaceIntegrator(integ);

         // evaluate d(psi^T R)/dx and contract with v
         NonlinearForm dfdx(&fes);
         dfdx.AddInteriorFaceIntegrator(
            new miso::InternalConvectionInterfaceIntegratorFluidTempRevSens(adjoint, *integ));

         // random perturbation
         double v = 0.3042434;
         double dfdx_v = dfdx.GetEnergy(state) * v;

         // now compute the finite-difference approximation...
         GridFunction r(&fes);
         setInputs(*integ, {
            {"fluid_temp", 10 + delta * v}
         });
         res.Mult(state, r);
         double dfdx_v_fd = adjoint * r;

         setInputs(*integ, {
            {"fluid_temp", 10 - delta * v}
         });
         res.Mult(state, r);
         dfdx_v_fd -= adjoint * r;
         dfdx_v_fd /= (2 * delta);

         std::cout << "dfdx_v: " << dfdx_v << " dfdx_v_fd: " << dfdx_v_fd << "\n";
         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-8));
      }
   }
}

TEST_CASE("ConvectionBCIntegrator::AssembleFaceGrad")
{
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
         res.AddBdrFaceIntegrator(new miso::ConvectionBCIntegrator);

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
         auto *integ = new miso::ConvectionBCIntegrator;
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
            new miso::ConvectionBCIntegratorMeshRevSens(state, adjoint, *integ));
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

TEST_CASE("ConvectionBCIntegratorHRevSens::GetFaceEnergy")
{
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
         auto *integ = new miso::ConvectionBCIntegrator;
         setInputs(*integ, {
            {"h", 10.0}
         });
         res.AddBdrFaceIntegrator(integ);

         // evaluate d(psi^T R)/dx and contract with v
         NonlinearForm dfdx(&fes);
         dfdx.AddBdrFaceIntegrator(
         // dfdx.AddBoundaryIntegrator(
            new miso::ConvectionBCIntegratorHRevSens(adjoint, *integ));

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

TEST_CASE("ConvectionBCIntegratorFluidTempRevSens::GetFaceEnergy")
{
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
         auto *integ = new miso::ConvectionBCIntegrator;
         setInputs(*integ, {
            {"fluid_temp", 100.0}
         });
         res.AddBdrFaceIntegrator(integ);

         // evaluate d(psi^T R)/dx and contract with v
         NonlinearForm dfdx(&fes);
         dfdx.AddBdrFaceIntegrator(
         // dfdx.AddBoundaryIntegrator(
            new miso::ConvectionBCIntegratorFluidTempRevSens(adjoint, *integ));

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
         res.AddBdrFaceIntegrator(new miso::OutfluxBCIntegrator);

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
         auto *integ = new miso::OutfluxBCIntegrator;
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
            new miso::OutfluxBCIntegratorMeshRevSens(state, adjoint, *integ));
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
