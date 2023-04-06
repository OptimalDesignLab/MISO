#include <random>

#include "catch.hpp"
#include "mfem.hpp"

#include "coefficient.hpp"
#include "euler_test_data.hpp"
#include "electromag_test_data.hpp"
#include "mfem_common_integ.hpp"

namespace
{

using namespace mfem;

double func(const Vector &x)
{
  return (x(0) + x(1) + x(2));
}

void funcRevDiff(const Vector &x, const double Q_bar, Vector &x_bar)
{
   x_bar.SetSize(3);
   x_bar = Q_bar;
}

} // namespace

TEST_CASE("BoundaryNormalIntegrator::GetFaceEnergy")
{
   using namespace mfem;

   //Adapted from TEST_CASE("ACLossFunctionalIntegrator::GetElementEnergy") in test_electromag_integ

   // generate a 8 element mesh, simple 2D domain, 0<=x<=1, 0<=y<=1
   int num_edge = 2;
   double x_length = 1.5;
   double y_length = 2.0;
   auto mesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE, 
                                     false, x_length, y_length);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   //Function Coefficient model Representing the Temperature Field (elfun)
   FunctionCoefficient Tfield_model(
      [](const mfem::Vector &x)
      {
         // T(x)=x(0)^2
         return std::pow(x(0),2);
      });
   
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         // Create the finite element collection and finite element space for the current order
         H1_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // extract mesh nodes and get their finite-element space
         auto &x_nodes = *mesh.GetNodes();
         auto &mesh_fes = *x_nodes.FESpace();

         // Create the temperature_field grid function by mapping the function coefficient to a grid function
         GridFunction temperature_field_test(&fes);
         temperature_field_test.ProjectCoefficient(Tfield_model);

         // Set the conductivity kappa to be a constant 1
         ConstantCoefficient kappa(1.0);

         // Define the functional integrator that will be used to compute the BoundaryFlux
         NonlinearForm functional(&fes);
         functional.AddBdrFaceIntegrator(new mach::BoundaryNormalIntegrator(kappa));

         // Compute the value of the boundary flux
         double BoundaryFlux = functional.GetEnergy(temperature_field_test);
         // std::cout << "BoundaryFlux = " << BoundaryFlux << "\n";

         /// TODO: Assert the computed boundary flux is correct
         double expected_BoundaryFlux = 2.0*(x_length*y_length); //2*the area for T(x)=x(0)^2
         //REQUIRE(BoundaryFlux == Approx(expected_BoundaryFlux)); // Assert the BoundaryNormalIntegrator is working as expected
      }
   }
}

TEST_CASE("VolumeIntegrator::GetElementEnergy (2D)")
{
   using namespace mfem;

   int num_edge = 2;
   auto smesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                      Element::TRIANGLE, true,
                                      2.0, 3.0);

   ParMesh mesh(MPI_COMM_WORLD, smesh);
   mesh.EnsureNodes();
   auto dim = mesh.Dimension();

   auto p = 1;
   H1_FECollection fec(p, dim);
   ParFiniteElementSpace fes(&mesh, &fec);

   ParNonlinearForm form(&fes);
   form.AddDomainIntegrator(new mach::VolumeIntegrator);

   HypreParVector tv(&fes);
   auto area = form.GetEnergy(tv);

   REQUIRE(area == Approx(6.0).margin(1e-10));
}

TEST_CASE("VolumeIntegrator::GetElementEnergy (3D)")
{
   using namespace mfem;

   int num_edge = 2;
   auto smesh = Mesh::MakeCartesian3D(num_edge, num_edge, num_edge,
                                      Element::TETRAHEDRON,
                                      2.0, 1.0, 1.0, true);

   ParMesh mesh(MPI_COMM_WORLD, smesh);
   mesh.EnsureNodes();
   auto dim = mesh.Dimension();

   auto p = 1;
   H1_FECollection fec(p, dim);
   ParFiniteElementSpace fes(&mesh, &fec);

   ParNonlinearForm form(&fes);
   form.AddDomainIntegrator(new mach::VolumeIntegrator);

   HypreParVector tv(&fes);
   auto volume = form.GetEnergy(tv);

   REQUIRE(volume == Approx(2.0).margin(1e-10));
}

TEST_CASE("VolumeIntegratorMeshSens::AssembleRHSElementVect (2D)")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   mfem::FunctionCoefficient model(
      [](const mfem::Vector &x)
      {
         double q = 0;
         for (int i = 0; i < x.Size(); ++i)
         {
            q += pow(x(i), 2);
         }
         return q;
      },
      [](const mfem::Vector &x, const double q_bar, mfem::Vector &x_bar)
      {
         for (int i = 0; i < x.Size(); ++i)
         {
            x_bar(i) += 2 * x(i) * q_bar;
         }
      });

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         // mesh.SetCurvature(p);

         H1_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize state
         GridFunction a(&fes);
         FunctionCoefficient pert(randState);
         a.ProjectCoefficient(pert);

         auto *integ = new mach::VolumeIntegrator(&model);
         NonlinearForm functional(&fes);
         functional.AddDomainIntegrator(integ);

         // extract mesh nodes and get their finite-element space
         auto &x_nodes = *mesh.GetNodes();
         auto &mesh_fes = *x_nodes.FESpace();

         // create v displacement field
         GridFunction v(&mesh_fes);
         VectorFunctionCoefficient v_pert(dim, randVectorState);
         v.ProjectCoefficient(v_pert);

         // initialize the vector that dJdx multiplies
         GridFunction p(&mesh_fes);
         p.ProjectCoefficient(v_pert);

         // evaluate dJdx and compute its product with p
         LinearForm dJdx(&mesh_fes);
         dJdx.AddDomainIntegrator(
            new mach::VolumeIntegratorMeshSens(a, *integ));
         dJdx.Assemble();
         double dJdx_dot_p = dJdx * p;

         // now compute the finite-difference approximation...
         GridFunction x_pert(x_nodes);
         x_pert.Add(-delta, p);
         mesh.SetNodes(x_pert);
         fes.Update();
         double dJdx_dot_p_fd = -functional.GetEnergy(a);
         x_pert.Add(2 * delta, p);
         mesh.SetNodes(x_pert);
         fes.Update();
         dJdx_dot_p_fd += functional.GetEnergy(a);
         dJdx_dot_p_fd /= (2 * delta);
         mesh.SetNodes(x_nodes); // remember to reset the mesh nodes
         fes.Update();

         REQUIRE(dJdx_dot_p == Approx(dJdx_dot_p_fd));
      }
   }
}

TEST_CASE("StateIntegrator::GetElementEnergy (3D)")
{
   using namespace mfem;

   int num_edge = 3;
   auto smesh = Mesh::MakeCartesian3D(num_edge, num_edge, num_edge,
                                      Element::TETRAHEDRON,
                                      2*M_PI, 1.0, 1.0, true);

   ParMesh mesh(MPI_COMM_WORLD, smesh);
   mesh.EnsureNodes();
   auto dim = mesh.Dimension();

   auto p = 1;
   H1_FECollection fec(p, dim);
   ParFiniteElementSpace fes(&mesh, &fec);

   ParNonlinearForm form(&fes);
   form.AddDomainIntegrator(new mach::StateIntegrator);

   ParGridFunction gf(&fes);
   FunctionCoefficient coeff([](const mfem::Vector &p){
      return sin(p(0)) * sin(p(0));
   });
   gf.ProjectCoefficient(coeff);
   HypreParVector tv(&fes);
   gf.GetTrueDofs(tv);

   ParNonlinearForm v_form(&fes);
   v_form.AddDomainIntegrator(new mach::VolumeIntegrator);
   auto volume = v_form.GetEnergy(tv);

   auto rms = sqrt(form.GetEnergy(tv) / volume);
   REQUIRE(rms == Approx(sqrt(2)/2).margin(1e-10));
}

TEST_CASE("MagnitudeCurlStateIntegrator::AssembleElementVector")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         H1_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize state
         GridFunction a(&fes);
         FunctionCoefficient pert(randState);
         a.ProjectCoefficient(pert);

         NonlinearForm functional(&fes);
         functional.AddDomainIntegrator(
            new mach::MagnitudeCurlStateIntegrator);

         // initialize the vector that dJdu multiplies
         GridFunction p(&fes);
         p.ProjectCoefficient(pert);

         // evaluate dJdu and compute its product with v
         GridFunction dJdu(&fes);
         functional.Mult(a, dJdu);
         double dJdu_dot_p = InnerProduct(dJdu, p);

         // now compute the finite-difference approximation...
         GridFunction q_pert(a);
         q_pert.Add(-delta, p);
         double dJdu_dot_p_fd = -functional.GetEnergy(q_pert);
         q_pert.Add(2 * delta, p);
         dJdu_dot_p_fd += functional.GetEnergy(q_pert);
         dJdu_dot_p_fd /= (2 * delta);

         REQUIRE(dJdu_dot_p == Approx(dJdu_dot_p_fd));
      }
   }
}

TEST_CASE("MagnitudeCurlStateIntegratorMeshSens::AssembleRHSElementVect")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         H1_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize state
         GridFunction a(&fes);
         FunctionCoefficient pert(randState);
         a.ProjectCoefficient(pert);

         auto *integ = new mach::MagnitudeCurlStateIntegrator;
         NonlinearForm functional(&fes);
         functional.AddDomainIntegrator(integ);

         // extract mesh nodes and get their finite-element space
         auto &x_nodes = *mesh.GetNodes();
         auto &mesh_fes = *x_nodes.FESpace();

         // create v displacement field
         GridFunction v(&mesh_fes);
         VectorFunctionCoefficient v_pert(dim, randVectorState);
         v.ProjectCoefficient(v_pert);

         // initialize the vector that dJdx multiplies
         GridFunction p(&mesh_fes);
         p.ProjectCoefficient(v_pert);

         // evaluate dJdx and compute its product with p
         LinearForm dJdx(&mesh_fes);
         dJdx.AddDomainIntegrator(
            new mach::MagnitudeCurlStateIntegratorMeshSens(a, *integ));
         dJdx.Assemble();
         double dJdx_dot_p = dJdx * p;

         // now compute the finite-difference approximation...
         GridFunction x_pert(x_nodes);
         x_pert.Add(-delta, p);
         mesh.SetNodes(x_pert);
         fes.Update();
         double dJdx_dot_p_fd = -functional.GetEnergy(a);
         x_pert.Add(2 * delta, p);
         mesh.SetNodes(x_pert);
         fes.Update();
         dJdx_dot_p_fd += functional.GetEnergy(a);
         dJdx_dot_p_fd /= (2 * delta);
         mesh.SetNodes(x_nodes); // remember to reset the mesh nodes
         fes.Update();

         REQUIRE(dJdx_dot_p == Approx(dJdx_dot_p_fd));
      }
   }
}

TEST_CASE("IEAggregateIntegratorNumerator::AssembleElementVector")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         H1_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize state
         GridFunction a(&fes);
         FunctionCoefficient a_field([](const mfem::Vector &x){
            return x(0) * x(0) + x(1) * x(1);
         });
         a.ProjectCoefficient(a_field);

         double actual_max = a.Max();

         NonlinearForm functional(&fes);

         auto *integ = new mach::IEAggregateIntegratorNumerator(1.0);
         setInputs(*integ, {{"true_max", 2.0}});
         functional.AddDomainIntegrator(integ);

         // initialize the vector that dJdu multiplies
         GridFunction p(&fes);
         FunctionCoefficient pert(randState);
         p.ProjectCoefficient(pert);

         // evaluate dJdu and compute its product with v
         GridFunction dJdu(&fes);
         functional.Mult(a, dJdu);
         double dJdu_dot_p = InnerProduct(dJdu, p);

         // now compute the finite-difference approximation...
         GridFunction q_pert(a);
         q_pert.Add(-delta, p);
         double dJdu_dot_p_fd = -functional.GetEnergy(q_pert);
         q_pert.Add(2 * delta, p);
         dJdu_dot_p_fd += functional.GetEnergy(q_pert);
         dJdu_dot_p_fd /= (2 * delta);

         REQUIRE(dJdu_dot_p == Approx(dJdu_dot_p_fd));
      }
   }
}

TEST_CASE("IEAggregateIntegratorNumeratorMeshSens::AssembleRHSElementVect")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         H1_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize state
         GridFunction a(&fes);
         // FunctionCoefficient pert(randState);
         FunctionCoefficient a_field([](const mfem::Vector &x){
            return x(0) * x(0) + x(1) * x(1);
         });
         a.ProjectCoefficient(a_field);

         double actual_max = a.Max();

         auto *integ = new mach::IEAggregateIntegratorNumerator(1.0);
         NonlinearForm functional(&fes);
         functional.AddDomainIntegrator(integ);

         // extract mesh nodes and get their finite-element space
         auto &x_nodes = *mesh.GetNodes();
         auto &mesh_fes = *x_nodes.FESpace();

         // create v displacement field
         GridFunction v(&mesh_fes);
         VectorFunctionCoefficient v_pert(dim, randVectorState);
         v.ProjectCoefficient(v_pert);

         // initialize the vector that dJdx multiplies
         GridFunction p(&mesh_fes);
         p.ProjectCoefficient(v_pert);

         // evaluate dJdx and compute its product with p
         LinearForm dJdx(&mesh_fes);
         dJdx.AddDomainIntegrator(
            new mach::IEAggregateIntegratorNumeratorMeshSens(a, *integ));
         dJdx.Assemble();
         double dJdx_dot_p = dJdx * p;

         // now compute the finite-difference approximation...
         GridFunction x_pert(x_nodes);
         x_pert.Add(-delta, p);
         mesh.SetNodes(x_pert);
         fes.Update();
         double dJdx_dot_p_fd = -functional.GetEnergy(a);
         x_pert.Add(2 * delta, p);
         mesh.SetNodes(x_pert);
         fes.Update();
         dJdx_dot_p_fd += functional.GetEnergy(a);
         dJdx_dot_p_fd /= (2 * delta);
         mesh.SetNodes(x_nodes); // remember to reset the mesh nodes
         fes.Update();

         REQUIRE(dJdx_dot_p == Approx(dJdx_dot_p_fd));
      }
   }
}

TEST_CASE("IEAggregateIntegratorDenominator::AssembleElementVector")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         H1_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize state
         GridFunction a(&fes);
         FunctionCoefficient a_field([](const mfem::Vector &x){
            return x(0) * x(0) + x(1) * x(1);
         });
         a.ProjectCoefficient(a_field);

         double actual_max = a.Max();

         NonlinearForm functional(&fes);

         auto *integ = new mach::IEAggregateIntegratorDenominator(1.0);
         setInputs(*integ, {{"true_max", 2.0}});
         functional.AddDomainIntegrator(integ);

         // initialize the vector that dJdu multiplies
         GridFunction p(&fes);
         FunctionCoefficient pert(randState);
         p.ProjectCoefficient(pert);

         // evaluate dJdu and compute its product with v
         GridFunction dJdu(&fes);
         functional.Mult(a, dJdu);
         double dJdu_dot_p = InnerProduct(dJdu, p);

         // now compute the finite-difference approximation...
         GridFunction q_pert(a);
         q_pert.Add(-delta, p);
         double dJdu_dot_p_fd = -functional.GetEnergy(q_pert);
         q_pert.Add(2 * delta, p);
         dJdu_dot_p_fd += functional.GetEnergy(q_pert);
         dJdu_dot_p_fd /= (2 * delta);

         REQUIRE(dJdu_dot_p == Approx(dJdu_dot_p_fd));
      }
   }
}

TEST_CASE("IEAggregateIntegratorDenominatorMeshSens::AssembleRHSElementVect")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         H1_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize state
         GridFunction a(&fes);
         // FunctionCoefficient pert(randState);
         FunctionCoefficient a_field([](const mfem::Vector &x){
            return x(0) * x(0) + x(1) * x(1);
         });
         a.ProjectCoefficient(a_field);

         double actual_max = a.Max();

         auto *integ = new mach::IEAggregateIntegratorDenominator(1.0);
         NonlinearForm functional(&fes);
         functional.AddDomainIntegrator(integ);

         // extract mesh nodes and get their finite-element space
         auto &x_nodes = *mesh.GetNodes();
         auto &mesh_fes = *x_nodes.FESpace();

         // create v displacement field
         GridFunction v(&mesh_fes);
         VectorFunctionCoefficient v_pert(dim, randVectorState);
         v.ProjectCoefficient(v_pert);

         // initialize the vector that dJdx multiplies
         GridFunction p(&mesh_fes);
         p.ProjectCoefficient(v_pert);

         // evaluate dJdx and compute its product with p
         LinearForm dJdx(&mesh_fes);
         dJdx.AddDomainIntegrator(
            new mach::IEAggregateIntegratorDenominatorMeshSens(a, *integ));
         dJdx.Assemble();
         double dJdx_dot_p = dJdx * p;

         // now compute the finite-difference approximation...
         GridFunction x_pert(x_nodes);
         x_pert.Add(-delta, p);
         mesh.SetNodes(x_pert);
         fes.Update();
         double dJdx_dot_p_fd = -functional.GetEnergy(a);
         x_pert.Add(2 * delta, p);
         mesh.SetNodes(x_pert);
         fes.Update();
         dJdx_dot_p_fd += functional.GetEnergy(a);
         dJdx_dot_p_fd /= (2 * delta);
         mesh.SetNodes(x_nodes); // remember to reset the mesh nodes
         fes.Update();

         REQUIRE(dJdx_dot_p == Approx(dJdx_dot_p_fd));
      }
   }
}

TEST_CASE("IECurlMagnitudeAggregateIntegratorNumerator::AssembleElementVector")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         H1_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize state
         GridFunction a(&fes);
         FunctionCoefficient a_field([](const mfem::Vector &x){
            return x(0) * x(0) + x(1) * x(1);
         });
         a.ProjectCoefficient(a_field);

         double actual_max = a.Max();

         NonlinearForm functional(&fes);
         functional.AddDomainIntegrator(
            new mach::IECurlMagnitudeAggregateIntegratorNumerator(1.0, actual_max));

         // initialize the vector that dJdu multiplies
         GridFunction p(&fes);
         FunctionCoefficient pert(randState);
         p.ProjectCoefficient(pert);

         // evaluate dJdu and compute its product with v
         GridFunction dJdu(&fes);
         functional.Mult(a, dJdu);
         double dJdu_dot_p = InnerProduct(dJdu, p);

         // now compute the finite-difference approximation...
         GridFunction q_pert(a);
         q_pert.Add(-delta, p);
         double dJdu_dot_p_fd = -functional.GetEnergy(q_pert);
         q_pert.Add(2 * delta, p);
         dJdu_dot_p_fd += functional.GetEnergy(q_pert);
         dJdu_dot_p_fd /= (2 * delta);

         REQUIRE(dJdu_dot_p == Approx(dJdu_dot_p_fd));
      }
   }
}

TEST_CASE("IECurlMagnitudeAggregateIntegratorNumeratorMeshSens::AssembleRHSElementVect")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         H1_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize state
         GridFunction a(&fes);
         // FunctionCoefficient pert(randState);
         FunctionCoefficient a_field([](const mfem::Vector &x){
            return x(0) * x(0) + x(1) * x(1);
         });
         a.ProjectCoefficient(a_field);

         double actual_max = a.Max();

         auto *integ = new mach::IECurlMagnitudeAggregateIntegratorNumerator(1.0, actual_max);
         NonlinearForm functional(&fes);
         functional.AddDomainIntegrator(integ);

         // extract mesh nodes and get their finite-element space
         auto &x_nodes = *mesh.GetNodes();
         auto &mesh_fes = *x_nodes.FESpace();

         // create v displacement field
         GridFunction v(&mesh_fes);
         VectorFunctionCoefficient v_pert(dim, randVectorState);
         v.ProjectCoefficient(v_pert);

         // initialize the vector that dJdx multiplies
         GridFunction p(&mesh_fes);
         p.ProjectCoefficient(v_pert);

         // evaluate dJdx and compute its product with p
         LinearForm dJdx(&mesh_fes);
         dJdx.AddDomainIntegrator(
            new mach::IECurlMagnitudeAggregateIntegratorNumeratorMeshSens(a, *integ));
         dJdx.Assemble();
         double dJdx_dot_p = dJdx * p;

         // now compute the finite-difference approximation...
         GridFunction x_pert(x_nodes);
         x_pert.Add(-delta, p);
         mesh.SetNodes(x_pert);
         fes.Update();
         double dJdx_dot_p_fd = -functional.GetEnergy(a);
         x_pert.Add(2 * delta, p);
         mesh.SetNodes(x_pert);
         fes.Update();
         dJdx_dot_p_fd += functional.GetEnergy(a);
         dJdx_dot_p_fd /= (2 * delta);
         mesh.SetNodes(x_nodes); // remember to reset the mesh nodes
         fes.Update();

         REQUIRE(dJdx_dot_p == Approx(dJdx_dot_p_fd));
      }
   }
}

TEST_CASE("IECurlMagnitudeAggregateIntegratorDenominator::AssembleElementVector")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   NonLinearCoefficient nu;
   // LinearCoefficient nu;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         H1_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize state
         GridFunction a(&fes);
         FunctionCoefficient a_field([](const mfem::Vector &x){
            return x(0) * x(0) + x(1) * x(1);
         });
         a.ProjectCoefficient(a_field);

         double actual_max = a.Max();

         NonlinearForm functional(&fes);
         functional.AddDomainIntegrator(
            new mach::IECurlMagnitudeAggregateIntegratorDenominator(1.0, actual_max));

         // initialize the vector that dJdu multiplies
         GridFunction p(&fes);
         FunctionCoefficient pert(randState);
         p.ProjectCoefficient(pert);

         // evaluate dJdu and compute its product with v
         GridFunction dJdu(&fes);
         functional.Mult(a, dJdu);
         double dJdu_dot_p = InnerProduct(dJdu, p);

         // now compute the finite-difference approximation...
         GridFunction q_pert(a);
         q_pert.Add(-delta, p);
         double dJdu_dot_p_fd = -functional.GetEnergy(q_pert);
         q_pert.Add(2 * delta, p);
         dJdu_dot_p_fd += functional.GetEnergy(q_pert);
         dJdu_dot_p_fd /= (2 * delta);

         REQUIRE(dJdu_dot_p == Approx(dJdu_dot_p_fd));
      }
   }
}

TEST_CASE("IECurlMagnitudeAggregateIntegratorDenominatorMeshSens::AssembleRHSElementVect")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   NonLinearCoefficient nu;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         H1_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize state
         GridFunction a(&fes);
         // FunctionCoefficient pert(randState);
         FunctionCoefficient a_field([](const mfem::Vector &x){
            return x(0) * x(0) + x(1) * x(1);
         });
         a.ProjectCoefficient(a_field);

         double actual_max = a.Max();

         auto *integ = new mach::IECurlMagnitudeAggregateIntegratorDenominator(1.0, actual_max);
         NonlinearForm functional(&fes);
         functional.AddDomainIntegrator(integ);

         // extract mesh nodes and get their finite-element space
         auto &x_nodes = *mesh.GetNodes();
         auto &mesh_fes = *x_nodes.FESpace();

         // create v displacement field
         GridFunction v(&mesh_fes);
         VectorFunctionCoefficient v_pert(dim, randVectorState);
         v.ProjectCoefficient(v_pert);

         // initialize the vector that dJdx multiplies
         GridFunction p(&mesh_fes);
         p.ProjectCoefficient(v_pert);

         // evaluate dJdx and compute its product with p
         LinearForm dJdx(&mesh_fes);
         dJdx.AddDomainIntegrator(
            new mach::IECurlMagnitudeAggregateIntegratorDenominatorMeshSens(a, *integ));
         dJdx.Assemble();
         double dJdx_dot_p = dJdx * p;

         // now compute the finite-difference approximation...
         GridFunction x_pert(x_nodes);
         x_pert.Add(-delta, p);
         mesh.SetNodes(x_pert);
         fes.Update();
         double dJdx_dot_p_fd = -functional.GetEnergy(a);
         x_pert.Add(2 * delta, p);
         mesh.SetNodes(x_pert);
         fes.Update();
         dJdx_dot_p_fd += functional.GetEnergy(a);
         dJdx_dot_p_fd /= (2 * delta);
         mesh.SetNodes(x_nodes); // remember to reset the mesh nodes
         fes.Update();

         REQUIRE(dJdx_dot_p == Approx(dJdx_dot_p_fd));
      }
   }
}

// TEST_CASE("IEAggregateDemagIntegratorNumerator::GetElementEnergy")
// {

//    // Logic much the same as GetElementEnergy tests in test_electromag_integ
//    using namespace mfem;
//    using namespace electromag_data;

//    // generate a 8 element mesh, simple 2D domain, 0<=x<=1, 0<=y<=1
//    int num_edge = 2;
//    auto mesh = Mesh::MakeCartesian2D(num_edge, num_edge,
//                                      Element::TRIANGLE);
//    // auto mesh =  Mesh::MakeCartesian3D(num_edge, num_edge, 1,
//    //                      mfem::Element::TETRAHEDRON,
//    //                      1.0, 1.0, 1.0/num_edge, true);
     
//    mesh.EnsureNodes();
//    const auto dim = mesh.SpaceDimension();

//    ///TODO: Function Coefficient model Representing the B Field (flux density)
//    VectorFunctionCoefficient Bfield_model(2, [](const Vector& x, Vector &B)
//    {
//       B(0) = std::pow(2,0.5)*x(0);
//       B(1) = 0;
//    });
//    // FunctionCoefficient B_x_field_model(
//    //    [](const mfem::Vector &x)
//    //    {
//    //       return 0.5*x(1);
//    //    });
//    // FunctionCoefficient B_y_field_model(
//    //    [](const mfem::Vector &x)
//    //    {
//    //       return -0.5*x(0);
//    //    });

//    //Function Coefficient model Representing the Temperature Field
//    FunctionCoefficient Tfield_model(
//       [](const mfem::Vector &x)
//       {
//          // x will be the point in space
//          double T = 0;
//          for (int i = 0; i < x.Size(); ++i)
//          {
//             T = 1.0; //constant temperature throughout mesh
//             // T = 77*x(0)+273.15; // temperature linearly dependent in the x(0) direction
//             // T = 63*x(1)+273.15; // temperature linearly dependent in the x(1) direction
//             // T = 30*std::pow(x(0),2)+273.15; // temperature quadratically dependent in the x(0) direction
//             // T = 77*x(0)+63*x(1)+273.15; // temperature linearly dependent in both x(0) and x(1) directions
//             // T = 30*std::pow(x(0),2) + 3*std::pow(x(1),2) +273.15; // temperature quadratically dependent in both x(0) and x(1) directions

//          }
//          return T;
//       });

//    // Aggregation parameter rho
//    const double rho = 10.0; 

//    ///TODO: Restore p <= 4
//    for (int p = 1; p <= 1; ++p)
//    {
//       DYNAMIC_SECTION("...for degree p = " << p)
//       {

//          // Create the finite element collection and finite element space for the current order
//          H1_FECollection fec(p, dim);
//          // L2_FECollection fec(p, dim);
//          FiniteElementSpace fes(&mesh, &fec, 2); // either ordering does not seem to make a difference

//          // extract mesh nodes and get their finite-element space
//          auto &x_nodes = *mesh.GetNodes();
//          auto &mesh_fes = *x_nodes.FESpace();

//          // Create the temperature_field grid function by mapping the function coefficient to a grid function
//          GridFunction temperature_field(&fes);
//          temperature_field.ProjectCoefficient(Tfield_model);

//          ///TODO: Create the flux density field (B) grid function by mapping the function coefficient to a grid function
//          GridFunction flux_density_field(&fes);
//          flux_density_field.ProjectCoefficient(Bfield_model); 
//          // GridFunction B_x_field(&fes);
//          // B_x_field.ProjectCoefficient(B_x_field_model);
//          // GridFunction B_y_field(&fes);
//          // B_y_field.ProjectCoefficient(B_y_field_model);

//          // Handle the knee point demag flux density
//          LinearCoefficient B_knee;

//          /// Magnetization
//          NonLinearVectorCoefficient mag_coeff(2, 1.0);
         
//          // Define the functional integrator that will be used to compute the numerator
//          // auto *integ = new mach::IEAggregateDemagIntegratorNumerator(rho, *B_knee, *mag_coeff, &temperature_field, &flux_density_field/*, &B_x_field, &B_y_field/*/);
//          auto *integ = new mach::IEAggregateDemagIntegratorDenominator(rho, B_knee, mag_coeff, &temperature_field, &flux_density_field/*, &B_x_field, &B_y_field/*/);
//          NonlinearForm functional(&fes);
//          functional.AddDomainIntegrator(integ);   

//          // Compute the value of the numerator int(g * exp_rho_g)
//          mfem::Vector dummy_vec(fes.GetTrueVSize());
//          auto demag_numerator = functional.GetEnergy(dummy_vec);

//          ///TODO: Ultimately compare to expected and add assertion
//       }
//    }   
// }

///TODO: Add TEST_CASE("IEAggregateDemagIntegratorNumerator::AssembleRHSElementVect")

///TODO: Add TEST_CASE("IEAggregateDemagIntegratorDenominator::GetElementEnergy")

///TODO: Add TEST_CASE("IEAggregateDemagIntegratorDenominator::AssembleRHSElementVect")

TEST_CASE("DiffusionIntegratorMeshSens::AssembleRHSElementVect")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3;
   double delta = 1e-5;

   int num_edge = 2;
   auto smesh = Mesh::MakeCartesian3D(num_edge, num_edge, num_edge,
                                      Element::TETRAHEDRON,
                                      1.0, 1.0, 1.0, true);

   ParMesh mesh(MPI_COMM_WORLD, smesh); 
   mesh.EnsureNodes();

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         H1_FECollection fec(p, dim);
         ParFiniteElementSpace fes(&mesh, &fec);

         // extract mesh nodes and get their finite-element space
         auto &x_nodes = dynamic_cast<ParGridFunction&>(*mesh.GetNodes());
         auto &mesh_fes = *x_nodes.ParFESpace();

         // we use A for finite-difference approximation
         ParBilinearForm A(&fes);
         A.AddDomainIntegrator(new DiffusionIntegrator());

         // initialize state and adjoint; here we randomly perturb a constant state
         ParGridFunction state(&fes), adjoint(&fes);
         FunctionCoefficient pert(randState);
         state.ProjectCoefficient(pert);
         adjoint.ProjectCoefficient(pert);

         // build the nonlinear form for d(psi^T R)/dx 
         ParLinearForm dfdx(&mesh_fes);
         auto integ = new mach::DiffusionIntegratorMeshSens;
         integ->setState(state);
         integ->setAdjoint(adjoint);
         dfdx.AddDomainIntegrator(integ);

         // initialize the vector that we use to perturb the mesh nodes
         ParGridFunction v(&mesh_fes);
         VectorFunctionCoefficient v_pert(dim, randVectorState);
         v.ProjectCoefficient(v_pert);

         // evaluate df/dx and contract with v
         dfdx.Assemble();
         double dfdx_v = dfdx * v;

         // now compute the finite-difference approximation...
         Vector res(A.Size());
         ParGridFunction x_pert(x_nodes);
         x_pert.Add(delta, v);
         mesh.SetNodes(x_pert);
         A.Assemble();
         A.Finalize();
         A.Mult(state, res);
         double dfdx_v_fd = adjoint * res;
         x_pert.Add(-2*delta, v);
         mesh.SetNodes(x_pert);
         A.Update();
         A.Assemble();
         A.Finalize();
         A.Mult(state, res);
         dfdx_v_fd -= adjoint * res;
         dfdx_v_fd /= (2*delta);
         mesh.SetNodes(x_nodes); // remember to reset the mesh nodes

         // std::cout << "dfdx_v: " << dfdx_v << "\n";
         // std::cout << "dfdx_v_fd: " << dfdx_v_fd << "\n";
         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-10));
      }
   }
}

TEST_CASE("VectorFEWeakDivergenceIntegratorMeshSens::AssembleRHSElementVect")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3;
   double delta = 1e-5;

   int num_edge = 2;
   auto smesh = Mesh::MakeCartesian3D(num_edge, num_edge, num_edge,
                                      Element::TETRAHEDRON,
                                      1.0, 1.0, 1.0, true);

   ParMesh mesh(MPI_COMM_WORLD, smesh); 
   mesh.EnsureNodes();

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         H1_FECollection h1_fec(p, dim);
         ParFiniteElementSpace h1_fes(&mesh, &h1_fec);

         ND_FECollection nd_fec(p, dim);
         ParFiniteElementSpace nd_fes(&mesh, &nd_fec);

         // extract mesh nodes and get their finite-element space
         auto &x_nodes = dynamic_cast<ParGridFunction&>(*mesh.GetNodes());
         auto &mesh_fes = *x_nodes.ParFESpace();

         // we use A for finite-difference approximation
         ParMixedBilinearForm W(&nd_fes, &h1_fes);
         W.AddDomainIntegrator(new VectorFEWeakDivergenceIntegrator);

         // initialize state and adjoint; here we randomly perturb a constant state
         ParGridFunction state(&nd_fes), adjoint(&h1_fes);
         FunctionCoefficient pert(randState);
         VectorFunctionCoefficient v_pert(3, randVectorState);
         state.ProjectCoefficient(v_pert);
         adjoint.ProjectCoefficient(pert);

         // build the nonlinear form for d(psi^T R)/dx 
         ParLinearForm dfdx(&mesh_fes);
         auto integ = new mach::VectorFEWeakDivergenceIntegratorMeshSens;
         integ->setState(state);
         integ->setAdjoint(adjoint);
         dfdx.AddDomainIntegrator(integ);

         // initialize the vector that we use to perturb the mesh nodes
         ParGridFunction v(&mesh_fes);
         v.ProjectCoefficient(v_pert);

         // evaluate df/dx and contract with v
         dfdx.Assemble();
         double dfdx_v = dfdx * v;

         // now compute the finite-difference approximation...
         Vector res(adjoint.Size());
         ParGridFunction x_pert(x_nodes);
         x_pert.Add(delta, v);
         mesh.SetNodes(x_pert);
         W.Assemble();
         W.Finalize();
         W.Mult(state, res);
         double dfdx_v_fd = adjoint * res;
         x_pert.Add(-2*delta, v);
         mesh.SetNodes(x_pert);
         W.Update();
         W.Assemble();
         W.Finalize();
         W.Mult(state, res);
         dfdx_v_fd -= adjoint * res;
         dfdx_v_fd /= (2*delta);
         mesh.SetNodes(x_nodes); // remember to reset the mesh nodes

         // std::cout << "dfdx_v: " << dfdx_v << "\n";
         // std::cout << "dfdx_v_fd: " << dfdx_v_fd << "\n";
         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-10));
      }
   }
}

/** Not differentiated, not needed since we use linear form version of MagneticLoad
TEST_CASE("VectorFECurlIntegratorMeshSens::AssembleRHSElementVect - (u, curl v)")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3;
   double delta = 1e-5;

   int num_edge = 2;
   auto smesh = Mesh::MakeCartesian3D(num_edge, num_edge, num_edge,
                                      Element::TETRAHEDRON,
                                      1.0, 1.0, 1.0, true);

   ParMesh mesh(MPI_COMM_WORLD, smesh); 
   mesh.EnsureNodes();

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         RT_FECollection rt_fec(p, dim);
         ParFiniteElementSpace rt_fes(&mesh, &rt_fec);

         ND_FECollection nd_fec(p, dim);
         ParFiniteElementSpace nd_fes(&mesh, &nd_fec);

         // extract mesh nodes and get their finite-element space
         auto &x_nodes = dynamic_cast<ParGridFunction&>(*mesh.GetNodes());
         auto &mesh_fes = *x_nodes.ParFESpace();

         // we use C for finite-difference approximation
         ParMixedBilinearForm C(&rt_fes, &nd_fes);
         C.AddDomainIntegrator(new VectorFECurlIntegrator);

         // initialize state and adjoint; here we randomly perturb a constant state
         ParGridFunction state(&rt_fes), adjoint(&nd_fes);
         VectorFunctionCoefficient pert(3, randVectorState);
         state.ProjectCoefficient(pert);
         adjoint.ProjectCoefficient(pert);

         // build the nonlinear form for d(psi^T R)/dx 
         ParLinearForm dfdx(&mesh_fes);
         auto integ = new mach::VectorFECurlIntegratorMeshSens;
         integ->setState(state);
         integ->setAdjoint(adjoint);
         dfdx.AddDomainIntegrator(integ);

         // initialize the vector that we use to perturb the mesh nodes
         ParGridFunction v(&mesh_fes);
         v.ProjectCoefficient(pert);

         // evaluate df/dx and contract with v
         dfdx.Assemble();
         double dfdx_v = dfdx * v;

         // now compute the finite-difference approximation...
         Vector res(adjoint.Size());
         ParGridFunction x_pert(x_nodes);
         x_pert.Add(delta, v);
         mesh.SetNodes(x_pert);
         C.Assemble();
         C.Finalize();
         C.Mult(state, res);
         double dfdx_v_fd = adjoint * res;
         x_pert.Add(-2*delta, v);
         mesh.SetNodes(x_pert);
         C.Update();
         C.Assemble();
         C.Finalize();
         C.Mult(state, res);
         dfdx_v_fd -= adjoint * res;
         dfdx_v_fd /= (2*delta);
         mesh.SetNodes(x_nodes); // remember to reset the mesh nodes

         // std::cout << "dfdx_v: " << dfdx_v << "\n";
         // std::cout << "dfdx_v_fd: " << dfdx_v_fd << "\n";
         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-10));
      }
   }
}

TEST_CASE("VectorFECurlIntegratorMeshSens::AssembleRHSElementVect - (curl u, v)")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3;
   double delta = 1e-5;

   int num_edge = 2;
   auto smesh = Mesh::MakeCartesian3D(num_edge, num_edge, num_edge,
                                      Element::TETRAHEDRON,
                                      1.0, 1.0, 1.0, true);

   ParMesh mesh(MPI_COMM_WORLD, smesh); 
   mesh.EnsureNodes();

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         RT_FECollection rt_fec(p, dim);
         ParFiniteElementSpace rt_fes(&mesh, &rt_fec);

         ND_FECollection nd_fec(p, dim);
         ParFiniteElementSpace nd_fes(&mesh, &nd_fec);

         // extract mesh nodes and get their finite-element space
         auto &x_nodes = dynamic_cast<ParGridFunction&>(*mesh.GetNodes());
         auto &mesh_fes = *x_nodes.ParFESpace();

         // we use C for finite-difference approximation
         ParMixedBilinearForm C(&nd_fes, &rt_fes);
         C.AddDomainIntegrator(new VectorFECurlIntegrator);

         // initialize state and adjoint; here we randomly perturb a constant state
         ParGridFunction state(&nd_fes), adjoint(&rt_fes);
         VectorFunctionCoefficient pert(3, randVectorState);
         state.ProjectCoefficient(pert);
         adjoint.ProjectCoefficient(pert);

         // build the nonlinear form for d(psi^T R)/dx 
         ParLinearForm dfdx(&mesh_fes);
         auto integ = new mach::VectorFECurlIntegratorMeshSens;
         integ->setState(state);
         integ->setAdjoint(adjoint);
         dfdx.AddDomainIntegrator(integ);

         // initialize the vector that we use to perturb the mesh nodes
         ParGridFunction v(&mesh_fes);
         v.ProjectCoefficient(pert);

         // evaluate df/dx and contract with v
         dfdx.Assemble();
         double dfdx_v = dfdx * v;

         // now compute the finite-difference approximation...
         Vector res(adjoint.Size());
         ParGridFunction x_pert(x_nodes);
         x_pert.Add(delta, v);
         mesh.SetNodes(x_pert);
         C.Assemble();
         C.Finalize();
         C.Mult(state, res);
         double dfdx_v_fd = adjoint * res;
         x_pert.Add(-2*delta, v);
         mesh.SetNodes(x_pert);
         C.Update();
         C.Assemble();
         C.Finalize();
         C.Mult(state, res);
         dfdx_v_fd -= adjoint * res;
         dfdx_v_fd /= (2*delta);
         mesh.SetNodes(x_nodes); // remember to reset the mesh nodes

         std::cout << "dfdx_v: " << dfdx_v << "\n";
         std::cout << "dfdx_v_fd: " << dfdx_v_fd << "\n";
         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-10));
      }
   }
}
*/

TEST_CASE("VectorFEMassIntegratorMeshSens::AssembleRHSElementVect")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3;
   double delta = 1e-5;

   int num_edge = 2;
   auto smesh = Mesh::MakeCartesian3D(num_edge, num_edge, num_edge,
                                      Element::TETRAHEDRON,
                                      1.0, 1.0, 1.0, true);

   ParMesh mesh(MPI_COMM_WORLD, smesh); 
   mesh.EnsureNodes();

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         ND_FECollection fec(p, dim);
         ParFiniteElementSpace fes(&mesh, &fec);

         // extract mesh nodes and get their finite-element space
         auto &x_nodes = dynamic_cast<ParGridFunction&>(*mesh.GetNodes());
         auto &mesh_fes = *x_nodes.ParFESpace();

         // we use A for finite-difference approximation
         ParBilinearForm A(&fes);
         A.AddDomainIntegrator(new VectorFEMassIntegrator);

         // initialize state and adjoint; here we randomly perturb a constant state
         ParGridFunction state(&fes), adjoint(&fes);
         VectorFunctionCoefficient pert(3, randVectorState);
         state.ProjectCoefficient(pert);
         adjoint.ProjectCoefficient(pert);

         // build the nonlinear form for d(psi^T R)/dx 
         ParLinearForm dfdx(&mesh_fes);
         auto integ = new mach::VectorFEMassIntegratorMeshSens;
         integ->setState(state);
         integ->setAdjoint(adjoint);
         dfdx.AddDomainIntegrator(integ);

         // initialize the vector that we use to perturb the mesh nodes
         ParGridFunction v(&mesh_fes);
         v.ProjectCoefficient(pert);

         // evaluate df/dx and contract with v
         dfdx.Assemble();
         double dfdx_v = dfdx * v;

         // now compute the finite-difference approximation...
         Vector res(A.Size());
         ParGridFunction x_pert(x_nodes);
         x_pert.Add(delta, v);
         mesh.SetNodes(x_pert);
         A.Assemble();
         A.Finalize();
         A.Mult(state, res);
         double dfdx_v_fd = adjoint * res;
         x_pert.Add(-2*delta, v);
         mesh.SetNodes(x_pert);
         A.Update();
         A.Assemble();
         A.Finalize();
         A.Mult(state, res);
         dfdx_v_fd -= adjoint * res;
         dfdx_v_fd /= (2*delta);
         mesh.SetNodes(x_nodes); // remember to reset the mesh nodes

         // std::cout << "dfdx_v: " << dfdx_v << "\n";
         // std::cout << "dfdx_v_fd: " << dfdx_v_fd << "\n";
         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-10));
      }
   }
}

TEST_CASE("VectorFEDomainLFIntegratorMeshSens::AssembleRHSElementVect")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3;
   double delta = 1e-5;

   int num_edge = 2;
   auto smesh = Mesh::MakeCartesian3D(num_edge, num_edge, num_edge,
                                      Element::TETRAHEDRON,
                                      1.0, 1.0, 1.0, true);

   ParMesh mesh(MPI_COMM_WORLD, smesh); 
   mesh.EnsureNodes();

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         ND_FECollection fec(p, dim);
         ParFiniteElementSpace fes(&mesh, &fec);

         // extract mesh nodes and get their finite-element space
         auto &x_nodes = dynamic_cast<ParGridFunction&>(*mesh.GetNodes());
         auto &mesh_fes = *x_nodes.ParFESpace();

         double vec_buffer[] = {1.2, 3.4, 2.8};
         Vector vec(vec_buffer, 3);
         VectorConstantCoefficient vec_coeff(vec);
         // we use b for finite-difference approximation
         ParLinearForm b(&fes);
         b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(vec_coeff));

         // initialize state and adjoint; here we randomly perturb a constant state
         ParGridFunction adjoint(&fes);
         VectorFunctionCoefficient pert(3, randVectorState);
         adjoint.ProjectCoefficient(pert);

         // build the nonlinear form for d(psi^T R)/dx 
         ParLinearForm dfdx(&mesh_fes);
         auto integ = new mach::VectorFEDomainLFIntegratorMeshSens(vec_coeff);
         integ->setAdjoint(adjoint);
         dfdx.AddDomainIntegrator(integ);

         // initialize the vector that we use to perturb the mesh nodes
         ParGridFunction v(&mesh_fes);
         v.ProjectCoefficient(pert);

         // evaluate df/dx and contract with v
         dfdx.Assemble();
         double dfdx_v = dfdx * v;

         // now compute the finite-difference approximation...
         ParGridFunction x_pert(x_nodes);
         x_pert.Add(delta, v);
         mesh.SetNodes(x_pert);
         b.Assemble();
         double dfdx_v_fd = adjoint * b;
         x_pert.Add(-2*delta, v);
         mesh.SetNodes(x_pert);
         b.Assemble();
         dfdx_v_fd -= adjoint * b;
         dfdx_v_fd /= (2*delta);
         mesh.SetNodes(x_nodes); // remember to reset the mesh nodes

         // std::cout << "dfdx_v: " << dfdx_v << "\n";
         // std::cout << "dfdx_v_fd: " << dfdx_v_fd << "\n";
         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-10));
      }
   }
}

TEST_CASE("VectorFEDomainLFCurlIntegratorMeshSens::AssembleRHSElementVect")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3;
   double delta = 1e-5;

   int num_edge = 2;
   auto smesh = Mesh::MakeCartesian3D(num_edge, num_edge, num_edge,
                                      Element::TETRAHEDRON,
                                      1.0, 1.0, 1.0, true);

   ParMesh mesh(MPI_COMM_WORLD, smesh); 
   mesh.EnsureNodes();

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         ND_FECollection fec(p, dim);
         ParFiniteElementSpace fes(&mesh, &fec);

         // extract mesh nodes and get their finite-element space
         auto &x_nodes = dynamic_cast<ParGridFunction&>(*mesh.GetNodes());
         auto &mesh_fes = *x_nodes.ParFESpace();

         double vec_buffer[] = {1.2, 3.4, 2.8};
         Vector vec(vec_buffer, 3);
         VectorConstantCoefficient vec_coeff(vec);
         // we use b for finite-difference approximation
         ParLinearForm b(&fes);
         auto *primal = new mach::VectorFEDomainLFCurlIntegrator(vec_coeff);
         b.AddDomainIntegrator(primal);

         // initialize state and adjoint; here we randomly perturb a constant state
         ParGridFunction adjoint(&fes);
         VectorFunctionCoefficient pert(3, randVectorState);
         adjoint.ProjectCoefficient(pert);

         // build the nonlinear form for d(psi^T R)/dx 
         ParLinearForm dfdx(&mesh_fes);
         auto integ = new mach::VectorFEDomainLFCurlIntegratorMeshSens(*primal);
         integ->setAdjoint(adjoint);
         dfdx.AddDomainIntegrator(integ);

         // initialize the vector that we use to perturb the mesh nodes
         ParGridFunction v(&mesh_fes);
         v.ProjectCoefficient(pert);

         // evaluate df/dx and contract with v
         dfdx.Assemble();
         double dfdx_v = dfdx * v;

         // now compute the finite-difference approximation...
         ParGridFunction x_pert(x_nodes);
         x_pert.Add(delta, v);
         mesh.SetNodes(x_pert);
         b.Assemble();
         double dfdx_v_fd = adjoint * b;
         x_pert.Add(-2*delta, v);
         mesh.SetNodes(x_pert);
         b.Assemble();
         dfdx_v_fd -= adjoint * b;
         dfdx_v_fd /= (2*delta);
         mesh.SetNodes(x_nodes); // remember to reset the mesh nodes

         // std::cout << "dfdx_v: " << dfdx_v << "\n";
         // std::cout << "dfdx_v_fd: " << dfdx_v_fd << "\n";
         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-10));
      }
   }
}

TEST_CASE("DomainLFIntegratorMeshSens::AssembleRHSElementVect")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   mfem::FunctionCoefficient model(
      [](const mfem::Vector &x)
      {
         double q = 0;
         for (int i = 0; i < x.Size(); ++i)
         {
            q += pow(x(i), 2);
            // q += x(i);
         }
         return q;

         // return 1.0;
      },
      [](const mfem::Vector &x, const double q_bar, mfem::Vector &x_bar)
      {
         for (int i = 0; i < x.Size(); ++i)
         {
            x_bar(i) += q_bar * 2 * x(i);
            // x_bar(i) += q_bar;
         }

         // x_bar = 0.0;
      });

   // mfem::ConstantCoefficient model(2.0);

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         H1_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize adjoint; here we randomly perturb a constant state
         GridFunction adjoint(&fes);
         FunctionCoefficient pert(randState);
         adjoint.ProjectCoefficient(pert);

         LinearForm res(&fes);
         auto *integ = new mach::DomainLFIntegrator(model, -1.0);
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
            new mach::DomainLFIntegratorMeshRevSens(adjoint, *integ));
         dfdx.Assemble();
         double dfdx_v = dfdx * v;

         // now compute the finite-difference approximation...
         GridFunction x_pert(x_nodes);
         x_pert.Add(delta, v);
         mesh.SetNodes(x_pert);
         fes.Update();
         res.Update();
         res.Assemble();
         double dfdx_v_fd = adjoint * res;
         x_pert.Add(-2 * delta, v);
         mesh.SetNodes(x_pert);
         fes.Update();
         res.Update();
         res.Assemble();
         dfdx_v_fd -= adjoint * res;
         dfdx_v_fd /= (2 * delta);
         mesh.SetNodes(x_nodes); // remember to reset the mesh nodes
         fes.Update();

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-8));
      }
   }
}

/** Not yet differentiated, class only needed if magnets are on the boundary
    and not normal to boundary
TEST_CASE("VectorFEBoundaryTangentLFIntegratorMeshSens::AssembleRHSElementVect")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3;
   double delta = 1e-5;

   int num_edge = 2;
   auto smesh = Mesh::MakeCartesian3D(num_edge, num_edge, num_edge,
                                      Element::TETRAHEDRON,
                                      1.0, 1.0, 1.0, true);

   ParMesh mesh(MPI_COMM_WORLD, smesh); 
   mesh.EnsureNodes();

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         ND_FECollection fec(p, dim);
         ParFiniteElementSpace fes(&mesh, &fec);

         // extract mesh nodes and get their finite-element space
         auto &x_nodes = dynamic_cast<ParGridFunction&>(*mesh.GetNodes());
         auto &mesh_fes = *x_nodes.ParFESpace();

         double vec_buffer[] = {1.2, 3.4, 2.8};
         Vector vec(vec_buffer, 3);
         VectorConstantCoefficient vec_coeff(vec);
         // we use b for finite-difference approximation
         ParLinearForm b(&fes);
         auto *primal = new mach::VectorFEBoundaryTangentLFIntegrator(vec_coeff);
         b.AddBoundaryIntegrator(primal);

         // initialize state and adjoint; here we randomly perturb a constant state
         ParGridFunction adjoint(&fes);
         VectorFunctionCoefficient pert(3, randVectorState);
         adjoint.ProjectCoefficient(pert);

         // build the nonlinear form for d(psi^T R)/dx 
         ParLinearForm dfdx(&mesh_fes);
         auto integ = new mach::VectorFEBoundaryTangentLFIntegratorMeshSens(*primal);
         integ->setAdjoint(adjoint);
         dfdx.AddBdrFaceIntegrator(integ);

         // initialize the vector that we use to perturb the mesh nodes
         ParGridFunction v(&mesh_fes);
         v.ProjectCoefficient(pert);

         // evaluate df/dx and contract with v
         dfdx.Assemble();
         double dfdx_v = dfdx * v;

         // now compute the finite-difference approximation...
         ParGridFunction x_pert(x_nodes);
         x_pert.Add(delta, v);
         mesh.SetNodes(x_pert);
         b.Assemble();
         double dfdx_v_fd = adjoint * b;
         x_pert.Add(-2*delta, v);
         mesh.SetNodes(x_pert);
         b.Assemble();
         dfdx_v_fd -= adjoint * b;
         dfdx_v_fd /= (2*delta);
         mesh.SetNodes(x_nodes); // remember to reset the mesh nodes

         std::cout << "dfdx_v: " << dfdx_v << "\n";
         std::cout << "dfdx_v_fd: " << dfdx_v_fd << "\n";
         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-10));
      }
   }
}
*/

// TEST_CASE("TestLFMeshSensIntegrator::AssembleRHSElementVect",
//           "[TestLFMeshSensIntegrator]")
// {
//    using namespace mfem;
//    using namespace electromag_data;

//    const int dim = 3; // templating is hard here because mesh constructors
//    double delta = 1e-5;

//    // generate a 2 element mesh
//    int num_edge = 2;
//    std::unique_ptr<Mesh> mesh(new Mesh(2, num_edge, num_edge,
//                                        Element::TETRAHEDRON,
//                                        true /* gen. edges */, 1.0, 1.0, 1.0, 
//                                        true));
//    mesh->EnsureNodes();
//    for (int p = 1; p <= 4; ++p)
//    {
//       DYNAMIC_SECTION("...for degree p = " << p)
//       {
//          // get the finite-element space for the state
//          std::unique_ptr<FiniteElementCollection> fec(
//              new ND_FECollection(p, dim));
//          std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
//              mesh.get(), fec.get()));

//          // we use res for finite-difference approximation
//          GridFunction A(fes.get());
//          VectorFunctionCoefficient pert(dim, electromag_data::randVectorState);
//          A.ProjectCoefficient(pert);

//          // extract mesh nodes and get their finite-element space
//          GridFunction *x_nodes = mesh->GetNodes();
//          FiniteElementSpace *mesh_fes = x_nodes->FESpace();

//          // ConstantCoefficient Q(1.0);
//          // FunctionCoefficient Q(func, funcRevDiff);
//          mach::SteinmetzCoefficient Q(1, 2, 4, 0.5, 0.6, A);

//          // initialize state; here we randomly perturb a constant state
//          GridFunction q(fes.get());
//          // VectorFunctionCoefficient pert(3, randState);
//          q.ProjectCoefficient(pert);

//          // initialize the vector that dJdx multiplies
//          GridFunction v(mesh_fes);
//          VectorFunctionCoefficient v_rand(dim, randVectorState);
//          v.ProjectCoefficient(v_rand);

//          // evaluate dJdx and compute its product with v
//          // GridFunction dJdx(*x_nodes);
//          LinearForm dJdx(mesh_fes);
//          dJdx.AddDomainIntegrator(
//             new mach::TestLFMeshSensIntegrator(Q));
//          dJdx.Assemble();
//          double dJdx_dot_v = dJdx * v;

//          // now compute the finite-difference approximation...
//          NonlinearForm functional(fes.get());
//          functional.AddDomainIntegrator(
//             new mach::TestLFIntegrator(Q));

//          double delta = 1e-5;
//          GridFunction x_pert(*x_nodes);
//          x_pert.Add(-delta, v);
//          mesh->SetNodes(x_pert);
//          fes->Update();
//          double dJdx_dot_v_fd = -functional.GetEnergy(q);
//          x_pert.Add(2 * delta, v);
//          mesh->SetNodes(x_pert);
//          fes->Update();
//          dJdx_dot_v_fd += functional.GetEnergy(q);
//          dJdx_dot_v_fd /= (2 * delta);
//          mesh->SetNodes(*x_nodes); // remember to reset the mesh nodes
//          fes->Update();

//          std::cout << dJdx_dot_v << "\n";
//          std::cout << dJdx_dot_v_fd << "\n";

//          REQUIRE(dJdx_dot_v == Approx(dJdx_dot_v_fd));
//       }
//    }
// }

// TEST_CASE("DomainResIntegrator::AssembleElementVector",
//           "[DomainResIntegrator]")
// {
//    using namespace mfem;
//    using namespace euler_data;

//    const int dim = 3; // templating is hard here because mesh constructors
//    double delta = 1e-5;

//    // generate a 8 element mesh
//    int num_edge = 2;
//    std::unique_ptr<Mesh> mesh = electromag_data::getMesh();
//                               //(new Mesh(num_edge, num_edge, num_edge, Element::TETRAHEDRON,
//                               //        true /* gen. edges */, 1.0, 1.0, 1.0, true));
//    mesh->EnsureNodes();
//    for (int p = 1; p <= 4; ++p)
//    {
//       DYNAMIC_SECTION("...for degree p = " << p)
//       {
//          // get the finite-element space for the state and adjoint
//          std::unique_ptr<FiniteElementCollection> fec(
//              new H1_FECollection(p, dim));
//          std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
//              mesh.get(), fec.get()));

//          // we use res for finite-difference approximation
//          std::unique_ptr<Coefficient> q1(new ConstantCoefficient(1));
//          std::unique_ptr<Coefficient> q2(new ConstantCoefficient(2));
//          std::unique_ptr<mach::MeshDependentCoefficient> Q;
//          Q.reset(new mach::MeshDependentCoefficient());
//          Q->addCoefficient(1, move(q1)); 
//          Q->addCoefficient(2, move(q2));
//          LinearForm res(fes.get());
//          res.AddDomainIntegrator(
//             new DomainLFIntegrator(*Q));

//          // initialize state and adjoint; here we randomly perturb a constant state
//          GridFunction state(fes.get()), adjoint(fes.get());
//          FunctionCoefficient pert(randState);
//          state.ProjectCoefficient(pert);
//          adjoint.ProjectCoefficient(pert);

//          // extract mesh nodes and get their finite-element space
//          GridFunction *x_nodes = mesh->GetNodes();
//          FiniteElementSpace *mesh_fes = x_nodes->FESpace();

//          // build the nonlinear form for d(psi^T R)/dx 
//          NonlinearForm dfdx_form(mesh_fes);
//          dfdx_form.AddDomainIntegrator(
//             new mach::DomainResIntegrator(*Q, &adjoint));

//          // initialize the vector that we use to perturb the mesh nodes
//          GridFunction v(mesh_fes);
//          VectorFunctionCoefficient v_rand(dim, randVectorState);
//          v.ProjectCoefficient(v_rand);

//          // evaluate df/dx and contract with v
//          GridFunction dfdx(*x_nodes);
//          dfdx_form.Mult(*x_nodes, dfdx);
//          double dfdx_v = dfdx * v;

//          // now compute the finite-difference approximation...
//          GridFunction x_pert(*x_nodes);
//          GridFunction r(fes.get());
//          x_pert.Add(delta, v);
//          mesh->SetNodes(x_pert);
//          res.Assemble();
//          double dfdx_v_fd = adjoint * res;
//          x_pert.Add(-2 * delta, v);
//          mesh->SetNodes(x_pert);
//          res.Assemble();
//          dfdx_v_fd -= adjoint * res;
//          dfdx_v_fd /= (2 * delta);
//          mesh->SetNodes(*x_nodes); // remember to reset the mesh nodes

//          REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-10));
//       }
//    }
// }

// TEST_CASE("MassResIntegrator::AssembleElementVector",
//           "[MassResIntegrator]")
// {
//    using namespace mfem;
//    using namespace euler_data;

//    const int dim = 3; // templating is hard here because mesh constructors
//    double delta = 1e-5;

//    // generate a 8 element mesh
//    int num_edge = 2;
//    std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, num_edge, Element::TETRAHEDRON,
//                                        true /* gen. edges */, 1.0, 1.0, 1.0, true));
//    mesh->EnsureNodes();
//    for (int p = 1; p <= 4; ++p)
//    {
//       DYNAMIC_SECTION("...for degree p = " << p)
//       {
//          // get the finite-element space for the state and adjoint
//          std::unique_ptr<FiniteElementCollection> fec(
//              new H1_FECollection(p, dim));
//          std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
//              mesh.get(), fec.get()));

//          // we use res for finite-difference approximation
//          std::unique_ptr<Coefficient> Q(new ConstantCoefficient(1));
//          BilinearForm res(fes.get());
//          res.AddDomainIntegrator(
//             new MassIntegrator(*Q));

//          // initialize state and adjoint; here we randomly perturb a constant state
//          GridFunction state(fes.get()), adjoint(fes.get());
//          FunctionCoefficient pert(randState);
//          state.ProjectCoefficient(pert);
//          adjoint.ProjectCoefficient(pert);

//          // extract mesh nodes and get their finite-element space
//          GridFunction *x_nodes = mesh->GetNodes();
//          FiniteElementSpace *mesh_fes = x_nodes->FESpace();

//          // build the nonlinear form for d(psi^T R)/dx 
//          NonlinearForm dfdx_form(mesh_fes);
//          dfdx_form.AddDomainIntegrator(
//             new mach::MassResIntegrator(*Q,
//                &state, &adjoint));

//          // initialize the vector that we use to perturb the mesh nodes
//          GridFunction v(mesh_fes);
//          VectorFunctionCoefficient v_rand(dim, randVectorState);
//          v.ProjectCoefficient(v_rand);

//          // evaluate df/dx and contract with v
//          GridFunction dfdx(*x_nodes);
//          dfdx_form.Mult(*x_nodes, dfdx);
//          double dfdx_v = dfdx * v;

//          // now compute the finite-difference approximation...
//          Vector resid(res.Size());
//          GridFunction x_pert(*x_nodes);
//          GridFunction r(fes.get());
//          x_pert.Add(delta, v);
//          mesh->SetNodes(x_pert);
//          res.Assemble();
//          res.Mult(state, resid);
//          double dfdx_v_fd = adjoint * resid;
//          x_pert.Add(-2*delta, v);
//          mesh->SetNodes(x_pert);
//          res.Update();
//          res.Assemble();
//          res.Mult(state, resid);
//          dfdx_v_fd -= adjoint * resid;
//          dfdx_v_fd /= (2*delta);
//          mesh->SetNodes(*x_nodes); // remember to reset the mesh nodes

//          REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-10));
//       }
//    }
// }

// TEST_CASE("DiffusionResIntegrator::AssembleElementVector",
//           "[DiffusionResIntegrator]")
// {
//    using namespace mfem;
//    using namespace euler_data;

//    const int dim = 3; // templating is hard here because mesh constructors
//    double delta = 1e-5;

//    // generate a 8 element mesh
//    int num_edge = 2;
//    std::unique_ptr<Mesh> mesh = electromag_data::getMesh();
//                                  //(new Mesh(num_edge, num_edge, num_edge, Element::TETRAHEDRON,
//                                  //true /* gen. edges */, 1.0, 1.0, 1.0, true));
//    mesh->EnsureNodes();
//    for (int p = 1; p <= 4; ++p)
//    {
//       DYNAMIC_SECTION("...for degree p = " << p)
//       {
//          // get the finite-element space for the state and adjoint
//          std::unique_ptr<FiniteElementCollection> fec(
//              new H1_FECollection(p, dim));
//          std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
//              mesh.get(), fec.get()));

//          // we use res for finite-difference approximation
//          std::unique_ptr<Coefficient> q1(new ConstantCoefficient(1));
//          std::unique_ptr<Coefficient> q2(new ConstantCoefficient(2));
//          std::unique_ptr<mach::MeshDependentCoefficient> Q;
//          Q.reset(new mach::MeshDependentCoefficient());
//          Q->addCoefficient(1, move(q1)); 
//          Q->addCoefficient(2, move(q2));
//          BilinearForm res(fes.get());
//          res.AddDomainIntegrator(
//             new DiffusionIntegrator(*Q));

//          // initialize state and adjoint; here we randomly perturb a constant state
//          GridFunction state(fes.get()), adjoint(fes.get());
//          FunctionCoefficient pert(randState);
//          state.ProjectCoefficient(pert);
//          adjoint.ProjectCoefficient(pert);

//          // extract mesh nodes and get their finite-element space
//          GridFunction *x_nodes = mesh->GetNodes();
//          FiniteElementSpace *mesh_fes = x_nodes->FESpace();

//          // build the nonlinear form for d(psi^T R)/dx 
//          NonlinearForm dfdx_form(mesh_fes);
//          dfdx_form.AddDomainIntegrator(
//             new mach::DiffusionResIntegrator(*Q,
//                &state, &adjoint));

//          // initialize the vector that we use to perturb the mesh nodes
//          GridFunction v(mesh_fes);
//          VectorFunctionCoefficient v_rand(dim, randVectorState);
//          v.ProjectCoefficient(v_rand);

//          // evaluate df/dx and contract with v
//          GridFunction dfdx(*x_nodes);
//          dfdx_form.Mult(*x_nodes, dfdx);
//          double dfdx_v = dfdx * v;

//          // now compute the finite-difference approximation...
//          Vector resid(res.Size());
//          GridFunction x_pert(*x_nodes);
//          GridFunction r(fes.get());
//          x_pert.Add(delta, v);
//          mesh->SetNodes(x_pert);
//          res.Assemble();
//          res.Mult(state, resid);
//          double dfdx_v_fd = adjoint * resid;
//          x_pert.Add(-2*delta, v);
//          mesh->SetNodes(x_pert);
//          res.Update();
//          res.Assemble();
//          res.Mult(state, resid);
//          dfdx_v_fd -= adjoint * resid;
//          dfdx_v_fd /= (2*delta);
//          mesh->SetNodes(*x_nodes); // remember to reset the mesh nodes

//          REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-10));
//       }
//    }
// }

// TEST_CASE("DiffusionResIntegrator::AssembleRHSElementVect",
//                        "[DiffusionResIntegrator]")
// {
//    using namespace mfem;
//    using namespace euler_data;

//    const int dim = 3; // templating is hard here because mesh constructors
//    double delta = 1e-5;

//    // generate a 8 element mesh
//    int num_edge = 2;
//    std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, num_edge, Element::TETRAHEDRON,
//                                        true /* gen. edges */, 1.0, 1.0, 1.0, true));
//    mesh->EnsureNodes();
//    for (int p = 1; p <= 4; ++p)
//    {
//       DYNAMIC_SECTION("...for degree p = " << p)
//       {
//          // get the finite-element space for the state and adjoint
//          std::unique_ptr<FiniteElementCollection> fec(
//              new H1_FECollection(p, dim));
//          std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
//              mesh.get(), fec.get()));

//          // we use res for finite-difference approximation
//          std::unique_ptr<Coefficient> Q(new ConstantCoefficient(1));
//          BilinearForm res(fes.get());
//          res.AddDomainIntegrator(
//             new DiffusionIntegrator(*Q));

//          // initialize state and adjoint; here we randomly perturb a constant state
//          GridFunction state(fes.get()), adjoint(fes.get());
//          FunctionCoefficient pert(randState);
//          state.ProjectCoefficient(pert);
//          adjoint.ProjectCoefficient(pert);

//          // extract mesh nodes and get their finite-element space
//          GridFunction *x_nodes = mesh->GetNodes();
//          FiniteElementSpace *mesh_fes = x_nodes->FESpace();

//          // build the nonlinear form for d(psi^T R)/dx 
//          LinearForm dfdx(mesh_fes);
//          dfdx.AddDomainIntegrator(
//             new mach::DiffusionResIntegrator(*Q,
//                &state, &adjoint));

//          // initialize the vector that we use to perturb the mesh nodes
//          GridFunction v(mesh_fes);
//          VectorFunctionCoefficient v_rand(dim, randVectorState);
//          v.ProjectCoefficient(v_rand);

//          // evaluate df/dx and contract with v
//          dfdx.Assemble();
//          double dfdx_v = dfdx * v;

//          // now compute the finite-difference approximation...
//          Vector resid(res.Size());
//          GridFunction x_pert(*x_nodes);
//          GridFunction r(fes.get());
//          x_pert.Add(delta, v);
//          mesh->SetNodes(x_pert);
//          res.Assemble();
//          res.Mult(state, resid);
//          double dfdx_v_fd = adjoint * resid;
//          x_pert.Add(-2*delta, v);
//          mesh->SetNodes(x_pert);
//          res.Update();
//          res.Assemble();
//          res.Mult(state, resid);
//          dfdx_v_fd -= adjoint * resid;
//          dfdx_v_fd /= (2*delta);
//          mesh->SetNodes(*x_nodes); // remember to reset the mesh nodes

//          REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-10));
//       }
//    }
// }

// TEST_CASE("BoundaryNormalResIntegrator::AssembleFaceVector",
//           "[BoundaryNormalResIntegrator]")
// {
//    using namespace mfem;
//    using namespace euler_data;

//    const int dim = 3; // templating is hard here because mesh constructors
//    double delta = 1e-5;

//    // generate a 8 element mesh
//    int num_edge = 2;
//    std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, num_edge, Element::TETRAHEDRON,
//                                        true /* gen. edges */, 1.0, 1.0, 1.0, true));
//    mesh->EnsureNodes();
//    for (int p = 1; p <= 4; ++p)
//    {
//       DYNAMIC_SECTION("...for degree p = " << p)
//       {
//          // get the finite-element space for the state and adjoint
//          std::unique_ptr<FiniteElementCollection> fec(
//              new H1_FECollection(p, dim));
//          std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
//              mesh.get(), fec.get()));

//          // we use res for finite-difference approximation
//          Vector V(dim); Array<int> attr;
//          attr.SetSize(mesh->bdr_attributes.Size(), 0);
//          attr[0] = 1;
//          V = 1.0;
//          std::unique_ptr<VectorCoefficient> Q(new VectorConstantCoefficient(V));
//          LinearForm res(fes.get());
//          res.AddBoundaryIntegrator(
//             new BoundaryNormalLFIntegrator(*Q), attr);

//          // initialize state and adjoint; here we randomly perturb a constant state
//          GridFunction state(fes.get()), adjoint(fes.get());
//          FunctionCoefficient pert(randState);
//          state.ProjectCoefficient(pert);
//          adjoint.ProjectCoefficient(pert);

//          // extract mesh nodes and get their finite-element space
//          GridFunction *x_nodes = mesh->GetNodes();
//          FiniteElementSpace *mesh_fes = x_nodes->FESpace();

//          // build the nonlinear form for d(psi^T R)/dx 
//          LinearForm dfdx_form(mesh_fes);
//          dfdx_form.AddBdrFaceIntegrator(
//             new mach::BoundaryNormalResIntegrator(*Q,
//                &state, &adjoint), attr);

//          // initialize the vector that we use to perturb the mesh nodes
//          GridFunction v(mesh_fes);
//          VectorFunctionCoefficient v_rand(dim, randVectorState);
//          v.ProjectCoefficient(v_rand);

//          // evaluate df/dx and contract with v
//          //GridFunction dfdx(*x_nodes);
//          //dfdx_form.Mult(*x_nodes, dfdx);
//          dfdx_form.Assemble();
//          double dfdx_v = dfdx_form * v;

//          // now compute the finite-difference approximation...
//          GridFunction x_pert(*x_nodes);
//          GridFunction r(fes.get());
//          x_pert.Add(delta, v);
//          mesh->SetNodes(x_pert);
//          res.Assemble();
//          double dfdx_v_fd = adjoint * res;
//          x_pert.Add(-2 * delta, v);
//          mesh->SetNodes(x_pert);
//          res.Assemble();
//          dfdx_v_fd -= adjoint * res;
//          dfdx_v_fd /= (2 * delta);
//          mesh->SetNodes(*x_nodes); // remember to reset the mesh nodes

//          REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-10));
//       }
//    }
// }