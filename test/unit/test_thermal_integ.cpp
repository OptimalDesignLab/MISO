#include <random>
#include <string>

#include "catch.hpp"
#include "mfem.hpp"

#include "euler_test_data.hpp"

#include "temp_integ.hpp"
#include "therm_integ.hpp"

// std::string two_region_mesh_str = R"(MFEM mesh v1.0

// #
// # MFEM Geometry Types (see mesh/geom.hpp):
// #
// # POINT       = 0
// # SEGMENT     = 1
// # TRIANGLE    = 2
// # SQUARE      = 3
// # TETRAHEDRON = 4
// # CUBE        = 5
// # PRISM       = 6
// #

// dimension
// 3

// elements
// 48
// 2 4 12 26 13 22
// 2 4 1 19 24 18
// 1 4 15 8 17 10
// 1 4 0 17 11 10
// 2 4 12 25 22 21
// 1 4 17 7 8 16
// 1 4 0 2 15 1
// 1 4 3 5 6 4
// 1 4 7 17 11 6
// 1 4 2 6 11 17
// 2 4 26 20 13 22
// 1 4 2 6 17 3
// 1 4 10 14 8 9
// 1 4 7 17 6 13
// 2 4 26 12 25 22
// 2 4 3 26 13 17
// 2 4 23 26 16 25
// 2 4 26 1 2 19
// 2 4 20 26 2 19
// 2 4 14 26 17 16
// 2 4 26 14 15 23
// 1 4 17 10 8 11
// 1 4 2 6 5 11
// 2 4 3 26 20 13
// 2 4 26 1 15 2
// 2 4 3 26 17 2
// 2 4 26 3 20 2
// 1 4 17 0 15 10
// 1 4 17 7 11 8
// 2 4 26 12 16 25
// 2 4 2 15 26 17
// 1 4 8 15 17 14
// 2 4 26 14 23 16
// 1 4 17 8 14 16
// 1 4 17 3 6 13
// 1 4 7 17 13 16
// 1 4 17 0 2 15
// 2 4 26 12 13 16
// 2 4 13 26 16 17
// 1 4 13 7 16 12
// 2 4 1 26 15 24
// 2 4 26 1 19 24
// 1 4 6 2 5 3
// 1 4 17 0 11 2
// 1 4 11 2 0 5
// 2 4 15 26 23 24
// 2 4 14 26 15 17
// 1 4 15 8 10 14

// boundary
// 56
// 1 2 2 0 5
// 1 2 0 2 1
// 1 2 3 5 4
// 1 2 5 3 2
// 2 2 11 0 10
// 2 2 0 11 5
// 2 2 6 5 11
// 2 2 5 6 4
// 2 2 8 10 9
// 2 2 10 8 11
// 2 2 7 11 8
// 2 2 11 7 6
// 3 2 6 3 4
// 3 2 3 6 13
// 3 2 7 13 6
// 3 2 13 7 12
// 4 2 17 14 15
// 4 2 14 17 16
// 4 2 2 15 1
// 4 2 15 2 17
// 4 2 13 16 17
// 4 2 16 13 12
// 4 2 3 17 2
// 4 2 17 3 13
// 5 2 19 1 2
// 5 2 1 19 18
// 5 2 20 2 3
// 5 2 2 20 19
// 6 2 15 0 1
// 6 2 0 15 10
// 6 2 14 10 15
// 6 2 10 14 9
// 7 2 16 7 8
// 7 2 7 16 12
// 7 2 14 8 9
// 7 2 8 14 16
// 8 2 13 20 3
// 8 2 20 13 22
// 8 2 12 22 13
// 8 2 22 12 21
// 9 2 24 1 18
// 9 2 1 24 15
// 9 2 23 15 24
// 9 2 15 23 14
// 10 2 25 12 16
// 10 2 12 25 21
// 10 2 23 16 14
// 10 2 16 23 25
// 11 2 26 23 24
// 11 2 23 26 25
// 11 2 19 24 18
// 11 2 24 19 26
// 11 2 22 25 26
// 11 2 25 22 21
// 11 2 20 26 19
// 11 2 26 20 22

// vertices
// 27
// 3
// 0 0 0
// 0 0.5 0
// 0 0.5 0.5
// 0 0.5 1
// 0 0 1
// 0 0 0.5
// 0.5 0 1
// 1 0 1
// 1 0 0.5
// 1 0 0
// 0.5 0 0
// 0.5 0 0.5
// 1 0.5 1
// 0.5 0.5 1
// 1 0.5 0
// 0.5 0.5 0
// 1 0.5 0.5
// 0.5 0.5 0.5
// 0 1 0
// 0 1 0.5
// 0 1 1
// 1 1 1
// 0.5 1 1
// 1 1 0
// 0.5 1 0
// 1 1 0.5
// 0.5 1 0.5)";

// TEST_CASE("InteriorBoundaryOutFluxInteg::AssembleFaceMatrix")
// {
//    using namespace mfem;
//    using namespace mach;

//    constexpr int dim = 3;
//    constexpr int num_edge = 2;

//    constexpr double delta = 1e-5;

//    std::stringstream meshStr;
//    meshStr << two_region_mesh_str;
//    std::unique_ptr<Mesh> mesh(new Mesh(meshStr));
   
//    mesh->bdr_attributes.Print();
//    for (int p = 1; p <= 4; ++p)
//    {
//       DYNAMIC_SECTION("... for degree p = " << p)
//       {
//          H1_FECollection fec(p, dim);
//          FiniteElementSpace fes(mesh.get(), &fec);

//          ConstantCoefficient k(1.0);
//          ConstantCoefficient h(5.0);
//          NonlinearForm res(&fes);
//          Array<int> conv_face(mesh->bdr_attributes.Max());
//          conv_face = 0;
//          conv_face[3] = 1;
//          res.AddInteriorFaceIntegrator(
//             new InteriorBoundaryOutFluxInteg(k, h, 1, 10));
//          // res.AddBdrFaceIntegrator(
//          //    new InteriorBoundaryOutFluxInteg(k, h, 1, 10), conv_face);

//          GridFunction theta(&fes);
//          FunctionCoefficient pert(euler_data::randBaselinePert);
//          theta.ProjectCoefficient(pert);

//          // initialize the vector that the Jacobian multiplies
//          GridFunction v(&fes);
//          FunctionCoefficient v_rand(euler_data::randState);
//          v.ProjectCoefficient(v_rand);

//          // evaluate the Jacobian and compute its product with v
//          Operator& Jac = res.GetGradient(theta);
//          GridFunction jac_v(&fes);
//          Jac.Mult(v, jac_v);

//          // now compute the finite-difference approximation...
//          GridFunction theta_pert(theta), r(&fes), jac_v_fd(&fes);
//          theta_pert.Add(-delta, v);
//          res.Mult(theta_pert, r);
//          theta_pert.Add(2*delta, v);
//          res.Mult(theta_pert, jac_v_fd);
//          jac_v_fd -= r;
//          jac_v_fd /= (2*delta);

//          for (int i = 0; i < jac_v.Size(); ++i)
//          {
//             std::cout << "jac_v = " << jac_v(i) << std::endl;
//             std::cout << "jac_v_fd = " << jac_v_fd(i) << std::endl;
//             REQUIRE( jac_v(i) == Approx(jac_v_fd(i)).margin(1e-10) );
//          }

//       }
//    }
// }

TEST_CASE("AggregateIntegratorNumerator/Denominator::GetEnergy")
{
   using namespace mfem;
   using namespace mach;

   constexpr int dim = 3;
   int num_edge = 2;
   std::unique_ptr<Mesh> mesh(
      new Mesh(Mesh::MakeCartesian3D(num_edge, num_edge, num_edge,
                                     Element::TETRAHEDRON,
                                     1.0, 1.0, 1.0, true)));
   
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("... for degree p = " << p)
      {
         H1_FECollection fec(p, dim);
         FiniteElementSpace fes(mesh.get(), &fec);

         GridFunction theta(&fes);
         FunctionCoefficient parabola([](const Vector &x)
         {
            return -(x(0)-0.5)*(x(0)-0.5) + -(x(1)-0.5)*(x(1)-0.5) + 100;
         });
         theta.ProjectCoefficient(parabola);
         {
               std::ofstream sol_ofs("theta.vtk");
               sol_ofs.precision(14);
               int refine = p+1;
               mesh->PrintVTK(sol_ofs, refine);
               theta.SaveVTK(sol_ofs, "Solution", refine);
               sol_ofs.close();
         }
         Vector max(2);
         max = 95;
         double rho = 20;

         NonlinearForm numer(&fes);
         numer.AddDomainIntegrator(
            new AggregateIntegratorNumerator(rho, max));

         NonlinearForm denom(&fes);
         denom.AddDomainIntegrator(
            new AggregateIntegratorDenominator(rho, max));

         double n = numer.GetEnergy(theta);
         double d = denom.GetEnergy(theta);

         double fun = n/d;
         // std::cout << "n: " << n << " d: " << d << " fun: " << fun << "\n";
      }
   }
}

TEST_CASE("AggregateIntegrator::AssembleVector", "[AggregateIntegrator]")
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 3; // templating is hard here because mesh constructors
   double delta = 1e-5;

   // generate a 2 element mesh
   int num_edge = 1;
   std::unique_ptr<Mesh> mesh(
      new Mesh(Mesh::MakeCartesian3D(num_edge, num_edge, num_edge,
                                     Element::TETRAHEDRON,
                                     1.0, 1.0, 1.0, true)));
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
             new H1_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             mesh.get(), fec.get()));

         std::unique_ptr<mach::AggregateIntegrator> func;

         const double scale = 0.01;

         Vector m;
         m.SetSize(2);
         m(0) = 0.5;
         m(1) = 0.5;

         double delta = 1e-5;

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         FunctionCoefficient pert(randState);
         q.ProjectCoefficient(pert);

         double rhoa = 5.0;
         NonlinearForm agg(fes.get());
         agg.AddDomainIntegrator(
            new mach::AggregateIntegrator(fes.get(), rhoa, m, &q));
         func.reset(new mach::AggregateIntegrator(fes.get(), rhoa, m));

         // initialize the vector that dJdu multiplies
         GridFunction v(fes.get());
         FunctionCoefficient v_rand(randState);
         v.ProjectCoefficient(v_rand);

         // evaluate dJdu and compute its product with v
         GridFunction dJdu(fes.get());
         agg.Mult(q, dJdu);
         double dJdu_dot_v = InnerProduct(dJdu, v);

         // now compute the finite-difference approximation...
         GridFunction q_pert(q);
         q_pert.Add(-delta, v);
         double dJdu_dot_v_fd = -func->GetIEAggregate(&q_pert);
         q_pert.Add(2 * delta, v);
         dJdu_dot_v_fd += func->GetIEAggregate(&q_pert);
         dJdu_dot_v_fd /= (2 * delta);
         // std::cout << "dJdu_dot_v = " << dJdu_dot_v << std::endl;
         // std::cout << "dJdu_dot_v_fd = " << dJdu_dot_v_fd << std::endl;
         REQUIRE(dJdu_dot_v == Approx(dJdu_dot_v_fd));
      }
   }
}

TEST_CASE("TempIntegrator::AssembleVector", "[TempIntegrator]")
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 3; // templating is hard here because mesh constructors
   double delta = 1e-5;

   // generate a 2 element mesh
   int num_edge = 1;
   std::unique_ptr<Mesh> mesh(
      new Mesh(Mesh::MakeCartesian3D(num_edge, num_edge, num_edge,
                                     Element::TETRAHEDRON,
                                     1.0, 1.0, 1.0, true)));
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
             new H1_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             mesh.get(), fec.get()));

         std::unique_ptr<mach::TempIntegrator> func;

         const double scale = 0.01;

         double delta = 1e-5;

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         FunctionCoefficient pert(randState);
         q.ProjectCoefficient(pert);

         NonlinearForm tempint(fes.get());
         tempint.AddDomainIntegrator(
            new mach::TempIntegrator(fes.get(), &q));
         func.reset(new mach::TempIntegrator(fes.get()));

         // initialize the vector that dJdu multiplies
         GridFunction v(fes.get());
         FunctionCoefficient v_rand(randState);
         v.ProjectCoefficient(v_rand);

         // evaluate dJdu and compute its product with v
         GridFunction dJdu(fes.get());
         tempint.Mult(q, dJdu);
         double dJdu_dot_v = InnerProduct(dJdu, v);

         // now compute the finite-difference approximation...
         GridFunction q_pert(q);
         q_pert.Add(-delta, v);
         double dJdu_dot_v_fd = -func->GetTemp(&q_pert);
         q_pert.Add(2 * delta, v);
         dJdu_dot_v_fd += func->GetTemp(&q_pert);
         dJdu_dot_v_fd /= (2 * delta);
         // std::cout << "dJdu_dot_v = " << dJdu_dot_v << std::endl;
         // std::cout << "dJdu_dot_v_fd = " << dJdu_dot_v_fd << std::endl;
         REQUIRE(dJdu_dot_v == Approx(dJdu_dot_v_fd));
      }
   }
}

TEST_CASE("AggregateResIntegrator::AssembleVector", "[AggregateResIntegrator]")
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 3; // templating is hard here because mesh constructors
   double delta = 1e-5;

   // generate a 2 element mesh
   int num_edge = 1;
   std::unique_ptr<Mesh> mesh(
      new Mesh(Mesh::MakeCartesian3D(num_edge, num_edge, num_edge,
                                     Element::TETRAHEDRON,
                                     1.0, 1.0, 1.0, true)));
   mesh->EnsureNodes();
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         // get the finite-element space for the state
         std::unique_ptr<FiniteElementCollection> fec(
             new H1_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             mesh.get(), fec.get()));

         std::unique_ptr<mach::AggregateResIntegrator> func;

         const double scale = 0.01;

         Vector m;
         m.SetSize(2);
         m(0) = 0.5;
         m(1) = 0.5;

         double delta = 1e-5;

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         FunctionCoefficient pert(randState);
         q.ProjectCoefficient(pert);

         // extract mesh nodes and get their finite-element space
         GridFunction *x_nodes = mesh->GetNodes();
         FiniteElementSpace *mesh_fes = x_nodes->FESpace();

         double rhoa = 5.0;
         
         // initialize the vector that dJdx multiplies
         GridFunction v(mesh_fes);
         VectorFunctionCoefficient v_rand(dim, randVectorState);
         v.ProjectCoefficient(v_rand);

         // evaluate dJdx and compute its product with v
         GridFunction dJdx(*x_nodes);
         NonlinearForm agg(mesh_fes);
         agg.AddDomainIntegrator(
            new mach::AggregateResIntegrator(fes.get(), rhoa, m, &q));
         func.reset(new mach::AggregateResIntegrator(fes.get(), rhoa, m));
         agg.Mult(*x_nodes, dJdx);
         double dJdx_dot_v = dJdx * v;

         // now compute the finite-difference approximation...
         GridFunction x_pert(*x_nodes);
         x_pert.Add(-delta, v);
         mesh->SetNodes(x_pert);
         double dJdx_dot_v_fd = -func->GetIEAggregate(&q);
         x_pert.Add(2 * delta, v);
         mesh->SetNodes(x_pert);
         dJdx_dot_v_fd += func->GetIEAggregate(&q);
         dJdx_dot_v_fd /= (2 * delta);
         // std::cout << "dJdx_dot_v = " << dJdx_dot_v << std::endl;
         // std::cout << "dJdx_dot_v_fd = " << dJdx_dot_v_fd << std::endl;
         REQUIRE(dJdx_dot_v == Approx(dJdx_dot_v_fd));
      }
   }
}

TEST_CASE("TempResIntegrator::AssembleVector", "[TempResIntegrator]")
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 3; // templating is hard here because mesh constructors
   double delta = 1e-5;

   // generate a 2 element mesh
   int num_edge = 1;
   std::unique_ptr<Mesh> mesh(
      new Mesh(Mesh::MakeCartesian3D(num_edge, num_edge, num_edge,
                                     Element::TETRAHEDRON,
                                     1.0, 1.0, 1.0, true)));
   mesh->EnsureNodes();
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         // get the finite-element space for the state
         std::unique_ptr<FiniteElementCollection> fec(
             new H1_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             mesh.get(), fec.get()));

         std::unique_ptr<mach::TempResIntegrator> func;

         const double scale = 0.01;

         double delta = 1e-5;

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         FunctionCoefficient pert(randState);
         q.ProjectCoefficient(pert);

         // extract mesh nodes and get their finite-element space
         GridFunction *x_nodes = mesh->GetNodes();
         FiniteElementSpace *mesh_fes = x_nodes->FESpace();
         
         // initialize the vector that dJdx multiplies
         GridFunction v(mesh_fes);
         VectorFunctionCoefficient v_rand(dim, randVectorState);
         v.ProjectCoefficient(v_rand);

         // evaluate dJdx and compute its product with v
         GridFunction dJdx(*x_nodes);
         NonlinearForm tempint(mesh_fes);
         tempint.AddDomainIntegrator(
            new mach::TempResIntegrator(fes.get(), &q));
         func.reset(new mach::TempResIntegrator(fes.get()));
         tempint.Mult(*x_nodes, dJdx);
         double dJdx_dot_v = dJdx * v;

         // now compute the finite-difference approximation...
         GridFunction x_pert(*x_nodes);
         x_pert.Add(-delta, v);
         mesh->SetNodes(x_pert);
         double dJdx_dot_v_fd = -func->GetTemp(&q);
         x_pert.Add(2 * delta, v);
         mesh->SetNodes(x_pert);
         dJdx_dot_v_fd += func->GetTemp(&q);
         dJdx_dot_v_fd /= (2 * delta);
         // std::cout << "dJdx_dot_v = " << dJdx_dot_v << std::endl;
         // std::cout << "dJdx_dot_v_fd = " << dJdx_dot_v_fd << std::endl;
         REQUIRE(dJdx_dot_v == Approx(dJdx_dot_v_fd));
      }
   }
}