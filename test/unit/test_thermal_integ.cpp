#include <random>

#include "catch.hpp"
#include "mfem.hpp"
#include "euler_fluxes.hpp"
#include "temp_integ.hpp"
#include "euler_test_data.hpp"

TEST_CASE("AggregateIntegrator::AssembleVector", "[AggregateIntegrator]")
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 3; // templating is hard here because mesh constructors
   double delta = 1e-5;

   // generate a 2 element mesh
   int num_edge = 1;
   std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, num_edge, Element::TETRAHEDRON,
                                       true /* gen. edges */, 1.0, 1.0, 1.0, true));
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
         std::cout << "dJdu_dot_v = " << dJdu_dot_v << std::endl;
         std::cout << "dJdu_dot_v_fd = " << dJdu_dot_v_fd << std::endl;
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
   std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, num_edge, Element::TETRAHEDRON,
                                       true /* gen. edges */, 1.0, 1.0, 1.0, true));
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
         std::cout << "dJdu_dot_v = " << dJdu_dot_v << std::endl;
         std::cout << "dJdu_dot_v_fd = " << dJdu_dot_v_fd << std::endl;
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
   std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, num_edge, Element::TETRAHEDRON,
                                       true /* gen. edges */, 1.0, 1.0, 1.0, true));
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
         std::cout << "dJdx_dot_v = " << dJdx_dot_v << std::endl;
         std::cout << "dJdx_dot_v_fd = " << dJdx_dot_v_fd << std::endl;
         REQUIRE(dJdx_dot_v == Approx(dJdx_dot_v_fd));
      }
   }
}