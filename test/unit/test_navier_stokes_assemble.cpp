#include <random>

#include "catch.hpp"
#include "mfem.hpp"

#include "euler_test_data.hpp"
#include "navier_stokes_integ.hpp"
#include "navier_stokes_fluxes.hpp"

TEST_CASE("ESViscousIntegrator::AssembleElementGrad", "[ESViscousIntegrator]")
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 2; // templating is hard here because mesh constructors
   int num_state = dim + 2;
   const double Re_num = 1.0;
   const double Pr_num = 1.0;
   const double vis = -1.0; // use Sutherland's law
   adept::Stack diff_stack;
   double delta = 1e-5;

   // generate a 2 element mesh
   int num_edge = 1;
   Mesh mesh(Mesh::MakeCartesian2D(num_edge, num_edge, Element::TRIANGLE,
                             true /* gen. edges */, 1.0, 1.0, true));
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
             new SBPCollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             &mesh, fec.get(), num_state, Ordering::byVDIM));
         NonlinearForm res(fes.get());
         res.AddDomainIntegrator(
             new mach::ESViscousIntegrator<2>(diff_stack, Re_num, Pr_num, vis));

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselineVectorPert<2>);
         q.ProjectCoefficient(pert);

         // initialize the vector that the Jacobian multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(num_state, randVectorState);
         v.ProjectCoefficient(v_rand);

         // evaluate the Jacobian and compute its product with v
         Operator &Jac = res.GetGradient(q);
         GridFunction jac_v(fes.get());
         Jac.Mult(v, jac_v);

         // now compute the finite-difference approximation...
         GridFunction q_pert(q), r(fes.get()), jac_v_fd(fes.get());
         q_pert.Add(-delta, v);
         res.Mult(q_pert, r);
         q_pert.Add(2 * delta, v);
         res.Mult(q_pert, jac_v_fd);
         jac_v_fd -= r;
         jac_v_fd /= (2 * delta);

         for (int i = 0; i < jac_v.Size(); ++i)
         {
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)));
         }
      }
   }
}

TEST_CASE("ESViscousIntegrator::AssembleElementGrad 3D", "[ESViscousIntegrator]")
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 3; // templating is hard here because mesh constructors
   int num_state = dim + 2;
   const double Re_num = 1.0;
   const double Pr_num = 1.0;
   const double vis = -1.0; // use Sutherland's law
   adept::Stack diff_stack;
   double delta = 1e-5;

   // generate a 8 element mesh
   int num_edge = 2;
   Mesh mesh(Mesh::MakeCartesian3D(num_edge, num_edge, num_edge, 
             Element::TETRAHEDRON, 1.0, 1.0, 1.0, Ordering::byVDIM));
   for (int p = 0; p <= 1; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
             new SBPCollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             &mesh, fec.get(), num_state, Ordering::byVDIM));
         NonlinearForm res(fes.get());
         res.AddDomainIntegrator(
             new mach::ESViscousIntegrator<dim>(diff_stack, Re_num, Pr_num, vis));

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselineVectorPert<dim>);
         q.ProjectCoefficient(pert);

         // initialize the vector that the Jacobian multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(num_state, randVectorState);
         v.ProjectCoefficient(v_rand);

         // evaluate the Jacobian and compute its product with v
         Operator &Jac = res.GetGradient(q);
         GridFunction jac_v(fes.get());
         Jac.Mult(v, jac_v);

         // now compute the finite-difference approximation...
         GridFunction q_pert(q), r(fes.get()), jac_v_fd(fes.get());
         q_pert.Add(-delta, v);
         res.Mult(q_pert, r);
         q_pert.Add(2 * delta, v);
         res.Mult(q_pert, jac_v_fd);
         jac_v_fd -= r;
         jac_v_fd /= (2 * delta);

         for (int i = 0; i < jac_v.Size(); ++i)
         {
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)));
         }
      }
   }
}

TEST_CASE("NoSlipAdiabaticWallBC::AssembleFaceGrad", "[NoSlipBC]")
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 2; // templating is hard here because mesh constructors
   int num_state = dim + 2;
   const double Re_num = 1.0;
   const double Pr_num = 1.0;
   const double vis = -1.0; // use Sutherland's law
   adept::Stack diff_stack;
   double delta = 1e-5;

   Vector q_ref(dim + 2);
   q_ref(0) = rho;
   q_ref(dim + 1) = rhoe;
   for (int di = 0; di < dim; ++di)
   {
      q_ref(di + 1) = rhou[di];
   }

   // generate a 2 element mesh
   int num_edge = 1;
   Mesh mesh(Mesh::MakeCartesian2D(num_edge, num_edge, Element::TRIANGLE,
                                   true /* gen. edges */, 1.0, 1.0, true));
   for (int p = 1; p <= 1; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
             new SBPCollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             &mesh, fec.get(), num_state, Ordering::byVDIM));

         NonlinearForm res(fes.get());
         res.AddBdrFaceIntegrator(
             new mach::NoSlipAdiabaticWallBC<dim>(diff_stack, fec.get(), Re_num,
                                                  Pr_num, q_ref, vis));

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselineVectorPert<dim>);
         q.ProjectCoefficient(pert);

         // initialize the vector that the Jacobian multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(num_state, randVectorState);
         v.ProjectCoefficient(v_rand);

         // evaluate the Jacobian and compute its product with v
         Operator &Jac = res.GetGradient(q);
         GridFunction jac_v(fes.get());
         Jac.Mult(v, jac_v);

         // now compute the finite-difference approximation...
         GridFunction q_pert(q), r(fes.get()), jac_v_fd(fes.get());
         q_pert.Add(-delta, v);
         res.Mult(q_pert, r);
         q_pert.Add(2 * delta, v);
         res.Mult(q_pert, jac_v_fd);
         jac_v_fd -= r;
         jac_v_fd /= (2 * delta);

         for (int i = 0; i < jac_v.Size(); ++i)
         {
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)));
         }
      }
   }
}

TEST_CASE("NoSlipAdiabaticWallBC::AssembleFaceGrad 3D", "[NoSlipBC]")
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 3; // templating is hard here because mesh constructors
   int num_state = dim + 2;
   const double Re_num = 1.0;
   const double Pr_num = 1.0;
   const double vis = -1.0; // use Sutherland's law
   adept::Stack diff_stack;
   double delta = 1e-5;

   Vector q_ref(dim + 2);
   q_ref(0) = rho;
   q_ref(dim + 1) = rhoe;
   for (int di = 0; di < dim; ++di)
   {
      q_ref(di + 1) = rhou[di];
   }

   // generate a 8 element mesh
   int num_edge = 2;
   Mesh mesh(Mesh::MakeCartesian3D(num_edge, num_edge, num_edge, 
             Element::TETRAHEDRON, 1.0, 1.0, 1.0, Ordering::byVDIM));
   for (int p = 0; p <= 1; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
             new SBPCollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             &mesh, fec.get(), num_state, Ordering::byVDIM));

         NonlinearForm res(fes.get());
         res.AddBdrFaceIntegrator(
             new mach::NoSlipAdiabaticWallBC<dim>(diff_stack, fec.get(), Re_num,
                                                  Pr_num, q_ref, vis));

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselineVectorPert<dim>);
         q.ProjectCoefficient(pert);

         // initialize the vector that the Jacobian multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(num_state, randVectorState);
         v.ProjectCoefficient(v_rand);

         // evaluate the Jacobian and compute its product with v
         Operator &Jac = res.GetGradient(q);
         GridFunction jac_v(fes.get());
         Jac.Mult(v, jac_v);

         // now compute the finite-difference approximation...
         GridFunction q_pert(q), r(fes.get()), jac_v_fd(fes.get());
         q_pert.Add(-delta, v);
         res.Mult(q_pert, r);
         q_pert.Add(2 * delta, v);
         res.Mult(q_pert, jac_v_fd);
         jac_v_fd -= r;
         jac_v_fd /= (2 * delta);

         for (int i = 0; i < jac_v.Size(); ++i)
         {
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)));
         }
      }
   }
}

TEST_CASE("ViscousSlipWallBC::AssembleFaceGrad", "[VisSlipWallBC]")
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 2; // templating is hard here because mesh constructors
   int num_state = dim + 2;
   const double Re_num = 1.0;
   const double Pr_num = 1.0;
   const double vis = -1.0; // use Sutherland's law
   adept::Stack diff_stack;
   double delta = 1e-5;

   // generate a 2 element mesh
   int num_edge = 1;
   Mesh mesh(Mesh::MakeCartesian2D(num_edge, num_edge, Element::TRIANGLE,
                                   true /* gen. edges */, 1.0, 1.0, true));
   for (int p = 1; p <= 1; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
             new SBPCollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             &mesh, fec.get(), num_state, Ordering::byVDIM));

         NonlinearForm res(fes.get());
         res.AddBdrFaceIntegrator(
             new mach::ViscousSlipWallBC<dim>(diff_stack, fec.get(), Re_num,
                                              Pr_num, vis));

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselineVectorPert<dim>);
         q.ProjectCoefficient(pert);

         // initialize the vector that the Jacobian multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(num_state, randVectorState);
         v.ProjectCoefficient(v_rand);

         // evaluate the Jacobian and compute its product with v
         Operator &Jac = res.GetGradient(q);
         GridFunction jac_v(fes.get());
         Jac.Mult(v, jac_v);

         // now compute the finite-difference approximation...
         GridFunction q_pert(q), r(fes.get()), jac_v_fd(fes.get());
         q_pert.Add(-delta, v);
         res.Mult(q_pert, r);
         q_pert.Add(2 * delta, v);
         res.Mult(q_pert, jac_v_fd);
         jac_v_fd -= r;
         jac_v_fd /= (2 * delta);

         for (int i = 0; i < jac_v.Size(); ++i)
         {
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)));
         }
      }
   }
}

TEST_CASE("ViscousSlipWallBC::AssembleFaceGrad 3D", "[VisSlipWallBC]")
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 3; // templating is hard here because mesh constructors
   int num_state = dim + 2;
   const double Re_num = 1.0;
   const double Pr_num = 1.0;
   const double vis = -1.0; // use Sutherland's law
   adept::Stack diff_stack;
   double delta = 1e-5;

   // generate a 8 element mesh
   int num_edge = 2;
   Mesh mesh(Mesh::MakeCartesian3D(num_edge, num_edge, num_edge, 
             Element::TETRAHEDRON, 1.0, 1.0, 1.0, Ordering::byVDIM));
   for (int p = 0; p <= 1; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
             new SBPCollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             &mesh, fec.get(), num_state, Ordering::byVDIM));

         NonlinearForm res(fes.get());
         res.AddBdrFaceIntegrator(
             new mach::ViscousSlipWallBC<dim>(diff_stack, fec.get(), Re_num,
                                              Pr_num, vis));

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselineVectorPert<dim>);
         q.ProjectCoefficient(pert);

         // initialize the vector that the Jacobian multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(num_state, randVectorState);
         v.ProjectCoefficient(v_rand);

         // evaluate the Jacobian and compute its product with v
         Operator &Jac = res.GetGradient(q);
         GridFunction jac_v(fes.get());
         Jac.Mult(v, jac_v);

         // now compute the finite-difference approximation...
         GridFunction q_pert(q), r(fes.get()), jac_v_fd(fes.get());
         q_pert.Add(-delta, v);
         res.Mult(q_pert, r);
         q_pert.Add(2 * delta, v);
         res.Mult(q_pert, jac_v_fd);
         jac_v_fd -= r;
         jac_v_fd /= (2 * delta);

         for (int i = 0; i < jac_v.Size(); ++i)
         {
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)));
         }
      }
   }
}

TEST_CASE("ViscousInflowWallBC::AssembleFaceGrad", "[VisInflowBC]")
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 2; // templating is hard here because mesh constructors
   int num_state = dim + 2;
   const double Re_num = 1.0;
   const double Pr_num = 1.0;
   const double vis = -1.0; // use Sutherland's law
   adept::Stack diff_stack;
   double delta = 1e-5;

   Vector q_in(dim + 2);
   q_in(0) = rho;
   q_in(dim + 1) = rhoe;
   for (int di = 0; di < dim; ++di)
   {
      q_in(di + 1) = rhou[di];
   }

   // generate a 2 element mesh
   int num_edge = 1;
   Mesh mesh(Mesh::MakeCartesian2D(num_edge, num_edge, Element::TRIANGLE,
                                   true /* gen. edges */, 1.0, 1.0, true));
   for (int p = 1; p <= 1; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
             new SBPCollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             &mesh, fec.get(), num_state, Ordering::byVDIM));

         NonlinearForm res(fes.get());
         res.AddBdrFaceIntegrator(
             new mach::ViscousInflowBC<dim>(diff_stack, fec.get(), Re_num,
                                            Pr_num, q_in, vis));

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselineVectorPert<dim>);
         q.ProjectCoefficient(pert);

         // initialize the vector that the Jacobian multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(num_state, randVectorState);
         v.ProjectCoefficient(v_rand);

         // evaluate the Jacobian and compute its product with v
         Operator &Jac = res.GetGradient(q);
         GridFunction jac_v(fes.get());
         Jac.Mult(v, jac_v);

         // now compute the finite-difference approximation...
         GridFunction q_pert(q), r(fes.get()), jac_v_fd(fes.get());
         q_pert.Add(-delta, v);
         res.Mult(q_pert, r);
         q_pert.Add(2 * delta, v);
         res.Mult(q_pert, jac_v_fd);
         jac_v_fd -= r;
         jac_v_fd /= (2 * delta);

         for (int i = 0; i < jac_v.Size(); ++i)
         {
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)));
         }
      }
   }
}

TEST_CASE("ViscousInflowWallBC::AssembleFaceGrad 3D", "[VisInflowBC]")
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 3; // templating is hard here because mesh constructors
   int num_state = dim + 2;
   const double Re_num = 1.0;
   const double Pr_num = 1.0;
   const double vis = -1.0; // use Sutherland's law
   adept::Stack diff_stack;
   double delta = 1e-5;

   Vector q_in(dim + 2);
   q_in(0) = rho;
   q_in(dim + 1) = rhoe;
   for (int di = 0; di < dim; ++di)
   {
      q_in(di + 1) = rhou[di];
   }

   // generate a 2 element mesh
   int num_edge = 2;
   Mesh mesh(Mesh::MakeCartesian3D(num_edge, num_edge, num_edge, 
             Element::TETRAHEDRON, 1.0, 1.0, 1.0, Ordering::byVDIM));
   for (int p = 0; p <= 1; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
             new SBPCollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             &mesh, fec.get(), num_state, Ordering::byVDIM));

         NonlinearForm res(fes.get());
         res.AddBdrFaceIntegrator(
             new mach::ViscousInflowBC<dim>(diff_stack, fec.get(), Re_num,
                                            Pr_num, q_in, vis));

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselineVectorPert<dim>);
         q.ProjectCoefficient(pert);

         // initialize the vector that the Jacobian multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(num_state, randVectorState);
         v.ProjectCoefficient(v_rand);

         // evaluate the Jacobian and compute its product with v
         Operator &Jac = res.GetGradient(q);
         GridFunction jac_v(fes.get());
         Jac.Mult(v, jac_v);

         // now compute the finite-difference approximation...
         GridFunction q_pert(q), r(fes.get()), jac_v_fd(fes.get());
         q_pert.Add(-delta, v);
         res.Mult(q_pert, r);
         q_pert.Add(2 * delta, v);
         res.Mult(q_pert, jac_v_fd);
         jac_v_fd -= r;
         jac_v_fd /= (2 * delta);

         for (int i = 0; i < jac_v.Size(); ++i)
         {
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)));
         }
      }
   }
}

TEST_CASE("ViscousOutflowBC::AssembleFaceGrad", "[VisOutflowBC]")
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 2; // templating is hard here because mesh constructors
   int num_state = dim + 2;
   const double Re_num = 1.0;
   const double Pr_num = 1.0;
   const double vis = -1.0; // use Sutherland's law
   adept::Stack diff_stack;
   double delta = 1e-5;

   Vector q_out(dim + 2);
   q_out(0) = rho;
   q_out(dim + 1) = rhoe;
   for (int di = 0; di < dim; ++di)
   {
      q_out(di + 1) = rhou[di];
   }

   // generate a 2 element mesh
   int num_edge = 1;
   Mesh mesh(Mesh::MakeCartesian2D(num_edge, num_edge, Element::TRIANGLE,
                                   true /* gen. edges */, 1.0, 1.0, true));
   for (int p = 1; p <= 1; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
             new SBPCollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             &mesh, fec.get(), num_state, Ordering::byVDIM));

         NonlinearForm res(fes.get());
         res.AddBdrFaceIntegrator(
             new mach::ViscousOutflowBC<dim>(diff_stack, fec.get(), Re_num,
                                             Pr_num, q_out, vis));

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselineVectorPert<dim>);
         q.ProjectCoefficient(pert);

         // initialize the vector that the Jacobian multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(num_state, randVectorState);
         v.ProjectCoefficient(v_rand);

         // evaluate the Jacobian and compute its product with v
         Operator &Jac = res.GetGradient(q);
         GridFunction jac_v(fes.get());
         Jac.Mult(v, jac_v);

         // now compute the finite-difference approximation...
         GridFunction q_pert(q), r(fes.get()), jac_v_fd(fes.get());
         q_pert.Add(-delta, v);
         res.Mult(q_pert, r);
         q_pert.Add(2 * delta, v);
         res.Mult(q_pert, jac_v_fd);
         jac_v_fd -= r;
         jac_v_fd /= (2 * delta);

         for (int i = 0; i < jac_v.Size(); ++i)
         {
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)));
         }
      }
   }
}

TEST_CASE("ViscousOutflowBC::AssembleFaceGrad 3D", "[VisOutflowBC]")
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 3; // templating is hard here because mesh constructors
   int num_state = dim + 2;
   const double Re_num = 1.0;
   const double Pr_num = 1.0;
   const double vis = -1.0; // use Sutherland's law
   adept::Stack diff_stack;
   double delta = 1e-5;

   Vector q_out(dim + 2);
   q_out(0) = rho;
   q_out(dim + 1) = rhoe;
   for (int di = 0; di < dim; ++di)
   {
      q_out(di + 1) = rhou[di];
   }

   // generate a 8 element mesh
   int num_edge = 2;
   Mesh mesh(Mesh::MakeCartesian3D(num_edge, num_edge, num_edge,
             Element::TETRAHEDRON, 1.0, 1.0, 1.0, Ordering::byVDIM));
   for (int p = 0; p <= 1; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
             new SBPCollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             &mesh, fec.get(), num_state, Ordering::byVDIM));

         NonlinearForm res(fes.get());
         res.AddBdrFaceIntegrator(
             new mach::ViscousOutflowBC<dim>(diff_stack, fec.get(), Re_num,
                                             Pr_num, q_out, vis));

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselineVectorPert<dim>);
         q.ProjectCoefficient(pert);

         // initialize the vector that the Jacobian multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(num_state, randVectorState);
         v.ProjectCoefficient(v_rand);

         // evaluate the Jacobian and compute its product with v
         Operator &Jac = res.GetGradient(q);
         GridFunction jac_v(fes.get());
         Jac.Mult(v, jac_v);

         // now compute the finite-difference approximation...
         GridFunction q_pert(q), r(fes.get()), jac_v_fd(fes.get());
         q_pert.Add(-delta, v);
         res.Mult(q_pert, r);
         q_pert.Add(2 * delta, v);
         res.Mult(q_pert, jac_v_fd);
         jac_v_fd -= r;
         jac_v_fd /= (2 * delta);

         for (int i = 0; i < jac_v.Size(); ++i)
         {
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)));
         }
      }
   }
}

TEST_CASE("ViscousFarFieldBC::AssembleFaceGrad", "[VisFarFieldBC]")
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 2; // templating is hard here because mesh constructors
   int num_state = dim + 2;
   const double Re_num = 1.0;
   const double Pr_num = 1.0;
   const double vis = -1.0; // use Sutherland's law
   adept::Stack diff_stack;
   double delta = 1e-5;

   Vector q_far(dim + 2);
   q_far(0) = rho;
   q_far(dim + 1) = rhoe;
   for (int di = 0; di < dim; ++di)
   {
      q_far(di + 1) = rhou[di];
   }

   // generate a 2 element mesh
   int num_edge = 1;
   Mesh mesh(Mesh::MakeCartesian2D(num_edge, num_edge, Element::TRIANGLE,
                                   true /* gen. edges */, 1.0, 1.0, true));
   for (int p = 1; p <= 1; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
             new SBPCollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             &mesh, fec.get(), num_state, Ordering::byVDIM));

         NonlinearForm res(fes.get());
         res.AddBdrFaceIntegrator(
             new mach::NoSlipAdiabaticWallBC<dim>(diff_stack, fec.get(), Re_num,
                                                  Pr_num, q_far, vis));

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselineVectorPert<dim>);
         q.ProjectCoefficient(pert);

         // initialize the vector that the Jacobian multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(num_state, randVectorState);
         v.ProjectCoefficient(v_rand);

         // evaluate the Jacobian and compute its product with v
         Operator &Jac = res.GetGradient(q);
         GridFunction jac_v(fes.get());
         Jac.Mult(v, jac_v);

         // now compute the finite-difference approximation...
         GridFunction q_pert(q), r(fes.get()), jac_v_fd(fes.get());
         q_pert.Add(-delta, v);
         res.Mult(q_pert, r);
         q_pert.Add(2 * delta, v);
         res.Mult(q_pert, jac_v_fd);
         jac_v_fd -= r;
         jac_v_fd /= (2 * delta);

         for (int i = 0; i < jac_v.Size(); ++i)
         {
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)));
         }
      }
   }
}

TEST_CASE("ViscousFarFieldBC::AssembleFaceGrad 3D", "[VisFarFieldBC]")
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 3; // templating is hard here because mesh constructors
   int num_state = dim + 2;
   const double Re_num = 1.0;
   const double Pr_num = 1.0;
   const double vis = -1.0; // use Sutherland's law
   adept::Stack diff_stack;
   double delta = 1e-5;

   Vector q_far(dim + 2);
   q_far(0) = rho;
   q_far(dim + 1) = rhoe;
   for (int di = 0; di < dim; ++di)
   {
      q_far(di + 1) = rhou[di];
   }

   // generate a 8 element mesh
   int num_edge = 2;
   Mesh mesh(Mesh::MakeCartesian3D(num_edge, num_edge, num_edge, 
             Element::TETRAHEDRON, 1.0, 1.0, 1.0, Ordering::byVDIM));
   for (int p = 0; p <= 1; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
             new SBPCollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             &mesh, fec.get(), num_state, Ordering::byVDIM));

         NonlinearForm res(fes.get());
         res.AddBdrFaceIntegrator(
             new mach::NoSlipAdiabaticWallBC<dim>(diff_stack, fec.get(), Re_num,
                                                  Pr_num, q_far, vis));

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselineVectorPert<dim>);
         q.ProjectCoefficient(pert);

         // initialize the vector that the Jacobian multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(num_state, randVectorState);
         v.ProjectCoefficient(v_rand);

         // evaluate the Jacobian and compute its product with v
         Operator &Jac = res.GetGradient(q);
         GridFunction jac_v(fes.get());
         Jac.Mult(v, jac_v);

         // now compute the finite-difference approximation...
         GridFunction q_pert(q), r(fes.get()), jac_v_fd(fes.get());
         q_pert.Add(-delta, v);
         res.Mult(q_pert, r);
         q_pert.Add(2 * delta, v);
         res.Mult(q_pert, jac_v_fd);
         jac_v_fd -= r;
         jac_v_fd /= (2 * delta);

         for (int i = 0; i < jac_v.Size(); ++i)
         {
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)));
         }
      }
   }
}

TEST_CASE("SurfaceForce::AssembleVector", "[Surface Force]")
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 2; // templating is hard here because mesh constructors
   int num_state = dim + 2;
   adept::Stack diff_stack;
   double delta = 1e-5;

   mfem::Vector drag_dir(dim);
   double aoa_fs = 5.0*M_PI/180;
   double Re = 1.0;
   double Pr = 0.72;
   drag_dir(0) = cos(aoa_fs);
   drag_dir(1) = sin(aoa_fs);

   mfem::Vector q_ref(dim + 2);
   q_ref(0) = rho;
   q_ref(dim + 1) = rhoe;
   for (int di = 0; di < dim; ++di)
   {
      q_ref(di + 1) = rhou[di];
   }

   // generate a 2 element mesh
   int num_edge = 2;
   Mesh mesh(Mesh::MakeCartesian2D(num_edge, num_edge, Element::TRIANGLE,
                                   true /* gen. edges */, 1.0, 1.0, true));
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
             new SBPCollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             &mesh, fec.get(), num_state, Ordering::byVDIM));

         NonlinearForm drag(fes.get());
         drag.AddBdrFaceIntegrator(
             new mach::SurfaceForce<dim>(diff_stack, fec.get(), dim + 2, Re, Pr,
                                         q_ref, drag_dir));

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselineVectorPert<dim>);
         q.ProjectCoefficient(pert);

         // initialize the vector that dJdu multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(num_state, randVectorState);
         v.ProjectCoefficient(v_rand);

         // evaluate dJdu and compute its product with v
         GridFunction dJdu(fes.get());
         drag.Mult(q, dJdu);
         double dJdu_dot_v = InnerProduct(dJdu, v);

         // now compute the finite-difference approximation...
         GridFunction q_pert(q);
         q_pert.Add(-delta, v);
         double dJdu_dot_v_fd = -drag.GetEnergy(q_pert);
         q_pert.Add(2 * delta, v);
         dJdu_dot_v_fd += drag.GetEnergy(q_pert);
         dJdu_dot_v_fd /= (2 * delta);
         REQUIRE(dJdu_dot_v == Approx(dJdu_dot_v_fd).margin(1e-10));
      }

      DYNAMIC_SECTION("(DSBP)...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
             new DSBPCollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             &mesh, fec.get(), num_state, Ordering::byVDIM));

         NonlinearForm drag(fes.get());
         drag.AddBdrFaceIntegrator(
             new mach::SurfaceForce<dim>(diff_stack, fec.get(), dim + 2, Re, Pr,
                                         q_ref, drag_dir));

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselineVectorPert<dim>);
         q.ProjectCoefficient(pert);

         // initialize the vector that dJdu multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(num_state, randVectorState);
         v.ProjectCoefficient(v_rand);

         // evaluate dJdu and compute its product with v
         GridFunction dJdu(fes.get());
         drag.Mult(q, dJdu);
         double dJdu_dot_v = InnerProduct(dJdu, v);

         // now compute the finite-difference approximation...
         GridFunction q_pert(q);
         q_pert.Add(-delta, v);
         double dJdu_dot_v_fd = -drag.GetEnergy(q_pert);
         q_pert.Add(2 * delta, v);
         dJdu_dot_v_fd += drag.GetEnergy(q_pert);
         dJdu_dot_v_fd /= (2 * delta);
         REQUIRE(dJdu_dot_v == Approx(dJdu_dot_v_fd).margin(1e-10));
      }
   }
}

TEST_CASE("SurfaceForce::AssembleVector 3D", "[Surface Force]")
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 3; // templating is hard here because mesh constructors
   int num_state = dim + 2;
   adept::Stack diff_stack;
   double delta = 1e-5;

   mfem::Vector drag_dir(dim);
   double aoa_fs = 5.0*M_PI/180;
   double Re = 1.0;
   double Pr = 0.72;
   drag_dir(0) = cos(aoa_fs);
   drag_dir(1) = sin(aoa_fs);
   drag_dir(2) = 0.0;

   mfem::Vector q_ref(dim + 2);
   q_ref(0) = rho;
   q_ref(dim + 1) = rhoe;
   for (int di = 0; di < dim; ++di)
   {
      q_ref(di + 1) = rhou[di];
   }

   // generate a 8 element mesh
   int num_edge = 2;
   Mesh mesh(Mesh::MakeCartesian3D(num_edge, num_edge, num_edge, Element::TETRAHEDRON,
                                   1.0, 1.0, 1.0, Ordering::byVDIM));
   for (int p = 0; p <= 1; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
             new SBPCollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             &mesh, fec.get(), num_state, Ordering::byVDIM));

         NonlinearForm drag(fes.get());
         drag.AddBdrFaceIntegrator(
             new mach::SurfaceForce<dim>(diff_stack, fec.get(), dim + 2, Re, Pr,
                                         q_ref, drag_dir));

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselineVectorPert<dim>);
         q.ProjectCoefficient(pert);

         // initialize the vector that dJdu multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(num_state, randVectorState);
         v.ProjectCoefficient(v_rand);

         // evaluate dJdu and compute its product with v
         GridFunction dJdu(fes.get());
         drag.Mult(q, dJdu);
         double dJdu_dot_v = InnerProduct(dJdu, v);

         // now compute the finite-difference approximation...
         GridFunction q_pert(q);
         q_pert.Add(-delta, v);
         double dJdu_dot_v_fd = -drag.GetEnergy(q_pert);
         q_pert.Add(2 * delta, v);
         dJdu_dot_v_fd += drag.GetEnergy(q_pert);
         dJdu_dot_v_fd /= (2 * delta);
         REQUIRE(dJdu_dot_v == Approx(dJdu_dot_v_fd).margin(1e-10));
      }
   }
}