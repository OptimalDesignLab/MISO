#include <random>

#include "catch.hpp"
#include "mfem.hpp"
#include "euler_fluxes.hpp"
#include "euler_integ.hpp"
#include "euler_test_data.hpp"

TEST_CASE("EulerIntegrator::AssembleElementGrad", "[EulerIntegrator]")
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 2; // templating is hard here because mesh constructors
   int num_state = dim + 2;
   adept::Stack diff_stack;
   double delta = 1e-5;

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

         NonlinearForm res(fes.get());
         res.AddDomainIntegrator(new miso::EulerIntegrator<2>(diff_stack));

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
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)).margin(1e-10));
         }
      }

      DYNAMIC_SECTION("(DSBP)...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
             new DSBPCollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             &mesh, fec.get(), num_state, Ordering::byVDIM));

         NonlinearForm res(fes.get());
         res.AddDomainIntegrator(new miso::EulerIntegrator<2>(diff_stack));

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
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)).margin(1e-10));
         }
      }
   }
}

TEST_CASE("EulerIntegrator::AssembleElementGrad3D", "[EulerIntegrator]")
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 3;
   int num_state = dim + 2;
   adept::Stack diff_stack;
   double delta = 1e-5;

   // generate a 2 element mesh - 3 nodes in x, y, and z - directions respectively.
   int num_edge = 1;
   Mesh mesh(Mesh::MakeCartesian3D(num_edge,num_edge,num_edge,
                                   Element::TETRAHEDRON,1.0,1.0,1.0, true));
   for (int p = 0; p <= 1; ++p)
   {  
      DYNAMIC_SECTION("...for degree p = " << p)
      {  
         std::unique_ptr<FiniteElementCollection> fec(
             new SBPCollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             &mesh, fec.get(), num_state, Ordering::byVDIM));

         NonlinearForm res(fes.get());
         res.AddDomainIntegrator(new miso::EulerIntegrator<dim>(diff_stack));

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselineVectorPert<3>);
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
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)).margin(1e-10));
         }
      }
   }
}

TEST_CASE("EulerIntegrator::AssembleElementGrad3D - 1 tet element", "[EulerIntegrator]")
{  
   using namespace mfem;
   using namespace euler_data;

   const int dim = 3;
   int num_state = dim + 2;
   adept::Stack diff_stack;
   double delta = 1e-5;

   int nv = 4, ne = 1, nb = 0, sdim = 3, attrib = 1;
   Mesh mesh(dim, nv, ne, nb, sdim);
   mesh.AddVertex(Vertex(0.,0.,0.)());
   mesh.AddVertex(Vertex(1.,0.,0.)());
   mesh.AddVertex(Vertex(0.,1.,0.)());
   mesh.AddVertex(Vertex(0.,0.,1.)());

   int idx[4] = {0,1,2,3};
   mesh.AddElement(new Tetrahedron(idx,attrib));
   mesh.FinalizeTetMesh(1,1,true);  

   for (int p = 0; p <= 1; ++p)
   {  
      DYNAMIC_SECTION("...for degree p = " << p)
      {  
         std::unique_ptr<FiniteElementCollection> fec(
             new SBPCollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             &mesh, fec.get(), num_state, Ordering::byVDIM));

         NonlinearForm res(fes.get());
         res.AddDomainIntegrator(new miso::EulerIntegrator<dim>(diff_stack));

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselineVectorPert<3>);
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
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)).margin(1e-10));
         }
      }
   }
}

TEST_CASE("SlipWallBC::AssembleFaceGrad", "[SlipWallBC]")
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 2; // templating is hard here because mesh constructors
   int num_state = dim + 2;
   adept::Stack diff_stack;
   double delta = 1e-5;

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

         NonlinearForm res(fes.get());
         res.AddBdrFaceIntegrator(new miso::SlipWallBC<dim>(diff_stack,
                                                            fec.get()));

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
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)).margin(1e-10));
         }
      }

      DYNAMIC_SECTION("(DSBP)...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
             new DSBPCollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             &mesh, fec.get(), num_state, Ordering::byVDIM));

         NonlinearForm res(fes.get());
         res.AddBdrFaceIntegrator(new miso::SlipWallBC<dim>(diff_stack,
                                                            fec.get()));

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
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)).margin(1e-10));
         }
      }
   }
}

TEST_CASE("SlipWallBC::AssembleFaceGrad3D", "[SlipWallBC]")
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 3; // templating is hard here because mesh constructors
   int num_state = dim + 2;
   adept::Stack diff_stack;
   double delta = 1e-5;

   // generate a 2 element mesh
   int num_edge = 2;
   Mesh mesh(Mesh::MakeCartesian3D(num_edge,num_edge,num_edge,
                                   Element::TETRAHEDRON,1.0,1.0,1.0, true));
   for (int p = 0; p <= 1; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
             new SBPCollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             &mesh, fec.get(), num_state, Ordering::byVDIM));

         NonlinearForm res(fes.get());
         res.AddBdrFaceIntegrator(new miso::SlipWallBC<dim>(diff_stack,
                                                            fec.get()));

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
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)).margin(1e-10));
         }
      }
   }
}

TEST_CASE("SlipWallBC::AssembleFaceGrad3D - 1 tet element", "[SlipWallBC]")
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 3; // templating is hard here because mesh constructors
   int num_state = dim + 2;
   adept::Stack diff_stack;
   double delta = 1e-5;

   int nv = 4, ne = 1, nb = 0, sdim = 3, attrib = 1;
   Mesh mesh(dim, nv, ne, nb, sdim);
   mesh.AddVertex(Vertex(0.,0.,0.)());
   mesh.AddVertex(Vertex(1.,0.,0.)());
   mesh.AddVertex(Vertex(0.,1.,0.)());
   mesh.AddVertex(Vertex(0.,0.,1.)());

   int idx[4] = {0,1,2,3};
   mesh.AddElement(new Tetrahedron(idx,attrib));
   mesh.FinalizeTetMesh(1,1,true); 

   for (int p = 0; p <= 1; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
             new SBPCollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             &mesh, fec.get(), num_state, Ordering::byVDIM));

         NonlinearForm res(fes.get());
         res.AddBdrFaceIntegrator(new miso::SlipWallBC<dim>(diff_stack,
                                                            fec.get()));

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
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)).margin(1e-10));
         }
      }
   }
}

TEST_CASE("PressureForce::AssembleVector", "[PressureForce]")
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 2; // templating is hard here because mesh constructors
   int num_state = dim + 2;
   adept::Stack diff_stack;
   double delta = 1e-5;

   mfem::Vector drag_dir(dim);
   double aoa_fs = 5.0*M_PI/180;
   drag_dir(0) = cos(aoa_fs);
   drag_dir(1) = sin(aoa_fs);

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
            new miso::PressureForce<dim>(diff_stack, fec.get(), drag_dir));

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
            new miso::PressureForce<dim>(diff_stack, fec.get(), drag_dir));

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

TEST_CASE("PressureForce::AssembleVector3D", "[PressureForce]")
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 3; // templating is hard here because mesh constructors
   int num_state = dim + 2;
   adept::Stack diff_stack;
   double delta = 1e-5;

   mfem::Vector drag_dir(dim);
   double aoa_fs = 5.0*M_PI/180;
   drag_dir(0) = cos(aoa_fs);
   drag_dir(1) = sin(aoa_fs);
   drag_dir(2) = 0.0;

   // generate a 2 element mesh
   int num_edge = 2;
   Mesh mesh(Mesh::MakeCartesian3D(num_edge,num_edge,num_edge,
                                   Element::TETRAHEDRON,1.0,1.0,1.0, true));
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
            new miso::PressureForce<dim>(diff_stack, fec.get(), drag_dir));

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


TEST_CASE("PressureForce::AssembleVector3D - 1 tet element", "[PressureForce]")
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 3; // templating is hard here because mesh constructors
   int num_state = dim + 2;
   adept::Stack diff_stack;
   double delta = 1e-5;

   mfem::Vector drag_dir(dim);
   double aoa_fs = 5.0*M_PI/180;
   drag_dir(0) = cos(aoa_fs);
   drag_dir(1) = sin(aoa_fs);
   drag_dir(2) = 0.0;

   int nv = 4, ne = 1, nb = 0, sdim = 3, attrib = 1;
   Mesh mesh(dim, nv, ne, nb, sdim);
   mesh.AddVertex(Vertex(0.,0.,0.)());
   mesh.AddVertex(Vertex(1.,0.,0.)());
   mesh.AddVertex(Vertex(0.,1.,0.)());
   mesh.AddVertex(Vertex(0.,0.,1.)());

   int idx[4] = {0,1,2,3};
   mesh.AddElement(new Tetrahedron(idx,attrib));
   mesh.FinalizeTetMesh(1,1,true); 

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
            new miso::PressureForce<dim>(diff_stack, fec.get(), drag_dir));

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

TEMPLATE_TEST_CASE_SIG("DyadicFluxIntegrator::AssembleElementGrad",
                       "[DyadicIntegrator]",
                       ((bool entvar), entvar), false, true)
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 2; // templating is hard here because mesh constructors
   int num_state = dim + 2;
   adept::Stack diff_stack;
   double delta = 1e-5;

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

         NonlinearForm res(fes.get());
         res.AddDomainIntegrator(
            new miso::IsmailRoeIntegrator<2,entvar>(diff_stack));

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselineVectorPert<2,entvar>);
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
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)).margin(1e-10));
         }
      }

      DYNAMIC_SECTION("(DSBP)...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
             new DSBPCollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             &mesh, fec.get(), num_state, Ordering::byVDIM));

         NonlinearForm res(fes.get());
         res.AddDomainIntegrator(new miso::IsmailRoeIntegrator<2, entvar>(diff_stack));

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselineVectorPert<2, entvar>);
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
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)).margin(1e-10));
         }
      }
   }
}

TEMPLATE_TEST_CASE_SIG("DyadicFluxIntegrator::AssembleElementGrad3D",
                       "[DyadicIntegrator]",
                       ((bool entvar), entvar), false, true)
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 3; // templating is hard here because mesh constructors
   int num_state = dim + 2;
   adept::Stack diff_stack;
   double delta = 1e-5;

   // generate a 2 element mesh
   int num_edge = 5;
   Mesh mesh(Mesh::MakeCartesian3D(num_edge,num_edge,num_edge,
                                   Element::TETRAHEDRON,1.0,1.0,1.0, true));
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
            new miso::IsmailRoeIntegrator<dim,entvar>(diff_stack));

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselineVectorPert<dim,entvar>);
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
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)).margin(1e-10));
         }
      }

   }
}


TEMPLATE_TEST_CASE_SIG("DyadicFluxIntegrator::AssembleElementGrad3D - 1 tet element",
                       "[DyadicIntegrator]",
                       ((bool entvar), entvar), false, true)
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 3; // templating is hard here because mesh constructors
   int num_state = dim + 2;
   adept::Stack diff_stack;
   double delta = 1e-5;

   int nv = 4, ne = 1, nb = 0, sdim = 3, attrib = 1;
   Mesh mesh(dim, nv, ne, nb, sdim);
   mesh.AddVertex(Vertex(0.,0.,0.)());
   mesh.AddVertex(Vertex(1.,0.,0.)());
   mesh.AddVertex(Vertex(0.,1.,0.)());
   mesh.AddVertex(Vertex(0.,0.,1.)());

   int idx[4] = {0,1,2,3};
   mesh.AddElement(new Tetrahedron(idx,attrib));
   mesh.FinalizeTetMesh(1,1,true); 

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
            new miso::IsmailRoeIntegrator<dim,entvar>(diff_stack));

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselineVectorPert<dim,entvar>);
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
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)).margin(1e-10));
         }
      }

   }
}

// JEH: This is redundant, as far as I can tell.
// TODO: add dim = 1, 3 once 3d sbp operators implemented
// TEST_CASE("InviscidFaceIntegrator::AssembleFaceGrad", "[InterfaceIntegrator]")
// {
//    using namespace euler_data;
//    using namespace mfem;
//    const int dim = 2;
//    double delta = 1e-5;
//    int num_state = dim + 2;
//    adept::Stack diff_stack;
//    double diss_coeff = 1.0;

//    // generate a 2 element mesh
//    int num_edge = 2;
//    Mesh mesh(Mesh::MakeCartesian2D(num_edge, num_edge, Element::TRIANGLE,
//                                    true /* gen. edges */, 1.0, 1.0, true));

//    const int max_degree = 4;
//    for (int p = 0; p <= max_degree; p++)
//    {
//       DYNAMIC_SECTION("Jacobian of Interface flux w.r.t state is correct (DSBP)" << p)
//       {
//          std::unique_ptr<FiniteElementCollection> fec(
//              new DSBPCollection(p, dim));
//          std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
//              &mesh, fec.get(), num_state, Ordering::byVDIM));

//          NonlinearForm res(fes.get());
//          res.AddInteriorFaceIntegrator(new miso::InterfaceIntegrator<dim>(
//              diff_stack, diss_coeff, fec.get()));

//          // initialize state; here we randomly perturb a constant state
//          GridFunction q(fes.get());
//          VectorFunctionCoefficient pert(num_state, randBaselineVectorPert<dim>);
//          q.ProjectCoefficient(pert);

//          // initialize the vector that the Jacobian multiplies
//          GridFunction v(fes.get());
//          VectorFunctionCoefficient v_rand(num_state, randVectorState);
//          v.ProjectCoefficient(v_rand);

//          // evaluate the Jacobian and compute its product with v
//          Operator &Jac = res.GetGradient(q);
//          GridFunction jac_v(fes.get());
//          Jac.Mult(v, jac_v);

//          // now compute the finite-difference approximation...
//          GridFunction q_pert(q), r(fes.get()), jac_v_fd(fes.get());
//          q_pert.Add(-delta, v);
//          res.Mult(q_pert, r);
//          q_pert.Add(2 * delta, v);
//          res.Mult(q_pert, jac_v_fd);
//          jac_v_fd -= r;
//          jac_v_fd /= (2 * delta);

//          for (int i = 0; i < jac_v.Size(); ++i)
//          {
//             REQUIRE(jac_v(i) == Approx(jac_v_fd(i)));
//          }
//       }
//    } // loop different order of elements
// }

TEMPLATE_TEST_CASE_SIG("EntStableLPSIntegrator::AssembleElementGrad using entvar",
                       "[LPSIntegrator]",
                       ((bool entvar), entvar), false, true)
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 2; // templating is hard here because mesh constructors
   int num_state = dim + 2;
   adept::Stack diff_stack;
   double delta = 1e-5;

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

         NonlinearForm res(fes.get());
         res.AddDomainIntegrator(
             new miso::EntStableLPSIntegrator<2, entvar>(diff_stack));

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselineVectorPert<2, entvar>);
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
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)).margin(1e-10));
         }
      }

      DYNAMIC_SECTION("(DSBP)...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
             new DSBPCollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             &mesh, fec.get(), num_state, Ordering::byVDIM));

         NonlinearForm res(fes.get());
         res.AddDomainIntegrator(new miso::EntStableLPSIntegrator<2, entvar>(diff_stack));

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselineVectorPert<2, entvar>);
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
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)).margin(1e-10));
         }
      }
   }
}

TEMPLATE_TEST_CASE_SIG("EntStableLPSIntegrator::AssembleElementGrad3D using entvar",
                       "[LPSIntegrator]",
                       ((bool entvar), entvar), false, true)
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 3; // templating is hard here because mesh constructors
   int num_state = dim + 2;
   adept::Stack diff_stack;
   double delta = 1e-5;

   // generate a 2 element mesh
   int num_edge = 2;
   Mesh mesh(Mesh::MakeCartesian3D(num_edge,num_edge,num_edge,
                                    Element::TETRAHEDRON,1.0,1.0,1.0, true));
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
             new miso::EntStableLPSIntegrator<dim, entvar>(diff_stack));

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselineVectorPert<dim, entvar>);
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
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)).margin(1e-10));
         }
      }
   }
}

TEMPLATE_TEST_CASE_SIG("EntStableLPSIntegrator::AssembleElementGrad3D(1 tet element) using entvar",
                       "[LPSIntegrator]",
                       ((bool entvar), entvar), false, true)
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 3; // templating is hard here because mesh constructors
   int num_state = dim + 2;
   adept::Stack diff_stack;
   double delta = 1e-5;

   int nv = 4, ne = 1, nb = 0, sdim = 3, attrib = 1;
   Mesh mesh(dim, nv, ne, nb, sdim);
   mesh.AddVertex(Vertex(0.,0.,0.)());
   mesh.AddVertex(Vertex(1.,0.,0.)());
   mesh.AddVertex(Vertex(0.,1.,0.)());
   mesh.AddVertex(Vertex(0.,0.,1.)());

   int idx[4] = {0,1,2,3};
   mesh.AddElement(new Tetrahedron(idx,attrib));
   mesh.FinalizeTetMesh(1,1,true); 

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
             new miso::EntStableLPSIntegrator<dim, entvar>(diff_stack));

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselineVectorPert<dim, entvar>);
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
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)).margin(1e-10));
         }
      }
   }
}

TEMPLATE_TEST_CASE_SIG("MassIntegrator::AssembleElementGrad",
                       "[MassIntegrator]", ((bool entvar), entvar), false, true)
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 2; // templating is hard here because mesh constructors
   int num_state = dim + 2;
   adept::Stack diff_stack;
   double delta = 1e-5;

   // generate an 8 element mesh
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

         // initialize state; we randomly perturb a constant state
         GridFunction u(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselineVectorPert<2, entvar>);
         u.ProjectCoefficient(pert);

         // initialize the vector that the Jacobian multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(num_state, randVectorState);
         v.ProjectCoefficient(v_rand);

         NonlinearForm res(fes.get());
         res.AddDomainIntegrator(
             new miso::MassIntegrator<2, entvar>(diff_stack));

         // evaluate the Jacobian and compute its product with v
         Operator &Jac = res.GetGradient(u);
         GridFunction jac_v(fes.get());
         Jac.Mult(v, jac_v);

         // now compute the finite-difference approximation...
         GridFunction u_pert(u), r(fes.get()), jac_v_fd(fes.get());
         u_pert.Add(-delta, v);
         res.Mult(u_pert, r);
         u_pert.Add(2 * delta, v);
         res.Mult(u_pert, jac_v_fd);
         jac_v_fd -= r;
         jac_v_fd /= (2 * delta);

         for (int i = 0; i < jac_v.Size(); ++i)
         {
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)).margin(1e-10));
         }
      }

      DYNAMIC_SECTION("(DSBP)...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
             new DSBPCollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             &mesh, fec.get(), num_state, Ordering::byVDIM));

         // initialize state and k = du/dt; here we randomly perturb a constant state
         GridFunction u(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselineVectorPert<2, entvar>);
         u.ProjectCoefficient(pert);

         // initialize the vector that the Jacobian multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(num_state, randVectorState);
         v.ProjectCoefficient(v_rand);

         NonlinearForm res(fes.get());
         res.AddDomainIntegrator(
             new miso::MassIntegrator<2, entvar>(diff_stack));

         // evaluate the Jacobian and compute its product with v
         Operator &Jac = res.GetGradient(u);
         GridFunction jac_v(fes.get());
         Jac.Mult(v, jac_v);

         // now compute the finite-difference approximation...
         GridFunction u_pert(u), r(fes.get()), jac_v_fd(fes.get());
         u_pert.Add(-delta, v);
         res.Mult(u_pert, r);
         u_pert.Add(2 * delta, v);
         res.Mult(u_pert, jac_v_fd);
         jac_v_fd -= r;
         jac_v_fd /= (2 * delta);

         for (int i = 0; i < jac_v.Size(); ++i)
         {
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)).margin(1e-10));
         }
      }
   }
}

TEMPLATE_TEST_CASE_SIG("MassIntegrator::AssembleElementGrad3D",
                       "[MassIntegrator]", ((bool entvar), entvar), false, true)
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 3; // templating is hard here because mesh constructors
   int num_state = dim + 2;
   adept::Stack diff_stack;
   double delta = 1e-5;

   // generate an 8 element mesh
   int num_edge = 5;
   Mesh mesh(Mesh::MakeCartesian3D(num_edge,num_edge,num_edge,
                                    Element::TETRAHEDRON,1.0,1.0,1.0, true));
   for (int p = 0; p <= 1; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
             new SBPCollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             &mesh, fec.get(), num_state, Ordering::byVDIM));

         // initialize state; we randomly perturb a constant state
         GridFunction u(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselineVectorPert<dim, entvar>);
         u.ProjectCoefficient(pert);

         // initialize the vector that the Jacobian multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(num_state, randVectorState);
         v.ProjectCoefficient(v_rand);

         NonlinearForm res(fes.get());
         res.AddDomainIntegrator(
             new miso::MassIntegrator<dim, entvar>(diff_stack));

         // evaluate the Jacobian and compute its product with v
         Operator &Jac = res.GetGradient(u);
         GridFunction jac_v(fes.get());
         Jac.Mult(v, jac_v);

         // now compute the finite-difference approximation...
         GridFunction u_pert(u), r(fes.get()), jac_v_fd(fes.get());
         u_pert.Add(-delta, v);
         res.Mult(u_pert, r);
         u_pert.Add(2 * delta, v);
         res.Mult(u_pert, jac_v_fd);
         jac_v_fd -= r;
         jac_v_fd /= (2 * delta);

         for (int i = 0; i < jac_v.Size(); ++i)
         {
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)).margin(1e-10));
         }
      }
   }
}


TEMPLATE_TEST_CASE_SIG("MassIntegrator::AssembleElementGrad3D - 1 tet element",
                       "[MassIntegrator]", ((bool entvar), entvar), false, true)
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 3; // templating is hard here because mesh constructors
   int num_state = dim + 2;
   adept::Stack diff_stack;
   double delta = 1e-5;

   int nv = 4, ne = 1, nb = 0, sdim = 3, attrib = 1;
   Mesh mesh(dim, nv, ne, nb, sdim);
   mesh.AddVertex(Vertex(0.,0.,0.)());
   mesh.AddVertex(Vertex(1.,0.,0.)());
   mesh.AddVertex(Vertex(0.,1.,0.)());
   mesh.AddVertex(Vertex(0.,0.,1.)());

   int idx[4] = {0,1,2,3};
   mesh.AddElement(new Tetrahedron(idx,attrib));
   mesh.FinalizeTetMesh(1,1,true); 

   for (int p = 0; p <= 1; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
             new SBPCollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             &mesh, fec.get(), num_state, Ordering::byVDIM));

         // initialize state; we randomly perturb a constant state
         GridFunction u(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselineVectorPert<dim, entvar>);
         u.ProjectCoefficient(pert);

         // initialize the vector that the Jacobian multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(num_state, randVectorState);
         v.ProjectCoefficient(v_rand);

         NonlinearForm res(fes.get());
         res.AddDomainIntegrator(
             new miso::MassIntegrator<dim, entvar>(diff_stack));

         // evaluate the Jacobian and compute its product with v
         Operator &Jac = res.GetGradient(u);
         GridFunction jac_v(fes.get());
         Jac.Mult(v, jac_v);

         // now compute the finite-difference approximation...
         GridFunction u_pert(u), r(fes.get()), jac_v_fd(fes.get());
         u_pert.Add(-delta, v);
         res.Mult(u_pert, r);
         u_pert.Add(2 * delta, v);
         res.Mult(u_pert, jac_v_fd);
         jac_v_fd -= r;
         jac_v_fd /= (2 * delta);

         for (int i = 0; i < jac_v.Size(); ++i)
         {
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)).margin(1e-10));
         }
      }
   }
}

// TODO: add dim = 1, 3 once 3d sbp operators implemented
TEMPLATE_TEST_CASE_SIG("InviscidFaceIntegrator::AssembleFaceGrad", 
                        "[InterfaceIntegrator]",
                       ((bool entvar), entvar), false, true)
{
   using namespace euler_data;
   using namespace mfem;
   const int dim = 2;
   double delta = 1e-5;
   int num_state = dim + 2;
   adept::Stack diff_stack;
   double diss_coeff = 1.0;

   // generate a 2 element mesh
   int num_edge = 2;
   Mesh mesh(Mesh::MakeCartesian2D(num_edge, num_edge, Element::TRIANGLE,
                                   true /* gen. edges */, 1.0, 1.0, true));

   const int max_degree = 4;
   for (int p = 0; p <= max_degree; p++)
   {
      DYNAMIC_SECTION("Jacobian of Interface flux w.r.t state is correct (DSBP)" << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
             new DSBPCollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             &mesh, fec.get(), num_state, Ordering::byVDIM));

         NonlinearForm res(fes.get());
         res.AddInteriorFaceIntegrator(
             new miso::InterfaceIntegrator<dim, entvar>(diff_stack, diss_coeff,
                                                        fec.get()));

         // initialize state; here we randomly perturb a constant state
         GridFunction w(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselineVectorPert<dim, entvar>);
         w.ProjectCoefficient(pert);

         // initialize the vector that the Jacobian multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(num_state, randVectorState);
         v.ProjectCoefficient(v_rand);

         // evaluate the Jacobian and compute its product with v
         Operator &Jac = res.GetGradient(w);
         GridFunction jac_v(fes.get());
         Jac.Mult(v, jac_v);

         // now compute the finite-difference approximation...
         GridFunction w_pert(w), r(fes.get()), jac_v_fd(fes.get());
         w_pert.Add(-delta, v);
         res.Mult(w_pert, r);
         w_pert.Add(2 * delta, v);
         res.Mult(w_pert, jac_v_fd);
         jac_v_fd -= r;
         jac_v_fd /= (2 * delta);

         for (int i = 0; i < jac_v.Size(); ++i)
         {
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)).margin(1e-10));
            //std::cout << "jac_v(" << i << ") = " << jac_v(i) << std::endl;
            //std::cout << "jac_v_fd = " << jac_v_fd(i) << std::endl;
         }
      }
   } // loop different order of elements
}

TEMPLATE_TEST_CASE_SIG("InviscidFaceIntegrator::AssembleFaceGrad3D", 
                        "[InterfaceIntegrator]",
                       ((bool entvar), entvar), false, true)
{
   using namespace euler_data;
   using namespace mfem;
   const int dim = 3;
   double delta = 1e-5;
   int num_state = dim + 2;
   adept::Stack diff_stack;
   double diss_coeff = 1.0;

   // generate a 2 element mesh
   int num_edge = 2;
   Mesh mesh(Mesh::MakeCartesian3D(num_edge,num_edge,num_edge,
                                    Element::TETRAHEDRON,1.0,1.0,1.0, true));
   const int max_degree = 1;
   for (int p = 0; p <= max_degree; ++p)
   {
      DYNAMIC_SECTION("Jacobian of Interface flux (3D) w.r.t state is correct (SBP) " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
             new SBPCollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             &mesh, fec.get(), num_state, Ordering::byVDIM));

         NonlinearForm res(fes.get());
         res.AddInteriorFaceIntegrator(
             new miso::InterfaceIntegrator<dim, entvar>(diff_stack, diss_coeff,
                                                        fec.get()));
         
         // initialize state; here we randomly perturb a constant state
         GridFunction w(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselineVectorPert<dim, entvar>);
         w.ProjectCoefficient(pert);
         
         // initialize the vector that the Jacobian multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(num_state, randVectorState);
         v.ProjectCoefficient(v_rand);

         // evaluate the Jacobian and compute its product with v
         Operator &Jac = res.GetGradient(w);
         GridFunction jac_v(fes.get());
         Jac.Mult(v, jac_v);

         // now compute the finite-difference approximation...
         GridFunction w_pert(w), r(fes.get()), jac_v_fd(fes.get());
         w_pert.Add(-delta, v);
         res.Mult(w_pert, r);
         w_pert.Add(2 * delta, v);
         res.Mult(w_pert, jac_v_fd);
         jac_v_fd -= r;
         jac_v_fd /= (2 * delta);

         for (int i = 0; i < jac_v.Size(); ++i)
         {
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)).margin(1e-10));
            //std::cout << "jac_v(" << i << ") = " << jac_v(i) << std::endl;
            //std::cout << "jac_v_fd = " << jac_v_fd(i) << std::endl;
         }
      }
   } // loop different order of elements
}

TEMPLATE_TEST_CASE_SIG("InviscidFaceIntegrator::AssembleFaceGrad3D - 1 tet element", 
                        "[InterfaceIntegrator]",
                       ((bool entvar), entvar), false, true)
{
   using namespace euler_data;
   using namespace mfem;
   const int dim = 3;
   double delta = 1e-5;
   int num_state = dim + 2;
   adept::Stack diff_stack;
   double diss_coeff = 1.0;

   int nv = 4, ne = 1, nb = 0, sdim = 3, attrib = 1;
   Mesh mesh(dim, nv, ne, nb, sdim);
   mesh.AddVertex(Vertex(0.,0.,0.)());
   mesh.AddVertex(Vertex(1.,0.,0.)());
   mesh.AddVertex(Vertex(0.,1.,0.)());
   mesh.AddVertex(Vertex(0.,0.,1.)());

   int idx[4] = {0,1,2,3};
   mesh.AddElement(new Tetrahedron(idx,attrib));
   mesh.FinalizeTetMesh(1,1,true); 

   const int max_degree = 1;
   for (int p = 0; p <= max_degree; ++p)
   {
      DYNAMIC_SECTION("Jacobian of Interface flux (3D) w.r.t state is correct (SBP) " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
             new SBPCollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             &mesh, fec.get(), num_state, Ordering::byVDIM));

         NonlinearForm res(fes.get());
         res.AddInteriorFaceIntegrator(
             new miso::InterfaceIntegrator<dim, entvar>(diff_stack, diss_coeff,
                                                        fec.get()));
         
         // initialize state; here we randomly perturb a constant state
         GridFunction w(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselineVectorPert<dim, entvar>);
         w.ProjectCoefficient(pert);
         
         // initialize the vector that the Jacobian multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(num_state, randVectorState);
         v.ProjectCoefficient(v_rand);

         // evaluate the Jacobian and compute its product with v
         Operator &Jac = res.GetGradient(w);
         GridFunction jac_v(fes.get());
         Jac.Mult(v, jac_v);

         // now compute the finite-difference approximation...
         GridFunction w_pert(w), r(fes.get()), jac_v_fd(fes.get());
         w_pert.Add(-delta, v);
         res.Mult(w_pert, r);
         w_pert.Add(2 * delta, v);
         res.Mult(w_pert, jac_v_fd);
         jac_v_fd -= r;
         jac_v_fd /= (2 * delta);

         for (int i = 0; i < jac_v.Size(); ++i)
         {
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)).margin(1e-10));
            //std::cout << "jac_v(" << i << ") = " << jac_v(i) << std::endl;
            //std::cout << "jac_v_fd = " << jac_v_fd(i) << std::endl;
         }
      }
   } // loop different order of elements
}