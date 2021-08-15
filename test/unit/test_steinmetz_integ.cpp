#include <random>

#include "catch.hpp"
#include "mfem.hpp"
#include "coefficient.hpp"
#include "mfem_common_integ.hpp"
#include "electromag_integ.hpp"
#include "euler_test_data.hpp"
#include "electromag_test_data.hpp"

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

TEST_CASE("DomainResIntegrator::AssembleElementVector",
          "[DomainResIntegrator for Steinmetz]")
{
   using namespace mfem;
   using namespace euler_data;
   using namespace mach;

   const int dim = 3; // templating is hard here because mesh constructors
   double delta = 1e-5;

   // generate a 8 element mesh
   int num_edge = 1;
   std::unique_ptr<Mesh> mesh = electromag_data::getMesh(2, 1);
                              //(new Mesh(num_edge, num_edge, num_edge, Element::TETRAHEDRON,
                              //        true /* gen. edges */, 1.0, 1.0, 1.0, true));
   mesh->ReorientTetMesh();
   mesh->EnsureNodes();

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         // get the finite-element space for the state and adjoint
         H1_FECollection fec(p, dim);
         ND_FECollection feca(p, dim);
         FiniteElementSpace fes(mesh.get(), &fec);
         FiniteElementSpace fesa(mesh.get(), &feca);

         // we use res for finite-difference approximation
         GridFunction A(&fesa);
         VectorFunctionCoefficient perta(dim, electromag_data::randVectorState);
         A.ProjectCoefficient(perta);
         // std::unique_ptr<Coefficient> q2(new FunctionCoefficient(func, funcRevDiff));
         std::unique_ptr<Coefficient> q2(new SteinmetzCoefficient(
                        1, 2, 4, 0.5, 0.6, A));
         // std::unique_ptr<mach::MeshDependentCoefficient> Q;
         // Q.reset(new mach::MeshDependentCoefficient());
         // Q->addCoefficient(1, move(q1)); 
         // Q->addCoefficient(2, move(q2));
         LinearForm res(&fes);
         res.AddDomainIntegrator(
            new DomainLFIntegrator(*q2));

         // initialize state and adjoint; here we randomly perturb a constant state
         GridFunction adjoint(&fes);
         FunctionCoefficient pert(electromag_data::randState);
         adjoint.ProjectCoefficient(pert);

         // extract mesh nodes and get their finite-element space
         auto *x_nodes = dynamic_cast<GridFunction*>(mesh->GetNodes());
         auto *mesh_fes = dynamic_cast<FiniteElementSpace*>(x_nodes->FESpace());

         // build the nonlinear form for d(psi^T R)/dx 
         NonlinearForm dfdx_form(mesh_fes);
         dfdx_form.AddDomainIntegrator(
            new mach::DomainResIntegrator(*q2, &adjoint));

         // initialize the vector that we use to perturb the mesh nodes
         GridFunction v(mesh_fes);
         VectorFunctionCoefficient v_rand(dim, electromag_data::randVectorState);
         v.ProjectCoefficient(v_rand);

         // evaluate df/dx and contract with v
         GridFunction dfdx(*x_nodes);
         dfdx_form.Mult(*x_nodes, dfdx);
         double dfdx_v = dfdx * v;

         // now compute the finite-difference approximation...
         GridFunction x_pert(*x_nodes);
         GridFunction r(&fes);
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

TEST_CASE("ThermalSensIntegrator::AssembleElementVector",
          "[ThermalSensIntegrator]")
{
   using namespace mfem;
   using namespace euler_data;
   using namespace mach;

   const int dim = 3; // templating is hard here because mesh constructors
   double delta = 1e-5;

   // generate a 8 element mesh
   int num_edge = 2;
   std::unique_ptr<Mesh> mesh = electromag_data::getMesh();
                              //(new Mesh(num_edge, num_edge, num_edge, Element::TETRAHEDRON,
                              //        true /* gen. edges */, 1.0, 1.0, 1.0, true));
   mesh->EnsureNodes();

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         // get the finite-element space for the state and adjoint
         std::unique_ptr<FiniteElementCollection> fec(
             new H1_FECollection(p, dim));
         std::unique_ptr<FiniteElementCollection> feca(
             new ND_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             mesh.get(), fec.get()));
         std::unique_ptr<FiniteElementSpace> fesa(new FiniteElementSpace(
             mesh.get(), feca.get()));

         // we use res for finite-difference approximation
         GridFunction A(fesa.get());
         VectorFunctionCoefficient perta(dim, electromag_data::randVectorState);
         A.ProjectCoefficient(perta);
         std::unique_ptr<Coefficient> q1(new ConstantCoefficient(1));
         std::unique_ptr<Coefficient> q2(new SteinmetzCoefficient(
                        1, 2, 4, 0.5, 0.6, A));
         std::unique_ptr<mach::MeshDependentCoefficient> Q;
         // Q.reset(new mach::MeshDependentCoefficient());
         // Q->addCoefficient(1, move(q1)); 
         // Q->addCoefficient(2, move(q2));
         std::unique_ptr<VectorCoefficient> QV(
            new SteinmetzVectorDiffCoefficient(1, 2, 4, 0.5, 0.6, A));
         LinearForm res(fes.get());
         res.AddDomainIntegrator(
            new DomainLFIntegrator(*q2));

         // initialize state and adjoint; here we randomly perturb a constant state
         GridFunction state(fes.get()), adjoint(fes.get());
         FunctionCoefficient pert(electromag_data::randState);
         state.ProjectCoefficient(pert);
         adjoint.ProjectCoefficient(pert);


         // build the nonlinear form for d(psi^T R)/dx 
         LinearForm dfdx_form(fesa.get());
         dfdx_form.AddDomainIntegrator(
            new mach::ThermalSensIntegrator(*QV, &adjoint));

         // initialize the vector that we use to perturb the vector potential
         GridFunction v(fesa.get());
         VectorFunctionCoefficient v_rand(dim, electromag_data::randVectorState);
         v.ProjectCoefficient(v_rand);

         // evaluate df/dx and contract with v
         //GridFunction dfdx(*x_nodes);
         dfdx_form.Assemble();
         double dfdx_v = dfdx_form * v;

         // now compute the finite-difference approximation...
         //GridFunction a_pert(A);
         A.Add(delta, v);
         res.Assemble();
         double dfdx_v_fd = adjoint * res;
         A.Add(-2 * delta, v);
         res.Assemble();
         dfdx_v_fd -= adjoint * res;
         dfdx_v_fd /= (2 * delta);
         A.Add(delta, v);

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-10));
      }
   }
}