#include <random>

#include "catch.hpp"
#include "mfem.hpp"
#include "euler_fluxes.hpp"
#include "euler_integ.hpp"
#include "euler_sens_integ.hpp"
#include "euler_test_data.hpp"
#include "navier_stokes_integ.hpp"
#include "navier_stokes_sens_integ.hpp"

TEMPLATE_TEST_CASE_SIG("ESViscousMeshSensIntegrator::AssembleElementVector",
                       "[ESViscousMeshSens]",
                       ((bool entvar), entvar), false)
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 2; // templating is hard here because mesh constructors
   double Re_num = 1;
   double Pr_num = 1;
   double mu = -1;
   int num_state = dim + 2;
   adept::Stack diff_stack; // needed for forward problem
   double delta = 1e-5;

   // generate a 8 element mesh
   int num_edge = 2;
   std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, Element::TRIANGLE,
                                       true /* gen. edges */, 1.0, 1.0, true));
   mesh->EnsureNodes();
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         // get the finite-element space for the state and adjoint
         std::unique_ptr<FiniteElementCollection> fec(
             new SBPCollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             mesh.get(), fec.get(), num_state, Ordering::byVDIM));

         // we use res for finite-difference approximation
         NonlinearForm res(fes.get());
         res.AddDomainIntegrator(
            new mach::ESViscousIntegrator<2>(diff_stack, Re_num, Pr_num, mu));

         // initialize state and adjoint; here we randomly perturb a constant state
         GridFunction state(fes.get()), adjoint(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselineVectorPert<2,entvar>);
         state.ProjectCoefficient(pert);
         adjoint.ProjectCoefficient(pert);

         // extract mesh nodes and get their finite-element space
         GridFunction *x_nodes = mesh->GetNodes();
         FiniteElementSpace *mesh_fes = x_nodes->FESpace();

         // build the linear form for d(psi^T R)/dx 
         LinearForm dfdx(mesh_fes);
         dfdx.AddDomainIntegrator(
            new mach::ESViscousMeshSensIntegrator<2,entvar>(
               state, adjoint, Re_num, Pr_num, mu));
         dfdx.Assemble();

         // initialize the vector that we use to perturb the mesh nodes
         GridFunction v(mesh_fes);
         VectorFunctionCoefficient v_rand(dim, randVectorState);
         v.ProjectCoefficient(v_rand);

         // contract dfdx with v
         double dfdx_v = dfdx * v;

         // now compute the finite-difference approximation...
         GridFunction x_pert(*x_nodes);
         GridFunction r(fes.get());
         x_pert.Add(delta, v);
         mesh->SetNodes(x_pert);
         res.Mult(state, r);
         double dfdx_v_fd = adjoint * r;
         x_pert.Add(-2 * delta, v);
         mesh->SetNodes(x_pert);
         res.Mult(state, r);
         dfdx_v_fd -= adjoint * r;
         dfdx_v_fd /= (2 * delta);
         mesh->SetNodes(*x_nodes); // remember to reset the mesh nodes

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-10));
      }
   }
}

