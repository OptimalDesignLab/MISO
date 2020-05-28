#include <random>

#include "catch.hpp"
#include "adept.h"
#include "mfem.hpp"
#include "euler_integ.hpp"
#include "euler_sens_integ.hpp"
#include "euler_test_data.hpp"
#include "euler_fluxes.hpp"

TEMPLATE_TEST_CASE_SIG("FarFieldBCDiff::AssembleRHSElementVect",
                       "[FarFieldBCDiff]",
                       ((bool entvar), entvar), false, true)
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 2; // templating is hard here because mesh constructors
   const int num_state = dim + 2;
   double delta = 1e-5;

   // generate a 8 element mesh
   int num_edge = 2;
   std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, Element::TRIANGLE,
                                       true /* gen. edges */, 1.0, 1.0, true));
    adept::Stack diff_stack;
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         // get the finite-element space for the state and adjoint
         std::unique_ptr<FiniteElementCollection> fec(
             new mfem::SBPCollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             mesh.get(), fec.get(), num_state,
                   Ordering::byVDIM));
        
        // generate far field conditions
        double mach = 1.0; double aoa = 3.0*(M_PI/180);
        mfem::Vector q_far(4); //mfem::Vector w_far(4);
        FreeStreamState2D(q_far, mach, aoa);
        // if (entvar)
        // {
        //     calcConservativeVars<xdouble, dim>(q_far.GetData(), w_far.GetData());
        // }
        // else
        // {
        //     w_far = q_far;
        // }

        // we use res for finite-difference approximation
        NonlinearForm res1(fes.get());
        NonlinearForm res2(fes.get());

         // initialize state and adjoint
         GridFunction state(fes.get()), adjoint(fes.get());
         VectorFunctionCoefficient pert(dim+2, randState);
         state.ProjectCoefficient(pert);
         adjoint.ProjectCoefficient(pert);

         // build the nonlinear form for d(psi^T R)/dx 
         LinearForm dfdx_form(fes.get());
         dfdx_form.AddBdrFaceIntegrator(
            new mach::FarFieldBCDiff<dim, entvar>(diff_stack, state, adjoint,
                        q_far, mach, aoa));

         // evaluate d(psi^T R)/dmach
         dfdx_form.Assemble();
         double dfdx = adjoint*dfdx_form;

         // now compute the finite-difference approximation...
         GridFunction dfdx_vect(fes.get());
         mach += delta;
         FreeStreamState2D(q_far, mach, aoa);
         res1.AddBdrFaceIntegrator(
            new mach::FarFieldBC<dim, entvar>(diff_stack, fec.get(), q_far));
        res1.Mult(state, dfdx_vect);
         double dfdx_fd = dfdx_vect*adjoint;
         mach -= 2*delta;
         FreeStreamState2D(q_far, mach, aoa);
         res2.AddBdrFaceIntegrator(
            new mach::FarFieldBC<dim, entvar>(diff_stack, fec.get(), q_far));
        res2.Mult(state, dfdx_vect);
         dfdx_fd -= dfdx_vect*adjoint;
         dfdx_fd /= (2 * delta);

         REQUIRE(dfdx == Approx(dfdx_fd).margin(1e-10));
      }
   }
}

TEMPLATE_TEST_CASE_SIG("PressureForceDiff::GetEnergy",
                       "[PressureForceDiff]",
                       ((bool entvar), entvar), false, true)
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 2; // templating is hard here because mesh constructors
   const int num_state = dim + 2;
   double delta = 1e-5;

   // generate a 8 element mesh
   int num_edge = 2;
   std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, Element::TRIANGLE,
                                       true /* gen. edges */, 1.0, 1.0, true));
    adept::Stack diff_stack;
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         // get the finite-element space for the state and adjoint
         std::unique_ptr<FiniteElementCollection> fec(
             new mfem::SBPCollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             mesh.get(), fec.get(), num_state,
                   Ordering::byVDIM));
        
        // generate drag direction
        double mach = 1.0; double aoa = 3.0*(M_PI/180);
        mfem::Vector drag_dir(dim); //mfem::Vector w_far(4);
        drag_dir = 0.0;
         drag_dir(0) = cos(aoa);
         drag_dir(1) = sin(aoa);
         drag_dir *= 1.0/pow(mach, 2.0);

        // we use res for finite-difference approximation
        NonlinearForm res1(fes.get());
        NonlinearForm res2(fes.get());

         // initialize state and adjoint
         GridFunction state(fes.get()), adjoint(fes.get());
         VectorFunctionCoefficient pert(dim+2, randState);
         state.ProjectCoefficient(pert);
         adjoint.ProjectCoefficient(pert);

         // build the nonlinear form for df/dmach
         NonlinearForm dfdx_form(fes.get());
         dfdx_form.AddBdrFaceIntegrator(
            new mach::PressureForceDiff<dim, entvar>(diff_stack, state, adjoint,
                        drag_dir, mach, aoa));

         // evaluate df/dmach
         double dfdx = dfdx_form.GetEnergy(state);

         // now compute the finite-difference approximation...
         GridFunction dfdx_vect(fes.get());
         mach += delta;
         drag_dir = 0.0;
         drag_dir(0) = cos(aoa);
         drag_dir(1) = sin(aoa);
         drag_dir *= 1.0/pow(mach, 2.0);
         res1.AddBdrFaceIntegrator(
            new mach::PressureForce<dim, entvar>(diff_stack, fec.get(), drag_dir));
         double dfdx_fd = res1.GetEnergy(state);
         mach -= 2*delta;
         drag_dir = 0.0;
         drag_dir(0) = cos(aoa);
         drag_dir(1) = sin(aoa);
         drag_dir *= 1.0/pow(mach, 2.0);
         res2.AddBdrFaceIntegrator(
            new mach::PressureForce<dim, entvar>(diff_stack, fec.get(), drag_dir));
         dfdx_fd -= res2.GetEnergy(state);
         dfdx_fd /= (2 * delta);

         REQUIRE(dfdx == Approx(dfdx_fd).margin(1e-10));
      }
   }
}