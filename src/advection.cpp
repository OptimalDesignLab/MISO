#include "advection.hpp"

namespace mach
{

AdvectionSolver::AdvectionSolver(OptionsParser &args,
                                 void (*vel_field)(const Vector &, Vector &))
   : AbstractSolver(args)
{
   num_state = 1;
   fes = new FiniteElementSpace(mesh, fec);  // TODO: handle parallel case
   cout << "Number of finite element unknowns: "
        << fes->GetTrueVSize() << endl;

   // set up the mass matrix
   mass = new BilinearForm(fes);
   mass->AddDomainIntegrator(new MassIntegrator);

   // set up the stiffness matrix
   velocity = new VectorFunctionCoefficient(mesh->Dimension(), vel_field);
   res = new BilinearForm(fes);
   static_cast<BilinearForm*>(res)->AddDomainIntegrator(
      new ConvectionIntegrator(*velocity, -1.0));
   // TODO: need to add an integrator for LPS 

   mass->Assemble();
   mass->Finalize();

   // Stiffness matrix assembly is not currently working; mfem produces the
   // error "IsoparametricTransformation::OrderGrad(...)"
   int skip_zeros = 0;
#if 0 
   static_cast<BilinearForm*>(res)->Assemble(skip_zeros);
   static_cast<BilinearForm*>(res)->Finalize(skip_zeros);
#endif   
}

}