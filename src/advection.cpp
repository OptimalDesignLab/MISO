#include "advection.hpp"

using namespace mfem;
using namespace std;

namespace mach
{

// TODO: this is just a copy of ConvectionIntegrator; need to specialize still
//
// We may need our own ElementTransformation, or something, since it makes more
// sense for us to precompute the transformation Jacobian, especially in 3D.
// Alternatively, we can pass a reference to the Jacobian (at each node) into
// the constructor for the Integrator?  But it won't know which element...
// Look at using IsoparametricTransformation as a model.  Note, we can use the
// same projection operators needed for LPS to project the inverse Jacobian onto
// the desired space.
void AdvectionIntegrator::AssembleElementMatrix(
   const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat)
{
   int nd = el.GetDof();
   int dim = el.GetDim();

#ifdef MFEM_THREAD_SAFE
   DenseMatrix dshape, adjJ, Q_ir;
   Vector shape, vec2, BdFidxT;
#endif
   elmat.SetSize(nd);
   dshape.SetSize(nd,dim);
   adjJ.SetSize(dim);
   shape.SetSize(nd);
   vec2.SetSize(dim);
   BdFidxT.SetSize(nd);

   Vector vec1;

   const IntegrationRule *ir = &el.GetNodes();

   // Evaluate the velocity at the integration points; Q_ir is dim x num_nodes
   Q.Eval(Q_ir, Trans, *ir);

   elmat = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      el.CalcDShape(ip, dshape);
      el.CalcShape(ip, shape);

      Trans.SetIntPoint(&ip);
      CalcAdjugate(Trans.Jacobian(), adjJ);
      Q_ir.GetColumnReference(i, vec1);
      vec1 *= alpha * ip.weight;

      adjJ.Mult(vec1, vec2);
      dshape.Mult(vec2, BdFidxT);

      AddMultVWt(shape, BdFidxT, elmat);
   }
}

AdvectionSolver::AdvectionSolver(OptionsParser &args,
                                 void (*vel_field)(const Vector &, Vector &))
   : AbstractSolver(args)
{
   // set the finite-element space and create (but do not initialize) the
   // state GridFunction
   num_state = 1;
   fes.reset(new FiniteElementSpace(mesh.get(), fec.get()));  // TODO: handle parallel case
   u.reset(new GridFunction(fes.get()));
   cout << "Number of finite element unknowns: "
        << fes->GetTrueVSize() << endl;

   // set up the mass matrix
   mass.reset(new BilinearForm(fes.get()));
   mass->AddDomainIntegrator(new MassIntegrator);
   mass->Assemble();
   mass->Finalize();

   // set up the stiffness matrix
   velocity.reset(new VectorFunctionCoefficient(mesh->Dimension(), vel_field));
   res.reset(new BilinearForm(fes.get()));
   static_cast<BilinearForm*>(res.get())->AddDomainIntegrator(
      new AdvectionIntegrator(*velocity, -1.0));
   // TODO: need to add an integrator for LPS 

   int skip_zeros = 0;
   static_cast<BilinearForm*>(res.get())->Assemble(skip_zeros);
   static_cast<BilinearForm*>(res.get())->Finalize(skip_zeros);

}

}