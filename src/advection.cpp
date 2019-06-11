#include "advection.hpp"
#include "diag_mass_integ.hpp"
#include "linear_evolver.hpp"

using namespace mfem;
using namespace std;

namespace mach
{

void AdvectionIntegrator::AssembleElementMatrix(
   const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat)
{
   int num_nodes = el.GetDof();
   int dim = el.GetDim();
#ifdef MFEM_THREAD_SAFE
   DenseMatrix vel, velhat, adjJ, D;
   Vector vel_i, velhat_i, Udi, H;
#endif
   elmat.SetSize(num_nodes);
   velhat.SetSize(dim, num_nodes); // scaled velocity in reference space
   adjJ.SetSize(dim); // adjJ = |J|*dxi/dx = adj(dx/dxi)
   Vector vel_i; // reference to vel at a node
   Vector velhat_i; // reference to velhat at a node
   Udi.SetSize(num_nodes); // reference to one component of velhat at all nodes
   static_cast<const SBP_TriangleElement&>(el).GetDiagNorm(H); // extract norm

   // Evaluate the velocity at the nodes and get the velocity in reference space
   // vel and velhat are dim x num_nodes
   vel_coeff.Eval(vel, Trans, el.GetNodes());
   for (int i = 0; i < num_nodes; i++)
   {
      vel.GetColumnReference(i, vel_i);
      velhat.GetColumnReference(i, velhat_i);
      CalcAdjugate(Trans.Jacobian(), adjJ);
      adjJ.Mult(vel_i, velhat_i);
   }
   // loop over the reference space dimensions and sum (transposed) operators
   elmat = 0.0;
   for (int di = 0; di < el.GetDim(); di++)
   {
      static_cast<const SBP_TriangleElement&>(el).GetOperator(di, D, true);
      velhat.GetRow(di, Udi);
      D.RightScaling(H); // This makes D_{di}^T = Q_{di}^T
      D.RightScaling(Udi); // This makes Q_{di}^T * diag(Udi)
      elmat.Add(alpha, D);
   }
}

AdvectionSolver::AdvectionSolver(OptionsParser &args,
                                 void (*vel_field)(const Vector &, Vector &))
   : AbstractSolver(args)
{
   // set the finite-element space and create (but do not initialize) the
   // state GridFunction
   num_state = 1;
   //fes.reset(new FiniteElementSpace(mesh.get(), fec.get()));  // TODO: handle parallel case
   fes.reset(new FiniteElementSpace(mesh.get(), fec.get(), num_state, Ordering::byVDIM)); 
   u.reset(new GridFunction(fes.get()));
   cout << "Number of finite element unknowns: "
        << fes->GetTrueVSize() << endl;

   // set up the mass matrix
   mass.reset(new BilinearForm(fes.get()));
   mass->AddDomainIntegrator(new DiagMassIntegrator(num_state));
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

   // define the time-dependent operator
   evolver.reset(new LinearEvolver(mass->SpMat(),
                 static_cast<BilinearForm*>(res.get())->SpMat()));
}

}