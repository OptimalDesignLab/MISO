#include "advection.hpp"
#include "sbp_fe.hpp"
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
   DenseMatrix vel, velhat, adjJ, Q;
   Vector vel_i, velhat_i, Udi;
#endif
   elmat.SetSize(num_nodes);
   velhat.SetSize(dim, num_nodes); // scaled velocity in reference space
   adjJ.SetSize(dim); // adjJ = |J|*dxi/dx = adj(dx/dxi)
   Vector vel_i; // reference to vel at a node
   Vector velhat_i; // reference to velhat at a node
   Udi.SetSize(num_nodes); // reference to one component of velhat at all nodes

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
      static_cast<const SBPTriangleElement&>(el).getWeakOperator(di, Q, true);
      velhat.GetRow(di, Udi);
      Q.RightScaling(Udi); // This makes Q_{di}^T * diag(Udi)
      elmat.Add(-alpha, Q); // minus sign accounts for integration by parts
   }
}


LPSIntegrator::LPSIntegrator(
    VectorCoefficient &velc, double a, double diss_coeff)
    : vel_coeff(velc)
{
   alpha = a;
   lps_coeff = diss_coeff;
}

void LPSIntegrator::AssembleElementMatrix(
   const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat)
{
   using SBP = const SBPTriangleElement&;
   int num_nodes = el.GetDof();
   int dim = el.GetDim();
#ifdef MFEM_THREAD_SAFE
   DenseMatrix vel, adjJ, P;
   Vector velhat_i, AH;
#endif
   elmat.SetSize(num_nodes);
   adjJ.SetSize(dim); // adjJ = |J|*dxi/dx = adj(dx/dxi)
   P.SetSize(num_nodes); // LPS projection operator
   velhat_i.SetSize(dim); // scaled velocity in reference space at a node
   AH.SetSize(num_nodes); // the scaling matrix for LPS

   // Get the scaling matrix, A, times the quadrature weights, H
   vel_coeff.Eval(vel, Trans, el.GetNodes());
   Vector vel_i; // reference to vel at a node
   const Vector &H = static_cast<SBP>(el).returnDiagNorm();
   for (int i = 0; i < num_nodes; i++)
   {
      vel.GetColumnReference(i, vel_i);
      CalcAdjugate(Trans.Jacobian(), adjJ);
      adjJ.Mult(vel_i, velhat_i);
      AH(i) = alpha*lps_coeff*H(i)*velhat_i.Norml2();
   }
   // Get the projection operator, construct LPS, and add to element matrix
   static_cast<SBP>(el).getLocalProjOperator(P);
   P.Transpose();
   MultADAt(P, AH, elmat);
}

AdvectionSolver::AdvectionSolver(const string &opt_file_name,
                                 void (*vel_field)(const Vector &, Vector &))
   : AbstractSolver(opt_file_name)
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

   // set up the LPS stabilization
   double lps_coeff = options["lps-coeff"].get<double>();
   static_cast<BilinearForm*>(res.get())->AddDomainIntegrator(
      new LPSIntegrator(*velocity, -1.0, lps_coeff));

   int skip_zeros = 0;
   static_cast<BilinearForm*>(res.get())->Assemble(skip_zeros);
   static_cast<BilinearForm*>(res.get())->Finalize(skip_zeros);

   // define the time-dependent operator
   evolver.reset(new LinearEvolver(mass->SpMat(),
                 static_cast<BilinearForm*>(res.get())->SpMat()));
}

}