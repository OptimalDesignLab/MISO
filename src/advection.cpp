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
   using SBP = const SBPTriangleElement&;
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
      Trans.SetIntPoint(&el.GetNodes().IntPoint(i));
      CalcAdjugate(Trans.Jacobian(), adjJ);
      adjJ.Mult(vel_i, velhat_i);
   }
   // loop over the reference space dimensions and sum (transposed) operators
   elmat = 0.0;
   for (int di = 0; di < el.GetDim(); di++)
   {
      static_cast<SBP>(el).getWeakOperator(di, Q, true);
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

   // Get AH: the scaling matrix, A, times the quadrature weights, H
   vel_coeff.Eval(vel, Trans, el.GetNodes());
   Vector vel_i; // reference to vel at a node
   const Vector &H = static_cast<SBP>(el).returnDiagNorm();
   for (int i = 0; i < num_nodes; i++)
   {
      vel.GetColumnReference(i, vel_i);
      Trans.SetIntPoint(&el.GetNodes().IntPoint(i));
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
   fes.reset(new SpaceType(static_cast<MeshType*>(mesh.get()), fec.get(), num_state, Ordering::byVDIM)); 
   u.reset(new GridFunType(static_cast<SpaceType*>(fes.get())));
#ifdef MFEM_USE_MPI
   cout << "Number of finite element unknowns: "
        << fes->GetTrueVSize() << endl;
#else 
   cout << "Number of finite element unknowns: "
        << fes->GlobalTrueVSize() << endl;
#endif
   //cout << "\tNumber of vertices = " << fes->GetNV() << endl;
   //cout << "\tNumber of vertex Dofs = " << fes->GetNVDofs() << endl;
   //cout << "\tNumber of edge Dofs = " << fes->GetNEDofs() << endl;
   //cout << "\tNumber of face Dofs = " << fes->GetNFDofs() << endl;
   //cout << "\tNumber of Boundary Edges = "<< fes->GetNBE() << endl;

   // set up the mass matrix
   mass.reset(new BilinearFormType(static_cast<SpaceType*>(fes.get())));
   mass->AddDomainIntegrator(new DiagMassIntegrator(num_state));
   mass->Assemble();
   mass->Finalize();

   // set up the stiffness matrix
   velocity.reset(new VectorFunctionCoefficient(mesh->Dimension(), vel_field));
   cout << "dimension is " << mesh->Dimension() << endl;
   stiff.reset(new BilinearFormType(static_cast<SpaceType*>(fes.get())));
   stiff->AddDomainIntegrator(new AdvectionIntegrator(*velocity, -1.0));
   // add the LPS stabilization
   double lps_coeff = options["lps-coeff"].get<double>();
   stiff->AddDomainIntegrator(new LPSIntegrator(*velocity, -1.0, lps_coeff));
   int skip_zeros = 0;
   stiff->Assemble(skip_zeros);
   stiff->Finalize(skip_zeros);

   // define the time-dependent operator
#ifdef MFEM_USE_MPI
   // The parallel bilinear forms return a pointer that this solver owns
   mass_matrix.reset(mass->ParallelAssemble());
   stiff_matrix.reset(stiff->ParallelAssemble());
#else
   mass_matrix.reset(new MatrixType(mass->SpMat()));
   stiff_matrix.reset(new MatrixType(stiff->SpMat()));
#endif
   evolver.reset(new LinearEvolver(*mass_matrix, *stiff_matrix));

}

} // namespace mach
