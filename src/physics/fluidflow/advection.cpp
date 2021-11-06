#include "evolver.hpp"
#include "sbp_fe.hpp"
#include "diag_mass_integ.hpp"
#include "advection.hpp"

#include <memory>

using namespace mfem;
using namespace std;

namespace mach
{
void AdvectionIntegrator::AssembleElementMatrix(const FiniteElement &el,
                                                ElementTransformation &Trans,
                                                DenseMatrix &elmat)
{
   using SBP = const SBPTriangleElement &;
   int num_nodes = el.GetDof();
   int dim = el.GetDim();
#ifdef MFEM_THREAD_SAFE
   DenseMatrix vel, velhat, adjJ, Q;
   Vector vel_i, velhat_i, Udi;
#endif
   elmat.SetSize(num_nodes);
   velhat.SetSize(dim, num_nodes);  // scaled velocity in reference space
   adjJ.SetSize(dim);               // adjJ = |J|*dxi/dx = adj(dx/dxi)
   Udi.SetSize(num_nodes);  // reference to one component of velhat at all nodes

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
      Q.RightScaling(Udi);   // This makes Q_{di}^T * diag(Udi)
      elmat.Add(-alpha, Q);  // minus sign accounts for integration by parts
   }
}

AdvectLPSIntegrator::AdvectLPSIntegrator(VectorCoefficient &velc,
                                         double a,
                                         double diss_coeff)
 : vel_coeff(velc), alpha(a), lps_coeff(diss_coeff)
{ }

void AdvectLPSIntegrator::AssembleElementMatrix(const FiniteElement &el,
                                                ElementTransformation &Trans,
                                                DenseMatrix &elmat)
{
   using SBP = const SBPTriangleElement &;
   int num_nodes = el.GetDof();
   int dim = el.GetDim();
#ifdef MFEM_THREAD_SAFE
   DenseMatrix vel, adjJ, P;
   Vector velhat_i, AH;
#endif
   elmat.SetSize(num_nodes);
   adjJ.SetSize(dim);      // adjJ = |J|*dxi/dx = adj(dx/dxi)
   P.SetSize(num_nodes);   // LPS projection operator
   velhat_i.SetSize(dim);  // scaled velocity in reference space at a node
   AH.SetSize(num_nodes);  // the scaling matrix for LPS

   // Get AH: the scaling matrix, A, times the quadrature weights, H
   vel_coeff.Eval(vel, Trans, el.GetNodes());
   Vector vel_i;  // reference to vel at a node
   const Vector &H = static_cast<SBP>(el).returnDiagNorm();
   for (int i = 0; i < num_nodes; i++)
   {
      vel.GetColumnReference(i, vel_i);
      Trans.SetIntPoint(&el.GetNodes().IntPoint(i));
      CalcAdjugate(Trans.Jacobian(), adjJ);
      adjJ.Mult(vel_i, velhat_i);
      AH(i) = alpha * lps_coeff * H(i) * velhat_i.Norml2();
   }
   // Get the projection operator, construct LPS, and add to element matrix
   static_cast<SBP>(el).getProjOperator(P);
   P.Transpose();
   MultADAt(P, AH, elmat);
}

template <int dim>
AdvectionSolver<dim>::AdvectionSolver(const nlohmann::json &json_options,
                                      void (*vel_field)(const Vector &,
                                                        Vector &),
                                      MPI_Comm comm)
 : AbstractSolver(json_options, nullptr, comm)
{
   // set up the stiffness matrix
   velocity = std::make_unique<VectorFunctionCoefficient>(mesh->Dimension(),
                                                          vel_field);
   *out << "dimension is " << mesh->Dimension() << endl;
   stiff =
       std::make_unique<BilinearFormType>(static_cast<SpaceType *>(fes.get()));
   stiff->AddDomainIntegrator(new AdvectionIntegrator(*velocity, -1.0));
   // add the LPS stabilization
   auto lps_coeff = options["space-dis"]["lps-coeff"].template get<double>();
   stiff->AddDomainIntegrator(
       new AdvectLPSIntegrator(*velocity, -1.0, lps_coeff));
   int skip_zeros = 0;
   stiff->Assemble(skip_zeros);
   stiff->Finalize(skip_zeros);

   // The parallel bilinear forms return a pointer that this solver owns
   stiff_matrix.reset(stiff->ParallelAssemble());

   Array<int> ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 0;
   /// This should overwrite the evolver defined in base class constructor
   /// TODO: This should not be necessary?
   evolver.reset(
       //   new LinearEvolver(*(mass_matrix), *(stiff_matrix), *(out)));
       new MachEvolver(ess_bdr,
                       nullptr,
                       mass.get(),
                       nullptr,
                       stiff.get(),
                       nullptr,
                       nullptr,
                       *(out),
                       0.0));
}

// explicit instantiation
template class AdvectionSolver<1>;
template class AdvectionSolver<2>;
template class AdvectionSolver<3>;

}  // namespace mach
