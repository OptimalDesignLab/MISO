#include "diag_mass_integ.hpp"

using namespace mfem;
using namespace std;

namespace mach
{
void DiagMassIntegrator::AssembleElementMatrix(
    const FiniteElement &el, ElementTransformation &Trans,
    DenseMatrix &elmat)
{   
   const IntegrationRule &ir = el.GetNodes();
   int num_nodes = ir.GetNPoints();
   elmat.SetSize(num_nodes*num_state);
   elmat = 0.0;
   double norm;
   // loop over the nodes of the SBP element
   // TODO: this does not take advantage of el.H diagonal norm
   for (int i = 0; i < num_nodes; i++)
   {
      // get the Jacobian (Trans.Weight) and cubature weight (node.weight)
      const IntegrationPoint &node = ir.IntPoint(i);
      Trans.SetIntPoint(&node);
      norm = node.weight * Trans.Weight();
      for (int k = 0; k < num_state; k++)
      {
         // Insert diagonal entries for each state 
         //elmat(i*num_state + k, i*num_state + k) = norm; // assumes states are ordered fastest;
         elmat(i + k*num_nodes, i + k*num_nodes) = norm;
      }
   }
}
}