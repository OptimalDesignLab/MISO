
#include "therm_integ.hpp"

using namespace mfem;
using namespace std;
#if 0
namespace mach
{

void HeatMassIntegrator::AssembleElementMatrix
( const FiniteElement &el, ElementTransformation &Trans,
  DenseMatrix &elmat )
{
   int nd = el.GetDof();
   // int dim = el.GetDim();
   double w;

#ifdef MFEM_THREAD_SAFE
   Vector shape;
#endif
   elmat.SetSize(nd);
   shape.SetSize(nd);

   const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, el, Trans);

   elmat = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      el.CalcShape(ip, shape);

      Trans.SetIntPoint (&ip);
      w = Trans.Weight() * ip.weight;
      // TODO: Need two coefficients, density and cv
      if (Q)
      {
         w *= Q -> Eval(Trans, ip);
      }

      AddMult_a_VVt(w, shape, elmat);
   }
}


} // namespace mach
#endif