#ifndef MACH_MMS_INTEG_DEF_DG
#define MACH_MMS_INTEG_DEF_DG

#include "mfem.hpp"

#include "mms_integ_dg.hpp"

namespace mach
{
template <typename Derived>
void InviscidMMSIntegrator<Derived>::AssembleElementVector(
    const mfem::FiniteElement &el,
    mfem::ElementTransformation &trans,
    const mfem::Vector &elfun,
    mfem::Vector &elvect)
{
   using namespace mfem;
   int num_nodes = el.GetDof();
#ifdef MFEM_THREAD_SAFE
   Vector x_i, src_i;
#endif
   elvect.SetSize(num_states * num_nodes);
   DenseMatrix res(elvect.GetData(), num_nodes, num_states);
   elvect = 0.0;
   shape.SetSize(num_nodes);
   int intorder = trans.OrderGrad(&el) + trans.Order() + el.GetOrder();
   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      ir = &IntRules.Get(el.GetGeomType(), intorder);
   }
   for (int i = 0; i < ir->GetNPoints(); ++i)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);
      trans.Transform(ip, x_i);
      el.CalcShape(ip, shape);
      double weight = trans.Weight() * ip.weight;
      src_i.SetSize(num_states);
      source(x_i, src_i);
      for (int n = 0; n < num_states; ++n)
      {
         for (int s = 0; s < num_nodes; ++s)
         {
            res(s, n) += weight * src_i(n) * shape(s);
         }
      }
   }
   res *= alpha;
}

template <typename Derived>
void InviscidMMSIntegrator<Derived>::AssembleElementGrad(
    const mfem::FiniteElement &el,
    mfem::ElementTransformation &trans,
    const mfem::Vector &elfun,
    mfem::DenseMatrix &elmat)
{
   using namespace mfem;
   int num_nodes = el.GetDof();
   elmat.SetSize(num_states * num_nodes);
   elmat = 0.0;
}

}  // namespace mach

#endif
