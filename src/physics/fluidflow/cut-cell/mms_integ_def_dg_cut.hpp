#ifndef MACH_MMS_INTEG_DEF_DG_CUT
#define MACH_MMS_INTEG_DEF_DG_CUT

#include "mfem.hpp"

#include "mms_integ_dg_cut.hpp"

namespace mach
{
#if 0
template <typename Derived>
void CutMMSIntegrator<Derived>::AssembleElementVector(
    const mfem::FiniteElement &el,
    mfem::ElementTransformation &trans,
    const mfem::Vector &elfun,
    mfem::Vector &elvect)
{
   using namespace mfem;
   const auto &sbp = dynamic_cast<const SBPFiniteElement &>(el);
   // const IntegrationRule &ir = sbp.GetNodes();
   int num_nodes = sbp.GetDof();
   // int dim = sbp.GetDim();
#ifdef MFEM_THREAD_SAFE
   Vector x_i, src_i;
#endif
   src_i.SetSize(num_states);
   elvect.SetSize(num_states * num_nodes);
   DenseMatrix res(elvect.GetData(), num_nodes, num_states);
   elvect = 0.0;
   for (int i = 0; i < num_nodes; ++i)
   {
      const IntegrationPoint &ip = el.GetNodes().IntPoint(i);
      trans.SetIntPoint(&ip);
      trans.Transform(ip, x_i);
      double weight = trans.Weight() * ip.weight;
      source(x_i, src_i);
      for (int n = 0; n < num_states; ++n)
      {
         res(i, n) += weight * src_i(n);
      }
   }
   res *= alpha;
}
#endif

template <typename Derived>
void CutMMSIntegrator<Derived>::AssembleElementVector(
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
   if (embeddedElements.at(trans.ElementNo) == true)
   {
      elvect = 0.0;
   }
   else
   {
      int intorder = trans.OrderGrad(&el) + trans.Order() + el.GetOrder();
      const IntegrationRule *ir;  // = IntRule;
      ir = cutSquareIntRules[trans.ElementNo];
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
   }
   res *= alpha;
}

template <typename Derived>
void CutMMSIntegrator<Derived>::AssembleElementGrad(
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
