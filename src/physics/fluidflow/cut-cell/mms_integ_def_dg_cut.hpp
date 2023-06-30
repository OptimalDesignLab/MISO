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

template <typename Derived>
void CutSensitivityMMSIntegrator<Derived>::calcTransformSens(
    const mfem::FiniteElement &el,
    mfem::ElementTransformation &trans,
    const IntegrationPoint &ip,
    mfem::DenseMatrix &el_dx)
{
   int dim = el.GetDim();
   /// calculate FD
   double delta = 1e-05;
   for (int i = 0; i < dim; ++i)
   {
      IntegrationPoint ip_p = ip;
      IntegrationPoint ip_m = ip;
      if (i == 0)
      {
         ip_p.x = ip.x + delta;
         ip_m.x = ip.x - delta;
      }
      else
      {
         ip_p.y = ip.y + delta;
         ip_m.y = ip.y - delta;
      }
      trans.SetIntPoint(&ip_p);
      Vector x_i_p(dim), x_i_m(dim);
      trans.Transform(ip_p, x_i_p);
      trans.SetIntPoint(&ip_m);
      trans.Transform(ip_m, x_i_m);
      x_i_p -= x_i_m;
      x_i_p /= 2.0 * delta;
      for (int j = 0; j < dim; ++j)
      {
         el_dx(j, i) = x_i_p(j);
      }
   }
}

template <typename Derived>
void CutSensitivityMMSIntegrator<Derived>::AssembleElementVector(
    const mfem::FiniteElement &el,
    mfem::ElementTransformation &trans,
    const mfem::Vector &elfun,
    mfem::Vector &elvect)
{
   using namespace mfem;
   int num_nodes = el.GetDof();
   int dim = el.GetDim();
#ifdef MFEM_THREAD_SAFE
   Vector x_i, src_i;
#endif
   Vector dxqi, dshape_xqi, src_Jac_a, dxqe_da;
   DenseMatrix src_Jac_xq;
   mfem::DenseMatrix del_xq;
   del_xq.SetSize(dim, dim);
   dxqe_da.SetSize(dim);
   elvect.SetSize(num_states * num_nodes);
   DenseMatrix res(elvect.GetData(), num_nodes, num_states);
   elvect = 0.0;
   shape.SetSize(num_nodes);
   dshape.SetSize(num_nodes, dim);
   dxqi.SetSize(dim);
   src_Jac_xq.SetSize(num_states, dim);
   src_Jac_a.SetSize(num_states);
   dshape_xqi.SetSize(num_nodes);
   if (embeddedElements.at(trans.ElementNo) == true)
   {
      elvect = 0.0;
   }
   else
   {
      int intorder = trans.OrderGrad(&el) + trans.Order() + el.GetOrder();
      const IntegrationRule *ir;  // = IntRule;
      ir = cutSquareIntRules[trans.ElementNo];
      const IntegrationRule *ir_a;  // = IntRule;
      ir_a = cutSquareIntRules_sens[trans.ElementNo];
      if (ir == NULL)
      {
         elvect = 0.0;
      }
      else
      {
         double delta = 1e-05;
         for (int i = 0; i < ir->GetNPoints(); ++i)
         {
            const IntegrationPoint &ip = ir->IntPoint(i);
            const IntegrationPoint &ip_a = ir_a->IntPoint(i);
            trans.SetIntPoint(&ip);
            trans.Transform(ip, x_i);
            calcTransformSens(el, trans, ip, del_xq);
            el.CalcShape(ip, shape);
            src_i.SetSize(num_states);
            source(x_i, src_i);
            double dwda_i = ip_a.weight;
            /// first term (dwda)
            for (int n = 0; n < num_states; ++n)
            {
               for (int s = 0; s < num_nodes; ++s)
               {
                  res(s, n) += trans.Weight() * src_i(n) * shape(s) * dwda_i;
               }
            }
            /// 2nd term (dshapeda)
            dxqi(0) = ip_a.x;
            dxqi(1) = ip_a.y;
            el.CalcDShape(ip, dshape);
            dshape.Mult(dxqi, dshape_xqi);
            for (int n = 0; n < num_states; ++n)
            {
               for (int s = 0; s < num_nodes; ++s)
               {
                  res(s, n) +=
                      trans.Weight() * src_i(n) * dshape_xqi(s) * ip.weight;
               }
            }
            /// third term (dsrc_dxq)
            sourceJac(x_i, src_Jac_xq);
            del_xq.Mult(dxqi, dxqe_da);
            src_Jac_xq.Mult(dxqe_da, src_Jac_a);

            for (int n = 0; n < num_states; ++n)
            {
               for (int s = 0; s < num_nodes; ++s)
               {
                  res(s, n) +=
                      trans.Weight() * src_Jac_a(n) * shape(s) * ip.weight;
               }
            }
         }
      }
   }
   res *= alpha;
}

}  // namespace mach

#endif
