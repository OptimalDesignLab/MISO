#ifndef MACH_POISSON_DG_DEF_CUT
#define MACH_POISSON_DG_DEF_CUT
#include "poisson_dg_cut.hpp"
using namespace blitz;
namespace mach
{
void CutDomainLFIntegrator::AssembleRHSElementVect(const FiniteElement &el,
                                                   ElementTransformation &Tr,
                                                   Vector &elvect)
{
   int dof = el.GetDof();
   shape.SetSize(dof);  // vector of size dof
   elvect.SetSize(dof);
   elvect = 0.0;
   const IntegrationRule *ir;
   ir = CutIntRules[Tr.ElementNo];
   if (ir == NULL)
   {
      // ir = &IntRules.Get(el.GetGeomType(),
      //                    oa * el.GetOrder() + ob + Tr.OrderW());
      ir = &IntRules.Get(el.GetGeomType(), oa * el.GetOrder() + ob);
   }
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetIntPoint(&ip);
      double val = Tr.Weight() * Q.Eval(Tr, ip);

      el.CalcShape(ip, shape);

      add(elvect, ip.weight * val, shape, elvect);
   }
}

void CutDomainLFIntegrator::AssembleDeltaElementVect(
    const FiniteElement &fe,
    ElementTransformation &Trans,
    Vector &elvect)
{
   MFEM_ASSERT(delta != NULL, "coefficient must be DeltaCoefficient");
   elvect.SetSize(fe.GetDof());
   fe.CalcPhysShape(Trans, elvect);
   elvect *= delta->EvalDelta(Trans, Trans.GetIntPoint());
}

void CutDomainIntegrator::AssembleRHSElementVect(const FiniteElement &el,
                                                 ElementTransformation &Tr,
                                                 Vector &elvect)
{
   int dof = el.GetDof();
   shape.SetSize(dof);  // vector of size dof
   elvect.SetSize(dof);
   elvect = 0.0;
   const IntegrationRule *ir;
   ir = CutIntRules[Tr.ElementNo];
   if (ir == NULL)
   {
      ir = &IntRules.Get(el.GetGeomType(), oa * el.GetOrder() + ob);
   }
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetIntPoint(&ip);
      double val = Tr.Weight() * Q.Eval(Tr, ip);

      el.CalcShape(ip, shape);

      add(elvect, ip.weight * val, shape, elvect);
   }
}

void CutDomainIntegrator::AssembleDeltaElementVect(const FiniteElement &fe,
                                                   ElementTransformation &Trans,
                                                   Vector &elvect)
{
   MFEM_ASSERT(delta != NULL, "coefficient must be DeltaCoefficient");
   elvect.SetSize(fe.GetDof());
   fe.CalcPhysShape(Trans, elvect);
   elvect *= delta->EvalDelta(Trans, Trans.GetIntPoint());
}
void CutDiffusionIntegrator::AssembleElementMatrix(const FiniteElement &el,
                                                   ElementTransformation &Trans,
                                                   DenseMatrix &elmat)
{
   int nd = el.GetDof();
   int dim = el.GetDim();
   int spaceDim = Trans.GetSpaceDim();
   bool square = (dim == spaceDim);
   double w;

#ifdef MFEM_THREAD_SAFE
   DenseMatrix dshape(nd, dim), dshapedxt(nd, spaceDim), invdfdx(dim, spaceDim);
#else
   dshape.SetSize(nd, dim);
   dshapedxt.SetSize(nd, spaceDim);
   invdfdx.SetSize(dim, spaceDim);
#endif
   elmat.SetSize(nd);
   elmat = 0.0;
   // elmat is identity for embedded elements
   if (EmbeddedElements.at(Trans.ElementNo) == true)
   {
      for (int k = 0; k < elmat.Size(); ++k)
      {
         elmat(k, k) = 1.0;
      }
   }
   else
   {
      // use Saye's quadrature rule for elements cut by boundary
      const IntegrationRule *ir;
      ir = CutIntRules[Trans.ElementNo];
      if (ir == NULL)
      {
         ir = IntRule ? IntRule : &GetRule(el, el);
      }
      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         el.CalcDShape(ip, dshape);
         Trans.SetIntPoint(&ip);
         w = Trans.Weight();
         w = ip.weight / (square ? w : w * w * w);
         // AdjugateJacobian = / adj(J),         if J is square
         //                    \ adj(J^t.J).J^t, otherwise
         Mult(dshape, Trans.AdjugateJacobian(), dshapedxt);
         if (!MQ)
         {
            if (Q)
            {
               w *= Q->Eval(Trans, ip);
            }
            AddMult_a_AAt(w, dshapedxt, elmat);
         }
         else
         {
            MQ->Eval(invdfdx, Trans, ip);
            invdfdx *= w;
            Mult(dshapedxt, invdfdx, dshape);
            AddMultABt(dshape, dshapedxt, elmat);
         }
      }
   }
}

const IntegrationRule &CutDiffusionIntegrator::GetRule(
    const FiniteElement &trial_fe,
    const FiniteElement &test_fe)
{
   int order;
   if (trial_fe.Space() == FunctionSpace::Pk)
   {
      order = trial_fe.GetOrder() + test_fe.GetOrder() - 2;
   }
   else
   {
      // order = 2*el.GetOrder() - 2;  // <-- this seems to work fine too
      order = trial_fe.GetOrder() + test_fe.GetOrder() + trial_fe.GetDim() - 1;
   }

   if (trial_fe.Space() == FunctionSpace::rQk)
   {
      return RefinedIntRules.Get(trial_fe.GetGeomType(), order);
   }
   return IntRules.Get(trial_fe.GetGeomType(), order);
}

void CutBoundaryFaceIntegrator::AssembleElementMatrix(
    const FiniteElement &el,
    ElementTransformation &Trans,
    DenseMatrix &elmat)
{
   int dim, ndof1, ndofs;
   bool kappa_is_nonzero = (kappa != 0.);
   double w, wq = 0.0;
   dim = el.GetDim();
   ndof1 = el.GetDof();
   nor.SetSize(dim);
   nh.SetSize(dim);
   ni.SetSize(dim);
   adjJ.SetSize(dim);
   if (MQ)
   {
      mq.SetSize(dim);
   }
   shape1.SetSize(ndof1);
   dshape1.SetSize(ndof1, dim);
   dshape1dn.SetSize(ndof1);
   ndofs = ndof1;
   elmat.SetSize(ndofs);
   elmat = 0.0;
   if (kappa_is_nonzero)
   {
      jmat.SetSize(ndofs);
      jmat = 0.;
   }
   const IntegrationRule *ir;
   ir = cutSegmentIntRules[Trans.ElementNo];
   if (ir == NULL)
   {
      // cout << "element is " << Trans.ElementNo << endl;
      elmat = 0.0;
   }
   // assemble: < {(Q \nabla u).n},[v] >      --> elmat
   //           kappa < {h^{-1} Q} [u],[v] >  --> jmat
   else
   {
      for (int p = 0; p < ir->GetNPoints(); p++)
      {
         const IntegrationPoint &ip = ir->IntPoint(p);
         IntegrationPoint eip1;
         eip1 = ip;
         Trans.SetIntPoint(&ip);
         double ds;
         if (dim == 1)
         {
            nor(0) = 2 * eip1.x - 1.0;
         }
         else
         {
            // CalcOrtho(Trans.Jacobian(), nor);
            //  double ds = sqrt((eip1.x*eip1.x) + (eip1.y*eip1.y));
            Vector v(dim);
            Trans.Transform(eip1, v);
            // double xc = 10.0;
            // double yc = 10.0;
            // double a = 4.0;
            // double b = 1.0;
            // double nx = 2 * (v(0) - xc) / (a * a);
            // double ny = 2 * (v(1) - yc) / (b * b);
            // ds = sqrt((nx * nx) + (ny * ny));
            // nor(0) = -nx / ds;
            // nor(1) = -ny / ds;
            TinyVector<double, 2> beta, xs;
            xs(0) = v(0);
            xs(1) = v(1);
            beta = phi.grad(xs);
            ds = mag(beta);
            double nx = beta(0);
            double ny = beta(1);
            nor(0) = nx / ds;
            nor(1) = ny / ds;
         }
         el.CalcShape(eip1, shape1);
         el.CalcDShape(eip1, dshape1);
         Trans.SetIntPoint(&eip1);
         w = ip.weight / Trans.Weight();
         if (!MQ)
         {
            if (Q)
            {
               w *= Q->Eval(Trans, eip1);
            }
            ni.Set(w, nor);
         }
         else
         {
            nh.Set(w, nor);
            MQ->Eval(mq, Trans, eip1);
            mq.MultTranspose(nh, ni);
         }
         CalcAdjugate(Trans.Jacobian(), adjJ);
         adjJ.Mult(ni, nh);
         if (kappa_is_nonzero)
         {
            wq = ni * nor;
            // wq = ip.weight*nor.Norml2();
         }
         // std::cout << "normal is " << nor(0) << " , " << nor(1) << endl;
         // std::cout << "norm is " << nor * nor << endl;
         // std::cout << "ds is " << ds << std::endl;
         // std::cout << "wq/ip.weight " << wq / ip.weight << std::endl;
         // Note: in the jump term, we use 1/h1 = |nor|/det(J1) which is
         // independent of Loc1 and always gives the size of element 1 in
         // direction perpendicular to the face. Indeed, for linear
         // transformation
         //     |nor|=measure(face)/measure(ref. face),
         //   det(J1)=measure(element)/measure(ref. element),
         // and the ratios measure(ref. element)/measure(ref. face) are
         // compatible for all element/face pairs.
         // For example: meas(ref. tetrahedron)/meas(ref. triangle) = 1/3, and
         // for any tetrahedron vol(tet)=(1/3)*height*area(base).
         // For interior faces: q_e/h_e=(q1/h1+q2/h2)/2.
         dshape1.Mult(nh, dshape1dn);
         for (int i = 0; i < ndof1; i++)
            for (int j = 0; j < ndof1; j++)
            {
               elmat(i, j) += shape1(i) * dshape1dn(j);
            }

         if (kappa_is_nonzero)
         {
            // only assemble the lower triangular part of jmat
            wq *= kappa;
            for (int i = 0; i < ndof1; i++)
            {
               const double wsi = wq * shape1(i);
               for (int j = 0; j <= i; j++)
               {
                  jmat(i, j) += wsi * shape1(j);
               }
            }
         }
      }

      // elmat := -elmat + sigma*elmat^t + jmat
      if (kappa_is_nonzero)
      {
         for (int i = 0; i < ndofs; i++)
         {
            for (int j = 0; j < i; j++)
            {
               double aij = elmat(i, j), aji = elmat(j, i), mij = jmat(i, j);
               elmat(i, j) = sigma * aji - aij + mij;
               elmat(j, i) = sigma * aij - aji + mij;
            }
            elmat(i, i) = (sigma - 1.) * elmat(i, i) + jmat(i, i);
         }
      }
      else
      {
         for (int i = 0; i < ndofs; i++)
         {
            for (int j = 0; j < i; j++)
            {
               double aij = elmat(i, j), aji = elmat(j, i);
               elmat(i, j) = sigma * aji - aij;
               elmat(j, i) = sigma * aij - aji;
            }
            elmat(i, i) *= (sigma - 1.);
         }
      }
   }
}

void CutDGDiffusionIntegrator::AssembleFaceMatrix(
    const FiniteElement &el1,
    const FiniteElement &el2,
    FaceElementTransformations &Trans,
    DenseMatrix &elmat)
{
   int dim, ndof1, ndof2, ndofs;
   bool kappa_is_nonzero = (kappa != 0.);
   double w, wq = 0.0;
   dim = el1.GetDim();
   ndof1 = el1.GetDof();
   nor.SetSize(dim);
   nh.SetSize(dim);
   ni.SetSize(dim);
   adjJ.SetSize(dim);
   if (MQ)
   {
      mq.SetSize(dim);
   }
   shape1.SetSize(ndof1);
   dshape1.SetSize(ndof1, dim);
   dshape1dn.SetSize(ndof1);
   if (Trans.Elem2No >= 0)
   {
      ndof2 = el2.GetDof();
      shape2.SetSize(ndof2);
      dshape2.SetSize(ndof2, dim);
      dshape2dn.SetSize(ndof2);
   }
   else
   {
      ndof2 = 0;
   }
   ndofs = ndof1 + ndof2;
   elmat.SetSize(ndofs);
   elmat = 0.0;
   if (immersedFaces[Trans.Face->ElementNo] == true)
   {
      elmat = 0.0;
   }
   else
   {
      if (kappa_is_nonzero)
      {
         jmat.SetSize(ndofs);
         jmat = 0.;
      }
      const IntegrationRule *ir;
      if (find(cutinteriorFaces.begin(),
               cutinteriorFaces.end(),
               Trans.Face->ElementNo) != cutinteriorFaces.end())
      {
         ir = cutInteriorFaceIntRules[Trans.Face->ElementNo];
      }
      else
      {
         ir = IntRule;
      }
      if (ir == NULL)
      {
         // a simple choice for the integration order; is this OK?
         int order;
         if (ndof2)
         {
            order = 2 * max(el1.GetOrder(), el2.GetOrder());
         }
         else
         {
            order = 2 * el1.GetOrder();
         }
         ir = &IntRules.Get(Trans.FaceGeom, order);
      }
      // assemble: < {(Q \nabla u).n},[v] >      --> elmat
      //           kappa < {h^{-1} Q} [u],[v] >  --> jmat
      // if (find(cutinteriorFaces.begin(), cutinteriorFaces.end(),
      // Trans.Face->ElementNo) != cutinteriorFaces.end())
      // {
      //    std::cout << "face is " << Trans.Face->ElementNo << " elements are "
      //    << Trans.Elem1No << " , " << Trans.Elem2No << std::endl;
      // }
      for (int p = 0; p < ir->GetNPoints(); p++)
      {
         const IntegrationPoint &ip = ir->IntPoint(p);
         IntegrationPoint eip1, eip2;
         Trans.Loc1.Transform(ip, eip1);
         Trans.Face->SetIntPoint(&ip);
         if (dim == 1)
         {
            nor(0) = 2 * eip1.x - 1.0;
         }
         else
         {
            CalcOrtho(Trans.Face->Jacobian(), nor);
         }
         el1.CalcShape(eip1, shape1);
         el1.CalcDShape(eip1, dshape1);
         Trans.Elem1->SetIntPoint(&eip1);
         w = ip.weight / Trans.Elem1->Weight();
         if (ndof2)
         {
            w /= 2;
         }
         if (!MQ)
         {
            if (Q)
            {
               w *= Q->Eval(*Trans.Elem1, eip1);
            }
            ni.Set(w, nor);
         }
         else
         {
            nh.Set(w, nor);
            MQ->Eval(mq, *Trans.Elem1, eip1);
            mq.MultTranspose(nh, ni);
         }
         CalcAdjugate(Trans.Elem1->Jacobian(), adjJ);
         adjJ.Mult(ni, nh);
         if (kappa_is_nonzero)
         {
            wq = ni * nor;
         }
         // Note: in the jump term, we use 1/h1 = |nor|/det(J1) which is
         // independent of Loc1 and always gives the size of element 1 in
         // direction perpendicular to the face. Indeed, for linear
         // transformation
         //     |nor|=measure(face)/measure(ref. face),
         //   det(J1)=measure(element)/measure(ref. element),
         // and the ratios measure(ref. element)/measure(ref. face) are
         // compatible for all element/face pairs.
         // For example: meas(ref. tetrahedron)/meas(ref. triangle) = 1/3, and
         // for any tetrahedron vol(tet)=(1/3)*height*area(base).
         // For interior faces: q_e/h_e=(q1/h1+q2/h2)/2.

         dshape1.Mult(nh, dshape1dn);
         for (int i = 0; i < ndof1; i++)
            for (int j = 0; j < ndof1; j++)
            {
               elmat(i, j) += shape1(i) * dshape1dn(j);
            }

         if (ndof2)
         {
            Trans.Loc2.Transform(ip, eip2);
            Vector v(dim);
            Trans.Elem2->Transform(eip2, v);
            el2.CalcShape(eip2, shape2);
            el2.CalcDShape(eip2, dshape2);
            Trans.Elem2->SetIntPoint(&eip2);
            w = ip.weight / 2 / Trans.Elem2->Weight();
            if (!MQ)
            {
               if (Q)
               {
                  w *= Q->Eval(*Trans.Elem2, eip2);
               }
               ni.Set(w, nor);
            }
            else
            {
               nh.Set(w, nor);
               MQ->Eval(mq, *Trans.Elem2, eip2);
               mq.MultTranspose(nh, ni);
            }
            CalcAdjugate(Trans.Elem2->Jacobian(), adjJ);
            adjJ.Mult(ni, nh);
            if (kappa_is_nonzero)
            {
               wq += ni * nor;
            }

            dshape2.Mult(nh, dshape2dn);

            for (int i = 0; i < ndof1; i++)
               for (int j = 0; j < ndof2; j++)
               {
                  elmat(i, ndof1 + j) += shape1(i) * dshape2dn(j);
               }

            for (int i = 0; i < ndof2; i++)
               for (int j = 0; j < ndof1; j++)
               {
                  elmat(ndof1 + i, j) -= shape2(i) * dshape1dn(j);
               }

            for (int i = 0; i < ndof2; i++)
               for (int j = 0; j < ndof2; j++)
               {
                  elmat(ndof1 + i, ndof1 + j) -= shape2(i) * dshape2dn(j);
               }
         }

         if (kappa_is_nonzero)
         {
            // only assemble the lower triangular part of jmat
            wq *= kappa;
            for (int i = 0; i < ndof1; i++)
            {
               const double wsi = wq * shape1(i);
               for (int j = 0; j <= i; j++)
               {
                  jmat(i, j) += wsi * shape1(j);
               }
            }
            if (ndof2)
            {
               for (int i = 0; i < ndof2; i++)
               {
                  const int i2 = ndof1 + i;
                  const double wsi = wq * shape2(i);
                  for (int j = 0; j < ndof1; j++)
                  {
                     jmat(i2, j) -= wsi * shape1(j);
                  }
                  for (int j = 0; j <= i; j++)
                  {
                     jmat(i2, ndof1 + j) += wsi * shape2(j);
                  }
               }
            }
         }
      }

      // elmat := -elmat + sigma*elmat^t + jmat
      if (kappa_is_nonzero)
      {
         for (int i = 0; i < ndofs; i++)
         {
            for (int j = 0; j < i; j++)
            {
               double aij = elmat(i, j), aji = elmat(j, i), mij = jmat(i, j);
               elmat(i, j) = sigma * aji - aij + mij;
               elmat(j, i) = sigma * aij - aji + mij;
            }
            elmat(i, i) = (sigma - 1.) * elmat(i, i) + jmat(i, i);
         }
      }
      else
      {
         for (int i = 0; i < ndofs; i++)
         {
            for (int j = 0; j < i; j++)
            {
               double aij = elmat(i, j), aji = elmat(j, i);
               elmat(i, j) = sigma * aji - aij;
               elmat(j, i) = sigma * aij - aji;
            }
            elmat(i, i) *= (sigma - 1.);
         }
      }
   }
}

void CutDGDirichletLFIntegrator::AssembleRHSElementVect(
    const FiniteElement &el,
    ElementTransformation &Trans,
    Vector &elvect)
{
   int dim, ndof;
   bool kappa_is_nonzero = (kappa != 0.);
   double w;
   dim = el.GetDim();
   ndof = el.GetDof();
   nor.SetSize(dim);
   nh.SetSize(dim);
   ni.SetSize(dim);
   adjJ.SetSize(dim);
   if (MQ)
   {
      mq.SetSize(dim);
   }
   shape.SetSize(ndof);
   dshape.SetSize(ndof, dim);
   dshape_dn.SetSize(ndof);

   elvect.SetSize(ndof);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   ir = cutSegmentIntRules[Trans.ElementNo];
   if (ir == NULL)
   {
      elvect = 0.0;
   }
   else
   {
      for (int p = 0; p < ir->GetNPoints(); p++)
      {
         const IntegrationPoint &ip = ir->IntPoint(p);
         IntegrationPoint eip;
         eip = ip;
         if (dim == 1)
         {
            nor(0) = 2 * eip.x - 1.0;
         }
         else
         {
            Vector v(dim);
            Trans.Transform(eip, v);
            // double xc = 10.0;
            // double yc = 10.0;
            // double a = 4.0;
            // double b = 1.0;
            // double nx = 2 * (v(0) - xc) / (a * a);
            // double ny = 2 * (v(1) - yc) / (b * b);
            // double ds = sqrt((nx * nx) + (ny * ny));
            // nor(0) = -nx / ds;
            // nor(1) = -ny / ds;
            TinyVector<double, 2> beta, xs;
            xs(0) = v(0);
            xs(1) = v(1);
            beta = phi.grad(xs);
            double ds = mag(beta);
            double nx = beta(0);
            double ny = beta(1);
            nor(0) = nx / ds;
            nor(1) = ny / ds;
         }
         el.CalcShape(eip, shape);
         el.CalcDShape(eip, dshape);
         Trans.SetIntPoint(&eip);
         // compute uD through the face transformation
         w = ip.weight * uD->Eval(Trans, ip) / Trans.Weight();
         if (!MQ)
         {
            if (Q)
            {
               w *= Q->Eval(Trans, eip);
            }
            ni.Set(w, nor);
         }
         else
         {
            nh.Set(w, nor);
            MQ->Eval(mq, Trans, eip);
            mq.MultTranspose(nh, ni);
         }
         CalcAdjugate(Trans.Jacobian(), adjJ);
         adjJ.Mult(ni, nh);

         dshape.Mult(nh, dshape_dn);
         elvect.Add(sigma, dshape_dn);

         if (kappa_is_nonzero)
         {
            elvect.Add(kappa * (ni * nor), shape);
         }
      }
   }
}

void CutDGDirichletLFIntegrator::AssembleDeltaElementVect(
    const FiniteElement &fe,
    ElementTransformation &Trans,
    Vector &elvect)
{
   MFEM_ASSERT(delta != NULL, "coefficient must be DeltaCoefficient");
   elvect.SetSize(fe.GetDof());
   fe.CalcPhysShape(Trans, elvect);
   elvect *= delta->EvalDelta(Trans, Trans.GetIntPoint());
}

void CutDGNeumannLFIntegrator::AssembleRHSElementVect(
    const FiniteElement &el,
    ElementTransformation &Trans,
    Vector &elvect)
{
   int dim, ndof;
   double w;
   Vector Qvec;
   dim = el.GetDim();
   ndof = el.GetDof();
   shape.SetSize(ndof);
   elvect.SetSize(ndof);
   elvect = 0.0;
   nor.SetSize(dim);
   const IntegrationRule *ir = IntRule;
   ir = cutSegmentIntRules[Trans.ElementNo];
   // elvect is zero for elements other than cut
   if (ir == NULL)
   {
      elvect = 0.0;
   }
   else
   {
      for (int p = 0; p < ir->GetNPoints(); p++)
      {
         const IntegrationPoint &ip = ir->IntPoint(p);
         el.CalcShape(ip, shape);
         Trans.SetIntPoint(&ip);
         // this evaluates the coefficient for the
         // integration points in physical space
         QN.Eval(Qvec, Trans, ip);
         Vector v(dim);
         // transform the integration point to original element
         Trans.Transform(ip, v);
         // double xc = 10.0;
         // double yc = 10.0;
         // double a = 4.0;
         // double b = 1.0;
         // double nx = 2 * (v(0) - xc) / (a * a);
         // double ny = 2 * (v(1) - yc) / (b * b);
         // double ds = sqrt((nx * nx) + (ny * ny));
         // nor(0) = -nx / ds;
         // nor(1) = -ny / ds;
            TinyVector<double, 2> beta, xs;
            xs(0) = v(0);
            xs(1) = v(1);
            beta = phi.grad(xs);
            double ds = mag(beta);
            double nx = beta(0);
            double ny = beta(1);
            nor(0) = nx / ds;
            nor(1) = ny / ds;
         elvect.Add(ip.weight * sqrt(Trans.Weight()) * (Qvec * nor), shape);
      }
   }
}
}  // namespace mach
#endif