// #include "mfem.hpp"

// #include "therm_integ.hpp"

// using namespace mfem;

// namespace mach
// {


// /// TODO: This is mostly right, but assumes that elmat will be DG sized
// ///       (twice what I'll actually see)
// ///       To fix, follow the path where ndof = 0 (boundary face), and choose the
// ///       element to use based on source_attr.
// void InteriorBoundaryOutFluxInteg::AssembleFaceMatrix(
//    const FiniteElement &el1,
//    const FiniteElement &el2,
//    FaceElementTransformations &face_trans,
//    DenseMatrix &elmat)
// {
//    int dim, ndof1, ndof2, ndofs;
//    bool kappa_is_nonzero = (kappa != 0.);
//    double w, wq = 0.0;

//    dim = el1.GetDim();
//    ndof1 = el1.GetDof();

//    nor.SetSize(dim);
//    nh.SetSize(dim);
//    ni.SetSize(dim);
//    adjJ.SetSize(dim);
//    if (MQ)
//    {
//       mq.SetSize(dim);
//    }

//    shape1.SetSize(ndof1);
//    dshape1.SetSize(ndof1, dim);
//    dshape1dn.SetSize(ndof1);
//    if (Trans.Elem2No >= 0)
//    {
//       ndof2 = el2.GetDof();
//       shape2.SetSize(ndof2);
//       dshape2.SetSize(ndof2, dim);
//       dshape2dn.SetSize(ndof2);
//    }
//    else
//    {
//       ndof2 = 0;
//    }

//    ndofs = ndof1 + ndof2;
//    elmat.SetSize(ndofs);
//    elmat = 0.0;
//    if (kappa_is_nonzero)
//    {
//       jmat.SetSize(ndofs);
//       jmat = 0.;
//    }

//    const IntegrationRule *ir = IntRule;
//    if (ir == NULL)
//    {
//       // a simple choice for the integration order; is this OK?
//       int order;
//       if (ndof2)
//       {
//          order = 2*max(el1.GetOrder(), el2.GetOrder());
//       }
//       else
//       {
//          order = 2*el1.GetOrder();
//       }
//       ir = &IntRules.Get(Trans.GetGeometryType(), order);
//    }

//    // assemble: < {(Q \nabla u).n},[v] >      --> elmat
//    //           kappa < {h^{-1} Q} [u],[v] >  --> jmat
//    for (int p = 0; p < ir->GetNPoints(); p++)
//    {
//       const IntegrationPoint &ip = ir->IntPoint(p);

//       // Set the integration point in the face and the neighboring elements
//       Trans.SetAllIntPoints(&ip);

//       // Access the neighboring elements' integration points
//       // Note: eip2 will only contain valid data if Elem2 exists
//       const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();
//       const IntegrationPoint &eip2 = Trans.GetElement2IntPoint();

//       if (dim == 1)
//       {
//          nor(0) = 2*eip1.x - 1.0;
//       }
//       else
//       {
//          CalcOrtho(Trans.Jacobian(), nor);
//       }

//       el1.CalcShape(eip1, shape1);
//       el1.CalcDShape(eip1, dshape1);
//       w = ip.weight/Trans.Elem1->Weight();
//       if (ndof2)
//       {
//          w /= 2;
//       }
//       if (!MQ)
//       {
//          if (Q)
//          {
//             w *= Q->Eval(*Trans.Elem1, eip1);
//          }
//          ni.Set(w, nor);
//       }
//       else
//       {
//          nh.Set(w, nor);
//          MQ->Eval(mq, *Trans.Elem1, eip1);
//          mq.MultTranspose(nh, ni);
//       }
//       CalcAdjugate(Trans.Elem1->Jacobian(), adjJ);
//       adjJ.Mult(ni, nh);
//       if (kappa_is_nonzero)
//       {
//          wq = ni * nor;
//       }
//       // Note: in the jump term, we use 1/h1 = |nor|/det(J1) which is
//       // independent of Loc1 and always gives the size of element 1 in
//       // direction perpendicular to the face. Indeed, for linear transformation
//       //     |nor|=measure(face)/measure(ref. face),
//       //   det(J1)=measure(element)/measure(ref. element),
//       // and the ratios measure(ref. element)/measure(ref. face) are
//       // compatible for all element/face pairs.
//       // For example: meas(ref. tetrahedron)/meas(ref. triangle) = 1/3, and
//       // for any tetrahedron vol(tet)=(1/3)*height*area(base).
//       // For interior faces: q_e/h_e=(q1/h1+q2/h2)/2.

//       dshape1.Mult(nh, dshape1dn);
//       for (int i = 0; i < ndof1; i++)
//          for (int j = 0; j < ndof1; j++)
//          {
//             elmat(i, j) += shape1(i) * dshape1dn(j);
//          }

//       if (ndof2)
//       {
//          el2.CalcShape(eip2, shape2);
//          el2.CalcDShape(eip2, dshape2);
//          w = ip.weight/2/Trans.Elem2->Weight();
//          if (!MQ)
//          {
//             if (Q)
//             {
//                w *= Q->Eval(*Trans.Elem2, eip2);
//             }
//             ni.Set(w, nor);
//          }
//          else
//          {
//             nh.Set(w, nor);
//             MQ->Eval(mq, *Trans.Elem2, eip2);
//             mq.MultTranspose(nh, ni);
//          }
//          CalcAdjugate(Trans.Elem2->Jacobian(), adjJ);
//          adjJ.Mult(ni, nh);
//          if (kappa_is_nonzero)
//          {
//             wq += ni * nor;
//          }

//          dshape2.Mult(nh, dshape2dn);

//          for (int i = 0; i < ndof1; i++)
//             for (int j = 0; j < ndof2; j++)
//             {
//                elmat(i, ndof1 + j) += shape1(i) * dshape2dn(j);
//             }

//          for (int i = 0; i < ndof2; i++)
//             for (int j = 0; j < ndof1; j++)
//             {
//                elmat(ndof1 + i, j) -= shape2(i) * dshape1dn(j);
//             }

//          for (int i = 0; i < ndof2; i++)
//             for (int j = 0; j < ndof2; j++)
//             {
//                elmat(ndof1 + i, ndof1 + j) -= shape2(i) * dshape2dn(j);
//             }
//       }
//    }
// }


// } // namespace mach
