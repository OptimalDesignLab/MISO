// #ifdef MFEM_USE_PUMI

// #include "apfMDS.h"
// #include "mfem.hpp"

// #include "therm_integ.hpp"

// using namespace mfem;

// namespace mach
// {

// void InteriorBoundaryOutFluxInteg::AssembleFaceVector(
//    const FiniteElement &el1,
//    const FiniteElement &el2,
//    FaceElementTransformations &face_trans,
//    const Vector &elfun,
//    Vector &elvect)
// {
//    int dim, ndof1, ndof2, ndofs;
//    double w1, w2;

//    dim = el1.GetDim();

//    nor.SetSize(dim);
//    nh.SetSize(dim);
//    adjJ.SetSize(dim);

//    ndof1 = el1.GetDof();
//    shape1.SetSize(ndof1);
//    dshape1.SetSize(ndof1, dim);
//    dshape1dn.SetSize(ndof1);

//    ndof2 = el2.GetDof();
//    shape2.SetSize(ndof2);
//    dshape2.SetSize(ndof2, dim);
//    dshape2dn.SetSize(ndof2);

//    Vector el1state(elfun.GetData(), ndof1);
//    Vector el2state(elfun.GetData() + ndof1, ndof2);

//    ndofs = ndof1 + ndof2;
//    elvect.SetSize(ndofs);
//    elvect = 0.0;

//    auto *ent = getMdsEntity(pumi_mesh, 2, face_trans.Face->ElementNo);
//    auto *me = pumi_mesh->toModel(ent);
//    int me_dim = pumi_mesh->getModelType(me);
//    if (me_dim != 2) {return;} // face classified on a model region
//    int tag = pumi_mesh->getModelTag(me);
//    if (faces.count(tag) == 0) {return;} // face is on a model face, but not one we want

//    Vector el1vect(elvect.GetData(), ndof1);
//    Vector el2vect(elvect.GetData() + ndof1, ndof2);

//    bool el1_is_source = face_trans.Elem1->Attribute == source_attr ? 
//                         true : false;

//    const IntegrationRule *ir = IntRule;
//    if (ir == NULL)
//    {
//       // a simple choice for the integration order; is this OK?
//       int order;
//       order = 2*std::max(el1.GetOrder(), el2.GetOrder());
//       ir = &IntRules.Get(face_trans.GetGeometryType(), order);
//    }

//    for (int p = 0; p < ir->GetNPoints(); p++)
//    {
//       const IntegrationPoint &ip = ir->IntPoint(p);

//       // Set the integration point in the face and the neighboring elements
//       face_trans.SetAllIntPoints(&ip);

//       // Access the neighboring elements' integration points
//       // Note: eip2 will only contain valid data if Elem2 exists
//       const IntegrationPoint &eip1 = face_trans.GetElement1IntPoint();
//       const IntegrationPoint &eip2 = face_trans.GetElement2IntPoint();

//       if (dim == 1)
//       {
//          nor(0) = 2*eip1.x - 1.0;
//       }
//       else
//       {
//          CalcOrtho(face_trans.Jacobian(), nor);
//       }

//       el1.CalcShape(eip1, shape1);
//       el2.CalcShape(eip2, shape2);

//       el1.CalcDShape(eip1, dshape1);
//       CalcAdjugate(face_trans.Elem1->Jacobian(), adjJ);
//       adjJ.Mult(nor, nh);
//       dshape1.Mult(nh, dshape1dn);

//       el2.CalcDShape(eip2, dshape2);
//       CalcAdjugate(face_trans.Elem2->Jacobian(), adjJ);
//       adjJ.Mult(nor, nh);
//       dshape2.Mult(nh, dshape2dn);
   
//       w1 = ip.weight;
//       w2 = ip.weight;

//       double el1flux, el2flux;
//       if (el1_is_source)
//       {
//          w1 *= nor.Norml2() * H.Eval(*face_trans.Elem1, eip1);
//          w2 *= -K.Eval(*face_trans.Elem2, eip2) / face_trans.Elem2->Weight();

//          el1flux = (shape1 * el1state) - ambient_temp;
//          el2flux = dshape2dn * el2state;
//       }
//       else 
//       {
//          w1 *= K.Eval(*face_trans.Elem1, eip1) / face_trans.Elem2->Weight();
//          w2 *= nor.Norml2() * H.Eval(*face_trans.Elem2, eip2);

//          el1flux = dshape1dn * el1state;
//          el2flux = (shape2 * el2state) - ambient_temp;
//       }

//       add(el1vect, w1*el1flux, shape1, el1vect);
//       add(el2vect, w2*el2flux, shape2, el2vect);
//    }
// }

// void InteriorBoundaryOutFluxInteg::AssembleFaceMatrix(
//    const FiniteElement &el1,
//    const FiniteElement &el2,
//    FaceElementTransformations &face_trans,
//    DenseMatrix &elmat)
// {
//    int dim, ndof1, ndof2, ndofs;
//    double w1, w2;

//    dim = el1.GetDim();

//    nor.SetSize(dim);
//    nh.SetSize(dim);
//    adjJ.SetSize(dim);

//    ndof1 = el1.GetDof();
//    shape1.SetSize(ndof1);
//    dshape1.SetSize(ndof1, dim);
//    dshape1dn.SetSize(ndof1);

//    ndof2 = el2.GetDof();
//    shape2.SetSize(ndof2);
//    dshape2.SetSize(ndof2, dim);
//    dshape2dn.SetSize(ndof2);

//    ndofs = ndof1 + ndof2;
//    elmat.SetSize(ndofs);
//    elmat = 0.0;

//    auto *ent = getMdsEntity(pumi_mesh, 2, face_trans.Face->ElementNo);
//    auto *me = pumi_mesh->toModel(ent);
//    int me_dim = pumi_mesh->getModelType(me);
//    if (me_dim != 2) {return;} // face classified on a model region
//    int tag = pumi_mesh->getModelTag(me);
//    if (faces.count(tag) == 0) {return;} // face is on a model face, but not one we want
//    bool el1_is_source = face_trans.Elem1->Attribute == source_attr ? 
//                         true : false;

//    const IntegrationRule *ir = IntRule;
//    if (ir == NULL)
//    {
//       int order;
//       order = 2*std::max(el1.GetOrder(), el2.GetOrder());
//       ir = &IntRules.Get(face_trans.GetGeometryType(), order);
//    }

//    for (int p = 0; p < ir->GetNPoints(); p++)
//    {
//       const IntegrationPoint &ip = ir->IntPoint(p);

//       // Set the integration point in the face and the neighboring elements
//       face_trans.SetAllIntPoints(&ip);

//       // Access the neighboring elements' integration points
//       // Note: eip2 will only contain valid data if Elem2 exists
//       const IntegrationPoint &eip1 = face_trans.GetElement1IntPoint();
//       const IntegrationPoint &eip2 = face_trans.GetElement2IntPoint();

//       if (dim == 1)
//       {
//          nor(0) = 2*eip1.x - 1.0;
//       }
//       else
//       {
//          CalcOrtho(face_trans.Jacobian(), nor);
//       }

//       el1.CalcShape(eip1, shape1);
//       el2.CalcShape(eip2, shape2);

//       el1.CalcDShape(eip1, dshape1);
//       CalcAdjugate(face_trans.Elem1->Jacobian(), adjJ);
//       adjJ.Mult(nor, nh);
//       dshape1.Mult(nh, dshape1dn);

//       el2.CalcDShape(eip2, dshape2);
//       CalcAdjugate(face_trans.Elem2->Jacobian(), adjJ);
//       adjJ.Mult(nor, nh);
//       dshape2.Mult(nh, dshape2dn);
   
//       w1 = ip.weight;
//       w2 = ip.weight;

//       if (el1_is_source)
//       {
//          w1 *= nor.Norml2() * H.Eval(*face_trans.Elem1, eip1);
//          w2 *= -K.Eval(*face_trans.Elem2, eip2) / face_trans.Elem2->Weight();

//          for (int i = 0; i < ndof1; i++)
//          {
//             for (int j = 0; j < ndof1; j++)
//             {
//                elmat(i, j) += w1 * shape1(i) * shape1(j);
//             }
//          }

//          for (int i = 0; i < ndof2; i++)
//          {
//             for (int j = 0; j < ndof2; j++)
//             {
//                elmat(ndof1 + i, ndof1 + j) += w2 * shape2(i) * dshape2dn(j);
//             }
//          }
//       }
//       else 
//       {
//          w1 *= K.Eval(*face_trans.Elem1, eip1) / face_trans.Elem1->Weight();
//          w2 *= nor.Norml2() * H.Eval(*face_trans.Elem2, eip2);

//          for (int i = 0; i < ndof1; i++)
//          {
//             for (int j = 0; j < ndof1; j++)
//             {
//                elmat(i, j) += w1 * shape1(i) * dshape1dn(j);
//             }
//          }

//          for (int i = 0; i < ndof2; i++)
//          {
//             for (int j = 0; j < ndof2; j++)
//             {
//                elmat(ndof1 + i, ndof1 + j) += w2 * shape2(i) * shape2(j);
//             }
//          }
//       }
//    }
// }

// } // namespace mach

// #endif
