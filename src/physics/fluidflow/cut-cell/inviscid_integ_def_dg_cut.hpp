#ifndef MACH_INVISCID_INTEG_DEF_DG_CUT
#define MACH_INVISCID_INTEG_DEF_DG_CUT

#include "mfem.hpp"

#include "utils.hpp"
#include "sbp_fe.hpp"
#include "inviscid_integ_dg_cut.hpp"

namespace mach
{
template <typename Derived>
double CutDGInviscidIntegrator<Derived>::GetElementEnergy(
    const mfem::FiniteElement &el,
    mfem::ElementTransformation &trans,
    const mfem::Vector &elfun)
{
   if (embeddedElements.at(trans.ElementNo) == true)
   {
      // cout << "embedded element " << trans.ElementNo << endl;
      return 0.0;
   }
   // int dof = el.GetDof();
   double energy;
   const IntegrationRule *ir;
   ir = cutSquareIntRules[trans.ElementNo];
   if (!ir)
   {
      ir = &(IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 3));  // <---
   }
   // if (ir == NULL)
   // {
   //    return 0.0;
   // }
   energy = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);
      energy += ip.weight * trans.Weight();
   }
   return energy;
}

template <typename Derived>
void CutDGInviscidIntegrator<Derived>::AssembleElementVector(
    const mfem::FiniteElement &el,
    mfem::ElementTransformation &trans,
    const mfem::Vector &elfun,
    mfem::Vector &elvect)
{
   using namespace mfem;
   using namespace std;
   const int num_nodes = el.GetDof();
   int dim = el.GetDim();
   elvect.SetSize(num_states * num_nodes);
   elvect = 0.0;
   if (embeddedElements.at(trans.ElementNo) == true)
   {
      elvect = 0.0;
   }
   else
   {
      DenseMatrix u_mat(elfun.GetData(), num_nodes, num_states);
      DenseMatrix res(elvect.GetData(), num_nodes, num_states);
      DenseMatrix adjJ_i, elflux, dshape, dshapedx;
      Vector shape, dxidx, dshapedxi, fluxi, u;
      u.SetSize(num_states);
      dxidx.SetSize(dim);
      fluxi.SetSize(num_states);
      dshapedxi.SetSize(num_nodes);
      shape.SetSize(num_nodes);
      dshape.SetSize(num_nodes, dim);
      dshapedx.SetSize(num_nodes, dim);
      elflux.SetSize(num_states, dim);
      adjJ_i.SetSize(dim);
      int intorder = trans.OrderGrad(&el) + trans.Order() + el.GetOrder();
      const IntegrationRule *ir;  // = IntRule;
      ir = cutSquareIntRules[trans.ElementNo];
      if (ir == NULL)
      {
         ir = &IntRules.Get(el.GetGeomType(), intorder);
      }
      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         trans.SetIntPoint(&ip);
         // Calculate the shape function
         el.CalcShape(ip, shape);
         // Compute the physical gradient
         el.CalcDShape(ip, dshape);
         // Mult(dshape, trans.AdjugateJacobian(), dshapedx);
         u_mat.MultTranspose(shape, u);
         CalcAdjugate(trans.Jacobian(), adjJ_i);
         for (int di = 0; di < dim; ++di)
         {
            adjJ_i.GetRow(di, dxidx);
            flux(dxidx, u, fluxi);
            dshape.GetColumn(di, dshapedxi);
            AddMult_a_VWt(-ip.weight, dshapedxi, fluxi, res);
         }
      }
      res *= alpha;
   }
}

template <typename Derived>
void CutDGInviscidIntegrator<Derived>::AssembleElementGrad(
    const mfem::FiniteElement &el,
    mfem::ElementTransformation &trans,
    const mfem::Vector &elfun,
    mfem::DenseMatrix &elmat)
{
   // int num_nodes = el.GetDof();
   // int ndof = elfun.Size();
   // elmat.SetSize(ndof);
   // elmat = 0.0;
   // double delta = 1e-5;
   // for (int i = 0; i < ndof; ++i)
   // {
   //    Vector elfun_plus(elfun);
   //    Vector elfun_minus(elfun);
   //    elfun_plus(i) += delta;
   //    Vector elvect_plus;
   //    AssembleElementVector(el, trans, elfun_plus, elvect_plus);
   //    elfun_minus(i) -= delta;
   //    Vector elvect_minus;
   //    AssembleElementVector(el, trans, elfun_minus, elvect_minus);

   //    elvect_plus -= elvect_minus;
   //    elvect_plus /= 2 * delta;

   //    for (int j = 0; j < ndof; ++j)
   //    {
   //       elmat(j, i) = elvect_plus(j);
   //    }
   // }
#if 1
   using namespace mfem;
   int num_nodes = el.GetDof();
   int dim = el.GetDim();
   int ndof = elfun.Size();
   elmat.SetSize(ndof);
   elmat = 0.0;
   if (embeddedElements.at(trans.ElementNo) == true)
   {
      elmat = 0.0;
   }
   else
   {
      DenseMatrix u_mat(elfun.GetData(), num_nodes, num_states);
      DenseMatrix adjJ_i, elflux, dshape, dshapedx;
      Vector shape, dxidx, dshapedxi, fluxi, u, ul;
      u.SetSize(num_states);
      ui.SetSize(num_states);
      dxidx.SetSize(dim);
      flux_jaci.SetSize(num_states);
      dshapedxi.SetSize(num_nodes);
      shape.SetSize(num_nodes);
      dshape.SetSize(num_nodes, dim);
      dshapedx.SetSize(num_nodes, dim);
      elflux.SetSize(num_states, dim);
      adjJ_i.SetSize(dim);
      int intorder = trans.OrderGrad(&el) + trans.Order() + el.GetOrder();
      const IntegrationRule *ir;  // = IntRule;
      ir = cutSquareIntRules[trans.ElementNo];
      if (ir == NULL)
      {
         ir = &IntRules.Get(el.GetGeomType(), intorder);
      }

      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         trans.SetIntPoint(&ip);
         // Calculate the shape function
         el.CalcShape(ip, shape);
         // Compute the physical gradient
         el.CalcDShape(ip, dshape);
         u_mat.MultTranspose(shape, u);
         CalcAdjugate(trans.Jacobian(), adjJ_i);
         double Q;
         for (int di = 0; di < dim; ++di)
         {
            dshape.GetColumn(di, dshapedxi);
            for (int l = 0; l < num_nodes; ++l)
            {
               adjJ_i.GetRow(di, dxidx);
               fluxJacState(dxidx, u, flux_jaci);
               for (int k = 0; k < num_nodes; ++k)
               {
                  Q = dshapedxi(k) * shape(l);
                  for (int n = 0; n < dim + 2; ++n)
                  {
                     for (int m = 0; m < dim + 2; ++m)
                     {
                        elmat(m * num_nodes + k, n * num_nodes + l) -=
                            ip.weight * Q * flux_jaci(m, n);
                     }
                  }
               }
            }
         }
      }
   }
#endif
}
template <typename Derived>
double CutDGEulerBoundaryIntegrator<Derived>::GetFaceEnergy(
    const mfem::FiniteElement &el_bnd,
    const mfem::FiniteElement &el_unused,
    mfem::FaceElementTransformations &trans,
    const mfem::Vector &elfun)
{
   using namespace mfem;
   const int num_nodes = el_bnd.GetDof();
   int dim = el_bnd.GetDim();
#ifdef MFEM_THREAD_SAFE
   Vector u_face, x, nrm, flux_face;
#endif
   u_face.SetSize(num_states);
   x.SetSize(dim);
   nrm.SetSize(dim);
   shape.SetSize(num_nodes);
   double fun = 0.0;  // initialize the functional value
   DenseMatrix u(elfun.GetData(), num_nodes, num_states);
   const IntegrationRule *ir;
   if (embeddedElements.at(trans.Elem1No) == true)
   {
      return 0;
   }
   // ir = cutBdrFaceIntRules[trans.Elem1No];
   ir = cutBdrFaceIntRules[trans.Elem1No];
   if (!ir)
   {
      // cout << "int rule not available for face with el id: " << trans.Elem1No
      // << endl;
      int intorder;
      intorder = trans.Elem1->OrderW() + 2 * el_bnd.GetOrder();
      ir = &IntRules.Get(trans.FaceGeom, intorder);
   }
   IntegrationPoint el_ip;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &face_ip = ir->IntPoint(i);
      trans.Loc1.Transform(face_ip, el_ip);
      trans.Elem1->Transform(el_ip, x);
      el_bnd.CalcShape(el_ip, shape);
      u.MultTranspose(shape, u_face);
      // get the normal vector, and then add contribution to function
      trans.Face->SetIntPoint(&face_ip);
      CalcOrtho(trans.Face->Jacobian(), nrm);
      // fun += face_ip.weight * trans.Weight();
      fun += bndryFun(x, nrm, u_face) * face_ip.weight * alpha;
   }
   // cout << "edge length for element " << trans.Elem1No << " : " << fun <<
   // endl;
   return fun;
}

template <typename Derived>
void CutDGEulerBoundaryIntegrator<Derived>::AssembleFaceVector(
    const mfem::FiniteElement &el_bnd,
    const mfem::FiniteElement &el_unused,
    mfem::FaceElementTransformations &trans,
    const mfem::Vector &elfun,
    mfem::Vector &elvect)
{
   using namespace mfem;
   const int dof = el_bnd.GetDof();
#ifdef MFEM_THREAD_SAFE
   Vector u_face, x, nrm, flux_face, shape;
#endif
   int dim = el_bnd.GetDim();
   u_face.SetSize(num_states);
   x.SetSize(dim);
   nrm.SetSize(dim);
   flux_face.SetSize(num_states);
   elvect.SetSize(num_states * dof);
   elvect = 0.0;
   shape.SetSize(dof);
   DenseMatrix u(elfun.GetData(), dof, num_states);
   DenseMatrix res(elvect.GetData(), dof, num_states);
   if (embeddedElements.at(trans.Elem1No) == true)
   {
      elvect = 0.0;
   }
   else
   {
      const IntegrationRule *ir;
      ir = cutBdrFaceIntRules[trans.Elem1No];
      if (!ir)
      {
         int intorder;
         intorder = trans.Elem1->OrderW() + 2 * el_bnd.GetOrder();
         ir = &IntRules.Get(trans.FaceGeom, intorder);
      }

      IntegrationPoint eip1;
      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         trans.Loc1.Transform(ip, eip1);
         trans.Elem1->Transform(eip1, x);
         el_bnd.CalcShape(eip1, shape);
         // get the normal vector and the flux on the face
         trans.Face->SetIntPoint(&ip);
         CalcOrtho(trans.Face->Jacobian(), nrm);
         // Interpolate elfun at the point
         u.MultTranspose(shape, u_face);
         flux(x, nrm, u_face, flux_face);
         flux_face *= ip.weight;
         // multiply by test function
         for (int n = 0; n < num_states; ++n)
         {
            for (int s = 0; s < dof; s++)
            {
               res(s, n) += shape(s) * flux_face(n);
            }
         }
      }
      res *= alpha;
   }
}

template <typename Derived>
void CutDGEulerBoundaryIntegrator<Derived>::AssembleFaceGrad(
    const mfem::FiniteElement &el_bnd,
    const mfem::FiniteElement &el_unused,
    mfem::FaceElementTransformations &trans,
    const mfem::Vector &elfun,
    mfem::DenseMatrix &elmat)
{
   using namespace mfem;
   int dim = el_bnd.GetDim();
   const int dof = el_bnd.GetDof();
   int ndof = elfun.Size();
   elmat.SetSize(ndof);
   u_face.SetSize(num_states);
   x.SetSize(dim);
   nrm.SetSize(dim);
   flux_face.SetSize(num_states);
   shape.SetSize(dof);
   DenseMatrix u(elfun.GetData(), dof, num_states);
   elmat = 0.0;
   flux_jac_face.SetSize(num_states);
   if (embeddedElements.at(trans.Elem1No) == true)
   {
      elmat = 0.0;
   }
   else
   {
      const IntegrationRule *ir;
      ir = cutBdrFaceIntRules[trans.Elem1No];
      if (!ir)
      {
         int intorder;
         intorder = trans.Elem1->OrderW() + 2 * el_bnd.GetOrder();
         ir = &IntRules.Get(trans.FaceGeom, intorder);
      }
      IntegrationPoint eip1;
      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         trans.Loc1.Transform(ip, eip1);
         trans.Elem1->Transform(eip1, x);
         el_bnd.CalcShape(eip1, shape);
         // get the normal vector and the flux on the face
         trans.Face->SetIntPoint(&ip);
         CalcOrtho(trans.Face->Jacobian(), nrm);
         // Interpolate elfun at the point
         u.MultTranspose(shape, u_face);
         // flux(x, nrm, u_face, flux_face);
         fluxJacState(x, nrm, u_face, flux_jac_face);
         for (int j = 0; j < dof; ++j)
         {
            for (int k = 0; k < dof; ++k)
            {
               double Q = shape(j) * shape(k);
               // multiply by test function
               for (int n = 0; n < num_states; ++n)
               {
                  for (int m = 0; m < num_states; ++m)
                  {
                     // res(j, n) += alpha*flux_face(n);
                     elmat(m * dof + k, n * dof + j) +=
                         ip.weight * Q * alpha * flux_jac_face(m, n);
                  }
               }
            }
         }
      }
   }
}
template <typename Derived>
double CutDGInviscidBoundaryIntegrator<Derived>::GetElementEnergy(
    const mfem::FiniteElement &el_bnd,
    mfem::ElementTransformation &trans,
    const mfem::Vector &elfun)
{
   using namespace mfem;
   const int num_nodes = el_bnd.GetDof();
   int dim = el_bnd.GetDim();
#ifdef MFEM_THREAD_SAFE
   Vector u_face, x, nrm, flux_face;
#endif
   u_face.SetSize(num_states);
   x.SetSize(dim);
   nrm.SetSize(dim);
   shape.SetSize(num_nodes);
   double fun = 0.0;  // initialize the functional value
   const IntegrationRule *ir;
   ir = cutSegmentIntRules[trans.ElementNo];
   if (!(ir))
   {
      return 0;
   }
   DenseMatrix u(elfun.GetData(), num_nodes, num_states);
   // IntegrationPoint el_ip;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &face_ip = ir->IntPoint(i);
      // get the normal vector, and then add contribution to function
      trans.SetIntPoint(&face_ip);
      trans.Transform(face_ip, x);
      el_bnd.CalcShape(face_ip, shape);
      // get the normal vector and the flux on the face
      // calculate normal vector using level-set function
      double nx;
      double ny;
      double ds;
      // double xc, yc;
      // xc = 0.0;
      // yc = 0.0;
      // nx = 2 * (x(0) - xc);
      // ny = 2 * (x(1) - yc);
      // ds = sqrt((nx * nx) + (ny * ny));
      // nrm(0) = phi.sign_phi * nx / ds;
      // nrm(1) = phi.sign_phi * ny / ds;
      /// n_hat = grad_phi/|\grad_phi|
      TinyVector<double, 2> beta, xs;
      xs(0) = x(0);
      xs(1) = x(1);
      beta = phi.grad(xs);
      ds = mag(beta);
      nx = beta(0);
      ny = beta(1);
      nrm(0) = nx / ds;
      nrm(1) = ny / ds;
      // Interpolate elfun at the point
      u.MultTranspose(shape, u_face);
      double area = sqrt(trans.Weight());
      //fun += face_ip.weight * alpha * area;
      fun += bndryFun(x, nrm, u_face) * face_ip.weight * sqrt(trans.Weight()) *
             alpha;
   }
   return fun;
}

template <typename Derived>
void CutDGInviscidBoundaryIntegrator<Derived>::AssembleElementVector(
    const mfem::FiniteElement &el_bnd,
    mfem::ElementTransformation &trans,
    const mfem::Vector &elfun,
    mfem::Vector &elvect)
{
   using namespace mfem;
   const int dof = el_bnd.GetDof();
#ifdef MFEM_THREAD_SAFE
   Vector u_face, x, nrm, flux_face, shape;
#endif
   int dim = el_bnd.GetDim();
   u_face.SetSize(num_states);
   x.SetSize(dim);
   nrm.SetSize(dim);
   flux_face.SetSize(num_states);
   elvect.SetSize(num_states * dof);
   elvect = 0.0;
   const IntegrationRule *ir;
   ir = cutSegmentIntRules[trans.ElementNo];
   if (ir == NULL)
   {
      elvect = 0.0;
   }
   else
   {
      shape.SetSize(dof);
      DenseMatrix u(elfun.GetData(), dof, num_states);
      DenseMatrix res(elvect.GetData(), dof, num_states);
      // IntegrationPoint eip1;
      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         trans.Transform(ip, x);
         trans.SetIntPoint(&ip);
         el_bnd.CalcShape(ip, shape);
         // get the normal vector and the flux on the face
         // calculate normal vector using level-set function
         double nx;
         double ny;
         double ds;
         // double xc, yc;
         // xc = 0.0;
         // yc = 0.0;
         // nx = 2 * (x(0) - xc);
         // ny = 2 * (x(1) - yc);
         // /// n_hat = grad_phi/|\grad_phi|
         // ds = sqrt((nx * nx) + (ny * ny));
         // nrm(0) = phi.sign_phi * nx / ds;
         // nrm(1) = phi.sign_phi * ny / ds;
         TinyVector<double, 2> beta, xs;
         xs(0) = x(0);
         xs(1) = x(1);
         beta = phi.grad(xs);
         ds = mag(beta);
         nx = beta(0);
         ny = beta(1);
         nrm(0) = nx / ds;
         nrm(1) = ny / ds;
         // Interpolate elfun at the point
         u.MultTranspose(shape, u_face);
         flux(x, nrm, u_face, flux_face);
         flux_face *= ip.weight * sqrt(trans.Weight());
         // multiply by test function
         for (int n = 0; n < num_states; ++n)
         {
            for (int s = 0; s < dof; s++)
            {
               res(s, n) += shape(s) * flux_face(n);
            }
         }
      }
      res *= alpha;
   }
}

template <typename Derived>
void CutDGInviscidBoundaryIntegrator<Derived>::AssembleElementGrad(
    const mfem::FiniteElement &el_bnd,
    mfem::ElementTransformation &trans,
    const mfem::Vector &elfun,
    mfem::DenseMatrix &elmat)
{
   using namespace mfem;
   int dim = el_bnd.GetDim();
   const int dof = el_bnd.GetDof();
   int ndof = elfun.Size();
   elmat.SetSize(ndof);
   u_face.SetSize(num_states);
   x.SetSize(dim);
   nrm.SetSize(dim);
   flux_face.SetSize(num_states);
   shape.SetSize(dof);
   DenseMatrix u(elfun.GetData(), dof, num_states);
   elmat = 0.0;
   flux_jac_face.SetSize(num_states);
   // int intorder;
   // intorder = trans.Elem1->OrderW() + 2 * el_bnd.GetOrder();
   const IntegrationRule *ir;  // = &IntRules.Get(trans.FaceGeom, intorder);
   ir = cutSegmentIntRules[trans.ElementNo];
   if (ir == NULL)
   {
      elmat = 0.0;
   }
   else
   {
      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         trans.Transform(ip, x);
         // get the normal vector and the flux on the face
         trans.SetIntPoint(&ip);
         el_bnd.CalcShape(ip, shape);
         double nx;
         double ny;
         double ds;
         // double xc, yc;
         // xc = 0.0;
         // yc = 0.0;
         // nx = 2 * (x(0) - xc);
         // ny = 2 * (x(1) - yc);
         // ds = sqrt((nx * nx) + (ny * ny));
         // nrm(0) = phi.sign_phi * nx / ds;
         // nrm(1) = phi.sign_phi * ny / ds;
         TinyVector<double, 2> beta, xs;
         xs(0) = x(0);
         xs(1) = x(1);
         beta = phi.grad(xs);
         ds = mag(beta);
         nx = beta(0);
         ny = beta(1);
         nrm(0) = nx / ds;
         nrm(1) = ny / ds;
         // Interpolate elfun at the point
         u.MultTranspose(shape, u_face);
         // flux(x, nrm, u_face, flux_face);
         fluxJacState(x, nrm, u_face, flux_jac_face);
         for (int j = 0; j < dof; ++j)
         {
            for (int k = 0; k < dof; ++k)
            {
               double Q = shape(j) * shape(k);
               // multiply by test function
               for (int n = 0; n < num_states; ++n)
               {
                  for (int m = 0; m < num_states; ++m)
                  {
                     // res(j, n) += alpha*flux_face(n);
                     elmat(m * dof + k, n * dof + j) += ip.weight * Q * alpha *
                                                        flux_jac_face(m, n) *
                                                        sqrt(trans.Weight());
                  }
               }
            }
         }
      }
   }
}

template <typename Derived>
double CutDGInviscidFaceIntegrator<Derived>::GetFaceEnergy(
    const mfem::FiniteElement &el_left,
    const mfem::FiniteElement &el_right,
    mfem::FaceElementTransformations &trans,
    const mfem::Vector &elfun)
{
   return 0.0;
}

template <typename Derived>
void CutDGInviscidFaceIntegrator<Derived>::AssembleFaceVector(
    const mfem::FiniteElement &el_left,
    const mfem::FiniteElement &el_right,
    mfem::FaceElementTransformations &trans,
    const mfem::Vector &elfun,
    mfem::Vector &elvect)
{
   using namespace mfem;
   using namespace std;
#ifdef MFEM_THREAD_SAFE
   Vector shape1, shape2, funval1, funval2, nrm, fluxN;
#endif

   // Compute the term <F.n(u),[w]> on the interior faces.
   int dim = el_left.GetDim();
   const int dof1 = el_left.GetDof();
   const int dof2 = el_right.GetDof();
   // int dim = el_left.GetDim();
   nrm.SetSize(dim);
   shape1.SetSize(dof1);
   shape2.SetSize(dof2);
   u_face_left.SetSize(num_states);
   u_face_right.SetSize(num_states);
   fluxN.SetSize(num_states);
   elvect.SetSize((dof1 + dof2) * num_states);
   elvect = 0.0;
   if (immersedFaces[trans.Face->ElementNo] == true)
   {
      return;
   }
   DenseMatrix elfun1_mat(elfun.GetData(), dof1, num_states);
   DenseMatrix elfun2_mat(
       elfun.GetData() + dof1 * num_states, dof2, num_states);

   DenseMatrix elvect1_mat(elvect.GetData(), dof1, num_states);
   DenseMatrix elvect2_mat(
       elvect.GetData() + dof1 * num_states, dof2, num_states);
   const IntegrationRule *ir;
   ir = cutInteriorFaceIntRules[trans.Face->ElementNo];
   if (ir == NULL)
   {
      // Integration order calculation from DGTraceIntegrator
      int intorder;
      if (trans.Elem2No >= 0)
         intorder = (min(trans.Elem1->OrderW(), trans.Elem2->OrderW()) +
                     2 * max(el_left.GetOrder(), el_right.GetOrder()));
      else
      {
         intorder = trans.Elem1->OrderW() + 2 * el_left.GetOrder();
      }
      if (el_left.Space() == FunctionSpace::Pk)
      {
         intorder++;
      }
      ir = &IntRules.Get(trans.GetGeometryType(), intorder);
   }

   // cout << "face elements are " << trans.Elem1No << " , " << trans.Elem2No <<
   // endl;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
#if 0
      IntegrationPoint eip1;
      IntegrationPoint eip2;
      trans.Loc1.Transform(ip, eip1);
      trans.Loc2.Transform(ip, eip2);

      // Calculate basis functions on both elements at the face
      el_left.CalcShape(eip1, shape1);
      el_right.CalcShape(eip2, shape2);
#endif

      trans.SetAllIntPoints(&ip);  // set face and element int. points
      // Calculate basis functions on both elements at the face
      el_left.CalcShape(trans.GetElement1IntPoint(), shape1);
      el_right.CalcShape(trans.GetElement2IntPoint(), shape2);
      // Interpolate elfun at the point
      elfun1_mat.MultTranspose(shape1, u_face_left);
      elfun2_mat.MultTranspose(shape2, u_face_right);

      trans.Face->SetIntPoint(&ip);

      // Get the normal vector and the flux on the face
      CalcOrtho(trans.Face->Jacobian(), nrm);

      flux(nrm, u_face_left, u_face_right, fluxN);

      fluxN *= ip.weight;
      for (int k = 0; k < num_states; k++)
      {
         for (int s = 0; s < dof1; s++)
         {
            elvect1_mat(s, k) += fluxN(k) * shape1(s);
         }
         for (int s = 0; s < dof2; s++)
         {
            elvect2_mat(s, k) -= fluxN(k) * shape2(s);
         }
      }
      elvect *= alpha;
   }
}

template <typename Derived>
void CutDGInviscidFaceIntegrator<Derived>::AssembleFaceGrad(
    const mfem::FiniteElement &el_left,
    const mfem::FiniteElement &el_right,
    mfem::FaceElementTransformations &trans,
    const mfem::Vector &elfun,
    mfem::DenseMatrix &elmat)
{
   using namespace mfem;
   using namespace std;
#ifdef MFEM_THREAD_SAFE
   Vector shape1, shape2, funval1, funval2, nrm, fluxN;
   DenseMatrix flux_jac_left, flux_jac_right;
#endif
   // Compute the term <F.n(u),[w]> on the interior faces.
   const int dof1 = el_left.GetDof();
   const int dof2 = el_right.GetDof();
   int dim = el_left.GetDim();
   nrm.SetSize(dim);
   flux_jac_left.SetSize(num_states);
   flux_jac_right.SetSize(num_states);
   shape1.SetSize(dof1);
   shape2.SetSize(dof2);
   u_face_left.SetSize(num_states);
   u_face_right.SetSize(num_states);
   fluxN.SetSize(num_states);
   elmat.SetSize((dof1 + dof2) * num_states);
   elmat = 0.0;
   if (immersedFaces[trans.Face->ElementNo] == true)
   {
      return;
   }
   DenseMatrix elfun1_mat(elfun.GetData(), dof1, num_states);
   DenseMatrix elfun2_mat(
       elfun.GetData() + dof1 * num_states, dof2, num_states);

   // Integration order calculation from DGTraceIntegrator
   const IntegrationRule *ir;
   ir = cutInteriorFaceIntRules[trans.Face->ElementNo];
   if (ir == NULL)
   {
      int intorder;
      if (trans.Elem2No >= 0)
         intorder = (min(trans.Elem1->OrderW(), trans.Elem2->OrderW()) +
                     2 * max(el_left.GetOrder(), el_right.GetOrder()));
      else
      {
         intorder = trans.Elem1->OrderW() + 2 * el_left.GetOrder();
      }

      ir = &IntRules.Get(trans.FaceGeom, intorder);
   }
   // cout << "face elements are " << trans.Elem1No << " , " << trans.Elem2No <<
   // endl;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      IntegrationPoint eip1;
      IntegrationPoint eip2;
      trans.Loc1.Transform(ip, eip1);
      trans.Loc2.Transform(ip, eip2);

      // Calculate basis functions on both elements at the face
      el_left.CalcShape(eip1, shape1);
      el_right.CalcShape(eip2, shape2);

      // Interpolate elfun at the point
      elfun1_mat.MultTranspose(shape1, u_face_left);
      elfun2_mat.MultTranspose(shape2, u_face_right);

      trans.Face->SetIntPoint(&ip);

      // Get the normal vector and the flux on the face
      CalcOrtho(trans.Face->Jacobian(), nrm);

      fluxJacStates(
          nrm, u_face_left, u_face_right, flux_jac_left, flux_jac_right);
      // insert flux Jacobians into element stiffness matrices
      const int offset = num_states * dof1;
      double Q;
      for (int k = 0; k < dof1; ++k)
      {
         for (int j = 0; j < dof2; ++j)
         {
            Q = shape1(k) * shape2(j);
            // multiply by test function
            for (int n = 0; n < num_states; ++n)
            {
               for (int m = 0; m < num_states; ++m)
               {
                  // res_left(i_left, n) += alpha*flux_face(n);
                  elmat(n * dof1 + k, offset + m * dof2 + j) +=
                      ip.weight * Q * flux_jac_right(n, m);
                  // res_right(i_right, n) -= alpha*flux_face(n);
                  elmat(offset + n * dof2 + j, m * dof1 + k) -=
                      ip.weight * Q * flux_jac_left(n, m);
               }
            }
         }
      }
      for (int j = 0; j < dof1; ++j)
      {
         for (int k = 0; k < dof1; ++k)
         {
            Q = shape1(j) * shape1(k);
            // multiply by test function
            for (int n = 0; n < num_states; ++n)
            {
               for (int m = 0; m < num_states; ++m)
               {
                  // res(j, n) += flux_face(n) * shape1(j);
                  elmat(m * dof1 + k, n * dof1 + j) +=
                      ip.weight * Q * alpha * flux_jac_left(m, n);
               }
            }
         }
      }
      for (int j = 0; j < dof2; ++j)
      {
         for (int k = 0; k < dof2; ++k)
         {
            Q = shape2(j) * shape2(k);
            // multiply by test function
            for (int n = 0; n < num_states; ++n)
            {
               for (int m = 0; m < num_states; ++m)
               {
                  // res(j, n) -= flux_face(n) * shape2(j);
                  elmat(offset + m * dof2 + k, offset + n * dof2 + j) -=
                      ip.weight * Q * alpha * flux_jac_right(m, n);
               }
            }
         }
      }
   }
}
}  // namespace mach

#endif
