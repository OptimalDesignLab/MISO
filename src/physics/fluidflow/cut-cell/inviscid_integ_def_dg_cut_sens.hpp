#ifndef MACH_INVISCID_INTEG_DEF_DG_CUT_SENS
#define MACH_INVISCID_INTEG_DEF_DG_CUT_SENS

#include "mfem.hpp"

#include "utils.hpp"
#include "sbp_fe.hpp"
#include "inviscid_integ_dg_cut_sens.hpp"

namespace mach
{
template <typename Derived>
double CutDGSensitivityInviscidIntegrator<Derived>::GetElementEnergy(
    const mfem::FiniteElement &el,
    mfem::ElementTransformation &trans,
    const mfem::Vector &elfun)
{
   using namespace mfem;
   using namespace std;
   const int num_nodes = el.GetDof();
   int dim = el.GetDim();
   mfem::Vector elvect;
   elvect.SetSize(num_states * num_nodes);
   elvect = 0.0;
   if (embeddedElements.at(trans.ElementNo) == true)
   {
      return 0.0;
   }
   else
   {
      // cout << "elem id: " << trans.ElementNo << endl;
      DenseMatrix u_mat(elfun.GetData(), num_nodes, num_states);
      // cout << "u_mat " << endl;
      // u_mat.Print();
      DenseMatrix res(elvect.GetData(), num_nodes, num_states);
      DenseMatrix adjJ_i, elflux, dshape, dshapedx, hessian;
      Vector shape, dxidx, dshapedxi, fluxi, u, d2shapedxi, dshapedxik;
      u.SetSize(num_states);
      dxidx.SetSize(dim);
      fluxi.SetSize(num_states);
      flux_jaci.SetSize(num_states);
      dshapedxi.SetSize(num_nodes);
      d2shapedxi.SetSize(num_nodes);
      dshapedxik.SetSize(num_nodes);
      shape.SetSize(num_nodes);
      dshape.SetSize(num_nodes, dim);
      dshapedx.SetSize(num_nodes, dim);
      int size = dim + 1;
      hessian.SetSize(num_nodes, size);
      elflux.SetSize(num_states, dim);
      adjJ_i.SetSize(dim);
      Vector dxqi, flux_jacqi, dudxqik;
      dxqi.SetSize(dim);
      dudxqik.SetSize(num_states);
      flux_jacqi.SetSize(num_states);
      int intorder = trans.OrderGrad(&el) + trans.Order() + el.GetOrder();
      const IntegrationRule *ir;  // = IntRule;
      ir = cutSquareIntRules[trans.ElementNo];
      const IntegrationRule *ir_a;  // = IntRule;
      ir_a = cutSquareIntRules_sens[trans.ElementNo];
      //   MFEM_ASSERT(ir->Size() == ir_a->Size(),
      //               " cut-cell int rule size and its sensitivity size don't "
      //               "match for a scalar design variable");
      if (ir == NULL)
      {
         return 0.0;
      }
      else
      {
         double delta = 1e-05;
         for (int i = 0; i < ir->GetNPoints(); i++)
         {
            const IntegrationPoint &ip = ir->IntPoint(i);
            const IntegrationPoint &ip_a = ir_a->IntPoint(i);
            trans.SetIntPoint(&ip);
            // Calculate the shape function
            el.CalcShape(ip, shape);
            // Compute the physical gradient
            el.CalcDShape(ip, dshape);
            el.CalcHessian(ip, hessian);
            // Mult(dshape, trans.AdjugateJacobian(), dshapedx);
            u_mat.MultTranspose(shape, u);
            CalcAdjugate(trans.Jacobian(), adjJ_i);
            double dwda_i = ip_a.weight;
            /// first term 
            // x (dwda)
            for (int di = 0; di < dim; ++di)
            {
               adjJ_i.GetRow(di, dxidx);
               flux(dxidx, u, fluxi);
               dshape.GetColumn(di, dshapedxi);
               AddMult_a_VWt(-dwda_i, dshapedxi, fluxi, res);
            }
            /// second term 
            // (dshape x [dFdxq x dxqda +  dFdyq x dyqda])
            dxqi(0) = ip_a.x;
            dxqi(1) = ip_a.y;
            for (int dik = 0; dik < dim; ++dik)
            {
               dshape.GetColumn(dik, dshapedxik);
               for (int di = 0; di < dim; ++di)
               {
                  adjJ_i.GetRow(di, dxidx);
                  fluxJacState(dxidx, u, flux_jaci);
                  dshape.GetColumn(di, dshapedxi);
                  u_mat.MultTranspose(dshapedxik, dudxqik);
                  flux_jaci.MultTranspose(dudxqik, flux_jacqi);
                  AddMult_a_VWt(
                      -dxqi(dik) * ip.weight, dshapedxi, flux_jacqi, res);
               }
            }
            /// third term 
            // ([d2shape(0) x Fx + d2shape(1) x Fy] dxqda +
            // [d2shape(1) x Fx + d2shape(2) x Fy] dyqda)
            for (int dik = 0; dik < dim; ++dik)
            {
               int dir = dik;
               for (int di = 0; di < dim; ++di)
               {
                  adjJ_i.GetRow(di, dxidx);
                  flux(dxidx, u, fluxi);
                  hessian.GetColumn(dir, d2shapedxi);
                  flux_jaci.MultTranspose(u, flux_jacqi);
                  AddMult_a_VWt(-dxqi(dik) * ip.weight, d2shapedxi, fluxi, res);
                  ++dir;
               }
            }
         }
         res *= alpha;
      }
   }
}

template <typename Derived>
void CutDGSensitivityInviscidIntegrator<Derived>::AssembleElementVector(
    const mfem::FiniteElement &el,
    mfem::ElementTransformation &trans,
    const mfem::Vector &elfun,
    mfem::Vector &elvect)
{
#if 1
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
     // cout << "elem id: " << trans.ElementNo << endl;
      DenseMatrix u_mat(elfun.GetData(), num_nodes, num_states);
      // cout << "u_mat " << endl;
      // u_mat.Print();
      DenseMatrix res(elvect.GetData(), num_nodes, num_states);
      DenseMatrix adjJ_i, elflux, dshape, dshapedx, hessian;
      Vector shape, dxidx, dshapedxi, fluxi, u, d2shapedxi, dshapedxik;
      u.SetSize(num_states);
      dxidx.SetSize(dim);
      fluxi.SetSize(num_states);
      flux_jaci.SetSize(num_states);
      dshapedxi.SetSize(num_nodes);
      d2shapedxi.SetSize(num_nodes);
      dshapedxik.SetSize(num_nodes);
      shape.SetSize(num_nodes);
      dshape.SetSize(num_nodes, dim);
      dshapedx.SetSize(num_nodes, dim);
      int size = dim + 1;
      hessian.SetSize(num_nodes, size);
      elflux.SetSize(num_states, dim);
      adjJ_i.SetSize(dim);
      Vector dxqi, flux_jacqi, dudxqik;
      dxqi.SetSize(dim);
      dudxqik.SetSize(num_states);
      flux_jacqi.SetSize(num_states);
      int intorder = trans.OrderGrad(&el) + trans.Order() + el.GetOrder();
      const IntegrationRule *ir;  // = IntRule;
      ir = cutSquareIntRules[trans.ElementNo];
      const IntegrationRule *ir_a;  // = IntRule;
      ir_a = cutSquareIntRules_sens[trans.ElementNo];
      //   MFEM_ASSERT(ir->Size() == ir_a->Size(),
      //               " cut-cell int rule size and its sensitivity size don't "
      //               "match for a scalar design variable");
      if (ir == NULL)
      {
         return ;
      }
      else
      {
         double delta = 1e-05;
         for (int i = 0; i < ir->GetNPoints(); i++)
         {
            const IntegrationPoint &ip = ir->IntPoint(i);
            const IntegrationPoint &ip_a = ir_a->IntPoint(i);
            trans.SetIntPoint(&ip);
            // Calculate the shape function
            el.CalcShape(ip, shape);
            // Compute the physical gradient
            el.CalcDShape(ip, dshape);
            el.CalcHessian(ip, hessian);
            // Mult(dshape, trans.AdjugateJacobian(), dshapedx);
            u_mat.MultTranspose(shape, u);
            CalcAdjugate(trans.Jacobian(), adjJ_i);
            double dwda_i = ip_a.weight;
            /// first term 
            // x (dwda)
            for (int di = 0; di < dim; ++di)
            {
               adjJ_i.GetRow(di, dxidx);
               flux(dxidx, u, fluxi);
               dshape.GetColumn(di, dshapedxi);
               AddMult_a_VWt(-dwda_i, dshapedxi, fluxi, res);
            }
            /// second term 
            // (dshape x [dFdxq x dxqda +  dFdyq x dyqda])
            dxqi(0) = ip_a.x;
            dxqi(1) = ip_a.y;
            for (int dik = 0; dik < dim; ++dik)
            {
               dshape.GetColumn(dik, dshapedxik);
               for (int di = 0; di < dim; ++di)
               {
                  adjJ_i.GetRow(di, dxidx);
                  fluxJacState(dxidx, u, flux_jaci);
                  dshape.GetColumn(di, dshapedxi);
                  u_mat.MultTranspose(dshapedxik, dudxqik);
                  flux_jaci.MultTranspose(dudxqik, flux_jacqi);
                  AddMult_a_VWt(
                      -dxqi(dik) * ip.weight, dshapedxi, flux_jacqi, res);
               }
            }
            /// third term 
            // ([d2shape(0) x Fx + d2shape(1) x Fy] dxqda +
            // [d2shape(1) x Fx + d2shape(2) x Fy] dyqda)
            for (int dik = 0; dik < dim; ++dik)
            {
               int dir = dik;
               for (int di = 0; di < dim; ++di)
               {
                  adjJ_i.GetRow(di, dxidx);
                  flux(dxidx, u, fluxi);
                  hessian.GetColumn(dir, d2shapedxi);
                  flux_jaci.MultTranspose(u, flux_jacqi);
                  AddMult_a_VWt(-dxqi(dik) * ip.weight, d2shapedxi, fluxi, res);
                  ++dir;
               }
            }
         }
         res *= alpha;
      }
   }
#endif
}
template <typename Derived>
void CutDGSensitivityInviscidBoundaryIntegrator<Derived>::calcNormalVec(
    Vector x,
    Vector &nrm)

{
   double nx;
   double ny;
   double ds;
   uvector<double, 2> beta, xs;
   xs(0) = x(0);
   xs(1) = x(1);
   beta = phi.grad(xs);
   ds = algoim::norm(beta);
   nx = beta(0);
   ny = beta(1);
   nrm(0) = nx / ds;
   nrm(1) = ny / ds;
}

template <typename Derived>
void CutDGSensitivityInviscidBoundaryIntegrator<Derived>::calcNormalSens(
    const mfem::FiniteElement &el_bnd,
    mfem::ElementTransformation &trans,
    const IntegrationPoint &ip,
    mfem::DenseMatrix &dndxq)
{
   int dim = el_bnd.GetDim();
   double delta = 1e-05;
   Vector pert(dim);
   for (int i = 0; i < dim; ++i)
   {
      Vector nrm_p(dim), nrm_m(dim);
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
      trans.Transform(ip_p, x);
      calcNormalVec(x, nrm_p);

      trans.Transform(ip_m, x);
      calcNormalVec(x, nrm_m);

      nrm_p -= nrm_m;
      nrm_p /= (2.0 * delta);
      for (int j = 0; j < dim; ++j)
      {
         dndxq(j, i) = nrm_p(j);
      }
   }
}

template <typename Derived>
double CutDGSensitivityInviscidBoundaryIntegrator<Derived>::GetElementEnergy(
    const mfem::FiniteElement &el_bnd,
    mfem::ElementTransformation &trans,
    const mfem::Vector &elfun)
{
   using namespace mfem;
   const int dof = el_bnd.GetDof();
   Vector elvect;
#ifdef MFEM_THREAD_SAFE
   Vector u_face, x, nrm, flux_face, shape, dshape;
#endif
   int dim = el_bnd.GetDim();
   mfem::DenseMatrix dndxq;
   dndxq.SetSize(dim, dim);
   Vector dxqi, dshape_xqi, dshapedxi;
   Vector flux_jac_uai, flux_jac_norai;
   DenseMatrix dshape;
   Vector dudxqi, dudai, dnda;
   dxqi.SetSize(dim);
   dnda.SetSize(dim);
   dudxqi.SetSize(num_states);
   dshapedxi.SetSize(dof);
   dudai.SetSize(num_states);
   u_face.SetSize(num_states);
   x.SetSize(dim);
   nrm.SetSize(dim);
   flux_face.SetSize(num_states);
   flux_jac_face.SetSize(num_states);
   flux_jac_uai.SetSize(num_states);
   flux_jac_norai.SetSize(num_states);
   flux_jac_dir.SetSize(num_states, dim);
   elvect.SetSize(num_states * dof);
   elvect = 0.0;
   const IntegrationRule *ir;
   ir = cutSegmentIntRules[trans.ElementNo];
   const IntegrationRule *ir_sens;
   ir_sens = cutSegmentIntRules_sens[trans.ElementNo];
   if (ir != NULL)
   {
      shape.SetSize(dof);
      dshape.SetSize(dof, dim);
      dshape_xqi.SetSize(dof);
      DenseMatrix u(elfun.GetData(), dof, num_states);
      DenseMatrix res(elvect.GetData(), dof, num_states);
      // IntegrationPoint eip1;
      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         const IntegrationPoint &ip_sens = ir_sens->IntPoint(i);
         trans.Transform(ip, x);
         trans.SetIntPoint(&ip);
         el_bnd.CalcShape(ip, shape);
         // get the normal vector and the flux on the face
         // calculate normal vector using level-set function
         double nx;
         double ny;
         double ds;
         double dwda_i = ip_sens.weight;
         dxqi(0) = ip_sens.x;
         dxqi(1) = ip_sens.y;
         uvector<double, 2> beta, xs;
         xs(0) = x(0);
         xs(1) = x(1);
         beta = phi.grad(xs);
         ds = algoim::norm(beta);
         nx = beta(0);
         ny = beta(1);
         nrm(0) = nx / ds;
         nrm(1) = ny / ds;
         // Interpolate elfun at the point
         u.MultTranspose(shape, u_face);
         // cout << "u_face " << endl;
         // u_face.Print();
         flux(x, nrm, u_face, flux_face);
         // cout << "flux " << endl;
         // flux_face.Print();
         // multiply by test function
         /// first term (dwda)
         for (int n = 0; n < num_states; ++n)
         {
            for (int s = 0; s < dof; s++)
            {
               res(s, n) +=
                   shape(s) * flux_face(n) * dwda_i * sqrt(trans.Weight());
            }
         }
         cout << " first term done " << endl;
         /// second term (dshapeda)
         el_bnd.CalcDShape(ip, dshape);
         dshape.Mult(dxqi, dshape_xqi);
         for (int n = 0; n < num_states; ++n)
         {
            for (int s = 0; s < dof; s++)
            {
               res(s, n) += dshape_xqi(s) * flux_face(n) * ip.weight *
                            sqrt(trans.Weight());
            }
         }
         cout << " 2nd term done " << endl;
         /// third term (dFdu x duda)
         fluxJacState(x, nrm, u_face, flux_jac_face);
         for (int di = 0; di < dim; ++di)
         {
            dshape.GetColumn(di, dshapedxi);
            u.MultTranspose(dshapedxi, dudxqi);
            dudxqi *= dxqi(di);
            dudai += dudxqi;
         }
         flux_jac_face.MultTranspose(dudai, flux_jac_uai);
         for (int n = 0; n < num_states; ++n)
         {
            for (int s = 0; s < dof; s++)
            {
               res(s, n) += shape(s) * flux_jac_uai(n) * ip.weight *
                            sqrt(trans.Weight());
            }
         }
         cout << " 3rd term done " << endl;
         /// fourth term (dFdn x dnda)
         fluxJacDir(x, nrm, u_face, flux_jac_dir);
         calcNormalSens(el_bnd, trans, ip, dndxq);
         dndxq.Mult(dxqi, dnda);
         flux_jac_dir.Mult(dnda, flux_jac_norai);
         for (int n = 0; n < num_states; ++n)
         {
            for (int s = 0; s < dof; s++)
            {
               res(s, n) += shape(s) * flux_jac_norai(n) * ip.weight *
                            sqrt(trans.Weight());
            }
         }
         cout << " 4th term done " << endl;
      }
      res *= alpha;
   }
   else
   {
      elvect = 0.0;
   }

#if 0
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
   double fun_sens = 0.0;
   const IntegrationRule *ir;
   ir = cutSegmentIntRules[trans.ElementNo];
   const IntegrationRule *ir_sens;
   ir_sens = cutSegmentIntRules_sens[trans.ElementNo];
   if (!(ir))
   {
      return fun;
   }
   DenseMatrix u(elfun.GetData(), num_nodes, num_states);
   // IntegrationPoint el_ip;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &face_ip = ir->IntPoint(i);
      const IntegrationPoint &face_ip_sens = ir_sens->IntPoint(i);
      // get the normal vector, and then add contribution to function
      trans.SetIntPoint(&face_ip);
      trans.Transform(face_ip, x);
      el_bnd.CalcShape(face_ip, shape);
      // get the normal vector and the flux on the face
      // calculate normal vector using level-set function
      double nx;
      double ny;
      double ds;
      /// n_hat = grad_phi/|\grad_phi|
      uvector<double, 2> beta, xs;
      xs(0) = x(0);
      xs(1) = x(1);
      beta = phi.grad(xs);
      ds = algoim::norm(beta);
      nx = beta(0);
      ny = beta(1);
      nrm(0) = nx / ds;
      nrm(1) = ny / ds;
      // Interpolate elfun at the point
      u.MultTranspose(shape, u_face);
      /// this is used for area test
      double area = sqrt(trans.Weight());
      fun += face_ip.weight * alpha * area;
      fun_sens += face_ip_sens.weight * alpha * area;
      // fun += bndryFun(x, nrm, u_face) * face_ip.weight * sqrt(trans.Weight())
      // *
      //        alpha;
   }
   return fun_sens;
#endif
}

template <typename Derived>
void CutDGSensitivityInviscidBoundaryIntegrator<Derived>::AssembleElementVector(
    const mfem::FiniteElement &el_bnd,
    mfem::ElementTransformation &trans,
    const mfem::Vector &elfun,
    mfem::Vector &elvect)
{
   using namespace mfem;
   const int dof = el_bnd.GetDof();
#ifdef MFEM_THREAD_SAFE
   Vector u_face, x, nrm, flux_face, shape, dshape;
#endif
   int dim = el_bnd.GetDim();
   mfem::DenseMatrix dndxq;
   dndxq.SetSize(dim, dim);
   Vector dxqi, dshape_xqi, dshapedxi;
   Vector flux_jac_uai, flux_jac_norai;
   DenseMatrix dshape;
   Vector dudxqi, dudai, dnda;
   dxqi.SetSize(dim);
   dnda.SetSize(dim);
   dudxqi.SetSize(num_states);
   dshapedxi.SetSize(dof);
   dudai.SetSize(num_states);
   u_face.SetSize(num_states);
   x.SetSize(dim);
   nrm.SetSize(dim);
   flux_face.SetSize(num_states);
   flux_jac_face.SetSize(num_states);
   flux_jac_uai.SetSize(num_states);
   flux_jac_norai.SetSize(num_states);
   flux_jac_dir.SetSize(num_states, dim);
   elvect.SetSize(num_states * dof);
   elvect = 0.0;
   const IntegrationRule *ir;
   ir = cutSegmentIntRules[trans.ElementNo];
   const IntegrationRule *ir_sens;
   ir_sens = cutSegmentIntRules_sens[trans.ElementNo];
   if (ir != NULL)
   {
      shape.SetSize(dof);
      dshape.SetSize(dof, dim);
      dshape_xqi.SetSize(dof);
      DenseMatrix u(elfun.GetData(), dof, num_states);
      DenseMatrix res(elvect.GetData(), dof, num_states);
      // IntegrationPoint eip1;
      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         const IntegrationPoint &ip_sens = ir_sens->IntPoint(i);
         trans.Transform(ip, x);
         trans.SetIntPoint(&ip);
         el_bnd.CalcShape(ip, shape);
         // get the normal vector and the flux on the face
         // calculate normal vector using level-set function
         double nx;
         double ny;
         double ds;
         double dwda_i = ip_sens.weight;
         dxqi(0) = ip_sens.x;
         dxqi(1) = ip_sens.y;
         uvector<double, 2> beta, xs;
         xs(0) = x(0);
         xs(1) = x(1);
         beta = phi.grad(xs);
         ds = algoim::norm(beta);
         nx = beta(0);
         ny = beta(1);
         nrm(0) = nx / ds;
         nrm(1) = ny / ds;
         // Interpolate elfun at the point
         u.MultTranspose(shape, u_face);
         flux(x, nrm, u_face, flux_face);
         /// first term (dwda)
         for (int n = 0; n < num_states; ++n)
         {
            for (int s = 0; s < dof; s++)
            {
               res(s, n) +=
                   shape(s) * flux_face(n) * dwda_i * sqrt(trans.Weight());
            }
         }
         cout << " first term done " << endl;
         /// second term (dshapeda)
         el_bnd.CalcDShape(ip, dshape);
         dshape.Mult(dxqi, dshape_xqi);
         for (int n = 0; n < num_states; ++n)
         {
            for (int s = 0; s < dof; s++)
            {
               res(s, n) += dshape_xqi(s) * flux_face(n) * ip.weight *
                            sqrt(trans.Weight());
            }
         }
         cout << " 2nd term done " << endl;
         /// third term (dFdu x duda)
         fluxJacState(x, nrm, u_face, flux_jac_face);
         for (int di = 0; di < dim; ++di)
         {
            dshape.GetColumn(di, dshapedxi);
            u.MultTranspose(dshapedxi, dudxqi);
            dudxqi *= dxqi(di);
            dudai += dudxqi;
         }
         flux_jac_face.MultTranspose(dudai, flux_jac_uai);
         for (int n = 0; n < num_states; ++n)
         {
            for (int s = 0; s < dof; s++)
            {
               res(s, n) += shape(s) * flux_jac_uai(n) * ip.weight *
                            sqrt(trans.Weight());
            }
         }
         cout << " 3rd term done " << endl;
         /// fourth term (dFdn x dnda)
         fluxJacDir(x, nrm, u_face, flux_jac_dir);
         calcNormalSens(el_bnd, trans, ip, dndxq);
         dndxq.Mult(dxqi, dnda);
         flux_jac_dir.Mult(dnda, flux_jac_norai);
         for (int n = 0; n < num_states; ++n)
         {
            for (int s = 0; s < dof; s++)
            {
               res(s, n) += shape(s) * flux_jac_norai(n) * ip.weight *
                            sqrt(trans.Weight());
            }
         }
         cout << " 4th term done " << endl;
      }
      res *= alpha;
   }
   else
   {
      elvect = 0.0;
   }
}

template <typename Derived>
void CutDGSensitivityInviscidFaceIntegrator<Derived>::calcFaceNormalSens(
    const mfem::FiniteElement &el_left,
    const mfem::FiniteElement &el_right,
    mfem::FaceElementTransformations &trans,
    const IntegrationPoint &ip,
    mfem::DenseMatrix &dndxq)
{
   int dim = el_left.GetDim();
   double delta = 1e-05;

   for (int i = 0; i < dim; ++i)
   {
      Vector nrm_p(dim), nrm_m(dim);
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
      trans.Face->SetIntPoint(&ip_p);
      // Get the normal vector and the flux on the face
      CalcOrtho(trans.Face->Jacobian(), nrm_p);

      trans.Face->SetIntPoint(&ip_m);
      // Get the normal vector and the flux on the face
      CalcOrtho(trans.Face->Jacobian(), nrm_m);

      nrm_p -= nrm_m;
      nrm_p /= (2.0 * delta);
      for (int j = 0; j < dim; ++j)
      {
         dndxq(j, i) = nrm_p(j);
      }
   }
}

template <typename Derived>
double CutDGSensitivityInviscidFaceIntegrator<Derived>::GetFaceEnergy(
    const mfem::FiniteElement &el_left,
    const mfem::FiniteElement &el_right,
    mfem::FaceElementTransformations &trans,
    const mfem::Vector &elfun)
{
    using namespace mfem;
   using namespace std;
#ifdef MFEM_THREAD_SAFE
   Vector shape1, shape2, funval1, funval2, nrm, fluxN;
#endif
   Vector elvect;
   // Compute the term <F.n(u),[w]> on the interior faces.
   int dim = el_left.GetDim();
   const int dof1 = el_left.GetDof();
   const int dof2 = el_right.GetDof();
   mfem::DenseMatrix dndxq;
   dndxq.SetSize(dim, dim);
   Vector dxqi, dshape1_xqi, dshape2_xqi, dshape1dxi, dshape2dxi;
   Vector flux_jac1_uai, flux_jac2_uai, flux_jac_norai;
   DenseMatrix dshape1, dshape2;
   Vector dudx1qi, duda1i, dnda;
   Vector dudx2qi, duda2i;
   dxqi.SetSize(dim);
   dnda.SetSize(dim);
   dudx1qi.SetSize(num_states);
   dudx2qi.SetSize(num_states);
   dshape1dxi.SetSize(dof1);
   dshape2dxi.SetSize(dof2);
   dshape1_xqi.SetSize(dof1);
   dshape2_xqi.SetSize(dof2);
   duda1i.SetSize(num_states);
   duda2i.SetSize(num_states);
   dshape1.SetSize(dof1, dim);
   dshape2.SetSize(dof2, dim);
   // int dim = el_left.GetDim();
   nrm.SetSize(dim);
   shape1.SetSize(dof1);
   shape2.SetSize(dof2);
   u_face_left.SetSize(num_states);
   u_face_right.SetSize(num_states);
   fluxN.SetSize(num_states);
   elvect.SetSize((dof1 + dof2) * num_states);
   elvect = 0.0;
   flux_jac_left.SetSize(num_states);
   flux_jac_right.SetSize(num_states);
   flux_jac1_uai.SetSize(num_states);
   flux_jac2_uai.SetSize(num_states);
   flux_jac_dir.SetSize(num_states, dim);
   flux_jac_norai.SetSize(num_states);
   if (immersedFaces[trans.Face->ElementNo] == true)
   {
      return 0.0;
   }
   DenseMatrix elfun1_mat(elfun.GetData(), dof1, num_states);
   DenseMatrix elfun2_mat(
       elfun.GetData() + dof1 * num_states, dof2, num_states);

   DenseMatrix elvect1_mat(elvect.GetData(), dof1, num_states);
   DenseMatrix elvect2_mat(
       elvect.GetData() + dof1 * num_states, dof2, num_states);
   IntegrationRule *ir;
   ir = cutInteriorFaceIntRules[trans.Face->ElementNo];
   const IntegrationRule *ir_sens;
   ir_sens = cutInteriorFaceIntRules_sens[trans.Face->ElementNo];
#if 0
   if (ir != NULL)
   {
      cout << "face elements are " << trans.Elem1No << " , " << trans.Elem2No
           << endl;
      double face_length = 0.0;
      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         trans.Face->SetIntPoint(&ip);
         face_length += ip.weight * trans.Weight();
         cout << "ip.x, ip.y : " << ip.x << " , " << ip.y << endl;
         cout << "ip.weight " << ip.weight << endl;
      }
      cout << "face length: " << face_length << endl;
   }
#endif
   if (ir == NULL)
   {
      return 0.0;
   }
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      IntegrationPoint &ip = ir->IntPoint(i);
      const IntegrationPoint &ip_sens = ir_sens->IntPoint(i);
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
      cout << "face normal " << endl;
      nrm.Print();
      cout << "face int rule sensitivities " << endl;
      cout << ip_sens.x << " , " << ip_sens.y << endl;
      flux(nrm, u_face_left, u_face_right, fluxN);
      double dwda_i = ip_sens.weight;
      /// first term (dwda)
      for (int k = 0; k < num_states; k++)
      {
         for (int s = 0; s < dof1; s++)
         {
            elvect1_mat(s, k) += fluxN(k) * dwda_i * shape1(s);
         }
         for (int s = 0; s < dof2; s++)
         {
            elvect2_mat(s, k) -= fluxN(k) * dwda_i * shape2(s);
         }
      }
      // cout << "Ist term done " << endl;
      /// second term (dshapeda)
      el_left.CalcDShape(trans.GetElement1IntPoint(), dshape1);
      el_right.CalcDShape(trans.GetElement2IntPoint(), dshape2);
      dxqi(0) = ip_sens.x;
      dxqi(1) = ip_sens.y;
      dshape1.Mult(dxqi, dshape1_xqi);
      dshape2.Mult(dxqi, dshape2_xqi);
      for (int k = 0; k < num_states; k++)
      {
         for (int s = 0; s < dof1; s++)
         {
            elvect1_mat(s, k) += fluxN(k) * ip.weight * dshape1_xqi(s);
         }
         for (int s = 0; s < dof2; s++)
         {
            elvect2_mat(s, k) -= fluxN(k) * ip.weight * dshape2_xqi(s);
         }
      }
      // cout << "2nd term done " << endl;
      /// third term (dFdu x duda)
      fluxJacStates(
          nrm, u_face_left, u_face_right, flux_jac_left, flux_jac_right);
      for (int di = 0; di < dim; ++di)
      {
         dshape1.GetColumn(di, dshape1dxi);
         elfun1_mat.MultTranspose(dshape1dxi, dudx1qi);
         dshape2.GetColumn(di, dshape2dxi);
         elfun2_mat.MultTranspose(dshape2dxi, dudx2qi);
         dudx1qi *= dxqi(di);
         duda1i += dudx1qi;
         dudx2qi *= dxqi(di);
         duda2i += dudx2qi;
      }
      flux_jac_left.MultTranspose(duda1i, flux_jac1_uai);
      flux_jac_right.MultTranspose(duda2i, flux_jac2_uai);
      flux_jac1_uai += flux_jac2_uai;
      for (int k = 0; k < num_states; k++)
      {
         for (int s = 0; s < dof1; s++)
         {
            elvect1_mat(s, k) += flux_jac1_uai(k) * ip.weight * shape1(s);
         }
         for (int s = 0; s < dof2; s++)
         {
            elvect2_mat(s, k) -= flux_jac1_uai(k) * ip.weight * shape2(s);
         }
      }
      // cout << "3rd term done " << endl;
      /// fourth term (dFdn x dnda)
      calcFaceNormalSens(el_left, el_right, trans, ip, dndxq);
      fluxJacDir(nrm, u_face_left, u_face_right, flux_jac_dir);
      dndxq.Mult(dxqi, dnda);
      flux_jac_dir.Mult(dnda, flux_jac_norai);
      for (int k = 0; k < num_states; k++)
      {
         for (int s = 0; s < dof1; s++)
         {
            elvect1_mat(s, k) += flux_jac_norai(k) * ip.weight * shape1(s);
         }
         for (int s = 0; s < dof2; s++)
         {
            elvect2_mat(s, k) -= flux_jac_norai(k) * ip.weight * shape2(s);
         }
      }
      // cout << "4th term done " << endl;
      elvect *= alpha;
   }
   return 0;
}

template <typename Derived>
void CutDGSensitivityInviscidFaceIntegrator<Derived>::AssembleFaceVector(
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
   mfem::DenseMatrix dndxq;
   dndxq.SetSize(dim, dim);
   Vector dxqi, dshape1_xqi, dshape2_xqi, dshape1dxi, dshape2dxi;
   Vector flux_jac1_uai, flux_jac2_uai, flux_jac_norai;
   DenseMatrix dshape1, dshape2;
   Vector dudx1qi, duda1i, dnda;
   Vector dudx2qi, duda2i;
   dxqi.SetSize(dim);
   dnda.SetSize(dim);
   dudx1qi.SetSize(num_states);
   dudx2qi.SetSize(num_states);
   dshape1dxi.SetSize(dof1);
   dshape2dxi.SetSize(dof2);
   dshape1_xqi.SetSize(dof1);
   dshape2_xqi.SetSize(dof2);
   duda1i.SetSize(num_states);
   duda2i.SetSize(num_states);
   dshape1.SetSize(dof1, dim);
   dshape2.SetSize(dof2, dim);
   // int dim = el_left.GetDim();
   nrm.SetSize(dim);
   shape1.SetSize(dof1);
   shape2.SetSize(dof2);
   u_face_left.SetSize(num_states);
   u_face_right.SetSize(num_states);
   fluxN.SetSize(num_states);
   elvect.SetSize((dof1 + dof2) * num_states);
   elvect = 0.0;
   flux_jac_left.SetSize(num_states);
   flux_jac_right.SetSize(num_states);
   flux_jac1_uai.SetSize(num_states);
   flux_jac2_uai.SetSize(num_states);
   flux_jac_dir.SetSize(num_states, dim);
   flux_jac_norai.SetSize(num_states);
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
   IntegrationRule *ir;
   ir = cutInteriorFaceIntRules[trans.Face->ElementNo];
   const IntegrationRule *ir_sens;
   ir_sens = cutInteriorFaceIntRules_sens[trans.Face->ElementNo];
#if 0
   if (ir != NULL)
   {
      cout << "face elements are " << trans.Elem1No << " , " << trans.Elem2No
           << endl;
      double face_length = 0.0;
      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         trans.Face->SetIntPoint(&ip);
         face_length += ip.weight * trans.Weight();
         cout << "ip.x, ip.y : " << ip.x << " , " << ip.y << endl;
         cout << "ip.weight " << ip.weight << endl;
      }
      cout << "face length: " << face_length << endl;
   }
#endif
   if (ir == NULL)
   {
      return;
   }
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      IntegrationPoint &ip = ir->IntPoint(i);
      const IntegrationPoint &ip_sens = ir_sens->IntPoint(i);
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
      cout << "face normal " << endl;
      nrm.Print();
      flux(nrm, u_face_left, u_face_right, fluxN);
      double dwda_i = ip_sens.weight;
      /// first term (dwda)
      for (int k = 0; k < num_states; k++)
      {
         for (int s = 0; s < dof1; s++)
         {
            elvect1_mat(s, k) += fluxN(k) * dwda_i * shape1(s);
         }
         for (int s = 0; s < dof2; s++)
         {
            elvect2_mat(s, k) -= fluxN(k) * dwda_i * shape2(s);
         }
      }
      // cout << "Ist term done " << endl;
      /// second term (dshapeda)
      el_left.CalcDShape(trans.GetElement1IntPoint(), dshape1);
      el_right.CalcDShape(trans.GetElement2IntPoint(), dshape2);
      dxqi(0) = ip_sens.x;
      dxqi(1) = ip_sens.y;
      dshape1.Mult(dxqi, dshape1_xqi);
      dshape2.Mult(dxqi, dshape2_xqi);
      for (int k = 0; k < num_states; k++)
      {
         for (int s = 0; s < dof1; s++)
         {
            elvect1_mat(s, k) += fluxN(k) * ip.weight * dshape1_xqi(s);
         }
         for (int s = 0; s < dof2; s++)
         {
            elvect2_mat(s, k) -= fluxN(k) * ip.weight * dshape2_xqi(s);
         }
      }
      // cout << "2nd term done " << endl;
      /// third term (dFdu x duda)
      fluxJacStates(
          nrm, u_face_left, u_face_right, flux_jac_left, flux_jac_right);
      for (int di = 0; di < dim; ++di)
      {
         dshape1.GetColumn(di, dshape1dxi);
         elfun1_mat.MultTranspose(dshape1dxi, dudx1qi);
         dshape2.GetColumn(di, dshape2dxi);
         elfun2_mat.MultTranspose(dshape2dxi, dudx2qi);
         dudx1qi *= dxqi(di);
         duda1i += dudx1qi;
         dudx2qi *= dxqi(di);
         duda2i += dudx2qi;
      }
      flux_jac_left.MultTranspose(duda1i, flux_jac1_uai);
      flux_jac_right.MultTranspose(duda2i, flux_jac2_uai);
      flux_jac1_uai += flux_jac2_uai;
      for (int k = 0; k < num_states; k++)
      {
         for (int s = 0; s < dof1; s++)
         {
            elvect1_mat(s, k) += flux_jac1_uai(k) * ip.weight * shape1(s);
         }
         for (int s = 0; s < dof2; s++)
         {
            elvect2_mat(s, k) -= flux_jac1_uai(k) * ip.weight * shape2(s);
         }
      }
      // cout << "3rd term done " << endl;
      #if 0
      /// fourth term (dFdn x dnda)
      calcFaceNormalSens(el_left, el_right, trans, ip, dndxq);
      fluxJacDir(nrm, u_face_left, u_face_right, flux_jac_dir);
      dndxq.Mult(dxqi, dnda);
      flux_jac_dir.Mult(dnda, flux_jac_norai);
      for (int k = 0; k < num_states; k++)
      {
         for (int s = 0; s < dof1; s++)
         {
            elvect1_mat(s, k) += flux_jac_norai(k) * ip.weight * shape1(s);
         }
         for (int s = 0; s < dof2; s++)
         {
            elvect2_mat(s, k) -= flux_jac_norai(k) * ip.weight * shape2(s);
         }
      }
      #endif
      // cout << "4th term done " << endl;
      elvect *= alpha;
   }
}

}  // namespace mach

#endif