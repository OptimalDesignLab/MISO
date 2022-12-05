#include <iostream>
#include "coefficient.hpp"
#include "electromag_integ.hpp"
#include "mach_input.hpp"

///TODO: Don't forget to uncomment. Commented out since ThreeStateCoefficient was causing some errors when trying make (even before make build_tests), so will come back to.
#include "cal2_kh_coefficient.hpp"
#include "cal2_ke_coefficient.hpp"

using namespace mfem;

namespace mach
{
double calcMagneticEnergy(ElementTransformation &trans,
                          const IntegrationPoint &ip,
                          StateCoefficient &nu,
                          double B)
{
   /// TODO: use a composite rule instead or find a way to just directly
   /// integrate B-H curve
   const IntegrationRule *ir = &IntRules.Get(Geometry::Type::SEGMENT, 20);

   /// compute int_0^{B} \nuB dB
   double en = 0.0;
   for (int j = 0; j < ir->GetNPoints(); j++)
   {
      const IntegrationPoint &segment_ip = ir->IntPoint(j);
      double xi = segment_ip.x * B;
      en += segment_ip.weight * xi * nu.Eval(trans, ip, xi);
   }
   en *= B;
   return en;
}

double calcMagneticEnergyDot(ElementTransformation &trans,
                             const IntegrationPoint &ip,
                             StateCoefficient &nu,
                             double B)
{
   /// TODO: use a composite rule instead or find a way to just directly
   /// integrate B-H curve
   const IntegrationRule *ir = &IntRules.Get(Geometry::Type::SEGMENT, 20);

   /// compute int_0^{B} \nuB dB
   double en = 0.0;
   double dendB = 0.0;
   for (int j = 0; j < ir->GetNPoints(); j++)
   {
      const IntegrationPoint &segment_ip = ir->IntPoint(j);
      double xi = segment_ip.x * B;
      double dxidB = segment_ip.x;
      en += segment_ip.weight * xi * nu.Eval(trans, ip, xi);

      dendB +=
          segment_ip.weight *
          (nu.Eval(trans, ip, xi) + xi * nu.EvalStateDeriv(trans, ip, xi)) *
          dxidB;
   }
   dendB *= B;
   dendB += en;
   return dendB;
}

double calcMagneticEnergyDoubleDot(ElementTransformation &trans,
                                   const IntegrationPoint &ip,
                                   StateCoefficient &nu,
                                   double B)
{
   /// TODO: use a composite rule instead or find a way to just directly
   /// integrate B-H curve
   const IntegrationRule *ir = &IntRules.Get(Geometry::Type::SEGMENT, 20);

   /// compute int_0^{B} \nuB dB
   double d2endB2 = 0.0;
   for (int j = 0; j < ir->GetNPoints(); j++)
   {
      const IntegrationPoint &segment_ip = ir->IntPoint(j);
      const double w = segment_ip.weight;
      const double xi = segment_ip.x * B;
      const double dxidB = segment_ip.x;

      const double nu_val = nu.Eval(trans, ip, xi);
      const double dnudB = nu.EvalStateDeriv(trans, ip, xi);
      const double d2nudB2 = nu.EvalState2ndDeriv(trans, ip, xi);

      d2endB2 += w * (2 * ((dxidB * nu_val) + (xi * dnudB * dxidB) +
                           (dxidB * dxidB * dnudB * B)) +
                      xi * d2nudB2 * dxidB * dxidB * B);
   }
   return d2endB2;
}

void NonlinearDiffusionIntegrator::AssembleElementVector(
    const FiniteElement &el,
    ElementTransformation &trans,
    const Vector &elfun,
    Vector &elvect)
{
   /// number of degrees of freedom
   int ndof = el.GetDof();
   elvect.SetSize(ndof);

   int dim = el.GetDim();
   int space_dim = trans.GetSpaceDim();

#ifdef MFEM_THREAD_SAFE
   DenseMatrix dshape;
   DenseMatrix dshapedxt;
#endif
   dshape.SetSize(ndof, dim);
   dshapedxt.SetSize(ndof, space_dim);

   double pointflux_buffer[3] = {};
   Vector pointflux(pointflux_buffer, space_dim);

   const IntegrationRule *ir = IntRule;
   if (ir == nullptr)
   {
      int order = [&]()
      {
         if (el.Space() == FunctionSpace::Pk)
         {
            return 2 * el.GetOrder() - 2;
         }
         else
         {
            // order = 2*el.GetOrder() - 2;  // <-- this seems to work fine too
            return 2 * el.GetOrder() + el.GetDim() - 1;
         }
      }();

      if (el.Space() == FunctionSpace::rQk)
      {
         ir = &RefinedIntRules.Get(el.GetGeomType(), order);
      }
      else
      {
         ir = &IntRules.Get(el.GetGeomType(), order);
      }
   }

   elvect = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);

      double trans_weight = trans.Weight();

      double w = alpha * ip.weight / trans_weight;

      el.CalcDShape(ip, dshape);
      Mult(dshape, trans.AdjugateJacobian(), dshapedxt);

      dshapedxt.MultTranspose(elfun, pointflux);

      const double pointflux_norm = pointflux.Norml2();
      const double pointflux_mag = pointflux_norm / trans_weight;

      double model_val = model.Eval(trans, ip, pointflux_mag);

      pointflux *= w * model_val;

      dshapedxt.AddMult(pointflux, elvect);
   }
}

void NonlinearDiffusionIntegrator::AssembleElementGrad(
    const mfem::FiniteElement &el,
    mfem::ElementTransformation &trans,
    const mfem::Vector &elfun,
    mfem::DenseMatrix &elmat)
{
   /// number of degrees of freedom
   int ndof = el.GetDof();
   elmat.SetSize(ndof);
   elmat = 0.0;

   int dim = el.GetDim();
   int space_dim = trans.GetSpaceDim();

#ifdef MFEM_THREAD_SAFE
   DenseMatrix dshape;
   DenseMatrix dshapedxt;
   DenseMatrix point_flux_2_dot;
   Vector scratch;
#endif
   dshape.SetSize(ndof, dim);
   dshapedxt.SetSize(ndof, space_dim);
   point_flux_2_dot.SetSize(ndof, space_dim);
   pointflux_norm_dot.SetSize(ndof);

   double pointflux_buffer[3] = {};
   Vector pointflux(pointflux_buffer, space_dim);

   const IntegrationRule *ir = IntRule;
   if (ir == nullptr)
   {
      int order = [&]()
      {
         if (el.Space() == FunctionSpace::Pk)
         {
            return 2 * el.GetOrder() - 2;
         }
         else
         {
            // order = 2*el.GetOrder() - 2;  // <-- this seems to work fine too
            return 2 * el.GetOrder() + el.GetDim() - 1;
         }
      }();

      if (el.Space() == FunctionSpace::rQk)
      {
         ir = &RefinedIntRules.Get(el.GetGeomType(), order);
      }
      else
      {
         ir = &IntRules.Get(el.GetGeomType(), order);
      }
   }

   elmat = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);

      double trans_weight = trans.Weight();

      double w = alpha * ip.weight / trans_weight;

      el.CalcDShape(ip, dshape);
      Mult(dshape, trans.AdjugateJacobian(), dshapedxt);

      dshapedxt.MultTranspose(elfun, pointflux);

      const double pointflux_norm = pointflux.Norml2();

      pointflux_norm_dot = 0.0;
      dshapedxt.AddMult_a(1.0 / pointflux_norm, pointflux, pointflux_norm_dot);

      const double pointflux_mag = pointflux_norm / trans_weight;
      pointflux_norm_dot /= trans_weight;

      double model_val = model.Eval(trans, ip, pointflux_mag);

      double model_deriv = model.EvalStateDeriv(trans, ip, pointflux_mag);
      pointflux_norm_dot *= model_deriv;

      point_flux_2_dot = dshapedxt;
      point_flux_2_dot *= model_val;

      if (abs(pointflux_norm) > 1e-14)
      {
         AddMultVWt(pointflux_norm_dot, pointflux, point_flux_2_dot);
      }
      point_flux_2_dot *= w;

      AddMultABt(dshapedxt, point_flux_2_dot, elmat);
   }
}

void NonlinearDiffusionIntegratorMeshRevSens::AssembleRHSElementVect(
    const FiniteElement &mesh_el,
    ElementTransformation &mesh_trans,
    Vector &mesh_coords_bar)
{
   const int element = mesh_trans.ElementNo;
   const auto &el = *state.FESpace()->GetFE(element);
   auto &trans = *state.FESpace()->GetElementTransformation(element);

   const int mesh_ndof = mesh_el.GetDof();
   const int ndof = el.GetDof();
   const int dim = el.GetDim();
   const int space_dim = trans.GetSpaceDim();
   const int curl_dim = space_dim;

   /// get the proper element, transformation, and state vector
#ifdef MFEM_THREAD_SAFE
   mfem::Array<int> vdofs;
   mfem::Vector elfun;
   mfem::Vector psi;
#endif
   auto *dof_tr = state.FESpace()->GetElementVDofs(element, vdofs);
   state.GetSubVector(vdofs, elfun);
   if (dof_tr != nullptr)
   {
      dof_tr->InvTransformPrimal(elfun);
   }

   dof_tr = adjoint.FESpace()->GetElementVDofs(element, vdofs);
   adjoint.GetSubVector(vdofs, psi);
   if (dof_tr != nullptr)
   {
      dof_tr->InvTransformPrimal(psi);
   }

#ifdef MFEM_THREAD_SAFE
   DenseMatrix dshape;
   DenseMatrix dshapedxt;
   DenseMatrix dshapedxt_bar;
   DenseMatrix PointMat_bar;
#else
   auto &dshape = integ.dshape;
   auto &dshapedxt = integ.dshapedxt;
#endif

   dshape.SetSize(ndof, dim);
   dshapedxt.SetSize(ndof, space_dim);
   dshapedxt_bar.SetSize(ndof, space_dim);
   PointMat_bar.SetSize(space_dim, mesh_ndof);

   double pointflux_buffer[3] = {};
   Vector pointflux(pointflux_buffer, space_dim);
   double pointflux_bar_buffer[3] = {};
   Vector pointflux_bar(pointflux_bar_buffer, space_dim);

   double curl_psi_buffer[3] = {};
   Vector curl_psi(curl_psi_buffer, dim);
   double curl_psi_bar_buffer[3] = {};
   Vector curl_psi_bar(curl_psi_bar_buffer, dim);

   // cast the ElementTransformation
   auto &isotrans = dynamic_cast<IsoparametricTransformation &>(mesh_trans);

   const IntegrationRule *ir = IntRule;
   if (ir == nullptr)
   {
      int order = [&]()
      {
         if (el.Space() == FunctionSpace::Pk)
         {
            return 2 * el.GetOrder() - 2;
         }
         else
         {
            // order = 2*el.GetOrder() - 2;  // <-- this seems to work fine too
            return 2 * el.GetOrder() + el.GetDim() - 1;
         }
      }();

      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   auto &alpha = integ.alpha;
   auto &model = integ.model;
   mesh_coords_bar.SetSize(mesh_ndof * space_dim);
   mesh_coords_bar = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);

      double trans_weight = trans.Weight();

      /// holds quadrature weight
      double w = alpha * ip.weight / trans_weight;

      el.CalcDShape(ip, dshape);
      Mult(dshape, trans.AdjugateJacobian(), dshapedxt);

      dshapedxt.MultTranspose(elfun, pointflux);
      dshapedxt.MultTranspose(psi, curl_psi);
      const double curl_psi_dot_pointflux = curl_psi * pointflux;

      const double pointflux_norm = pointflux.Norml2();
      const double pointflux_mag = pointflux_norm / trans_weight;

      double model_val = model.Eval(trans, ip, pointflux_mag);

      /// dummy functional for adjoint-weighted residual
      // fun += model_val * curl_psi_dot_pointflux * w;

      /// start reverse pass
      double fun_bar = 1.0;

      /// fun += model_val * curl_psi_dot_pointflux * w;
      double model_val_bar = fun_bar * curl_psi_dot_pointflux * w;
      double curl_psi_dot_pointflux_bar = fun_bar * model_val * w;
      double w_bar = fun_bar * model_val * curl_psi_dot_pointflux;

      /// double model_val = model.Eval(trans, ip, pointflux_mag);
      double pointflux_mag_bar = 0.0;
      const double dmodeldpointflux_mag =
          model.EvalStateDeriv(trans, ip, pointflux_mag);
      pointflux_mag_bar += model_val_bar * dmodeldpointflux_mag;

      /// const double pointflux_mag = pointflux_norm / trans_weight;
      double pointflux_norm_bar = 0.0;
      double trans_weight_bar = 0.0;
      pointflux_norm_bar += pointflux_mag_bar / trans_weight;
      trans_weight_bar -=
          pointflux_mag_bar * pointflux_norm / pow(trans_weight, 2);

      /// const double pointflux_norm = pointflux.Norml2();
      pointflux_bar = 0.0;
      add(pointflux_bar,
          pointflux_norm_bar / pointflux_norm,
          pointflux,
          pointflux_bar);

      /// const double curl_psi_dot_pointflux = curl_psi * pointflux;
      curl_psi_bar = 0.0;
      add(curl_psi_bar, curl_psi_dot_pointflux_bar, pointflux, curl_psi_bar);
      add(pointflux_bar, curl_psi_dot_pointflux_bar, curl_psi, pointflux_bar);

      dshapedxt_bar = 0.0;
      /// dshapedxt.MultTranspose(psi, curl_psi);
      AddMultVWt(psi, curl_psi_bar, dshapedxt_bar);
      /// dshapedxt.MultTranspose(elfun, pointflux);
      AddMultVWt(elfun, pointflux_bar, dshapedxt_bar);

      /// Mult(dshape, trans.AdjugateJacobian(), dshapedxt);
      double adj_jac_bar_buffer[9] = {};
      DenseMatrix adj_jac_bar(adj_jac_bar_buffer, space_dim, space_dim);
      MultAtB(dshape, dshapedxt_bar, adj_jac_bar);

      PointMat_bar = 0.0;
      isotrans.AdjugateJacobianRevDiff(adj_jac_bar, PointMat_bar);

      /// double w = alpha * ip.weight / trans_weight;
      trans_weight_bar -= w_bar * alpha * ip.weight / pow(trans_weight, 2);

      isotrans.WeightRevDiff(trans_weight_bar, PointMat_bar);

      // code to insert PointMat_bar into mesh_coords_bar;
      for (int j = 0; j < mesh_ndof; ++j)
      {
         for (int k = 0; k < curl_dim; ++k)
         {
            mesh_coords_bar(k * mesh_ndof + j) += PointMat_bar(k, j);
         }
      }
   }
}

void MagnetizationSource2DIntegrator::AssembleRHSElementVect(
    const mfem::FiniteElement &el,
    mfem::ElementTransformation &trans,
    mfem::Vector &elvect)
{
   /// number of degrees of freedom
   int ndof = el.GetDof();
   elvect.SetSize(ndof);

   int dim = el.GetDim();
   int space_dim = trans.GetSpaceDim();
   if (space_dim != 2)
   {
      mfem_error(
          "MagnetizationSource2DIntegrator only supports 2D space dim!\n");
   }

#ifdef MFEM_THREAD_SAFE
   DenseMatrix dshape;
   DenseMatrix dshapedxt;
   Vector scratch;
#endif
   dshape.SetSize(ndof, dim);
   dshapedxt.SetSize(ndof, space_dim);
   scratch.SetSize(ndof);

   double mag_flux_buffer[3] = {};
   Vector mag_flux(mag_flux_buffer, space_dim);

   const IntegrationRule *ir = IntRule;
   if (ir == nullptr)
   {
      int order = [&]()
      {
         if (el.Space() == FunctionSpace::Pk)
         {
            return 2 * el.GetOrder() - 2;
         }
         else
         {
            // order = 2*el.GetOrder() - 2;  // <-- this seems to work fine too
            return 2 * el.GetOrder() + el.GetDim() - 1;
         }
      }();

      if (el.Space() == FunctionSpace::rQk)
      {
         ir = &RefinedIntRules.Get(el.GetGeomType(), order);
      }
      else
      {
         ir = &IntRules.Get(el.GetGeomType(), order);
      }
   }

   elvect = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);

      double w = alpha * ip.weight;

      el.CalcDShape(ip, dshape);
      Mult(dshape, trans.AdjugateJacobian(), dshapedxt);

      M.Eval(mag_flux, trans, ip);
      mag_flux *= w;

      scratch = 0.0;
      Vector grad_column;
      dshapedxt.GetColumnReference(0, grad_column);
      scratch.Add(mag_flux(1), grad_column);

      dshapedxt.GetColumnReference(1, grad_column);
      scratch.Add(-mag_flux(0), grad_column);

      elvect += scratch;
   }
}

void MagnetizationSource2DIntegratorMeshRevSens::AssembleRHSElementVect(
    const FiniteElement &mesh_el,
    ElementTransformation &mesh_trans,
    Vector &mesh_coords_bar)
{
   const int element = mesh_trans.ElementNo;
   const auto &el = *adjoint.FESpace()->GetFE(element);
   auto &trans = *adjoint.FESpace()->GetElementTransformation(element);

   const int mesh_ndof = mesh_el.GetDof();
   const int ndof = el.GetDof();
   const int dim = el.GetDim();
   const int space_dim = trans.GetSpaceDim();
   const int curl_dim = space_dim;

   /// get the proper element, transformation, and state vector
#ifdef MFEM_THREAD_SAFE
   mfem::Array<int> vdofs;
   mfem::Vector psi;
#endif
   auto *dof_tr = adjoint.FESpace()->GetElementVDofs(element, vdofs);
   adjoint.GetSubVector(vdofs, psi);
   if (dof_tr != nullptr)
   {
      dof_tr->InvTransformPrimal(psi);
   }

#ifdef MFEM_THREAD_SAFE
   DenseMatrix dshape;
   DenseMatrix dshapedxt;
   DenseMatrix dshapedxt_bar;
   DenseMatrix PointMat_bar;
   Vector scratch_bar;
#else
   auto &dshape = integ.dshape;
   auto &dshapedxt = integ.dshapedxt;
#endif

   dshape.SetSize(ndof, dim);
   dshapedxt.SetSize(ndof, space_dim);

   dshapedxt_bar.SetSize(ndof, space_dim);
   scratch_bar.SetSize(ndof);
   PointMat_bar.SetSize(space_dim, mesh_ndof);

   double mag_flux_buffer[3] = {};
   Vector mag_flux(mag_flux_buffer, space_dim);
   double mag_flux_bar_buffer[3] = {};
   Vector mag_flux_bar(mag_flux_bar_buffer, space_dim);

   // cast the ElementTransformation
   auto &isotrans = dynamic_cast<IsoparametricTransformation &>(mesh_trans);

   const IntegrationRule *ir = IntRule;
   if (ir == nullptr)
   {
      int order = [&]()
      {
         if (el.Space() == FunctionSpace::Pk)
         {
            return 2 * el.GetOrder() - 2;
         }
         else
         {
            // order = 2*el.GetOrder() - 2;  // <-- this seems to work fine too
            return 2 * el.GetOrder() + el.GetDim() - 1;
         }
      }();

      if (el.Space() == FunctionSpace::rQk)
      {
         ir = &RefinedIntRules.Get(el.GetGeomType(), order);
      }
      else
      {
         ir = &IntRules.Get(el.GetGeomType(), order);
      }
   }

   auto &alpha = integ.alpha;
   auto &M = integ.M;
   mesh_coords_bar.SetSize(mesh_ndof * space_dim);
   mesh_coords_bar = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);

      double w = alpha * ip.weight;

      el.CalcDShape(ip, dshape);
      Mult(dshape, trans.AdjugateJacobian(), dshapedxt);

      M.Eval(mag_flux, trans, ip);
      // mag_flux *= w;

      Vector grad_column_0;
      dshapedxt.GetColumnReference(0, grad_column_0);

      Vector grad_column_1;
      dshapedxt.GetColumnReference(1, grad_column_1);

      // scratch = 0.0;
      // add(mag_flux(1), grad_column_0, -mag_flux(0), grad_column_1, scratch);

      // const double psi_dot_scratch = psi * scratch;

      // elvect += scratch;
      /// dummy functional for adjoint-weighted residual
      // fun += psi_dot_scratch * w;

      /// start reverse pass
      double fun_bar = 1.0;

      /// fun += psi_dot_scratch * w;
      double psi_dot_scratch_bar = fun_bar * w;
      // double w_bar = fun_bar * psi_dot_scratch;

      /// const double psi_dot_scratch = psi * scratch;
      scratch_bar = 0.0;
      scratch_bar.Add(psi_dot_scratch_bar, psi);

      /// add(mag_flux(1), grad_column_0, -mag_flux(0), grad_column_1, scratch);
      /// Vector grad_column_1;
      /// dshapedxt.GetColumnReference(1, grad_column_1);
      /// Vector grad_column_0;
      /// dshapedxt.GetColumnReference(0, grad_column_0);
      dshapedxt_bar = 0.0;
      Vector grad_bar_column_1;
      dshapedxt_bar.GetColumnReference(1, grad_bar_column_1);
      Vector grad_bar_column_0;
      dshapedxt_bar.GetColumnReference(0, grad_bar_column_0);

      mag_flux_bar(1) = grad_column_0 * scratch_bar;
      mag_flux_bar(0) = -(grad_column_1 * scratch_bar);

      grad_bar_column_0.Add(mag_flux(1), scratch_bar);
      grad_bar_column_1.Add(-mag_flux(0), scratch_bar);

      /// M.Eval(mag_flux, trans, ip);
      PointMat_bar = 0.0;
      M.EvalRevDiff(mag_flux_bar, trans, ip, PointMat_bar);

      /// Mult(dshape, trans.AdjugateJacobian(), dshapedxt);
      double adj_jac_bar_buffer[9] = {};
      DenseMatrix adj_jac_bar(adj_jac_bar_buffer, space_dim, space_dim);
      MultAtB(dshape, dshapedxt_bar, adj_jac_bar);

      isotrans.AdjugateJacobianRevDiff(adj_jac_bar, PointMat_bar);

      // code to insert PointMat_bar into mesh_coords_bar;
      for (int j = 0; j < mesh_ndof; ++j)
      {
         for (int k = 0; k < curl_dim; ++k)
         {
            mesh_coords_bar(k * mesh_ndof + j) += PointMat_bar(k, j);
         }
      }
   }
}

void CurlCurlNLFIntegrator::AssembleElementVector(const FiniteElement &el,
                                                  ElementTransformation &trans,
                                                  const Vector &elfun,
                                                  Vector &elvect)
{
   /// number of degrees of freedom
   int ndof = el.GetDof();
   int dim = el.GetDim();
   int dimc = (dim == 3) ? 3 : 1;
   elvect.SetSize(ndof);
   elvect = 0.0;

#ifdef MFEM_THREAD_SAFE
   DenseMatrix curlshape(ndof, dimc), curlshape_dFt(ndof, dimc), M;
   // Vector b_vec(dimc);
#else
   curlshape.SetSize(ndof, dimc);
   curlshape_dFt.SetSize(ndof, dimc);
   // b_vec.SetSize(dimc);
#endif

   double b_vec_buffer[3] = {};
   Vector b_vec(b_vec_buffer, dimc);

   const IntegrationRule *ir = IntRule;
   if (ir == nullptr)
   {
      int order = [&]()
      {
         if (el.Space() == FunctionSpace::Pk)
         {
            return 2 * el.GetOrder() - 1;
         }
         else
         {
            return 2 * el.GetOrder();
         }
      }();

      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      b_vec = 0.0;
      const IntegrationPoint &ip = ir->IntPoint(i);

      trans.SetIntPoint(&ip);

      double w = ip.weight / trans.Weight();
      w *= alpha;

      if (dim == 3)
      {
         el.CalcCurlShape(ip, curlshape);
         MultABt(curlshape, trans.Jacobian(), curlshape_dFt);
      }
      else
      {
         el.CalcCurlShape(ip, curlshape_dFt);
      }
      curlshape_dFt.AddMultTranspose(elfun, b_vec);
      const double b_vec_norm = b_vec.Norml2();
      const double b_mag = b_vec_norm / trans.Weight();

      double model_val = model.Eval(trans, ip, b_mag);
      model_val *= w;
      b_vec *= model_val;

      curlshape_dFt.AddMult(b_vec, elvect);
   }
}

void CurlCurlNLFIntegrator::AssembleElementGrad(
    const mfem::FiniteElement &el,
    mfem::ElementTransformation &trans,
    const mfem::Vector &elfun,
    mfem::DenseMatrix &elmat)
{
   /// number of degrees of freedom
   int ndof = el.GetDof();
   int dim = el.GetDim();
   int dimc = (dim == 3) ? 3 : 1;
   elmat.SetSize(ndof);
   elmat = 0.0;

#ifdef MFEM_THREAD_SAFE
   DenseMatrix curlshape(ndof, dimc), curlshape_dFt(ndof, dimc);
   Vector scratch(ndof);
#else
   curlshape.SetSize(ndof, dimc);
   curlshape_dFt.SetSize(ndof, dimc);
   scratch.SetSize(ndof);
#endif

   double b_vec_buffer[3] = {};
   Vector b_vec(b_vec_buffer, dimc);

   const IntegrationRule *ir = IntRule;
   if (ir == nullptr)
   {
      int order = [&]()
      {
         if (el.Space() == FunctionSpace::Pk)
         {
            return 2 * el.GetOrder() - 1;
         }
         else
         {
            return 2 * el.GetOrder();
         }
      }();

      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      trans.SetIntPoint(&ip);

      double w = ip.weight / trans.Weight();
      w *= alpha;

      if (dim == 3)
      {
         el.CalcCurlShape(ip, curlshape);
         MultABt(curlshape, trans.Jacobian(), curlshape_dFt);
      }
      else
      {
         el.CalcCurlShape(ip, curlshape_dFt);
      }

      /// calculate B = curl(A)
      b_vec = 0.0;
      curlshape_dFt.MultTranspose(elfun, b_vec);
      b_vec /= trans.Weight();
      const double b_mag = b_vec.Norml2();

      /////////////////////////////////////////////////////////////////////////
      /// calculate first term of Jacobian
      /////////////////////////////////////////////////////////////////////////

      /// evaluate material model at ip
      double model_val = model.Eval(trans, ip, b_mag);
      /// multiply material value by integration weight
      model_val *= w;
      /// add first term to elmat
      AddMult_a_AAt(model_val, curlshape_dFt, elmat);
      // elmat.PrintMatlab(); std::cout << "\n";

      /////////////////////////////////////////////////////////////////////////
      /// calculate second term of Jacobian
      /////////////////////////////////////////////////////////////////////////
      if (abs(b_mag) > 1e-14)
      {
         /// calculate curl(N_i) dot curl(A), need to store in a DenseMatrix
         /// so we can take outer product of result to generate matrix
         scratch = 0.0;
         curlshape_dFt.Mult(b_vec, scratch);

         /// evaluate the derivative of the material model with respect to
         /// the norm of the grid function associated with the model at the
         /// point defined by ip, and scale by integration point weight
         double model_deriv = model.EvalStateDeriv(trans, ip, b_mag);
         model_deriv *= w;
         model_deriv /= b_mag;

         /// add second term to elmat
         // AddMult_a_AAt(model_deriv, temp_matrix, elmat);
         AddMult_a_VVt(model_deriv, scratch, elmat);

         // for (int i = 0; i < ndof; ++i)
         // {
         //    for (int j = 0; j < ndof; ++j)
         //    {
         //       try
         //       {
         //          if (!isfinite(elmat(i,j)))
         //          {
         //             throw MachException("nan!");
         //          }
         //       }
         //       catch(const std::exception& e)
         //       {
         //          std::cerr << e.what() << '\n';
         //       }
         //    }
         // }
      }
   }
}

void CurlCurlNLFIntegratorStateRevSens::AssembleRHSElementVect(
    const mfem::FiniteElement &el,
    mfem::ElementTransformation &trans,
    mfem::Vector &state_bar)
{
   /// get the proper element, transformation, and state and adjoint vectors
   int element = trans.ElementNo;
   auto *dof_tr = state.FESpace()->GetElementVDofs(element, vdofs);
   state.GetSubVector(vdofs, elfun);
   if (dof_tr != nullptr)
   {
      dof_tr->InvTransformPrimal(elfun);
   }

   dof_tr = adjoint.FESpace()->GetElementVDofs(element, vdofs);
   adjoint.GetSubVector(vdofs, psi);
   if (dof_tr != nullptr)
   {
      dof_tr->InvTransformPrimal(psi);
   }

   DenseMatrix elmat;
   integ.AssembleElementGrad(el, trans, elfun, elmat);

   state_bar.SetSize(psi.Size());
   elmat.MultTranspose(psi, state_bar);
}

void CurlCurlNLFIntegratorStateFwdSens::AssembleRHSElementVect(
    const mfem::FiniteElement &el,
    mfem::ElementTransformation &trans,
    mfem::Vector &res_dot)
{
   /// get the proper element, transformation, and state_dot vector
   int element = trans.ElementNo;
   auto *dof_tr = state.FESpace()->GetElementVDofs(element, vdofs);
   state.GetSubVector(vdofs, elfun);
   if (dof_tr != nullptr)
   {
      dof_tr->InvTransformPrimal(elfun);
   }

   dof_tr = state_dot.FESpace()->GetElementVDofs(element, vdofs);
   state_dot.GetSubVector(vdofs, elfun_dot);
   if (dof_tr != nullptr)
   {
      dof_tr->InvTransformPrimal(elfun_dot);
   }

   DenseMatrix elmat;
   integ.AssembleElementGrad(el, trans, elfun, elmat);

   res_dot.SetSize(elfun_dot.Size());
   elmat.Mult(elfun_dot, res_dot);
}

void CurlCurlNLFIntegratorMeshRevSens::AssembleRHSElementVect(
    const FiniteElement &mesh_el,
    ElementTransformation &mesh_trans,
    Vector &mesh_coords_bar)
{
   /// get the proper element, transformation, and state vector
#ifdef MFEM_THREAD_SAFE
   mfem::Array<int> vdofs;
   mfem::Vector elfun, psi;
#endif
   int element = mesh_trans.ElementNo;
   const auto &el = *state.FESpace()->GetFE(element);
   auto &trans = *state.FESpace()->GetElementTransformation(element);

   const int ndof = mesh_el.GetDof();
   const int el_ndof = el.GetDof();
   const int dim = el.GetDim();
   const int dimc = (dim == 3) ? 3 : 1;
   mesh_coords_bar.SetSize(ndof * dimc);
   mesh_coords_bar = 0.0;

   auto *dof_tr = state.FESpace()->GetElementVDofs(element, vdofs);
   state.GetSubVector(vdofs, elfun);
   if (dof_tr != nullptr)
   {
      dof_tr->InvTransformPrimal(elfun);
   }

   dof_tr = adjoint.FESpace()->GetElementVDofs(element, vdofs);
   adjoint.GetSubVector(vdofs, psi);
   if (dof_tr != nullptr)
   {
      dof_tr->InvTransformPrimal(psi);
   }

#ifdef MFEM_THREAD_SAFE
   DenseMatrix curlshape(el_ndof, dimc), curlshape_dFt(el_ndof, dimc);
   DenseMatrix curlshape_dFt_bar(
       dimc, el_ndof);  // transposed dimensions of curlshape_dFt so I don't
                        // have to transpose J later
   DenseMatrix PointMat_bar(dimc, ndof);
#else
   auto &curlshape = integ.curlshape;
   auto &curlshape_dFt = integ.curlshape_dFt;
   curlshape.SetSize(el_ndof, dimc);
   curlshape_dFt.SetSize(el_ndof, dimc);
   curlshape_dFt_bar.SetSize(
       dimc, el_ndof);  // transposed dimensions of curlshape_dFt so I don't
                        // have to transpose J later
   PointMat_bar.SetSize(dimc, ndof);
#endif
   auto &nu = integ.model;

   /// these vector's size is the spatial dimension we can stack allocate
   double b_vec_buffer[3] = {};
   Vector b_vec(b_vec_buffer, dim);
   double curl_psi_buffer[3] = {};
   Vector curl_psi(curl_psi_buffer, dim);

   // cast the ElementTransformation
   auto &isotrans = dynamic_cast<IsoparametricTransformation &>(trans);

   const IntegrationRule *ir = IntRule;
   if (ir == nullptr)
   {
      int order = [&]()
      {
         if (el.Space() == FunctionSpace::Pk)
         {
            return 2 * el.GetOrder() - 1;
         }
         else
         {
            return 2 * el.GetOrder();
         }
      }();

      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      isotrans.SetIntPoint(&ip);

      const double w = ip.weight / trans.Weight();

      if (dim == 3)
      {
         el.CalcCurlShape(ip, curlshape);
         MultABt(curlshape, isotrans.Jacobian(), curlshape_dFt);
      }
      else
      {
         el.CalcCurlShape(ip, curlshape_dFt);
      }

      b_vec = 0.0;
      curl_psi = 0.0;
      curlshape_dFt.AddMultTranspose(elfun, b_vec);
      curlshape_dFt.AddMultTranspose(psi, curl_psi);
      const double curl_psi_dot_b = curl_psi * b_vec;

      const double b_vec_norm = b_vec.Norml2();
      const double b_mag = b_vec_norm / trans.Weight();

      const double nu_val = nu.Eval(isotrans, ip, b_mag);

      /// dummy functional for adjoint-weighted residual
      // fun += nu_val * curl_psi_dot_b * w;

      /// start reverse pass
      double fun_bar = 1.0;

      /// fun += nu_val * curl_psi_dot_b * w;
      double nu_val_bar = fun_bar * curl_psi_dot_b * w;
      double curl_psi_dot_b_bar = fun_bar * nu_val * w;
      double w_bar = fun_bar * nu_val * curl_psi_dot_b;

      /// double nu_val = nu.Eval(isotrans, ip, b_mag);
      double b_mag_bar = 0.0;
      const double dnudb = nu.EvalStateDeriv(isotrans, ip, b_mag);
      b_mag_bar += nu_val_bar * dnudb;

      /// const double b_mag = b_vec_norm / trans.Weight();
      double b_vec_norm_bar = 0.0;
      double trans_weight_bar = 0.0;
      b_vec_norm_bar += b_mag_bar / trans.Weight();
      trans_weight_bar -= b_mag_bar * b_vec_norm / pow(trans.Weight(), 2);

      /// const double b_vec_norm = b_vec.Norml2();
      double b_vec_bar_buffer[3] = {};
      Vector b_vec_bar(b_vec_bar_buffer, dim);
      b_vec_bar = 0.0;
      add(b_vec_bar, b_vec_norm_bar / b_vec_norm, b_vec, b_vec_bar);

      /// const double curl_psi_dot_b = curl_psi * b_vec;
      double curl_psi_bar_buffer[3] = {};
      Vector curl_psi_bar(curl_psi_bar_buffer, dim);
      curl_psi_bar = 0.0;
      add(curl_psi_bar, curl_psi_dot_b_bar, b_vec, curl_psi_bar);
      add(b_vec_bar, curl_psi_dot_b_bar, curl_psi, b_vec_bar);

      curlshape_dFt_bar = 0.0;
      /// curlshape_dFt.AddMultTranspose(psi, curl_psi);
      AddMultVWt(curl_psi_bar, psi, curlshape_dFt_bar);
      /// curlshape_dFt.AddMultTranspose(elfun, b_vec);
      AddMultVWt(b_vec_bar, elfun, curlshape_dFt_bar);

      /// MultABt(curlshape, trans.Jacobian(), curlshape_dFt);
      double jac_bar_buffer[9] = {};
      DenseMatrix jac_bar(jac_bar_buffer, dim, dim);
      jac_bar = 0.0;
      AddMult(curlshape_dFt_bar, curlshape, jac_bar);

      /// const double w = ip.weight / trans.Weight();
      trans_weight_bar -= w_bar * ip.weight / pow(trans.Weight(), 2);

      PointMat_bar = 0.0;
      isotrans.WeightRevDiff(PointMat_bar);
      PointMat_bar *= trans_weight_bar;

      isotrans.JacobianRevDiff(jac_bar, PointMat_bar);

      // code to insert PointMat_bar into mesh_coords_bar;
      for (int j = 0; j < ndof; ++j)
      {
         for (int d = 0; d < dimc; ++d)
         {
            mesh_coords_bar(d * ndof + j) += PointMat_bar(d, j);
         }
      }
   }
}

void MagnetizationIntegrator::AssembleElementVector(
    const FiniteElement &el,
    ElementTransformation &trans,
    const Vector &elfun,
    Vector &elvect)
{
   // std::cout << "mag integ\n";
   /// number of degrees of freedom
   int ndof = el.GetDof();
   int dim = el.GetDim();

   /// I believe this takes advantage of a 2D problem not having
   /// a properly defined curl? Need more investigation
   int dimc = (dim == 3) ? 3 : 1;

#ifdef MFEM_THREAD_SAFE
   DenseMatrix curlshape(ndof, dimc), curlshape_dFt(ndof, dimc), M;
   Vector b_vec(dimc) mag_vec(dimc);
#else
   curlshape.SetSize(ndof, dimc);
   curlshape_dFt.SetSize(ndof, dimc);
   b_vec.SetSize(dimc);
   mag_vec.SetSize(dimc);
#endif

   elvect.SetSize(ndof);

   const IntegrationRule *ir = IntRule;
   if (ir == nullptr)
   {
      int order = [&]()
      {
         if (el.Space() == FunctionSpace::Pk)
         {
            return 2 * el.GetOrder() - 2;
         }
         else
         {
            return 2 * el.GetOrder();
         }
      }();

      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   elvect = 0.0;

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      trans.SetIntPoint(&ip);

      /// holds quadrature weight
      double w = alpha * ip.weight / trans.Weight();

      if (dim == 3)
      {
         el.CalcCurlShape(ip, curlshape);
         MultABt(curlshape, trans.Jacobian(), curlshape_dFt);
      }
      else
      {
         el.CalcCurlShape(ip, curlshape_dFt);
      }

      b_vec = 0.0;
      curlshape_dFt.AddMultTranspose(elfun, b_vec);
      double nu_val = nu->Eval(trans, ip, b_vec.Norml2());
      nu_val *= w;

      mag_vec = 0.0;
      mag->Eval(mag_vec, trans, ip);
      mag_vec *= nu_val;

      curlshape_dFt.AddMult(mag_vec, elvect);
   }
}

void MagnetizationIntegrator::AssembleElementGrad(
    const mfem::FiniteElement &el,
    mfem::ElementTransformation &trans,
    const mfem::Vector &elfun,
    mfem::DenseMatrix &elmat)
{
   elmat = 0.0;
   /*
   /// number of degrees of freedom
   int ndof = el.GetDof();
   int dim = el.GetDim();

   /// I believe this takes advantage of a 2D problem not having
   /// a properly defined curl? Need more investigation
   int dimc = (dim == 3) ? 3 : 1;

   /// holds quadrature weight
   double w;

#ifdef MFEM_THREAD_SAFE
   DenseMatrix curlshape(ndof,dimc), curlshape_dFt(ndof,dimc);
   Vector b_vec(dimc), mag_vec(dimc), temp_vec(ndof), temp_vec2(ndof);
#else
   curlshape.SetSize(ndof,dimc);
   curlshape_dFt.SetSize(ndof,dimc);
   b_vec.SetSize(dimc);
   mag_vec.SetSize(dimc);
   temp_vec.SetSize(ndof);
   temp_vec2.SetSize(ndof);
#endif

   elmat.SetSize(ndof);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order;
      if (el.Space() == FunctionSpace::Pk)
      {
         order = 2*el.GetOrder() - 2;
      }
      else
      {
         order = 2*el.GetOrder();
      }

      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   elmat = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      trans.SetIntPoint(&ip);

      w = ip.weight / trans.Weight();
      w *= alpha;

      if ( dim == 3 )
      {
         el.CalcCurlShape(ip, curlshape);
         MultABt(curlshape, trans.Jacobian(), curlshape_dFt);
      }
      else
      {
         el.CalcCurlShape(ip, curlshape_dFt);
      }

      /// calculate B = curl(A)
      b_vec = 0.0;
      curlshape_dFt.MultTranspose(elfun, b_vec);
      const double b_mag = b_vec.Norml2();

      if (abs(b_mag) > 1e-14)
      {
         /// TODO - is this thread safe?
         /// calculate curl(N_i) dot curl(A), need to store in a DenseMatrix
         /// so we can take outer product of result to generate matrix
         temp_vec = 0.0;
         curlshape_dFt.Mult(b_vec, temp_vec);
         DenseMatrix temp_matrix(temp_vec.GetData(), ndof, 1);

         mag_vec = 0.0;
         mag->Eval(mag_vec, trans, ip);

         temp_vec2 = 0.0;
         curlshape_dFt.Mult(mag_vec, temp_vec2);
         DenseMatrix temp_matrix2(temp_vec2.GetData(), ndof, 1);

         /// evaluate the derivative of the material model with respect to
         /// the norm of the grid function associated with the model at the
point
         /// defined by ip, and scale by integration point weight
         double nu_deriv = nu->EvalStateDeriv(trans, ip, b_mag);
         nu_deriv *= w;
         nu_deriv /= b_mag;

         AddMult_a_ABt(nu_deriv, temp_matrix2, temp_matrix, elmat);
      }
   }
   */
}

/** moved/replaced in mfem_common_integ.xpp
void VectorFECurldJdXIntegerator::AssembleRHSElementVect(
   const FiniteElement &mesh_el,
   ElementTransformation &mesh_trans,
   Vector &elvect)
{
   /// get the proper element, transformation, and adjoint and m vector
   Array<int> adj_vdofs, state_vdofs;
   Vector elfun, psi;
   Vector elfun_proj;
   int element = mesh_trans.ElementNo;

   /// get the ND elements used for curl shape
   const FiniteElement &nd_el = *adjoint->FESpace()->GetFE(element);
   ElementTransformation &nd_trans =
*adjoint->FESpace()->GetElementTransformation(element);

   /// get the RT elements used for V shape
   const FiniteElement &rt_el = *state->FESpace()->GetFE(element);
   ElementTransformation &rt_trans =
*state->FESpace()->GetElementTransformation(element);

   adjoint->FESpace()->GetElementVDofs(element, adj_vdofs);
   state->FESpace()->GetElementVDofs(element, state_vdofs);



   const IntegrationRule *ir = NULL;
   {
      int order = rt_el.GetOrder() + nd_el.GetOrder() - 1; // <--
      ir = &IntRules.Get(rt_el.GetGeomType(), order);
   }

   elfun_proj.SetSize(state_vdofs.Size());
   rt_el.Project(*vec_coeff, rt_trans, elfun_proj);

   state->GetSubVector(state_vdofs, elfun);
   adjoint->GetSubVector(adj_vdofs, psi);

   Vector diff(elfun);
   diff -= elfun_proj;

   int ndof = mesh_el.GetDof();
   int nd_ndof = nd_el.GetDof();
   int rt_ndof = rt_el.GetDof();
   int dim = nd_el.GetDim();
   int dimc = (dim == 3) ? 3 : 1;
   elvect.SetSize(ndof*dimc);
   elvect = 0.0;
#ifdef MFEM_THREAD_SAFE
   DenseMatrix curlshape(nd_ndof,dimc), curlshape_dFt(nd_ndof,dimc);
   DenseMatrix vshape(rt_ndof, dimc), vshape_dFt(rt_ndof, dimc);
   Vector m_vec(dimc), m_hat(dimc), curl_psi(dimc), curl_psi_hat(dimc);
#else
   curlshape.SetSize(nd_ndof,dimc);
   curlshape_dFt.SetSize(nd_ndof,dimc);
   vshape.SetSize(rt_ndof, dimc);
   vshape_dFt.SetSize(rt_ndof, dimc);
   m_vec.SetSize(dimc);
   m_hat.SetSize(dimc);
   curl_psi.SetSize(dimc);
   curl_psi_hat.SetSize(dimc);
#endif
   DenseMatrix PointMat_bar(dimc, ndof);

   // cast the ElementTransformation
   IsoparametricTransformation &isotrans =
   dynamic_cast<IsoparametricTransformation&>(nd_trans);

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      PointMat_bar = 0.0;
      m_vec = 0.0;
      m_hat = 0.0;
      curl_psi_hat = 0.0;
      curl_psi = 0.0;

      const IntegrationPoint &ip = ir->IntPoint(i);

      isotrans.SetIntPoint(&ip);

      if ( dim == 3 )
      {
         nd_el.CalcCurlShape(ip, curlshape);
         MultABt(curlshape, isotrans.Jacobian(), curlshape_dFt);

         rt_el.CalcVShape(ip, vshape);
         MultABt(vshape, isotrans.Jacobian(), vshape_dFt);
      }
      else
      {
         nd_el.CalcCurlShape(ip, curlshape_dFt);
         rt_el.CalcVShape(ip, vshape_dFt);
      }
      curlshape.AddMultTranspose(psi, curl_psi_hat);
      curlshape_dFt.AddMultTranspose(psi, curl_psi);
      vshape.AddMultTranspose(elfun, m_hat);
      vshape_dFt.AddMultTranspose(elfun, m_vec);

      double nu_val = nu->Eval(isotrans, ip);

      double curl_psi_dot_m = curl_psi * m_vec;

      // nu * (\partial a^T b / \partial J) / |J|
      DenseMatrix Jac_bar(3);
      MultVWt(m_vec, curl_psi_hat, Jac_bar);
      AddMultVWt(curl_psi, m_hat, Jac_bar);
      Jac_bar *= nu_val / isotrans.Weight();

      // (- nu * a^T b / |J|^2)  * \partial |J| / \partial X
      isotrans.WeightRevDiff(PointMat_bar);
      PointMat_bar *= -nu_val * curl_psi_dot_m / pow(isotrans.Weight(), 2.0);

      isotrans.JacobianRevDiff(Jac_bar, PointMat_bar);

      // sensitivity with respect to the projection of the coefficient
      if (vec_coeff)
      {
         Vector P_bar(rt_ndof);
         vshape_dFt.Mult(curl_psi, P_bar);
         P_bar *= 1 / isotrans.Weight();
         rt_el.ProjectRevDiff(P_bar, *vec_coeff, isotrans, PointMat_bar);
      }

      for (int j = 0; j < ndof ; ++j)
      {
         for (int d = 0; d < dimc; ++d)
         {
            elvect(d*ndof + j) += alpha * ip.weight * PointMat_bar(d,j);
         }
      }
   }
}
*/

/** moved/replaced in mfem_common_integ.xpp
void VectorFEMassdJdXIntegerator::AssembleRHSElementVect(
   const FiniteElement &mesh_el,
   ElementTransformation &mesh_trans,
   Vector &elvect)
{
   /// get the proper element, transformation, and adjoint and m vector
   Array<int> adj_vdofs, state_vdofs;
   Vector elfun, psi;
   int element = mesh_trans.ElementNo;

   /// get the ND elements used for adjoint and J
   const FiniteElement &el = *adjoint->FESpace()->GetFE(element);
   ElementTransformation &trans =
*adjoint->FESpace()->GetElementTransformation(element);

   adjoint->FESpace()->GetElementVDofs(element, adj_vdofs);
   state->FESpace()->GetElementVDofs(element, state_vdofs);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // int order = 2 * el.GetOrder();
      int order = trans.OrderW() + 2 * el.GetOrder();
      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   state->GetSubVector(state_vdofs, elfun);
   adjoint->GetSubVector(adj_vdofs, psi);

   int ndof = mesh_el.GetDof();
   int el_ndof = el.GetDof();
   int dim = el.GetDim();
   int dimc = (dim == 3) ? 3 : 1;
   elvect.SetSize(ndof*dimc);
   elvect = 0.0;
#ifdef MFEM_THREAD_SAFE
   DenseMatrix vshape(el_ndof, dimc), vshape_dFt(el_ndof, dimc);
   Vector v_j_hat(dimc), v_j_vec(dimc), v_psi_hat(dimc), v_psi_vec(dimc);
#else
   vshape.SetSize(el_ndof, dimc);
   vshape_dFt.SetSize(el_ndof, dimc);
   v_j_hat.SetSize(dimc);
   v_j_vec.SetSize(dimc);
   v_psi_hat.SetSize(dimc);
   v_psi_vec.SetSize(dimc);
#endif
   DenseMatrix PointMat_bar(dimc, ndof);

   // cast the ElementTransformation
   IsoparametricTransformation &isotrans =
   dynamic_cast<IsoparametricTransformation&>(trans);

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      PointMat_bar = 0.0;
      v_j_hat = 0.0;
      v_j_vec = 0.0;
      v_psi_hat = 0.0;
      v_psi_vec = 0.0;

      const IntegrationPoint &ip = ir->IntPoint(i);

      isotrans.SetIntPoint(&ip);

      if ( dim == 3 )
      {
         el.CalcVShape(ip, vshape);
         Mult(vshape, isotrans.AdjugateJacobian(), vshape_dFt);
      }
      else
      {
         el.CalcVShape(ip, vshape_dFt);
      }
      vshape.AddMultTranspose(psi, v_psi_hat);
      vshape_dFt.AddMultTranspose(psi, v_psi_vec);
      vshape.AddMultTranspose(elfun, v_j_hat);
      vshape_dFt.AddMultTranspose(elfun, v_j_vec);

      double v_psi_dot_v_j = v_psi_vec * v_j_vec;

      // nu * (\partial a^T b / \partial J) / |J|
      DenseMatrix Jac_bar(3);
      MultVWt(v_j_hat, v_psi_vec, Jac_bar);
      AddMultVWt(v_psi_hat, v_j_vec, Jac_bar);
      Jac_bar *=  1 / isotrans.Weight();

      // (- a^T b / |J|^2)  * \partial |J| / \partial X
      isotrans.WeightRevDiff(PointMat_bar);
      PointMat_bar *= -v_psi_dot_v_j / pow(isotrans.Weight(), 2.0);

      isotrans.AdjugateJacobianRevDiff(Jac_bar, PointMat_bar);

      // sensitivity with respect to the projection of the coefficient
      if (vec_coeff)
      {
         Vector P_bar(el_ndof);
         vshape_dFt.Mult(v_psi_vec, P_bar);
         P_bar *= 1 / isotrans.Weight();
         el.ProjectRevDiff(P_bar, *vec_coeff, isotrans, PointMat_bar);
      }

      for (int j = 0; j < ndof ; ++j)
      {
         for (int d = 0; d < dimc; ++d)
         {
            elvect(d*ndof + j) += alpha * ip.weight * PointMat_bar(d,j);
         }
      }
   }
}
*/

/** moved/replaced in mfem_common_integ.xpp
void VectorFEWeakDivergencedJdXIntegrator::AssembleRHSElementVect(
   const FiniteElement &mesh_el,
   ElementTransformation &mesh_trans,
   Vector &elvect)
{
   /// get the proper element, transformation, and adjoint and m vector
   Array<int> adj_vdofs, state_vdofs;
   Vector elfun, psi;
   int element = mesh_trans.ElementNo;

   /// get the H1 elements used for curl shape
   const FiniteElement &h1_el = *adjoint->FESpace()->GetFE(element);
   ElementTransformation &h1_trans =
*adjoint->FESpace()->GetElementTransformation(element);

   /// get the ND elements used for V shape
   const FiniteElement &nd_el = *state->FESpace()->GetFE(element);

   adjoint->FESpace()->GetElementVDofs(element, adj_vdofs);
   state->FESpace()->GetElementVDofs(element, state_vdofs);

   state->GetSubVector(state_vdofs, elfun);
   adjoint->GetSubVector(adj_vdofs, psi);

   int ndof = mesh_el.GetDof();
   int h1_ndof = h1_el.GetDof();
   int nd_ndof = nd_el.GetDof();
   int dim = nd_el.GetDim();
   int dimc = (dim == 3) ? 3 : 1;

   const IntegrationRule *ir = NULL;
   {
      int order = (nd_el.Space() == FunctionSpace::Pk) ?
                  (nd_el.GetOrder() + h1_el.GetOrder() - 1) :
                  (nd_el.GetOrder() + h1_el.GetOrder() + 2*(dim-2));
      ir = &IntRules.Get(h1_el.GetGeomType(), order);
   }

   elvect.SetSize(ndof*dimc);
   elvect = 0.0;
#ifdef MFEM_THREAD_SAFE
   DenseMatrix dshape(h1_ndof,dimc), dshape_dFt(h1_ndof,dimc);
   DenseMatrix vshape(nd_ndof, dimc), vshape_dFt(nd_ndof, dimc);
   Vector v_vec(dimc), v_hat(dimc), d_psi(dimc), d_psi_hat(dimc);
#else
   dshape.SetSize(h1_ndof,dimc);
   dshape_dFt.SetSize(h1_ndof,dimc);
   vshape.SetSize(nd_ndof, dimc);
   vshape_dFt.SetSize(nd_ndof, dimc);
   v_vec.SetSize(dimc);
   v_hat.SetSize(dimc);
   d_psi.SetSize(dimc);
   d_psi_hat.SetSize(dimc);
#endif
   DenseMatrix PointMat_bar(dimc, ndof);

   // cast the ElementTransformation
   IsoparametricTransformation &isotrans =
   dynamic_cast<IsoparametricTransformation&>(h1_trans);

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      PointMat_bar = 0.0;
      v_vec = 0.0;
      v_hat = 0.0;
      d_psi_hat = 0.0;
      d_psi = 0.0;

      const IntegrationPoint &ip = ir->IntPoint(i);

      isotrans.SetIntPoint(&ip);

      if ( dim == 3 )
      {
         nd_el.CalcVShape(ip, vshape);
         Mult(vshape, isotrans.AdjugateJacobian(), vshape_dFt);
      }
      else
      {
         nd_el.CalcVShape(ip, vshape_dFt);
      }
      h1_el.CalcDShape(ip, dshape);
      Mult(dshape, isotrans.AdjugateJacobian(), dshape_dFt);

      dshape.AddMultTranspose(psi, d_psi_hat);
      dshape_dFt.AddMultTranspose(psi, d_psi);
      vshape.AddMultTranspose(elfun, v_hat);
      vshape_dFt.AddMultTranspose(elfun, v_vec);

      double d_psi_dot_v = d_psi * v_vec;

      // (\partial a^T b / \partial J) / |J|
      DenseMatrix Jac_bar(3);
      MultVWt(d_psi_hat, v_vec, Jac_bar);
      AddMultVWt(v_hat, d_psi, Jac_bar);
      Jac_bar *= 1 / isotrans.Weight();

      // (- a^T b / |J|^2)  * \partial |J| / \partial X
      isotrans.WeightRevDiff(PointMat_bar);
      PointMat_bar *= -d_psi_dot_v / pow(isotrans.Weight(), 2.0);

      isotrans.AdjugateJacobianRevDiff(Jac_bar, PointMat_bar);

      // sensitivity with respect to the projection of the coefficient
      if (vec_coeff)
      {
         Vector P_bar(nd_ndof);
         vshape_dFt.Mult(d_psi, P_bar);
         P_bar *= 1 / isotrans.Weight();
         nd_el.ProjectRevDiff(P_bar, *vec_coeff, isotrans, PointMat_bar);
      }

      for (int j = 0; j < ndof ; ++j)
      {
         for (int d = 0; d < dimc; ++d)
         {
            /// NOTE: this is -= instead of += since the weight is negated in
            /// the original integrator (line 1312 in bilininteg.cpp)
            elvect(d*ndof + j) -= alpha * ip.weight * PointMat_bar(d,j);
         }
      }
   }
}
*/

/** moved/replaced in mfem_common_integ.xpp
void VectorFEDomainLFMeshSensInteg::AssembleRHSElementVect(
   const FiniteElement &mesh_el,
   ElementTransformation &mesh_trans,
   Vector &elvect)
{
   /// get the proper element, transformation, and adjoint and m vector
   Array<int> adj_vdofs, state_vdofs;
   Vector elfun, psi;
   int element = mesh_trans.ElementNo;

   /// get the ND elements used for adjoint and J
   const FiniteElement &el = *adjoint->FESpace()->GetFE(element);
   ElementTransformation &trans =
*adjoint->FESpace()->GetElementTransformation(element);

   adjoint->FESpace()->GetElementVDofs(element, adj_vdofs);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // int order = 2 * el.GetOrder();
      int order = trans.OrderW() + 2 * el.GetOrder();
      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   adjoint->GetSubVector(adj_vdofs, psi);

   int ndof = mesh_el.GetDof();
   int el_ndof = el.GetDof();
   int dim = el.GetDim();
   int dimc = (dim == 3) ? 3 : 1;
   elvect.SetSize(ndof*dimc);
   elvect = 0.0;
#ifdef MFEM_THREAD_SAFE
   DenseMatrix vshape(el_ndof, dimc), vshape_dFt(el_ndof, dimc);
   Vector v_psi_vec(dimc);
#else
   vshape.SetSize(el_ndof, dimc);
   vshape_dFt.SetSize(el_ndof, dimc);
   v_psi_vec.SetSize(dimc);
   v_psi_hat.SetSize(dimc);
#endif
   DenseMatrix PointMat_bar(dimc, ndof);
   Vector vec(dimc);
   vec = 0.0;

   // cast the ElementTransformation
   IsoparametricTransformation &isotrans =
   dynamic_cast<IsoparametricTransformation&>(trans);

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      PointMat_bar = 0.0;
      v_psi_vec = 0.0;
      v_psi_hat = 0.0;

      const IntegrationPoint &ip = ir->IntPoint(i);

      isotrans.SetIntPoint(&ip);

      vec_coeff.Eval(vec, isotrans, ip);

      if ( dim == 3 )
      {
         el.CalcVShape(ip, vshape);
         Mult(vshape, isotrans.AdjugateJacobian(), vshape_dFt);
      }
      else
      {
         el.CalcVShape(ip, vshape_dFt);
      }
      vshape.AddMultTranspose(psi, v_psi_hat);
      vshape_dFt.AddMultTranspose(psi, v_psi_vec);

      // \partial a^T b / \partial K
      DenseMatrix Jac_bar(3);
      MultVWt(v_psi_hat, vec, Jac_bar);
      isotrans.AdjugateJacobianRevDiff(Jac_bar, PointMat_bar);

      // sensitivity with respect to the projection of the coefficient
      vec_coeff.EvalRevDiff(v_psi_vec, isotrans, ip, PointMat_bar);

      for (int j = 0; j < ndof ; ++j)
      {
         for (int d = 0; d < dimc; ++d)
         {
            elvect(d*ndof + j) += alpha * ip.weight * PointMat_bar(d,j);
         }
      }
   }
}
*/

/** moved/replaced in mfem_common_integ.xpp
void GridFuncMeshSensIntegrator::AssembleRHSElementVect(
   const FiniteElement &mesh_el,
   ElementTransformation &mesh_trans,
   Vector &elvect)
{
   /// get the proper element, transformation, and adjoint and m vector
   Array<int> adj_vdofs;
   Vector psi;
   int element = mesh_trans.ElementNo;

   /// get the elements used the adjoint dofs
   const FiniteElement &el = *adjoint->FESpace()->GetFE(element);
   ElementTransformation &trans =
*adjoint->FESpace()->GetElementTransformation(element);

   adjoint->FESpace()->GetElementVDofs(element, adj_vdofs);
   adjoint->GetSubVector(adj_vdofs, psi);

   int ndof = mesh_el.GetDof();
   int dim = el.GetDim();
   int dimc = (dim == 3) ? 3 : 1;

   elvect.SetSize(ndof*dimc);
   elvect = 0.0;

   DenseMatrix PointMat_bar(dimc, ndof);

   // cast the ElementTransformation
   IsoparametricTransformation &isotrans =
   dynamic_cast<IsoparametricTransformation&>(trans);

   PointMat_bar = 0.0;
   el.ProjectRevDiff(psi, *vec_coeff, isotrans, PointMat_bar);

   for (int j = 0; j < ndof ; ++j)
   {
      for (int d = 0; d < dimc; ++d)
      {
         elvect(d*ndof + j) += alpha * PointMat_bar(d,j);
      }
   }
}
*/

double MagneticEnergyIntegrator::GetElementEnergy(const FiniteElement &el,
                                                  ElementTransformation &trans,
                                                  const Vector &elfun)
{
   /// number of degrees of freedom
   int ndof = el.GetDof();
   int dim = el.GetDim();
   int space_dim = trans.GetSpaceDim();
   int curl_dim = space_dim;

#ifdef MFEM_THREAD_SAFE
   DenseMatrix curlshape;
   DenseMatrix curlshape_dFt;
#endif
   curlshape.SetSize(ndof, curl_dim);
   curlshape_dFt.SetSize(ndof, curl_dim);

   double b_vec_buffer[3] = {};
   Vector b_vec(b_vec_buffer, curl_dim);

   const IntegrationRule *ir = IntRule;
   if (ir == nullptr)
   {
      int order = [&]()
      {
         if (el.Space() == FunctionSpace::Pk)
         {
            return 2 * el.GetOrder() - 2;
         }
         else
         {
            return 2 * el.GetOrder();
         }
      }();

      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   double fun = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);

      /// holds quadrature weight
      const double trans_weight = trans.Weight();
      const double w = ip.weight * trans_weight;

      if (dim == 3)
      {
         el.CalcCurlShape(ip, curlshape);
         MultABt(curlshape, trans.Jacobian(), curlshape_dFt);
      }
      else
      {
         el.CalcDShape(ip, curlshape);
         Mult(curlshape, trans.AdjugateJacobian(), curlshape_dFt);
      }

      b_vec = 0.0;
      curlshape_dFt.AddMultTranspose(elfun, b_vec);
      const double b_vec_norm = b_vec.Norml2();
      const double b_mag = b_vec_norm / trans_weight;

      const double energy = calcMagneticEnergy(trans, ip, nu, b_mag);
      fun += energy * w;
   }
   return fun;
}

void MagneticEnergyIntegrator::AssembleElementVector(
    const FiniteElement &el,
    ElementTransformation &trans,
    const Vector &elfun,
    Vector &elfun_bar)
{
   /// number of degrees of freedom
   int ndof = el.GetDof();
   int dim = el.GetDim();
   int space_dim = trans.GetSpaceDim();
   int curl_dim = space_dim;

#ifdef MFEM_THREAD_SAFE
   DenseMatrix curlshape;
   DenseMatrix curlshape_dFt;
#endif
   curlshape.SetSize(ndof, curl_dim);
   curlshape_dFt.SetSize(ndof, curl_dim);

   // Vector curlshape_dFt_bar_buffer(curl_dim * ndof);

   double b_vec_buffer[3] = {};
   Vector b_vec(b_vec_buffer, curl_dim);

   double b_vec_bar_buffer[3] = {};
   Vector b_vec_bar(b_vec_bar_buffer, curl_dim);

   const IntegrationRule *ir = IntRule;
   if (ir == nullptr)
   {
      int order = [&]()
      {
         if (el.Space() == FunctionSpace::Pk)
         {
            return 2 * el.GetOrder() - 2;
         }
         else
         {
            return 2 * el.GetOrder();
         }
      }();

      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   elfun_bar.SetSize(ndof);
   elfun_bar = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);

      /// holds quadrature weight
      double trans_weight = trans.Weight();
      const double w = ip.weight * trans_weight;

      if (dim == 3)
      {
         el.CalcCurlShape(ip, curlshape);
         MultABt(curlshape, trans.Jacobian(), curlshape_dFt);
      }
      else
      {
         el.CalcDShape(ip, curlshape);
         Mult(curlshape, trans.AdjugateJacobian(), curlshape_dFt);
      }

      b_vec = 0.0;
      curlshape_dFt.AddMultTranspose(elfun, b_vec);

      const double b_vec_norm = b_vec.Norml2();
      const double b_mag = b_vec_norm / trans_weight;

      // const double energy = calcMagneticEnergy(trans, ip, nu, b_mag);
      // fun += energy * w;
      double fun_bar = 1.0;

      double energy_bar = 0.0;
      // double w_bar = 0.0;
      energy_bar += fun_bar * w;
      // w_bar += fun_bar * energy;

      /// const double energy = calcMagneticEnergy(trans, ip, nu, b_mag);
      double b_mag_bar = 0.0;
      const double energy_dot = calcMagneticEnergyDot(trans, ip, nu, b_mag);
      b_mag_bar += energy_bar * energy_dot;

      double b_vec_norm_bar = 0.0;
      // double trans_weight_bar = 0.0;
      /// const double b_mag = b_vec_norm / trans_weight;
      b_vec_norm_bar += b_mag_bar / trans_weight;
      // trans_weight_bar -= b_mag_bar * b_vec_norm / pow(trans_weight, 2);

      b_vec_bar = 0.0;
      /// const double b_vec_norm = b_vec.Norml2();
      add(b_vec_bar, b_vec_norm_bar / b_vec_norm, b_vec, b_vec_bar);

      /// curlshape_dFt.AddMultTranspose(elfun, b_vec);
      curlshape_dFt.AddMult(b_vec_bar, elfun_bar);
   }
}

void MagneticEnergyIntegratorMeshSens::AssembleRHSElementVect(
    const FiniteElement &mesh_el,
    ElementTransformation &mesh_trans,
    Vector &mesh_coords_bar)
{
   /// get the proper element, transformation, and state vector
#ifdef MFEM_THREAD_SAFE
   Array<int> vdofs;
   Vector elfun;
#endif
   const int element = mesh_trans.ElementNo;
   const auto &el = *state.FESpace()->GetFE(element);
   auto &trans = *state.FESpace()->GetElementTransformation(element);

   const int ndof = mesh_el.GetDof();
   const int el_ndof = el.GetDof();
   const int dim = el.GetDim();
   const int space_dim = trans.GetSpaceDim();
   const int curl_dim = space_dim;

   auto *dof_tr = state.FESpace()->GetElementVDofs(element, vdofs);
   state.GetSubVector(vdofs, elfun);
   if (dof_tr != nullptr)
   {
      dof_tr->InvTransformPrimal(elfun);
   }

   auto &nu = integ.nu;

#ifdef MFEM_THREAD_SAFE
   DenseMatrix curlshape(el_ndof, curl_dim);
   DenseMatrix curlshape_dFt(el_ndof, curl_dim);
   DenseMatrix curlshape_dFt_bar(curl_dim, el_ndof);
   DenseMatrix PointMat_bar(curl_dim, ndof);
#else
   auto &curlshape = integ.curlshape;
   auto &curlshape_dFt = integ.curlshape_dFt;
#endif
   curlshape.SetSize(el_ndof, curl_dim);
   curlshape_dFt.SetSize(el_ndof, curl_dim);
   curlshape_dFt_bar.SetSize(curl_dim, el_ndof);
   PointMat_bar.SetSize(curl_dim, ndof);

   // Vector curlshape_dFt_bar_buffer(curl_dim * ndof);

   double b_vec_buffer[3] = {};
   Vector b_vec(b_vec_buffer, curl_dim);

   double b_vec_bar_buffer[3] = {};
   Vector b_vec_bar(b_vec_bar_buffer, curl_dim);

   // cast the ElementTransformation
   auto &isotrans = dynamic_cast<IsoparametricTransformation &>(trans);

   const IntegrationRule *ir = IntRule;
   if (ir == nullptr)
   {
      int order = [&]()
      {
         if (el.Space() == FunctionSpace::Pk)
         {
            return 2 * el.GetOrder() - 2;
         }
         else
         {
            return 2 * el.GetOrder();
         }
      }();

      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   mesh_coords_bar.SetSize(ndof * space_dim);
   mesh_coords_bar = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);

      /// holds quadrature weight
      double trans_weight = trans.Weight();
      const double w = ip.weight * trans_weight;

      if (dim == 3)
      {
         el.CalcCurlShape(ip, curlshape);
         MultABt(curlshape, trans.Jacobian(), curlshape_dFt);
      }
      else
      {
         el.CalcDShape(ip, curlshape);
         Mult(curlshape, trans.AdjugateJacobian(), curlshape_dFt);
      }

      b_vec = 0.0;
      curlshape_dFt.AddMultTranspose(elfun, b_vec);

      const double b_vec_norm = b_vec.Norml2();
      const double b_mag = b_vec_norm / trans_weight;
      const double energy = calcMagneticEnergy(trans, ip, nu, b_mag);

      // fun += energy * w;
      double fun_bar = 1.0;

      double energy_bar = 0.0;
      double w_bar = 0.0;
      energy_bar += fun_bar * w;
      w_bar += fun_bar * energy;

      /// const double energy = calcMagneticEnergy(trans, ip, nu, b_mag);
      double b_mag_bar = 0.0;
      const double energy_dot = calcMagneticEnergyDot(trans, ip, nu, b_mag);
      b_mag_bar += energy_bar * energy_dot;

      double b_vec_norm_bar = 0.0;
      double trans_weight_bar = 0.0;
      /// const double b_mag = b_vec_norm / trans_weight;
      b_vec_norm_bar += b_mag_bar / trans_weight;
      trans_weight_bar -= b_mag_bar * b_vec_norm / pow(trans_weight, 2);

      b_vec_bar = 0.0;
      /// const double b_vec_norm = b_vec.Norml2();
      add(b_vec_bar, b_vec_norm_bar / b_vec_norm, b_vec, b_vec_bar);

      PointMat_bar = 0.0;
      if (dim == 3)
      {
         /// curlshape_dFt.AddMultTranspose(elfun, b_vec);
         // transposed dimensions of curlshape_dFt so I don't have to transpose
         // jac_bar later
         // DenseMatrix curlshape_dFt_bar_(curlshape_dFt_bar_buffer.GetData(),
         // curl_dim, el_ndof); DenseMatrix curlshape_dFt_bar_(curl_dim,
         // el_ndof);
         curlshape_dFt_bar.SetSize(curl_dim, el_ndof);
         MultVWt(b_vec_bar, elfun, curlshape_dFt_bar);

         /// MultABt(curlshape, trans.Jacobian(), curlshape_dFt);
         double jac_bar_buffer[9] = {};
         DenseMatrix jac_bar(jac_bar_buffer, space_dim, space_dim);
         jac_bar = 0.0;
         AddMult(curlshape_dFt_bar, curlshape, jac_bar);
         isotrans.JacobianRevDiff(jac_bar, PointMat_bar);
      }
      else  // Dealing with scalar H1 field representing Az
      {
         /// curlshape_dFt.AddMultTranspose(elfun, b_vec);
         // DenseMatrix curlshape_dFt_bar_(curlshape_dFt_bar_buffer.GetData(),
         // el_ndof, curl_dim); DenseMatrix curlshape_dFt_bar_(el_ndof,
         // curl_dim);
         curlshape_dFt_bar.SetSize(el_ndof, curl_dim);
         MultVWt(elfun, b_vec_bar, curlshape_dFt_bar);

         /// Mult(curlshape, trans.AdjugateJacobian(), curlshape_dFt);
         double adj_bar_buffer[9] = {};
         DenseMatrix adj_bar(adj_bar_buffer, space_dim, space_dim);
         // adj_bar = 0.0;
         MultAtB(curlshape, curlshape_dFt_bar, adj_bar);
         isotrans.AdjugateJacobianRevDiff(adj_bar, PointMat_bar);
      }

      /// const double w = ip.weight * trans_weight;
      trans_weight_bar += w_bar * ip.weight;

      // double trans_weight = trans.Weight();
      isotrans.WeightRevDiff(trans_weight_bar, PointMat_bar);

      /// code to insert PointMat_bar into mesh_coords_bar;
      for (int j = 0; j < ndof; ++j)
      {
         for (int d = 0; d < space_dim; ++d)
         {
            mesh_coords_bar(d * ndof + j) += PointMat_bar(d, j);
         }
      }

      /*
      /// holds quadrature weight
      const double w = ip.weight * trans.Weight();
      if (dim == 3)
      {
         el.CalcCurlShape(ip, curlshape);
         MultABt(curlshape, trans.Jacobian(), curlshape_dFt);
      }
      else
      {
         el.CalcCurlShape(ip, curlshape_dFt);
      }

      b_vec = 0.0;
      curlshape_dFt.AddMultTranspose(elfun, b_vec);
      const double b_vec_norm = b_vec.Norml2();
      const double b_mag = b_vec_norm / trans.Weight();
      const double energy = calcMagneticEnergy(trans, ip, nu, b_mag);
      // fun += energy * w;

      /// start reverse pass
      double fun_bar = 1.0;

      /// fun += qp_en * w;
      double energy_bar = 0.0;
      double w_bar = 0.0;
      energy_bar += fun_bar * w;
      w_bar += fun_bar * energy;

      /// const double energy = calcMagneticEnergy(trans, ip, nu, b_mag);
      double b_mag_bar = 0.0;
      const double energy_dot = calcMagneticEnergyDot(trans, ip, nu, b_mag);
      b_mag_bar += energy_bar * energy_dot;

      std::cout << "b_mag_bar: " << b_mag_bar << "\n";

      double b_vec_norm_bar = 0.0;
      double trans_weight_bar = 0.0;
      /// const double b_mag = b_vec_norm / trans.Weight();
      b_vec_norm_bar += b_mag_bar / trans.Weight();
      trans_weight_bar -= b_mag_bar * b_vec_norm / pow(trans.Weight(), 2);

      Vector b_vec_bar(curl_dim);
      b_vec_bar = 0.0;
      /// const double b_vec_norm = b_vec.Norml2();
      add(b_vec_bar, b_vec_norm_bar / b_vec_norm, b_vec, b_vec_bar);

      /// curlshape_dFt.AddMultTranspose(elfun, b_vec);
      MultVWt(b_vec_bar, elfun, curlshape_dFt_bar);

      /// MultABt(curlshape, trans.Jacobian(), curlshape_dFt);
      DenseMatrix jac_bar(curl_dim);
      jac_bar = 0.0;
      AddMult(curlshape_dFt_bar, curlshape, jac_bar);

      /// const double w = ip.weight * trans.Weight();
      trans_weight_bar += w_bar * ip.weight;

      PointMat_bar = 0.0;
      isotrans.WeightRevDiff(PointMat_bar);
      PointMat_bar *= trans_weight_bar;

      isotrans.JacobianRevDiff(jac_bar, PointMat_bar);

      // code to insert PointMat_bar into mesh_coords_bar;
      for (int j = 0; j < ndof; ++j)
      {
         for (int d = 0; d < curl_dim; ++d)
         {
            mesh_coords_bar(d * ndof + j) += PointMat_bar(d, j);
         }
      }
      */
   }
}

/** commenting out co-energy stuff since I'm stopping maintaining it
double MagneticCoenergyIntegrator::GetElementEnergy(
   const FiniteElement &el,
   ElementTransformation &trans,
   const Vector &elfun)
{
   /// number of degrees of freedom
   int ndof = el.GetDof();
   int dim = el.GetDim();

   /// I believe this takes advantage of a 2D problem not having
   /// a properly defined curl? Need more investigation
   int dimc = (dim == 3) ? 3 : 1;

   /// holds quadrature weight
   double w;

#ifdef MFEM_THREAD_SAFE
   DenseMatrix curlshape(ndof,dimc), curlshape_dFt(ndof,dimc), M;
   Vector b_vec(dimc);
#else
   curlshape.SetSize(ndof,dimc);
   curlshape_dFt.SetSize(ndof,dimc);
   b_vec.SetSize(dimc);
#endif

   const IntegrationRule *ir = NULL;
   const IntegrationRule *segment_ir = NULL;
   if (ir == NULL)
   {
      int order;
      if (el.Space() == FunctionSpace::Pk)
      {
         order = 2*el.GetOrder() - 2;
      }
      else
      {
         order = 2*el.GetOrder();
      }

      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   if (segment_ir == NULL)
   {
      // int order;
      // if (el.Space() == FunctionSpace::Pk)
      // {
      //    order = 2*el.GetOrder() - 2;
      // }
      // else
      // {
      //    order = 2*el.GetOrder();
      // }

      // segment_ir = &IntRules.Get(Geometry::Type::SEGMENT, 2*(order+1));
      segment_ir = &IntRules.Get(Geometry::Type::SEGMENT, 12);
   }

   double fun = 0.0;

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      trans.SetIntPoint(&ip);
      w = ip.weight;

      if ( dim == 3 )
      {
         el.CalcCurlShape(ip, curlshape);
         MultABt(curlshape, trans.Jacobian(), curlshape_dFt);
      }
      else
      {
         el.CalcCurlShape(ip, curlshape_dFt);
      }

      b_vec = 0.0;
      curlshape_dFt.AddMultTranspose(elfun, b_vec);
      b_vec /= trans.Weight();
      const double b_mag = b_vec.Norml2();
      const double nu_val = nu->Eval(trans, ip, b_mag);

      const double qp_en = integrateBH(segment_ir, trans, ip,
                                       0.0, nu_val * b_mag);

      fun += qp_en * w;
   }
   return fun;
}

void MagneticCoenergyIntegrator::AssembleElementVector(
   const mfem::FiniteElement &el,
   mfem::ElementTransformation &trans,
   const mfem::Vector &elfun,
   mfem::Vector &elvect)
{
   /// number of degrees of freedom
   int ndof = el.GetDof();
   int dim = el.GetDim();

   elvect.SetSize(ndof);
   elvect = 0.0;

   /// I believe this takes advantage of a 2D problem not having
   /// a properly defined curl? Need more investigation
   int dimc = (dim == 3) ? 3 : 1;

   /// holds quadrature weight
   double w;

#ifdef MFEM_THREAD_SAFE
   DenseMatrix curlshape(ndof,dimc), curlshape_dFt(ndof,dimc), M;
   Vector b_vec(dimc), temp_vec(ndof);
#else
   curlshape.SetSize(ndof,dimc);
   curlshape_dFt.SetSize(ndof,dimc);
   b_vec.SetSize(dimc);
   temp_vec.SetSize(ndof);
#endif

   const IntegrationRule *ir = NULL;
   const IntegrationRule *segment_ir = NULL;
   {
      int order;
      if (el.Space() == FunctionSpace::Pk)
      {
         order = 2*el.GetOrder() - 2;
      }
      else
      {
         order = 2*el.GetOrder();
      }

      ir = &IntRules.Get(el.GetGeomType(), order);
   }
   /// TODO make segment's integration much higher than elements
   {
      // int order;
      // if (el.Space() == FunctionSpace::Pk)
      // {
      //    order = 2*el.GetOrder() - 2;
      // }
      // else
      // {
      //    order = 2*el.GetOrder();
      // }

      // segment_ir = &IntRules.Get(Geometry::Type::SEGMENT, 2*(order+1));
      segment_ir = &IntRules.Get(Geometry::Type::SEGMENT, 12);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      trans.SetIntPoint(&ip);

      w = ip.weight / trans.Weight();

      if ( dim == 3 )
      {
         el.CalcCurlShape(ip, curlshape);
         MultABt(curlshape, trans.Jacobian(), curlshape_dFt);
      }
      else
      {
         el.CalcCurlShape(ip, curlshape_dFt);
      }

      b_vec = 0.0;
      curlshape_dFt.AddMultTranspose(elfun, b_vec);
      b_vec /= trans.Weight();
      const double b_mag = b_vec.Norml2();
      const double nu_val = nu->Eval(trans, ip, b_mag);
      const double dnu_dB = nu->EvalStateDeriv(trans, ip, b_mag);

      /// temp_vec = curl(N_i) dot curl(A)
      temp_vec = 0.0;
      curlshape_dFt.Mult(b_vec, temp_vec);
      double dwp_dh = RevADintegrateBH(segment_ir, trans, ip,
                                       0, nu_val * b_mag);
      temp_vec *= dwp_dh*(dnu_dB + nu_val/b_mag);
      temp_vec *= w;
      elvect += temp_vec;
   }
}

double MagneticCoenergyIntegrator::integrateBH(
   const IntegrationRule *ir,
   ElementTransformation &trans,
   const IntegrationPoint &old_ip,
   double lower_bound,
   double upper_bound)
{
   /// compute int_0^{\nu*B} \frac{H}{\nu} dH
   double qp_en = 0.0;
   for (int j = 0; j < ir->GetNPoints(); j++)
   {
      const IntegrationPoint &ip = ir->IntPoint(j);
      double xi = ip.x * (upper_bound - lower_bound);
      qp_en += ip.weight * xi / nu->Eval(trans, old_ip, xi);
   }
   qp_en *= (upper_bound - lower_bound);
   return qp_en;
}

double MagneticCoenergyIntegrator::FDintegrateBH(
   const IntegrationRule *ir,
   ElementTransformation &trans,
   const IntegrationPoint &old_ip,
   double lower_bound,
   double upper_bound)
{
   double delta = 1e-5;

   double fd_val;
   fd_val = integrateBH(ir, trans, old_ip, lower_bound, upper_bound + delta);
   fd_val -= integrateBH(ir, trans, old_ip, lower_bound, upper_bound -
delta); return fd_val / (2*delta);
}

double MagneticCoenergyIntegrator::RevADintegrateBH(
   const IntegrationRule *ir,
   ElementTransformation &trans,
   const IntegrationPoint &old_ip,
   double lower_bound,
   double upper_bound)
{
   /// compute int_0^{\nu*B} \frac{H}{\nu} dH
   double qp_en = 0.0;
   for (int j = 0; j < ir->GetNPoints(); j++)
   {
      const IntegrationPoint &ip = ir->IntPoint(j);
      double xi = ip.x * (upper_bound - lower_bound);
      qp_en += ip.weight * xi / nu->Eval(trans, old_ip, xi);
   }

   /// insert forward code here to compute qp_en, but not the last part
   /// where you multiply by (upper_bound - lower_bound)
   /// start reverse mode for int_0^{\nu*B} \frac{H}{\nu} dH
   // return qp_en*(upper_bound - lower_bound);
   double upper_bound_bar = qp_en;
   double qp_en_bar = (upper_bound - lower_bound);
   for (int j = 0; j < ir->GetNPoints(); j++)
   {
      const IntegrationPoint &ip = ir->IntPoint(j);
      double xi = ip.x * (upper_bound - lower_bound);
      // qp_en += ip.weight * xi / nu->Eval(trans, old_ip, xi);
      double xi_bar = qp_en_bar * ip.weight / nu->Eval(trans, old_ip, xi);
      xi_bar -= (qp_en_bar * ip.weight * xi * nu->EvalStateDeriv(trans,
old_ip, xi) / pow(nu->Eval(trans, old_ip, xi), 2.0));
      // double xi = ip.x * (upper_bound - lower_bound);
      upper_bound_bar += ip.x*xi_bar;
   }
   return upper_bound_bar;
}

void MagneticCoenergyIntegrator::AssembleRHSElementVect(
   const FiniteElement &mesh_el,
   ElementTransformation &mesh_trans,
   Vector &elvect)
{
   /// get the proper element, transformation, and state vector
   Array<int> vdofs; Vector elfun;
   int element = mesh_trans.ElementNo;
   const FiniteElement *el = state.FESpace()->GetFE(element);
   ElementTransformation *trans =
state.FESpace()->GetElementTransformation(element);
   state.FESpace()->GetElementVDofs(element, vdofs);

   const IntegrationRule *ir = NULL;
   const IntegrationRule *segment_ir = NULL;
   {
      int order;
      if (el->Space() == FunctionSpace::Pk)
      {
         order = 2*el->GetOrder() - 2;
      }
      else
      {
         order = 2*el->GetOrder();
      }

      ir = &IntRules.Get(el->GetGeomType(), order);
   }
   /// TODO make segment's integration much higher than elements
   {
      int order;
      if (el->Space() == FunctionSpace::Pk)
      {
         order = 2*el->GetOrder() - 2;
      }
      else
      {
         order = 2*el->GetOrder();
      }

      segment_ir = &IntRules.Get(Geometry::Type::SEGMENT, 12);
   }
   state.GetSubVector(vdofs, elfun);

   int ndof = mesh_el.GetDof();
   int el_ndof = el->GetDof();
   int dim = el->GetDim();
   int dimc = (dim == 3) ? 3 : 1;
   elvect.SetSize(ndof*dimc);
   elvect = 0.0;
#ifdef MFEM_THREAD_SAFE
   DenseMatrix curlshape(ndof,dimc), curlshape_dFt(ndof,dimc), M;
   Vector b_vec(dimc), b_hat(dimc);
#else
   curlshape.SetSize(el_ndof,dimc);
   curlshape_dFt.SetSize(el_ndof,dimc);
   b_vec.SetSize(dimc);
   b_hat.SetSize(dimc);
#endif
   DenseMatrix PointMat_bar(dimc, ndof);

   // cast the ElementTransformation
   IsoparametricTransformation &isotrans =
   dynamic_cast<IsoparametricTransformation&>(*trans);

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      PointMat_bar = 0.0;

      const IntegrationPoint &ip = ir->IntPoint(i);
      trans->SetIntPoint(&ip);
      // double w = ip.weight / trans->Weight();
      if ( dim == 3 )
      {
         el->CalcCurlShape(ip, curlshape);
         MultABt(curlshape, trans->Jacobian(), curlshape_dFt);
      }
      else
      {
         el->CalcCurlShape(ip, curlshape_dFt);
      }

      b_hat = 0.0;
      b_vec = 0.0;
      curlshape.AddMultTranspose(elfun, b_hat);
      curlshape_dFt.AddMultTranspose(elfun, b_vec);
      b_vec /= trans->Weight();
      const double b_mag = b_vec.Norml2();

      const double nu_val = nu->Eval(*trans, ip, b_mag);
      const double dnu_dB = nu->EvalStateDeriv(*trans, ip, b_mag);

      const double wp = integrateBH(segment_ir, *trans, ip,
                              0.0, nu_val * b_mag);

      // start reverse sweep
      const double dwp_dh = RevADintegrateBH(segment_ir, *trans, ip,
                                             0.0, nu_val * b_mag);

      DenseMatrix BB_hatT(3);
      MultVWt(b_vec, b_hat, BB_hatT);
      BB_hatT *= dwp_dh*(dnu_dB + nu_val/b_mag); // / trans->Weight();

      isotrans.WeightRevDiff(PointMat_bar);
      PointMat_bar *= -wp / pow(trans->Weight(), 2.0);
      isotrans.JacobianRevDiff(BB_hatT, PointMat_bar);

      for (int j = 0; j < ndof ; ++j)
      {
         for (int d = 0; d < dimc; ++d)
         {
            elvect(d*ndof + j) += ip.weight * PointMat_bar(d,j);
         }
      }
   }
}
*/

double BNormIntegrator::GetElementEnergy(const FiniteElement &el,
                                         ElementTransformation &trans,
                                         const Vector &elfun)
{
   /// number of degrees of freedom
   int ndof = el.GetDof();
   int dim = el.GetDim();

   /// I believe this takes advantage of a 2D problem not having
   /// a properly defined curl? Need more investigation
   int dimc = (dim == 3) ? 3 : 1;

#ifdef MFEM_THREAD_SAFE
   DenseMatrix curlshape(ndof, dimc), curlshape_dFt(ndof, dimc), M;
   Vector b_vec(dimc);
#else
   curlshape.SetSize(ndof, dimc);
   curlshape_dFt.SetSize(ndof, dimc);
   b_vec.SetSize(dimc);
#endif

   const IntegrationRule *ir = IntRule;
   if (ir == nullptr)
   {
      int order = [&]()
      {
         if (el.Space() == FunctionSpace::Pk)
         {
            return 2 * el.GetOrder() - 2;
         }
         else
         {
            return 2 * el.GetOrder();
         }
      }();

      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   double fun = 0.0;

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      b_vec = 0.0;
      const IntegrationPoint &ip = ir->IntPoint(i);

      trans.SetIntPoint(&ip);

      /// holds quadrature weight
      auto w = ip.weight * trans.Weight();

      if (dim == 3)
      {
         el.CalcCurlShape(ip, curlshape);
         MultABt(curlshape, trans.Jacobian(), curlshape_dFt);
      }
      else
      {
         el.CalcCurlShape(ip, curlshape_dFt);
      }

      curlshape_dFt.AddMultTranspose(elfun, b_vec);
      const double b_mag = b_vec.Norml2() / trans.Weight();
      fun += b_mag * w;
   }
   return fun;
}

void BNormIntegrator::AssembleElementVector(const mfem::FiniteElement &el,
                                            mfem::ElementTransformation &trans,
                                            const mfem::Vector &elfun,
                                            mfem::Vector &elvect)
{
   /// number of degrees of freedom
   int ndof = el.GetDof();
   int dim = el.GetDim();

   elvect.SetSize(ndof);
   elvect = 0.0;

   /// I believe this takes advantage of a 2D problem not having
   /// a properly defined curl? Need more investigation
   int dimc = (dim == 3) ? 3 : 1;

#ifdef MFEM_THREAD_SAFE
   DenseMatrix curlshape(ndof, dimc), curlshape_dFt(ndof, dimc), M;
   Vector b_vec(dimc), temp_vec(ndof);
#else
   curlshape.SetSize(ndof, dimc);
   curlshape_dFt.SetSize(ndof, dimc);
   b_vec.SetSize(dimc);
   temp_vec.SetSize(ndof);
#endif

   const IntegrationRule *ir = IntRule;
   if (ir == nullptr)
   {
      int order = [&]()
      {
         if (el.Space() == FunctionSpace::Pk)
         {
            return 2 * el.GetOrder() - 2;
         }
         else
         {
            return 2 * el.GetOrder();
         }
      }();

      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      b_vec = 0.0;
      const IntegrationPoint &ip = ir->IntPoint(i);

      trans.SetIntPoint(&ip);

      /// holds quadrature weight
      auto w = ip.weight / trans.Weight();

      if (dim == 3)
      {
         el.CalcCurlShape(ip, curlshape);
         MultABt(curlshape, trans.Jacobian(), curlshape_dFt);
      }
      else
      {
         el.CalcCurlShape(ip, curlshape_dFt);
      }

      curlshape_dFt.AddMultTranspose(elfun, b_vec);
      double b_mag = b_vec.Norml2() / trans.Weight();

      /// temp_vec = curl(N_i) dot curl(A)
      temp_vec = 0.0;
      curlshape_dFt.Mult(b_vec, temp_vec);
      temp_vec /= b_mag;
      temp_vec *= w;
      elvect += temp_vec;
   }
}

double BNormSquaredIntegrator::GetElementEnergy(const FiniteElement &el,
                                                ElementTransformation &trans,
                                                const Vector &elfun)
{
   /// number of degrees of freedom
   int ndof = el.GetDof();
   int dim = el.GetDim();

   /// I believe this takes advantage of a 2D problem not having
   /// a properly defined curl? Need more investigation
   int dimc = (dim == 3) ? 3 : 1;

#ifdef MFEM_THREAD_SAFE
   DenseMatrix curlshape(ndof, dimc), curlshape_dFt(ndof, dimc);
#else
   curlshape.SetSize(ndof, dimc);
   curlshape_dFt.SetSize(ndof, dimc);
#endif

   double b_vec_buffer[3] = {};
   Vector b_vec(b_vec_buffer, dimc);

   const IntegrationRule *ir = IntRule;
   if (ir == nullptr)
   {
      int order = [&]()
      {
         if (el.Space() == FunctionSpace::Pk)
         {
            return 2 * el.GetOrder() - 2;
         }
         else
         {
            return 2 * el.GetOrder();
         }
      }();

      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   double fun = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);

      if (dim == 3)
      {
         el.CalcCurlShape(ip, curlshape);
         MultABt(curlshape, trans.Jacobian(), curlshape_dFt);
      }
      else
      {
         el.CalcCurlShape(ip, curlshape_dFt);
      }

      b_vec = 0.0;
      curlshape_dFt.AddMultTranspose(elfun, b_vec);

      auto trans_weight = trans.Weight();
      const double b_mag = b_vec.Norml2() / trans_weight;
      fun += b_mag * b_mag * ip.weight * trans_weight;
   }
   return fun;
}

void BNormdJdx::AssembleRHSElementVect(const FiniteElement &mesh_el,
                                       ElementTransformation &mesh_trans,
                                       Vector &elvect)
{
   /// get the proper element, transformation, and state vector
   Array<int> vdofs;
   Vector elfun;
   int element = mesh_trans.ElementNo;
   const FiniteElement &el = *state.FESpace()->GetFE(element);
   ElementTransformation *trans =
       state.FESpace()->GetElementTransformation(element);

   const IntegrationRule *ir = IntRule;
   if (ir == nullptr)
   {
      int order = [&]()
      {
         if (el.Space() == FunctionSpace::Pk)
         {
            return 2 * el.GetOrder() - 2;
         }
         else
         {
            return 2 * el.GetOrder();
         }
      }();

      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   auto *dof_tr = state.FESpace()->GetElementVDofs(element, vdofs);
   state.GetSubVector(vdofs, elfun);
   if (dof_tr != nullptr)
   {
      dof_tr->InvTransformPrimal(elfun);
   }

   int ndof = mesh_el.GetDof();
   int el_ndof = el.GetDof();
   int dim = el.GetDim();
   int dimc = (dim == 3) ? 3 : 1;
   elvect.SetSize(ndof * dimc);
   elvect = 0.0;
#ifdef MFEM_THREAD_SAFE
   DenseMatrix curlshape(ndof, dimc), curlshape_dFt(ndof, dimc), M;
   Vector b_vec(dimc);
#else
   curlshape.SetSize(el_ndof, dimc);
   curlshape_dFt.SetSize(el_ndof, dimc);
   b_vec.SetSize(dimc);
#endif
   DenseMatrix PointMat_bar(dimc, ndof);
   Vector b_hat(dimc);

   // cast the ElementTransformation
   auto &isotrans = dynamic_cast<IsoparametricTransformation &>(*trans);

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trans->SetIntPoint(&ip);
      double w = ip.weight;
      if (dim == 3)
      {
         el.CalcCurlShape(ip, curlshape);
         MultABt(curlshape, trans->Jacobian(), curlshape_dFt);
      }
      else
      {
         el.CalcCurlShape(ip, curlshape_dFt);
      }

      b_vec = 0.0;
      curlshape_dFt.AddMultTranspose(elfun, b_vec);
      // start reverse sweep
      PointMat_bar = 0.0;

      /** the following computes `\partial (||B||/|J|) / \partial X`
       * This is useful elsewhere, but for the test to pass we need
       * `\partial (||B||/|J|)*(w*|J|) / \partial X`...

      double weight_bar = -b_vec.Norml2() / pow(trans->Weight(), 2.0);
      isotrans.WeightRevDiff(PointMat_bar);
      PointMat_bar *= weight_bar;

      b_hat = 0.0;
      curlshape.AddMultTranspose(elfun, b_hat);
      DenseMatrix BB_hatT(3);
      MultVWt(b_vec, b_hat, BB_hatT);
      BB_hatT *= 1.0 / (trans->Weight() * b_vec.Norml2());
      isotrans.JacobianRevDiff(BB_hatT, PointMat_bar);
      */

      b_hat = 0.0;
      curlshape.AddMultTranspose(elfun, b_hat);
      DenseMatrix BB_hatT(3);
      MultVWt(b_vec, b_hat, BB_hatT);
      BB_hatT *= 1.0 / b_vec.Norml2();
      isotrans.JacobianRevDiff(BB_hatT, PointMat_bar);

      // code to insert PointMat_bar into elvect;
      for (int j = 0; j < ndof; ++j)
      {
         for (int d = 0; d < dimc; ++d)
         {
            elvect(d * ndof + j) += w * PointMat_bar(d, j);
         }
      }
   }
}

double nuBNormIntegrator::GetElementEnergy(const FiniteElement &el,
                                           ElementTransformation &trans,
                                           const Vector &elfun)
{
   /// number of degrees of freedom
   int ndof = el.GetDof();
   int dim = el.GetDim();

   /// I believe this takes advantage of a 2D problem not having
   /// a properly defined curl? Need more investigation
   int dimc = (dim == 3) ? 3 : 1;

#ifdef MFEM_THREAD_SAFE
   DenseMatrix curlshape(ndof, dimc), curlshape_dFt(ndof, dimc), M;
   Vector b_vec(dimc);
#else
   curlshape.SetSize(ndof, dimc);
   curlshape_dFt.SetSize(ndof, dimc);
   b_vec.SetSize(dimc);
#endif

   const IntegrationRule *ir = IntRule;
   if (ir == nullptr)
   {
      int order = [&]()
      {
         if (el.Space() == FunctionSpace::Pk)
         {
            return 2 * el.GetOrder() - 2;
         }
         else
         {
            return 2 * el.GetOrder();
         }
      }();

      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   double fun = 0.0;

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      b_vec = 0.0;
      const IntegrationPoint &ip = ir->IntPoint(i);

      trans.SetIntPoint(&ip);

      /// holds quadrature weight
      auto w = ip.weight / trans.Weight();

      if (dim == 3)
      {
         el.CalcCurlShape(ip, curlshape);
         MultABt(curlshape, trans.Jacobian(), curlshape_dFt);
      }
      else
      {
         el.CalcCurlShape(ip, curlshape_dFt);
      }

      curlshape_dFt.AddMultTranspose(elfun, b_vec);
      fun += nu->Eval(trans, ip, b_vec.Norml2()) * b_vec.Norml2() * w;
   }
   return fun;
}

void nuBNormIntegrator::AssembleElementVector(
    const mfem::FiniteElement &el,
    mfem::ElementTransformation &trans,
    const mfem::Vector &elfun,
    mfem::Vector &elvect)
{
   /// number of degrees of freedom
   int ndof = el.GetDof();
   int dim = el.GetDim();

   elvect.SetSize(ndof);
   elvect = 0.0;

   /// I believe this takes advantage of a 2D problem not having
   /// a properly defined curl? Need more investigation
   int dimc = (dim == 3) ? 3 : 1;

#ifdef MFEM_THREAD_SAFE
   DenseMatrix curlshape(ndof, dimc), curlshape_dFt(ndof, dimc), M;
   Vector b_vec(dimc), temp_vec(ndof);
#else
   curlshape.SetSize(ndof, dimc);
   curlshape_dFt.SetSize(ndof, dimc);
   b_vec.SetSize(dimc);
   temp_vec.SetSize(ndof);
#endif

   const IntegrationRule *ir = IntRule;
   if (ir == nullptr)
   {
      int order = [&]()
      {
         if (el.Space() == FunctionSpace::Pk)
         {
            return 2 * el.GetOrder() - 2;
         }
         else
         {
            return 2 * el.GetOrder();
         }
      }();

      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      b_vec = 0.0;
      const IntegrationPoint &ip = ir->IntPoint(i);

      trans.SetIntPoint(&ip);

      auto w = ip.weight / trans.Weight();

      if (dim == 3)
      {
         el.CalcCurlShape(ip, curlshape);
         MultABt(curlshape, trans.Jacobian(), curlshape_dFt);
      }
      else
      {
         el.CalcCurlShape(ip, curlshape_dFt);
      }

      curlshape_dFt.AddMultTranspose(elfun, b_vec);
      double b_mag = b_vec.Norml2();

      /// temp_vec = curl(N_i) dot curl(A)
      temp_vec = 0.0;
      curlshape_dFt.Mult(b_vec, temp_vec);
      double nu_val = nu->Eval(trans, ip, b_mag);
      double dnu_dB = nu->EvalStateDeriv(trans, ip, b_mag);
      temp_vec *= (dnu_dB + nu_val / b_mag);
      temp_vec *= w;
      elvect += temp_vec;
   }
}

void nuBNormdJdx::AssembleRHSElementVect(const FiniteElement &mesh_el,
                                         ElementTransformation &mesh_trans,
                                         Vector &elvect)
{
   /// get the proper element, transformation, and state vector
   Array<int> vdofs;
   Vector elfun;
   int element = mesh_trans.ElementNo;
   const FiniteElement &el = *state.FESpace()->GetFE(element);
   ElementTransformation *trans =
       state.FESpace()->GetElementTransformation(element);

   const IntegrationRule *ir = IntRule;
   if (ir == nullptr)
   {
      int order = [&]()
      {
         if (el.Space() == FunctionSpace::Pk)
         {
            return 2 * el.GetOrder() - 2;
         }
         else
         {
            return 2 * el.GetOrder();
         }
      }();

      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   auto *dof_tr = state.FESpace()->GetElementVDofs(element, vdofs);
   state.GetSubVector(vdofs, elfun);
   if (dof_tr != nullptr)
   {
      dof_tr->InvTransformPrimal(elfun);
   }

   int ndof = mesh_el.GetDof();
   int el_ndof = el.GetDof();
   int dim = el.GetDim();
   int dimc = (dim == 3) ? 3 : 1;
   elvect.SetSize(ndof * dimc);
   elvect = 0.0;
#ifdef MFEM_THREAD_SAFE
   DenseMatrix curlshape(ndof, dimc), curlshape_dFt(ndof, dimc), M;
   Vector b_vec(dimc);
#else
   curlshape.SetSize(el_ndof, dimc);
   curlshape_dFt.SetSize(el_ndof, dimc);
   b_vec.SetSize(dimc);
#endif
   // DenseMatrix PointMat_bar(dimc, ndof);
   DenseMatrix PointMat_bar_1(dimc, ndof);
   DenseMatrix PointMat_bar_2(dimc, ndof);
   DenseMatrix PointMat_bar_3(dimc, ndof);
   // Vector DofVal(elfun.Size());

   Vector b_hat(dimc);

   // cast the ElementTransformation
   auto &isotrans = dynamic_cast<IsoparametricTransformation &>(*trans);

   for (int i = 0; i < ir->GetNPoints(); ++i)
   {
      PointMat_bar_1 = 0.0;
      PointMat_bar_2 = 0.0;
      PointMat_bar_3 = 0.0;
      b_vec = 0.0;
      b_hat = 0.0;
      const IntegrationPoint &ip = ir->IntPoint(i);

      trans->SetIntPoint(&ip);
      if (dim == 3)
      {
         el.CalcCurlShape(ip, curlshape);
         MultABt(curlshape, trans->Jacobian(), curlshape_dFt);
      }
      else
      {
         el.CalcCurlShape(ip, curlshape_dFt);
      }
      curlshape.AddMultTranspose(elfun, b_hat);
      curlshape_dFt.AddMultTranspose(elfun, b_vec);

      double nu_val = nu->Eval(*trans, ip, b_vec.Norml2());
      double nu_deriv = nu->EvalStateDeriv(*trans, ip, b_vec.Norml2());

      Vector dNormBdB(b_vec);
      dNormBdB /= b_vec.Norml2();
      DenseMatrix dBdJ(b_hat.Size(), b_vec.Size());
      MultVWt(dNormBdB, b_hat, dBdJ);
      isotrans.JacobianRevDiff(dBdJ, PointMat_bar_1);
      PointMat_bar_1 *= nu_val / isotrans.Weight();

      isotrans.WeightRevDiff(PointMat_bar_2);
      PointMat_bar_2 *= -nu_val * b_vec.Norml2() / pow(isotrans.Weight(), 2);

      isotrans.JacobianRevDiff(dBdJ, PointMat_bar_3);
      PointMat_bar_3 *= b_vec.Norml2() * nu_deriv / isotrans.Weight();
      // for (int i = 0; i < ir->GetNPoints(); i++)
      // {
      //    b_vec = 0.0;
      //    const IntegrationPoint &ip = ir->IntPoint(i);
      //    trans->SetIntPoint(&ip);
      //    double w = ip.weight / trans->Weight();
      //    if ( dim == 3 )
      //    {
      //       el->CalcCurlShape(ip, curlshape);
      //       MultABt(curlshape, trans->Jacobian(), curlshape_dFt);
      //    }
      //    else
      //    {
      //       el->CalcCurlShape(ip, curlshape_dFt);
      //    }
      //    curlshape_dFt.AddMultTranspose(elfun, b_vec);
      //    // start reverse sweep

      //    PointMat_bar = 0.0;
      //    // fun += b_vec.Norml2() * w;
      //    Vector b_vec_bar(b_vec);
      //    b_vec_bar *= w / b_vec.Norml2();
      //    double w_bar = b_vec.Norml2();
      //    // curlshape_dFt.AddMultTranspose(elfun, b_vec);
      //    DenseMatrix curlshape_dFt_bar(elfun.Size(), b_vec_bar.Size());
      //    MultVWt(elfun, b_vec_bar, curlshape_dFt_bar);
      //    // MultABt(curlshape, trans.Jacobian(), curlshape_dFt);
      //    DenseMatrix Jac_bar(3);
      //    MultAtB(curlshape_dFt_bar, curlshape, Jac_bar);
      //    // w = ip.weight / trans.Weight();
      //    double weight_bar = -w_bar*ip.weight/pow(trans->Weight(), 2.0);
      //    isotrans.WeightRevDiff(PointMat_bar);
      //    PointMat_bar *= weight_bar;
      //    // This is out of order because WeightRevDiff needs to scale
      //    PointMat_bar first isotrans.JacobianRevDiff(Jac_bar,
      //    PointMat_bar);
      //    // code to insert PointMat_bar into elvect;

      for (int j = 0; j < ndof; ++j)
      {
         for (int d = 0; d < dimc; ++d)
         {
            elvect(d * ndof + j) +=
                ip.weight * (PointMat_bar_1(d, j) + PointMat_bar_2(d, j) +
                             PointMat_bar_3(d, j));
            // elvect(d*ndof + j) += PointMat_bar(d,j);
         }
      }
   }
}

double nuFuncIntegrator::GetElementEnergy(const FiniteElement &el,
                                          ElementTransformation &trans,
                                          const Vector &elfun)
{
   /// number of degrees of freedom
   int ndof = el.GetDof();
   int dim = el.GetDim();

   /// I believe this takes advantage of a 2D problem not having
   /// a properly defined curl? Need more investigation
   int dimc = (dim == 3) ? 3 : 1;

#ifdef MFEM_THREAD_SAFE
   DenseMatrix curlshape(ndof, dimc), curlshape_dFt(ndof, dimc), M;
   Vector b_vec(dimc);
#else
   curlshape.SetSize(ndof, dimc);
   curlshape_dFt.SetSize(ndof, dimc);
   b_vec.SetSize(dimc);
#endif

   const IntegrationRule *ir = mfem::NonlinearFormIntegrator::IntRule;
   if (ir == nullptr)
   {
      int order = [&]()
      {
         if (el.Space() == FunctionSpace::Pk)
         {
            return 2 * el.GetOrder() - 2;
         }
         else
         {
            return 2 * el.GetOrder();
         }
      }();

      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   double fun = 0.0;

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      b_vec = 0.0;
      const IntegrationPoint &ip = ir->IntPoint(i);

      trans.SetIntPoint(&ip);

      auto w = ip.weight / trans.Weight();

      if (dim == 3)
      {
         el.CalcCurlShape(ip, curlshape);
         MultABt(curlshape, trans.Jacobian(), curlshape_dFt);
      }
      else
      {
         el.CalcCurlShape(ip, curlshape_dFt);
      }

      curlshape_dFt.AddMultTranspose(elfun, b_vec);
      const double b_mag = b_vec.Norml2() / trans.Weight();
      const double nu_val = nu->Eval(trans, ip, b_mag);
      fun += nu_val * w;
   }
   return fun;
}

void nuFuncIntegrator::AssembleRHSElementVect(const FiniteElement &mesh_el,
                                              ElementTransformation &mesh_trans,
                                              Vector &elvect)
{
   /// get the proper element, transformation, and state vector
   Array<int> vdofs;
   Vector elfun;
   int element = mesh_trans.ElementNo;
   const FiniteElement &el = *state->FESpace()->GetFE(element);
   ElementTransformation *trans =
       state->FESpace()->GetElementTransformation(element);

   const IntegrationRule *ir = mfem::NonlinearFormIntegrator::IntRule;
   if (ir == nullptr)
   {
      int order = [&]()
      {
         if (el.Space() == FunctionSpace::Pk)
         {
            return 2 * el.GetOrder() - 2;
         }
         else
         {
            return 2 * el.GetOrder();
         }
      }();

      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   auto *dof_tr = state->FESpace()->GetElementVDofs(element, vdofs);
   state->GetSubVector(vdofs, elfun);
   if (dof_tr != nullptr)
   {
      dof_tr->InvTransformPrimal(elfun);
   }

   int ndof = mesh_el.GetDof();
   int el_ndof = el.GetDof();
   int dim = el.GetDim();
   int dimc = (dim == 3) ? 3 : 1;
   elvect.SetSize(ndof * dimc);
   elvect = 0.0;
#ifdef MFEM_THREAD_SAFE
   DenseMatrix curlshape(ndof, dimc), curlshape_dFt(ndof, dimc), M;
   Vector b_vec(dimc);
#else
   curlshape.SetSize(el_ndof, dimc);
   curlshape_dFt.SetSize(el_ndof, dimc);
   b_vec.SetSize(dimc);
#endif
   DenseMatrix PointMat_bar(dimc, ndof);
   DenseMatrix PointMat_bar_2(dimc, ndof);
   Vector b_hat(dimc);

   // cast the ElementTransformation
   auto &isotrans = dynamic_cast<IsoparametricTransformation &>(*trans);

   for (int i = 0; i < ir->GetNPoints(); ++i)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      trans->SetIntPoint(&ip);
      if (dim == 3)
      {
         el.CalcCurlShape(ip, curlshape);
         MultABt(curlshape, trans->Jacobian(), curlshape_dFt);
      }
      else
      {
         el.CalcCurlShape(ip, curlshape_dFt);
      }
      b_vec = 0.0;
      curlshape_dFt.AddMultTranspose(elfun, b_vec);
      const double b_mag = b_vec.Norml2() / isotrans.Weight();
      const double nu_val = nu->Eval(*trans, ip, b_mag);

      // reverse pass

      PointMat_bar = 0.0;
      b_hat = 0.0;
      curlshape.AddMultTranspose(elfun, b_hat);

      const double nu_deriv = nu->EvalStateDeriv(*trans, ip, b_mag);

      DenseMatrix dBdJ(b_hat.Size(), b_vec.Size());
      dBdJ = 0.0;
      AddMult_a_VWt(
          1.0 / (b_vec.Norml2() * isotrans.Weight()), b_vec, b_hat, dBdJ);

      isotrans.WeightRevDiff(PointMat_bar);
      PointMat_bar *= -b_vec.Norml2() / pow(isotrans.Weight(), 2);

      isotrans.JacobianRevDiff(dBdJ, PointMat_bar);

      PointMat_bar *= nu_deriv / isotrans.Weight();

      PointMat_bar_2 = 0.0;
      isotrans.WeightRevDiff(PointMat_bar_2);
      PointMat_bar_2 *= -nu_val / pow(isotrans.Weight(), 2);

      for (int j = 0; j < ndof; ++j)
      {
         for (int d = 0; d < dimc; ++d)
         {
            elvect(d * ndof + j) +=
                ip.weight * (PointMat_bar(d, j) + PointMat_bar_2(d, j));
         }
      }
   }
}

void ThermalSensIntegrator::AssembleRHSElementVect(
    const FiniteElement &nd_el,
    ElementTransformation &nd_trans,
    Vector &elvect)
{
   /// get the proper element, transformation, and adjoint vector
   int element = nd_trans.ElementNo;
   const auto &el = *adjoint->FESpace()->GetFE(element);
   auto &trans = *adjoint->FESpace()->GetElementTransformation(element);

   Array<int> vdofs;
   adjoint->FESpace()->GetElementVDofs(element, vdofs);
   Vector psi;
   adjoint->GetSubVector(vdofs, psi);

   const IntegrationRule *ir =
       &IntRules.Get(el.GetGeomType(), oa * el.GetOrder() + ob);

   int h1_dof = el.GetDof();
   shape.SetSize(h1_dof);
   int nd_dof = nd_el.GetDof();
   elvect.SetSize(nd_dof);
   elvect = 0.0;

   Vector V(nd_dof);

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);
      el.CalcShape(ip, shape);

      double Q_bar = trans.Weight() * (psi * shape);  // d(psi^T R)/dQ
      Q.Eval(V, trans, ip);                           // evaluate dQ/dA
      add(elvect, ip.weight * Q_bar, V, elvect);
   }
}

// void setInputs(DCLossFunctionalIntegrator &integ, const MachInputs
// &inputs)
// {
//    setValueFromInputs(inputs, "rms_current", integ.rms_current);
// }

double DCLossFunctionalIntegrator::GetElementEnergy(
    const FiniteElement &el,
    ElementTransformation &trans,
    const Vector &elfun)
{
   // Obtain correct element, DOFs, etc for temperature field
   const int element = trans.ElementNo;

   const FiniteElement *temp_el=nullptr;
   if (temperature_field != nullptr)
   {
      temp_el = temperature_field->FESpace()->GetFE(element);

      auto *dof_tr = temperature_field->FESpace()->GetElementVDofs(element, vdofs);
      temperature_field->GetSubVector(vdofs, temp_elfun);
      if (dof_tr != nullptr)
      {
         dof_tr->InvTransformPrimal(temp_elfun);
      }
      
      int ndof = temp_el->GetDof();
      shape.SetSize(ndof);
      
   }

   //Should be fine to leave el as is in this scope (rather than replace with temp_el)
   const IntegrationRule *ir = IntRule;
   if (ir == nullptr)
   {
      int order = [&]()
      {
         if (el.Space() == FunctionSpace::Pk)
         {
            return 2 * el.GetOrder() - 2;
         }
         else
         {
            return 2 * el.GetOrder();
         }
      }();

      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   double fun = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {      
      const IntegrationPoint &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip); 
      const double trans_weight = trans.Weight();
      const double w = ip.weight * trans_weight;

      double temperature;

      if (temperature_field != nullptr)
      {
         temp_el->CalcPhysShape(trans, shape);
         temperature = shape * temp_elfun; //Take dot product between shape and elfun to get the value at the integration point
      }
      else
      {
         ///TODO: Change default value of 100 if needed (be consistent throughout)
         temperature = 100; 
      }

      const double sigma_v = sigma.Eval(trans, ip, temperature);
      // const double sigma_v = sigma.Eval(trans, ip);
      fun += w / sigma_v;
   
   }
   return fun;
}

void DCLossFunctionalIntegratorMeshSens::AssembleRHSElementVect(
    const mfem::FiniteElement &mesh_el,
    mfem::ElementTransformation &mesh_trans,
    mfem::Vector &mesh_coords_bar)
{
   const int element = mesh_trans.ElementNo;
   const auto &el = *state.FESpace()->GetFE(element);
   auto &trans = *state.FESpace()->GetElementTransformation(element);

   const int mesh_ndof = mesh_el.GetDof();
   const int space_dim = mesh_trans.GetSpaceDim();

   PointMat_bar.SetSize(space_dim, mesh_ndof);

   // cast the ElementTransformation
   auto &isotrans = dynamic_cast<IsoparametricTransformation &>(trans);

   const IntegrationRule *ir = IntRule;
   if (ir == nullptr)
   {
      int order = [&]()
      {
         if (el.Space() == FunctionSpace::Pk)
         {
            return 2 * el.GetOrder() - 2;
         }
         else
         {
            return 2 * el.GetOrder();
         }
      }();

      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   auto &sigma = integ.sigma;
   auto *temperature_field = integ.temperature_field;

   // Obtain correct element, DOFs, etc for temperature field
   // Logic from DCLossFunctionalIntegrator
   const FiniteElement *temp_el=nullptr;
   if (temperature_field != nullptr)
   {
      
      temp_el = temperature_field->FESpace()->GetFE(element);

      auto *dof_tr = temperature_field->FESpace()->GetElementVDofs(element, vdofs);
      temperature_field->GetSubVector(vdofs, temp_elfun);
      if (dof_tr != nullptr)
      {
         dof_tr->InvTransformPrimal(temp_elfun);
      }
      
      int ndof = temp_el->GetDof();
      shape.SetSize(ndof);
   }

   mesh_coords_bar.SetSize(mesh_ndof * space_dim);
   mesh_coords_bar = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const auto &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);

      double trans_weight = trans.Weight();
      double w = ip.weight * trans_weight;

      // Logic from DCLossFunctionalIntegrator
      double temperature;
      if (temperature_field != nullptr)
      {
         temp_el->CalcPhysShape(trans, shape); // Alternative to CalcShape, used by ACLossFunctionalIntegrator. Difference between CalcPhysShape and CalcShape?
         temperature = shape * temp_elfun; //Take dot product between shape and elfun to get the value at the integration point
      }
      else
      {
         ///TODO: Change default value of 100 if needed (be consistent throughout)
         temperature = 100; 
      }

      const double sigma_v = sigma.Eval(trans, ip, temperature);
      // const double sigma_v = sigma.Eval(trans, ip);
      // fun += w / sigma_v;

      /// Start reverse pass...
      double fun_bar = 1.0;

      /// fun += w / sigma_v;
      double w_bar = fun_bar / sigma_v;
      double sigma_v_bar = -fun_bar * w / pow(sigma_v, 2);

      // std::cout << "sigma_v_bar: " << sigma_v_bar << "\n";
      // std::cout << "fun_bar: " << fun_bar << " w: " << w << " sigma_v: " <<
      // sigma_v << "\n";

      /// const double sigma_v = sigma.Eval(trans, ip);
      PointMat_bar = 0.0;
      ///TODO: Stick with default EvalRevDiff or adapt?
      sigma.EvalRevDiff(sigma_v_bar, trans, ip, PointMat_bar);

      /// double w = ip.weight * trans_weight;
      double trans_weight_bar = w_bar * ip.weight;

      /// double trans_weight = trans.Weight();
      isotrans.WeightRevDiff(trans_weight_bar, PointMat_bar);

      /// code to insert PointMat_bar into mesh_coords_bar;
      for (int j = 0; j < mesh_ndof; ++j)
      {
         for (int d = 0; d < space_dim; ++d)
         {
            mesh_coords_bar(d * mesh_ndof + j) += PointMat_bar(d, j);
         }
      }
   }
}

void setInputs(DCLossFunctionalDistributionIntegrator &integ,
               const MachInputs &inputs)
{
   setValueFromInputs(inputs, "wire_length", integ.wire_length);
   setValueFromInputs(inputs, "rms_current", integ.rms_current);
   setValueFromInputs(inputs, "strand_radius", integ.strand_radius);
   setValueFromInputs(inputs, "strands_in_hand", integ.strands_in_hand);
}

void DCLossFunctionalDistributionIntegrator::AssembleRHSElementVect(
    const mfem::FiniteElement &el,
    mfem::ElementTransformation &trans,
    mfem::Vector &elvect)
{
   // Obtain correct element, DOFs, etc for temperature field
   // Logic from DCLossFunctionalIntegrator
   const int element = trans.ElementNo;

   const FiniteElement *temp_el=nullptr;
   if (temperature_field != nullptr)
   {
      temp_el = temperature_field->FESpace()->GetFE(element);

      auto *dof_tr = temperature_field->FESpace()->GetElementVDofs(element, vdofs);
      temperature_field->GetSubVector(vdofs, temp_elfun);
      if (dof_tr != nullptr)
      {
         dof_tr->InvTransformPrimal(temp_elfun);
      }
      
      int ndof = temp_el->GetDof();
      shape.SetSize(ndof);
      
   }

   const auto *ir = IntRule;
   if (ir == nullptr)
   {
      int order = [&]()
      {
         if (el.Space() == FunctionSpace::Pk)
         {
            return 2 * el.GetOrder() - 1;
         }
         else
         {
            return 2 * el.GetOrder();
         }
      }();

      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   elvect = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      // Logic from DCLossFunctionalIntegrator
      const IntegrationPoint &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip); 
      const double trans_weight = trans.Weight();
      const double w = ip.weight * trans_weight;

      double temperature;

      if (temperature_field != nullptr)
      {
         temp_el->CalcPhysShape(trans, shape); // Alternative to CalcShape, used by ACLossFunctionalIntegrator. Difference between CalcPhysShape and CalcShape?
         temperature = shape * temp_elfun; //Take dot product between shape and elfun to get the value at the integration point
      }
      else
      {
         ///TODO: Change default value of 100 if needed (be consistent throughout)
         temperature = 100; 
      }

      const double sigma_v = sigma.Eval(trans, ip, temperature);
      // const double sigma_v = sigma.Eval(trans, ip);

      double strand_area = M_PI * pow(strand_radius, 2);
      double R = wire_length / (strand_area * strands_in_hand * sigma_v);

      double loss = pow(rms_current, 2) * R;
      // not sure about this... but it matches MotorCAD's values
      loss *= sqrt(2);

      elvect.Add(loss * w, shape);
   }
   ///TODO: Logic is up to date now. Need to finish the implementation and then test
}

double ACLossFunctionalIntegrator::GetElementEnergy(
    const FiniteElement &el,
    ElementTransformation &trans,
    const Vector &elfun)
{
   int ndof = el.GetDof(); // number of degrees of freedom
   // std::cout << "el.GetDof() = " << ndof << "\n";
   shape.SetSize(ndof);

   // Obtain correct element, DOFs, etc for temperature field
   // Logic from DCLossFunctionalIntegrator
   const int element = trans.ElementNo;
   // std::cout << "element = " << element << "\n";   

   const FiniteElement *temp_el=nullptr;
   if (temperature_field != nullptr)
   {
      temp_el = temperature_field->FESpace()->GetFE(element);

      auto *dof_tr = temperature_field->FESpace()->GetElementVDofs(element, vdofs);
      temperature_field->GetSubVector(vdofs, temp_elfun);
      if (dof_tr != nullptr)
      {
         dof_tr->InvTransformPrimal(temp_elfun);
      }
      
      int ndof = temp_el->GetDof();
      // std::cout << "temp_el->GetDof() = " << ndof << "\n";
      temp_shape.SetSize(ndof);
      
   }
   
   ///TODO: Perhaps enforce higher order integration rules to pass the ACLossFunctionalIntegrator test in test_electromag_integ.cpp
   const IntegrationRule *ir = IntRule;
   if (ir == nullptr)
   {
      int order = [&]()
      {
         if (el.Space() == FunctionSpace::Pk)
         {
            return 2 * el.GetOrder() - 2;
            // return 2 * el.GetOrder() - 1;
            // return 10;
         }
         else
         {
            return 2 * el.GetOrder();
         }
      }();

      // std::cout << "order = " << order << "\n";
      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   // std::cout << "ir->GetNPoints() = " << ir->GetNPoints() << "\n";

   double fun = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);

      /// holds quadrature weight
      double trans_weight = trans.Weight();
      const double w = ip.weight * trans_weight;

      el.CalcPhysShape(trans, shape);
      for (int j = 0; j < ndof; j++)
      {
         if (el.GetOrder() == 1)
         {
            // std::cout << "shape[" << j << "]= " << shape[j] << "\n";
            // std::cout << "elfun[" << j << "]= " << elfun[j] << "\n";
         }         
      }
      const auto b_mag = shape * elfun;

      // Logic from DCLossFunctionalIntegrator
      double temperature;

      if (temperature_field != nullptr)
      {
         temp_el->CalcPhysShape(trans, temp_shape);
         for (int j = 0; j < ndof; j++)
         {
            if (el.GetOrder() == 1)
            {
               // std::cout << "temp_shape[" << j << "]= " << temp_shape[j] << "\n";
               // std::cout << "temp_elfun[" << j << "]= " << temp_elfun[j] << "\n";
            }
         }
         temperature = temp_shape * temp_elfun; //Take dot product between shape and elfun to get the value at the integration point
      }
      else
      {
         ///TODO: Change default value of 100 if needed (be consistent throughout)
         temperature = 100; 
      }

      const auto sigma_val = sigma.Eval(trans, ip, temperature);
      // const auto sigma_val = sigma.Eval(trans, ip);

      ///TODO: Comment out or remove once finish debugging
      if (el.GetOrder() ==1)
      {
         // std::cout << "trans_weight = " << trans_weight << "\n";
         // std::cout << "ip.weight = " << ip.weight << "\n";
         // std::cout << "w = ip.weight * trans_weight = " << w << "\n";
         // std::cout << "b_mag = " << b_mag << "\n";
         // std::cout << "temperature = " << temperature << "\n";
         // std::cout << "sigma_val = " << sigma_val << "\n";
      }

      const auto loss = sigma_val * pow(b_mag, 2);
      fun += loss * w;
   }
   return fun;
}

// void ACLossFunctionalIntegrator::AssembleElementVector(const FiniteElement
// &el,
//                                              ElementTransformation &trans,
//                                              const Vector &elfun,
//                                              Vector &elfun_bar)
// {
//    /// number of degrees of freedom
//    int ndof = el.GetDof();

// #ifdef MFEM_THREAD_SAFE
//    Vector shape(ndof);
// #else
//    shape.SetSize(ndof);
// #endif

//    const IntegrationRule *ir = IntRule;
//    if (ir == nullptr)
//    {
//       int order = [&]()
//       {
//          if (el.Space() == FunctionSpace::Pk)
//          {
//             return 2 * el.GetOrder() - 2;
//          }
//          else
//          {
//             return 2 * el.GetOrder();
//          }
//       }();

//       ir = &IntRules.Get(el.GetGeomType(), order);
//    }

//    elfun_bar.SetSize(ndof);
//    elfun_bar = 0.0;
//    for (int i = 0; i < ir->GetNPoints(); i++)
//    {
//       const IntegrationPoint &ip = ir->IntPoint(i);
//       trans.SetIntPoint(&ip);

//       /// holds quadrature weight
//       double trans_weight = trans.Weight();
//       const double w = ip.weight * trans_weight;

//       el.CalcPhysShape(trans, shape);
//       const auto b_mag = shape * elfun;

//       const auto sigma_v = sigma.Eval(trans, ip);

//       const auto loss = sigma_v * pow(b_mag, 2);
//       // fun += loss * w;

//       /// Start reverse pass...
//       double fun_bar = 1.0;

//       /// fun += loss * w;
//       double loss_bar = fun_bar * w;

//       /// const double loss = sigma_v * pow(b_mag, 2);
//       double b_mag_bar = loss_bar * sigma_v * 2 * b_mag;

//       /// const double b_mag = shape * elfun;
//       elfun_bar.Add(b_mag_bar, shape);
//    }
// }

void ACLossFunctionalIntegratorMeshSens::AssembleRHSElementVect(
    const mfem::FiniteElement &mesh_el,
    mfem::ElementTransformation &mesh_trans,
    mfem::Vector &mesh_coords_bar)
{
   const int element = mesh_trans.ElementNo;
   const auto &el = *peak_flux.FESpace()->GetFE(element);
   auto &trans = *peak_flux.FESpace()->GetElementTransformation(element);

   const int ndof = el.GetDof();
   const int mesh_ndof = mesh_el.GetDof();
   const int space_dim = mesh_trans.GetSpaceDim();

   auto *dof_tr = peak_flux.FESpace()->GetElementVDofs(element, vdofs);
   peak_flux.GetSubVector(vdofs, elfun);
   if (dof_tr != nullptr)
   {
      dof_tr->InvTransformPrimal(elfun);
   }

// #ifdef MFEM_THREAD_SAFE
//    mfem::Vector shape;
//    mfem::Vector shape_bar;
// #else
//    auto &shape = integ.shape;
// #endif

   shape.SetSize(ndof);
   shape_bar.SetSize(ndof);
   PointMat_bar.SetSize(space_dim, mesh_ndof);

   // cast the ElementTransformation
   auto &isotrans = dynamic_cast<IsoparametricTransformation &>(trans);

   const IntegrationRule *ir = IntRule;
   if (ir == nullptr)
   {
      int order = [&]()
      {
         if (el.Space() == FunctionSpace::Pk)
         {
            return 2 * el.GetOrder() - 2;
         }
         else
         {
            return 2 * el.GetOrder();
         }
      }();

      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   auto &sigma = integ.sigma;
   auto *temperature_field = integ.temperature_field;

   // Obtain correct element, DOFs, etc for temperature field
   // Logic from DCLossFunctionalIntegrator
   const FiniteElement *temp_el=nullptr;
   if (temperature_field != nullptr)
   {
      
      temp_el = temperature_field->FESpace()->GetFE(element);

      auto *dof_tr = temperature_field->FESpace()->GetElementVDofs(element, vdofs);
      temperature_field->GetSubVector(vdofs, temp_elfun);
      if (dof_tr != nullptr)
      {
         dof_tr->InvTransformPrimal(temp_elfun);
      }
      
      int ndof = temp_el->GetDof();
      temp_shape.SetSize(ndof);
   }

   mesh_coords_bar.SetSize(mesh_ndof * space_dim);
   mesh_coords_bar = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const auto &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);

      double trans_weight = trans.Weight();
      double w = ip.weight * trans_weight;

      el.CalcPhysShape(trans, shape);
      const double b_mag = shape * elfun;

      // Logic from DCLossFunctionalIntegrator
      double temperature;
      if (temperature_field != nullptr)
      {
         temp_el->CalcPhysShape(trans, temp_shape); 
         temperature = temp_shape * temp_elfun; //Take dot product between shape and elfun to get the value at the integration point
      }
      else
      {
         ///TODO: Change default value of 100 if needed (be consistent throughout)
         temperature = 100; 
      }

      const double sigma_v = sigma.Eval(trans, ip, temperature);
      // const double sigma_v = sigma.Eval(trans, ip);

      const double loss = sigma_v * pow(b_mag, 2);
      // fun += loss * w;

      ///TODO: Determine if any of the below needs to be changed
      /// Start reverse pass...
      double fun_bar = 1.0;

      /// fun += loss * w;
      double loss_bar = fun_bar * w;
      double w_bar = fun_bar * loss;

      /// const double loss = sigma_v * pow(b_mag, 2);
      double sigma_v_bar = loss_bar * pow(b_mag, 2);
      double b_mag_bar = loss_bar * sigma_v * 2 * b_mag;

      /// const double sigma_v = sigma.Eval(trans, ip);
      PointMat_bar = 0.0;
      ///TODO: Adapt EvalRevDiff if needed
      sigma.EvalRevDiff(sigma_v_bar, trans, ip, PointMat_bar);
      // std::cout << "PointMat_bar(0,0)" << PointMat_bar(0,0) << "\n";
      /// const double b_mag = shape * elfun;
      shape_bar = 0.0;
      shape_bar.Add(b_mag_bar, elfun);
      // std::cout << "shape_bar(0)=" << shape_bar(0) << "\n";

      /// el.CalcPhysShape(trans, shape);
      el.CalcPhysShapeRevDiff(trans, shape_bar, PointMat_bar);

      /// double w = ip.weight * trans_weight;
      double trans_weight_bar = w_bar * ip.weight;

      /// double trans_weight = trans.Weight();
      isotrans.WeightRevDiff(trans_weight_bar, PointMat_bar);

      /// code to insert PointMat_bar into mesh_coords_bar;
      for (int j = 0; j < mesh_ndof; ++j)
      {
         for (int d = 0; d < space_dim; ++d)
         {
            mesh_coords_bar(d * mesh_ndof + j) += PointMat_bar(d, j);
            // std::cout << "PointMat_bar(" << d << "," << j << ")=" << PointMat_bar(d, j) << "\n";
            // std::cout << "mesh_coords_bar(" << d*mesh_ndof+j << ")=" << mesh_coords_bar(d * mesh_ndof + j) << "\n";
         }
      }
   }
}

void ACLossFunctionalIntegratorPeakFluxSens::AssembleRHSElementVect(
    const mfem::FiniteElement &el,
    mfem::ElementTransformation &trans,
    mfem::Vector &elfun_bar)
{
   const int ndof = el.GetDof();

#ifdef MFEM_THREAD_SAFE
   mfem::Vector elfun;
#endif
   const int element = trans.ElementNo;
   auto *dof_tr = peak_flux.FESpace()->GetElementVDofs(element, vdofs);
   peak_flux.GetSubVector(vdofs, elfun);
   if (dof_tr != nullptr)
   {
      dof_tr->InvTransformPrimal(elfun);
   }

#ifdef MFEM_THREAD_SAFE
   mfem::Vector shape;
#else
   auto &shape = integ.shape;
#endif

   shape.SetSize(ndof);

   const IntegrationRule *ir = IntRule;
   if (ir == nullptr)
   {
      int order = [&]()
      {
         if (el.Space() == FunctionSpace::Pk)
         {
            return 2 * el.GetOrder() - 2;
         }
         else
         {
            return 2 * el.GetOrder();
         }
      }();

      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   auto &sigma = integ.sigma;

   elfun_bar.SetSize(ndof);
   elfun_bar = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);

      /// holds quadrature weight
      double trans_weight = trans.Weight();
      const double w = ip.weight * trans_weight;

      el.CalcPhysShape(trans, shape);
      const auto b_mag = shape * elfun;

      const auto sigma_v = sigma.Eval(trans, ip);

      // const auto loss = sigma_v * pow(b_mag, 2);
      // fun += loss * w;

      /// Start reverse pass...
      double fun_bar = 1.0;

      /// fun += loss * w;
      double loss_bar = fun_bar * w;

      /// const double loss = sigma_v * pow(b_mag, 2);
      double b_mag_bar = loss_bar * sigma_v * 2 * b_mag;

      /// const double b_mag = shape * elfun;
      elfun_bar.Add(b_mag_bar, shape);
   }
}

void setInputs(ACLossFunctionalDistributionIntegrator &integ,
               const MachInputs &inputs)
{
   setValueFromInputs(inputs, "frequency", integ.freq);
   setValueFromInputs(inputs, "strand_radius", integ.radius);
   setValueFromInputs(inputs, "stack_length", integ.stack_length);
   setValueFromInputs(inputs, "strands_in_hand", integ.strands_in_hand);
   setValueFromInputs(inputs, "num_turns", integ.num_turns);
   setValueFromInputs(inputs, "num_slots", integ.num_slots);
}

void ACLossFunctionalDistributionIntegrator::AssembleRHSElementVect(
    const mfem::FiniteElement &el,
    mfem::ElementTransformation &trans,
    mfem::Vector &elvect)
{
   int ndof = el.GetDof();

   const int element = trans.ElementNo;
   const auto &flux_el = *peak_flux.FESpace()->GetFE(element);
   auto &flux_trans = *peak_flux.FESpace()->GetElementTransformation(element);
   const int flux_ndof = flux_el.GetDof();

#ifdef MFEM_THREAD_SAFE
   mfem::Array<int> vdofs;
   mfem::Vector elfun;
#endif

   auto *dof_tr = peak_flux.FESpace()->GetElementVDofs(element, vdofs);
   peak_flux.GetSubVector(vdofs, elfun);
   if (dof_tr != nullptr)
   {
      dof_tr->InvTransformPrimal(elfun);
   }

#ifdef MFEM_THREAD_SAFE
   mfem::Vector shape;
   mfem::Vector flux_shape;
#endif
   shape.SetSize(ndof);
   flux_shape.SetSize(flux_ndof);
   elvect.SetSize(ndof);

   // Obtain correct element, DOFs, etc for temperature field
   // Logic from DCLossFunctionalIntegrator
   const FiniteElement *temp_el=nullptr;
   if (temperature_field != nullptr)
   {
      temp_el = temperature_field->FESpace()->GetFE(element);

      auto *dof_tr = temperature_field->FESpace()->GetElementVDofs(element, vdofs);
      temperature_field->GetSubVector(vdofs, temp_elfun);
      if (dof_tr != nullptr)
      {
         dof_tr->InvTransformPrimal(temp_elfun);
      }
      
      int ndof = temp_el->GetDof();
      shape.SetSize(ndof);
      
   }

   const auto *ir = IntRule;
   if (ir == nullptr)
   {
      int order = [&]()
      {
         if (el.Space() == FunctionSpace::Pk)
         {
            return 2 * el.GetOrder() - 1;
         }
         else
         {
            return 2 * el.GetOrder();
         }
      }();

      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   elvect = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);

      el.CalcPhysShape(trans, shape);

      flux_el.CalcPhysShape(flux_trans, flux_shape);
      const auto b_mag = flux_shape * elfun;

      /// holds quadrature weight
      const double w = ip.weight * trans.Weight();

      double temperature;

      if (temperature_field != nullptr)
      {
         temp_el->CalcPhysShape(trans, shape); // Alternative to CalcShape, used by ACLossFunctionalIntegrator. Difference between CalcPhysShape and CalcShape?
         temperature = shape * temp_elfun; //Take dot product between shape and elfun to get the value at the integration point
      }
      else
      {
         ///TODO: Change default value of 100 if needed (be consistent throughout)
         temperature = 100; 
      }

      const double sigma_v = sigma.Eval(trans, ip, temperature);
      // const double sigma_v = sigma.Eval(trans, ip);

      double loss = stack_length * M_PI * pow(radius, 4) *
                    pow(2 * M_PI * freq * b_mag, 2) * sigma_v / 32.0;
      loss *= 2 * strands_in_hand * num_turns * num_slots;

      elvect.Add(loss * w, shape);
   }
   ///TODO: Logic is up to date now. Need to finish the implementation and then test
}

/// HybridACLossFunctionalIntegrator is a Dead class. Not putting any more time into
// void setInputs(HybridACLossFunctionalIntegrator &integ,
//                const MachInputs &inputs)
// {
//    // auto it = inputs.find("diam");
//    // if (it != inputs.end())
//    // {
//    //    integ.diam = it->second.getValue();
//    // }
//    // it = inputs.find("diam");
//    // if (it != inputs.end())
//    // {
//    //    integ.freq = it->second.getValue();
//    // }
//    // it = inputs.find("fill-factor");
//    // if (it != inputs.end())
//    // {
//    //    integ.fill_factor = it->second.getValue();
//    // }
//    setValueFromInputs(inputs, "diam", integ.diam);
//    setValueFromInputs(inputs, "freq", integ.freq);
//    setValueFromInputs(inputs, "fill-factor", integ.fill_factor);
// }

/// HybridACLossFunctionalIntegrator is a Dead class. Not putting any more time into
/// In cpp and hpp, Have only the immediate below HybridACLossFunctionalIntegrator uncommented if want mfem::Coefficient logic for test_acloss_functional
// double HybridACLossFunctionalIntegrator::GetElementEnergy(
//     const FiniteElement &el,
//     ElementTransformation &trans,
//     const Vector &elfun)
// {
//    /// number of degrees of freedom
//    int ndof = el.GetDof();
//    int dim = el.GetDim();

//    /// I believe this takes advantage of a 2D problem not having
//    /// a properly defined curl? Need more investigation
//    int dimc = (dim == 3) ? 3 : 1;

// #ifdef MFEM_THREAD_SAFE
//    DenseMatrix curlshape(ndof, dimc), curlshape_dFt(ndof, dimc);
//    Vector b_vec(dimc);
// #else
//    curlshape.SetSize(ndof, dimc);
//    curlshape_dFt.SetSize(ndof, dimc);
//    b_vec.SetSize(dimc);
// #endif

//    const IntegrationRule *ir = IntRule;
//    if (ir == nullptr)
//    {
//       int order = [&]()
//       {
//          if (el.Space() == FunctionSpace::Pk)
//          {
//             return 2 * el.GetOrder() - 2;
//          }
//          else
//          {
//             return 2 * el.GetOrder();
//          }
//       }();

//       ir = &IntRules.Get(el.GetGeomType(), order);
//    }

//    double fun = 0.0;

//    for (int i = 0; i < ir->GetNPoints(); i++)
//    {
//       b_vec = 0.0;
//       const IntegrationPoint &ip = ir->IntPoint(i);

//       trans.SetIntPoint(&ip);

//       auto w = ip.weight * trans.Weight();
//       if (dim == 3)
//       {
//          el.CalcCurlShape(ip, curlshape);
//          MultABt(curlshape, trans.Jacobian(), curlshape_dFt);
//       }
//       else
//       {
//          el.CalcCurlShape(ip, curlshape_dFt);
//       }

//       curlshape_dFt.AddMultTranspose(elfun, b_vec);
//       const double b_mag = b_vec.Norml2() / trans.Weight();
//       ///TODO: double temperature = (after incorporate temperature field)
//       // double temperature = 100; /// temporary. Will remove after incorporate temperature field
//       // const double sigma_val = sigma.Eval(trans, ip, temperature);
//       const double sigma_val = sigma.Eval(trans, ip);

//       const double loss = std::pow(diam, 2) * sigma_val *
//                           std::pow(2 * M_PI * freq * b_mag, 2) / 128.0;
//       fun += loss * fill_factor * w;
//    }
//    return fun;
// }

/// HybridACLossFunctionalIntegrator is a Dead class. Not putting any more time into
/// In cpp and hpp, Have only the immediate below HybridACLossFunctionalIntegrator uncommented for StateCoefficient logic for test_acloss_functional (this will be the one that remains)
// double HybridACLossFunctionalIntegrator::GetElementEnergy(
//     const FiniteElement &el,
//     ElementTransformation &trans,
//     const Vector &elfun)
// {
//    /// number of degrees of freedom
//    int ndof = el.GetDof();
//    int dim = el.GetDim();

//    /// I believe this takes advantage of a 2D problem not having
//    /// a properly defined curl? Need more investigation
//    int dimc = (dim == 3) ? 3 : 1;

// // #ifdef MFEM_THREAD_SAFE
//    DenseMatrix curlshape(ndof, dimc), curlshape_dFt(ndof, dimc);
//    Vector b_vec(dimc);
// // #else
// //    curlshape.SetSize(ndof, dimc);
// //    curlshape_dFt.SetSize(ndof, dimc);
// //    b_vec.SetSize(dimc);
// // #endif

//    // Obtain correct element, DOFs, etc for temperature field
//    // Logic from DCLossFunctionalIntegrator
//    const int element = trans.ElementNo;
//    const FiniteElement *temp_el=nullptr;
//    if (temperature_field != nullptr)
//    {
//       temp_el = temperature_field->FESpace()->GetFE(element);

//       auto *dof_tr = temperature_field->FESpace()->GetElementVDofs(element, vdofs);
//       temperature_field->GetSubVector(vdofs, temp_elfun);
//       if (dof_tr != nullptr)
//       {
//          dof_tr->InvTransformPrimal(temp_elfun);
//       }
      
//       int ndof = temp_el->GetDof();
//       shape.SetSize(ndof);
      
//    }

//    const IntegrationRule *ir = IntRule;
//    if (ir == nullptr)
//    {
//       int order = [&]()
//       {
//          if (el.Space() == FunctionSpace::Pk)
//          {
//             return 2 * el.GetOrder() - 2;
//          }
//          else
//          {
//             return 2 * el.GetOrder();
//          }
//       }();

//       ir = &IntRules.Get(el.GetGeomType(), order);
//    }

//    double fun = 0.0;

//    for (int i = 0; i < ir->GetNPoints(); i++)
//    {
//       b_vec = 0.0;
//       const IntegrationPoint &ip = ir->IntPoint(i);

//       trans.SetIntPoint(&ip);

//       auto w = ip.weight * trans.Weight();
//       if (dim == 3)
//       {
//          el.CalcCurlShape(ip, curlshape);
//          MultABt(curlshape, trans.Jacobian(), curlshape_dFt);
//       }
//       else
//       {
//          el.CalcCurlShape(ip, curlshape_dFt);
//       }

//       curlshape_dFt.AddMultTranspose(elfun, b_vec);
//       const double b_mag = b_vec.Norml2() / trans.Weight();
   
//       double temperature;

//       if (temperature_field != nullptr)
//       {
//          ///TODO: Edit the logic here. Had to make temperature a vector coefficient in test_acloss_functional.bin, and CalcShape cannot work with vector, only scalar
//          temp_el->CalcPhysShape(trans, shape); // Alternative to CalcShape, used by ACLossFunctionalIntegrator. Difference between CalcPhysShape and CalcShape?
//          temperature = shape * temp_elfun; //Take dot product between shape and elfun to get the value at the integration point
//       }
//       else
//       {
//          ///TODO: Change default value of 100 if needed (be consistent throughout)
//          temperature = 100; 
//       }

//       const double sigma_val = sigma.Eval(trans, ip, temperature);
//       // const double sigma_val = sigma.Eval(trans, ip);

//       const double loss = std::pow(diam, 2) * sigma_val *
//                           std::pow(2 * M_PI * freq * b_mag, 2) / 128.0;
//       fun += loss * fill_factor * w;
//    }
//    return fun;
// }

double ForceIntegrator3::GetElementEnergy(const FiniteElement &el,
                                          ElementTransformation &trans,
                                          const Vector &elfun)
{
   if (attrs.count(trans.Attribute) == 1)
   {
      return 0.0;
   }
   /// get the proper element, transformation, and v vector
#ifdef MFEM_THREAD_SAFE
   Array<int> vdofs;
   Vector vfun;
#endif
   int element = trans.ElementNo;
   const auto &v_el = *v.FESpace()->GetFE(element);
   v.FESpace()->GetElementVDofs(element, vdofs);
   v.GetSubVector(vdofs, vfun);
   DenseMatrix dXds(vfun.GetData(), v_el.GetDof(), v_el.GetDim());
   if (vfun.Normlinf() < 1e-14)
   {
      return 0.0;
   }
   /// number of degrees of freedom
   int ndof = el.GetDof();
   int dim = el.GetDim();
   int space_dim = trans.GetSpaceDim();
   int curl_dim = space_dim;

#ifdef MFEM_THREAD_SAFE
   DenseMatrix dshape;
   DenseMatrix curlshape;
   DenseMatrix curlshape_dFt;
   DenseMatrix dBdX;
#endif
   dshape.SetSize(v_el.GetDof(), v_el.GetDim());
   curlshape.SetSize(ndof, curl_dim);
   curlshape_dFt.SetSize(ndof, curl_dim);
   dBdX.SetSize(v_el.GetDim(), v_el.GetDof());

   // DenseMatrix dBdX(v_el.GetDim(), v_el.GetDof());
   // PointMat_bar.SetSize(space_dim, v_el.GetDof());

   double b_vec_buffer[3] = {};
   Vector b_vec(b_vec_buffer, curl_dim);

   // double b_vec_bar_buffer[3] = {};
   // Vector b_vec_bar(b_vec_bar_buffer, curl_dim);

   // cast the ElementTransformation
   auto &isotrans = dynamic_cast<IsoparametricTransformation &>(trans);

   const IntegrationRule *ir = IntRule;
   if (ir == nullptr)
   {
      int order = [&]()
      {
         if (el.Space() == FunctionSpace::Pk)
         {
            return 2 * el.GetOrder() - 2;
         }
         else
         {
            return 2 * el.GetOrder();
         }
      }();

      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   double fun = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);

      const double trans_weight = trans.Weight();
      /// holds quadrature weight
      double w = ip.weight * trans_weight;

      if (dim == 3)
      {
         el.CalcCurlShape(ip, curlshape);
         MultABt(curlshape, trans.Jacobian(), curlshape_dFt);
      }
      else  // Dealing with scalar H1 field representing Az
      {
         /// Not exactly the curl matrix, but since we just want the magnitude
         /// of the curl it's okay
         el.CalcDShape(ip, curlshape);
         Mult(curlshape, trans.AdjugateJacobian(), curlshape_dFt);
      }

      // b_vec = 0.0;
      curlshape_dFt.MultTranspose(elfun, b_vec);
      const double b_vec_norm = b_vec.Norml2();
      const double b_mag = b_vec_norm / trans_weight;

      /// compute d(b_mag)/dJ
      double db_magdJ_buffer[9] = {};
      DenseMatrix db_magdJ(db_magdJ_buffer, space_dim, space_dim);
      db_magdJ = 0.0;
      if (dim == 3)
      {
         double b_hat_buffer[3] = {};
         Vector b_hat(b_hat_buffer, curl_dim);
         b_hat = 0.0;

         curlshape.AddMultTranspose(elfun, b_hat);
         double BB_hatT_buffer[9] = {};
         DenseMatrix BB_hatT(BB_hatT_buffer, curl_dim, curl_dim);
         MultVWt(b_vec, b_hat, BB_hatT);

         db_magdJ.Add(-b_vec_norm / pow(trans_weight, 2),
                      trans.AdjugateJacobian());
         db_magdJ.Transpose();

         db_magdJ.Add(1.0 / (trans_weight * b_vec_norm), BB_hatT);
      }
      else
      {
         double b_adjJT_buffer[3] = {};
         Vector b_adjJT(b_adjJT_buffer, curl_dim);
         trans.AdjugateJacobian().Mult(b_vec, b_adjJT);

         double a = -1 / (b_vec_norm * pow(trans_weight, 2));

         AddMult_a_VWt(a, b_vec, b_adjJT, db_magdJ);
      }

      /// contract d(b_mag)/dJ with dJ/dX
      dBdX = 0.0;
      isotrans.JacobianRevDiff(db_magdJ, dBdX);

      double dBds = 0.0;
      for (int j = 0; j < v_el.GetDof(); ++j)
      {
         for (int k = 0; k < space_dim; ++k)
         {
            dBds += dBdX(k, j) * dXds(j, k);
         }
      }
      const double energy_dot = calcMagneticEnergyDot(trans, ip, nu, b_mag);
      auto force = dBds * energy_dot;

      v_el.CalcDShape(ip, dshape);
      double JinvdJds_buffer[9] = {};
      DenseMatrix JinvdJds(JinvdJds_buffer, space_dim, space_dim);
      double dJds_buffer[9] = {};
      DenseMatrix dJds(dJds_buffer, space_dim, space_dim);
      MultAtB(dXds, dshape, dJds);
      Mult(trans.InverseJacobian(), dJds, JinvdJds);
      double JinvdJdsTrace = JinvdJds.Trace();

      const double energy = calcMagneticEnergy(trans, ip, nu, b_mag);
      double force2 = energy * JinvdJdsTrace;
      fun -= (force + force2) * w;
   }
   return fun;
}

void ForceIntegrator3::AssembleElementVector(const FiniteElement &el,
                                             ElementTransformation &trans,
                                             const Vector &elfun,
                                             Vector &elfun_bar)
{
   /// number of degrees of freedom
   int ndof = el.GetDof();
   int dim = el.GetDim();
   int space_dim = trans.GetSpaceDim();
   int curl_dim = space_dim;

   elfun_bar.SetSize(ndof);
   elfun_bar = 0.0;
   if (attrs.count(trans.Attribute) == 1)
   {
      return;
   }

   /// get the proper element, transformation, and v vector
#ifdef MFEM_THREAD_SAFE
   Array<int> vdofs;
   Vector vfun;
#endif
   int element = trans.ElementNo;
   const auto &v_el = *v.FESpace()->GetFE(element);
   v.FESpace()->GetElementVDofs(element, vdofs);
   v.GetSubVector(vdofs, vfun);
   DenseMatrix dXds(vfun.GetData(), v_el.GetDof(), v_el.GetDim());
   if (vfun.Normlinf() < 1e-14)
   {
      return;
   }

#ifdef MFEM_THREAD_SAFE
   DenseMatrix curlshape;
   DenseMatrix curlshape_dFt;
   DenseMatrix curlshape_dFt_bar;
   DenseMatrix dBdX;
#endif
   dshape.SetSize(v_el.GetDof(), v_el.GetDim());
   curlshape.SetSize(ndof, curl_dim);
   curlshape_dFt.SetSize(ndof, curl_dim);
   dBdX.SetSize(v_el.GetDim(), v_el.GetDof());

   double b_vec_buffer[3] = {};
   Vector b_vec(b_vec_buffer, curl_dim);

   double b_vec_bar_buffer[3] = {};
   Vector b_vec_bar(b_vec_bar_buffer, curl_dim);

   // cast the ElementTransformation
   auto &isotrans = dynamic_cast<IsoparametricTransformation &>(trans);

   const IntegrationRule *ir = IntRule;
   if (ir == nullptr)
   {
      int order = [&]()
      {
         if (el.Space() == FunctionSpace::Pk)
         {
            return 2 * el.GetOrder() - 2;
         }
         else
         {
            return 2 * el.GetOrder();
         }
      }();

      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);

      double trans_weight = trans.Weight();

      /// holds quadrature weight
      const double w = ip.weight * trans_weight;

      if (dim == 3)
      {
         el.CalcCurlShape(ip, curlshape);
         MultABt(curlshape, trans.Jacobian(), curlshape_dFt);
      }
      else  // Dealing with scalar H1 field representing Az
      {
         /// Not exactly the curl matrix, but since we just want the magnitude
         /// of the curl it's okay
         el.CalcDShape(ip, curlshape);
         Mult(curlshape, trans.AdjugateJacobian(), curlshape_dFt);
      }

      b_vec = 0.0;
      curlshape_dFt.AddMultTranspose(elfun, b_vec);
      const double b_vec_norm = b_vec.Norml2();
      const double b_mag = b_vec_norm / trans_weight;

      /// compute d(b_mag)/dJ
      double db_magdJ_buffer[9] = {};
      DenseMatrix db_magdJ(db_magdJ_buffer, space_dim, space_dim);
      db_magdJ = 0.0;
      if (dim == 3)
      {
         double b_hat_buffer[3] = {};
         Vector b_hat(b_hat_buffer, curl_dim);
         b_hat = 0.0;

         curlshape.AddMultTranspose(elfun, b_hat);
         double BB_hatT_buffer[9] = {};
         DenseMatrix BB_hatT(BB_hatT_buffer, curl_dim, curl_dim);
         MultVWt(b_vec, b_hat, BB_hatT);

         db_magdJ.Add(-b_vec_norm / pow(trans_weight, 2),
                      trans.AdjugateJacobian());
         db_magdJ.Transpose();

         db_magdJ.Add(1.0 / (trans_weight * b_vec_norm), BB_hatT);
      }
      else
      {
         double b_adjJT_buffer[3] = {};
         Vector b_adjJT(b_adjJT_buffer, curl_dim);
         trans.AdjugateJacobian().Mult(b_vec, b_adjJT);

         double a = -1 / (b_vec_norm * pow(trans_weight, 2));

         AddMult_a_VWt(a, b_vec, b_adjJT, db_magdJ);
      }

      /// contract d(b_mag)/dJ with dJ/dX
      dBdX = 0.0;
      isotrans.JacobianRevDiff(db_magdJ, dBdX);

      double dBds = 0.0;
      for (int j = 0; j < v_el.GetDof(); ++j)
      {
         for (int k = 0; k < space_dim; ++k)
         {
            dBds += dBdX(k, j) * dXds(j, k);
         }
      }
      const double energy_dot = calcMagneticEnergyDot(trans, ip, nu, b_mag);
      // auto force = dBds * energy_dot;

      v_el.CalcDShape(ip, dshape);
      double JinvdJds_buffer[9] = {};
      DenseMatrix JinvdJds(JinvdJds_buffer, space_dim, space_dim);
      double dJds_buffer[9] = {};
      DenseMatrix dJds(dJds_buffer, space_dim, space_dim);
      MultAtB(dXds, dshape, dJds);
      Mult(trans.InverseJacobian(), dJds, JinvdJds);
      double JinvdJdsTrace = JinvdJds.Trace();

      // const double energy = calcMagneticEnergy(trans, ip, nu, b_mag);
      // double force2 = energy * JinvdJdsTrace;
      // fun -= (force + force2) * w;

      /// start reverse pass
      double fun_bar = 1.0;

      /// fun -= (force + force2) * w;
      double force_bar = 0.0;
      double force2_bar = 0.0;
      force_bar -= fun_bar * w;
      force2_bar -= fun_bar * w;

      /// double force2 = energy * JinvdJdsTrace;
      double energy_bar = force2_bar * JinvdJdsTrace;

      /// const double energy = calcMagneticEnergy(trans, ip, nu, b_mag);
      double b_mag_bar = 0.0;
      b_mag_bar += energy_bar * energy_dot;

      /// auto force = dBds * energy_dot;
      double dBds_bar = force_bar * energy_dot;
      double energy_dot_bar = force_bar * dBds;

      /// double energy_dot = calcMagneticEnergyDot(trans, ip, nu, b_mag);
      auto energy_double_dot =
          calcMagneticEnergyDoubleDot(trans, ip, nu, b_mag);
      b_mag_bar += energy_dot_bar * energy_double_dot;

      DenseMatrix dBdX_bar(v_el.GetDim(), v_el.GetDof());
      dBdX_bar = 0.0;  // same shape as dBdX
      for (int j = 0; j < v_el.GetDof(); ++j)
      {
         for (int k = 0; k < space_dim; ++k)
         {
            /// dBds += dBdX(k, j) * dXds(j, k);
            dBdX_bar(k, j) += dBds_bar * dXds(j, k);
         }
      }

      /// isotrans.JacobianRevDiff(db_magdJ, dBdX);
      /// aka AddMultABt(db_magdJ, dshape, dBdX);
      double db_magdJ_bar_buffer[9] = {};
      DenseMatrix db_magdJ_bar(db_magdJ_bar_buffer, space_dim, space_dim);
      Mult(dBdX_bar, dshape, db_magdJ_bar);
      // db_magdJ_bar = 0.0;
      // v_el.CalcDShape(ip, dshape);
      // AddMult(dBdX_bar, dshape, db_magdJ_bar);

      double b_vec_norm_bar = 0.0;
      // double trans_weight_bar = 0.0;
      if (dim == 3)
      {
         double b_hat_buffer[3] = {};
         Vector b_hat(b_hat_buffer, curl_dim);
         curlshape.MultTranspose(elfun, b_hat);

         double BB_hatT_buffer[9] = {};
         DenseMatrix BB_hatT(BB_hatT_buffer, curl_dim, curl_dim);
         MultVWt(b_vec, b_hat, BB_hatT);

         double BB_hatT_bar_buffer[9] = {};
         DenseMatrix BB_hatT_bar(BB_hatT_bar_buffer, curl_dim, curl_dim);
         BB_hatT_bar = 0.0;

         /// db_magdJ.Add(1.0 / (b_vec_norm * trans_weight), BB_hatT);
         BB_hatT_bar.Add(1.0 / (b_vec_norm * trans_weight), db_magdJ_bar);

         for (int j = 0; j < curl_dim; ++j)
         {
            for (int k = 0; k < curl_dim; ++k)
            {
               b_vec_norm_bar -= db_magdJ_bar(j, k) * BB_hatT(j, k) /
                                 (pow(b_vec_norm, 2) * trans_weight);
               // trans_weight_bar -= db_magdJ_bar(j, k) * BB_hatT(j, k) /
               //                     (b_vec_norm * pow(trans_weight, 2));
            }
         }

         /// db_magdJ.Transpose();
         db_magdJ_bar.Transpose();

         /// db_magdJ.Add(-b_vec_norm / pow(trans_weight, 2),
         ///              trans.AdjugateJacobian());
         for (int j = 0; j < curl_dim; ++j)
         {
            for (int k = 0; k < curl_dim; ++k)
            {
               b_vec_norm_bar -= db_magdJ_bar(j, k) *
                                 trans.AdjugateJacobian()(j, k) /
                                 pow(trans_weight, 2);
               // trans_weight_bar += db_magdJ_bar(j, k) * b_vec_norm *
               //                     trans.AdjugateJacobian()(j, k) /
               //                     pow(trans_weight, 3);
            }
         }

         double b_hat_bar_buffer[3] = {};
         Vector b_hat_bar(b_hat_bar_buffer, curl_dim);

         /// MultVWt(b_vec, b_hat, BB_hatT);
         BB_hatT_bar.Mult(b_hat, b_vec_bar);
         BB_hatT_bar.MultTranspose(b_vec, b_hat_bar);

         /// curlshape.AddMultTranspose(elfun, b_hat);
         curlshape.AddMult(b_hat_bar, elfun_bar);
      }
      else
      {
         double b_adjJT_buffer[3] = {};
         Vector b_adjJT(b_adjJT_buffer, curl_dim);
         trans.AdjugateJacobian().Mult(b_vec, b_adjJT);

         double a = -1 / (b_vec_norm * pow(trans_weight, 2));

         /// AddMult_a_VWt(a, b_vec, b_adjJT, db_magdJ);
         double b_adjJT_var_buffer[3] = {};
         Vector b_adjJT_bar(b_adjJT_var_buffer, curl_dim);

         b_vec_bar = 0.0;
         db_magdJ_bar.AddMult_a(a, b_adjJT, b_vec_bar);
         b_adjJT_bar = 0.0;
         db_magdJ_bar.AddMultTranspose_a(a, b_vec, b_adjJT_bar);
         double a_bar = 0;
         for (int j = 0; j < space_dim; ++j)
         {
            for (int k = 0; k < space_dim; ++k)
            {
               a_bar += db_magdJ_bar(j, k) * db_magdJ(j, k);
            }
         }
         a_bar /= a;

         /// double a = -1 / (b_vec_norm * pow(trans_weight, 2));
         b_vec_norm_bar += a_bar / pow(b_vec_norm * trans_weight, 2);
         // trans_weight_bar = 2 * a_bar / (b_vec_norm * pow(trans_weight, 3));

         /// trans.AdjugateJacobian().Mult(b_vec, b_adjJT);
         trans.AdjugateJacobian().AddMultTranspose(b_adjJT_bar, b_vec_bar);
      }

      /// const double b_mag = b_vec_norm / trans_weight;
      b_vec_norm_bar += b_mag_bar / trans_weight;
      // trans_weight_bar -= b_mag_bar * b_vec_norm / pow(trans_weight, 2);

      /// const double b_vec_norm = b_vec.Norml2();
      add(b_vec_bar, b_vec_norm_bar / b_vec_norm, b_vec, b_vec_bar);

      /// curlshape_dFt.MultTranspose(elfun, b_vec);
      curlshape_dFt.AddMult(b_vec_bar, elfun_bar);
   }
}

void ForceIntegratorMeshSens3::AssembleRHSElementVect(
    const FiniteElement &mesh_el,
    ElementTransformation &mesh_trans,
    Vector &mesh_coords_bar)
{
   const int element = mesh_trans.ElementNo;
   const auto &el = *state.FESpace()->GetFE(element);
   auto &trans = *state.FESpace()->GetElementTransformation(element);

   const int mesh_ndof = mesh_el.GetDof();
   const int ndof = el.GetDof();
   const int dim = el.GetDim();
   const int space_dim = trans.GetSpaceDim();
   const int curl_dim = space_dim;
   mesh_coords_bar.SetSize(mesh_ndof * space_dim);
   mesh_coords_bar = 0.0;

   auto &attrs = force_integ.attrs;
   if (attrs.count(trans.Attribute) == 1)
   {
      return;
   }

   auto &v = force_integ.v;

#ifdef MFEM_THREAD_SAFE
   Array<int> vdofs;
   Vector vfun;
   Vector elfun;
#else
   auto &vdofs = force_integ.vdofs;
   auto &vfun = force_integ.vfun;
#endif

   /// get the proper element, transformation, and v vector
   const auto &v_el = *v.FESpace()->GetFE(element);
   v.FESpace()->GetElementVDofs(element, vdofs);
   v.GetSubVector(vdofs, vfun);
   DenseMatrix dXds(vfun.GetData(), v_el.GetDof(), v_el.GetDim());
   if (vfun.Normlinf() < 1e-14)
   {
      return;
   }

   /// get the proper element, transformation, and state vector
   auto *dof_tr = state.FESpace()->GetElementVDofs(element, vdofs);
   state.GetSubVector(vdofs, elfun);
   if (dof_tr != nullptr)
   {
      dof_tr->InvTransformPrimal(elfun);
   }

#ifdef MFEM_THREAD_SAFE
   DenseMatrix dshape;
   DenseMatrix curlshape;
   DenseMatrix curlshape_dFt;
   DenseMatrix dBdX;
   DenseMatrix PointMat_bar;
#else
   auto &dshape = force_integ.dshape;
   auto &curlshape = force_integ.curlshape;
   auto &curlshape_dFt = force_integ.curlshape_dFt;
   auto &dBdX = force_integ.dBdX;
#endif
   dshape.SetSize(v_el.GetDof(), v_el.GetDim());
   curlshape.SetSize(ndof, curl_dim);
   curlshape_dFt.SetSize(ndof, curl_dim);
   dBdX.SetSize(v_el.GetDim(), v_el.GetDof());
   PointMat_bar.SetSize(space_dim, mesh_ndof);

   DenseMatrix curlshape_dFt_bar;

   double b_vec_buffer[3] = {};
   Vector b_vec(b_vec_buffer, curl_dim);

   double b_vec_bar_buffer[3] = {};
   Vector b_vec_bar(b_vec_bar_buffer, curl_dim);

   // cast the ElementTransformation
   auto &isotrans = dynamic_cast<IsoparametricTransformation &>(trans);

   const IntegrationRule *ir = IntRule;
   if (ir == nullptr)
   {
      int order = [&]()
      {
         if (el.Space() == FunctionSpace::Pk)
         {
            return 2 * el.GetOrder() - 2;
         }
         else
         {
            return 2 * el.GetOrder();
         }
      }();

      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   auto &nu = force_integ.nu;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);

      double trans_weight = trans.Weight();

      /// holds quadrature weight
      const double w = ip.weight * trans_weight;

      if (dim == 3)
      {
         el.CalcCurlShape(ip, curlshape);
         MultABt(curlshape, trans.Jacobian(), curlshape_dFt);
      }
      else  // Dealing with scalar H1 field representing Az
      {
         /// Not exactly the curl matrix, but since we just want the magnitude
         /// of the curl it's okay
         el.CalcDShape(ip, curlshape);
         Mult(curlshape, trans.AdjugateJacobian(), curlshape_dFt);
      }

      b_vec = 0.0;
      curlshape_dFt.AddMultTranspose(elfun, b_vec);
      const double b_vec_norm = b_vec.Norml2();
      const double b_mag = b_vec_norm / trans_weight;

      /// compute d(b_mag)/dJ
      double db_magdJ_buffer[9] = {};
      DenseMatrix db_magdJ(db_magdJ_buffer, space_dim, space_dim);
      db_magdJ = 0.0;
      if (dim == 3)
      {
         double b_hat_buffer[3] = {};
         Vector b_hat(b_hat_buffer, curl_dim);
         b_hat = 0.0;

         curlshape.AddMultTranspose(elfun, b_hat);
         double BB_hatT_buffer[9] = {};
         DenseMatrix BB_hatT(BB_hatT_buffer, curl_dim, curl_dim);
         MultVWt(b_vec, b_hat, BB_hatT);

         db_magdJ.Add(-b_vec_norm / pow(trans_weight, 2),
                      trans.AdjugateJacobian());
         db_magdJ.Transpose();

         db_magdJ.Add(1.0 / (trans_weight * b_vec_norm), BB_hatT);
      }
      else
      {
         double b_adjJT_buffer[3] = {};
         Vector b_adjJT(b_adjJT_buffer, curl_dim);
         trans.AdjugateJacobian().Mult(b_vec, b_adjJT);

         double a = -1 / (b_vec_norm * pow(trans_weight, 2));

         AddMult_a_VWt(a, b_vec, b_adjJT, db_magdJ);
      }

      /// contract d(b_mag)/dJ with dJ/dX
      dBdX = 0.0;
      isotrans.JacobianRevDiff(db_magdJ, dBdX);

      double dBds = 0.0;
      for (int j = 0; j < v_el.GetDof(); ++j)
      {
         for (int k = 0; k < space_dim; ++k)
         {
            dBds += dBdX(k, j) * dXds(j, k);
         }
      }
      const double energy_dot = calcMagneticEnergyDot(trans, ip, nu, b_mag);
      auto force = dBds * energy_dot;

      v_el.CalcDShape(ip, dshape);
      double JinvdJds_buffer[9] = {};
      DenseMatrix JinvdJds(JinvdJds_buffer, space_dim, space_dim);
      double dJds_buffer[9] = {};
      DenseMatrix dJds(dJds_buffer, space_dim, space_dim);
      MultAtB(dXds, dshape, dJds);
      Mult(trans.InverseJacobian(), dJds, JinvdJds);
      double JinvdJdsTrace = JinvdJds.Trace();

      const double energy = calcMagneticEnergy(trans, ip, nu, b_mag);
      double force2 = energy * JinvdJdsTrace;
      // fun -= (force + force2) * w;

      /// start reverse pass
      double fun_bar = 1.0;

      /// fun -= (force + force2) * w;
      double force_bar = 0.0;
      double force2_bar = 0.0;
      double w_bar = 0.0;
      force_bar -= fun_bar * w;
      force2_bar -= fun_bar * w;
      w_bar -= fun_bar * (force + force2);

      /// double force2 = energy * JinvdJdsTrace;
      double energy_bar = force2_bar * JinvdJdsTrace;
      double JinvdJdsTrace_bar = force2_bar * energy;

      /// const double energy = calcMagneticEnergy(trans, ip, nu, b_mag);
      double b_mag_bar = 0.0;
      b_mag_bar += energy_bar * energy_dot;

      /// double JinvdJdsTrace = JinvdJds.Trace();
      double JinvdJds_bar_buffer[9] = {};
      DenseMatrix JinvdJds_bar(JinvdJds_bar_buffer, space_dim, space_dim);
      JinvdJds_bar.Diag(JinvdJdsTrace_bar, space_dim);

      /// Mult(trans.InverseJacobian(), dJds, JinvdJds);
      double dJds_bar_buffer[9] = {};
      DenseMatrix dJds_bar(dJds_bar_buffer, space_dim, space_dim);
      double inv_jac_bar_buffer[9] = {};
      DenseMatrix inv_jac_bar(inv_jac_bar_buffer, space_dim, space_dim);
      MultABt(JinvdJds_bar, dJds, inv_jac_bar);
      MultAtB(trans.InverseJacobian(), JinvdJds_bar, dJds_bar);

      /// Matrix inverse reverse mode rule:
      /// C = A^-1,
      /// A_bar = -C^T * C_bar * C^T
      double scratch_buffer[9] = {};
      DenseMatrix scratch(scratch_buffer, space_dim, space_dim);
      double jac_bar_buffer[9] = {};
      DenseMatrix jac_bar(jac_bar_buffer, space_dim, space_dim);
      jac_bar = 0.0;
      MultAtB(trans.InverseJacobian(), inv_jac_bar, scratch);
      AddMult_a_ABt(-1.0, scratch, trans.InverseJacobian(), jac_bar);

      /// MultAtB(dXds, dshape, dJds); // does not depend on mesh nodes
      /// v_el.CalcDShape(ip, dshape); // does not depend on mesh nodes

      /// auto force = dBds * energy_dot;
      double dBds_bar = force_bar * energy_dot;
      double energy_dot_bar = force_bar * dBds;

      /// double energy_dot = calcMagneticEnergyDot(trans, ip, nu, b_mag);
      auto energy_double_dot =
          calcMagneticEnergyDoubleDot(trans, ip, nu, b_mag);
      b_mag_bar += energy_dot_bar * energy_double_dot;

      /// TODO: replace with class matrix
      DenseMatrix dBdX_bar(v_el.GetDim(), v_el.GetDof());
      dBdX_bar = 0.0;  // same shape as dBdX
      for (int j = 0; j < v_el.GetDof(); ++j)
      {
         for (int k = 0; k < space_dim; ++k)
         {
            /// dBds += dBdX(k, j) * dXds(j, k);
            dBdX_bar(k, j) += dBds_bar * dXds(j, k);
         }
      }

      /// isotrans.JacobianRevDiff(db_magdJ, dBdX);
      /// aka AddMultABt(db_magdJ, dshape, dBdX);
      double db_magdJ_bar_buffer[9] = {};
      DenseMatrix db_magdJ_bar(db_magdJ_bar_buffer, space_dim, space_dim);
      Mult(dBdX_bar, dshape, db_magdJ_bar);

      // /// isotrans.JacobianRevDiff(dBmdJ, dBdX);
      // /// aka AddMultABt(dBmdJ, dshape, dBdX);
      // DenseMatrix dBmdJ_bar(dimc);
      // dBmdJ_bar = 0.0;
      // v_el.CalcDShape(ip, dshape);
      // AddMult(dBdX_bar, dshape, dBmdJ_bar);

      double b_vec_norm_bar = 0.0;
      double trans_weight_bar = 0.0;
      double adj_jac_bar_buffer[9] = {};
      DenseMatrix adj_jac_bar(adj_jac_bar_buffer, space_dim, space_dim);
      if (dim == 3)
      {
         double b_hat_buffer[3] = {};
         Vector b_hat(b_hat_buffer, curl_dim);
         curlshape.MultTranspose(elfun, b_hat);

         double BB_hatT_buffer[9] = {};
         DenseMatrix BB_hatT(BB_hatT_buffer, curl_dim, curl_dim);
         MultVWt(b_vec, b_hat, BB_hatT);

         double BB_hatT_bar_buffer[9] = {};
         DenseMatrix BB_hatT_bar(BB_hatT_bar_buffer, curl_dim, curl_dim);
         BB_hatT_bar = 0.0;

         /// db_magdJ.Add(1.0 / (b_vec_norm * trans_weight), BB_hatT);
         BB_hatT_bar.Add(1.0 / (b_vec_norm * trans_weight), db_magdJ_bar);

         for (int j = 0; j < curl_dim; ++j)
         {
            for (int k = 0; k < curl_dim; ++k)
            {
               b_vec_norm_bar -= db_magdJ_bar(j, k) * BB_hatT(j, k) /
                                 (pow(b_vec_norm, 2) * trans_weight);
               trans_weight_bar -= db_magdJ_bar(j, k) * BB_hatT(j, k) /
                                   (b_vec_norm * pow(trans_weight, 2));
            }
         }

         /// db_magdJ.Transpose();
         db_magdJ_bar.Transpose();

         /// db_magdJ.Add(-b_vec_norm / pow(trans_weight, 2),
         ///              trans.AdjugateJacobian());
         adj_jac_bar.Add(-b_vec_norm / pow(trans_weight, 2), db_magdJ_bar);
         for (int j = 0; j < space_dim; ++j)
         {
            for (int k = 0; k < space_dim; ++k)
            {
               b_vec_norm_bar -= db_magdJ_bar(j, k) *
                                 trans.AdjugateJacobian()(j, k) /
                                 pow(trans_weight, 2);
               trans_weight_bar += 2 * db_magdJ_bar(j, k) * b_vec_norm *
                                   trans.AdjugateJacobian()(j, k) /
                                   pow(trans_weight, 3);
            }
         }

         // double b_hat_bar_buffer[3] = {};
         // Vector b_hat_bar(b_hat_bar_buffer, curl_dim);

         /// MultVWt(b_vec, b_hat, BB_hatT);
         BB_hatT_bar.Mult(b_hat, b_vec_bar);
         // BB_hatT_bar.MultTranspose(b_vec, b_hat_bar);
         // does not depend on mesh nodes

         /// curlshape.AddMultTranspose(elfun, b_hat);
         // does not depend on mesh nodes
      }
      else
      {
         double b_adjJT_buffer[3] = {};
         Vector b_adjJT(b_adjJT_buffer, curl_dim);
         trans.AdjugateJacobian().Mult(b_vec, b_adjJT);

         double a = -1 / (b_vec_norm * pow(trans_weight, 2));

         /// AddMult_a_VWt(a, b_vec, b_adjJT, db_magdJ);
         double b_adjJT_var_buffer[3] = {};
         Vector b_adjJT_bar(b_adjJT_var_buffer, curl_dim);

         b_vec_bar = 0.0;
         db_magdJ_bar.AddMult_a(a, b_adjJT, b_vec_bar);
         b_adjJT_bar = 0.0;
         db_magdJ_bar.AddMultTranspose_a(a, b_vec, b_adjJT_bar);
         double a_bar = 0;
         for (int j = 0; j < space_dim; ++j)
         {
            for (int k = 0; k < space_dim; ++k)
            {
               a_bar += db_magdJ_bar(j, k) * db_magdJ(j, k);
            }
         }
         a_bar /= a;

         /// double a = -1 / (b_vec_norm * pow(trans_weight, 2));
         b_vec_norm_bar += a_bar / pow(b_vec_norm * trans_weight, 2);
         trans_weight_bar = 2 * a_bar / (b_vec_norm * pow(trans_weight, 3));

         /// trans.AdjugateJacobian().Mult(b_vec, b_adjJT);
         trans.AdjugateJacobian().AddMultTranspose(b_adjJT_bar, b_vec_bar);
         MultVWt(b_adjJT_bar, b_vec, adj_jac_bar);
      }

      /// const double b_mag = b_vec_norm / trans.Weight();
      b_vec_norm_bar += b_mag_bar / trans.Weight();
      trans_weight_bar -= b_mag_bar * b_vec_norm / pow(trans.Weight(), 2);

      /// const double b_vec_norm = b_vec.Norml2();
      add(b_vec_bar, b_vec_norm_bar / b_vec_norm, b_vec, b_vec_bar);

      if (dim == 3)
      {
         /// curlshape_dFt.AddMultTranspose(elfun, b_vec);
         // transposed dimensions of curlshape_dFt
         // so I don't have to transpose jac_bar later
         curlshape_dFt_bar.SetSize(curl_dim, ndof);
         MultVWt(b_vec_bar, elfun, curlshape_dFt_bar);

         /// MultABt(curlshape, trans.Jacobian(), curlshape_dFt);
         AddMult(curlshape_dFt_bar, curlshape, jac_bar);
      }
      else  // Dealing with scalar H1 field representing Az
      {
         /// curlshape_dFt.AddMultTranspose(elfun, b_vec);
         curlshape_dFt_bar.SetSize(ndof, curl_dim);
         MultVWt(elfun, b_vec_bar, curlshape_dFt_bar);

         /// Mult(curlshape, trans.AdjugateJacobian(), curlshape_dFt);
         MultAtB(curlshape, curlshape_dFt_bar, scratch);
         adj_jac_bar += scratch;
      }

      PointMat_bar = 0.0;
      isotrans.AdjugateJacobianRevDiff(adj_jac_bar, PointMat_bar);

      /// const double w = ip.weight * trans.Weight();
      trans_weight_bar += w_bar * ip.weight;

      isotrans.WeightRevDiff(trans_weight_bar, PointMat_bar);

      isotrans.JacobianRevDiff(jac_bar, PointMat_bar);

      // code to insert PointMat_bar into mesh_coords_bar;
      for (int j = 0; j < mesh_ndof; ++j)
      {
         for (int k = 0; k < curl_dim; ++k)
         {
            mesh_coords_bar(k * mesh_ndof + j) += PointMat_bar(k, j);
         }
      }
   }
}

double ForceIntegrator::GetElementEnergy(const FiniteElement &el,
                                         ElementTransformation &trans,
                                         const Vector &elfun)
{
   if (attrs.count(trans.Attribute) == 1)
   {
      return 0.0;
   }
   /// get the proper element, transformation, and v vector
#ifdef MFEM_THREAD_SAFE
   Array<int> vdofs;
   Vector vfun;
#endif
   int element = trans.ElementNo;
   const auto &v_el = *v.FESpace()->GetFE(element);
   v.FESpace()->GetElementVDofs(element, vdofs);
   v.GetSubVector(vdofs, vfun);
   DenseMatrix dXds(vfun.GetData(), v_el.GetDof(), v_el.GetDim());
   if (vfun.Normlinf() < 1e-14)
   {
      return 0.0;
   }
   /// number of degrees of freedom
   int ndof = el.GetDof();
   int dim = el.GetDim();
   int space_dim = trans.GetSpaceDim();
   int curl_dim = space_dim;

#ifdef MFEM_THREAD_SAFE
   DenseMatrix dshape(v_el.GetDof(), v_el.GetDim());
   DenseMatrix curlshape(ndof, curl_dim), curlshape_dFt(ndof, curl_dim);
   DenseMatrix dBdX(v_el.GetDim(), v_el.GetDof());
   Vector b_vec(curl_dim), b_hat(curl_dim);
#else
   dshape.SetSize(v_el.GetDof(), v_el.GetDim());
   curlshape.SetSize(ndof, curl_dim);
   curlshape_dFt.SetSize(ndof, curl_dim);
   dBdX.SetSize(v_el.GetDim(), v_el.GetDof());
   b_vec.SetSize(curl_dim);
   b_hat.SetSize(curl_dim);
#endif

   // cast the ElementTransformation
   auto &isotrans = dynamic_cast<IsoparametricTransformation &>(trans);

   const IntegrationRule *ir = IntRule;
   if (ir == nullptr)
   {
      int order = [&]()
      {
         if (el.Space() == FunctionSpace::Pk)
         {
            return 2 * el.GetOrder() - 2;
         }
         else
         {
            return 2 * el.GetOrder();
         }
      }();

      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   double fun = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);

      const double trans_weight = trans.Weight();
      /// holds quadrature weight
      double w = ip.weight * trans_weight;
      if (dim == 3)
      {
         el.CalcCurlShape(ip, curlshape);
         MultABt(curlshape, trans.Jacobian(), curlshape_dFt);
      }
      else  // Dealing with scalar H1 field representing Az
      {
         /// Not exactly the curl matrix, but since we just want the magnitude
         /// of the curl its okay
         el.CalcDShape(ip, curlshape);
         Mult(curlshape, trans.AdjugateJacobian(), curlshape_dFt);
      }

      b_vec = 0.0;
      curlshape_dFt.AddMultTranspose(elfun, b_vec);
      const double b_vec_norm = b_vec.Norml2();
      const double b_mag = b_vec_norm / trans_weight;

      /// the following computes `\partial (||B||/|J|) / \partial J`
      double dBmdJ_buffer[9] = {};
      DenseMatrix dBmdJ(dBmdJ_buffer, curl_dim, curl_dim);
      dBmdJ = 0.0;
      if (dim == 3)
      {
         b_hat = 0.0;
         curlshape.AddMultTranspose(elfun, b_hat);
         double BB_hatT_buffer[9] = {};
         DenseMatrix BB_hatT(BB_hatT_buffer, curl_dim, curl_dim);
         MultVWt(b_vec, b_hat, BB_hatT);

         auto inv_jac_transposed = trans.InverseJacobian();
         inv_jac_transposed.Transpose();

         Add(1.0 / (trans_weight * b_vec_norm),
             BB_hatT,
             -b_vec_norm / trans_weight,
             inv_jac_transposed,
             dBmdJ);
      }
      else
      {
         b_hat = 0.0;
         trans.InverseJacobian().Mult(b_vec, b_hat);
         AddMult_a_VWt(-1 / b_vec_norm, b_vec, b_hat, dBmdJ);
      }

      /// and then contracts with \partial J / \partial X
      dBdX = 0.0;
      isotrans.JacobianRevDiff(dBmdJ, dBdX);

      double dBds = 0.0;
      for (int j = 0; j < v_el.GetDof(); ++j)
      {
         for (int d = 0; d < curl_dim; ++d)
         {
            dBds += dBdX(d, j) * dXds(j, d);
         }
      }
      const double energy_dot = calcMagneticEnergyDot(trans, ip, nu, b_mag);
      auto force = dBds * energy_dot;

      v_el.CalcDShape(ip, dshape);
      double JinvdJds_buffer[9] = {};
      DenseMatrix JinvdJds(JinvdJds_buffer, space_dim, space_dim);
      double dJds_buffer[9] = {};
      DenseMatrix dJds(dJds_buffer, space_dim, space_dim);
      MultAtB(dXds, dshape, dJds);
      Mult(trans.InverseJacobian(), dJds, JinvdJds);
      double JinvdJdsTrace = JinvdJds.Trace();

      const double energy = calcMagneticEnergy(trans, ip, nu, b_mag);
      double force2 = energy * JinvdJdsTrace;
      fun -= (force + force2) * w;
   }
   return fun;
}

void ForceIntegrator::AssembleElementVector(const mfem::FiniteElement &el,
                                            mfem::ElementTransformation &trans,
                                            const mfem::Vector &elfun,
                                            mfem::Vector &elfun_bar)
{
   /// number of degrees of freedom
   const int ndof = el.GetDof();
   const int dim = el.GetDim();
   const int dimc = (dim == 3) ? 3 : 1;

   elfun_bar.SetSize(ndof);
   elfun_bar = 0.0;
   if (attrs.count(trans.Attribute) == 1)
   {
      return;
   }

   /// get the proper element, transformation, and v vector
#ifdef MFEM_THREAD_SAFE
   Array<int> vdofs;
   Vector vfun;
#endif
   const int element = trans.ElementNo;
   const auto &v_el = *v.FESpace()->GetFE(element);
   v.FESpace()->GetElementVDofs(element, vdofs);
   v.GetSubVector(vdofs, vfun);
   DenseMatrix dXds(vfun.GetData(), v_el.GetDof(), v_el.GetDim());
   if (vfun.Normlinf() < 1e-14)
   {
      return;
   }

#ifdef MFEM_THREAD_SAFE
   DenseMatrix dshape(v_el.GetDof(), v_el.GetDim());
   DenseMatrix curlshape(ndof, dimc), curlshape_dFt(ndof, dimc);
   DenseMatrix dBdX(v_el.GetDim(), v_el.GetDof());
   Vector b_vec(dimc), b_hat(dimc);
#else
   dshape.SetSize(v_el.GetDof(), v_el.GetDim());
   curlshape.SetSize(ndof, dimc);
   curlshape_dFt.SetSize(ndof, dimc);
   dBdX.SetSize(v_el.GetDim(), v_el.GetDof());
   b_vec.SetSize(dimc);
   b_hat.SetSize(dimc);
#endif

   // cast the ElementTransformation
   auto &isotrans = dynamic_cast<IsoparametricTransformation &>(trans);

   const IntegrationRule *ir = IntRule;
   if (ir == nullptr)
   {
      int order = [&]()
      {
         if (el.Space() == FunctionSpace::Pk)
         {
            return 2 * el.GetOrder() - 2;
         }
         else
         {
            return 2 * el.GetOrder();
         }
      }();

      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      /// forward pass
      const IntegrationPoint &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);

      /// holds quadrature weight
      const double w = ip.weight * trans.Weight();
      if (dim == 3)
      {
         el.CalcCurlShape(ip, curlshape);
         MultABt(curlshape, trans.Jacobian(), curlshape_dFt);
      }
      else
      {
         el.CalcCurlShape(ip, curlshape_dFt);
      }

      b_vec = 0.0;
      curlshape_dFt.AddMultTranspose(elfun, b_vec);
      const double b_vec_norm = b_vec.Norml2();
      const double b_mag = b_vec_norm / trans.Weight();

      /// the following computes `\partial (||B||/|J|) / \partial J`
      DenseMatrix dBmdJ(dimc);
      b_hat = 0.0;
      curlshape.AddMultTranspose(elfun, b_hat);
      DenseMatrix BB_hatT(dimc);
      MultVWt(b_vec, b_hat, BB_hatT);

      auto inv_jac_transposed = trans.InverseJacobian();
      inv_jac_transposed.Transpose();

      Add(1.0 / (trans.Weight() * b_vec_norm),
          BB_hatT,
          -b_vec_norm / trans.Weight(),
          inv_jac_transposed,
          dBmdJ);

      /// and then contracts with \partial J / \partial X
      dBdX = 0.0;
      isotrans.JacobianRevDiff(dBmdJ, dBdX);

      double dBds = 0.0;
      for (int j = 0; j < v_el.GetDof(); ++j)
      {
         for (int d = 0; d < dimc; ++d)
         {
            dBds += dBdX(d, j) * dXds(j, d);
         }
      }
      const double energy_dot = calcMagneticEnergyDot(trans, ip, nu, b_mag);
      // dBds *= energy_dot;
      // auto force = dBds * energy_dot;

      v_el.CalcDShape(ip, dshape);
      DenseMatrix JinvdJds(3);
      DenseMatrix dJds(3);
      MultAtB(dXds, dshape, dJds);
      Mult(trans.InverseJacobian(), dJds, JinvdJds);

      // const double energy = calcMagneticEnergy(trans, ip, nu, b_mag);
      // double force2 = energy * JinvdJds.Trace();

      // fun -= (force + force2) * w;

      /// start reverse pass
      double fun_bar = 1.0;
      double force_bar = 0.0;
      double force2_bar = 0.0;

      /// fun -= (force + force2) * w;
      force_bar -= fun_bar * w;
      force2_bar -= fun_bar * w;

      /// double force2 = energy * JinvdJds.Trace();
      double energy_bar = force2_bar * JinvdJds.Trace();

      /// const double energy = calcMagneticEnergy(trans, ip, nu, b_mag);
      double b_mag_bar = 0.0;
      b_mag_bar += energy_bar * energy_dot;

      /// auto force = dBds * energy_dot;
      double dBds_bar = force_bar * energy_dot;
      double energy_dot_bar = force_bar * dBds;

      /// const double energy_dot = calcMagneticEnergyDot(trans, ip, nu,
      /// b_mag);
      auto energy_double_dot =
          calcMagneticEnergyDoubleDot(trans, ip, nu, b_mag);
      b_mag_bar += energy_dot_bar * energy_double_dot;

      DenseMatrix dBdX_bar(v_el.GetDim(), v_el.GetDof());
      dBdX_bar = 0.0;  // same shape as dBdX
      for (int j = 0; j < v_el.GetDof(); ++j)
      {
         for (int d = 0; d < dimc; ++d)
         {
            /// dBds += dBdX(d, j) * dXds(j, d);
            dBdX_bar(d, j) += dBds_bar * dXds(j, d);
         }
      }

      /// isotrans.JacobianRevDiff(dBmdJ, dBdX);
      /// aka AddMultABt(dBmdJ, dshape, dBdX);
      DenseMatrix dBmdJ_bar(dimc);
      dBmdJ_bar = 0.0;
      v_el.CalcDShape(ip, dshape);
      AddMult(dBdX_bar, dshape, dBmdJ_bar);

      /// Add(1.0 / (trans.Weight() * b_vec_norm), BB_hatT,
      ///     -b_vec_norm / trans.Weight(), inv_jac_transposed, dBmdJ);

      DenseMatrix BB_hatT_bar(dBmdJ_bar);
      BB_hatT_bar *= 1.0 / (trans.Weight() * b_vec_norm);

      double b_vec_norm_bar = 0.0;
      for (int j = 0; j < dimc; ++j)
      {
         for (int d = 0; d < dimc; ++d)
         {
            b_vec_norm_bar -=
                dBmdJ_bar(j, d) *
                (BB_hatT(j, d) / (trans.Weight() * pow(b_vec_norm, 2)) +
                 inv_jac_transposed(j, d) / trans.Weight());
         }
      }

      /// MultVWt(b_vec, b_hat, BB_hatT);
      Vector b_vec_bar(dimc);
      Vector b_hat_bar(dimc);
      BB_hatT_bar.Mult(b_hat, b_vec_bar);
      BB_hatT_bar.MultTranspose(b_vec, b_hat_bar);

      /// curlshape.AddMultTranspose(elfun, b_hat);
      curlshape.AddMult(b_hat_bar, elfun_bar);

      /// const double b_mag = b_vec_norm / trans.Weight();
      b_vec_norm_bar += b_mag_bar / trans.Weight();

      /// const double b_vec_norm = b_vec.Norml2();
      add(b_vec_bar, b_vec_norm_bar / b_vec_norm, b_vec, b_vec_bar);

      /// curlshape_dFt.AddMultTranspose(elfun, b_vec);
      curlshape_dFt.AddMult(b_vec_bar, elfun_bar);
   }
}

void ForceIntegratorMeshSens::AssembleRHSElementVect(
    const FiniteElement &mesh_el,
    ElementTransformation &mesh_trans,
    Vector &mesh_coords_bar)
{
   auto &nu = force_integ.nu;
   auto &v = force_integ.v;
   auto &attrs = force_integ.attrs;

   /// get the proper element, transformation, and state vector
#ifdef MFEM_THREAD_SAFE
   Array<int> vdofs;
   Vector elfun, vfun;
#endif
   auto &vdofs = force_integ.vdofs;
   auto &vfun = force_integ.vfun;

   const int element = mesh_trans.ElementNo;
   const auto &el = *state.FESpace()->GetFE(element);
   auto &trans = *state.FESpace()->GetElementTransformation(element);

   const int ndof = mesh_el.GetDof();
   const int el_ndof = el.GetDof();
   const int dim = el.GetDim();
   const int dimc = (dim == 3) ? 3 : 1;
   mesh_coords_bar.SetSize(ndof * dimc);
   mesh_coords_bar = 0.0;
   if (attrs.count(trans.Attribute) == 1)
   {
      return;
   }
   auto *dof_tr = state.FESpace()->GetElementVDofs(element, vdofs);
   state.GetSubVector(vdofs, elfun);
   if (dof_tr != nullptr)
   {
      dof_tr->InvTransformPrimal(elfun);
   }

   /// get the proper element, transformation, and v vector
   // const int element = trans.ElementNo;
   const auto &v_el = *v.FESpace()->GetFE(element);
   v.FESpace()->GetElementVDofs(element, vdofs);
   v.GetSubVector(vdofs, vfun);
   DenseMatrix dXds(vfun.GetData(), v_el.GetDof(), v_el.GetDim());
   if (vfun.Normlinf() < 1e-14)
   {
      return;
   }

#ifdef MFEM_THREAD_SAFE
   DenseMatrix dshape(v_el.GetDof(), v_el.GetDim());
   DenseMatrix curlshape(el_ndof, dimc);
   DenseMatrix curlshape_dFt(el_ndof, dimc);
   DenseMatrix dBdX(v_el.GetDim(), v_el.GetDof());
   Vector b_vec(dimc), b_hat(dimc);
   DenseMatrix PointMat_bar(dimc, ndof);
#else
   auto &dshape = force_integ.dshape;
   auto &curlshape = force_integ.curlshape;
   auto &curlshape_dFt = force_integ.curlshape_dFt;
   auto &dBdX = force_integ.dBdX;
   auto &b_vec = force_integ.b_vec;
   auto &b_hat = force_integ.b_hat;
   dshape.SetSize(v_el.GetDof(), v_el.GetDim());
   curlshape.SetSize(el_ndof, dimc);
   curlshape_dFt.SetSize(el_ndof, dimc);
   dBdX.SetSize(v_el.GetDim(), v_el.GetDof());
   b_vec.SetSize(dimc);
   b_hat.SetSize(dimc);
   PointMat_bar.SetSize(dimc, ndof);
#endif

   // cast the ElementTransformation
   auto &isotrans = dynamic_cast<IsoparametricTransformation &>(trans);

   const IntegrationRule *ir = IntRule;
   if (ir == nullptr)
   {
      int order = [&]()
      {
         if (el.Space() == FunctionSpace::Pk)
         {
            return 2 * el.GetOrder() - 2;
         }
         else
         {
            return 2 * el.GetOrder();
         }
      }();

      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);

      /// holds quadrature weight
      const double w = ip.weight * trans.Weight();
      if (dim == 3)
      {
         el.CalcCurlShape(ip, curlshape);
         MultABt(curlshape, trans.Jacobian(), curlshape_dFt);
      }
      else
      {
         el.CalcCurlShape(ip, curlshape_dFt);
      }

      b_vec = 0.0;
      curlshape_dFt.AddMultTranspose(elfun, b_vec);
      const double b_vec_norm = b_vec.Norml2();
      const double b_mag = b_vec_norm / trans.Weight();

      /// the following computes `\partial (||B||/|J|) / \partial J`
      DenseMatrix dBmdJ(dimc);
      b_hat = 0.0;
      curlshape.AddMultTranspose(elfun, b_hat);
      DenseMatrix BB_hatT(dimc);
      MultVWt(b_vec, b_hat, BB_hatT);

      auto inv_jac_transposed = trans.InverseJacobian();
      inv_jac_transposed.Transpose();

      Add(1.0 / (trans.Weight() * b_vec_norm),
          BB_hatT,
          -b_vec_norm / trans.Weight(),
          inv_jac_transposed,
          dBmdJ);

      /// and then contracts with \partial J / \partial X
      dBdX = 0.0;
      isotrans.JacobianRevDiff(dBmdJ, dBdX);

      double dBds = 0.0;
      for (int j = 0; j < v_el.GetDof(); ++j)
      {
         for (int d = 0; d < dimc; ++d)
         {
            dBds += dBdX(d, j) * dXds(j, d);
         }
      }
      const double energy_dot = calcMagneticEnergyDot(trans, ip, nu, b_mag);
      auto force = dBds * energy_dot;
      // auto force = dBds;

      v_el.CalcDShape(ip, dshape);
      DenseMatrix JinvdJds(3);
      DenseMatrix dJds(3);
      MultAtB(dXds, dshape, dJds);
      Mult(trans.InverseJacobian(), dJds, JinvdJds);
      double JinvdJdsTrace = JinvdJds.Trace();

      const double energy = calcMagneticEnergy(trans, ip, nu, b_mag);
      double force2 = energy * JinvdJdsTrace;
      // double force2 = 0.0;
      // fun -= (force + force2) * w;

      /// start reverse pass
      double fun_bar = 1.0;

      /// fun -= (force + force2) * w;
      double force_bar = 0.0;
      double force2_bar = 0.0;
      double w_bar = 0.0;
      force_bar -= fun_bar * w;
      force2_bar -= fun_bar * w;
      w_bar -= fun_bar * (force + force2);

      /// double force2 = energy * JinvdJdsTrace;
      double energy_bar = force2_bar * JinvdJdsTrace;
      double JinvdJdsTrace_bar = force2_bar * energy;

      /// const double energy = calcMagneticEnergy(trans, ip, nu, b_mag);
      double b_mag_bar = 0.0;
      b_mag_bar += energy_bar * energy_dot;

      /// double JinvdJdsTrace = JinvdJds.Trace();
      DenseMatrix JinvdJds_bar;
      JinvdJds_bar.Diag(JinvdJdsTrace_bar, dimc);

      /// Mult(trans.InverseJacobian(), dJds, JinvdJds);
      DenseMatrix dJds_bar(dimc);
      DenseMatrix jac_bar(dimc);
      jac_bar = 0.0;
      Mult(inv_jac_transposed, JinvdJds_bar, dJds_bar);
      MultABt(dJds_bar, JinvdJds, jac_bar);
      jac_bar *= -1.0;

      /// MultAtB(dXds, dshape, dJds); // does not depend on mesh nodes

      /// auto force = dBds * energy_dot;
      double dBds_bar = force_bar * energy_dot;
      double energy_dot_bar = force_bar * dBds;
      // double dBds_bar = force_bar;

      /// const double energy_dot = calcMagneticEnergyDot(trans, ip, nu,
      /// b_mag);
      auto energy_double_dot =
          calcMagneticEnergyDoubleDot(trans, ip, nu, b_mag);
      b_mag_bar += energy_dot_bar * energy_double_dot;

      DenseMatrix dBdX_bar(v_el.GetDim(), v_el.GetDof());
      dBdX_bar = 0.0;  // same shape as dBdX
      for (int j = 0; j < v_el.GetDof(); ++j)
      {
         for (int d = 0; d < dimc; ++d)
         {
            /// dBds += dBdX(d, j) * dXds(j, d);
            dBdX_bar(d, j) += dBds_bar * dXds(j, d);
         }
      }

      /// isotrans.JacobianRevDiff(dBmdJ, dBdX);
      /// aka AddMultABt(dBmdJ, dshape, dBdX);
      DenseMatrix dBmdJ_bar(dimc);
      dBmdJ_bar = 0.0;
      v_el.CalcDShape(ip, dshape);
      AddMult(dBdX_bar, dshape, dBmdJ_bar);

      /// Add(1.0 / (trans.Weight() * b_vec_norm), BB_hatT,
      ///     -b_vec_norm / trans.Weight(), inv_jac_transposed, dBmdJ);
      DenseMatrix BB_hatT_bar(dBmdJ_bar);
      BB_hatT_bar *= 1.0 / (trans.Weight() * b_vec_norm);

      DenseMatrix inv_jac_transposed_bar(dBmdJ_bar);
      inv_jac_transposed_bar *= -b_vec_norm / trans.Weight();

      double b_vec_norm_bar = 0.0;
      double trans_weight_bar = 0.0;
      for (int j = 0; j < dimc; ++j)
      {
         for (int d = 0; d < dimc; ++d)
         {
            b_vec_norm_bar -=
                dBmdJ_bar(j, d) *
                (BB_hatT(j, d) / (trans.Weight() * pow(b_vec_norm, 2)) +
                 inv_jac_transposed(j, d) / trans.Weight());

            trans_weight_bar -=
                dBmdJ_bar(j, d) *
                (BB_hatT(j, d) / (pow(trans.Weight(), 2) * b_vec_norm) -
                 inv_jac_transposed(j, d) * b_vec_norm /
                     pow(trans.Weight(), 2));
         }
      }

      /// inv_jac_transposed.Transpose();
      inv_jac_transposed_bar.Transpose();  // this could be the issue

      /// auto inv_jac_transposed = trans.InverseJacobian();
      DenseMatrix inv_jac_barinv_jacT(dimc);
      Mult(inv_jac_transposed_bar, inv_jac_transposed, inv_jac_barinv_jacT);
      inv_jac_barinv_jacT *= -1.0;
      AddMult(inv_jac_transposed, inv_jac_barinv_jacT, jac_bar);

      /// MultVWt(b_vec, b_hat, BB_hatT);
      Vector b_vec_bar(dimc);
      b_vec_bar = 0.0;
      BB_hatT_bar.Mult(b_hat, b_vec_bar);

      /// curlshape.AddMultTranspose(elfun, b_hat); // no effect from mesh
      /// coords

      /// const double b_mag = b_vec_norm / trans.Weight();
      b_vec_norm_bar += b_mag_bar / trans.Weight();
      trans_weight_bar -= b_mag_bar * b_vec_norm / pow(trans.Weight(), 2);

      /// const double b_vec_norm = b_vec.Norml2();
      add(b_vec_bar, b_vec_norm_bar / b_vec_norm, b_vec, b_vec_bar);

      /// curlshape_dFt.AddMultTranspose(elfun, b_vec);
      DenseMatrix curlshape_dFt_bar(
          dimc, el_ndof);  // transposed dimensions of curlshape_dFt so I
                           // don't have to transpose J later
      MultVWt(b_vec_bar, elfun, curlshape_dFt_bar);

      /// MultABt(curlshape, trans.Jacobian(), curlshape_dFt);
      AddMult(curlshape_dFt_bar, curlshape, jac_bar);

      /// const double w = ip.weight * trans.Weight();
      trans_weight_bar += w_bar * ip.weight;

      PointMat_bar = 0.0;
      isotrans.WeightRevDiff(PointMat_bar);
      PointMat_bar *= trans_weight_bar;

      isotrans.JacobianRevDiff(jac_bar, PointMat_bar);

      // code to insert PointMat_bar into mesh_coords_bar;
      for (int j = 0; j < ndof; ++j)
      {
         for (int d = 0; d < dimc; ++d)
         {
            mesh_coords_bar(d * ndof + j) += PointMat_bar(d, j);
         }
      }
   }
}

void setInputs(SteinmetzLossIntegrator &integ, const MachInputs &inputs)
{
   setValueFromInputs(inputs, "frequency", integ.freq);
   if (!integ.name.empty())
   {
      setValueFromInputs(
          inputs, "max_flux_magnitude:" + integ.name, integ.max_flux_mag);
   }
   else
   {
      setValueFromInputs(inputs, "max_flux_magnitude", integ.max_flux_mag);
   }
}

double SteinmetzLossIntegrator::GetElementEnergy(
    const mfem::FiniteElement &el,
    mfem::ElementTransformation &trans,
    const mfem::Vector &elfun)
{
   const auto *ir = IntRule;
   if (ir == nullptr)
   {
      int order = [&]()
      {
         if (el.Space() == FunctionSpace::Pk)
         {
            return 2 * el.GetOrder() - 2;
         }
         else
         {
            return 2 * el.GetOrder();
         }
      }();

      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   double fun = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);

      /// holds quadrature weight
      double trans_weight = trans.Weight();
      double w = ip.weight * trans_weight;

      auto rho_v = rho.Eval(trans, ip);
      auto k_s_v = k_s.Eval(trans, ip);
      auto alpha_v = alpha.Eval(trans, ip);
      auto beta_v = beta.Eval(trans, ip);

      fun += rho_v * k_s_v * pow(freq, alpha_v) * pow(max_flux_mag, beta_v) * w;
   }
   return fun;
}

void setInputs(CAL2CoreLossIntegrator &integ, const MachInputs &inputs)
{
   setValueFromInputs(inputs, "frequency", integ.freq);
   // Temperature and max flux density handled below
   ///TODO: Determine if the flux that is being used is correct
}

double CAL2CoreLossIntegrator::GetElementEnergy(
    const mfem::FiniteElement &el,
    mfem::ElementTransformation &trans,
    const mfem::Vector &elfun)
{
   int ndof = el.GetDof();

   // Using flux logic from ACLossFunctionalDistributionIntegrator and adapting as needed
   const int element = trans.ElementNo;
   const auto &flux_el = *peak_flux.FESpace()->GetFE(element);
   auto &flux_trans = *peak_flux.FESpace()->GetElementTransformation(element);
   const int flux_ndof = flux_el.GetDof();
   
   auto *dof_tr = peak_flux.FESpace()->GetElementVDofs(element, vdofs);
   peak_flux.GetSubVector(vdofs, flux_elfun);
   if (dof_tr != nullptr)
   {
      dof_tr->InvTransformPrimal(flux_elfun);
   }

   shape.SetSize(ndof);
   flux_shape.SetSize(flux_ndof);

   // Using temperature logic from DCLossFunctional Integrator
   const FiniteElement *temp_el=nullptr;
   if (temperature_field != nullptr)
   {
      temp_el = temperature_field->FESpace()->GetFE(element);

      auto *dof_tr = temperature_field->FESpace()->GetElementVDofs(element, vdofs);
      temperature_field->GetSubVector(vdofs, temp_elfun);
      if (dof_tr != nullptr)
      {
         dof_tr->InvTransformPrimal(temp_elfun);
      }
      
      int ndof = temp_el->GetDof();
      shape.SetSize(ndof); // shape will pertain to temperature
      
   }

   //Should be fine to leave el as is in this scope (rather than replace with temp_el)
   const IntegrationRule *ir = IntRule;
   if (ir == nullptr)
   {
      int order = [&]()
      {
         if (el.Space() == FunctionSpace::Pk)
         {
            return 2 * el.GetOrder() - 1;
         }
         else
         {
            return 2 * el.GetOrder();
         }
      }();

      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   double fun = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip); 
      const double trans_weight = trans.Weight();
      const double w = ip.weight * trans_weight;

      el.CalcPhysShape(trans, shape); // may be unused

      double temperature;

      if (temperature_field != nullptr)
      {
         temp_el->CalcPhysShape(trans, shape); // Alternative to CalcShape, used by ACLossFunctionalIntegrator. Difference between CalcPhysShape and CalcShape?
         temperature = shape * temp_elfun; //Take dot product between shape and elfun to get the value at the integration point
      }
      else
      {
         temperature = 100;
      }

      flux_el.CalcPhysShape(flux_trans, flux_shape);
      const auto max_flux_mag = flux_shape * flux_elfun;

      // Compute the values of the variable hysteresis and eddy current loss coefficients at the integration point
      // kh(f,T,Bm) and ke(f,T,Bm)
      auto kh_v = CAL2_kh.Eval(trans, ip, temperature, freq, max_flux_mag);
      auto ke_v = CAL2_ke.Eval(trans, ip, temperature, freq, max_flux_mag);

      fun += kh_v * freq * std::pow(max_flux_mag,2) * w; // Add the hysteresis loss constribution
      fun += ke_v * std::pow(freq,2) * std::pow(max_flux_mag,2) * w; // Add the eddy current loss constribution
   }
   return fun;
}

double SteinmetzLossIntegratorFreqSens::GetElementEnergy(
    const mfem::FiniteElement &el,
    mfem::ElementTransformation &trans,
    const mfem::Vector &elfun)
{
   const auto *ir = IntRule;
   if (ir == nullptr)
   {
      int order = [&]()
      {
         if (el.Space() == FunctionSpace::Pk)
         {
            return 2 * el.GetOrder() - 2;
         }
         else
         {
            return 2 * el.GetOrder();
         }
      }();

      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   auto &rho = integ.rho;
   auto &k_s = integ.k_s;
   auto &alpha = integ.alpha;
   auto &beta = integ.beta;
   auto freq = integ.freq;
   auto max_flux_mag = integ.max_flux_mag;

   double sens = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);

      /// holds quadrature weight
      double trans_weight = trans.Weight();
      double w = ip.weight * trans_weight;

      auto rho_v = rho.Eval(trans, ip);
      auto k_s_v = k_s.Eval(trans, ip);
      auto alpha_v = alpha.Eval(trans, ip);
      auto beta_v = beta.Eval(trans, ip);

      // fun += rho_v * k_s_v * pow(freq, alpha_v) * pow(max_flux_mag, beta_v) *
      // w;
      sens += rho_v * k_s_v * alpha_v * pow(freq, alpha_v - 1) *
              pow(max_flux_mag, beta_v) * w;
   }
   return sens;
}

double SteinmetzLossIntegratorMaxFluxSens::GetElementEnergy(
    const mfem::FiniteElement &el,
    mfem::ElementTransformation &trans,
    const mfem::Vector &elfun)
{
   const auto *ir = IntRule;
   if (ir == nullptr)
   {
      int order = [&]()
      {
         if (el.Space() == FunctionSpace::Pk)
         {
            return 2 * el.GetOrder() - 2;
         }
         else
         {
            return 2 * el.GetOrder();
         }
      }();

      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   auto &rho = integ.rho;
   auto &k_s = integ.k_s;
   auto &alpha = integ.alpha;
   auto &beta = integ.beta;
   auto freq = integ.freq;
   auto max_flux_mag = integ.max_flux_mag;

   double sens = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);

      /// holds quadrature weight
      double trans_weight = trans.Weight();
      double w = ip.weight * trans_weight;

      auto rho_v = rho.Eval(trans, ip);
      auto k_s_v = k_s.Eval(trans, ip);
      auto alpha_v = alpha.Eval(trans, ip);
      auto beta_v = beta.Eval(trans, ip);

      // fun += rho_v * k_s_v * pow(freq, alpha_v) * pow(max_flux_mag, beta_v) *
      // w;
      sens += rho_v * k_s_v * pow(freq, alpha_v) * beta_v *
              pow(max_flux_mag, beta_v - 1) * w;
   }
   return sens;
}

void SteinmetzLossIntegratorMeshSens::AssembleRHSElementVect(
    const mfem::FiniteElement &mesh_el,
    mfem::ElementTransformation &mesh_trans,
    mfem::Vector &mesh_coords_bar)
{
   const int element = mesh_trans.ElementNo;
   const auto &el = *state.FESpace()->GetFE(element);
   auto &trans = *state.FESpace()->GetElementTransformation(element);

   const int mesh_ndof = mesh_el.GetDof();
   const int space_dim = mesh_trans.GetSpaceDim();

   PointMat_bar.SetSize(space_dim, mesh_ndof);

   // cast the ElementTransformation
   auto &isotrans = dynamic_cast<IsoparametricTransformation &>(trans);

   const auto *ir = IntRule;
   if (ir == nullptr)
   {
      int order = [&]()
      {
         if (el.Space() == FunctionSpace::Pk)
         {
            return 2 * el.GetOrder() - 2;
         }
         else
         {
            return 2 * el.GetOrder();
         }
      }();

      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   auto &rho = integ.rho;
   auto &k_s = integ.k_s;
   auto &alpha = integ.alpha;
   auto &beta = integ.beta;
   auto freq = integ.freq;
   auto max_flux_mag = integ.max_flux_mag;

   mesh_coords_bar.SetSize(mesh_ndof * space_dim);
   mesh_coords_bar = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const auto &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);

      double trans_weight = trans.Weight();
      double w = ip.weight * trans_weight;

      auto rho_v = rho.Eval(trans, ip);
      auto k_s_v = k_s.Eval(trans, ip);
      auto alpha_v = alpha.Eval(trans, ip);
      auto beta_v = beta.Eval(trans, ip);

      // fun += rho_v * k_s_v * pow(freq, alpha_v) * pow(max_flux_mag, beta_v) *
      // w;

      /// Start reverse pass...
      /// fun += rho_v * k_s_v * pow(freq, alpha_v) * pow(max_flux_mag, beta_v)
      /// * w;
      double fun_bar = 1.0;
      double rho_v_bar =
          fun_bar * k_s_v * pow(freq, alpha_v) * pow(max_flux_mag, beta_v) * w;
      double k_s_v_bar =
          fun_bar * rho_v * pow(freq, alpha_v) * pow(max_flux_mag, beta_v) * w;
      // double freq_bar = fun_bar * rho_v * k_s_v * alpha_v *
      //                   pow(freq, alpha_v - 1) * pow(max_flux_mag, beta_v) *
      //                   w;
      double alpha_v_bar = fun_bar * rho_v * k_s_v * pow(freq, alpha_v) *
                           log(freq) * pow(max_flux_mag, beta_v) * w;
      // double max_flux_mag_bar = fun_bar * rho_v * k_s_v * pow(freq, alpha_v)
      // *
      //                           beta_v * pow(max_flux_mag, beta_v - 1) * w;
      double beta_v_bar = fun_bar * rho_v * k_s_v * pow(freq, alpha_v) *
                          pow(max_flux_mag, beta_v) * log(max_flux_mag) * w;
      double w_bar = fun_bar * rho_v * k_s_v * pow(freq, alpha_v) *
                     pow(max_flux_mag, beta_v);

      /// auto beta_v = beta.Eval(trans, ip);
      PointMat_bar = 0.0;
      beta.EvalRevDiff(beta_v_bar, trans, ip, PointMat_bar);

      /// auto alpha_v = alpha.Eval(trans, ip);
      alpha.EvalRevDiff(alpha_v_bar, trans, ip, PointMat_bar);

      /// auto k_s_v = k_s.Eval(trans, ip);
      k_s.EvalRevDiff(k_s_v_bar, trans, ip, PointMat_bar);

      /// auto rho_v = rho.Eval(trans, ip);
      rho.EvalRevDiff(rho_v_bar, trans, ip, PointMat_bar);

      /// double w = ip.weight * trans_weight;
      double trans_weight_bar = w_bar * ip.weight;

      /// double trans_weight = trans.Weight();
      isotrans.WeightRevDiff(trans_weight_bar, PointMat_bar);

      /// code to insert PointMat_bar into mesh_coords_bar;
      for (int j = 0; j < mesh_ndof; ++j)
      {
         for (int d = 0; d < space_dim; ++d)
         {
            mesh_coords_bar(d * mesh_ndof + j) += PointMat_bar(d, j);
         }
      }
   }
}

void setInputs(SteinmetzLossDistributionIntegrator &integ,
               const MachInputs &inputs)
{
   setValueFromInputs(inputs, "frequency", integ.freq);

   if (!integ.name.empty())
   {
      setValueFromInputs(
          inputs, "max_flux_magnitude:" + integ.name, integ.max_flux_mag);
   }
   else
   {
      setValueFromInputs(inputs, "max_flux_magnitude", integ.max_flux_mag);
   }
}

void SteinmetzLossDistributionIntegrator::AssembleRHSElementVect(
    const mfem::FiniteElement &el,
    mfem::ElementTransformation &trans,
    mfem::Vector &elvect)
{
   int ndof = el.GetDof();

#ifdef MFEM_THREAD_SAFE
   Vector shape;
#endif
   shape.SetSize(ndof);
   elvect.SetSize(ndof);

   const auto *ir = IntRule;
   if (ir == nullptr)
   {
      int order = [&]()
      {
         if (el.Space() == FunctionSpace::Pk)
         {
            return 2 * el.GetOrder() - 1;
         }
         else
         {
            return 2 * el.GetOrder();
         }
      }();

      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   elvect = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);

      el.CalcPhysShape(trans, shape);

      /// holds quadrature weight
      const double w = ip.weight * trans.Weight();

      double rho_v = rho.Eval(trans, ip);
      double k_s_v = k_s.Eval(trans, ip);
      double alpha_v = alpha.Eval(trans, ip);
      double beta_v = beta.Eval(trans, ip);

      double loss =
          rho_v * k_s_v * pow(freq, alpha_v) * pow(max_flux_mag, beta_v);

      elvect.Add(loss * w, shape);
   }
}

}  // namespace mach
