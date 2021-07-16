#include "electromag_integ.hpp"

#ifdef MFEM_USE_PUMI
#include "apfMDS.h"
#include "PCU.h"
#include "apfConvert.h"
#include "crv.h"
#include "gmi.h"
#endif // MFEM_USE_PUMI

#include "coefficient.hpp"
#include "solver.hpp"


using namespace mfem;

namespace mach
{

double calcMagneticEnergy(
   ElementTransformation &trans,
   const IntegrationPoint &ip,
   StateCoefficient &nu,
   double B)
{
   /// TODO: use a composite rule instead or find a way to just directly
   /// integrate B-H curve
   const IntegrationRule *ir = &IntRules.Get(Geometry::Type::SEGMENT, 40);

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

double calcMagneticEnergyDot(
   ElementTransformation &trans,
   const IntegrationPoint &ip,
   StateCoefficient &nu,
   double B)
{
   /// TODO: use a composite rule instead or find a way to just directly
   /// integrate B-H curve
   const IntegrationRule *ir = &IntRules.Get(Geometry::Type::SEGMENT, 40);

   /// compute int_0^{B} \nuB dB
   double en = 0.0;
   double dendB = 0.0;
   for (int j = 0; j < ir->GetNPoints(); j++)
   {
      const IntegrationPoint &segment_ip = ir->IntPoint(j);
      double xi = segment_ip.x * B;
      double dxidB = segment_ip.x;
      en += segment_ip.weight * xi * nu.Eval(trans, ip, xi);

      dendB += segment_ip.weight * (nu.Eval(trans, ip, xi)
                     +  xi * nu.EvalStateDeriv(trans, ip, xi)) * dxidB;
   }
   dendB *= B;
   dendB += en;
   return dendB;
}

double calcMagneticEnergyDoubleDot(
   ElementTransformation &trans,
   const IntegrationPoint &ip,
   StateCoefficient &nu,
   double B)
{
   /// TODO: use a composite rule instead or find a way to just directly
   /// integrate B-H curve
   const IntegrationRule *ir = &IntRules.Get(Geometry::Type::SEGMENT, 40);

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

      d2endB2 += w * (2*((dxidB * nu_val)+(xi * dnudB * dxidB)+(dxidB * dxidB * dnudB * B))
                     + xi * d2nudB2*dxidB*dxidB * B);
   }
   return d2endB2;
}

void CurlCurlNLFIntegrator::AssembleElementVector(
   const FiniteElement &el,
   ElementTransformation &trans,
   const Vector &elfun, Vector &elvect)
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

   elvect.SetSize(ndof);

   const IntegrationRule *ir = NULL;
   {
      int order;
      if (el.Space() == FunctionSpace::Pk)
      {
         order = 2*el.GetOrder() - 1;
      }
      else
      {
         order = 2*el.GetOrder();
      }

      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   elvect = 0.0;

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      b_vec = 0.0;
      const IntegrationPoint &ip = ir->IntPoint(i);

      trans.SetIntPoint(&ip);

      w = ip.weight;
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
      curlshape_dFt.AddMultTranspose(elfun, b_vec);
      b_vec /= trans.Weight();

      double model_val = model->Eval(trans, ip, b_vec.Norml2());
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

   /// I believe this takes advantage of a 2D problem not having
   /// a properly defined curl? Need more investigation
   int dimc = (dim == 3) ? 3 : 1;

   /// holds quadrature weight
   double w;

#ifdef MFEM_THREAD_SAFE
   DenseMatrix curlshape(ndof,dimc), curlshape_dFt(ndof,dimc);
   Vector b_vec(dimc), temp_vec(ndof);
#else
   curlshape.SetSize(ndof,dimc);
   curlshape_dFt.SetSize(ndof,dimc);
   b_vec.SetSize(dimc);
   temp_vec.SetSize(ndof);
#endif

   elmat.SetSize(ndof);

   const IntegrationRule *ir = NULL;
   {
      int order;
      if (el.Space() == FunctionSpace::Pk)
      {
         order = 2*el.GetOrder() - 1;
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
      b_vec /= trans.Weight();
      const double b_mag = b_vec.Norml2();

      /////////////////////////////////////////////////////////////////////////
      /// calculate first term of Jacobian
      /////////////////////////////////////////////////////////////////////////

      /// evaluate material model at ip
      double model_val = model->Eval(trans, ip, b_mag);
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
         /// TODO: is this thread safe?
         /// calculate curl(N_i) dot curl(A), need to store in a DenseMatrix so we
         /// can take outer product of result to generate matrix
         temp_vec = 0.0;
         curlshape_dFt.Mult(b_vec, temp_vec);
         // DenseMatrix temp_matrix(temp_vec.GetData(), ndof, 1);

         /// evaluate the derivative of the material model with respect to the
         /// norm of the grid function associated with the model at the point
         /// defined by ip, and scale by integration point weight
         double model_deriv = model->EvalStateDeriv(trans, ip, b_mag);
         model_deriv *= w;
         model_deriv /= b_mag;
      
         /// add second term to elmat
         // AddMult_a_AAt(model_deriv, temp_matrix, elmat);
         AddMult_a_VVt(model_deriv, temp_vec, elmat);

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

double CurlCurlNLFIntegrator::GetElementEnergy(
   const mfem::FiniteElement &el,
   mfem::ElementTransformation &trans,
   const mfem::Vector &elfun)
{
   /// get the proper element, transformation, and state vector
   Array<int> vdofs;
   Vector psi; 
   int element = trans.ElementNo;
   state->FESpace()->GetElementVDofs(element, vdofs);

   const IntegrationRule *ir = NULL;
   {
      int order;
      if (el.Space() == FunctionSpace::Pk)
      {
         order = 2*el.GetOrder() - 1;
      }
      else
      {
         order = 2*el.GetOrder();
      }

      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   adjoint->GetSubVector(vdofs, psi);

   int ndof = el.GetDof();
   int dim = el.GetDim();
   int dimc = (dim == 3) ? 3 : 1;
#ifdef MFEM_THREAD_SAFE
   DenseMatrix curlshape(ndof,dimc), curlshape_dFt(el_ndof,dimc);
   Vector b_vec(dimc), curl_psi(dimc);
#else
   curlshape.SetSize(ndof,dimc);
   curlshape_dFt.SetSize(ndof,dimc);
   b_vec.SetSize(dimc);
   curl_psi.SetSize(dimc);
#endif

   /// holds quadrature weight
   double w;


   double fun = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      b_vec = 0.0;
      curl_psi = 0.0;

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

      curlshape_dFt.AddMultTranspose(elfun, b_vec);
      curlshape_dFt.AddMultTranspose(psi, curl_psi);

      double b_mag = b_vec.Norml2();
      double model_val = model->Eval(trans, ip, b_mag);

      fun += model_val * (curl_psi * b_vec) * w;
      // fun += (curl_psi * b_vec) * ip.weight;
   }
   return fun;
}

void CurlCurlNLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &mesh_el,
   ElementTransformation &mesh_trans,
   Vector &elvect)
{
   /// get the proper element, transformation, and state vector
   Array<int> vdofs;
   Vector elfun, psi; 
   int element = mesh_trans.ElementNo;
   const FiniteElement *el = state->FESpace()->GetFE(element);
   ElementTransformation *trans = state->FESpace()->GetElementTransformation(element);
   state->FESpace()->GetElementVDofs(element, vdofs);

   const IntegrationRule *ir = NULL;
   {
      int order;
      if (el->Space() == FunctionSpace::Pk)
      {
         order = 2*el->GetOrder() - 1;
      }
      else
      {
         order = 2*el->GetOrder();
      }

      ir = &IntRules.Get(el->GetGeomType(), order);
   }

   state->GetSubVector(vdofs, elfun);
   adjoint->GetSubVector(vdofs, psi);

   int ndof = mesh_el.GetDof();
   int el_ndof = el->GetDof();
   int dim = el->GetDim();
   int dimc = (dim == 3) ? 3 : 1;
   elvect.SetSize(ndof*dimc);
   elvect = 0.0;
#ifdef MFEM_THREAD_SAFE
   DenseMatrix curlshape(el_ndof,dimc), curlshape_dFt(el_ndof,dimc);
   Vector b_vec(dimc), b_hat(dimc), curl_psi(dimc), curl_psi_hat(dimc);
#else
   curlshape.SetSize(el_ndof,dimc);
   curlshape_dFt.SetSize(el_ndof,dimc);
   b_vec.SetSize(dimc);
   b_hat.SetSize(dimc);
   curl_psi.SetSize(dimc);
   curl_psi_hat.SetSize(dimc);
#endif
   DenseMatrix PointMat_bar(dimc, ndof);
   
   // cast the ElementTransformation
   IsoparametricTransformation &isotrans =
   dynamic_cast<IsoparametricTransformation&>(*trans);

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      PointMat_bar = 0.0;
      b_hat = 0.0;
      b_vec = 0.0;
      curl_psi_hat = 0.0;
      curl_psi = 0.0;

      const IntegrationPoint &ip = ir->IntPoint(i);

      isotrans.SetIntPoint(&ip);

      if ( dim == 3 )
      {
         el->CalcCurlShape(ip, curlshape);
         MultABt(curlshape, isotrans.Jacobian(), curlshape_dFt);
      }
      else
      {
         el->CalcCurlShape(ip, curlshape_dFt);
      }

      curlshape.AddMultTranspose(elfun, b_hat);
      curlshape_dFt.AddMultTranspose(elfun, b_vec);
      curlshape.AddMultTranspose(psi, curl_psi_hat);
      curlshape_dFt.AddMultTranspose(psi, curl_psi);

      double b_mag = b_vec.Norml2();
      double model_val = model->Eval(isotrans, ip, b_mag);
      double model_deriv = model->EvalStateDeriv(isotrans, ip, b_mag);

      double curl_psi_dot_b = curl_psi * b_vec;

      // nu * (\partial a^T b / \partial J) / |J|
      DenseMatrix Jac_bar(3);
      MultVWt(b_vec, curl_psi_hat, Jac_bar);
      AddMultVWt(curl_psi, b_hat, Jac_bar);
      Jac_bar *= model_val / isotrans.Weight();

      // (\partial nu / \partial J) * a^T b / |J|
      // (\partial nu / \partial ||B||) * B / ||B|| * B_hat * a^T b
      AddMult_a_VWt(model_deriv * curl_psi_dot_b / (b_mag * isotrans.Weight()),
                    b_vec, b_hat, Jac_bar);

      // (- nu * a^T b / |J|^2)  * \partial |J| / \partial X
      isotrans.WeightRevDiff(PointMat_bar);
      PointMat_bar *= -model_val * curl_psi_dot_b / pow(trans->Weight(), 2.0);

      isotrans.JacobianRevDiff(Jac_bar, PointMat_bar);

      for (int j = 0; j < ndof ; ++j)
      {
         for (int d = 0; d < dimc; ++d)
         {
            elvect(d*ndof + j) += ip.weight * PointMat_bar(d,j);
         }
      }
   }
}

void MagnetizationIntegrator::AssembleElementVector(
   const FiniteElement &el,
   ElementTransformation &trans,
   const Vector &elfun, Vector &elvect)
{
   // std::cout << "mag integ\n";
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
   Vector b_vec(dimc) mag_vec(dimc);
#else
   curlshape.SetSize(ndof,dimc);
   curlshape_dFt.SetSize(ndof,dimc);
   b_vec.SetSize(dimc);
   mag_vec.SetSize(dimc);
#endif

   elvect.SetSize(ndof);

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

   elvect = 0.0;

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
         /// calculate curl(N_i) dot curl(A), need to store in a DenseMatrix so we
         /// can take outer product of result to generate matrix
         temp_vec = 0.0;
         curlshape_dFt.Mult(b_vec, temp_vec);
         DenseMatrix temp_matrix(temp_vec.GetData(), ndof, 1);

         mag_vec = 0.0;
         mag->Eval(mag_vec, trans, ip);

         temp_vec2 = 0.0;
         curlshape_dFt.Mult(mag_vec, temp_vec2);
         DenseMatrix temp_matrix2(temp_vec2.GetData(), ndof, 1);

         /// evaluate the derivative of the material model with respect to the
         /// norm of the grid function associated with the model at the point
         /// defined by ip, and scale by integration point weight
         double nu_deriv = nu->EvalStateDeriv(trans, ip, b_mag);
         nu_deriv *= w;
         nu_deriv /= b_mag;

         AddMult_a_ABt(nu_deriv, temp_matrix2, temp_matrix, elmat);
      }
   }
   */
}

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
   ElementTransformation &nd_trans = *adjoint->FESpace()->GetElementTransformation(element);

   /// get the RT elements used for V shape
   const FiniteElement &rt_el = *state->FESpace()->GetFE(element);
   ElementTransformation &rt_trans = *state->FESpace()->GetElementTransformation(element);

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
         rt_el.Project_RevDiff(P_bar, *vec_coeff, isotrans, PointMat_bar);
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
   ElementTransformation &trans = *adjoint->FESpace()->GetElementTransformation(element);

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
         el.Project_RevDiff(P_bar, *vec_coeff, isotrans, PointMat_bar);
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
   ElementTransformation &h1_trans = *adjoint->FESpace()->GetElementTransformation(element);

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
         nd_el.Project_RevDiff(P_bar, *vec_coeff, isotrans, PointMat_bar);
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
   ElementTransformation &trans = *adjoint->FESpace()->GetElementTransformation(element);

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
   ElementTransformation &trans = *adjoint->FESpace()->GetElementTransformation(element);

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
   el.Project_RevDiff(psi, *vec_coeff, isotrans, PointMat_bar);

   for (int j = 0; j < ndof ; ++j)
   {
      for (int d = 0; d < dimc; ++d)
      {
         elvect(d*ndof + j) += alpha * PointMat_bar(d,j);
      }
   }  
}

double MagneticEnergyIntegrator::GetElementEnergy(
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

#ifdef MFEM_THREAD_SAFE
   DenseMatrix curlshape(ndof,dimc), curlshape_dFt(ndof,dimc), M;
   Vector b_vec(dimc);
#else
   curlshape.SetSize(ndof,dimc);
   curlshape_dFt.SetSize(ndof,dimc);
   b_vec.SetSize(dimc);
#endif

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

   double fun = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);

      /// holds quadrature weight
      const double w = ip.weight * trans.Weight();

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
      const double b_mag = b_vec.Norml2() / trans.Weight();

      const double qp_en = calcMagneticEnergy(trans, ip, nu, b_mag);
      fun += qp_en * w;
   }
   return fun;
}

void MagneticEnergyIntegrator::AssembleElementVector(
   const FiniteElement &el,
   ElementTransformation &trans,
   const Vector &elfun,
   Vector &elvect)
{
   /// number of degrees of freedom
   int ndof = el.GetDof();
   int dim = el.GetDim();

   /// I believe this takes advantage of a 2D problem not having
   /// a properly defined curl? Need more investigation
   int dimc = (dim == 3) ? 3 : 1;

#ifdef MFEM_THREAD_SAFE
   DenseMatrix curlshape(ndof,dimc), curlshape_dFt(ndof,dimc), M;
   Vector b_vec(dimc);
#else
   curlshape.SetSize(ndof,dimc);
   curlshape_dFt.SetSize(ndof,dimc);
   b_vec.SetSize(dimc);
#endif

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

   elvect.SetSize(ndof);
   elvect = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);

      /// holds quadrature weight
      const double w = ip.weight / trans.Weight();

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
      const double b_mag = b_vec.Norml2() / trans.Weight();

      const double energy_dot = calcMagneticEnergyDot(trans, ip, nu, b_mag);

      b_vec *= energy_dot * w / b_mag;
      curlshape_dFt.AddMult(b_vec, elvect);
   }
   return;
}

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
   fd_val -= integrateBH(ir, trans, old_ip, lower_bound, upper_bound - delta);
   return fd_val / (2*delta);
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
      xi_bar -= (qp_en_bar * ip.weight * xi * nu->EvalStateDeriv(trans, old_ip, xi) / 
               pow(nu->Eval(trans, old_ip, xi), 2.0));
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
   ElementTransformation *trans = state.FESpace()->GetElementTransformation(element);
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

double BNormIntegrator::GetElementEnergy(
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

   double fun = 0.0;

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      b_vec = 0.0;
      const IntegrationPoint &ip = ir->IntPoint(i);

      trans.SetIntPoint(&ip);

      w = ip.weight * trans.Weight();

      if ( dim == 3 )
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

void BNormIntegrator::AssembleElementVector(
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

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      b_vec = 0.0;
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

void BNormdJdx::AssembleRHSElementVect(
   const FiniteElement &mesh_el,
   ElementTransformation &mesh_trans,
   Vector &elvect)
{
   /// get the proper element, transformation, and state vector
   Array<int> vdofs; Vector elfun; 
   int element = mesh_trans.ElementNo;
   const FiniteElement *el = state.FESpace()->GetFE(element);
   ElementTransformation *trans = state.FESpace()->GetElementTransformation(element);
   state.FESpace()->GetElementVDofs(element, vdofs);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
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
   state.GetSubVector(vdofs, elfun);

   int ndof = mesh_el.GetDof();
   int el_ndof = el->GetDof();
   int dim = el->GetDim();
   int dimc = (dim == 3) ? 3 : 1;
   elvect.SetSize(ndof*dimc);
   elvect = 0.0;
#ifdef MFEM_THREAD_SAFE
   DenseMatrix curlshape(ndof,dimc), curlshape_dFt(ndof,dimc), M;
   Vector b_vec(dimc);
#else
   curlshape.SetSize(el_ndof,dimc);
   curlshape_dFt.SetSize(el_ndof,dimc);
   b_vec.SetSize(dimc);
#endif
   DenseMatrix PointMat_bar(dimc, ndof);
   Vector b_hat(dimc);
   
   // cast the ElementTransformation
   IsoparametricTransformation &isotrans =
   dynamic_cast<IsoparametricTransformation&>(*trans);

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trans->SetIntPoint(&ip);
      double w = ip.weight;
      if ( dim == 3 )
      {
         el->CalcCurlShape(ip, curlshape);
         MultABt(curlshape, trans->Jacobian(), curlshape_dFt);
      }
      else
      {
         el->CalcCurlShape(ip, curlshape_dFt);
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
      for (int j = 0; j < ndof ; ++j)
      {
         for (int d = 0; d < dimc; ++d)
         {
            elvect(d*ndof + j) += w * PointMat_bar(d,j);
         }
      }
   }
}

double nuBNormIntegrator::GetElementEnergy(
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

   double fun = 0.0;

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      b_vec = 0.0;
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

      curlshape_dFt.AddMultTranspose(elfun, b_vec);
      fun += nu->Eval(trans, ip, b_vec.Norml2())*b_vec.Norml2() * w;
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

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      b_vec = 0.0;
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

      curlshape_dFt.AddMultTranspose(elfun, b_vec);
      double b_mag = b_vec.Norml2();

      /// temp_vec = curl(N_i) dot curl(A)
      temp_vec = 0.0;
      curlshape_dFt.Mult(b_vec, temp_vec);
      double nu_val = nu->Eval(trans, ip, b_mag);
      double dnu_dB = nu->EvalStateDeriv(trans, ip, b_mag);
      temp_vec *= (dnu_dB + nu_val/b_mag);
      temp_vec *= w;
      elvect += temp_vec;
   }
}

void nuBNormdJdx::AssembleRHSElementVect(
   const FiniteElement &mesh_el,
   ElementTransformation &mesh_trans,
   Vector &elvect)
{
   /// get the proper element, transformation, and state vector
   Array<int> vdofs; Vector elfun; 
   int element = mesh_trans.ElementNo;
   const FiniteElement *el = state.FESpace()->GetFE(element);
   ElementTransformation *trans = state.FESpace()->GetElementTransformation(element);
   state.FESpace()->GetElementVDofs(element, vdofs);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
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
   state.GetSubVector(vdofs, elfun);

   int ndof = mesh_el.GetDof();
   int el_ndof = el->GetDof();
   int dim = el->GetDim();
   int dimc = (dim == 3) ? 3 : 1;
   elvect.SetSize(ndof*dimc);
   elvect = 0.0;
#ifdef MFEM_THREAD_SAFE
   DenseMatrix curlshape(ndof,dimc), curlshape_dFt(ndof,dimc), M;
   Vector b_vec(dimc);
#else
   curlshape.SetSize(el_ndof,dimc);
   curlshape_dFt.SetSize(el_ndof,dimc);
   b_vec.SetSize(dimc);
#endif
   // DenseMatrix PointMat_bar(dimc, ndof);
   DenseMatrix PointMat_bar_1(dimc, ndof);
   DenseMatrix PointMat_bar_2(dimc, ndof);
   DenseMatrix PointMat_bar_3(dimc, ndof);
   // Vector DofVal(elfun.Size());

   Vector b_hat(dimc);
   
   // cast the ElementTransformation
   IsoparametricTransformation &isotrans =
   dynamic_cast<IsoparametricTransformation&>(*trans);

   for (int i = 0; i < ir->GetNPoints(); ++i)
   {
      PointMat_bar_1 = 0.0;
      PointMat_bar_2 = 0.0;
      PointMat_bar_3 = 0.0;
      b_vec = 0.0;
      b_hat = 0.0;
      const IntegrationPoint &ip = ir->IntPoint(i);

      trans->SetIntPoint(&ip);
      if ( dim == 3 )
      {
         el->CalcCurlShape(ip, curlshape);
         MultABt(curlshape, trans->Jacobian(), curlshape_dFt);
      }
      else
      {
         el->CalcCurlShape(ip, curlshape_dFt);
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
      PointMat_bar_2 *= - nu_val * b_vec.Norml2() / pow(isotrans.Weight(),2);



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
   //    // This is out of order because WeightRevDiff needs to scale PointMat_bar first
   //    isotrans.JacobianRevDiff(Jac_bar, PointMat_bar);
   //    // code to insert PointMat_bar into elvect;


      for (int j = 0; j < ndof ; ++j)
      {
         for (int d = 0; d < dimc; ++d)
         {
            elvect(d*ndof + j) += ip.weight * (PointMat_bar_1(d,j)
                                             + PointMat_bar_2(d,j)
                                             + PointMat_bar_3(d,j));
            // elvect(d*ndof + j) += PointMat_bar(d,j);
         }
      }
   }
}

double nuFuncIntegrator::GetElementEnergy(
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

   double fun = 0.0;

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      b_vec = 0.0;
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

      curlshape_dFt.AddMultTranspose(elfun, b_vec);
      const double b_mag = b_vec.Norml2() / trans.Weight();
      const double nu_val = nu->Eval(trans, ip, b_mag);
      fun += nu_val * w;
   }
   return fun;
}

void nuFuncIntegrator::AssembleRHSElementVect(
   const FiniteElement &mesh_el,
   ElementTransformation &mesh_trans,
   Vector &elvect)
{
   /// get the proper element, transformation, and state vector
   Array<int> vdofs; Vector elfun; 
   int element = mesh_trans.ElementNo;
   const FiniteElement *el = state->FESpace()->GetFE(element);
   ElementTransformation *trans = state->FESpace()->GetElementTransformation(element);
   state->FESpace()->GetElementVDofs(element, vdofs);

   const IntegrationRule *ir = NULL;
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
   state->GetSubVector(vdofs, elfun);

   int ndof = mesh_el.GetDof();
   int el_ndof = el->GetDof();
   int dim = el->GetDim();
   int dimc = (dim == 3) ? 3 : 1;
   elvect.SetSize(ndof*dimc);
   elvect = 0.0;
#ifdef MFEM_THREAD_SAFE
   DenseMatrix curlshape(ndof,dimc), curlshape_dFt(ndof,dimc), M;
   Vector b_vec(dimc);
#else
   curlshape.SetSize(el_ndof,dimc);
   curlshape_dFt.SetSize(el_ndof,dimc);
   b_vec.SetSize(dimc);
#endif
   DenseMatrix PointMat_bar(dimc, ndof);
   DenseMatrix PointMat_bar_2(dimc, ndof);
   Vector b_hat(dimc);
   
   // cast the ElementTransformation
   IsoparametricTransformation &isotrans =
   dynamic_cast<IsoparametricTransformation&>(*trans);

   for (int i = 0; i < ir->GetNPoints(); ++i)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      trans->SetIntPoint(&ip);
      if ( dim == 3 )
      {
         el->CalcCurlShape(ip, curlshape);
         MultABt(curlshape, trans->Jacobian(), curlshape_dFt);
      }
      else
      {
         el->CalcCurlShape(ip, curlshape_dFt);
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
      AddMult_a_VWt(1.0 / (b_vec.Norml2() * isotrans.Weight()), b_vec, b_hat, dBdJ);

      isotrans.WeightRevDiff(PointMat_bar);
      PointMat_bar *= - b_vec.Norml2() / pow(isotrans.Weight(),2);

      isotrans.JacobianRevDiff(dBdJ, PointMat_bar);
   
      PointMat_bar *= nu_deriv / isotrans.Weight();

      PointMat_bar_2 = 0.0;
      isotrans.WeightRevDiff(PointMat_bar_2);
      PointMat_bar_2 *= - nu_val / pow(isotrans.Weight(),2);

      for (int j = 0; j < ndof ; ++j)
      {
         for (int d = 0; d < dimc; ++d)
         {
            elvect(d*ndof + j) += ip.weight * (PointMat_bar(d,j) + PointMat_bar_2(d,j));
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

   const IntegrationRule *ir = &IntRules.Get(
            el.GetGeomType(), oa * el.GetOrder() + ob);

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

      double Q_bar = trans.Weight() * (psi * shape); // d(psi^T R)/dQ
      Q.Eval(V, trans, ip); // evaluate dQ/dA
      add(elvect, ip.weight * Q_bar, V, elvect);
   }
}

void setInput(DCLossFunctionalIntegrator &integ,
              const std::string &name,
              const MachInput &input)
{
   if (name == "current-density")
   {
      integ.current_density = input.getValue();
   }
   else if (name == "fill-factor")
   {
      integ.fill_factor = input.getValue();
   }
}

double DCLossFunctionalIntegrator::GetElementEnergy(
   const FiniteElement &el,
   ElementTransformation &trans,
   const Vector &elfun)
{
   /// I believe this takes advantage of a 2D problem not having
   /// a properly defined curl? Need more investigation
   int dim = el.GetDim();
   int dimc = (dim == 3) ? 3 : 1;

   const IntegrationRule *ir = NULL;
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

#ifdef MFEM_THREAD_SAFE
   double current_vec_data[3];
   Vector current_vec(current_vec_data, dimc);
#else
   current_vec.SetSize(dimc);
#endif

   double fun = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      trans.SetIntPoint(&ip);

      double w = ip.weight * trans.Weight();
      
      current.Eval(current_vec, trans, ip);
      current_vec *= current_density;
      const double current2 = current_vec*current_vec;
      if (current2 < 1e-14)
         continue;
      const double sigma_val = sigma.Eval(trans, ip);
      const double loss = current2 / sigma_val;
      fun += loss * fill_factor * w;
   }
   return fun;
}

void setInput(HybridACLossFunctionalIntegrator &integ,
              const std::string &name,
              const MachInput &input)
{
   if (name == "diam")
   {
      integ.diam = input.getValue();
   }
   else if (name == "frequency")
   {
      integ.freq = input.getValue();
   }
   else if (name == "fill-factor")
   {
      integ.fill_factor = input.getValue();
   }
}

double HybridACLossFunctionalIntegrator::GetElementEnergy(
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
   DenseMatrix curlshape(ndof,dimc), curlshape_dFt(ndof,dimc);
   Vector b_vec(dimc);
#else
   curlshape.SetSize(ndof,dimc);
   curlshape_dFt.SetSize(ndof,dimc);
   b_vec.SetSize(dimc);
#endif

   const IntegrationRule *ir = NULL;
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

   double fun = 0.0;

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      b_vec = 0.0;
      const IntegrationPoint &ip = ir->IntPoint(i);

      trans.SetIntPoint(&ip);

      w = ip.weight * trans.Weight();
      if ( dim == 3 )
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
      
      const double sigma_val = sigma.Eval(trans, ip);

      const double loss = std::pow(diam, 2) * sigma_val
                           * std::pow(2*M_PI * freq * b_mag, 2) / 128.0;
      fun += loss * fill_factor * w;
   }
   return fun;
}

double ForceIntegrator::GetElementEnergy(
   const FiniteElement &el,
   ElementTransformation &trans,
   const Vector &elfun)
{
   if (attrs.count(trans.Attribute) == 1)
   {
      return 0.0;
   }
   /// get the proper element, transformation, and v vector
   Array<int> vdofs; Vector vfun; 
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

   /// I believe this takes advantage of a 2D problem not having
   /// a properly defined curl? Need more investigation
   int dimc = (dim == 3) ? 3 : 1;

#ifdef MFEM_THREAD_SAFE
   DenseMatrix dshape(v_el.GetDof(), v_el.GetDim());
   DenseMatrix curlshape(ndof,dimc), curlshape_dFt(ndof,dimc);
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
   auto &isotrans = dynamic_cast<IsoparametricTransformation&>(trans);

   const IntegrationRule *ir = NULL;
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

   double fun = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);

      /// holds quadrature weight
      const double w = ip.weight * trans.Weight();
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
      const double b_vec_norm = b_vec.Norml2();
      const double b_mag = b_vec_norm / trans.Weight();

      // /// the following computes `\partial (||B||/|J|) / \partial X`
      // dBdX = 0.0;      
      // double weight_bar = -b_vec_norm / pow(trans.Weight(), 2.0);
      // isotrans.WeightRevDiff(dBdX);
      // dBdX *= weight_bar;

      // b_hat = 0.0;
      // curlshape.AddMultTranspose(elfun, b_hat);
      // DenseMatrix BB_hatT(3);
      // MultVWt(b_vec, b_hat, BB_hatT);
      // BB_hatT *= 1.0 / (trans.Weight() * b_vec_norm);
      // isotrans.JacobianRevDiff(BB_hatT, dBdX);

      // double force = 0.0;
      // for (int j = 0; j < v_el.GetDof() ; ++j)
      // {
      //    for (int d = 0; d < dimc; ++d)
      //    {
      //       force += dBdX(d, j) * dXds(j, d);
      //    }
      // }
      const double energy_dot = calcMagneticEnergyDot(trans, ip, nu, b_mag);
      // force *= energy_dot;
      double force = energy_dot;

      /// This and below to force2 has been verified
      // CalcDShape
      v_el.CalcDShape(ip, dshape);
      DenseMatrix JinvdJds(3);
      DenseMatrix dJds(3);
      MultAtB(dXds, dshape, dJds);
      Mult(trans.InverseJacobian(), dJds, JinvdJds);

      const double energy = calcMagneticEnergy(trans, ip, nu, b_mag);
      double force2 = energy * JinvdJds.Trace();
      fun -= (force + force2) * w;
      // fun -= force2 * w;
      // fun -= force * w;
   }
   return fun;
}

void ForceIntegrator::AssembleElementVector(
   const mfem::FiniteElement &el, 
   mfem::ElementTransformation &trans,
   const mfem::Vector &elfun,
   mfem::Vector &elvect)
{
   /// number of degrees of freedom
   const int ndof = el.GetDof();
   const int dim = el.GetDim();
   const int dimc = (dim == 3) ? 3 : 1;

   elvect.SetSize(ndof);
   elvect = 0.0;
   if (attrs.count(trans.Attribute) == 1)
   {
      return;
   }

   /// get the proper element, transformation, and v vector
   Array<int> vdofs; Vector vfun; 
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
   DenseMatrix curlshape(ndof,dimc), curlshape_dFt(ndof,dimc);
   DenseMatrix dBdX(v_el.GetDim(), v_el.GetDof());
   Vector b_vec(dimc), b_hat(dimc);
   Vector dBmdA(ndof);
#else
   dshape.SetSize(v_el.GetDof(), v_el.GetDim());
   curlshape.SetSize(ndof, dimc);
   curlshape_dFt.SetSize(ndof, dimc);
   dBdX.SetSize(v_el.GetDim(), v_el.GetDof());
   b_vec.SetSize(dimc);
   b_hat.SetSize(dimc);
   dBmdA.SetSize(ndof);
#endif

   // cast the ElementTransformation
   auto &isotrans = dynamic_cast<IsoparametricTransformation&>(trans);

   const IntegrationRule *ir = NULL;
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

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      /// forward pass
      const IntegrationPoint &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);

      /// holds quadrature weight
      const double w = ip.weight * trans.Weight();
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
      const double b_vec_norm = b_vec.Norml2();
      const double b_mag = b_vec_norm / trans.Weight();

      // /// the following computes `\partial (||B||/|J|) / \partial X`
      // dBdX = 0.0;      
      // double weight_bar = -b_vec_norm / pow(trans.Weight(), 2.0);
      // isotrans.WeightRevDiff(dBdX);
      // dBdX *= weight_bar;

      // b_hat = 0.0;
      // curlshape.AddMultTranspose(elfun, b_hat);
      // DenseMatrix BB_hatT(3);
      // MultVWt(b_vec, b_hat, BB_hatT);
      // BB_hatT *= 1.0 / (trans.Weight() * b_vec_norm);
      // isotrans.JacobianRevDiff(BB_hatT, dBdX);

      // double force = 0.0;
      // for (int j = 0; j < v_el.GetDof() ; ++j)
      // {
      //    for (int d = 0; d < dimc; ++d)
      //    {
      //       force += dBdX(d, j) * dXds(j, d);
      //    }
      // }
      const double energy_dot = calcMagneticEnergyDot(trans, ip, nu, b_mag);
      // force *= energy_dot;
      double force = energy_dot;

      v_el.CalcDShape(ip, dshape);
      DenseMatrix JinvdJds(3);
      DenseMatrix dJds(3);
      MultAtB(dXds, dshape, dJds);
      Mult(trans.InverseJacobian(), dJds, JinvdJds);

      const double energy = calcMagneticEnergy(trans, ip, nu, b_mag);
      double force2 = energy * JinvdJds.Trace();
      
      // fun -= (force + force2) * w;

      /// start reverse pass
      double fun_bar = 1.0;
      double force_bar = 0.0;
      double force2_bar = 0.0;

      /// fun -= (force + force2) * w;
      force_bar -= fun_bar * w;
      force2_bar -= fun_bar * w;

      /// double force2 = energy * JinvdJds.Trace();
      double energy_bar = 0.0;
      /// double JinvdJdsTrace_bar = 0.0; // not needed for state derivative
      energy_bar += JinvdJds.Trace() * force2_bar;
      /// JinvdJdsTrace_bar += JinvdJdsTrace_bar + energy * force2_bar; // not needed for state derivative

      /// const double energy = calcMagneticEnergy(trans, ip, nu, b_mag);
      double b_mag_bar = 0.0;
      b_mag_bar += energy_bar * energy_dot;

      /// force *= energy_dot;
      /// force = energy_dot;
      double energy_dot_bar = force_bar;
      // double energy_dot_bar = force_bar * force;
      force_bar *= energy_dot;

      /// const double energy_dot = calcMagneticEnergyDot(trans, ip, nu, b_mag);
      auto energy_double_dot = calcMagneticEnergyDoubleDot(trans, ip, nu, b_mag);
      b_mag_bar += energy_dot_bar * energy_double_dot;

      // DenseMatrix dBdX_bar(v_el.GetDim(), v_el.GetDof());
      // dBdX_bar = 0.0; // same shape as dBdX
      // for (int j = 0; j < v_el.GetDof() ; ++j)
      // {
      //    for (int d = 0; d < dimc; ++d)
      //    {
      //       // force += dBdX(d, j) * dXds(j, d);
      //       dBdX_bar(d, j) += force_bar * dXds(j, d);
      //    }
      // }

      // /// isotrans.JacobianRevDiff(BB_hatT, dBdX);
      // DenseMatrix BB_hatT_bar(dimc);
      // BB_hatT_bar = 0.0;
      // /// isotrans.JacobianRevDiffRevDiff(dBdX_bar, BB_hatT_bar);
      // /// this is from looking at code for JacobianRevDiff and replacing it with
      // /// multiplication by `dshape`
      // AddMult(dBdX_bar, dshape, BB_hatT_bar);

      // /// BB_hatT *= 1.0 / (trans.Weight() * b_vec_norm);
      // /// BB_hatT_temp = BB_hatT / (trans.Weight() * b_vec_norm);
      double b_vec_norm_bar = 0.0;
      // for (int j = 0; j < dimc ; ++j)
      // {
      //    for (int d = 0; d < dimc; ++d)
      //    {
      //       b_vec_norm_bar += BB_hatT_bar(j,d) * BB_hatT(j, d);
      //    }
      // }
      // b_vec_norm_bar *= - 1.0 / (trans.Weight() * pow(b_vec_norm, 2));

      // /// MultVWt(b_vec, b_hat, BB_hatT);
      // Vector b_bar(dimc), b_hat_bar(dimc);
      // BB_hatT_bar.Mult(b_hat, b_bar);
      // BB_hatT_bar.Mult(b_vec, b_hat_bar);

      // /// curlshape.AddMultTranspose(elfun, b_hat);
      // curlshape.AddMult(b_hat_bar, elvect);

      // /// dBdX *= weight_bar;
      // dBdX_bar *= weight_bar;

      // /// isotrans.WeightRevDiff(dBdX); // no input from state
      // /// double weight_bar = -b_vec_norm / pow(trans.Weight(), 2.0);
      // b_vec_norm_bar +=  -weight_bar / pow(trans.Weight(), 2);

      /// const double b_mag = b_vec_norm / trans.Weight();   
      b_vec_norm_bar += b_mag_bar / trans.Weight();

      /// const double b_vec_norm = b_vec.Norml2();
      auto b_vec_bar = b_vec;
      b_vec_bar *= b_vec_norm_bar / b_vec_norm;

      /// curlshape_dFt.AddMultTranspose(elfun, b_vec);
      curlshape_dFt.AddMult(b_vec_bar, elvect);
   }
   return;
}

double dBdsIntegrator::GetElementEnergy(
   const FiniteElement &el,
   ElementTransformation &trans,
   const Vector &elfun)
{
   /// get the proper element, transformation, and v vector
   Array<int> vdofs; Vector vfun; 
   int element = trans.ElementNo;
   const auto &v_el = *v.FESpace()->GetFE(element);
   v.FESpace()->GetElementVDofs(element, vdofs);
   v.GetSubVector(vdofs, vfun);
   DenseMatrix dXds(vfun.GetData(), v_el.GetDof(), v_el.GetDim());

   /// number of degrees of freedom
   int ndof = el.GetDof();
   int dim = el.GetDim();

   /// I believe this takes advantage of a 2D problem not having
   /// a properly defined curl? Need more investigation
   int dimc = (dim == 3) ? 3 : 1;

#ifdef MFEM_THREAD_SAFE
   DenseMatrix dshape(v_el.GetDof(), v_el.GetDim());
   DenseMatrix curlshape(ndof,dimc), curlshape_dFt(ndof,dimc);
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
   auto &isotrans = dynamic_cast<IsoparametricTransformation&>(trans);

   const IntegrationRule *ir = NULL;
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

   double fun = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);

      /// holds quadrature weight
      const double w = ip.weight * trans.Weight();
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
      const double b_vec_norm = b_vec.Norml2();

      /// the following computes `\partial (||B||/|J|) / \partial J`
      b_hat = 0.0;
      curlshape.AddMultTranspose(elfun, b_hat);
      DenseMatrix BB_hatT(3);
      MultVWt(b_vec, b_hat, BB_hatT);

      DenseMatrix dBmdJ(dimc);
      DenseMatrix inv_jac_transposed = trans.InverseJacobian();
      inv_jac_transposed.Transpose();

      BB_hatT = 0.0;
      inv_jac_transposed = 0.0; // but changing it here does...
      Add(1.0 / (trans.Weight() * b_vec_norm), BB_hatT,
          -b_vec_norm / trans.Weight(), inv_jac_transposed, dBmdJ);

      /// and then contracts with \partial J / \partial X
      dBdX = 0.0;
      isotrans.JacobianRevDiff(dBmdJ, dBdX);

      double dBds = 0.0;
      for (int j = 0; j < v_el.GetDof() ; ++j)
      {
         for (int d = 0; d < dimc; ++d)
         {
            dBds += dBdX(d, j) * dXds(j, d);
         }
      }

      fun -= dBds * w;
   }
   return fun;
}

void dBdsIntegrator::AssembleElementVector(
   const mfem::FiniteElement &el, 
   mfem::ElementTransformation &trans,
   const mfem::Vector &elfun,
   mfem::Vector &elvect)
{
   /// number of degrees of freedom
   const int ndof = el.GetDof();
   const int dim = el.GetDim();
   const int dimc = (dim == 3) ? 3 : 1;

   elvect.SetSize(ndof);
   elvect = 0.0;

   /// get the proper element, transformation, and v vector
   Array<int> vdofs; Vector vfun; 
   const int element = trans.ElementNo;
   const auto &v_el = *v.FESpace()->GetFE(element);
   v.FESpace()->GetElementVDofs(element, vdofs);
   v.GetSubVector(vdofs, vfun);
   DenseMatrix dXds(vfun.GetData(), v_el.GetDof(), v_el.GetDim());

#ifdef MFEM_THREAD_SAFE
   DenseMatrix dshape(v_el.GetDof(), v_el.GetDim());
   DenseMatrix curlshape(ndof,dimc), curlshape_dFt(ndof,dimc);
   DenseMatrix dBdX(v_el.GetDim(), v_el.GetDof());
   Vector b_vec(dimc), b_hat(dimc);
   Vector dBmdA(ndof);
#else
   dshape.SetSize(v_el.GetDof(), v_el.GetDim());
   curlshape.SetSize(ndof, dimc);
   curlshape_dFt.SetSize(ndof, dimc);
   dBdX.SetSize(v_el.GetDim(), v_el.GetDof());
   b_vec.SetSize(dimc);
   b_hat.SetSize(dimc);
   dBmdA.SetSize(ndof);
#endif

   // cast the ElementTransformation
   auto &isotrans = dynamic_cast<IsoparametricTransformation&>(trans);

   const IntegrationRule *ir = NULL;
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

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      /// forward pass
      const IntegrationPoint &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);

      /// holds quadrature weight
      const double w = ip.weight * trans.Weight();
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
      const double b_vec_norm = b_vec.Norml2();
      const double b_mag = b_vec_norm / trans.Weight();

      /// the following computes `\partial (||B||/|J|) / \partial J`
      b_hat = 0.0;
      curlshape.AddMultTranspose(elfun, b_hat);
      DenseMatrix BB_hatT(3);
      MultVWt(b_vec, b_hat, BB_hatT);

      DenseMatrix dBmdJ(dimc);
      DenseMatrix inv_jac_transposed = trans.InverseJacobian();
      inv_jac_transposed.Transpose();

      BB_hatT = 0.0;
      inv_jac_transposed = 0.0; // changing this does not affect derivative output...
      Add(1.0 / (trans.Weight() * b_vec_norm), BB_hatT,
          -b_vec_norm / trans.Weight(), inv_jac_transposed, dBmdJ);

      dBmdJ.Print();
      /// and then contracts with \partial J / \partial X
      dBdX = 0.0;
      isotrans.JacobianRevDiff(dBmdJ, dBdX);

      double dBds = 0.0;
      for (int j = 0; j < v_el.GetDof() ; ++j)
      {
         for (int d = 0; d < dimc; ++d)
         {
            dBds += dBdX(d, j) * dXds(j, d);
         }
      }
      
      // fun -= dBds * w;

      /// start reverse pass
      double fun_bar = 1.0;
      double dBds_bar = 0.0;

      /// fun -= dBds * w;
      dBds_bar -= fun_bar * w;

      DenseMatrix dBdX_bar(v_el.GetDim(), v_el.GetDof());
      dBdX_bar = 0.0; // same shape as dBdX
      for (int j = 0; j < v_el.GetDof() ; ++j)
      {
         for (int d = 0; d < dimc; ++d)
         {
            // dBds += dBdX(d, j) * dXds(j, d);
            dBdX_bar(d, j) += dBds_bar * dXds(j, d);
         }
      }

      /// isotrans.JacobianRevDiff(dBmdJ, dBdX);
      DenseMatrix dBmdJ_bar(dimc);
      dBmdJ_bar = 0.0;
      /// this is from looking at code for JacobianRevDiff and replacing it with
      /// multiplication by `dshape`
      v_el.CalcDShape(ip, dshape);
      AddMult(dBdX_bar, dshape, dBmdJ_bar);

      /// Add(1.0 / (trans.Weight() * b_vec_norm), BB_hatT,
      ///     -b_vec_norm / trans.Weight(), inv_jac_transposed, dBmdJ);

      DenseMatrix BB_hatT_bar(dBmdJ_bar);
      BB_hatT_bar *= -1.0 / (trans.Weight() * b_vec_norm);

      double b_vec_norm_bar = 0.0;
      for (int j = 0; j < dimc ; ++j)
      {
         for (int d = 0; d < dimc; ++d)
         {
            b_vec_norm_bar -= dBmdJ_bar(j,d) * 
                              (BB_hatT(j, d) / (trans.Weight() * pow(b_vec_norm, 2)
                              + inv_jac_transposed(j,d) / trans.Weight()));
         }
      }

      /// MultVWt(b_vec, b_hat, BB_hatT);
      Vector b_bar(dimc), b_hat_bar(dimc);
      BB_hatT_bar.Mult(b_hat, b_bar);
      BB_hatT_bar.Mult(b_vec, b_hat_bar);

      /// curlshape.AddMultTranspose(elfun, b_hat);
      curlshape.AddMult(b_hat_bar, elvect);
      // elvect.Print();

      /// const double b_vec_norm = b_vec.Norml2();
      auto b_vec_bar = b_vec;
      b_vec_bar *= b_vec_norm_bar / b_vec_norm;

      /// curlshape_dFt.AddMultTranspose(elfun, b_vec);
      curlshape_dFt.AddMult(b_vec_bar, elvect);
      // elvect.Print();
   }
   return;
}

} // namespace mach
