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
using namespace std;

namespace mach
{

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
      b_vec = 0.0;
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
      double model_val = model->Eval(trans, ip, b_vec.Norml2());
      model_val *= w;
      b_vec *= model_val;

      // std::cout << "above curl curl add mult\n";
      curlshape_dFt.AddMult(b_vec, elvect);
      // std::cout << "below curl curl add mult\n";
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
         DenseMatrix temp_matrix(temp_vec.GetData(), ndof, 1);

         /// evaluate the derivative of the material model with respect to the
         /// norm of the grid function associated with the model at the point
         /// defined by ip, and scale by integration point weight
         double model_deriv = model->EvalStateDeriv(trans, ip, b_mag);
         model_deriv *= w;
         model_deriv /= b_mag;
      
         /// add second term to elmat
         AddMult_a_AAt(model_deriv, temp_matrix, elmat);
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
      double model_val = nu->Eval(trans, ip, b_vec.Norml2());
      model_val *= w;

      double el_en = b_vec*b_vec;
      el_en *= 0.5 * model_val;

      fun += el_en;
   }
   return fun;
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

   const IntegrationRule *ir = IntRule;
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
      int order;
      if (el.Space() == FunctionSpace::Pk)
      {
         order = 2*el.GetOrder() - 2;
      }
      else
      {
         order = 2*el.GetOrder();
      }

      // segment_ir = &IntRules.Get(Geometry::Type::SEGMENT, 2*(order+1));
      segment_ir = &IntRules.Get(Geometry::Type::SEGMENT, 12);
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
      double nu_val = nu->Eval(trans, ip, b_vec.Norml2());

      double lower_bound = 0.0;
      double upper_bound = nu_val * b_vec.Norml2();
      /// compute int_0^{\nu*B} \frac{H}{\nu} dH
      // double qp_en = 0.0;
      // for (int j = 0; j < segment_ir->GetNPoints(); j++)
      // {
      //    const IntegrationPoint &segment_ip = segment_ir->IntPoint(j);
      //    double xi = segment_ip.x * (upper_bound - lower_bound);
      //    qp_en += segment_ip.weight * xi / nu->Eval(trans, ip, xi);
      // }
      // qp_en *= (upper_bound - lower_bound);
      double qp_en = integrateBH(segment_ir, trans, ip,
                                 lower_bound, upper_bound);

      // double fd_int = FDintegrateBH(segment_ir, trans, ip, lower_bound, upper_bound);
      // double ad_int = RevADintegrateBH(segment_ir, trans, ip, lower_bound, upper_bound);
      // double b_mag = b_vec.Norml2();
      // std::cout << "Finite differenced integral: " << fd_int << "\n";
      // std::cout << "AD integral: " << ad_int << "\n";
      // std::cout << "B mag: " << b_mag << "\n";

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

   const IntegrationRule *ir = IntRule;
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
      int order;
      if (el.Space() == FunctionSpace::Pk)
      {
         order = 2*el.GetOrder() - 2;
      }
      else
      {
         order = 2*el.GetOrder();
      }

      // segment_ir = &IntRules.Get(Geometry::Type::SEGMENT, 2*(order+1));
      segment_ir = &IntRules.Get(Geometry::Type::SEGMENT, 12);
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
      double nu_val = nu->Eval(trans, ip, b_mag);
      double dnu_dB = nu->EvalStateDeriv(trans, ip, b_mag);

      /// temp_vec = curl(N_i) dot curl(A)
      temp_vec = 0.0;
      curlshape_dFt.Mult(b_vec, temp_vec);
      // temp_vec *= (nu->EvalStateDeriv(trans, ip, b_mag) + nu_val / b_mag);
      // temp_vec *= nu_val;
      double dwp_dh = RevADintegrateBH(segment_ir, trans, ip,
                                       0, nu_val * b_mag);
      temp_vec *= dwp_dh*(dnu_dB + nu_val/b_mag); // (dnu_dB + nu_val/b_mag)
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

void MagneticCoenergydJdx::AssembleRHSElementVect(
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
   int order = 2*el->GetOrder() + trans->OrderW();
   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      ir = &IntRules.Get(el->GetGeomType(), order);
   }
   state.GetSubVector(vdofs, elfun);

   int dof = mesh_el.GetDof(), dim = el->GetDim();
   elvect.SetSize(dof*dim);
   elvect = 0.0;
   DenseMatrix PointMat_bar(dim, dof);
   DenseMatrix vshape(dof, dim);
   Vector DofVal(elfun.Size());
   
   // cast the ElementTransformation
   IsoparametricTransformation &isotrans =
   dynamic_cast<IsoparametricTransformation&>(*trans);

   const int attr = mesh_trans.Attribute;
   for (int i = 0; i < ir->GetNPoints(); ++i)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      trans->SetIntPoint(&ip);
      el->CalcVShape(ip, vshape);

      isotrans.WeightRevDiff(PointMat_bar);
      for (int j = 0; j < dof ; ++j)
      {
         for (int d = 0; d < dim; ++d)
         {
            elvect(d*dof + j) += PointMat_bar(d,j);
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
      fun += b_vec.Norml2() * w;
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
      double b_mag = b_vec.Norml2();

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
   int order = 2*el->GetOrder() + trans->OrderW();
   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
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
   Vector DofVal(elfun.Size());
   
   // cast the ElementTransformation
   IsoparametricTransformation &isotrans =
   dynamic_cast<IsoparametricTransformation&>(*trans);

   const int attr = mesh_trans.Attribute;
   for (int i = 0; i < ir->GetNPoints(); ++i)
   {
      b_vec = 0.0;
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
      isotrans.WeightRevDiff(PointMat_bar);
      curlshape_dFt.AddMultTranspose(elfun, b_vec);

      for (int j = 0; j < ndof ; ++j)
      {
         for (int d = 0; d < dimc; ++d)
         {
            elvect(d*ndof + j) += PointMat_bar(d,j);
         }
      }
      elvect *= b_vec.Norml2() * ip.weight;
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

// ForceIntegrator::ForceIntegrator(AbstractSolver *_solver,
//                                  std::unordered_set<int> _regions,
//                                  std::unordered_set<int> _free_regions,
//                                  StateCoefficient *_nu,
//                                  Vector _dir)
//    : solver(_solver), regions(_regions), free_regions(_free_regions), nu(_nu),
//      dir(_dir)
// {
// #ifndef MFEM_USE_PUMI
//    throw MachException("ForceIntegrator::ForceIntegrator()\n"
//                         "\tusing ForceIntegrator requires PUMI!\n");
// }

// double ForceIntegrator::GetElementEnergy(const FiniteElement &el,
//                                          ElementTransformation &trans,
//                                          const Vector &elfun)
// {
//    throw MachException("ForceIntegrator::ForceIntegrator()\n"
//                         "\tusing ForceIntegrator requires PUMI!\n");
// }
// #else
//    /// TODO: Call pumi APIs to get a list of mesh face indices that are on the
//    ///       boundary of the regions given in regions

//    // std::unordered_set<int> face_list;

//    apf::Mesh2 *pumi_mesh = solver->getPumiMesh();
//    /// get the underlying gmi model
//    auto *model = pumi_mesh->getModel();

//    /// find the model faces that define the interface between moving and fixed
//    /// parts
//    for (auto &free_region_tag : free_regions)
//    {
//       auto *free_region = gmi_find(model, 3, free_region_tag);
//       auto *adjacent_faces = gmi_adjacent(model, free_region, 2);
//       for (int i = 0; i < adjacent_faces->n; ++i)
//       {
//          auto adjacent_face = adjacent_faces->e[i];
//          for (auto &moving_region_tag : regions)
//          {
//             auto *moving_region = gmi_find(model, 3, moving_region_tag);
//             if (gmi_is_in_closure_of(model, adjacent_face, moving_region))
//             {
//                int face_tag = gmi_tag(model, adjacent_face);
//                face_list.insert(face_tag);
//             }
//          }
//       }
//       gmi_free_set(adjacent_faces);
//    }

//    /// loop over all mesh faces
//    apf::MeshEntity *e;
//    apf::MeshIterator* ent_it = pumi_mesh->begin(2);
//    while ((e = pumi_mesh->iterate(ent_it)))
//    {
//       int e_tag = gmi_tag(model, (gmi_ent*)pumi_mesh->toModel(e));
//       auto search = face_list.find(e_tag);
//       if (search != face_list.end())
//       {
//          auto r_1 = pumi_mesh->getUpward(e,0);
//          int r_1_tag = gmi_tag(model, (gmi_ent*)pumi_mesh->toModel(r_1));
//          auto search_tet = free_regions.find(r_1_tag);
//          if (search_tet != free_regions.end())
//          {
//             el_ids.insert(apf::getMdsIndex(pumi_mesh, r_1));
//          }
//          else
//          {
//             auto r_2 = pumi_mesh->getUpward(e,1);
//             el_ids.insert(apf::getMdsIndex(pumi_mesh, r_2));
//          }
//       }
//    }
//    pumi_mesh->end(ent_it);
// }

// double ForceIntegrator::GetElementEnergy(const FiniteElement &el,
//                                          ElementTransformation &trans,
//                                          const Vector &elfun)
// {
//    /// number of degrees of freedom
//    int ndof = el.GetDof();
//    int dim = el.GetDim();

//    /// I believe this takes advantage of a 2D problem not having
//    /// a properly defined curl? Need more investigation
//    int dimc = (dim == 3) ? 3 : 1;

//    /// holds quadrature weight
//    double w;

// #ifdef MFEM_THREAD_SAFE
//    DenseMatrix curlshape(ndof,dimc), curlshape_dFt(ndof,dimc), M;
//    Vector b_vec(dimc);
// #else
//    curlshape.SetSize(ndof,dimc);
//    curlshape_dFt.SetSize(ndof,dimc);
//    b_vec.SetSize(dimc);
// #endif

//    const IntegrationRule *ir = IntRule;
//    if (ir == NULL)
//    {
//       int order;
//       if (el.Space() == FunctionSpace::Pk)
//       {
//          order = 2*el.GetOrder() - 2;
//       }
//       else
//       {
//          order = 2*el.GetOrder();
//       }

//       ir = &IntRules.Get(el.GetGeomType(), order);
//    }

//    double fun = 0.0;

//    for (int i = 0; i < ir->GetNPoints(); i++)
//    {
//       b_vec = 0.0;
//       const IntegrationPoint &ip = ir->IntPoint(i);

//       trans.SetIntPoint(&ip);

//       w = ip.weight / trans.Weight();

//       if ( dim == 3 )
//       {
//          el.CalcCurlShape(ip, curlshape);
//          MultABt(curlshape, trans.Jacobian(), curlshape_dFt);
//       }
//       else
//       {
//          el.CalcCurlShape(ip, curlshape_dFt);
//       }

//       curlshape_dFt.AddMultTranspose(elfun, b_vec);
//       double model_val = nu->Eval(trans, ip, b_vec.Norml2());
//       model_val *= w;

//       double el_en = b_vec*b_vec;
//       el_en *= 0.5 * model_val;

//       fun += el_en;
//    }
//    return fun;
// }
// #endif

// double ForceIntegrator::GetFaceEnergy(const FiniteElement &el1,
//                                       const FiniteElement &el2,
//                                       FaceElementTransformations &Tr,
//                                       const Vector &elfun)
// {

// }

// double VWTorqueIntegrator::GetElementEnergy(const FiniteElement &el,
//                                             ElementTransformation &Tr,
//                                             const Vector &elfun)
// {

//    return 0.0;
// }

} // namespace mach
