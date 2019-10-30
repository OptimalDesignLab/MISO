#include "electromag_integ.hpp"

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

      /////////////////////////////////////////////////////////////////////////
      /// calculate second term of Jacobian
      /////////////////////////////////////////////////////////////////////////

      /// TODO - is this thread safe?
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

void MagnetizationIntegrator::AssembleElementVector(
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
   Vector b_vec(dimc) mag_vec(dimc);
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

      /// TODO - is this thread safe?
      /// calculate curl(N_i) dot curl(A), need to store in a DenseMatrix so we
      /// can take outer product of result to generate matrix
      temp_vec = 0.0;
      curlshape_dFt.Mult(b_vec, temp_vec);
      DenseMatrix temp_matrix(temp_vec.GetData(), ndof, 1);

      mag_vec = 0.0;
      mag->Eval(mag_vec, trans, ip);

      temp_vec2 = 0.0;
      curlshape_dFt.Mult(mag_vec, temp_vec);
      DenseMatrix temp_matrix2(temp_vec.GetData(), ndof, 1);

      /// evaluate the derivative of the material model with respect to the
      /// norm of the grid function associated with the model at the point
      /// defined by ip, and scale by integration point weight
      double nu_deriv = nu->EvalStateDeriv(trans, ip, b_mag);
      nu_deriv *= w;
      nu_deriv /= b_mag;

      /// add second term to elmat
      AddMult_a_ABt(nu_deriv, temp_matrix, temp_matrix2, elmat);
   }
}

} // namespace mach