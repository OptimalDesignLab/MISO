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
	// number of degrees of freedom
	int ndof = el.GetDof();
   int dim = el.GetDim();

	// I believe this takes advantage of a 2D problem not having 
	// a properly defined curl? Need more investigation
   int dimc = (dim == 3) ? 3 : 1;

   // holds quadrature weight
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
      curlshape_dFt.AddMult(b_vec, elvect);

      double model_val = 0.0;
      model->Eval(trans, b_vec.Norml2(), model_val);
      model_val *= w;
      elvect *= model_val;   
   }
}

void CurlCurlNLFIntegrator::AssembleElementGrad(
    const mfem::FiniteElement &el,
    mfem::ElementTransformation &trans,
    const mfem::Vector &elfun,
    mfem::DenseMatrix &elmat)
{
   // number of degrees of freedom
	int ndof = el.GetDof();
   int dim = el.GetDim();

	// I believe this takes advantage of a 2D problem not having 
	// a properly defined curl? Need more investigation
   int dimc = (dim == 3) ? 3 : 1;

   // holds quadrature weight
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

      /////////////////////////////////////////////////////////////////////////
      /// calculate first term of Jacobian
      /////////////////////////////////////////////////////////////////////////

      /// calculate B = curl(A)
      curlshape_dFt.AddMultTranspose(elfun, b_vec);

      /// evaluate material model with norm of b_vec
      double model_val = 0.0;
      model->Eval(trans, b_vec.Norml2(), model_val);

      /// multiply material value by integration weight
      model_val *= w;

      /// add first term to elmat
      AddMult_a_AAt(model_val, curlshape_dFt, elmat);

      /////////////////////////////////////////////////////////////////////////
      /// calculate second term of Jacobian
      /////////////////////////////////////////////////////////////////////////

      // calculate curl(N_i) dot curl(A), need to store in a DenseMatrix so we
      // can take outer product of result to generate matrix
      temp_vec = 0.0;
      curlshape_dFt.AddMult(b_vec, temp_vec);
      DenseMatrix temp_matrix(temp_vec.GetData(), ndof, 1);

      // evaluate derivative of material model with norm of b_vec
      double model_deriv = 0.0;
      model->EvalDerivState(trans, b_vec.Norml2(), model_deriv);

      // scale derivative by weight and devide by norm of b_vec
      model_deriv *= w;
      model_deriv /= b_vec.Norml2();

      // add second term to elmat
      AddMult_a_AAt(model_deriv, temp_matrix, elmat);
   }
}

} // namespace mach