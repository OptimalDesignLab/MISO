#include "electromag_integ.hpp"

using namespace mfem;
using namespace std;

namespace mach
{

void CurlCurlNLFIntegrator::AssembleElementVector(
    const FiniteElement &el,
    ElementTransformation &Trans,
    const Vector &elfun, Vector &elvect)
{
	// number of degrees of freedom
	int ndof = el.GetDof();
   int dim = el.GetDim();

	// I believe this takes advantage of a 2D problem not having 
	// a properly defined curl? Need more investigation
   int dimc = (dim == 3) ? 3 : 1;
   double w;

#ifdef MFEM_THREAD_SAFE
   DenseMatrix curlshape(ndof,dimc), curlshape_dFt(ndof,dimc), M;
   Vector tempVec(ndof);
#else
   curlshape.SetSize(ndof,dimc);
   curlshape_dFt.SetSize(ndof,dimc);
   tempVec.SetSize(ndof);
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
      tempVec = 0.0;
      const IntegrationPoint &ip = ir->IntPoint(i);

      Trans.SetIntPoint (&ip);

      w = ip.weight / Trans.Weight();

      if ( dim == 3 )
      {
         el.CalcCurlShape(ip, curlshape);
         MultABt(curlshape, Trans.Jacobian(), curlshape_dFt);
      }
      else
      {
         el.CalcCurlShape(ip, curlshape_dFt);
      }

      curlshape_dFt.AddMultTranspose(elfun, tempVec);
      curlshape_dFt.AddMult(tempVec, elvect);

      tempVec = 0.0;
      model->Eval(Trans, elfun, tempVec);
      tempVec *= w;
      multiplyElementwise(tempVec, elvect);      
   }
}


} // namespace mach