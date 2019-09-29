#include "coefficient.hpp"

using namespace mfem;

namespace mach
{

double ReluctivityCoefficient::Eval(ElementTransformation &trans,
												const IntegrationPoint &ip)
{
	Vector B;
   magnetic_flux_GF->GetVectorValue(trans.ElementNo, ip, B);

	if (Bmodel)
   {
      return ((*Bmodel)(B));
   }
   else
   {
		double temp;
		temperature_GF->GetValue(trans.ElementNo, ip, temp);
      return (*BTmodel)(B, temp);
   }
}


} // namespace mach