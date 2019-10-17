#include "coefficient.hpp"

using namespace mfem;

namespace mach
{

double MeshDependentCoefficient::Eval(ElementTransformation &trans,
                                      const IntegrationPoint &ip,
                                      const double state)
{
   // given the attribute, extract the coefficient value from the map
   std::map<const int, Coefficient*>::iterator it;
   int this_att = trans.Attribute;
   Coefficient *coeff;
	double value;
   it = material_map.find(this_att);
   if (it != material_map.end())
   {
      coeff = it->second;
      value = Eval(coeff, trans, ip, state);
   }
   else
   {
      value = 0.0; // avoid compile warning
      std::cerr << "MeshDependentCoefficient attribute " << it->first
                << " not found" << std::endl;
      mfem_error();
   }
   return value;
}

double MeshDependentCoefficient::EvalStateDeriv(ElementTransformation &trans,
                                      				const IntegrationPoint &ip,
                                                const double state)
{
   // given the attribute, extract the coefficient value from the map
   std::map<const int, Coefficient*>::iterator it;
   int this_att = trans.Attribute;
	double value;
   it = material_map.find(this_att);
   if (it != material_map.end())
   {
      Coefficient *coeff = it->second;
		value = EvalStateDeriv(coeff, trans, ip, state);
   }
   else
   {
      value = 0.0; // avoid compile warning
      std::cerr << "MeshDependentCoefficient attribute " << it->first
                << " not found" << std::endl;
      mfem_error();
   }
   return value;
}

double ReluctivityCoefficient::Eval(ElementTransformation &trans,
												const IntegrationPoint &ip,
                                    const double state)
{
	if (Bmodel)
   {
      return ((*Bmodel)(state));
   }
   else
   {
		const double temp = temperature_GF->GetValue(trans.ElementNo, ip);
      return (*BTmodel)(state, temp);
   }
}

double ReluctivityCoefficient::EvalStateDeriv(ElementTransformation &trans,
												          const IntegrationPoint &ip,
                                              const double state)
{
   mfem_error("Not yet implemented!");
   return 0.0;
}

} // namespace mach