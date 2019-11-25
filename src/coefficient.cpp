#include "coefficient.hpp"

using namespace mfem;

namespace mach
{

double MeshDependentCoefficient::Eval(ElementTransformation &trans,
                                      const IntegrationPoint &ip,
                                      const double state)
{
   // given the attribute, extract the coefficient value from the map
   int this_att = trans.Attribute;
   Coefficient *coeff;
	double value;
   auto it = material_map.find(this_att);
   if (it != material_map.end())
   {
      coeff = it->second.get();
      value = Eval(coeff, trans, ip, state);
   }
   else if (default_coeff)
   {
      value = Eval(default_coeff.get(), trans, ip, state);
   }
   else // if attribute not found and no default set, evaluate to zero
   {
      value = 0.0;
   }
   // std::cout << "nu val in eval: " << value << "\n";
   return value;
}

double MeshDependentCoefficient::EvalStateDeriv(ElementTransformation &trans,
                                      				const IntegrationPoint &ip,
                                                const double state)
{
   // given the attribute, extract the coefficient value from the map
   int this_att = trans.Attribute;
   Coefficient *coeff;
	double value;
   auto it = material_map.find(this_att);
   if (it != material_map.end())
   {
      coeff = it->second.get();
		value = EvalStateDeriv(coeff, trans, ip, state);
   }
   else if (default_coeff)
   {
      value = EvalStateDeriv(default_coeff.get(), trans, ip, state);
   }
   else // if attribute not found in material map default to zero
   {
      value = 0.0;
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

void VectorMeshDependentCoefficient::Eval(Vector &vec,
                                          ElementTransformation &trans,
                                          const IntegrationPoint &ip)
{
   // given the attribute, extract the coefficient value from the map
   int this_att = trans.Attribute;
   VectorCoefficient *coeff;
   auto it = material_map.find(this_att);
   if (it != material_map.end())
   {
      // std::cout << "attr found\n";
      coeff = it->second.get();
      coeff->Eval(vec, trans, ip);
      // std::cout << "mag_vec in eval: ";
      // vec.Print();
   }
   else if (default_coeff)
   {
      default_coeff->Eval(vec, trans, ip);
   }
   else // if attribute not found and no default set, set the output to be zero
   {
      vec = 0.0;
   }
   // std::cout << "mag_vec in eval: ";
   // vec.Print();
}

} // namespace mach