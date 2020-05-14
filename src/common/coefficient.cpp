#include "coefficient.hpp"

using namespace mfem;

namespace mach
{

double MeshDependentCoefficient::Eval(ElementTransformation &trans,
                                      const IntegrationPoint &ip)
{
   // given the attribute, extract the coefficient value from the map
   int this_att = trans.Attribute;
   Coefficient *coeff;
	double value;
   auto it = material_map.find(this_att);
   if (it != material_map.end())
   {
      coeff = it->second.get();
      value = coeff->Eval(trans, ip);
   }
   else if (default_coeff)
   {
      value = default_coeff->Eval(trans, ip);
   }
   else // if attribute not found and no default set, evaluate to zero
   {
      value = 0.0;
   }
   // std::cout << "nu val in eval: " << value << "\n";
   return value;
}

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

ReluctivityCoefficient::ReluctivityCoefficient(std::vector<double> B,
                                               std::vector<double> H)
   : temperature_GF(NULL), b_h_curve()
{
   b_h_curve.set_boundary(Spline::second_deriv, 0.0,
                          Spline::second_deriv, 0.0);
   b_h_curve.set_points(B, H);
}

double ReluctivityCoefficient::Eval(ElementTransformation &trans,
												const IntegrationPoint &ip,
                                    const double state)
{
	if (temperature_GF)
   {
      throw MachException(
         "Temperature dependent reluctivity is not currently supported!");
   }
   else
   {
      return b_h_curve.deriv(1, state);
   }
}

double ReluctivityCoefficient::EvalStateDeriv(ElementTransformation &trans,
												          const IntegrationPoint &ip,
                                              const double state)
{
   if (temperature_GF)
   {
      throw MachException(
         "Temperature dependent reluctivity is not currently supported!");
   }
   else
   {
      return b_h_curve.deriv(2, state);
   }
}

MagneticFluxCoefficient::MagneticFluxCoefficient(std::vector<double> B,
                                                 std::vector<double> H)
   : temperature_GF(NULL), b_h_curve()
{
   b_h_curve.set_boundary(Spline::second_deriv, 0.0,
                          Spline::second_deriv, 0.0);
   b_h_curve.set_points(H, B);
}

double MagneticFluxCoefficient::Eval(ElementTransformation &trans,
												 const IntegrationPoint &ip,
                                     const double state)
{
	if (temperature_GF)
   {
      throw MachException(
         "Temperature dependent reluctivity is not currently supported!");
   }
   else
   {
      return b_h_curve(state);
   }
}

double MagneticFluxCoefficient::EvalStateDeriv(ElementTransformation &trans,
												           const IntegrationPoint &ip,
                                               const double state)
{
   if (temperature_GF)
   {
      throw MachException(
         "Temperature dependent reluctivity is not currently supported!");
   }
   else
   {
      return b_h_curve.deriv(1, state);
   }
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

void VectorMeshDependentCoefficient::EvalRevDiff(
   const Vector &V_bar,
   ElementTransformation &trans,
   const IntegrationPoint &ip,
   DenseMatrix &PointMat_bar)
{
   // given the attribute, extract the coefficient value from the map
   int this_att = trans.Attribute;
   VectorCoefficient *coeff;
   auto it = material_map.find(this_att);
   if (it != material_map.end())
   {
      coeff = it->second.get();
      coeff->EvalRevDiff(V_bar, trans, ip, PointMat_bar);
   }
   else if (default_coeff)
   {
      default_coeff->EvalRevDiff(V_bar, trans, ip, PointMat_bar);
   }
   // if attribute not found and no default set, don't change PointMat_bar
   return;
}

double SteinmetzCoefficient::Eval(ElementTransformation &trans,
                                  const IntegrationPoint &ip)
{
   if (B)
   {
      Vector b;
      B->GetVectorValue(trans.ElementNo, ip, b);
      double bMag = b.Norml2();
      return rho*(kh*freq*pow(bMag, alpha) + ke*freq*freq*bMag*bMag);
   }
   else
      return 0.0;
}

double ElementFunctionCoefficient::Eval (ElementTransformation &trans,
                                       const IntegrationPoint &ip)
{
   double x[3];
   Vector transip(x, 3);
   
   trans.Transform(ip, transip);
   
   int ei = trans.ElementNo;

   if (Function)
   {
      return (*Function)(transip, ei);
   }
   else
   {
      return (*TDFunction)(transip, ei, GetTime());
   }
}

} // namespace mach