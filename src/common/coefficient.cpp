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

void MeshDependentCoefficient::EvalRevDiff(
   const double &Q_bar,
   ElementTransformation &trans,
   const IntegrationPoint &ip,
   DenseMatrix &PointMat_bar)
{
   // given the attribute, extract the coefficient value from the map
   int this_att = trans.Attribute;
   Coefficient *coeff;
   auto it = material_map.find(this_att);
   if (it != material_map.end())
   {
      coeff = it->second.get();
      coeff->EvalRevDiff(Q_bar, trans, ip, PointMat_bar);
   }
   else if (default_coeff)
   {
      default_coeff->EvalRevDiff(Q_bar, trans, ip, PointMat_bar);
   }
   // if attribute not found and no default set, don't change PointMat_bar
   return;
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
   if (A)
   {
      Array<int> vdofs; Vector a(trans.GetSpaceDim()); 
      const FiniteElement *el = A->FESpace()->GetFE(trans.ElementNo);
      A->FESpace()->GetElementVDofs(trans.ElementNo, vdofs);
      DenseMatrix J = trans.Jacobian(); //Element Jacobian
      DenseMatrix C(a.Size(), J.Width()); //Curl Shape Functions
      el->CalcCurlShape(ip, C);
      A->GetSubVector(vdofs, a);
      
      //Compute Magnetic Field
      Vector b(a.Size()); Vector bh(a.Size());
      C.MultTranspose(a, bh); //C^T a
      J.Mult(bh, b); //J C^T a

      double bMag = b.Norml2();
      return rho*(kh*freq*pow(bMag, alpha) + ke*freq*freq*bMag*bMag);
   }
   else
      return 0.0;
}

void SteinmetzCoefficient::EvalRevDiff(const double &Q_bar,
    						                  ElementTransformation &trans,
    						                  const IntegrationPoint &ip,
    						                  DenseMatrix &PointMat_bar)
{
   if (A)
   {
      Array<int> vdofs; Vector a(trans.GetSpaceDim()); 
      const FiniteElement *el = A->FESpace()->GetFE(trans.ElementNo);
      ElementTransformation *Tr = A->FESpace()->GetElementTransformation(trans.ElementNo);
      Tr->SetIntPoint(&ip);
      A->FESpace()->GetElementVDofs(trans.ElementNo, vdofs);
      DenseMatrix J = Tr->Jacobian(); //Element Jacobian
      DenseMatrix C(a.Size(), J.Width()); //Curl Shape Functions
      el->CalcCurlShape(ip, C);
      A->GetSubVector(vdofs, a);
      DenseMatrix jac_bar(trans.GetSpaceDim()); jac_bar = 0.0;
      
      //Compute Magnetic Field
      Vector b(a.Size()); Vector bh(a.Size());
      C.MultTranspose(a, bh); //C^T a
      J.Mult(bh, b); //J C^T a

      double bMag = b.Norml2();
      //rho*(kh*freq*pow(bMag, alpha) + ke*freq*freq*bMag*bMag);
      double dS = rho*(alpha*kh*freq*pow(bMag, alpha-2) + 2*ke*freq*freq); //dS/dBmag * 1/Bmag
      AddMult_a_VWt(dS, b, bh, jac_bar); // B*Bh^T

      // cast the ElementTransformation
      IsoparametricTransformation &isotrans =
         dynamic_cast<IsoparametricTransformation&>(*Tr);

      isotrans.JacobianRevDiff(jac_bar, PointMat_bar);
   }
   else
      return;
}

void SteinmetzVectorDiffCoefficient::Eval(Vector &V, 
                                             ElementTransformation &T,
                                             const IntegrationPoint &ip)
{
   if (A)
   {
      Array<int> vdofs; Vector a(T.GetSpaceDim()); 
      const FiniteElement *el = A->FESpace()->GetFE(T.ElementNo);
      A->FESpace()->GetElementVDofs(T.ElementNo, vdofs);
      DenseMatrix J = T.Jacobian(); //Element Jacobian
      DenseMatrix C(a.Size(), J.Width()); //Curl Shape Functions
      el->CalcCurlShape(ip, C);
      A->GetSubVector(vdofs, a);
      
      //Compute Magnetic Field
      Vector b(a.Size()); Vector bh(a.Size());
      C.MultTranspose(a, bh); //C^T a
      J.Mult(bh, b); //J C^T a

      //Compute Derivative w.r.t. a
      Vector zw(b.Size()); Vector z(b.Size());
      J.MultTranspose(b, zw);
      C.Mult(zw, z);
      double bMag = b.Norml2();
      double dS = rho*(alpha*kh*freq*pow(bMag, alpha-2) + 2*ke*freq*freq); //dS/dBmag * 1/Bmag
      V.Set(dS, z);
   }
   else
      V =  0.0;
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