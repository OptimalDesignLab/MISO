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
   const double Q_bar,
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

ReluctivityCoefficient::ReluctivityCoefficient(const std::vector<double> &B,
                                               const std::vector<double> &H)
   : b_max(B[B.size()-1]), b_h(B.size())
{
   /// set control points of B-H spline
   auto ctrlp = b_h.controlPoints();
   for (size_t i = 0; i < B.size(); ++i)
   {
      ctrlp[2*i] = B[i];
      ctrlp[(2*i) + 1] = H[i];
   }
   b_h.setControlPoints(ctrlp);

   /// take derivatives of B-H curve
   nu = b_h.derive();
   dnudb = nu.derive();
}

double ReluctivityCoefficient::Eval(ElementTransformation &trans,
                                    const IntegrationPoint &ip,
                                    const double state)
{
   if (state <= b_max)
   {
      auto t = b_h.bisect(state, 1e-8).knot();
      auto dbhdt = nu.eval(t).result();
      return dbhdt[1] / dbhdt[0]; // dH/dt / dB/dt
   }
   else
   {
      return 1 / (4e-7*M_PI); // assumed fully saturated
   }
}

double ReluctivityCoefficient::EvalStateDeriv(ElementTransformation &trans,
                                             const IntegrationPoint &ip,
                                             const double state)
{
   if (state <= b_max)
   {
      auto t = b_h.bisect(state, 1e-8).knot();
      auto d2bhdt2 = dnudb.eval(t).result();
      return d2bhdt2[1] / d2bhdt2[0]; // d2H/dt2 / d2B/dt2
   }
   else
   {
      return 0.0; // assumed fully saturated
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
      int dim = trans.GetSpaceDim();
      Array<int> vdofs;
      Vector elfun;
      A->FESpace()->GetElementVDofs(trans.ElementNo, vdofs);
      A->GetSubVector(vdofs, elfun);

      auto &el = *A->FESpace()->GetFE(trans.ElementNo);
      int ndof = el.GetDof();

      DenseMatrix curlshape(ndof,dim);
      DenseMatrix curlshape_dFt(ndof,dim);
      Vector b_vec(dim);
      b_vec = 0.0;

      trans.SetIntPoint(&ip);

      el.CalcCurlShape(ip, curlshape);
      MultABt(curlshape, trans.Jacobian(), curlshape_dFt);
      curlshape_dFt.AddMultTranspose(elfun, b_vec);

      double b_mag = b_vec.Norml2();

      double S = rho*(kh*freq*std::pow(b_mag, alpha) + ke*freq*freq*b_mag*b_mag);
      return S;
   }
   else
      return 0.0;
}

void SteinmetzCoefficient::EvalRevDiff(const double Q_bar,
    						                  ElementTransformation &trans,
    						                  const IntegrationPoint &ip,
    						                  DenseMatrix &PointMat_bar)
{
   if (A)
   {
      int dim = trans.GetSpaceDim();
      Array<int> vdofs;
      Vector elfun;
      A->FESpace()->GetElementVDofs(trans.ElementNo, vdofs);
      A->GetSubVector(vdofs, elfun);

      auto &el = *A->FESpace()->GetFE(trans.ElementNo);
      int ndof = el.GetDof();

      DenseMatrix curlshape(ndof,dim);
      DenseMatrix curlshape_dFt(ndof,dim);
      Vector b_vec(dim);
      Vector b_hat(dim);
      b_vec = 0.0;
      b_hat = 0.0;

      trans.SetIntPoint(&ip);

      el.CalcCurlShape(ip, curlshape);
      MultABt(curlshape, trans.Jacobian(), curlshape_dFt);
      curlshape_dFt.AddMultTranspose(elfun, b_vec);
      curlshape.AddMultTranspose(elfun, b_hat);

      double b_mag = b_vec.Norml2();
      // double S = rho*(kh*freq*std::pow(b_mag, alpha) + ke*freq*freq*b_mag*b_mag);
      double dS = rho*(alpha*kh*freq*std::pow(b_mag, alpha-2) + 2*ke*freq*freq);

      DenseMatrix Jac_bar(3);
      MultVWt(b_vec, b_hat, Jac_bar);
      Jac_bar *= dS;

      // cast the ElementTransformation
      IsoparametricTransformation &isotrans =
         dynamic_cast<IsoparametricTransformation&>(trans);

      DenseMatrix loc_PointMat_bar(PointMat_bar.Height(), PointMat_bar.Width());
      loc_PointMat_bar = 0.0;

      isotrans.JacobianRevDiff(Jac_bar, loc_PointMat_bar);

      PointMat_bar.Add(Q_bar, loc_PointMat_bar);

   }
   else
      return;
}

void SteinmetzVectorDiffCoefficient::Eval(Vector &V, 
                                          ElementTransformation &trans,
                                          const IntegrationPoint &ip)
{
   if (A)
   {
      int dim = trans.GetSpaceDim();
      Array<int> vdofs;
      Vector elfun;
      A->FESpace()->GetElementVDofs(trans.ElementNo, vdofs);
      A->GetSubVector(vdofs, elfun);

      auto &el = *A->FESpace()->GetFE(trans.ElementNo);
      int ndof = el.GetDof();

      DenseMatrix curlshape(ndof,dim);
      DenseMatrix curlshape_dFt(ndof,dim);
      Vector b_vec(dim);
      Vector temp_vec(ndof);
      b_vec = 0.0;
      temp_vec = 0.0;

      trans.SetIntPoint(&ip);

      el.CalcCurlShape(ip, curlshape);
      MultABt(curlshape, trans.Jacobian(), curlshape_dFt);
      curlshape_dFt.AddMultTranspose(elfun, b_vec);
      double b_mag = b_vec.Norml2();

      V = 0.0;
      curlshape_dFt.Mult(b_vec, temp_vec);
      V = temp_vec;
      double dS = rho*(alpha*kh*freq*std::pow(b_mag, alpha-2) + 2*ke*freq*freq);

      V *= dS;
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