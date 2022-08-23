#include <cmath>

#include "tinysplinecxx.h"

#include "coefficient.hpp"

using namespace mfem;

namespace mach
{
double ParameterContinuationCoefficient::lambda = 0.0;

double ParameterContinuationCoefficient::Eval(ElementTransformation &trans,
                                              const IntegrationPoint &ip,
                                              const double state)
{
   const double lin_comp = (1 - lambda) * linear->Eval(trans, ip);
   const double nonlin_comp = lambda * nonlinear->Eval(trans, ip, state);
   return lin_comp + nonlin_comp;
}

double ParameterContinuationCoefficient::EvalStateDeriv(
    ElementTransformation &trans,
    const IntegrationPoint &ip,
    const double state)
{
   return lambda * nonlinear->EvalStateDeriv(trans, ip, state);
}

double MeshDependentCoefficient::Eval(ElementTransformation &trans,
                                      const IntegrationPoint &ip)
{
   // given the attribute, extract the coefficient value from the map
   int this_att = trans.Attribute;
   double value = NAN;
   auto it = material_map.find(this_att);
   if (it != material_map.end())
   {
      auto *coeff = it->second.get();
      value = coeff->Eval(trans, ip);
   }
   else if (default_coeff)
   {
      value = default_coeff->Eval(trans, ip);
   }
   else  // if attribute not found and no default set, evaluate to zero
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
   double value = NAN;
   auto it = material_map.find(this_att);
   if (it != material_map.end())
   {
      auto *coeff = it->second.get();
      auto *state_coeff = dynamic_cast<StateCoefficient *>(coeff);
      if (state_coeff != nullptr)
      {
         value = state_coeff->Eval(trans, ip, state);
      }
      else
      {
         value = coeff->Eval(trans, ip);
      }
   }
   else if (default_coeff)
   {
      auto *state_coeff = dynamic_cast<StateCoefficient *>(default_coeff.get());
      if (state_coeff != nullptr)
      {
         value = state_coeff->Eval(trans, ip, state);
      }
      else
      {
         value = default_coeff->Eval(trans, ip);
      }
   }
   else  // if attribute not found and no default set, evaluate to zero
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
   double value = NAN;
   auto it = material_map.find(this_att);
   if (it != material_map.end())
   {
      auto *coeff = it->second.get();
      auto *state_coeff = dynamic_cast<StateCoefficient *>(coeff);
      if (state_coeff != nullptr)
      {
         value = state_coeff->EvalStateDeriv(trans, ip, state);
      }
      else
      {
         value = 0.0;
      }
   }
   else if (default_coeff)
   {
      auto *state_coeff = dynamic_cast<StateCoefficient *>(default_coeff.get());
      if (state_coeff != nullptr)
      {
         value = state_coeff->EvalStateDeriv(trans, ip, state);
      }
      else
      {
         value = 0.0;
      }
   }
   else  // if attribute not found in material map default to zero
   {
      value = 0.0;
   }
   return value;
}

double MeshDependentCoefficient::EvalState2ndDeriv(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    const double state)
{
   // given the attribute, extract the coefficient value from the map
   int this_att = trans.Attribute;
   double value = NAN;
   auto it = material_map.find(this_att);
   if (it != material_map.end())
   {
      auto *coeff = it->second.get();
      auto *state_coeff = dynamic_cast<StateCoefficient *>(coeff);
      if (state_coeff != nullptr)
      {
         value = state_coeff->EvalState2ndDeriv(trans, ip, state);
      }
      else
      {
         value = 0.0;
      }
   }
   else if (default_coeff)
   {
      auto *state_coeff = dynamic_cast<StateCoefficient *>(default_coeff.get());
      if (state_coeff != nullptr)
      {
         value = state_coeff->EvalState2ndDeriv(trans, ip, state);
      }
      else
      {
         value = 0.0;
      }
   }
   else  // if attribute not found in material map default to zero
   {
      value = 0.0;
   }
   return value;
}

void MeshDependentCoefficient::EvalRevDiff(const double Q_bar,
                                           ElementTransformation &trans,
                                           const IntegrationPoint &ip,
                                           DenseMatrix &PointMat_bar)
{
   // given the attribute, extract the coefficient value from the map
   int this_att = trans.Attribute;
   Coefficient *coeff = nullptr;
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
}

std::unique_ptr<mach::MeshDependentCoefficient> constructMaterialCoefficient(
    const std::string &name,
    const nlohmann::json &components,
    const nlohmann::json &materials,
    double default_val)
{
   auto material_coeff = std::make_unique<mach::MeshDependentCoefficient>();
   /// loop over all components, construct coeff for each
   for (const auto &component : components)
   {
      int attr = component.value("attr", -1);

      const auto &material = component["material"];
      std::string material_name;
      if (material.is_string())
      {
         material_name = material.get<std::string>();
      }
      else
      {
         material_name = material["name"].get<std::string>();
      }

      double val = materials[material_name].value(name, default_val);

      if (-1 != attr)
      {
         auto coeff = std::make_unique<mfem::ConstantCoefficient>(val);
         material_coeff->addCoefficient(attr, move(coeff));
      }
      else
      {
         for (const auto &attribute : component["attrs"])
         {
            auto coeff = std::make_unique<mfem::ConstantCoefficient>(val);
            material_coeff->addCoefficient(attribute, move(coeff));
         }
      }
   }
   return material_coeff;
}

// NonlinearReluctivityCoefficient::NonlinearReluctivityCoefficient(
//     const std::vector<double> &B,
//     const std::vector<double> &H)
//  // : b_max(B[B.size()-1]), nu(H.size(), 1, 3)
//  : b_max(B[B.size() - 1]),
//    h_max(H[H.size() - 1]),
//    bh(std::make_unique<tinyspline::BSpline>(H.size(), 1, 3))
// {
//    std::vector<double> knots(B);
//    for (int i = 0; i < B.size(); ++i)
//    {
//       knots[i] = knots[i] / b_max;
//    }
//    bh->setControlPoints(H);
//    bh->setKnots(knots);

//    dbdh = std::make_unique<tinyspline::BSpline>(bh->derive());
//    // dnudb = nu.derive();
// }

// double NonlinearReluctivityCoefficient::Eval(ElementTransformation &trans,
//                                              const IntegrationPoint &ip,
//                                              const double state)
// {
//    constexpr double nu0 = 1 / (4e-7 * M_PI);
//    // std::cout << "eval state state: " << state << "\n";
//    if (state <= 1e-14)
//    {
//       double t = state / b_max;
//       double nu = dbdh->eval(t).result()[0] / b_max;
//       return nu;
//    }
//    else if (state <= b_max)
//    {
//       double t = state / b_max;
//       double nu = bh->eval(t).result()[0] / state;
//       // std::cout << "eval state nu: " << nu << "\n";
//       return nu;
//    }
//    else
//    {
//       return (h_max - nu0 * b_max) / state + nu0;
//    }
// }

// double NonlinearReluctivityCoefficient::EvalStateDeriv(
//     ElementTransformation &trans,
//     const IntegrationPoint &ip,
//     const double state)
// {
//    constexpr double nu0 = 1 / (4e-7 * M_PI);

//    /// TODO: handle state == 0
//    if (state <= b_max)
//    {
//       double t = state / b_max;
//       double h = bh->eval(t).result()[0];
//       return dbdh->eval(t).result()[0] / (state * b_max) - h / pow(state, 2);
//    }
//    else
//    {
//       return -(h_max - nu0 * b_max) / pow(state, 2);
//    }
// }

// NonlinearReluctivityCoefficient::~NonlinearReluctivityCoefficient() =
// default;

// /// namespace for TEAM 13 B-H curve fit
// namespace
// {
// /** unused
// double team13h(double b_hat)
// {
//    const double h =
//        exp((0.0011872363994136887 * pow(b_hat, 2) * (15 * pow(b_hat, 2)
//        - 9.0) -
//             0.19379133411847338 * pow(b_hat, 2) -
//             0.012675319795245974 * b_hat * (3 * pow(b_hat, 2) - 1.0) +
//             0.52650810858405916 * b_hat + 0.77170389255937188) /
//            (-0.037860246476916264 * pow(b_hat, 2) +
//             0.085040155318288846 * b_hat + 0.1475250808150366)) -
//        31;
//    return h;
// }
// */

// double team13dhdb_hat(double b_hat)
// {
//    const double dhdb_hat =
//        (-0.0013484718812450662 * pow(b_hat, 5) +
//         0.0059829967461202211 * pow(b_hat, 4) +
//         0.0040413617616232578 * pow(b_hat, 3) -
//         0.013804440762666015 * pow(b_hat, 2) - 0.0018970139190370716 * b_hat
//         + 0.013917259962808418) *
//        exp((0.017808545991205332 * pow(b_hat, 4) -
//             0.038025959385737926 * pow(b_hat, 3) -
//             0.20447646171319658 * pow(b_hat, 2) + 0.53918342837930511 * b_hat
//             + 0.77170389255937188) /
//            (-0.037860246476916264 * pow(b_hat, 2) +
//             0.085040155318288846 * b_hat + 0.1475250808150366)) /
//        (0.0014333982632928504 * pow(b_hat, 4) -
//         0.0064392824815713142 * pow(b_hat, 3) -
//         0.0039388438258098624 * pow(b_hat, 2) + 0.025091111571707653 * b_hat
//         + 0.02176364946948308);
//    return dhdb_hat;
// }

// double team13d2hdb_hat2(double b_hat)
// {
//    const double d2hdb_hat2 =
//        (1.8183764145086082e-6 * pow(b_hat, 10) -
//         1.6135805755447689e-5 * pow(b_hat, 9) +
//         2.2964027416433258e-5 * pow(b_hat, 8) +
//         0.00010295509167249583 * pow(b_hat, 7) -
//         0.0001721199302193437 * pow(b_hat, 6) -
//         0.00031470749218644612 * pow(b_hat, 5) +
//         0.00054873370082066282 * pow(b_hat, 4) +
//         0.00078428896855240252 * pow(b_hat, 3) -
//         0.00020176627749697931 * pow(b_hat, 2) -
//         0.00054403666453702558 * b_hat - 0.00019679534359955033) *
//        exp((0.017808545991205332 * pow(b_hat, 4) -
//             0.038025959385737926 * pow(b_hat, 3) -
//             0.20447646171319658 * pow(b_hat, 2) + 0.53918342837930511 * b_hat
//             + 0.77170389255937188) /
//            (-0.037860246476916264 * pow(b_hat, 2) +
//             0.085040155318288846 * b_hat + 0.1475250808150366)) /
//        (2.0546305812109595e-6 * pow(b_hat, 8) -
//         1.8460112651872795e-5 * pow(b_hat, 7) +
//         3.0172495078875982e-5 * pow(b_hat, 6) +
//         0.00012265776759231136 * pow(b_hat, 5) -
//         0.0002452310649846335 * pow(b_hat, 4) -
//         0.00047794451332165656 * pow(b_hat, 3) +
//         0.00045811664722395466 * pow(b_hat, 2) + 0.001092148314092672 * b_hat
//         + 0.00047365643823053111);
//    return d2hdb_hat2;
// }

// double team13b_hat(double b)
// {
//    const double b_hat = 1.10803324099723 * b + 1.10803324099723 * atan(20 *
//    b) -
//                         0.9944598337950139;
//    return b_hat;
// }

// double team13db_hatdb(double b)
// {
//    const double db_hatdb = (443.213296398892 * pow(b, 2) + 23.26869806094183)
//    /
//                            (400 * pow(b, 2) + 1);
//    return db_hatdb;
// }

// double team13d2b_hatdb2(double b)
// {
//    const double d2b_hatdb2 =
//        -17728.53185595568 * b / pow(400 * pow(b, 2) + 1, 2);
//    return d2b_hatdb2;
// }

// }  // anonymous namespace

void VectorMeshDependentCoefficient::Eval(Vector &vec,
                                          ElementTransformation &trans,
                                          const IntegrationPoint &ip)
{
   // given the attribute, extract the coefficient value from the map
   int this_att = trans.Attribute;
   VectorCoefficient *coeff = nullptr;
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
   else  // if attribute not found and no default set, set the output to be zero
   {
      vec = 0.0;
   }
   // std::cout << "mag_vec in eval: ";
   // vec.Print();
}

void VectorMeshDependentCoefficient::EvalRevDiff(const Vector &V_bar,
                                                 ElementTransformation &trans,
                                                 const IntegrationPoint &ip,
                                                 DenseMatrix &PointMat_bar)
{
   // given the attribute, extract the coefficient value from the map
   int this_att = trans.Attribute;
   VectorCoefficient *coeff = nullptr;
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
}

double SteinmetzCoefficient::Eval(ElementTransformation &trans,
                                  const IntegrationPoint &ip)
{
   int dim = trans.GetSpaceDim();
   // Array<int> vdofs;
   // Vector elfun;
   // A.FESpace()->GetElementVDofs(trans.ElementNo, vdofs);
   // A.GetSubVector(vdofs, elfun);

   // auto &el = *A.FESpace()->GetFE(trans.ElementNo);
   // int ndof = el.GetDof();

   // DenseMatrix curlshape(ndof,dim);
   // DenseMatrix curlshape_dFt(ndof,dim);
   Vector b_vec(dim);
   b_vec = 0.0;

   trans.SetIntPoint(&ip);

   /// TODO: this changes how I differentiate
   A.GetCurl(trans, b_vec);

   // el.CalcCurlShape(ip, curlshape);
   // MultABt(curlshape, trans.Jacobian(), curlshape_dFt);
   // curlshape_dFt.AddMultTranspose(elfun, b_vec);
   // curlshape.AddMultTranspose(elfun, b_vec);

   double b_mag = b_vec.Norml2();

   // double S = rho*(kh*freq*std::pow(b_mag, alpha) +
   // ke*freq*freq*b_mag*b_mag);

   double S = rho * ks * std::pow(freq, alpha) * std::pow(b_mag, beta);
   return -S;
}

void SteinmetzCoefficient::EvalRevDiff(const double Q_bar,
                                       ElementTransformation &trans,
                                       const IntegrationPoint &ip,
                                       DenseMatrix &PointMat_bar)
{
   int dim = trans.GetSpaceDim();
   Array<int> vdofs;
   Vector elfun;
   A.FESpace()->GetElementVDofs(trans.ElementNo, vdofs);
   A.GetSubVector(vdofs, elfun);

   const auto &el = *A.FESpace()->GetFE(trans.ElementNo);
   int ndof = el.GetDof();

   DenseMatrix curlshape(ndof, dim);
   DenseMatrix curlshape_dFt(ndof, dim);
   Vector b_vec(dim);
   Vector b_hat(dim);
   b_vec = 0.0;
   b_hat = 0.0;

   trans.SetIntPoint(&ip);

   el.CalcCurlShape(ip, curlshape);
   MultABt(curlshape, trans.Jacobian(), curlshape_dFt);
   curlshape_dFt.AddMultTranspose(elfun, b_vec);
   curlshape.AddMultTranspose(elfun, b_hat);

   // double b_mag = b_vec.Norml2();
   // double S = rho*(kh*freq*std::pow(b_mag, alpha) +
   // ke*freq*freq*b_mag*b_mag); double dS = rho*(alpha*kh*freq*std::pow(b_mag,
   // alpha-2) + 2*ke*freq*freq);
   double dS = 1.0;

   DenseMatrix Jac_bar(3);
   MultVWt(b_vec, b_hat, Jac_bar);
   Jac_bar *= dS;

   // cast the ElementTransformation
   auto &isotrans = dynamic_cast<IsoparametricTransformation &>(trans);

   DenseMatrix loc_PointMat_bar(PointMat_bar.Height(), PointMat_bar.Width());
   loc_PointMat_bar = 0.0;

   isotrans.JacobianRevDiff(Jac_bar, loc_PointMat_bar);

   PointMat_bar.Add(Q_bar, loc_PointMat_bar);
}

void SteinmetzVectorDiffCoefficient::Eval(Vector &V,
                                          ElementTransformation &trans,
                                          const IntegrationPoint &ip)
{
   int dim = trans.GetSpaceDim();
   Array<int> vdofs;
   Vector elfun;
   A.FESpace()->GetElementVDofs(trans.ElementNo, vdofs);
   A.GetSubVector(vdofs, elfun);

   const auto &el = *A.FESpace()->GetFE(trans.ElementNo);
   int ndof = el.GetDof();

   DenseMatrix curlshape(ndof, dim);
   DenseMatrix curlshape_dFt(ndof, dim);
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
   double dS = rho * (alpha * kh * freq * std::pow(b_mag, alpha - 2) +
                      2 * ke * freq * freq);

   V *= dS;
}

double ElementFunctionCoefficient::Eval(ElementTransformation &trans,
                                        const IntegrationPoint &ip)
{
   double x[3];
   Vector transip(x, 3);

   trans.Transform(ip, transip);

   int ei = trans.ElementNo;

   if (Function != nullptr)
   {
      return (*Function)(transip, ei);
   }
   else
   {
      return (*TDFunction)(transip, ei, GetTime());
   }
}

}  // namespace mach