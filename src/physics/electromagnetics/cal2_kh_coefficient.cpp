#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "mfem.hpp"
#include "nlohmann/json.hpp"
#include "tinysplinecxx.h"

#include "cal2_kh_coefficient.hpp"
#include "utils.hpp"

#include "coefficient.hpp"
#include "mach_input.hpp"

namespace
{
/// TODO: Ensure all states will be doubles (they may be grid functions or something else)
class PolyVarHysteresisLossCoeff : public mach::ThreeStateCoefficient
{
public:
   /// \brief Define a model to represent the polynomial fit for the variable hysteresis coefficient
   ///      empirically derived from a data source (NASA, Carpenter, ADA, etc.)
   /// \param[in] T0 - the lower temperature used for curve fitting kh(f,B) and ke(f,B)
   /// \param[in] kh_T0 - vector of variable hysteresis loss coefficients at temperature T0, empirically found.
   ///                     kh_T0=[kh0_T0, kh1_T0, kh2_T0, ...], kh_T0(B)=kh0_T0+kh1_T0*B+kh2_T0*B^2...
   /// \param[in] T1 - the upper temperature used for curve fitting kh(f,B) and kh(f,B)
   /// \param[in] kh_T1 - vector of variable hysteresis loss coefficients at temperature T1, empirically found.
   ///                     kh_T1=[kh0_T1, kh1_T1, kh2_T1, ...], kh_T1(B)=kh0_T1+kh1_T1*B+kh2_T1*B^2...
   PolyVarHysteresisLossCoeff(const double &T0,
                        const std::vector<double> &kh_T0,
                        const double &T1,
                        const std::vector<double> &kh_T1);

   /// \brief Evaluate the variable hysteresis coefficient in the element described by trans at the
   /// point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double Eval(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               double state1,
               double state2,
               double state3) override;

   /// \brief Evaluate the derivative of the variable hysteresis coefficient in the element with respect to the 1st state variable
   ///   described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double EvalDerivS1(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               double state1,
               double state2,
               double state3) override;

   /// \brief Evaluate the derivative of the variable hysteresis coefficient in the element with respect to the 2nd state variable
   ///   described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double EvalDerivS2(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               double state1,
               double state2,
               double state3) override;

   /// \brief Evaluate the derivative of the variable hysteresis coefficient in the element with respect to the 3rd state variable
   ///   described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double EvalDerivS3(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               double state1,
               double state2,
               double state3) override;

   /// \brief Evaluate the second derivative of the variable hysteresis coefficient in the element with respect to the 1st state variable
   ///   described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double Eval2ndDerivS1(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               double state1,
               double state2,
               double state3) override;

   /// \brief Evaluate the derivative of the variable hysteresis coefficient in the element with respect to the 2nd state variable
   ///   described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double Eval2ndDerivS2(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               double state1,
               double state2,
               double state3) override;

   /// \brief Evaluate the derivative of the variable hysteresis coefficient in the element with respect to the 3rd state variable
   ///   described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double Eval2ndDerivS3(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               double state1,
               double state2,
               double state3) override;

   /// \brief Evaluate the second derivative of the variable hysteresis coefficient in the element with respect to the 1st then 2nd state variable
   ///   described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double Eval2ndDerivS1S2(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               double state1,
               double state2,
               double state3) override;

   /// \brief Evaluate the second derivative of the variable hysteresis coefficient in the element with respect to the 1st then 3rd state variable
   ///   described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double Eval2ndDerivS1S3(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               double state1,
               double state2,
               double state3) override;

   /// \brief Evaluate the second derivative of the variable hysteresis coefficient in the element with respect to the 2nd then 3rd state variable
   ///   described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double Eval2ndDerivS2S3(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               double state1,
               double state2,
               double state3) override;

   ///TODO: Likely not necessary because of Eval2ndDerivS1S2
   /// \brief Evaluate the derivative of the variable hysteresis coefficient in the element with respect to the 2nd then 1st state variable
   ///   described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double Eval2ndDerivS2S1(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               double state1,
               double state2,
               double state3) override;

   ///TODO: Likely not necessary because of Eval2ndDerivS1S3
   /// \brief Evaluate the derivative of the variable hysteresis coefficient in the element with respect to the 3rd then 1st state variable
   ///   described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double Eval2ndDerivS3S1(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               double state1,
               double state2,
               double state3) override;

   ///TODO: Likely not necessary because of Eval2ndDerivS2S3
   /// \brief Evaluate the derivative of the variable hysteresis coefficient in the element with respect to the 3rd then 2nd state variable
   ///   described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double Eval2ndDerivS3S2(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               double state1,
               double state2,
               double state3) override;

   /// TODO: Adapt EvalRevDiff if needed for variable hysteresis coefficient
   void EvalRevDiff(const double Q_bar,
                    mfem::ElementTransformation &trans,
                    const mfem::IntegrationPoint &ip,
                    mfem::DenseMatrix &PointMat_bar) override
   { }

protected:
/// TODO: determine protected outputs that will be useful (protected meaning child classes can access too, but other classes cannot) 
   double T0, T1;
   std::vector<double> kh_T0, kh_T1;
};

/// TODO: this default variable hysteresis loss coefficient will need to be altered
/// TODO: this will need some work -- COME BACK TO THIS (possibly even remove)
std::unique_ptr<mfem::Coefficient> constructDefaultCAL2_khCoeff(
    const std::string &material_name,
    const nlohmann::json &materials)
{
   auto CAL2_kh = materials[material_name].value("kh", 1.0);
   return std::make_unique<mfem::ConstantCoefficient>(kh);
}

std::unique_ptr<mfem::Coefficient> constructCAL2_khCoeff(
    const nlohmann::json &component,
    const nlohmann::json &materials)
{
   std::unique_ptr<mfem::Coefficient> temp_coeff; // temp=temporary, not temperature
   const auto &material = component["material"];

   /// If "material" is a string, it is interpreted to be the name of a 
   /// material. We default to a CAL2_kh coeff of ///TODO: (insert default value here) unless
   /// there is a different value in the material library
   if (material.is_string())
   {
      const auto &material_name = material.get<std::string>();
      ///TODO: Ensure this Default CAL2_kh coefficient is in order
      temp_coeff = constructDefaultCAL2_khCoeff(material_name, materials);
   }
   else
   {
      const auto &material_name = material["name"].get<std::string>();

      ///TODO: Perhaps soften this if statement (impose default T0 and T1 values, for example)
      if (materials[material_name]["core_loss"]["CAL2"].contains("T0") && 
         materials[material_name]["core_loss"]["CAL2"].contains("kh_T0") && 
         materials[material_name]["core_loss"]["CAL2"].contains("kh_T1"))
      {
         double T0 = material["T0"].get<double>();
         std::vector<double> kh_T0 = materials[material_name]["core_loss"]
                                 ["CAL2"]["kh_T0"].get<std::vector<double>>();
         double T1 = material["T1"].get<double>();
         std::vector<double> kh_T1 = materials[material_name]["core_loss"]
                                 ["CAL2"]["kh_T1"].get<std::vector<double>>();

         temp_coeff = std::make_unique<PolyVarHysteresisLossCoeff>(
            T0, kh_T0, T1, kh_T1);
      }
      else
      {
         std::string error_msg =
               "Insufficient information to compute CAL2 variable hysteresis loss coefficient for material \"";
         error_msg += material_name;
         error_msg += "\"!\n";
         throw mach::MachException(error_msg);
      }
   }
   return temp_coeff;
}

}  // anonymous namespace

namespace mach
{
double CAL2khCoefficient::Eval(mfem::ElementTransformation &trans,
                                    const mfem::IntegrationPoint &ip)
{
   return CAL2_kh.Eval(trans, ip);
}

double CAL2khCoefficient::Eval(mfem::ElementTransformation &trans,
                                    const mfem::IntegrationPoint &ip,
                                    double state1,
                                    double state2,
                                    double state3)
{
   return CAL2_kh.Eval(trans, ip, state1, state2, state3);
}

double CAL2khCoefficient::EvalDerivS1(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state1,
    double state2,
    double state3)
{
   return CAL2_kh.EvalDerivS1(trans, ip, state1, state2, state3);
}

double CAL2khCoefficient::EvalDerivS2(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state1,
    double state2,
    double state3)
{
   return CAL2_kh.EvalDerivS2(trans, ip, state1, state2, state3);
}

double CAL2khCoefficient::EvalDerivS3(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state1,
    double state2,
    double state3)
{
   return CAL2_kh.EvalDerivS3(trans, ip, state1, state2, state3);
}

double CAL2khCoefficient::Eval2ndDerivS1(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state1,
    double state2,
    double state3)
{
   return CAL2_kh.Eval2ndDerivS1(trans, ip, state1, state2, state3);
}

double CAL2khCoefficient::Eval2ndDerivS2(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state1,
    double state2,
    double state3)
{
   return CAL2_kh.Eval2ndDerivS2(trans, ip, state1, state2, state3);
}

double CAL2khCoefficient::Eval2ndDerivS3(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state1,
    double state2,
    double state3)
{
   return CAL2_kh.Eval2ndDerivS3(trans, ip, state1, state2, state3);
}

double CAL2khCoefficient::EvalDerivS1S2(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state1,
    double state2,
    double state3)
{
   return CAL2_kh.EvalDerivS1S2(trans, ip, state1, state2, state3);
}

double CAL2khCoefficient::EvalDerivS1S3(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state1,
    double state2,
    double state3)
{
   return CAL2_kh.EvalDerivS1S3(trans, ip, state1, state2, state3);
}

double CAL2khCoefficient::EvalDerivS2S3(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state1,
    double state2,
    double state3)
{
   return CAL2_kh.EvalDerivS2S3(trans, ip, state1, state2, state3);
}

///TODO: Likely not necessary because of Eval2ndDerivS1S2
double CAL2khCoefficient::EvalDerivS2S1(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state1,
    double state2,
    double state3)
{
   return CAL2_kh.EvalDerivS2S1(trans, ip, state1, state2, state3);
}

///TODO: Likely not necessary because of Eval2ndDerivS1S3
double CAL2khCoefficient::EvalDerivS3S1(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state1,
    double state2,
    double state3)
{
   return CAL2_kh.EvalDerivS3S1(trans, ip, state1, state2, state3);
}

///TODO: Likely not necessary because of Eval2ndDerivS2S3
double CAL2keCoefficient::EvalDerivS3S2(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state1,
    double state2,
    double state3)
{
   return CAL2_kh.EvalDerivS3S2(trans, ip, state1, state2, state3);
}

/// TODO: Adapt if needed
void CAL2keCoefficient::EvalRevDiff(const double Q_bar,
                                         mfem::ElementTransformation &trans,
                                         const mfem::IntegrationPoint &ip,
                                         mfem::DenseMatrix &PointMat_bar)
{
   CAL2_kh.EvalRevDiff(Q_bar, trans, ip, PointMat_bar);
}

/// TODO: Change CAL2_kh(std::make_unique<mfem::ConstantCoefficient>(1.0)) line IF the equivalent lines...
/// std::unique_ptr<mfem::Coefficient> constructDefaultCAL2_khCoeff( from earlier change
CAL2khCoefficient::CAL2khCoefficient(const nlohmann::json &CAL2_kh_options,
                                               const nlohmann::json &materials)
 : CAL2_kh(std::make_unique<mfem::ConstantCoefficient>(1.0)) 
{
   /// loop over all components, construct a CAL2_kh loss coefficient for each
   for (const auto &component : CAL2_kh_options["components"])
   {
      int attr = component.value("attr", -1);
      if (-1 != attr)
      {
         CAL2_kh.addCoefficient(attr,
                           constructDefaultCAL2_khCoeff(component, materials));
      }
      else
      {
         for (const auto &attribute : component["attrs"])
         {
            CAL2_kh.addCoefficient(attribute,
                              constructCAL2_khCoeff(component, materials));
         }
      }
   }
}

}  // namespace mach

namespace
{
PolyVarHysteresisLossCoeff::PolyVarHysteresisLossCoeff(const double &T0,
                        const std::vector<double> &kh_T0,
                        const double &T1,
                        const std::vector<double> &kh_T1)
 : T0(T0), kh_T0(kh_T0), T1(T1), kh_T1(kh_T1)

///TODO: As needed, add in more definitions of protected class members here
{
  
///TODO: As needed, add in calculations of protected class members here

}

double PolyVarHysteresisLossCoeff::Eval(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           const double state1,
                           const double state2,
                           const double state3)
{
   ///TODO: As needed, utilize logic of protected class members to eval CAL2_kh

   // Assuming state1=temperature, state2=frequency, state3=max alternating flux density
   auto T = state1;
   auto f = state2;
   auto Bm = state3;

   double kh_T0_f_B = 0.0;
   for (int i = 0; i < kh_T0.size(); ++i)
   {
      kh_T0_f_B += kh_T0[i]*pow(Bm,i);
   }
   double kh_T1_f_B = 0.0;
   for (int i = 0; i < kh_T1.size(); ++i)
   {
      kh_T1_f_B += kh_T1[i]*pow(Bm,i);
   }
   double D_hyst = (kh_T1_f_B-kh_T0_f_B)/((T1-T0)*kh_T0_f_B);
   double kth = 1+(T-T0)*D_hyst;
   
   double CAL2_kh = kth*kh_T0_f_B;

   return CAL2_kh;
}

double PolyVarHysteresisLossCoeff::EvalDerivS1(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           const double state1,
                           const double state2,
                           const double state3)
{
   ///TODO: As needed, utilize logic of protected class members to eval CAL2_kh

   ///TODO: Derived overall derivative in Overleaf. Need to modify for just the CAL2_kh, code the below, then uncomment
   /*
   double dCAL2_khdf = 
   return dCAL2_khdf;
   */
}

double PolyVarHysteresisLossCoeff::EvalDerivS2(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           const double state1,
                           const double state2,
                           const double state3)
{
   ///TODO: As needed, utilize logic of protected class members to eval CAL2_kh

   ///TODO: Derived overall derivative in Overleaf. Need to modify for just the CAL2_kh, code the below, then uncomment
   /*
   double dCAL2_khdB = 
   return dCAL2_khdB;
   */
}

double PolyVarHysteresisLossCoeff::EvalDerivS3(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           const double state1,
                           const double state2,
                           const double state3)
{
   ///TODO: As needed, utilize logic of protected class members to eval CAL2_kh

   ///TODO: Derived overall derivative in Overleaf. Need to modify for just the CAL2_kh, code the below, then uncomment
   /*
   double dCAL2_khdT = 
   return dCAL2_khdT;
   */
}

double PolyVarHysteresisLossCoeff::Eval2ndDerivS1(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           const double state1,
                           const double state2,
                           const double state3)
{
   ///TODO: As needed, utilize logic of protected class members to eval CAL2_kh

   ///TODO: Derive and code the below, then uncomment
   /*
   double d2CAL2_khdf2 = 
   return d2CAL2_khdf2;
   */
}

double PolyVarHysteresisLossCoeff::Eval2ndDerivS2(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           const double state1,
                           const double state2,
                           const double state3)
{
   ///TODO: As needed, utilize logic of protected class members to eval CAL2_kh

   ///TODO: Derive and code the below, then uncomment
   /*
   double d2CAL2_khdB2 = 
   return d2CAL2_khdB2;
   */
}

double PolyVarHysteresisLossCoeff::Eval2ndDerivS3(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           const double state1,
                           const double state2,
                           const double state3)
{
   ///TODO: As needed, utilize logic of protected class members to eval CAL2_kh

   ///TODO: Derive and code the below, then uncomment
   /*
   double d2CAL2_khdT2 = 
   return d2CAL2_khdT2;
   */
}

double PolyVarHysteresisLossCoeff::Eval2ndDerivS1S2(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           const double state1,
                           const double state2,
                           const double state3)
{
   ///TODO: As needed, utilize logic of protected class members to eval CAL2_kh

   ///TODO: Derive and code the below, then uncomment
   /*
   double d2CAL2_khdfdB = 
   return d2CAL2_khdfdB;
   */
}

double PolyVarHysteresisLossCoeff::Eval2ndDerivS1S3(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           const double state1,
                           const double state2,
                           const double state3)
{
   ///TODO: As needed, utilize logic of protected class members to eval CAL2_kh

   ///TODO: Derive and code the below, then uncomment
   /*
   double d2CAL2_khdfdT = 
   return d2CAL2_khdfdT;
   */
}

double PolyVarHysteresisLossCoeff::Eval2ndDerivS2S3(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           const double state1,
                           const double state2,
                           const double state3)
{
   ///TODO: As needed, utilize logic of protected class members to eval CAL2_kh

   ///TODO: Derive and code the below, then uncomment
   /*
   double d2CAL2_khdBdT = 
   return d2CAL2_khdBdT;
   */
}

///TODO: Likely not necessary because of Eval2ndDerivS1S2
double PolyVarHysteresisLossCoeff::Eval2ndDerivS2S1(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           const double state1,
                           const double state2,
                           const double state3)
{
   ///TODO: As needed, utilize logic of protected class members to eval CAL2_kh

   ///TODO: Derive and code the below, then uncomment
   /*
   double d2CAL2_khdBdf = 
   return d2CAL2_khdBdf;
   */
}

///TODO: Likely not necessary because of Eval2ndDerivS1S3
double PolyVarHysteresisLossCoeff::Eval2ndDerivS3S1(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           const double state1,
                           const double state2,
                           const double state3)
{
   ///TODO: As needed, utilize logic of protected class members to eval CAL2_kh

   ///TODO: Derive and code the below, then uncomment
   /*
   double d2CAL2_khdTdf = 
   return d2CAL2_khdTdf;
   */
}

///TODO: Likely not necessary because of Eval2ndDerivS2S3
double PolyVarHysteresisLossCoeff::Eval2ndDerivS3S2(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           const double state1,
                           const double state2,
                           const double state3)
{
   ///TODO: As needed, utilize logic of protected class members to eval CAL2_kh

   ///TODO: Derive and code the below, then uncomment
   /*
   double d2CAL2_khdTdB = 
   return d2CAL2_khdTdB;
   */
}

///TODO: is there a need to code EvalRevDiff for variable hysteresis coefficient method here? I'm thinking not

}

///TODO: After code derivatives, circle back around and check this and hpp file for logic/consistency

