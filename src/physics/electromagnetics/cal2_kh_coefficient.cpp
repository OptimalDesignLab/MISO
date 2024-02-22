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
#include "miso_input.hpp"

namespace
{
class PolyVarHysteresisLossCoeff : public miso::ThreeStateCoefficient
{
public:
   /// \brief Define a model to represent the polynomial fit for the variable
   /// hysteresis coefficient
   ///      empirically derived from a data source (NASA, Carpenter, ADA, etc.)
   /// \param[in] T0 - the lower temperature used for curve fitting kh(f,B) and
   /// ke(f,B) \param[in] kh_T0 - vector of variable hysteresis loss
   /// coefficients at temperature T0, empirically found.
   ///                     kh_T0=[kh0_T0, kh1_T0, kh2_T0, ...],
   ///                     kh_T0(B)=kh0_T0+kh1_T0*B+kh2_T0*B^2...
   /// \param[in] T1 - the upper temperature used for curve fitting kh(f,B) and
   /// kh(f,B) \param[in] kh_T1 - vector of variable hysteresis loss
   /// coefficients at temperature T1, empirically found.
   ///                     kh_T1=[kh0_T1, kh1_T1, kh2_T1, ...],
   ///                     kh_T1(B)=kh0_T1+kh1_T1*B+kh2_T1*B^2...
   PolyVarHysteresisLossCoeff(const double &T0,
                              const std::vector<double> &kh_T0,
                              const double &T1,
                              const std::vector<double> &kh_T1);

   /// \brief Evaluate the variable hysteresis coefficient in the element
   /// described by trans at the point ip. \note When this method is called, the
   /// caller must make sure that the IntegrationPoint associated with trans is
   /// the same as ip. This can be achieved by calling trans.SetIntPoint(&ip).
   double Eval(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               double state1,
               double state2,
               double state3) override;

   /// \brief Evaluate the derivative of the variable hysteresis coefficient in
   /// the element with respect to the 1st state variable
   ///   described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double EvalDerivS1(mfem::ElementTransformation &trans,
                      const mfem::IntegrationPoint &ip,
                      double state1,
                      double state2,
                      double state3) override;

   /// \brief Evaluate the derivative of the variable hysteresis coefficient in
   /// the element with respect to the 2nd state variable
   ///   described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double EvalDerivS2(mfem::ElementTransformation &trans,
                      const mfem::IntegrationPoint &ip,
                      double state1,
                      double state2,
                      double state3) override;

   /// \brief Evaluate the derivative of the variable hysteresis coefficient in
   /// the element with respect to the 3rd state variable
   ///   described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double EvalDerivS3(mfem::ElementTransformation &trans,
                      const mfem::IntegrationPoint &ip,
                      double state1,
                      double state2,
                      double state3) override;

   /// \brief Evaluate the second derivative of the variable hysteresis
   /// coefficient in the element with respect to the 1st state variable
   ///   described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double Eval2ndDerivS1(mfem::ElementTransformation &trans,
                         const mfem::IntegrationPoint &ip,
                         double state1,
                         double state2,
                         double state3) override;

   /// \brief Evaluate the derivative of the variable hysteresis coefficient in
   /// the element with respect to the 2nd state variable
   ///   described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double Eval2ndDerivS2(mfem::ElementTransformation &trans,
                         const mfem::IntegrationPoint &ip,
                         double state1,
                         double state2,
                         double state3) override;

   /// \brief Evaluate the derivative of the variable hysteresis coefficient in
   /// the element with respect to the 3rd state variable
   ///   described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double Eval2ndDerivS3(mfem::ElementTransformation &trans,
                         const mfem::IntegrationPoint &ip,
                         double state1,
                         double state2,
                         double state3) override;

   /// \brief Evaluate the second derivative of the variable hysteresis
   /// coefficient in the element with respect to the 1st then 2nd state
   /// variable
   ///   described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double Eval2ndDerivS1S2(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           double state1,
                           double state2,
                           double state3) override;

   /// \brief Evaluate the second derivative of the variable hysteresis
   /// coefficient in the element with respect to the 1st then 3rd state
   /// variable
   ///   described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double Eval2ndDerivS1S3(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           double state1,
                           double state2,
                           double state3) override;

   /// \brief Evaluate the second derivative of the variable hysteresis
   /// coefficient in the element with respect to the 2nd then 3rd state
   /// variable
   ///   described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double Eval2ndDerivS2S3(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           double state1,
                           double state2,
                           double state3) override;

   /// TODO: Likely not necessary because of Eval2ndDerivS1S2
   ///  \brief Evaluate the derivative of the variable hysteresis coefficient in
   ///  the element with respect to the 2nd then 1st state variable
   ///    described by trans at the point ip.
   ///  \note When this method is called, the caller must make sure that the
   ///  IntegrationPoint associated with trans is the same as ip. This can be
   ///  achieved by calling trans.SetIntPoint(&ip).
   double Eval2ndDerivS2S1(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           double state1,
                           double state2,
                           double state3) override;

   /// TODO: Likely not necessary because of Eval2ndDerivS1S3
   ///  \brief Evaluate the derivative of the variable hysteresis coefficient in
   ///  the element with respect to the 3rd then 1st state variable
   ///    described by trans at the point ip.
   ///  \note When this method is called, the caller must make sure that the
   ///  IntegrationPoint associated with trans is the same as ip. This can be
   ///  achieved by calling trans.SetIntPoint(&ip).
   double Eval2ndDerivS3S1(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           double state1,
                           double state2,
                           double state3) override;

   /// TODO: Likely not necessary because of Eval2ndDerivS2S3
   ///  \brief Evaluate the derivative of the variable hysteresis coefficient in
   ///  the element with respect to the 3rd then 2nd state variable
   ///    described by trans at the point ip.
   ///  \note When this method is called, the caller must make sure that the
   ///  IntegrationPoint associated with trans is the same as ip. This can be
   ///  achieved by calling trans.SetIntPoint(&ip).
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
   double T0;
   std::vector<double> kh_T0;
   double T1;
   std::vector<double> kh_T1;
};

/// TODO: this default variable hysteresis loss coefficient will need to be
/// altered
/// TODO: this will need some work -- COME BACK TO THIS (possibly even remove)
std::unique_ptr<mfem::Coefficient> constructDefaultCAL2_khCoeff(
    const std::string &material_name,
    const nlohmann::json &materials)
{
   auto CAL2_kh = materials[material_name].value("kh", 1.0);
   return std::make_unique<mfem::ConstantCoefficient>(CAL2_kh);
}

// Get T0, kh_T0, T1, and kh_T1 from JSON
/// TODO: Add in argument of const std::string &model if decide to combine
/// Steinmetz and CAL2 into one
void getTsAndKhs(const nlohmann::json &material,
                 const nlohmann::json &materials,
                 double &T0,
                 std::vector<double> &kh_T0,
                 double &T1,
                 std::vector<double> &kh_T1)
{
   const auto &material_name = material["name"].get<std::string>();

   // Assign T0 based on material options, else refer to material library
   if (material["core_loss"].contains("T0"))
   {
      T0 = material["core_loss"]["T0"].get<double>();
   }
   else
   {
      T0 = materials[material_name]["core_loss"]["CAL2"]["T0"].get<double>();
   }

   // Assign kh_T0 based on material options, else refer to material library
   if (material["core_loss"].contains("kh_T0"))
   {
      kh_T0 = material["core_loss"]["kh_T0"].get<std::vector<double>>();
   }
   else
   {
      kh_T0 = materials[material_name]["core_loss"]["CAL2"]["kh_T0"]
                  .get<std::vector<double>>();
   }

   // Assign T1 based on material options, else refer to material library
   if (material["core_loss"].contains("T1"))
   {
      T1 = material["core_loss"]["T1"].get<double>();
   }
   else
   {
      T1 = materials[material_name]["core_loss"]["CAL2"]["T1"].get<double>();
   }

   // Assign kh_T1 based on material options, else refer to material library
   if (material["core_loss"].contains("kh_T1"))
   {
      kh_T1 = material["core_loss"]["kh_T1"].get<std::vector<double>>();
   }
   else
   {
      kh_T1 = materials[material_name]["core_loss"]["CAL2"]["kh_T1"]
                  .get<std::vector<double>>();
   }
}

// Construct the kh coefficient
std::unique_ptr<mfem::Coefficient> constructCAL2_khCoeff(
    const nlohmann::json &component,
    const nlohmann::json &materials)
{
   std::unique_ptr<mfem::Coefficient>
       temp_coeff;  // temp=temporary, not temperature
   const auto &material = component["material"];

   /// If "material" is a string, it is interpreted to be the name of a
   /// material. We default to a CAL2_kh coeff of 1 ///TODO: (change this value
   /// as needed) unless there is a different value in the material library
   if (material.is_string())
   {
      const auto &material_name = material.get<std::string>();
      /// TODO: Ensure this Default CAL2_kh coefficient is in order
      temp_coeff = constructDefaultCAL2_khCoeff(material_name, materials);
   }
   else
   {
      const auto &material_name = material["name"].get<std::string>();

      if (material.contains("core_loss"))
      {
         // Declare variables
         double T0;
         std::vector<double> kh_T0;
         double T1;
         std::vector<double> kh_T1;

         // Obtain the necessary parameters from the JSON
         getTsAndKhs(material, materials, T0, kh_T0, T1, kh_T1);

         // Can now construct the coefficient accordingly
         temp_coeff =
             std::make_unique<PolyVarHysteresisLossCoeff>(T0, kh_T0, T1, kh_T1);

         /// TODO: Add this error handing in if add in multiple models or
         /// combine Steinmetz and CAL2 into one
         // else
         // {
         //    std::string error_msg =
         //          "Insufficient information to compute CAL2 variable
         //          hysteresis loss coefficient for material \"";
         //    error_msg += material_name;
         //    error_msg += "\"!\n";
         //    throw miso::MISOException(error_msg);
         // }
      }
      else
      {
         // Doesn't have the core loss JSON structure; assign it default
         // coefficient
         temp_coeff = constructDefaultCAL2_khCoeff(material_name, materials);
      }
   }
   return temp_coeff;
}

}  // anonymous namespace

namespace miso
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

double CAL2khCoefficient::EvalDerivS1(mfem::ElementTransformation &trans,
                                      const mfem::IntegrationPoint &ip,
                                      double state1,
                                      double state2,
                                      double state3)
{
   return CAL2_kh.EvalDerivS1(trans, ip, state1, state2, state3);
}

double CAL2khCoefficient::EvalDerivS2(mfem::ElementTransformation &trans,
                                      const mfem::IntegrationPoint &ip,
                                      double state1,
                                      double state2,
                                      double state3)
{
   return CAL2_kh.EvalDerivS2(trans, ip, state1, state2, state3);
}

double CAL2khCoefficient::EvalDerivS3(mfem::ElementTransformation &trans,
                                      const mfem::IntegrationPoint &ip,
                                      double state1,
                                      double state2,
                                      double state3)
{
   return CAL2_kh.EvalDerivS3(trans, ip, state1, state2, state3);
}

double CAL2khCoefficient::Eval2ndDerivS1(mfem::ElementTransformation &trans,
                                         const mfem::IntegrationPoint &ip,
                                         double state1,
                                         double state2,
                                         double state3)
{
   return CAL2_kh.Eval2ndDerivS1(trans, ip, state1, state2, state3);
}

double CAL2khCoefficient::Eval2ndDerivS2(mfem::ElementTransformation &trans,
                                         const mfem::IntegrationPoint &ip,
                                         double state1,
                                         double state2,
                                         double state3)
{
   return CAL2_kh.Eval2ndDerivS2(trans, ip, state1, state2, state3);
}

double CAL2khCoefficient::Eval2ndDerivS3(mfem::ElementTransformation &trans,
                                         const mfem::IntegrationPoint &ip,
                                         double state1,
                                         double state2,
                                         double state3)
{
   return CAL2_kh.Eval2ndDerivS3(trans, ip, state1, state2, state3);
}

double CAL2khCoefficient::Eval2ndDerivS1S2(mfem::ElementTransformation &trans,
                                           const mfem::IntegrationPoint &ip,
                                           double state1,
                                           double state2,
                                           double state3)
{
   return CAL2_kh.Eval2ndDerivS1S2(trans, ip, state1, state2, state3);
}

double CAL2khCoefficient::Eval2ndDerivS1S3(mfem::ElementTransformation &trans,
                                           const mfem::IntegrationPoint &ip,
                                           double state1,
                                           double state2,
                                           double state3)
{
   return CAL2_kh.Eval2ndDerivS1S3(trans, ip, state1, state2, state3);
}

double CAL2khCoefficient::Eval2ndDerivS2S3(mfem::ElementTransformation &trans,
                                           const mfem::IntegrationPoint &ip,
                                           double state1,
                                           double state2,
                                           double state3)
{
   return CAL2_kh.Eval2ndDerivS2S3(trans, ip, state1, state2, state3);
}

/// TODO: Likely not necessary because of Eval2ndDerivS1S2
double CAL2khCoefficient::Eval2ndDerivS2S1(mfem::ElementTransformation &trans,
                                           const mfem::IntegrationPoint &ip,
                                           double state1,
                                           double state2,
                                           double state3)
{
   return CAL2_kh.Eval2ndDerivS2S1(trans, ip, state1, state2, state3);
}

/// TODO: Likely not necessary because of Eval2ndDerivS1S3
double CAL2khCoefficient::Eval2ndDerivS3S1(mfem::ElementTransformation &trans,
                                           const mfem::IntegrationPoint &ip,
                                           double state1,
                                           double state2,
                                           double state3)
{
   return CAL2_kh.Eval2ndDerivS3S1(trans, ip, state1, state2, state3);
}

/// TODO: Likely not necessary because of Eval2ndDerivS2S3
double CAL2khCoefficient::Eval2ndDerivS3S2(mfem::ElementTransformation &trans,
                                           const mfem::IntegrationPoint &ip,
                                           double state1,
                                           double state2,
                                           double state3)
{
   return CAL2_kh.Eval2ndDerivS3S2(trans, ip, state1, state2, state3);
}

/// TODO: Adapt if needed
void CAL2khCoefficient::EvalRevDiff(const double Q_bar,
                                    mfem::ElementTransformation &trans,
                                    const mfem::IntegrationPoint &ip,
                                    mfem::DenseMatrix &PointMat_bar)
{
   CAL2_kh.EvalRevDiff(Q_bar, trans, ip, PointMat_bar);
}

/// TODO: Change CAL2_kh(std::make_unique<mfem::ConstantCoefficient>(1.0)) line
/// IF the equivalent lines... std::unique_ptr<mfem::Coefficient>
/// constructDefaultCAL2_khCoeff( from earlier change
CAL2khCoefficient::CAL2khCoefficient(const nlohmann::json &CAL2_kh_options,
                                     const nlohmann::json &materials)
 : CAL2_kh(std::make_unique<mfem::ConstantCoefficient>(1.0))
{
   if (CAL2_kh_options.contains("components"))
   {
      /// Options are being passed in. Loop over the components within and
      /// construct a CAL2_kh loss coefficient for each
      for (const auto &component : CAL2_kh_options["components"])
      {
         int attr = component.value("attr", -1);
         if (-1 != attr)
         {
            CAL2_kh.addCoefficient(
                attr, constructDefaultCAL2_khCoeff(component, materials));
         }
         else
         {
            for (const auto &attribute : component["attrs"])
            {
               CAL2_kh.addCoefficient(
                   attribute, constructCAL2_khCoeff(component, materials));
            }
         }
      }
   }
   else
   {
      /// Components themselves are being passed in. Loop over the components
      /// and construct a CAL2_kh loss coefficient for each
      auto components = CAL2_kh_options;
      for (const auto &component : components)
      {
         int attr = component.value("attr", -1);
         if (-1 != attr)
         {
            CAL2_kh.addCoefficient(
                attr, constructDefaultCAL2_khCoeff(component, materials));
         }
         else
         {
            for (const auto &attribute : component["attrs"])
            {
               CAL2_kh.addCoefficient(
                   attribute, constructCAL2_khCoeff(component, materials));
            }
         }
      }
   }
}

}  // namespace miso

namespace
{
PolyVarHysteresisLossCoeff::PolyVarHysteresisLossCoeff(
    const double &T0,
    const std::vector<double> &kh_T0,
    const double &T1,
    const std::vector<double> &kh_T1)
 : T0(T0), kh_T0(kh_T0), T1(T1), kh_T1(kh_T1)

{ }

double PolyVarHysteresisLossCoeff::Eval(mfem::ElementTransformation &trans,
                                        const mfem::IntegrationPoint &ip,
                                        const double state1,
                                        const double state2,
                                        const double state3)
{
   // Assuming state1=temperature, state2=frequency, state3=max alternating flux
   // density
   auto T = state1;
   // auto f = state2;
   auto B_m = state3;

   double kh_T0_f_B = 0.0;
   for (int i = 0; i < static_cast<int>(kh_T0.size()); ++i)
   {
      kh_T0_f_B += kh_T0[i] * std::pow(B_m, i);
   }
   double kh_T1_f_B = 0.0;
   for (int i = 0; i < static_cast<int>(kh_T1.size()); ++i)
   {
      kh_T1_f_B += kh_T1[i] * std::pow(B_m, i);
   }
   double D_hyst = (kh_T1_f_B - kh_T0_f_B) / ((T1 - T0) * kh_T0_f_B);
   double kth = 1 + (T - T0) * D_hyst;

   double CAL2_kh = kth * kh_T0_f_B;

   return CAL2_kh;
}

double PolyVarHysteresisLossCoeff::EvalDerivS1(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    const double state1,
    const double state2,
    const double state3)
{
   // Assuming state1=temperature, state2=frequency, state3=max alternating flux
   // density First derivative with respect to temperature

   auto T = state1;
   // auto f = state2;
   auto B_m = state3;

   double kh_T0_f_B = 0.0;
   for (int i = 0; i < static_cast<int>(kh_T0.size()); ++i)
   {
      kh_T0_f_B += kh_T0[i] * std::pow(B_m, i);
   }
   double kh_T1_f_B = 0.0;
   for (int i = 0; i < static_cast<int>(kh_T1.size()); ++i)
   {
      kh_T1_f_B += kh_T1[i] * std::pow(B_m, i);
   }
   double D_hyst = (kh_T1_f_B - kh_T0_f_B) / ((T1 - T0) * kh_T0_f_B);
   // double kth = 1+(T-T0)*D_hyst;
   double dkthdT = D_hyst;

   // double CAL2_kh = kth*kh_T0_f_B;

   double dCAL2_khdT = dkthdT * kh_T0_f_B;
   return dCAL2_khdT;
}

double PolyVarHysteresisLossCoeff::EvalDerivS2(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    const double state1,
    const double state2,
    const double state3)
{
   // Assuming state1=temperature, state2=frequency, state3=max alternating flux
   // density First derivative with respect to frequency

   // Frequency is not explicitly used to calculate the coefficient itself.
   // Frequency is used to explicitly calculate the specific core loss (next
   // level up) Therefore, all CAL2 coefficient derivatives with respect to
   // frequency are 0
   double dCAL2_khdf = 0;
   return dCAL2_khdf;
}

double PolyVarHysteresisLossCoeff::EvalDerivS3(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    const double state1,
    const double state2,
    const double state3)
{
   // Assuming state1=temperature, state2=frequency, state3=max alternating flux
   // density First derivative with respect to max alternating flux density

   auto T = state1;
   // auto f = state2;
   auto B_m = state3;

   // double kh_T0_f_B = 0.0;
   double dkh_T0_f_BdB_m = 0.0;
   for (int i = 0; i < static_cast<int>(kh_T0.size()); ++i)
   {
      // kh_T0_f_B += kh_T0[i]*std::pow(B_m,i);
      dkh_T0_f_BdB_m += i * kh_T0[i] * std::pow(B_m, i - 1);
   }
   // double kh_T1_f_B = 0.0;
   double dkh_T1_f_BdB_m = 0.0;
   for (int i = 0; i < static_cast<int>(kh_T1.size()); ++i)
   {
      // kh_T1_f_B += kh_T1[i]*std::pow(B_m,i);
      dkh_T1_f_BdB_m += i * kh_T1[i] * std::pow(B_m, i - 1);
   }
   // double D_hyst = (kh_T1_f_B-kh_T0_f_B)/((T1-T0)*kh_T0_f_B);
   // double kth = 1+(T-T0)*D_hyst;

   // double CAL2_kh = kth*kh_T0_f_B;

   double dCAL2_khdB_m = dkh_T0_f_BdB_m + ((T - T0) / (T1 - T0)) *
                                              (dkh_T1_f_BdB_m - dkh_T0_f_BdB_m);
   return dCAL2_khdB_m;
}

double PolyVarHysteresisLossCoeff::Eval2ndDerivS1(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    const double state1,
    const double state2,
    const double state3)
{
   // Assuming state1=temperature, state2=frequency, state3=max alternating flux
   // density Second derivative with respect to temperature

   // double D_hyst = (kh_T1_f_B-kh_T0_f_B)/((T1-T0)*kh_T0_f_B);
   // double kth = 1+(T-T0)*D_hyst;
   // double dkthdT = D_hyst;

   // double CAL2_kh = kth*kh_T0_f_B;
   // double dCAL2_khdT = dkthdT*kh_T0_f_B;

   // As seen, CAL2 coefficient merely linear in temperature
   // Thus, 2nd and higher order derivatives w/r/t temperature will be 0
   double d2CAL2_khdT2 = 0;
   return d2CAL2_khdT2;
}

double PolyVarHysteresisLossCoeff::Eval2ndDerivS2(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    const double state1,
    const double state2,
    const double state3)
{
   // Assuming state1=temperature, state2=frequency, state3=max alternating flux
   // density Second derivative with respect to frequency

   // Frequency is not explicitly used to calculate the coefficient itself.
   // Frequency is used to explicitly calculate the specific core loss (next
   // level up) Therefore, all CAL2 coefficient derivatives with respect to
   // frequency are 0
   double d2CAL2_khdf2 = 0;
   return d2CAL2_khdf2;
}

double PolyVarHysteresisLossCoeff::Eval2ndDerivS3(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    const double state1,
    const double state2,
    const double state3)
{
   // Assuming state1=temperature, state2=frequency, state3=max alternating flux
   // density Second derivative with respect to max alternating flux density

   auto T = state1;
   // auto f = state2;
   auto B_m = state3;

   // double dkh_T0_f_BdB_m = 0.0;
   double d2kh_T0_f_BdB_m2 = 0.0;
   for (int i = 0; i < static_cast<int>(kh_T0.size()); ++i)
   {
      // dkh_T0_f_BdB_m += i*kh_T0[i]*std::pow(B_m,i-1);
      d2kh_T0_f_BdB_m2 += (i - 1) * i * kh_T0[i] * std::pow(B_m, i - 2);
   }
   // double dkh_T1_f_BdB_m = 0.0;
   double d2kh_T1_f_BdB_m2 = 0.0;
   for (int i = 0; i < static_cast<int>(kh_T1.size()); ++i)
   {
      // dkh_T1_f_BdB_m += i*kh_T1[i]*std::pow(B_m,i-1);
      d2kh_T1_f_BdB_m2 += (i - 1) * i * kh_T1[i] * std::pow(B_m, i - 2);
   }
   // double D_hyst = (kh_T1_f_B-kh_T0_f_B)/((T1-T0)*kh_T0_f_B);
   // double kth = 1+(T-T0)*D_hyst;

   // double CAL2_kh = kth*kh_T0_f_B;

   double d2CAL2_khdB_m2 =
       d2kh_T0_f_BdB_m2 +
       ((T - T0) / (T1 - T0)) * (d2kh_T1_f_BdB_m2 - d2kh_T0_f_BdB_m2);
   return d2CAL2_khdB_m2;
}

double PolyVarHysteresisLossCoeff::Eval2ndDerivS1S2(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    const double state1,
    const double state2,
    const double state3)
{
   // Assuming state1=temperature, state2=frequency, state3=max alternating flux
   // density Derivative with respect to temperature then frequency

   // Frequency is not explicitly used to calculate the coefficient itself.
   // Frequency is used to explicitly calculate the specific core loss (next
   // level up) Therefore, all CAL2 coefficient derivatives with respect to
   // frequency are 0
   double d2CAL2_khdTdf = 0;
   return d2CAL2_khdTdf;
}

double PolyVarHysteresisLossCoeff::Eval2ndDerivS1S3(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    const double state1,
    const double state2,
    const double state3)
{
   // Assuming state1=temperature, state2=frequency, state3=max alternating flux
   // density Derivative with respect to temperature then max alternating flux
   // density

   auto T = state1;
   // auto f = state2;
   auto B_m = state3;

   // double kh_T0_f_B = 0.0;
   double dkh_T0_f_BdB_m = 0.0;
   for (int i = 0; i < static_cast<int>(kh_T0.size()); ++i)
   {
      // kh_T0_f_B += kh_T0[i]*std::pow(B_m,i);
      dkh_T0_f_BdB_m += i * kh_T0[i] * std::pow(B_m, i - 1);
   }
   // double kh_T1_f_B = 0.0;
   double dkh_T1_f_BdB_m = 0.0;
   for (int i = 0; i < static_cast<int>(kh_T1.size()); ++i)
   {
      // kh_T1_f_B += kh_T1[i]*std::pow(B_m,i);
      dkh_T1_f_BdB_m += i * kh_T1[i] * std::pow(B_m, i - 1);
   }
   // double D_hyst = (kh_T1_f_B-kh_T0_f_B)/((T1-T0)*kh_T0_f_B);
   // double kth = 1+(T-T0)*D_hyst;
   // double dkthdT = D_hyst;

   // double CAL2_kh = kth*kh_T0_f_B;
   // double dCAL2_khdT = dkthdT*kh_T0_f_B = (kh_T1_f_B-kh_T0_f_B)/(T1-T0);

   double d2CAL2_khdTdB_m = (dkh_T1_f_BdB_m - dkh_T0_f_BdB_m) / (T1 - T0);
   return d2CAL2_khdTdB_m;
}

double PolyVarHysteresisLossCoeff::Eval2ndDerivS2S3(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    const double state1,
    const double state2,
    const double state3)
{
   // Assuming state1=temperature, state2=frequency, state3=max alternating flux
   // density Derivative with respect to frequency then max alternating flux
   // density

   // Frequency is not explicitly used to calculate the coefficient itself.
   // Frequency is used to explicitly calculate the specific core loss (next
   // level up) Therefore, all CAL2 coefficient derivatives with respect to
   // frequency are 0
   double d2CAL2_khdfdB_m = 0;
   return d2CAL2_khdfdB_m;
}

/// TODO: Likely not necessary because of Eval2ndDerivS1S2
double PolyVarHysteresisLossCoeff::Eval2ndDerivS2S1(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    const double state1,
    const double state2,
    const double state3)
{
   // Assuming state1=temperature, state2=frequency, state3=max alternating flux
   // density Derivative with respect to frequency then temperature

   // Frequency is not explicitly used to calculate the coefficient itself.
   // Frequency is used to explicitly calculate the specific core loss (next
   // level up) Therefore, all CAL2 coefficient derivatives with respect to
   // frequency are 0
   double d2CAL2_khdfdT = 0;
   return d2CAL2_khdfdT;
}

/// TODO: Likely not necessary because of Eval2ndDerivS1S3
double PolyVarHysteresisLossCoeff::Eval2ndDerivS3S1(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    const double state1,
    const double state2,
    const double state3)
{
   // Assuming state1=temperature, state2=frequency, state3=max alternating flux
   // density Derivative with respect to max alternating flux density then
   // temperature

   auto T = state1;
   // auto f = state2;
   auto B_m = state3;

   // double kh_T0_f_B = 0.0;
   double dkh_T0_f_BdB_m = 0.0;
   for (int i = 0; i < static_cast<int>(kh_T0.size()); ++i)
   {
      // kh_T0_f_B += kh_T0[i]*std::pow(B_m,i);
      dkh_T0_f_BdB_m += i * kh_T0[i] * std::pow(B_m, i - 1);
   }
   // double kh_T1_f_B = 0.0;
   double dkh_T1_f_BdB_m = 0.0;
   for (int i = 0; i < static_cast<int>(kh_T1.size()); ++i)
   {
      // kh_T1_f_B += kh_T1[i]*std::pow(B_m,i);
      dkh_T1_f_BdB_m += i * kh_T1[i] * std::pow(B_m, i - 1);
   }
   // double D_hyst = (kh_T1_f_B-kh_T0_f_B)/((T1-T0)*kh_T0_f_B);
   // double kth = 1+(T-T0)*D_hyst;
   // double dkthdT = D_hyst;

   // double CAL2_kh = kth*kh_T0_f_B;
   // double dCAL2_khdB_m =
   // dkh_T0_f_BdB_m+((T-T0)/(T1-T0))*(dkh_T1_f_BdB_m-dkh_T0_f_BdB_m);

   double d2CAL2_khdB_mdT = (dkh_T1_f_BdB_m - dkh_T0_f_BdB_m) / (T1 - T0);
   return d2CAL2_khdB_mdT;
}

/// TODO: Likely not necessary because of Eval2ndDerivS2S3
double PolyVarHysteresisLossCoeff::Eval2ndDerivS3S2(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    const double state1,
    const double state2,
    const double state3)
{
   // Assuming state1=temperature, state2=frequency, state3=max alternating flux
   // density Derivative with respect to max alternating flux density then
   // frequency

   // Frequency is not explicitly used to calculate the coefficient itself.
   // Frequency is used to explicitly calculate the specific core loss (next
   // level up) Therefore, all CAL2 coefficient derivatives with respect to
   // frequency are 0
   double d2CAL2_khdB_mdf = 0;
   return d2CAL2_khdB_mdf;
}

/// TODO: is there a need to code EvalRevDiff for variable hysteresis
/// coefficient method here? I'm thinking not

}  // namespace

/// TODO: After code derivatives, circle back around and check this and hpp file
/// for logic/consistency
