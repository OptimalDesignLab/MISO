#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "mfem.hpp"
#include "nlohmann/json.hpp"
#include "tinysplinecxx.h"

#include "cal2_ke_coefficient.hpp"
#include "utils.hpp"

#include "coefficient.hpp"
#include "mach_input.hpp"

namespace
{
class PolyVarEddyCurrentLossCoeff : public mach::ThreeStateCoefficient
{
public:
   /// \brief Define a model to represent the polynomial fit for the variable eddy current coefficient
   ///      empirically derived from a data source (NASA, Carpenter, ADA, etc.)
   /// \param[in] T0 - the lower temperature used for curve fitting kh(f,B) and ke(f,B)
   /// \param[in] ke_T0 - vector of variable eddy current loss coefficients at temperature T0, empirically found.
   ///                     ke_T0=[ke0_T0, ke1_T0, ke2_T0, ...], ke_T0(B)=ke0_T0+ke1_T0*B+ke2_T0*B^2...
   /// \param[in] T1 - the upper temperature used for curve fitting ke(f,B) and ke(f,B)
   /// \param[in] ke_T1 - vector of variable eddy current loss coefficients at temperature T1, empirically found.
   ///                     ke_T1=[ke0_T1, ke1_T1, ke2_T1, ...], ke_T1(B)=ke0_T1+ke1_T1*B+ke2_T1*B^2...
   PolyVarEddyCurrentLossCoeff(const double &T0,
                        const std::vector<double> &ke_T0,
                        const double &T1,
                        const std::vector<double> &ke_T1);

   /// \brief Evaluate the variable eddy current coefficient in the element described by trans at the
   /// point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double Eval(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               double state1,
               double state2,
               double state3) override;

   /// \brief Evaluate the derivative of the variable eddy current coefficient in the element with respect to the 1st state variable
   ///   described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double EvalDerivS1(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               double state1,
               double state2,
               double state3) override;

   /// \brief Evaluate the derivative of the variable eddy current coefficient in the element with respect to the 2nd state variable
   ///   described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double EvalDerivS2(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               double state1,
               double state2,
               double state3) override;

   /// \brief Evaluate the derivative of the variable eddy current coefficient in the element with respect to the 3rd state variable
   ///   described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double EvalDerivS3(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               double state1,
               double state2,
               double state3) override;

   /// \brief Evaluate the second derivative of the variable eddy current coefficient in the element with respect to the 1st state variable
   ///   described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double Eval2ndDerivS1(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               double state1,
               double state2,
               double state3) override;

   /// \brief Evaluate the derivative of the variable eddy current coefficient in the element with respect to the 2nd state variable
   ///   described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double Eval2ndDerivS2(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               double state1,
               double state2,
               double state3) override;

   /// \brief Evaluate the derivative of the variable eddy current coefficient in the element with respect to the 3rd state variable
   ///   described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double Eval2ndDerivS3(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               double state1,
               double state2,
               double state3) override;

   /// \brief Evaluate the second derivative of the variable eddy current coefficient in the element with respect to the 1st then 2nd state variable
   ///   described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double Eval2ndDerivS1S2(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               double state1,
               double state2,
               double state3) override;

   /// \brief Evaluate the second derivative of the variable eddy current coefficient in the element with respect to the 1st then 3rd state variable
   ///   described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double Eval2ndDerivS1S3(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               double state1,
               double state2,
               double state3) override;

   /// \brief Evaluate the second derivative of the variable eddy current coefficient in the element with respect to the 2nd then 3rd state variable
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
   /// \brief Evaluate the derivative of the variable eddy current coefficient in the element with respect to the 2nd then 1st state variable
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
   /// \brief Evaluate the derivative of the variable eddy current coefficient in the element with respect to the 3rd then 1st state variable
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
   /// \brief Evaluate the derivative of the variable eddy current coefficient in the element with respect to the 3rd then 2nd state variable
   ///   described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double Eval2ndDerivS3S2(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               double state1,
               double state2,
               double state3) override;

   /// TODO: Adapt EvalRevDiff if needed for variable eddy current coefficient
   void EvalRevDiff(const double Q_bar,
                    mfem::ElementTransformation &trans,
                    const mfem::IntegrationPoint &ip,
                    mfem::DenseMatrix &PointMat_bar) override
   { }

protected:
   double T0;
   std::vector<double> ke_T0;
   double T1;
   std::vector<double> ke_T1;
};

/// TODO: this default variable eddy current loss coefficient will need to be altered
/// TODO: this will need some work -- COME BACK TO THIS (possibly even remove)
std::unique_ptr<mfem::Coefficient> constructDefaultCAL2_keCoeff(
    const std::string &material_name,
    const nlohmann::json &materials)
{
   auto CAL2_ke = materials[material_name].value("ke", 1.0);
   return std::make_unique<mfem::ConstantCoefficient>(CAL2_ke);
}

// Get T0, ke_T0, T1, and ke_T1 from JSON
///TODO: Add in argument of const std::string &model if decide to combine Steinmetz and CAL2 into one
void getTsAndKes(const nlohmann::json &material,
                  const nlohmann::json &materials,
                  double &T0, 
                  std::vector<double> &ke_T0,
                  double &T1, 
                  std::vector<double> &ke_T1)
{
   const auto &material_name = material["name"].get<std::string>();
    
   // Assign T0 based on material options, else refer to material library
   if (material["core_loss"].contains("T0"))
   {
      T0 = material["core_loss"]["T0"].get<double>();
   }
   else
   {
      T0 = materials[material_name]["core_loss"]
                              ["CAL2"]["T0"].get<double>();
   }

   // Assign ke_T0 based on material options, else refer to material library
   if (material["core_loss"].contains("ke_T0"))
   {
      ke_T0 = material["core_loss"]
                              ["ke_T0"].get<std::vector<double>>();
   }  
   else
   {
      ke_T0 = materials[material_name]["core_loss"]
                              ["CAL2"]["ke_T0"].get<std::vector<double>>();
   }         
   
   // Assign T1 based on material options, else refer to material library
   if (material["core_loss"].contains("T1"))
   {
      T1 = material["core_loss"]["T1"].get<double>();
   }
   else
   {
      T1 = materials[material_name]["core_loss"]
                              ["CAL2"]["T1"].get<double>();
   }

   // Assign ke_T1 based on material options, else refer to material library
   if (material["core_loss"].contains("ke_T1"))
   {
      ke_T1 = material["core_loss"]
                              ["ke_T1"].get<std::vector<double>>();
   }  
   else
   {
      ke_T1 = materials[material_name]["core_loss"]
                              ["CAL2"]["ke_T1"].get<std::vector<double>>();
   }
}
      
// Construct the ke coefficient
std::unique_ptr<mfem::Coefficient> constructCAL2_keCoeff(
    const nlohmann::json &component,
    const nlohmann::json &materials)
{
   std::unique_ptr<mfem::Coefficient> temp_coeff; // temp=temporary, not temperature
   const auto &material = component["material"];

   /// If "material" is a string, it is interpreted to be the name of a 
   /// material. We default to a CAL2_ke coeff of 1 ///TODO: (change this value as needed) unless
   /// there is a different value in the material library
   if (material.is_string())
   {
      const auto &material_name = material.get<std::string>();
      ///TODO: Ensure this Default CAL2_ke coefficient is in order
      temp_coeff = constructDefaultCAL2_keCoeff(material_name, materials);
   }
   else
   {
      const auto &material_name = material["name"].get<std::string>();
      
      if (material.contains("core_loss"))
      {

         // Declare variables
         double T0;
         std::vector<double> ke_T0;
         double T1;
         std::vector<double> ke_T1;

         // Obtain the necessary parameters from the JSON
         getTsAndKes(material, materials, T0, ke_T0, T1, ke_T1);

         // Can now construct the coefficient accordingly
         temp_coeff = std::make_unique<PolyVarEddyCurrentLossCoeff>(
               T0, ke_T0, T1, ke_T1);

         ///TODO: Add this error handing in if add in multiple models or combine Steinmetz and CAL2 into one
         // else
         // {
         //    std::string error_msg =
         //          "Insufficient information to compute CAL2 variable eddy current loss coefficient for material \"";
         //    error_msg += material_name;
         //    error_msg += "\"!\n";
         //    throw mach::MachException(error_msg);
         // }
      }
      else
      {
         // Doesn't have the core loss JSON structure; assign it default coefficient
         temp_coeff = constructDefaultCAL2_keCoeff(material_name, materials);
      }
   }
   return temp_coeff;
}

}  // anonymous namespace

namespace mach
{
double CAL2keCoefficient::Eval(mfem::ElementTransformation &trans,
                                    const mfem::IntegrationPoint &ip)
{
   return CAL2_ke.Eval(trans, ip);
}

double CAL2keCoefficient::Eval(mfem::ElementTransformation &trans,
                                    const mfem::IntegrationPoint &ip,
                                    double state1,
                                    double state2,
                                    double state3)
{
   return CAL2_ke.Eval(trans, ip, state1, state2, state3);
}

double CAL2keCoefficient::EvalDerivS1(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state1,
    double state2,
    double state3)
{
   return CAL2_ke.EvalDerivS1(trans, ip, state1, state2, state3);
}

double CAL2keCoefficient::EvalDerivS2(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state1,
    double state2,
    double state3)
{
   return CAL2_ke.EvalDerivS2(trans, ip, state1, state2, state3);
}

double CAL2keCoefficient::EvalDerivS3(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state1,
    double state2,
    double state3)
{
   return CAL2_ke.EvalDerivS3(trans, ip, state1, state2, state3);
}

double CAL2keCoefficient::Eval2ndDerivS1(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state1,
    double state2,
    double state3)
{
   return CAL2_ke.Eval2ndDerivS1(trans, ip, state1, state2, state3);
}

double CAL2keCoefficient::Eval2ndDerivS2(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state1,
    double state2,
    double state3)
{
   return CAL2_ke.Eval2ndDerivS2(trans, ip, state1, state2, state3);
}

double CAL2keCoefficient::Eval2ndDerivS3(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state1,
    double state2,
    double state3)
{
   return CAL2_ke.Eval2ndDerivS3(trans, ip, state1, state2, state3);
}

double CAL2keCoefficient::Eval2ndDerivS1S2(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state1,
    double state2,
    double state3)
{
   return CAL2_ke.Eval2ndDerivS1S2(trans, ip, state1, state2, state3);
}

double CAL2keCoefficient::Eval2ndDerivS1S3(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state1,
    double state2,
    double state3)
{
   return CAL2_ke.Eval2ndDerivS1S3(trans, ip, state1, state2, state3);
}

double CAL2keCoefficient::Eval2ndDerivS2S3(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state1,
    double state2,
    double state3)
{
   return CAL2_ke.Eval2ndDerivS2S3(trans, ip, state1, state2, state3);
}

///TODO: Likely not necessary because of Eval2ndDerivS1S2
double CAL2keCoefficient::Eval2ndDerivS2S1(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state1,
    double state2,
    double state3)
{
   return CAL2_ke.Eval2ndDerivS2S1(trans, ip, state1, state2, state3);
}

///TODO: Likely not necessary because of Eval2ndDerivS1S3
double CAL2keCoefficient::Eval2ndDerivS3S1(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state1,
    double state2,
    double state3)
{
   return CAL2_ke.Eval2ndDerivS3S1(trans, ip, state1, state2, state3);
}

///TODO: Likely not necessary because of Eval2ndDerivS2S3
double CAL2keCoefficient::Eval2ndDerivS3S2(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state1,
    double state2,
    double state3)
{
   return CAL2_ke.Eval2ndDerivS3S2(trans, ip, state1, state2, state3);
}

/// TODO: Adapt if needed
void CAL2keCoefficient::EvalRevDiff(const double Q_bar,
                                         mfem::ElementTransformation &trans,
                                         const mfem::IntegrationPoint &ip,
                                         mfem::DenseMatrix &PointMat_bar)
{
   CAL2_ke.EvalRevDiff(Q_bar, trans, ip, PointMat_bar);
}

/// TODO: Change CAL2_ke(std::make_unique<mfem::ConstantCoefficient>(1.0)) line IF the equivalent lines...
/// std::unique_ptr<mfem::Coefficient> constructDefaultCAL2_keCoeff( from earlier change
CAL2keCoefficient::CAL2keCoefficient(const nlohmann::json &CAL2_ke_options,
                                               const nlohmann::json &materials)
 : CAL2_ke(std::make_unique<mfem::ConstantCoefficient>(1.0)) 
{
   if (CAL2_ke_options.contains("components"))
   {
      /// Options are being passed in. Loop over the components within and construct a CAL2_ke loss coefficient for each
      for (const auto &component : CAL2_ke_options["components"])
      {
         int attr = component.value("attr", -1);
         if (-1 != attr)
         {
            CAL2_ke.addCoefficient(attr,
                              constructDefaultCAL2_keCoeff(component, materials));
         }
         else
         {
            for (const auto &attribute : component["attrs"])
            {
               CAL2_ke.addCoefficient(attribute,
                                 constructCAL2_keCoeff(component, materials));
            }
         }
      }
   }
   else
   {
      /// Components themselves are being passed in. Loop over the components and construct a CAL2_ke loss coefficient for each
      auto components = CAL2_ke_options;
      for (const auto &component : components)
      {
         int attr = component.value("attr", -1);
         if (-1 != attr)
         {
            CAL2_ke.addCoefficient(attr,
                              constructDefaultCAL2_keCoeff(component, materials));
         }
         else
         {
            for (const auto &attribute : component["attrs"])
            {
               CAL2_ke.addCoefficient(attribute,
                                 constructCAL2_keCoeff(component, materials));
            }
         }
      }
   }
}

}  // namespace mach

namespace
{
PolyVarEddyCurrentLossCoeff::PolyVarEddyCurrentLossCoeff(const double &T0,
                        const std::vector<double> &ke_T0,
                        const double &T1,
                        const std::vector<double> &ke_T1)
 : T0(T0), ke_T0(ke_T0), T1(T1), ke_T1(ke_T1)

///TODO: As needed, add in more definitions of protected class members here
{
  
///TODO: As needed, add in calculations of protected class members here

}

double PolyVarEddyCurrentLossCoeff::Eval(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           const double state1,
                           const double state2,
                           const double state3)
{
   ///TODO: As needed, utilize logic of protected class members to eval CAL2_ke

   // Assuming state1=temperature, state2=frequency, state3=max alternating flux density
   auto T = state1;
   auto f = state2;
   auto Bm = state3;

   double ke_T0_f_B = 0.0;
   for (int i = 0; i < static_cast<int>(ke_T0.size()); ++i)
   {
      ke_T0_f_B += ke_T0[i]*std::pow(Bm,i);
   }
   double ke_T1_f_B = 0.0;
   for (int i = 0; i < static_cast<int>(ke_T1.size()); ++i)
   {
      ke_T1_f_B += ke_T1[i]*std::pow(Bm,i);
   }
   double D_eddy = (ke_T1_f_B-ke_T0_f_B)/((T1-T0)*ke_T0_f_B);
   double kte = 1+(T-T0)*D_eddy;
   
   double CAL2_ke = kte*ke_T0_f_B;

   return CAL2_ke;
}

double PolyVarEddyCurrentLossCoeff::EvalDerivS1(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           const double state1,
                           const double state2,
                           const double state3)
{
   ///TODO: As needed, utilize logic of protected class members to eval CAL2_ke

   ///TODO: Derived overall derivative in Overleaf. Need to modify for just the CAL2_ke, code the below, then uncomment
   /*
   double dCAL2_kedf = 
   return dCAL2_kedf;
   */
}

double PolyVarEddyCurrentLossCoeff::EvalDerivS2(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           const double state1,
                           const double state2,
                           const double state3)
{
   ///TODO: As needed, utilize logic of protected class members to eval CAL2_ke

   ///TODO: Derived overall derivative in Overleaf. Need to modify for just the CAL2_ke, code the below, then uncomment
   /*
   double dCAL2_kedB = 
   return dCAL2_kedB;
   */
}

double PolyVarEddyCurrentLossCoeff::EvalDerivS3(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           const double state1,
                           const double state2,
                           const double state3)
{
   ///TODO: As needed, utilize logic of protected class members to eval CAL2_ke

   ///TODO: Derived overall derivative in Overleaf. Need to modify for just the CAL2_ke, code the below, then uncomment
   /*
   double dCAL2_kedT = 
   return dCAL2_kedT;
   */
}

double PolyVarEddyCurrentLossCoeff::Eval2ndDerivS1(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           const double state1,
                           const double state2,
                           const double state3)
{
   ///TODO: As needed, utilize logic of protected class members to eval CAL2_ke

   ///TODO: Derive and code the below, then uncomment
   /*
   double d2CAL2_kedf2 = 
   return d2CAL2_kedf2;
   */
}

double PolyVarEddyCurrentLossCoeff::Eval2ndDerivS2(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           const double state1,
                           const double state2,
                           const double state3)
{
   ///TODO: As needed, utilize logic of protected class members to eval CAL2_ke

   ///TODO: Derive and code the below, then uncomment
   /*
   double d2CAL2_kedB2 = 
   return d2CAL2_kedB2;
   */
}

double PolyVarEddyCurrentLossCoeff::Eval2ndDerivS3(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           const double state1,
                           const double state2,
                           const double state3)
{
   ///TODO: As needed, utilize logic of protected class members to eval CAL2_ke

   ///TODO: Derive and code the below, then uncomment
   /*
   double d2CAL2_kedT2 = 
   return d2CAL2_kedT2;
   */
}

double PolyVarEddyCurrentLossCoeff::Eval2ndDerivS1S2(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           const double state1,
                           const double state2,
                           const double state3)
{
   ///TODO: As needed, utilize logic of protected class members to eval CAL2_ke

   ///TODO: Derive and code the below, then uncomment
   /*
   double d2CAL2_kedfdB = 
   return d2CAL2_kedfdB;
   */
}

double PolyVarEddyCurrentLossCoeff::Eval2ndDerivS1S3(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           const double state1,
                           const double state2,
                           const double state3)
{
   ///TODO: As needed, utilize logic of protected class members to eval CAL2_ke

   ///TODO: Derive and code the below, then uncomment
   /*
   double d2CAL2_kedfdT = 
   return d2CAL2_kedfdT;
   */
}

double PolyVarEddyCurrentLossCoeff::Eval2ndDerivS2S3(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           const double state1,
                           const double state2,
                           const double state3)
{
   ///TODO: As needed, utilize logic of protected class members to eval CAL2_ke

   ///TODO: Derive and code the below, then uncomment
   /*
   double d2CAL2_kedBdT = 
   return d2CAL2_kedBdT;
   */
}

///TODO: Likely not necessary because of Eval2ndDerivS1S2
double PolyVarEddyCurrentLossCoeff::Eval2ndDerivS2S1(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           const double state1,
                           const double state2,
                           const double state3)
{
   ///TODO: As needed, utilize logic of protected class members to eval CAL2_ke

   ///TODO: Derive and code the below, then uncomment
   /*
   double d2CAL2_kedBdf = 
   return d2CAL2_kedBdf;
   */
}

///TODO: Likely not necessary because of Eval2ndDerivS1S3
double PolyVarEddyCurrentLossCoeff::Eval2ndDerivS3S1(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           const double state1,
                           const double state2,
                           const double state3)
{
   ///TODO: As needed, utilize logic of protected class members to eval CAL2_ke

   ///TODO: Derive and code the below, then uncomment
   /*
   double d2CAL2_kedTdf = 
   return d2CAL2_kedTdf;
   */
}

///TODO: Likely not necessary because of Eval2ndDerivS2S3
double PolyVarEddyCurrentLossCoeff::Eval2ndDerivS3S2(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           const double state1,
                           const double state2,
                           const double state3)
{
   ///TODO: As needed, utilize logic of protected class members to eval CAL2_ke

   ///TODO: Derive and code the below, then uncomment
   /*
   double d2CAL2_kedTdB = 
   return d2CAL2_kedTdB;
   */
}

///TODO: is there a need to code EvalRevDiff for variable eddy current coefficient method here? I'm thinking not

}

