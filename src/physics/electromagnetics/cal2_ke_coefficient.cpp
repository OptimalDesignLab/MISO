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
/// TODO: Ensure all states will be doubles (they may be grid functions or something else)
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
/// TODO: determine protected outputs that will be useful (protected meaning child classes can access too, but other classes cannot) 
   double T0, T1;
   std::vector<double> ke_T0, ke_T1;
};

/// TODO: this default variable eddy current loss coefficient will need to be altered
/// TODO: this will need some work -- COME BACK TO THIS (possibly even remove)
std::unique_ptr<mfem::Coefficient> constructDefaultCAL2_keCoeff(
    const std::string &material_name,
    const nlohmann::json &materials)
{
   auto CAL2_ke = materials[material_name].value("ke", 1.0);
   return std::make_unique<mfem::ConstantCoefficient>(ke);
}

std::unique_ptr<mfem::Coefficient> constructCAL2_keCoeff(
    const nlohmann::json &component,
    const nlohmann::json &materials)
{
   std::unique_ptr<mfem::Coefficient> temp_coeff; // temp=temporary, not temperature
   const auto &material = component["material"];

   /// If "material" is a string, it is interpreted to be the name of a 
   /// material. We default to a CAL2_ke coeff of ///TODO: (insert default value here) unless
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

      ///TODO: Perhaps soften this if statement (impose default T0 and T1 values, for example)
      if (materials[material_name]["core_loss"]["CAL2"].contains("T0") && 
         materials[material_name]["core_loss"]["CAL2"].contains("ke_T0") && 
         materials[material_name]["core_loss"]["CAL2"].contains("ke_T1"))
      {
         double T0 = material["T0"].get<double>();
         std::vector<double> ke_T0 = materials[material_name]["core_loss"]
                                 ["CAL2"]["ke_T0"].get<std::vector<double>>();
         double T1 = material["T1"].get<double>();
         std::vector<double> ke_T1 = materials[material_name]["core_loss"]
                                 ["CAL2"]["ke_T1"].get<std::vector<double>>();

         temp_coeff = std::make_unique<PolyVarEddyCurrentLossCoeff>(
            T0, ke_T0, T1, ke_T1);
      }
      else
      {
         std::string error_msg =
               "Insufficient information to compute CAL2 variable eddy current loss coefficient for material \"";
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

double CAL2keCoefficient::EvalDerivS1S2(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state1,
    double state2,
    double state3)
{
   return CAL2_ke.EvalDerivS1S2(trans, ip, state1, state2, state3);
}

double CAL2keCoefficient::EvalDerivS1S3(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state1,
    double state2,
    double state3)
{
   return CAL2_ke.EvalDerivS1S3(trans, ip, state1, state2, state3);
}

double CAL2keCoefficient::EvalDerivS2S3(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state1,
    double state2,
    double state3)
{
   return CAL2_ke.EvalDerivS2S3(trans, ip, state1, state2, state3);
}

///TODO: Likely not necessary because of Eval2ndDerivS1S2
double CAL2keCoefficient::EvalDerivS2S1(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state1,
    double state2,
    double state3)
{
   return CAL2_ke.EvalDerivS2S1(trans, ip, state1, state2, state3);
}

///TODO: Likely not necessary because of Eval2ndDerivS1S3
double CAL2keCoefficient::EvalDerivS3S1(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state1,
    double state2,
    double state3)
{
   return CAL2_ke.EvalDerivS3S1(trans, ip, state1, state2, state3);
}

///TODO: Likely not necessary because of Eval2ndDerivS2S3
double CAL2keCoefficient::EvalDerivS3S2(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state1,
    double state2,
    double state3)
{
   return CAL2_ke.EvalDerivS3S2(trans, ip, state1, state2, state3);
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
   /// loop over all components, construct a CAL2_ke loss coefficient for each
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
   for (int i = 0; i < ke_T0.size(); ++i)
   {
      ke_T0_f_B += ke_T0[i]*pow(Bm,i);
   }
   double ke_T1_f_B = 0.0;
   for (int i = 0; i < ke_T1.size(); ++i)
   {
      ke_T1_f_B += ke_T1[i]*pow(Bm,i);
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

///TODO: After code derivatives, circle back around and check this and hpp file for logic/consistency

