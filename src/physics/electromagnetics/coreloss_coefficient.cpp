#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "mfem.hpp"
#include "nlohmann/json.hpp"
#include "tinysplinecxx.h"

#include "coreloss_coefficient.hpp"
#include "utils.hpp"

#include "coefficient.hpp"
#include "mach_input.hpp"

namespace
{
/// TODO: If needed, uncomment the below global variables and/or add in necessary global variables
/// permeability of free space
// constexpr double mu_0 = 4e-7 * M_PI;
// constexpr double nu0 = 1 / mu_0;

class TempDepCoreLossCoefficient : public mach::ThreeStateCoefficient
{
public:
   /// \brief Define a model to represent the temperature dependent core losses as calculated by the two term loss separation model, 
   ///      empirically derived from a data source (NASA, Carpenter, ADA, etc.)
   /// \param[in] rho - TODO: material density? 
   /// \param[in] kh - vector of variable hysteresis loss coefficients, empirically found.
   ///                     kh=[kh0, kh1, kh2, ...], kh(B)=kh0+kh1*B+kh2*B^2...
   /// \param[in] ke - vector of variable eddy current loss coefficients, empirically found.
   ///                     ke=[ke0, ke1, ke2, ...], ke(B)=ke0+ke1*B+ke2*B^2...
   /// \param[in] A - magnetic vector potential GridFunction
   TempDepCoreLossCoefficient(double &rho,
                        const std::vector<double> &kh,
                        const std::vector<double> &ke,
                        mfem::GridFunction &A);

   ///TODO: Figure out how to handle A here AND throughout file

   /// \brief Evaluate the temp dep core loss in the element described by trans at the
   /// point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double Eval(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               double state1,
               double state2,
               double state3) override;

   /// \brief Evaluate the derivative of the temp dep core loss in the element with respect to the 1st state variable
   ///   described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double EvalDerivS1(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               double state1,
               double state2,
               double state3) override;

   /// \brief Evaluate the derivative of the temp dep core loss in the element with respect to the 2nd state variable
   ///   described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double EvalDerivS2(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               double state1,
               double state2,
               double state3) override;

   /// \brief Evaluate the derivative of the temp dep core loss in the element with respect to the 3rd state variable
   ///   described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double EvalDerivS3(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               double state1,
               double state2,
               double state3) override;

   /// \brief Evaluate the second derivative of the temp dep core loss in the element with respect to the 1st state variable
   ///   described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double Eval2ndDerivS1(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               double state1,
               double state2,
               double state3) override;

   /// \brief Evaluate the derivative of the temp dep core loss in the element with respect to the 2nd state variable
   ///   described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double Eval2ndDerivS2(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               double state1,
               double state2,
               double state3) override;

   /// \brief Evaluate the derivative of the temp dep core loss in the element with respect to the 3rd state variable
   ///   described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double Eval2ndDerivS3(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               double state1,
               double state2,
               double state3) override;

   /// \brief Evaluate the second derivative of the temp dep core loss in the element with respect to the 1st then 2nd state variable
   ///   described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double Eval2ndDerivS1S2(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               double state1,
               double state2,
               double state3) override;

   /// \brief Evaluate the second derivative of the temp dep core loss in the element with respect to the 1st then 3rd state variable
   ///   described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double Eval2ndDerivS1S3(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               double state1,
               double state2,
               double state3) override;

   /// \brief Evaluate the second derivative of the temp dep core loss in the element with respect to the 2nd then 3rd state variable
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
   /// \brief Evaluate the derivative of the temp dep core loss in the element with respect to the 2nd then 1st state variable
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
   /// \brief Evaluate the derivative of the temp dep core loss in the element with respect to the 3rd then 1st state variable
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
   /// \brief Evaluate the derivative of the temp dep core loss in the element with respect to the 3rd then 2nd state variable
   ///   described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double Eval2ndDerivS3S2(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               double state1,
               double state2,
               double state3) override;

   /// TODO: Adapt EvalRevDiff as needed for temp dep core losses
   void EvalRevDiff(double Q_bar,
                    mfem::ElementTransformation &trans,
                    const mfem::IntegrationPoint &ip,
                    mfem::DenseMatrix &PointMat_bar) override;

protected:
/// TODO: determine protected outputs that will be useful (protected meaning child classes can access too, but other classes cannot) 
   double rho, kh, ke;
   mfem::GridFunction &A;
};

/// Brought this in from coefficient.hpp file (and adapted) so have the option to compute core losses as were done before
/// Translated from an mfem::Coeffient to convert to a mach::ThreeStateCoefficient
/// TODO: Remove/overwrite the SteinmetzCoefficient class in coefficient.cpp and .hpp as needed
class SteinmetzCoefficient : public mach::ThreeStateCoefficient
{
public:
   /// \brief Define a coefficient to represent the Steinmetz core losses as calculated by the modified Steinmetz equation, 
   ///      empirically derived from a data source
   /// \param[in] rho - TODO: material density?
   /// \param[in] alpha - empirically derived material dependent constant that frequency is raised to (f^alpha)
   /// \param[in] ks - empirically derived material dependent constant that scales the losses through multiplication
   /// \param[in] beta - empirically derived material dependent constant that flux density is raised to (B^beta)
   /// \param[in] A - magnetic vector potential GridFunction
   SteinmetzCoefficient(double &rho,
                        double &alpha,
                        double &ks,
                        double &beta,
                        mfem::GridFunction &A);
   ///TODO: Do I need to add in the equivalent of lines 589 and 590 of coefficient.hpp here? Probably not because doing that in implementation in lower anonymous namespace

   /// \brief Evaluate the Steinmetz core loss in the element described by trans at the
   /// point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double Eval(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               double state1,
               double state2,
               double state3) override;

   ///TODO: Remove if needed. This method was not previously specifically defined for Steinmetz
   /// \brief Evaluate the derivative of the Steinmetz core loss in the element with respect to the 1st state variable
   ///   described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double EvalDerivS1(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               double state1,
               double state2,
               double state3) override;

   ///TODO: Remove if needed. This method was not previously specifically defined for Steinmetz
   /// \brief Evaluate the derivative of the Steinmetz core loss in the element with respect to the 2nd state variable
   ///   described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double EvalDerivS2(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               double state1,
               double state2,
               double state3) override;
   
   ///TODO: Remove if needed. This method was not previously specifically defined for Steinmetz
   /// \brief Evaluate the derivative of the Steinmetz core loss in the element with respect to the 3rd state variable
   ///   described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double EvalDerivS3(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               double state1,
               double state2,
               double state3) override;

   ///TODO: Remove if needed. This method was not previously specifically defined for Steinmetz
   /// \brief Evaluate the second derivative of the Steinmetz core loss in the element with respect to the 1st state variable
   ///   described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double Eval2ndDerivS1(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               double state1,
               double state2,
               double state3) override;

   ///TODO: Remove if needed. This method was not previously specifically defined for Steinmetz
   /// \brief Evaluate the derivative of the Steinmetz core loss in the element with respect to the 2nd state variable
   ///   described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double Eval2ndDerivS2(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               double state1,
               double state2,
               double state3) override;

   ///TODO: Remove if needed. This method was not previously specifically defined for Steinmetz
   /// \brief Evaluate the derivative of the Steinmetz core loss in the element with respect to the 3rd state variable
   ///   described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double Eval2ndDerivS3(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               double state1,
               double state2,
               double state3) override;

   ///TODO: Remove if needed. This method was not previously specifically defined for Steinmetz
   /// \brief Evaluate the second derivative of the Steinmetz core loss in the element with respect to the 1st then 2nd state variable
   ///   described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double Eval2ndDerivS1S2(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               double state1,
               double state2,
               double state3) override;

   ///TODO: Remove if needed. This method was not previously specifically defined for Steinmetz
   /// \brief Evaluate the second derivative of the Steinmetz core loss in the element with respect to the 1st then 3rd state variable
   ///   described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double Eval2ndDerivS1S3(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               double state1,
               double state2,
               double state3) override;

   ///TODO: Remove if needed. This method was not previously specifically defined for Steinmetz
   /// \brief Evaluate the second derivative of the Steinmetz core loss in the element with respect to the 2nd then 3rd state variable
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
   /// \brief Evaluate the derivative of the Steinmetz core loss in the element with respect to the 2nd then 1st state variable
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
   /// \brief Evaluate the derivative of the Steinmetz core loss in the element with respect to the 3rd then 1st state variable
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
   /// \brief Evaluate the derivative of the Steinmetz core loss in the element with respect to the 3rd then 2nd state variable
   ///   described by trans at the point ip.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double Eval2ndDerivS3S2(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               double state1,
               double state2,
               double state3) override;

   /// Evaluate the derivative of the Steinmetz coefficient with respect to x
   void EvalRevDiff(double Q_bar,
                    mfem::ElementTransformation &trans,
                    const mfem::IntegrationPoint &ip,
                    mfem::DenseMatrix &PointMat_bar) override;

private:
   // double rho, alpha, freq, kh, ke;
   double rho, alpha, ks, beta;
   mfem::GridFunction &A;
};

/// TODO: this default core loss coefficient will need to be altered
/// TODO: this will need some work -- COME BACK TO THIS (possibly even remove)
std::unique_ptr<mfem::Coefficient> constructDefaultCoreLossCoeff(
    const std::string &material_name,
    const nlohmann::json &materials)
{
   auto pFe = materials[material_name].value("pFe", 1.0);
   return std::make_unique<mfem::ConstantCoefficient>(pFe);
}

std::unique_ptr<mfem::Coefficient> constructCoreLossCoeff(
    const nlohmann::json &component,
    const nlohmann::json &materials)
{
   std::unique_ptr<mfem::Coefficient> temp_coeff; // temp=temporary, not temperature
   const auto &material = component["material"];

   /// If "material" is a string, it is interpreted to be the name of a 
   /// material. We default to a core loss of ///TODO: (insert default value here) unless
   /// there is a different value in the material library
   if (material.is_string())
   {
      const auto &material_name = material.get<std::string>();
      ///TODO: Ensure this Default core loss coefficient is in order
      temp_coeff = constructDefaultCoreLossCoeff(material_name, materials);
   }
   else
   {
      const auto &material_name = material["name"].get<std::string>();

      ///TODO: Move this obtaining of density as needed, and/or change default value
      double rho = materials[material_name].value("rho", 1.0);

      if (material.contains("kh") && material.contains("ke"))
      {
         std::vector<double> kh = material["kh"].get<std::vector<double>>();
         std::vector<double> ke = material["ke"].get<std::vector<double>>();

         ///TODO: temp_coeff = std::make_unique<TempDepCoreLossCoefficient...
      }
      else if (material.contains("alpha") && material.contains("beta") && material.contains("ks"))
      {
         double alpha = material["alpha"].get<double>();
         double beta = material["beta"].get<double>();
         double ks = material["ks"].get<double>();

         ///TODO: temp_coeff = std::make_unique<SteinmetzCoefficient...
      }
      else
      {
         std::string error_msg =
               "Insufficient information to compute core loss for material \"";
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
double CoreLossCoefficient::Eval(mfem::ElementTransformation &trans,
                                    const mfem::IntegrationPoint &ip)
{
   return pFe.Eval(trans, ip);
}

double CoreLossCoefficient::Eval(mfem::ElementTransformation &trans,
                                    const mfem::IntegrationPoint &ip,
                                    double state1,
                                    double state2,
                                    double state3)
{
   return pFe.Eval(trans, ip, state1, state2, state3);
}

double CoreLossCoefficient::EvalDerivS1(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state1,
    double state2,
    double state3)
{
   return pFe.EvalDerivS1(trans, ip, state1, state2, state3);
}

double CoreLossCoefficient::EvalDerivS2(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state1,
    double state2,
    double state3)
{
   return pFe.EvalDerivS2(trans, ip, state1, state2, state3);
}

double CoreLossCoefficient::EvalDerivS3(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state1,
    double state2,
    double state3)
{
   return pFe.EvalDerivS3(trans, ip, state1, state2, state3);
}

double CoreLossCoefficient::Eval2ndDerivS1(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state1,
    double state2,
    double state3)
{
   return pFe.Eval2ndDerivS1(trans, ip, state1, state2, state3);
}

double TempDepCoreLossCoefficient::Eval2ndDerivS2(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state1,
    double state2,
    double state3)
{
   return pFe.Eval2ndDerivS2(trans, ip, state1, state2, state3);
}

double TempDepCoreLossCoefficient::Eval2ndDerivS3(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state1,
    double state2,
    double state3)
{
   return pFe.Eval2ndDerivS3(trans, ip, state1, state2, state3);
}

double TempDepCoreLossCoefficient::EvalDerivS1S2(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state1,
    double state2,
    double state3)
{
   return pFe.EvalDerivS1S2(trans, ip, state1, state2, state3);
}

double TempDepCoreLossCoefficient::EvalDerivS1S3(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state1,
    double state2,
    double state3)
{
   return pFe.EvalDerivS1S3(trans, ip, state1, state2, state3);
}

double TempDepCoreLossCoefficient::EvalDerivS2S3(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state1,
    double state2,
    double state3)
{
   return pFe.EvalDerivS2S3(trans, ip, state1, state2, state3);
}

///TODO: Likely not necessary because of Eval2ndDerivS1S2
double TempDepCoreLossCoefficient::EvalDerivS2S1(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state1,
    double state2,
    double state3)
{
   return pFe.EvalDerivS2S1(trans, ip, state1, state2, state3);
}

///TODO: Likely not necessary because of Eval2ndDerivS1S3
double TempDepCoreLossCoefficient::EvalDerivS3S1(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state1,
    double state2,
    double state3)
{
   return pFe.EvalDerivS3S1(trans, ip, state1, state2, state3);
}

///TODO: Likely not necessary because of Eval2ndDerivS2S3
double TempDepCoreLossCoefficient::EvalDerivS3S2(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state1,
    double state2,
    double state3)
{
   return pFe.EvalDerivS3S2(trans, ip, state1, state2, state3);
}

/// TODO: Adapt if keeping, remove if not
void TempDepCoreLossCoefficient::EvalRevDiff(const double Q_bar,
                                         mfem::ElementTransformation &trans,
                                         const mfem::IntegrationPoint &ip,
                                         mfem::DenseMatrix &PointMat_bar)
{
   pFe.EvalRevDiff(Q_bar, trans, ip, PointMat_bar);
}

/// TODO: Change pFe(std::make_unique<mfem::ConstantCoefficient>(1.0)) line IF the equivalent lines...
/// std::unique_ptr<mfem::Coefficient> constructDefaultCoreLossCoeff( from earlier change
CoreLossCoefficient::CoreLossCoefficient(const nlohmann::json &pFe_options,
                                               const nlohmann::json &materials)
 : pFe(std::make_unique<mfem::ConstantCoefficient>(1.0)) 
{
   /// loop over all components, construct a core loss coefficient for each
   for (const auto &component : pFe_options["components"])
   {
      int attr = component.value("attr", -1);
      if (-1 != attr)
      {
         pFe.addCoefficient(attr,
                           constructDefaultCoreLossCoeff(component, materials));
      }
      else
      {
         for (const auto &attribute : component["attrs"])
         {
            pFe.addCoefficient(attribute,
                              constructCoreLossCoeff(component, materials));
         }
      }
   }
}

}  // namespace mach

namespace
{
TempDepCoreLossCoefficient::TempDepCoreLossCoefficient(
   double &rho,
   const std::vector<double> &kh,
   const std::vector<double> &ke,
   mfem::GridFunction &A)
 : rho(rho), kh(kh), ke(ke), A(A)

///TODO: As needed, add in more definitions of protected class members here
{
  
///TODO: As needed, add in calculations of protected class members here

}

double TempDepCoreLossCoefficient::Eval(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           const double state1,
                           const double state2,
                           const double state3)
{
   ///TODO: As needed, utilize logic of protected class members to eval pFe

   ///TODO: Derive and code the below, then uncomment
   /*
   double pFe = 
   return pFe;
   */
}

double TempDepCoreLossCoefficient::EvalDerivS1(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           const double state1,
                           const double state2,
                           const double state3)
{
   ///TODO: As needed, utilize logic of protected class members to eval pFe

   ///TODO: Derive and code the below, then uncomment
   /*
   double dpFedf = 
   return dpFedf;
   */
}

double TempDepCoreLossCoefficient::EvalDerivS2(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           const double state1,
                           const double state2,
                           const double state3)
{
   ///TODO: As needed, utilize logic of protected class members to eval pFe

   ///TODO: Derive and code the below, then uncomment
   /*
   double dpFedB = 
   return dpFedB;
   */
}

double TempDepCoreLossCoefficient::EvalDerivS3(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           const double state1,
                           const double state2,
                           const double state3)
{
   ///TODO: As needed, utilize logic of protected class members to eval pFe

   ///TODO: Derive and code the below, then uncomment
   /*
   double dpFedT = 
   return dpFedT;
   */
}

double TempDepCoreLossCoefficient::Eval2ndDerivS1(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           const double state1,
                           const double state2,
                           const double state3)
{
   ///TODO: As needed, utilize logic of protected class members to eval pFe

   ///TODO: Derive and code the below, then uncomment
   /*
   double d2pFedf2 = 
   return d2pFedf2;
   */
}

double TempDepCoreLossCoefficient::Eval2ndDerivS2(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           const double state1,
                           const double state2,
                           const double state3)
{
   ///TODO: As needed, utilize logic of protected class members to eval pFe

   ///TODO: Derive and code the below, then uncomment
   /*
   double d2pFedB2 = 
   return d2pFedB2;
   */
}

double TempDepCoreLossCoefficient::Eval2ndDerivS3(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           const double state1,
                           const double state2,
                           const double state3)
{
   ///TODO: As needed, utilize logic of protected class members to eval pFe

   ///TODO: Derive and code the below, then uncomment
   /*
   double d2pFedT2 = 
   return d2pFedT2;
   */
}

double TempDepCoreLossCoefficient::Eval2ndDerivS1S2(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           const double state1,
                           const double state2,
                           const double state3)
{
   ///TODO: As needed, utilize logic of protected class members to eval pFe

   ///TODO: Derive and code the below, then uncomment
   /*
   double d2pFedfdB = 
   return d2pFedfdB;
   */
}

double TempDepCoreLossCoefficient::Eval2ndDerivS1S3(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           const double state1,
                           const double state2,
                           const double state3)
{
   ///TODO: As needed, utilize logic of protected class members to eval pFe

   ///TODO: Derive and code the below, then uncomment
   /*
   double d2pFedfdT = 
   return d2pFedfdT;
   */
}

double TempDepCoreLossCoefficient::Eval2ndDerivS2S3(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           const double state1,
                           const double state2,
                           const double state3)
{
   ///TODO: As needed, utilize logic of protected class members to eval pFe

   ///TODO: Derive and code the below, then uncomment
   /*
   double d2pFedBdT = 
   return d2pFedBdT;
   */
}

///TODO: Likely not necessary because of Eval2ndDerivS1S2
double TempDepCoreLossCoefficient::Eval2ndDerivS2S1(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           const double state1,
                           const double state2,
                           const double state3)
{
   ///TODO: As needed, utilize logic of protected class members to eval pFe

   ///TODO: Derive and code the below, then uncomment
   /*
   double d2pFedBdf = 
   return d2pFedBdf;
   */
}

///TODO: Likely not necessary because of Eval2ndDerivS1S3
double TempDepCoreLossCoefficient::Eval2ndDerivS3S1(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           const double state1,
                           const double state2,
                           const double state3)
{
   ///TODO: As needed, utilize logic of protected class members to eval pFe

   ///TODO: Derive and code the below, then uncomment
   /*
   double d2pFedTdf = 
   return d2pFedTdf;
   */
}

///TODO: Likely not necessary because of Eval2ndDerivS2S3
double TempDepCoreLossCoefficient::Eval2ndDerivS3S2(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           const double state1,
                           const double state2,
                           const double state3)
{
   ///TODO: As needed, utilize logic of protected class members to eval pFe

   ///TODO: Derive and code the below, then uncomment
   /*
   double d2pFedTdB = 
   return d2pFedTdB;
   */
}

///TODO: is there a need to code EvalRevDiff for temp dep core loss method here?

SteinmetzCoefficient::SteinmetzCoefficient(
   double &rho,
   const double &alpha,
   const double &ks,
   const double &beta,
   mfem::GridFunction &A)
 : rho(rho), alpha(alpha), ks(ks), beta(beta), A(A)

///TODO: As needed, add in more definitions of protected class members here
{
  
///TODO: As needed, add in calculations of protected class members here

}

double SteinmetzCoefficient::Eval(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           const double state1,
                           const double state2,
                           const double state3)
{
   ///TODO: As needed, utilize logic of protected class members to eval pFe

   ///TODO: Derive and code the below, add in pseudo if statement and paste in code from coefficient.cpp, then uncomment
   /*

   double pFe = 
   return pFe;
   */
}

double SteinmetzCoefficient::EvalDerivS1(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           const double state1,
                           const double state2,
                           const double state3)
{
   ///TODO: As needed, utilize logic of protected class members to eval pFe

   ///TODO: Derive and code the below, add in pseudo if statement and paste in code from coefficient.cpp, then uncomment
   /*
   double dpFedf = 
   return dpFedf;
   */
}

double SteinmetzCoefficient::EvalDerivS2(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           const double state1,
                           const double state2,
                           const double state3)
{
   ///TODO: As needed, utilize logic of protected class members to eval pFe

   ///TODO: Derive and code the below, add in pseudo if statement and paste in code from coefficient.cpp, then uncomment
   /*
   double dpFedB = 
   return dpFedB;
   */
}

double SteinmetzCoefficient::EvalDerivS3(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           const double state1,
                           const double state2,
                           const double state3)
{
   ///TODO: As needed, utilize logic of protected class members to eval pFe

   ///TODO: Derive and code the below, add in pseudo if statement and paste in code from coefficient.cpp, then uncomment
   /*
   double dpFedT = 
   return dpFedT;
   */
}

double SteinmetzCoefficient::Eval2ndDerivS1(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           const double state1,
                           const double state2,
                           const double state3)
{
   ///TODO: As needed, utilize logic of protected class members to eval pFe

   ///TODO: Derive and code the below, add in pseudo if statement and paste in code from coefficient.cpp, then uncomment
   /*
   double d2pFedf2 = 
   return d2pFedf2;
   */
}

double SteinmetzCoefficient::Eval2ndDerivS2(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           const double state1,
                           const double state2,
                           const double state3)
{
   ///TODO: As needed, utilize logic of protected class members to eval pFe

   ///TODO: Derive and code the below, add in pseudo if statement and paste in code from coefficient.cpp, then uncomment
   /*
   double d2pFedB2 = 
   return d2pFedB2;
   */
}

double SteinmetzCoefficient::Eval2ndDerivS3(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           const double state1,
                           const double state2,
                           const double state3)
{
   ///TODO: As needed, utilize logic of protected class members to eval pFe

   ///TODO: Derive and code the below, add in pseudo if statement and paste in code from coefficient.cpp, then uncomment
   /*
   double d2pFedT2 = 
   return d2pFedT2;
   */
}

double SteinmetzCoefficient::Eval2ndDerivS1S2(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           const double state1,
                           const double state2,
                           const double state3)
{
   ///TODO: As needed, utilize logic of protected class members to eval pFe

   ///TODO: Derive and code the below, add in pseudo if statement and paste in code from coefficient.cpp, then uncomment
   /*
   double d2pFedfdB = 
   return d2pFedfdB;
   */
}

double SteinmetzCoefficient::Eval2ndDerivS1S3(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           const double state1,
                           const double state2,
                           const double state3)
{
   ///TODO: As needed, utilize logic of protected class members to eval pFe

   ///TODO: Derive and code the below, add in pseudo if statement and paste in code from coefficient.cpp, then uncomment
   /*
   double d2pFedfdT = 
   return d2pFedfdT;
   */
}

double SteinmetzCoefficient::Eval2ndDerivS2S3(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           const double state1,
                           const double state2,
                           const double state3)
{
   ///TODO: As needed, utilize logic of protected class members to eval pFe

   ///TODO: Derive and code the below, add in pseudo if statement and paste in code from coefficient.cpp, then uncomment
   /*
   double d2pFedBdT = 
   return d2pFedBdT;
   */
}

///TODO: Likely not necessary because of Eval2ndDerivS1S2
double SteinmetzCoefficient::Eval2ndDerivS2S1(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           const double state1,
                           const double state2,
                           const double state3)
{
   ///TODO: As needed, utilize logic of protected class members to eval pFe

   ///TODO: Derive and code the below, add in pseudo if statement and paste in code from coefficient.cpp, then uncomment
   /*
   double d2pFedBdf = 
   return d2pFedBdf;
   */
}

///TODO: Likely not necessary because of Eval2ndDerivS1S3
double SteinmetzCoefficient::Eval2ndDerivS3S1(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           const double state1,
                           const double state2,
                           const double state3)
{
   ///TODO: As needed, utilize logic of protected class members to eval pFe

   ///TODO: Derive and code the below, add in pseudo if statement and paste in code from coefficient.cpp, then uncomment
   /*
   double d2pFedTdf = 
   return d2pFedTdf;
   */
}

///TODO: Likely not necessary because of Eval2ndDerivS2S3
double SteinmetzCoefficient::Eval2ndDerivS3S2(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           const double state1,
                           const double state2,
                           const double state3)
{
   ///TODO: As needed, utilize logic of protected class members to eval pFe

   ///TODO: Derive and code the below, add in pseudo if statement and paste in code from coefficient.cpp, then uncomment
   /*
   double d2pFedTdB = 
   return d2pFedTdB;
   */
}

///TODO: is there a need to code EvalRevDiff for Steinmetz method here?

}

///TODO: After code derivatives, circle back around to ConstructCoreLossCoeff and check this and hpp file for logic/consistency

