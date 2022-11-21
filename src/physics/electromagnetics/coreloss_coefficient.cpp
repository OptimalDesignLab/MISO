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
/// TODO: Ensure all states will be doubles (they may be grid functions or something else)
class TempDepCoreLossCoefficient : public mach::ThreeStateCoefficient
{
public:
   /// \brief Define a model to represent the temperature dependent core losses as calculated by the two term loss separation model, 
   ///      empirically derived from a data source (NASA, Carpenter, ADA, etc.)
   /// \param[in] rho - TODO: material density? 
   /// \param[in] T0 - the lower temperature used for curve fitting kh(f,B) and ke(f,B)
   /// \param[in] kh_T0 - vector of variable hysteresis loss coefficients at temperature T0, empirically found.
   ///                     kh_T0=[kh0_T0, kh1_T0, kh2_T0, ...], kh_T0(B)=kh0_T0+kh1_T0*B+kh2_T0*B^2...
   /// \param[in] ke_T0 - vector of variable eddy current loss coefficients at temperature T0, empirically found.
   ///                     ke_T0=[ke0_T0, ke1_T0, ke2_T0, ...], ke_T0(B)=ke0_T0+ke1_T0*B+ke2_T0*B^2...
   /// \param[in] T1 - the upper temperature used for curve fitting kh(f,B) and ke(f,B)
   /// \param[in] kh_T1 - vector of variable hysteresis loss coefficients at temperature T1, empirically found.
   ///                     kh_T1=[kh0_T1, kh1_T1, kh2_T1, ...], kh_T1(B)=kh0_T1+kh1_T1*B+kh2_T1*B^2...
   /// \param[in] ke_T1 - vector of variable eddy current loss coefficients at temperature T1, empirically found.
   ///                     ke_T1=[ke0_T1, ke1_T1, ke2_T1, ...], ke_T1(B)=ke0_T1+ke1_T1*B+ke2_T1*B^2...
   /// \param[in] A - magnetic vector potential GridFunction
   TempDepCoreLossCoefficient(double &rho,
                        const double &T0,
                        const std::vector<double> &kh_T0,
                        const std::vector<double> &ke_T0,
                        const double &T1,
                        const std::vector<double> &kh_T1,
                        const std::vector<double> &ke_T1,
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

   /// TODO: Adapt EvalRevDiff if needed for temp dep core losses
   void EvalRevDiff(const double Q_bar,
                    mfem::ElementTransformation &trans,
                    const mfem::IntegrationPoint &ip,
                    mfem::DenseMatrix &PointMat_bar) override
   { }

protected:
/// TODO: determine protected outputs that will be useful (protected meaning child classes can access too, but other classes cannot) 
   double rho, T0, T1;
   std::vector<double> kh_T0, ke_T0, kh_T1, ke_T1;
   mfem::GridFunction &A;
};

/// Brought this in from coefficient.hpp file (and adapted) so have the option to compute core losses as were done before
/// Translated from an mfem::Coeffient to a mach::ThreeStateCoefficient
/// TODO: UPDATE/REMOVE THIS CLASS. NEED TO FOCUS EFFORT ON SteinmetzLossIntegrator
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

      ///TODO: Perhaps soften this if statement (impose default T0 and T1 values, for example)
      if (material.contains("T0") && material.contains("kh_T0") && material.contains("ke_T0") && material.contains("T1") && material.contains("kh_T1") && material.contains("ke_T1"))
      {
         double T0 = material["T0"].get<double>();
         std::vector<double> kh_T0 = material["kh_T0"].get<std::vector<double>>();
         std::vector<double> ke_T0 = material["ke_T0"].get<std::vector<double>>();
         double T1 = material["T1"].get<double>();
         std::vector<double> kh_T1 = material["kh_T1"].get<std::vector<double>>();
         std::vector<double> ke_T1 = material["ke_T1"].get<std::vector<double>>();

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

double CoreLossCoefficient::Eval2ndDerivS2(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state1,
    double state2,
    double state3)
{
   return pFe.Eval2ndDerivS2(trans, ip, state1, state2, state3);
}

double CoreLossCoefficient::Eval2ndDerivS3(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state1,
    double state2,
    double state3)
{
   return pFe.Eval2ndDerivS3(trans, ip, state1, state2, state3);
}

double CoreLossCoefficient::EvalDerivS1S2(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state1,
    double state2,
    double state3)
{
   return pFe.EvalDerivS1S2(trans, ip, state1, state2, state3);
}

double CoreLossCoefficient::EvalDerivS1S3(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state1,
    double state2,
    double state3)
{
   return pFe.EvalDerivS1S3(trans, ip, state1, state2, state3);
}

double CoreLossCoefficient::EvalDerivS2S3(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state1,
    double state2,
    double state3)
{
   return pFe.EvalDerivS2S3(trans, ip, state1, state2, state3);
}

///TODO: Likely not necessary because of Eval2ndDerivS1S2
double CoreLossCoefficient::EvalDerivS2S1(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state1,
    double state2,
    double state3)
{
   return pFe.EvalDerivS2S1(trans, ip, state1, state2, state3);
}

///TODO: Likely not necessary because of Eval2ndDerivS1S3
double CoreLossCoefficient::EvalDerivS3S1(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state1,
    double state2,
    double state3)
{
   return pFe.EvalDerivS3S1(trans, ip, state1, state2, state3);
}

///TODO: Likely not necessary because of Eval2ndDerivS2S3
double CoreLossCoefficient::EvalDerivS3S2(
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    double state1,
    double state2,
    double state3)
{
   return pFe.EvalDerivS3S2(trans, ip, state1, state2, state3);
}

/// TODO: Adapt if needed
void CoreLossCoefficient::EvalRevDiff(const double Q_bar,
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
   const double &T0,
   const std::vector<double> &kh_T0,
   const std::vector<double> &ke_T0,
   const double &T1,
   const std::vector<double> &kh_T1,
   const std::vector<double> &ke_T1,
   mfem::GridFunction &A)
 : rho(rho), T0(T0), kh_T0(kh_T0), ke_T0(ke_T0), T1(T1), kh_T1(kh_T1), ke_T1(ke_T1), A(A)

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

   // Assuming state1=temperature, state2=frequency, state3=max alternating flux density
   auto T = state1;
   auto f = state2;
   auto Bm = state3;

   ///TODO: Double check the below is all good
   // First, the hysteresis loss term subequations
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
   double kh_T_f_B = kth*kh_T0_f_B;

   // Second, the eddy current loss term subequations
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
   double ke_T_f_B = kte*ke_T0_f_B;

   // Can now calculate specific core losses (W/kg)
   double pFe = kh_T_f_B*f*std::pow(Bm,2)+ke_T_f_B*std::pow(f,2)*std::pow(Bm,2);
   ///TODO: Return negative pFe (-pFe) below to be consistent with SteinmetzCoefficient::Eval from coefficient.cpp?
   return pFe;
}

double TempDepCoreLossCoefficient::EvalDerivS1(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           const double state1,
                           const double state2,
                           const double state3)
{
   ///TODO: As needed, utilize logic of protected class members to eval pFe

   ///TODO: Derived in Overleaf. Just need to code the below, then uncomment
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

   ///TODO: Derived in Overleaf. Just need to code the below, then uncomment
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

   ///TODO: Derived in Overleaf. Just need to code the below, then uncomment
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

///TODO: is there a need to code EvalRevDiff for temp dep core loss method here? No, but YES for Steinmetz (copy from coefficient.cpp)

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

   // Assuming state1=temperature, state2=frequency, state3=max alternating flux density
   auto T = state1; // not used for Steinmetz
   auto f = state2;
   auto Bm = state3; 

   double pFe = rho * ks * std::pow(f, alpha) * std::pow(Bm, beta)
   ///TODO: Return negative pFe (-pFe) below to be consistent with SteinmetzCoefficient::Eval from coefficient.cpp?
   return pFe;
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

///TODO: Code EvalRevDiff for Steinmetz method here (copy from coefficient.cpp)

}

///TODO: After code derivatives, circle back around to ConstructCoreLossCoeff and check this and hpp file for logic/consistency

