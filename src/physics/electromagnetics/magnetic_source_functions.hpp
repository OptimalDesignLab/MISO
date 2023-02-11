#ifndef MACH_MAGNETIC_SOURCE_FUNCTIONS
#define MACH_MAGNETIC_SOURCE_FUNCTIONS

#include <map>
#include <string>

#include "adept.h"
#include "mfem.hpp"

#include "coefficient.hpp"
#include "mach_input.hpp"

#include "remnant_flux_coefficient.hpp"

namespace mach
{ 
class MagnetizationCoefficient : public VectorStateCoefficient // formerly an mfem::VectorCoefficient
{
public:
   friend void setInputs(MagnetizationCoefficient &mag_coeff,
                         const MachInputs &inputs)
   { }

   void Eval(mfem::Vector &V,
             mfem::ElementTransformation &trans,
             const mfem::IntegrationPoint &ip) override;

   void Eval(mfem::Vector &V,
             mfem::ElementTransformation &trans,
             const mfem::IntegrationPoint &ip, 
             double state) override;

   ///TODO: If needed, and once implemented at coefficient level, bring in EvalStateDeriv and EvalState2ndDeriv 

   void EvalRevDiff(const mfem::Vector &V_bar,
                    mfem::ElementTransformation &trans,
                    const mfem::IntegrationPoint &ip,
                    mfem::DenseMatrix &PointMat_bar) override;

   MagnetizationCoefficient(adept::Stack &diff_stack,
                            const nlohmann::json &magnet_options,
                            const nlohmann::json &materials,
                            int vdim = 3);

private:
   /// The underlying coefficient that does all the heavy lifting
   VectorMeshDependentStateCoefficient mag_coeff; // formerly an mfem::VectorCoefficient
   /// Map that holds the remnant flux for each magnet material group
   // std::map<std::string, double> remnant_flux_map;
   std::map<std::string, mfem::ConstantCoefficient> remnant_flux_map;
   /// The UNSCALED BY REMNANT FLUX underlying magnetization coefficient
   std::map<int, mfem::VectorFunctionCoefficient> no_B_r_mag_coeffs;
   /// Map that holds the remanent flux temperature coefficient for each magnet material group
   std::map<std::string, double> alpha_B_r_map;
   /// Map that holds the remanent flux reference temperature for each magnet material group
   std::map<std::string, double> T_ref_map;
};

}  // namespace mach

#endif
