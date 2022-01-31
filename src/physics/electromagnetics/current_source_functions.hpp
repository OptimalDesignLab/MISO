#ifndef MACH_CURRENT_SOURCE_FUNCTIONS
#define MACH_CURRENT_SOURCE_FUNCTIONS

#include <map>
#include <memory>
#include <string>

#include "adept.h"
#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "coefficient.hpp"
#include "mach_input.hpp"

namespace mach
{
class CurrentDensityCoefficient : public mfem::VectorCoefficient
{
public:
   friend void setInputs(CurrentDensityCoefficient &current,
                         const MachInputs &inputs);

   void Eval(mfem::Vector &V,
             mfem::ElementTransformation &trans,
             const mfem::IntegrationPoint &ip) override;

   void EvalRevDiff(const mfem::Vector &V_bar,
                    mfem::ElementTransformation &trans,
                    const mfem::IntegrationPoint &ip,
                    mfem::DenseMatrix &PointMat_bar) override;

   CurrentDensityCoefficient(adept::Stack &diff_stack,
                             const nlohmann::json &current_options,
                             int vdim = 3);

private:
   /// The underlying coefficient that does all the heavy lifting
   VectorMeshDependentCoefficient current_coeff;
   /// Map that holds coefficients for each current group so that the scalar
   /// input may be set for each group
   std::map<std::string, mfem::ConstantCoefficient> group_map;
   /// Map that owns all of the underlying source coefficients
   std::map<int, mfem::VectorFunctionCoefficient> source_coeffs;
   /// Inputs to be passed by reference to source-wrapping lambdas
   std::map<std::string, double> cached_inputs;
};

// /// Construct vector coefficient that describes the current source direction
// /// \param[in] options - JSON options dictionary that maps mesh element
// /// attributes to known current source functions
// std::unique_ptr<mfem::VectorCoefficient> constructCurrent(
//     const nlohmann::json &current_options);

}  // namespace mach

#endif
