#include <string>

#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "miso_input.hpp"
#include "miso_integrator.hpp"

namespace miso
{
void setInputs(std::vector<MISOIntegrator> &integrators,
               const MISOInputs &inputs)
{
   for (auto &integ : integrators)
   {
      setInputs(integ, inputs);
   }
}

void setInputs(MISOIntegrator &integ, const MISOInputs &inputs)
{
   integ.self_->setInputs_(inputs);
}

void setOptions(std::vector<MISOIntegrator> &integrators,
                const nlohmann::json &options)
{
   for (auto &integ : integrators)
   {
      setOptions(integ, options);
   }
}

void setOptions(MISOIntegrator &integ, const nlohmann::json &options)
{
   integ.self_->setOptions_(options);
}

}  // namespace miso
