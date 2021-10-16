#include <string>

#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "mach_input.hpp"
#include "mach_integrator.hpp"

namespace mach
{
void setInputs(std::vector<MachIntegrator> &integrators,
               const MachInputs &inputs)
{
   for (auto &integ : integrators)
   {
      setInputs(integ, inputs);
   }
}

void setInputs(MachIntegrator &integ, const MachInputs &inputs)
{
   integ.self_->setInputs_(inputs);
}

void setOptions(std::vector<MachIntegrator> &integrators,
                const nlohmann::json &options)
{
   for (auto &integ : integrators)
   {
      setOptions(integ, options);
   }
}

void setOptions(MachIntegrator &integ, const nlohmann::json &options)
{
   integ.self_->setOptions_(options);
}

}  // namespace mach
