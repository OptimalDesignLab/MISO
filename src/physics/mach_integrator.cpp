#include <string>

#include "mfem.hpp"

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

}  // namespace mach
