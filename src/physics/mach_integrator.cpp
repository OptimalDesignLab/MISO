#include <string>

#include "mfem.hpp"

#include "mach_input.hpp"
#include "mach_integrator.hpp"

namespace mach
{

void setScalarInputs(std::vector<MachIntegrator> &integrators,
                    const MachInputs &inputs)
{
   for (auto &input : inputs)
   {
      setScalarInput(integrators, input.first, input.second);
   }
}

void setScalarInput(std::vector<MachIntegrator> &integrators,
                    const std::string &name,
                    const MachInput &input)
{
   for (auto &integrator : integrators)
   {
      setInput(integrator, name, input);
   }
}

void setInput(MachIntegrator &integ,
              const std::string &name,
              const MachInput &input)
{
   integ.self_->setInput_(name, input);
}

void setInput(mfem::NonlinearFormIntegrator &integ,
              const std::string &name,
              const MachInput &input)
{
   // do nothing for default integrator
}

} // namespace mach
