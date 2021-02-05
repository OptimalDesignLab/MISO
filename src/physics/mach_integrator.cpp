#include <string>

#include "mfem.hpp"

#include "mach_input.hpp"
#include "mach_integrator.hpp"

namespace mach
{

void setInput(const MachIntegrator &integ,
              const std::string &name,
              const MachInput &input)
{
   integ.self_->setInput_(name, input);
}

void setInput(const mfem::NonlinearFormIntegrator &integ,
              const std::string &name,
              const MachInput &input)
{
   // do nothing for default integrator
}

} // namespace mach
