#include <string>

#include "mach_input.hpp"
#include "mach_integrator.hpp"

namespace mach
{

void setInput(const MachIntegrator &x,
              const std::string &name,
              const MachInput &input)
{ x.self_->setInput_(name, input); }

} // namespace mach
