#include <string>

#include "mfem.hpp"

#include "mach_input.hpp"
#include "mach_integrator.hpp"
#include "mach_load.hpp"

namespace mach
{

void setInputs(MachLoad &load,
               const MachInputs &inputs)
{
   load.self_->setInputs_(inputs);
}

void assemble(MachLoad &load,
              mfem::HypreParVector &tv)
{
   load.self_->assemble_(tv);
}

} // namespace mach
