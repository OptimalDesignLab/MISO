#include <vector>

#include "mfem.hpp"

#include "mach_input.hpp"
#include "mach_integrator.hpp"
#include "mach_linearform.hpp"

namespace mach
{

void setInputs(MachLinearForm &load,
               const MachInputs &inputs)
{
   setScalarInputs(load.dlfi, inputs);
   setScalarInputs(load.blfi, inputs);
   setScalarInputs(load.flfi, inputs);
}

void assemble(MachLinearForm &load,
              mfem::HypreParVector &tv)
{
   load.lf.Assemble();
   load.lf.ParallelAssemble(tv);
}

} // namespace mach
