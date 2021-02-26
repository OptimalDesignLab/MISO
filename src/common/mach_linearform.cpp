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

void addLoad(MachLinearForm &load,
             mfem::Vector &tv)
{
   load.lf.Assemble();
   load.lf.ParallelAssemble(load.scratch);
   add(tv, load.scratch, tv);
}

} // namespace mach
