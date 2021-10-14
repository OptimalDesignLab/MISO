#include <vector>

#include "mfem.hpp"

#include "mach_input.hpp"
#include "mach_integrator.hpp"
#include "mach_nonlinearform.hpp"

namespace mach
{
void setInputs(MachNonlinearForm &form, const MachInputs &inputs)
{
   setScalarInputs(form.integs, inputs);
}

void evaluate(MachNonlinearForm &form, const MachInputs &inputs, 
              mfem::Vector &res_vec)
{
   // TODO
}
