#include <vector>

#include "mfem.hpp"

#include "utils.hpp"
#include "mach_input.hpp"
#include "mach_integrator.hpp"
#include "mach_nonlinearform.hpp"

namespace mach
{
void setInputs(MachNonlinearForm &form, const MachInputs &inputs)
{
   setScalarInputs(form.integs, inputs);
}

void evaluate(MachNonlinearForm &form,
              const MachInputs &inputs,
              mfem::Vector &res_vec)
{
   auto pfes = form.nf.ParFESpace();
   auto state = bufferToHypreParVector(inputs.at("state").getField(), *pfes);
   form.nf.Mult(state, res_vec);
}

}  // namespace mach