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
   for (const auto &in : inputs)
   {
      const auto &input = in.second;
      if (input.isField())
      {
         const auto &name = in.first;
         auto it = form.nf_fields->find(name);
         if (it != form.nf_fields->end())
         {
            auto &field = it->second;
            field.GetTrueVector().SetDataAndSize(
                input.getField(), field.ParFESpace()->GetTrueVSize());
            field.SetFromTrueVector();
         }
      }
   }
   setInputs(form.integs, inputs);
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