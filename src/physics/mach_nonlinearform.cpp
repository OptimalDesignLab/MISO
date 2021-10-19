#include <vector>

#include "mfem.hpp"

#include "utils.hpp"
#include "mach_input.hpp"
#include "mach_integrator.hpp"
#include "mach_nonlinearform.hpp"

namespace mach
{
int getSize(const MachNonlinearForm &form)
{
   return form.nf.FESpace()->GetTrueVSize();
}

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

void setOptions(MachNonlinearForm &form, const nlohmann::json &options)
{
   setOptions(form.integs, options);
}

void evaluate(MachNonlinearForm &form,
              const MachInputs &inputs,
              mfem::Vector &res_vec)
{
   auto pfes = form.nf.ParFESpace();
   auto state = bufferToHypreParVector(inputs.at("state").getField(), *pfes);
   form.nf.Mult(state, res_vec);
}

void getJacobian(MachNonlinearForm &form,
                 const MachInputs &inputs,
                 std::string wrt,
                 mfem::Operator &jacobian)
{
   auto pfes = form.nf.ParFESpace();
   auto state = bufferToHypreParVector(inputs.at("state").getField(), *pfes);
   jacobian = form.nf.GetGradient(state);
}

mfem::Operator &getJacobian(MachNonlinearForm &form,
                            const MachInputs &inputs,
                            std::string wrt)
{
   auto pfes = form.nf.ParFESpace();
   auto state = bufferToHypreParVector(inputs.at("state").getField(), *pfes);
   return form.nf.GetGradient(state);
}

}  // namespace mach