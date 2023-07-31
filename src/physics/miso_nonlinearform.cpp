#include <vector>

#include "mfem.hpp"

#include "utils.hpp"
#include "miso_input.hpp"
#include "miso_integrator.hpp"
#include "miso_nonlinearform.hpp"

namespace miso
{
int getSize(const MISONonlinearForm &form)
{
   return form.nf.FESpace()->GetTrueVSize();
}

void setInputs(MISONonlinearForm &form, const MISOInputs &inputs)
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

void setOptions(MISONonlinearForm &form, const nlohmann::json &options)
{
   setOptions(form.integs, options);
}

void evaluate(MISONonlinearForm &form,
              const MISOInputs &inputs,
              mfem::Vector &res_vec)
{
   auto *pfes = form.nf.ParFESpace();
   auto state = bufferToHypreParVector(inputs.at("state").getField(), *pfes);
   form.nf.Mult(state, res_vec);
}

mfem::Operator &getJacobian(MISONonlinearForm &form,
                            const MISOInputs &inputs,
                            std::string wrt)
{
   auto *pfes = form.nf.ParFESpace();
   auto state = bufferToHypreParVector(inputs.at("state").getField(), *pfes);
   return form.nf.GetGradient(state);
}

}  // namespace miso