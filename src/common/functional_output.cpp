#include <cmath>

#include "mfem.hpp"

#include "miso_input.hpp"
#include "miso_integrator.hpp"
#include "functional_output.hpp"

using namespace mfem;

namespace miso
{
void setInputs(FunctionalOutput &output, const MISOInputs &inputs)
{
   for (const auto &in : inputs)
   {
      const auto &input = in.second;
      if (input.isField())
      {
         const auto &name = in.first;
         auto it = output.func_fields->find(name);
         if (it != output.func_fields->end())
         {
            auto &field = it->second;
            field.GetTrueVector().SetDataAndSize(
                input.getField(), field.ParFESpace()->GetTrueVSize());
            field.SetFromTrueVector();
         }
      }
   }
   setInputs(output.integs, inputs);
}

void setOptions(FunctionalOutput &output, const nlohmann::json &options)
{
   setOptions(output.integs, options);
}

double calcOutput(FunctionalOutput &output, const MISOInputs &inputs)
{
   setInputs(output, inputs);

   auto &state = output.func_fields->at("state");
   return output.output.GetEnergy(state);
}

double calcOutputPartial(FunctionalOutput &output,
                         const std::string &wrt,
                         const MISOInputs &inputs)
{
   return NAN;
}

void calcOutputPartial(FunctionalOutput &output,
                       const std::string &wrt,
                       const MISOInputs &inputs,
                       HypreParVector &partial)
{
   setInputs(output, inputs);

   if (wrt == "state")
   {
      auto *fes = output.output.ParFESpace();
      HypreParVector state(fes->GetComm(),
                           fes->GlobalTrueVSize(),
                           inputs.at("state").getField(),
                           fes->GetTrueDofOffsets());
      output.output.Mult(state, partial);
   }
   else
   {
      output.output_sens.at(wrt).Assemble();
      output.output_sens.at(wrt).ParallelAssemble(partial);
   }
}

}  // namespace miso
