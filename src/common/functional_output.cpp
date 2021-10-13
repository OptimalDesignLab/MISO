#include <cmath>

#include "mfem.hpp"

#include "mach_input.hpp"
#include "mach_integrator.hpp"
#include "functional_output.hpp"

using namespace mfem;

namespace mach
{
void setInputs(FunctionalOutput &output, const MachInputs &inputs)
{
   for (const auto &in : inputs)
   {
      const auto &input = in.second;
      if (input.isField())
      {
         const auto &name = in.first;
         auto &field = output.func_fields.at(name);
         field.GetTrueVector().SetDataAndSize(
             input.getField(), field.ParFESpace()->GetTrueVSize());
         field.SetFromTrueVector();
      }
   }
   setInputs(output.integs, inputs);
}

double calcOutput(FunctionalOutput &output, const MachInputs &inputs)
{
   setInputs(output, inputs);

   auto &state = output.func_fields.at("state");
   return output.output.GetEnergy(state);
}

double calcOutputPartial(FunctionalOutput &output,
                         const std::string &wrt,
                         const MachInputs &inputs)
{
   return NAN;
}

void calcOutputPartial(FunctionalOutput &output,
                       const std::string &wrt,
                       const MachInputs &inputs,
                       HypreParVector &partial)
{
   setInputs(output.integs, inputs);

   if (wrt == "state")
   {
      // HypreParVector state(fes->GetComm(),
      //                      fes->GlobalTrueVSize(),
      //                      inputs.at("state").getField(),
      //                      fes->GetTrueDofOffsets());
      // output.output.Mult(state, partial);
   }
   else
   {
      output.output_sens.at(wrt).Assemble();
      output.output_sens.at(wrt).ParallelAssemble(partial);
   }
}

}  // namespace mach
