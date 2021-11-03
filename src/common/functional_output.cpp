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
      if (std::holds_alternative<InputVector>(input))
      {
         const auto &name = in.first;
         auto it = output.func_fields->find(name);
         if (it != output.func_fields->end())
         {
            auto &field = it->second;
            setVectorFromInput(input, field.GetTrueVector());
            // if (field.GetTrueVector().Size() !=
            //     field.ParFESpace()->GetTrueVSize())
            // {
            //    throw MachException("Input field " + name +
            //                        " is wrong size!\n"
            //                        "Size is " +
            //                        field.GetTrueVector().Size() +
            //                        ", should be " +
            //                        field.ParFESpace()->GetTrueVSize() +
            //                        "!\n");
            // }
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

double calcOutput(FunctionalOutput &output, const MachInputs &inputs)
{
   setInputs(output, inputs);
   Vector state;
   setVectorFromInputs(inputs, "state", state, false, true);
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
   setInputs(output, inputs);

   if (wrt == "state")
   {
      // auto *fes = output.output.ParFESpace();
      // HypreParVector state(fes->GetComm(),
      //                      fes->GlobalTrueVSize(),
      //                      inputs.at("state").getField(),
      //                      fes->GetTrueDofOffsets());
      Vector state;
      setVectorFromInputs(inputs, "state", state, false, true);
      output.output.Mult(state, partial);
   }
   else
   {
      output.output_sens.at(wrt).Assemble();
      output.output_sens.at(wrt).ParallelAssemble(partial);
   }
}

}  // namespace mach
