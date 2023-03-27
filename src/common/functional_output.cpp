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
   for (const auto &[name, input] : inputs)
   {
      if (std::holds_alternative<InputVector>(input))
      {
         if (output.func_fields != nullptr)
         {
            auto it = output.func_fields->find(name);
            if (it != output.func_fields->end())
            {
               auto &field = it->second;
               mfem::Vector field_tv;
               setVectorFromInput(input, field_tv);

               field.distributeSharedDofs(field_tv);
            }
         }
      }
   }
   setInputs(output.integs, inputs);
}

void setOptions(FunctionalOutput &output, const nlohmann::json &options)
{
   setOptions(output.integs, options);
   if (options.contains("attributes"))
   {
      auto attributes = options["attributes"].get<std::vector<int>>();
      for (auto &marker : output.domain_markers)
      {
         attrVecToArray(attributes, marker);
      }
      for (auto &marker : output.bdr_markers)
      {
         attrVecToArray(attributes, marker);
      }
   }
}

double calcOutput(FunctionalOutput &output, const MachInputs &inputs)
{
   setInputs(output, inputs);
   Vector state;
   setVectorFromInputs(inputs, "state", state);
   if (state.Size() != output.output.ParFESpace()->GetTrueVSize())
   {
      output.scratch.SetSize(output.output.ParFESpace()->GetTrueVSize());
      state.NewDataAndSize(output.scratch.GetData(),
                           output.output.ParFESpace()->GetTrueVSize());
   }
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
                       Vector &partial)
{
   setInputs(output, inputs);

   if (wrt == "state")
   {
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

double jacobianVectorProduct(FunctionalOutput &output,
                             const mfem::Vector &wrt_dot,
                             const std::string &wrt)
{
   output.scratch.SetSize(wrt_dot.Size());
   if (wrt == "state")
   {
      Vector state;
      output.func_fields->at("state").setTrueVec(state);
      output.output.Mult(state, output.scratch);
   }
   else
   {
      if (wrt_dot.Size() == 1)
      {
         Vector state;
         output.func_fields->at("state").setTrueVec(state);
         const auto &state_gf = output.func_fields->at("state").gridFunc();

         auto wrt_key = wrt.substr(0, wrt.find(':'));
         // output.scratch(0) =
         // output.output_scalar_sens.at(wrt).GetEnergy(state);
         // TODO: need to confirm what OpenMDAO expects in terms of parallel
         // outputs: should we give it the already reduced value or not?
         output.scratch(0) =
             output.output_scalar_sens.at(wrt_key).GetEnergy(state_gf);
      }
      else
      {
         output.output_sens.at(wrt).Assemble();
         output.output_sens.at(wrt).ParallelAssemble(output.scratch);
      }
   }

   return InnerProduct(output.scratch, wrt_dot);
}

double vectorJacobianProduct(FunctionalOutput &output,
                             const mfem::Vector &out_bar,
                             const std::string &wrt)
{
   // Vector state;
   // output.func_fields->at("state").setTrueVec(state);
   const auto &state_gf = output.func_fields->at("state").gridFunc();

   auto wrt_key = wrt.substr(0, wrt.find(':'));
   // double sens = output.output_scalar_sens.at(wrt_key).GetEnergy(state);
   // TODO: need to confirm what OpenMDAO expects in terms of parallel outputs:
   // should we give it the already reduced value or not?
   double sens = output.output_scalar_sens.at(wrt_key).GetEnergy(state_gf);

   return sens * out_bar(0);
}

void vectorJacobianProduct(FunctionalOutput &output,
                           const mfem::Vector &out_bar,
                           const std::string &wrt,
                           mfem::Vector &wrt_bar)
{
   output.scratch.SetSize(wrt_bar.Size());
   if (wrt == "state")
   {
      Vector state;
      output.func_fields->at("state").setTrueVec(state);
      output.output.Mult(state, output.scratch);
   }
   else
   {
      output.output_sens.at(wrt).Assemble();
      output.output_sens.at(wrt).ParallelAssemble(output.scratch);
   }

   wrt_bar.Add(out_bar(0), output.scratch);
}

}  // namespace mach
