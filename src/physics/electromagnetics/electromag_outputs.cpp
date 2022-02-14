#include <string>

#include "mfem.hpp"

#include "mach_input.hpp"
#include "electromag_outputs.hpp"

namespace mach
{
void setInputs(ACLossFunctional &output,
               const MachInputs &inputs)
{
   setInputs(output.output, inputs);
}

double calcOutput(ACLossFunctional &output,
                  const MachInputs &inputs)
{
   auto fun_inputs = inputs;
   fun_inputs["state"] = inputs.at("flux_magnitude");

   mfem::Vector flux_state;
   setVectorFromInputs(inputs, "flux_magnitude", flux_state, false, true);

   auto &flux_mag = output.fields.at("flux_magnitude");
   flux_mag.distributeSharedDofs(flux_state);
   mfem::ParaViewDataCollection pv("FluxMag", &flux_mag.mesh());
   pv.SetPrefixPath("ParaView");
   pv.SetLevelsOfDetail(3);
   pv.SetDataFormat(mfem::VTKFormat::ASCII);
   pv.SetHighOrderOutput(true);
   pv.RegisterField("FluxMag", &flux_mag.gridFunc());
   pv.Save();

   return calcOutput(output.output, fun_inputs);
}

 double calcOutputPartial(ACLossFunctional &output,
                          const std::string &wrt,
                          const MachInputs &inputs)
{
   return calcOutputPartial(output.output, wrt, inputs);
}

 void calcOutputPartial(ACLossFunctional &output,
                        const std::string &wrt,
                        const MachInputs &inputs,
                        mfem::Vector &partial)
{
   calcOutputPartial(output.output, wrt, inputs, partial);
}

ACLossFunctional::ACLossFunctional(
   std::map<std::string, FiniteElementState> &fields,
   mfem::Coefficient &sigma)
 : output(fields.at("flux_magnitude").space(), fields), fields(fields)
{
   output.addOutputDomainIntegrator(
      new ACLossFunctionalIntegrator(sigma));
}


}  // namespace mach
