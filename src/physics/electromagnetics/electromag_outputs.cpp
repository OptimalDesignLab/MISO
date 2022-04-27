#include <string>

#include "mfem.hpp"

#include "mach_input.hpp"
#include "electromag_outputs.hpp"
#include "nlohmann/json.hpp"

namespace mach
{
void setOptions(ACLossFunctional &output, const nlohmann::json &options)
{
   setOptions(output.output, options);
}

void setInputs(ACLossFunctional &output, const MachInputs &inputs)
{
   setInputs(output.output, inputs);
}

double calcOutput(ACLossFunctional &output, const MachInputs &inputs)
{
   auto fun_inputs = inputs;
   fun_inputs["state"] = inputs.at("peak_flux");

   mfem::Vector flux_state;
   setVectorFromInputs(inputs, "peak_flux", flux_state, false, true);

   auto &flux_mag = output.fields.at("peak_flux");
   flux_mag.distributeSharedDofs(flux_state);
   mfem::ParaViewDataCollection pv("FluxMag", &flux_mag.mesh());
   pv.SetPrefixPath("ParaView");
   pv.SetLevelsOfDetail(3);
   pv.SetDataFormat(mfem::VTKFormat::BINARY);
   pv.SetHighOrderOutput(true);
   pv.RegisterField("FluxMag", &flux_mag.gridFunc());
   pv.Save();

   double loss = calcOutput(output.output, fun_inputs);
   double volume = calcOutput(output.volume, fun_inputs);
   std::cout << "Loss: " << loss << "\n";
   std::cout << "volume: " << volume << "\n";
   std::cout << "loss / volume: " << loss / volume << "\n";
   return loss / volume;
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
    mfem::Coefficient &sigma,
    const nlohmann::json &options)
 : output(fields.at("peak_flux").space(), fields),
   volume(fields.at("peak_flux").space(), fields),
   fields(fields)
{
   if (options.contains("attributes"))
   {
      auto attributes = options["attributes"].get<std::vector<int>>();
      output.addOutputDomainIntegrator(new ACLossFunctionalIntegrator(sigma), attributes);
      volume.addOutputDomainIntegrator(new VolumeIntegrator, attributes);
   }
   else
   {
      output.addOutputDomainIntegrator(new ACLossFunctionalIntegrator(sigma));
      volume.addOutputDomainIntegrator(new VolumeIntegrator);
   }
   setOptions(*this, options);
}

}  // namespace mach
