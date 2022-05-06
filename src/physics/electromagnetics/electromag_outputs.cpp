#include <string>

#include "mfem.hpp"

#include "coefficient.hpp"
#include "mach_input.hpp"
#include "electromag_outputs.hpp"
#include "nlohmann/json.hpp"

namespace mach
{
void setOptions(ACLossFunctional &output, const nlohmann::json &options)
{
   // output.num_strands = options.value("num_strands", output.num_strands);
   setOptions(output.output, options);
}

void setInputs(ACLossFunctional &output, const MachInputs &inputs)
{
   setValueFromInputs(inputs, "strand_radius", output.radius);
   setValueFromInputs(inputs, "frequency", output.freq);
   setValueFromInputs(inputs, "stack_length", output.stack_length);
   setValueFromInputs(inputs, "num_strands", output.num_strands);

   setInputs(output.output, inputs);
}

double calcOutput(ACLossFunctional &output, const MachInputs &inputs)
{
   auto fun_inputs = inputs;
   fun_inputs["state"] = inputs.at("peak_flux");

   // mfem::Vector flux_state;
   // setVectorFromInputs(inputs, "peak_flux", flux_state, false, true);
   // auto &flux_mag = output.fields.at("peak_flux");
   // flux_mag.distributeSharedDofs(flux_state);
   // mfem::ParaViewDataCollection pv("FluxMag", &flux_mag.mesh());
   // pv.SetPrefixPath("ParaView");
   // pv.SetLevelsOfDetail(3);
   // pv.SetDataFormat(mfem::VTKFormat::BINARY);
   // pv.SetHighOrderOutput(true);
   // pv.RegisterField("FluxMag", &flux_mag.gridFunc());
   // pv.Save();

   double loss = calcOutput(output.output, fun_inputs);

   loss *= output.stack_length * M_PI * pow(output.radius, 4) *
           pow(2 * M_PI * output.freq, 2) / 32.0;
   loss *= output.num_strands;

   double volume = calcOutput(output.volume, fun_inputs);
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
   volume(fields.at("peak_flux").space(), fields)
// fields(fields)
{
   if (options.contains("attributes"))
   {
      auto attributes = options["attributes"].get<std::vector<int>>();
      output.addOutputDomainIntegrator(new ACLossFunctionalIntegrator(sigma),
                                       attributes);
      volume.addOutputDomainIntegrator(new VolumeIntegrator, attributes);
   }
   else
   {
      output.addOutputDomainIntegrator(new ACLossFunctionalIntegrator(sigma));
      volume.addOutputDomainIntegrator(new VolumeIntegrator);
   }
   setOptions(*this, options);
}

void setOptions(CoreLossFunctional &output, const nlohmann::json &options)
{
   setOptions(output.output, options);
}

void setInputs(CoreLossFunctional &output, const MachInputs &inputs)
{
   setInputs(output.output, inputs);
}

double calcOutput(CoreLossFunctional &output, const MachInputs &inputs)
{
   auto fun_inputs = inputs;
   fun_inputs["state"] = inputs.at("peak_flux");

   double loss = calcOutput(output.output, fun_inputs);
   // std::cout << "Core loss: " << loss << "\n";
   return loss;
}

double calcOutputPartial(CoreLossFunctional &output,
                         const std::string &wrt,
                         const MachInputs &inputs)
{
   return calcOutputPartial(output.output, wrt, inputs);
}

void calcOutputPartial(CoreLossFunctional &output,
                       const std::string &wrt,
                       const MachInputs &inputs,
                       mfem::Vector &partial)
{
   calcOutputPartial(output.output, wrt, inputs, partial);
}

CoreLossFunctional::CoreLossFunctional(
    std::map<std::string, FiniteElementState> &fields,
    const nlohmann::json &components,
    const nlohmann::json &materials,
    const nlohmann::json &options)
 : output(fields.at("peak_flux").space(), fields),
   rho(constructMaterialCoefficient("rho", components, materials)),
   k_s(constructMaterialCoefficient("ks", components, materials)),
   alpha(constructMaterialCoefficient("alpha", components, materials)),
   beta(constructMaterialCoefficient("beta", components, materials))
{
   if (options.contains("attributes"))
   {
      auto attributes = options["attributes"].get<std::vector<int>>();
      output.addOutputDomainIntegrator(
          new SteinmetzLossIntegrator(*rho, *k_s, *alpha, *beta), attributes);
   }
   else
   {
      output.addOutputDomainIntegrator(
          new SteinmetzLossIntegrator(*rho, *k_s, *alpha, *beta));
   }
}

}  // namespace mach
