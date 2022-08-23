#include <string>

#include "data_logging.hpp"
#include "electromag_integ.hpp"
#include "mach_integrator.hpp"
#include "mfem.hpp"

#include "coefficient.hpp"
#include "mach_input.hpp"
#include "electromag_outputs.hpp"
#include "nlohmann/json.hpp"

namespace mach
{
void setOptions(ForceFunctional &output, const nlohmann::json &options)
{
   auto &&attrs = options["attributes"].get<std::unordered_set<int>>();
   auto &&axis = options["axis"].get<std::vector<double>>();

   auto space_dim = output.fields.at("vforce").mesh().SpaceDimension();
   mfem::VectorConstantCoefficient axis_vector(
       mfem::Vector(&axis[0], space_dim));

   auto &v = output.fields.at("vforce").gridFunc();
   v = 0.0;
   for (const auto &attr : attrs)
   {
      v.ProjectCoefficient(axis_vector, attr);
   }
}

void setOptions(TorqueFunctional &output, const nlohmann::json &options)
{
   auto &&attrs = options["attributes"].get<std::unordered_set<int>>();
   auto &&axis = options["axis"].get<std::vector<double>>();
   auto &&about = options["about"].get<std::vector<double>>();
   mfem::Vector axis_vector(&axis[0], axis.size());
   axis_vector /= axis_vector.Norml2();

   auto space_dim = output.fields.at("vtorque").mesh().SpaceDimension();
   mfem::Vector about_vector(&about[0], space_dim);
   double r_data[3];
   mfem::Vector r(r_data, space_dim);

   mfem::VectorFunctionCoefficient v_vector(
       space_dim,
       [&axis_vector, &about_vector, &r, space_dim](const mfem::Vector &x,
                                                    mfem::Vector &v)
       {
          subtract(x, about_vector, r);
          if (space_dim == 3)
          {
             // r /= r.Norml2();
             v(0) = axis_vector(1) * r(2) - axis_vector(2) * r(1);
             v(1) = axis_vector(2) * r(0) - axis_vector(0) * r(2);
             v(2) = axis_vector(0) * r(1) - axis_vector(1) * r(0);
          }
          else
          {
             v(0) = -axis_vector(2) * r(1);
             v(1) = axis_vector(2) * r(0);
          }
          // if (v.Norml2() > 1e-12)
          //    v /= v.Norml2();
       });

   auto &v = output.fields.at("vtorque").gridFunc();
   v = 0.0;
   for (const auto &attr : attrs)
   {
      v.ProjectCoefficient(v_vector, attr);
   }
}

double calcOutput(DCLossFunctional &output, const MachInputs &inputs)
{
   setInputs(output, inputs);

   /// rho = electrical resistivity, doesn't depend on any state
   /// so we just integrate with a dummy state vector
   output.scratch.SetSize(output.output.ParFESpace()->GetTrueVSize());
   double rho = output.output.GetEnergy(output.scratch);

   double strand_area = M_PI * pow(output.strand_radius, 2);
   double R = output.wire_length * rho / (strand_area * output.strands_in_hand);

   double loss = pow(output.rms_current, 2) * R;
   loss *= sqrt(2);

   double volume = calcOutput(output.volume, inputs);
   return loss / volume;
}

DCLossFunctional::DCLossFunctional(
    std::map<std::string, FiniteElementState> &fields,
    mfem::Coefficient &sigma,
    const nlohmann::json &options)
 : FunctionalOutput(fields.at("state").space(), fields), volume(fields, options)
{
   if (options.contains("attributes"))
   {
      auto attributes = options["attributes"].get<std::vector<int>>();
      addOutputDomainIntegrator(new DCLossFunctionalIntegrator(sigma),
                                attributes);
   }
   else
   {
      addOutputDomainIntegrator(new DCLossFunctionalIntegrator(sigma));
   }
}

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
   setValueFromInputs(inputs, "strands_in_hand", output.strands_in_hand);
   setValueFromInputs(inputs, "num_turns", output.num_turns);
   setValueFromInputs(inputs, "num_slots", output.num_slots);

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

   loss *= output.stack_length * M_PI * pow(output.radius, 4) *
           pow(2 * M_PI * output.freq, 2) / 32.0;
   loss *= 2 * output.strands_in_hand * output.num_turns * output.num_slots;

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
   volume(fields, options),
   fields(fields)
{
   if (options.contains("attributes"))
   {
      auto attributes = options["attributes"].get<std::vector<int>>();
      output.addOutputDomainIntegrator(new ACLossFunctionalIntegrator(sigma),
                                       attributes);
   }
   else
   {
      output.addOutputDomainIntegrator(new ACLossFunctionalIntegrator(sigma));
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
   return calcOutput(output.output, inputs);
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
          new SteinmetzLossIntegrator(*rho, *k_s, *alpha, *beta, "stator"),
          attributes);
   }
   else
   {
      output.addOutputDomainIntegrator(
          new SteinmetzLossIntegrator(*rho, *k_s, *alpha, *beta));
   }
}

void setOptions(EMHeatSourceOutput &output, const nlohmann::json &options)
{
   setOptions(output.lf, options);
}

void setInputs(EMHeatSourceOutput &output, const MachInputs &inputs)
{
   setInputs(output.lf, inputs);
}

void calcOutput(EMHeatSourceOutput &output,
                const MachInputs &inputs,
                mfem::Vector &out_vec)
{
   setInputs(output, inputs);

   out_vec = 0.0;
   addLoad(output.lf, out_vec);
}

EMHeatSourceOutput::EMHeatSourceOutput(
    std::map<std::string, FiniteElementState> &fields,
    mfem::Coefficient &rho,
    mfem::Coefficient &sigma,
    const nlohmann::json &components,
    const nlohmann::json &materials,
    const nlohmann::json &options)
 : lf(fields.at("temperature").space(), fields),
   k_s(constructMaterialCoefficient("ks", components, materials)),
   alpha(constructMaterialCoefficient("alpha", components, materials)),
   beta(constructMaterialCoefficient("beta", components, materials))
{
   // auto stator_attrs = components["stator"]["attrs"].get<std::vector<int>>();
   // lf.addDomainIntegrator(new SteinmetzLossDistributionIntegrator(
   //                            rho, *k_s, *alpha, *beta, "stator"),
   //                        stator_attrs);

   auto winding_attrs = components["windings"]["attrs"].get<std::vector<int>>();
   lf.addDomainIntegrator(new DCLossFunctionalDistributionIntegrator(sigma),
                          winding_attrs);
   // lf.addDomainIntegrator(new ACLossFunctionalDistributionIntegrator(
   //                            fields.at("peak_flux").gridFunc(), sigma),
   //                        winding_attrs);
}

}  // namespace mach
