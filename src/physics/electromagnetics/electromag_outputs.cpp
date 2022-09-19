#include <cmath>
#include <string>

#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "coefficient.hpp"
#include "common_outputs.hpp"
#include "data_logging.hpp"
#include "electromag_integ.hpp"
#include "functional_output.hpp"
#include "mach_input.hpp"
// #include "mach_integrator.hpp"

#include "electromag_outputs.hpp"

namespace mach
{
void setOptions(ForceFunctional &output, const nlohmann::json &options)
{
   auto &&attrs = options["attributes"].get<std::unordered_set<int>>();
   auto &&axis = options["axis"].get<std::vector<double>>();

   auto space_dim = output.fields.at("vforce").mesh().SpaceDimension();
   mfem::VectorConstantCoefficient axis_vector(
       mfem::Vector(axis.data(), space_dim));

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
   mfem::Vector axis_vector(axis.data(), axis.size());
   axis_vector /= axis_vector.Norml2();

   auto space_dim = output.fields.at("vtorque").mesh().SpaceDimension();
   mfem::Vector about_vector(about.data(), space_dim);
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

   double rho = calcOutput(output.resistivity, inputs);

   double strand_area = M_PI * pow(output.strand_radius, 2);
   double R = output.wire_length * rho / (strand_area * output.strands_in_hand);

   double loss = pow(output.rms_current, 2) * R * sqrt(2);

   double volume = calcOutput(output.volume, inputs);
   return loss / volume;
}

double jacobianVectorProduct(DCLossFunctional &output,
                             const mfem::Vector &wrt_dot,
                             const std::string &wrt)
{
   const MachInputs &inputs = *output.inputs;
   if (wrt.rfind("wire_length", 0) == 0)
   {
      std::cout << "wire_length_dot = " << wrt_dot(0) << "\n";

      double rho = calcOutput(output.resistivity, inputs);

      double strand_area = M_PI * pow(output.strand_radius, 2);
      // double R =
      //     output.wire_length * rho / (strand_area * output.strands_in_hand);
      double R_dot = rho / (strand_area * output.strands_in_hand) * wrt_dot(0);

      // double loss = pow(output.rms_current, 2) * R * sqrt(2);
      double loss_dot = pow(output.rms_current, 2) * sqrt(2) * R_dot;

      double volume = calcOutput(output.volume, inputs);

      // double dc_loss = loss / volume;
      double dc_loss_dot = 1 / volume * loss_dot;
      std::cout << "dc_loss_dot = " << dc_loss_dot << "\n";
      return dc_loss_dot;
   }
   else if (wrt.rfind("rms_current", 0) == 0)
   {
      double rho = calcOutput(output.resistivity, inputs);

      double strand_area = M_PI * pow(output.strand_radius, 2);
      double R =
          output.wire_length * rho / (strand_area * output.strands_in_hand);

      // double loss = pow(output.rms_current, 2) * R * sqrt(2);
      double loss_dot = 2 * output.rms_current * R * sqrt(2) * wrt_dot(0);

      double volume = calcOutput(output.volume, inputs);

      // double dc_loss = loss / volume;
      double dc_loss_dot = 1 / volume * loss_dot;
      return dc_loss_dot;
   }
   else if (wrt.rfind("strand_radius", 0) == 0)
   {
      double rho = calcOutput(output.resistivity, inputs);

      double strand_area = M_PI * pow(output.strand_radius, 2);
      double strand_area_dot = M_PI * 2 * output.strand_radius * wrt_dot(0);

      // double R =
      //     output.wire_length * rho / (strand_area * output.strands_in_hand);
      double R_dot = -output.wire_length * rho /
                     (pow(strand_area, 2) * output.strands_in_hand) *
                     strand_area_dot;

      // double loss = pow(output.rms_current, 2) * R * sqrt(2);
      double loss_dot = pow(output.rms_current, 2) * sqrt(2) * R_dot;

      double volume = calcOutput(output.volume, inputs);

      // double dc_loss = loss / volume;
      double dc_loss_dot = 1 / volume * loss_dot;
      return dc_loss_dot;
   }
   else if (wrt.rfind("strands_in_hand", 0) == 0)
   {
      double rho = calcOutput(output.resistivity, inputs);

      double strand_area = M_PI * pow(output.strand_radius, 2);

      // double R =
      //     output.wire_length * rho / (strand_area * output.strands_in_hand);
      double R_dot = -output.wire_length * rho /
                     (strand_area * pow(output.strands_in_hand, 2)) *
                     wrt_dot(0);

      // double loss = pow(output.rms_current, 2) * R * sqrt(2);
      double loss_dot = pow(output.rms_current, 2) * sqrt(2) * R_dot;

      double volume = calcOutput(output.volume, inputs);

      // double dc_loss = loss / volume;
      double dc_loss_dot = 1 / volume * loss_dot;
      return dc_loss_dot;
   }
   else if (wrt.rfind("mesh_coords", 0) == 0)
   {
      double rho = calcOutput(output.resistivity, inputs);
      double rho_dot = jacobianVectorProduct(output.resistivity, wrt_dot, wrt);

      double strand_area = M_PI * pow(output.strand_radius, 2);

      double R =
          output.wire_length * rho / (strand_area * output.strands_in_hand);
      double R_dot =
          output.wire_length / (strand_area * output.strands_in_hand) * rho_dot;

      double loss = pow(output.rms_current, 2) * R * sqrt(2);
      double loss_dot = pow(output.rms_current, 2) * sqrt(2) * R_dot;

      double volume = calcOutput(output.volume, inputs);
      double volume_dot = jacobianVectorProduct(output.volume, wrt_dot, wrt);

      // double dc_loss = loss / volume;
      double dc_loss_dot =
          loss_dot / volume - loss / pow(volume, 2) * volume_dot;

      return dc_loss_dot;
   }
   else
   {
      return 0.0;
   }
}

void jacobianVectorProduct(DCLossFunctional &output,
                           const mfem::Vector &wrt_dot,
                           const std::string &wrt,
                           mfem::Vector &out_dot)
{ }

double vectorJacobianProduct(DCLossFunctional &output,
                             const mfem::Vector &out_bar,
                             const std::string &wrt)
{
   const MachInputs &inputs = *output.inputs;
   if (wrt.rfind("wire_length", 0) == 0)
   {
      double rho = calcOutput(output.resistivity, inputs);

      double strand_area = M_PI * pow(output.strand_radius, 2);
      // double R = output.wire_length * rho / (strand_area *
      // output.strands_in_hand);

      // double loss = pow(output.rms_current, 2) * R * sqrt(2);

      double volume = calcOutput(output.volume, inputs);
      // double dc_loss = loss / volume;

      /// Start reverse pass...
      double dc_loss_bar = out_bar(0);

      /// double dc_loss = loss / volume;
      double loss_bar = dc_loss_bar / volume;
      // double volume_bar = -dc_loss_bar * loss / pow(volume, 2);

      /// double volume = calcOutput(output.volume, inputs);
      // volume does not depend on any of the inputs except mesh coords

      /// double loss = pow(output.rms_current, 2) * R * sqrt(2);
      // double rms_current_bar = loss_bar * 2 * output.rms_current * R *
      // sqrt(2);
      double R_bar = loss_bar * pow(output.rms_current, 2) * sqrt(2);

      /// double R = output.wire_length * rho / (strand_area *
      /// output.strands_in_hand);
      double wire_length_bar =
          R_bar * rho / (strand_area * output.strands_in_hand);
      // double rho_bar = R_bar * output.wire_length / (strand_area *
      // output.strands_in_hand); double strand_area_bar = -R_bar *
      // output.wire_length * rho / (pow(strand_area,2) *
      // output.strands_in_hand); double strands_in_hand_bar = -R_bar *
      // output.wire_length * rho / (strand_area * pow(output.strands_in_hand,
      // 2));

      /// double strand_area = M_PI * pow(output.strand_radius, 2);
      // double strand_radius_bar = strand_area_bar * M_PI * 2 *
      // output.strand_radius;

      /// double rho = output.output.GetEnergy(output.scratch);
      // rho does not depend on any of the inputs except mesh coords

      return wire_length_bar;
   }
   else if (wrt.rfind("rms_current", 0) == 0)
   {
      double rho = calcOutput(output.resistivity, inputs);

      double strand_area = M_PI * pow(output.strand_radius, 2);
      double R =
          output.wire_length * rho / (strand_area * output.strands_in_hand);

      // double loss = pow(output.rms_current, 2) * R * sqrt(2);

      double volume = calcOutput(output.volume, inputs);
      // double dc_loss = loss / volume;

      /// Start reverse pass...
      double dc_loss_bar = out_bar(0);

      /// double dc_loss = loss / volume;
      double loss_bar = dc_loss_bar / volume;
      // double volume_bar = -dc_loss_bar * loss / pow(volume, 2);

      /// double volume = calcOutput(output.volume, inputs);
      // volume does not depend on any of the inputs except mesh coords

      /// double loss = pow(output.rms_current, 2) * R * sqrt(2);
      double rms_current_bar = loss_bar * 2 * output.rms_current * R * sqrt(2);
      // double R_bar = loss_bar * pow(output.rms_current, 2) * sqrt(2);

      /// double R = output.wire_length * rho / (strand_area *
      /// output.strands_in_hand);
      // double wire_length_bar = R_bar * rho / (strand_area *
      // output.strands_in_hand); double rho_bar = R_bar * output.wire_length /
      // (strand_area * output.strands_in_hand); double strand_area_bar = -R_bar
      // * output.wire_length * rho / (pow(strand_area,2) *
      // output.strands_in_hand); double strands_in_hand_bar = -R_bar *
      // output.wire_length * rho / (strand_area * pow(output.strands_in_hand,
      // 2));

      /// double strand_area = M_PI * pow(output.strand_radius, 2);
      // double strand_radius_bar = strand_area_bar * M_PI * 2 *
      // output.strand_radius;

      /// double rho = output.output.GetEnergy(output.scratch);
      // rho does not depend on any of the inputs except mesh coords

      return rms_current_bar;
   }
   else if (wrt.rfind("strand_radius", 0) == 0)
   {
      double rho = calcOutput(output.resistivity, inputs);

      double strand_area = M_PI * pow(output.strand_radius, 2);
      // double R = output.wire_length * rho / (strand_area *
      // output.strands_in_hand);

      // double loss = pow(output.rms_current, 2) * R * sqrt(2);

      double volume = calcOutput(output.volume, inputs);
      // double dc_loss = loss / volume;

      /// Start reverse pass...
      double dc_loss_bar = out_bar(0);

      /// double dc_loss = loss / volume;
      double loss_bar = dc_loss_bar / volume;
      // double volume_bar = -dc_loss_bar * loss / pow(volume, 2);

      /// double volume = calcOutput(output.volume, inputs);
      // volume does not depend on any of the inputs except mesh coords

      /// double loss = pow(output.rms_current, 2) * R * sqrt(2);
      // double rms_current_bar = loss_bar * 2 * output.rms_current * R *
      // sqrt(2);
      double R_bar = loss_bar * pow(output.rms_current, 2) * sqrt(2);

      /// double R = output.wire_length * rho / (strand_area *
      /// output.strands_in_hand);
      // double wire_length_bar = R_bar * rho / (strand_area *
      // output.strands_in_hand); double rho_bar = R_bar * output.wire_length /
      // (strand_area * output.strands_in_hand);
      double strand_area_bar = -R_bar * output.wire_length * rho /
                               (pow(strand_area, 2) * output.strands_in_hand);
      // double strands_in_hand_bar = -R_bar * output.wire_length * rho /
      // (strand_area * pow(output.strands_in_hand, 2));

      /// double strand_area = M_PI * pow(output.strand_radius, 2);
      double strand_radius_bar =
          strand_area_bar * M_PI * 2 * output.strand_radius;

      /// double rho = output.output.GetEnergy(output.scratch);
      // rho does not depend on any of the inputs except mesh coords

      return strand_radius_bar;
   }
   else if (wrt.rfind("strands_in_hand", 0) == 0)
   {
      double rho = calcOutput(output.resistivity, inputs);

      double strand_area = M_PI * pow(output.strand_radius, 2);
      // double R = output.wire_length * rho / (strand_area *
      // output.strands_in_hand);

      // double loss = pow(output.rms_current, 2) * R * sqrt(2);

      double volume = calcOutput(output.volume, inputs);
      // double dc_loss = loss / volume;

      /// Start reverse pass...
      double dc_loss_bar = out_bar(0);

      /// double dc_loss = loss / volume;
      double loss_bar = dc_loss_bar / volume;
      // double volume_bar = -dc_loss_bar * loss / pow(volume, 2);

      /// double volume = calcOutput(output.volume, inputs);
      // volume does not depend on any of the inputs except mesh coords

      /// double loss = pow(output.rms_current, 2) * R * sqrt(2);
      // double rms_current_bar = loss_bar * 2 * output.rms_current * R *
      // sqrt(2);
      double R_bar = loss_bar * pow(output.rms_current, 2) * sqrt(2);

      /// double R = output.wire_length * rho / (strand_area *
      /// output.strands_in_hand);
      // double wire_length_bar = R_bar * rho / (strand_area *
      // output.strands_in_hand); double rho_bar = R_bar * output.wire_length /
      // (strand_area * output.strands_in_hand); double strand_area_bar = -R_bar
      // * output.wire_length * rho / (pow(strand_area,2) *
      // output.strands_in_hand);
      double strands_in_hand_bar =
          -R_bar * output.wire_length * rho /
          (strand_area * pow(output.strands_in_hand, 2));

      /// double strand_area = M_PI * pow(output.strand_radius, 2);
      // double strand_radius_bar = strand_area_bar * M_PI * 2 *
      // output.strand_radius;

      /// double rho = output.output.GetEnergy(output.scratch);
      // rho does not depend on any of the inputs except mesh coords

      return strands_in_hand_bar;
   }
   else
   {
      return 0.0;
   }
}

void vectorJacobianProduct(DCLossFunctional &output,
                           const mfem::Vector &out_bar,
                           const std::string &wrt,
                           mfem::Vector &wrt_bar)
{
   const MachInputs &inputs = *output.inputs;
   if (wrt.rfind("mesh_coords", 0) == 0)
   {
      double rho = calcOutput(output.resistivity, inputs);

      double strand_area = M_PI * pow(output.strand_radius, 2);
      double R =
          output.wire_length * rho / (strand_area * output.strands_in_hand);

      double loss = pow(output.rms_current, 2) * R * sqrt(2);

      double volume = calcOutput(output.volume, inputs);
      // double dc_loss = loss / volume;

      /// Start reverse pass...
      double dc_loss_bar = out_bar(0);

      /// double dc_loss = loss / volume;
      double loss_bar = dc_loss_bar / volume;
      double volume_bar = -dc_loss_bar * loss / pow(volume, 2);

      /// double volume = calcOutput(output.volume, inputs);
      mfem::Vector vol_bar_vec(&volume_bar, 1);
      vectorJacobianProduct(output.volume, vol_bar_vec, wrt, wrt_bar);

      /// double loss = pow(output.rms_current, 2) * R * sqrt(2);
      // double rms_current_bar =
      //     loss_bar * 2 * output.rms_current * R * sqrt(2);
      double R_bar = loss_bar * pow(output.rms_current, 2) * sqrt(2);

      /// double R =
      ///     output.wire_length * rho / (strand_area * output.strands_in_hand);
      // double wire_length_bar =
      //     R_bar * rho / (strand_area * output.strands_in_hand);
      double rho_bar =
          R_bar * output.wire_length / (strand_area * output.strands_in_hand);
      // double strand_area_bar = -R_bar * output.wire_length * rho /
      //                          (pow(strand_area, 2) *
      //                          output.strands_in_hand);
      // double strands_in_hand_bar =
      //     -R_bar * output.wire_length * rho /
      //     (strand_area * pow(output.strands_in_hand, 2));

      /// double strand_area = M_PI * pow(output.strand_radius, 2);
      // double strand_radius_bar =
      //     strand_area_bar * M_PI * 2 * output.strand_radius;

      /// double rho = calcOutput(output.resistivity, inputs);
      mfem::Vector rho_bar_vec(&rho_bar, 1);
      vectorJacobianProduct(output.resistivity, rho_bar_vec, wrt, wrt_bar);
   }
}

DCLossFunctional::DCLossFunctional(
    std::map<std::string, FiniteElementState> &fields,
    mfem::Coefficient &sigma,
    const nlohmann::json &options)
 : resistivity(fields.at("state").space(), fields), volume(fields, options)
{
   if (options.contains("attributes"))
   {
      auto attributes = options["attributes"].get<std::vector<int>>();
      resistivity.addOutputDomainIntegrator(
          new DCLossFunctionalIntegrator(sigma), attributes);
   }
   else
   {
      resistivity.addOutputDomainIntegrator(
          new DCLossFunctionalIntegrator(sigma));
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
