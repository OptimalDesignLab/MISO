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

double EOutputsGlobalVariableCounter = 0;

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
   if (EOutputsGlobalVariableCounter < 10)
   {
      std::cout << "Ultimately remove EOutputsGlobalVariableCounter from "
                   "electromag_outputs.cpp\n";
      // std::cout << "rho = calcOutput(output.resistivity, inputs) = " << rho
      // << "\n"; std::cout << "strand_area = M_PI * pow(output.strand_radius,
      // 2) = PI * " << output.strand_radius << "^2 = " << strand_area << "\n";
      // std::cout << "R = output.wire_length * rho / (strand_area *
      // output.strands_in_hand) = " << output.wire_length << " * " << rho << "/
      // (" << strand_area << "*" << output.strands_in_hand << ") = " << R <<
      // "\n"; std::cout << "loss = sqrt(2) * pow(output.rms_current, 2) * R =
      // sqrt(2) *" << output.rms_current << "^2 * " << R << " = " << loss <<
      // "\n"; std::cout << "volume = calcOutput(output.volume, inputs) = " <<
      // volume << "\n"; std::cout << "loss/volume = " << loss/volume << "\n";
      EOutputsGlobalVariableCounter++;
   }
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
   // Adding in w/r/t temperature for the future. Untested.
   else if (wrt.rfind("temperature", 0) == 0)
   {
      std::cout << "temperature_dot = " << wrt_dot(0) << "\n";

      // double rho = calcOutput(output.resistivity, inputs); //not needed, rho
      // is linear in T, goes away

      double strand_area = M_PI * pow(output.strand_radius, 2);
      // double R =
      //     output.wire_length * rho / (strand_area * output.strands_in_hand);
      double R_dot = output.wire_length /
                     (strand_area * output.strands_in_hand) * wrt_dot(0);

      // double loss = pow(output.rms_current, 2) * R * sqrt(2);
      double loss_dot = pow(output.rms_current, 2) * sqrt(2) * R_dot;

      double volume = calcOutput(output.volume, inputs);

      // double dc_loss = loss / volume;
      double dc_loss_dot = 1 / volume * loss_dot;
      std::cout << "dc_loss_dot = " << dc_loss_dot << "\n";
      return dc_loss_dot;
   }
   // Adding in w/r/t temperature field for the future. Untested.
   else if (wrt.rfind("temperature_field", 0) == 0)
   {
      // double rho = calcOutput(output.resistivity, inputs);
      /// TODO: Determine if rho_dot computes correctly
      double rho_dot = jacobianVectorProduct(output.resistivity, wrt_dot, wrt);

      double strand_area = M_PI * pow(output.strand_radius, 2);
      // double R =
      //     output.wire_length * rho / (strand_area * output.strands_in_hand);
      double R_dot =
          output.wire_length / (strand_area * output.strands_in_hand) * rho_dot;

      // double loss = pow(output.rms_current, 2) * R * sqrt(2);
      double loss_dot = pow(output.rms_current, 2) * sqrt(2) * R_dot;

      double volume = calcOutput(output.volume, inputs);

      // double dc_loss = loss / volume;
      double dc_loss_dot = 1 / volume * loss_dot;
      std::cout << "dc_loss_dot = " << dc_loss_dot << "\n";
      return dc_loss_dot;
   }
   else
   {
      return 0.0;
   }
}

// void jacobianVectorProduct(DCLossFunctional &output,
//                            const mfem::Vector &wrt_dot,
//                            const std::string &wrt,
//                            mfem::Vector &out_dot)
// { }

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
   // Adding in w/r/t temperature for the future. Untested.
   else if (wrt.rfind("temperature", 0) == 0)
   {
      // double rho = calcOutput(output.resistivity, inputs); //not needed, rho
      // is linear in T, goes away

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
          output.wire_length * R_bar / (strand_area * output.strands_in_hand);

      return wire_length_bar;
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
   // Adding in w/r/t temperature field for the future. Untested.
   else if (wrt.rfind("temperature_field", 0) == 0)
   {
      // double rho = calcOutput(output.resistivity, inputs);

      double strand_area = M_PI * pow(output.strand_radius, 2);
      // double R =
      //     output.wire_length * rho / (strand_area * output.strands_in_hand);

      // double loss = pow(output.rms_current, 2) * R * sqrt(2);

      double volume = calcOutput(output.volume, inputs);
      // double dc_loss = loss / volume;

      /// Start reverse pass...
      double dc_loss_bar = out_bar(0);

      /// double dc_loss = loss / volume;
      double loss_bar = dc_loss_bar / volume;
      // double volume_bar = -dc_loss_bar * loss / pow(volume, 2);

      /// double volume = calcOutput(output.volume, inputs);
      // mfem::Vector vol_bar_vec(&volume_bar, 1);
      // vectorJacobianProduct(output.volume, vol_bar_vec, wrt, wrt_bar);

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

// Made sigma a StateCoefficient (was formerly an mfem::coefficient)
/// Also make this functional see the pre-computed temperature field
DCLossFunctional::DCLossFunctional(
    std::map<std::string, FiniteElementState> &fields,
    StateCoefficient &sigma,
    const nlohmann::json &options)
 : resistivity(fields.at("state").space(), fields), volume(fields, options)
{
   // Making the integrator see the temperature field
   const auto &temp_field_iter =
       fields.find("temperature");  // find where temperature field is
   mfem::GridFunction *temperature_field =
       nullptr;  // default temperature field to null pointer
   if (temp_field_iter != fields.end())
   {
      // If temperature field exists, turn it into a grid function
      auto &temp_field = temp_field_iter->second;
      temperature_field = &temp_field.gridFunc();
   }

   // Assign the integrator used to compute the DC losses
   if (options.contains("attributes"))
   {
      auto attributes = options["attributes"].get<std::vector<int>>();
      resistivity.addOutputDomainIntegrator(
          new DCLossFunctionalIntegrator(sigma, temperature_field), attributes);
      // std::cout << "TODO: Ultimately remove from DCLF in Eoutputs cpp.
      // attributes=\n"; for (const auto &attribute : attributes)
      // {
      //    std::cout << attribute << ", ";
      // }
      // std::cout << "])\n";
   }
   else
   {
      resistivity.addOutputDomainIntegrator(
          new DCLossFunctionalIntegrator(sigma, temperature_field));
      std::cout << "In the else\n";
   }
}

void setOptions(ACLossFunctional &output, const nlohmann::json &options)
{
   // output.num_strands = options.value("num_strands", output.num_strands);
   setOptions(output.output, options);
}

void setInputs(ACLossFunctional &output, const MachInputs &inputs)
{
   output.inputs = inputs;
   output.inputs["state"] = inputs.at("peak_flux");

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
   setInputs(output, inputs);

   // mfem::Vector flux_state;
   // setVectorFromInputs(inputs, "peak_flux", flux_state, false, true);
   /// TODO: Remove once done debugging
   // std::cout << "flux_state.Size() = " << flux_state.Size() << "\n";
   // std::cout << "flux_state=np.array([";
   // for (int j = 0; j < flux_state.Size(); j++) {std::cout <<
   // flux_state.Elem(j) << ", ";} std::cout << "])\n";
   // auto &flux_mag = output.fields.at("peak_flux");
   // flux_mag.distributeSharedDofs(flux_state);
   // mfem::ParaViewDataCollection pv("FluxMag", &flux_mag.mesh());
   // pv.SetPrefixPath("ParaView");
   // pv.SetLevelsOfDetail(3);
   // pv.SetDataFormat(mfem::VTKFormat::BINARY);
   // pv.SetHighOrderOutput(true);
   // pv.RegisterField("FluxMag", &flux_mag.gridFunc());
   // pv.Save();

   double sigma_b2 = calcOutput(output.output, output.inputs);

   double strand_loss = sigma_b2 * output.stack_length * M_PI *
                        pow(output.radius, 4) * pow(2 * M_PI * output.freq, 2) /
                        8.0;

   double num_strands =
       2 * output.strands_in_hand * output.num_turns * output.num_slots;

   double loss = num_strands * strand_loss;

   double volume = calcOutput(output.volume, output.inputs);

   return loss / volume;
}

double jacobianVectorProduct(ACLossFunctional &output,
                             const mfem::Vector &wrt_dot,
                             const std::string &wrt)
{
   if (wrt.rfind("strand_radius", 0) == 0)
   {
      double sigma_b2 = calcOutput(output.output, output.inputs);

      // double strand_loss = sigma_b2 * output.stack_length * M_PI *
      //                      pow(output.radius, 4) *
      //                      pow(2 * M_PI * output.freq, 2) / 8.0;
      double strand_loss_dot =
          4 * sigma_b2 * output.stack_length * M_PI * pow(output.radius, 3) *
          pow(2 * M_PI * output.freq, 2) / 8.0 * wrt_dot(0);

      double num_strands =
          2 * output.strands_in_hand * output.num_turns * output.num_slots;

      // double loss = num_strands * strand_loss;
      double loss_dot = num_strands * strand_loss_dot;

      double volume = calcOutput(output.volume, output.inputs);

      return loss_dot / volume;
   }
   else if (wrt.rfind("frequency", 0) == 0)
   {
      double sigma_b2 = calcOutput(output.output, output.inputs);

      // double strand_loss = sigma_b2 * output.stack_length * M_PI *
      //                      pow(output.radius, 4) *
      //                      pow(2 * M_PI * output.freq, 2) / 8.0;
      double strand_loss_dot = 2 * sigma_b2 * output.stack_length * M_PI *
                               pow(output.radius, 3) * output.freq *
                               pow(2 * M_PI, 2) / 8.0 * wrt_dot(0);

      double num_strands =
          2 * output.strands_in_hand * output.num_turns * output.num_slots;

      // double loss = num_strands * strand_loss;
      double loss_dot = num_strands * strand_loss_dot;

      double volume = calcOutput(output.volume, output.inputs);

      return loss_dot / volume;
   }
   else if (wrt.rfind("stack_length", 0) == 0)
   {
      double sigma_b2 = calcOutput(output.output, output.inputs);

      // double strand_loss = sigma_b2 * output.stack_length * M_PI *
      //                      pow(output.radius, 4) *
      //                      pow(2 * M_PI * output.freq, 2) / 8.0;
      double strand_loss_dot = sigma_b2 * M_PI * pow(output.radius, 4) *
                               pow(2 * M_PI * output.freq, 2) / 8.0 *
                               wrt_dot(0);

      double num_strands =
          2 * output.strands_in_hand * output.num_turns * output.num_slots;

      // double loss = num_strands * strand_loss;
      double loss_dot = num_strands * strand_loss_dot;

      double volume = calcOutput(output.volume, output.inputs);

      return loss_dot / volume;
   }
   else if (wrt.rfind("strands_in_hand", 0) == 0)
   {
      double sigma_b2 = calcOutput(output.output, output.inputs);

      double strand_loss = sigma_b2 * output.stack_length * M_PI *
                           pow(output.radius, 4) *
                           pow(2 * M_PI * output.freq, 2) / 8.0;

      // double num_strands =
      //     2 * output.strands_in_hand * output.num_turns * output.num_slots;

      double num_strands_dot =
          2 * output.num_turns * output.num_slots * wrt_dot(0);

      // double loss = num_strands * strand_loss;
      double loss_dot = strand_loss * num_strands_dot;

      double volume = calcOutput(output.volume, output.inputs);

      return loss_dot / volume;
   }
   else if (wrt.rfind("num_turns", 0) == 0)
   {
      double sigma_b2 = calcOutput(output.output, output.inputs);

      double strand_loss = sigma_b2 * output.stack_length * M_PI *
                           pow(output.radius, 4) *
                           pow(2 * M_PI * output.freq, 2) / 8.0;

      // double num_strands =
      //     2 * output.strands_in_hand * output.num_turns * output.num_slots;

      double num_strands_dot =
          2 * output.strands_in_hand * output.num_slots * wrt_dot(0);

      // double loss = num_strands * strand_loss;
      double loss_dot = strand_loss * num_strands_dot;

      double volume = calcOutput(output.volume, output.inputs);

      return loss_dot / volume;
   }
   else if (wrt.rfind("num_slots", 0) == 0)
   {
      double sigma_b2 = calcOutput(output.output, output.inputs);

      double strand_loss = sigma_b2 * output.stack_length * M_PI *
                           pow(output.radius, 4) *
                           pow(2 * M_PI * output.freq, 2) / 8.0;

      // double num_strands =
      //     2 * output.strands_in_hand * output.num_turns * output.num_slots;

      double num_strands_dot =
          2 * output.strands_in_hand * output.num_turns * wrt_dot(0);

      // double loss = num_strands * strand_loss;
      double loss_dot = strand_loss * num_strands_dot;

      double volume = calcOutput(output.volume, output.inputs);

      return loss_dot / volume;
   }
   else if (wrt.rfind("mesh_coords", 0) == 0)
   {
      double sigma_b2 = calcOutput(output.output, output.inputs);
      double sigma_b2_dot = jacobianVectorProduct(output.output, wrt_dot, wrt);

      double strand_loss = sigma_b2 * output.stack_length * M_PI *
                           pow(output.radius, 4) *
                           pow(2 * M_PI * output.freq, 2) / 8.0;

      double strand_loss_dot =
          output.stack_length * M_PI * pow(output.radius, 4) *
          pow(2 * M_PI * output.freq, 2) / 8.0 * sigma_b2_dot;

      double num_strands =
          2 * output.strands_in_hand * output.num_turns * output.num_slots;

      double loss = num_strands * strand_loss;
      double loss_dot = num_strands * strand_loss_dot;

      double volume = calcOutput(output.volume, output.inputs);
      double volume_dot = jacobianVectorProduct(output.volume, wrt_dot, wrt);

      return loss_dot / volume - loss / pow(volume, 2) * volume_dot;
   }
   else if (wrt.rfind("peak_flux", 0) == 0)
   {
      // double sigma_b2 = calcOutput(output.output, output.inputs);
      double sigma_b2_dot = jacobianVectorProduct(output.output, wrt_dot, wrt);

      // double strand_loss = sigma_b2 * output.stack_length * M_PI *
      //                      pow(output.radius, 4) *
      //                      pow(2 * M_PI * output.freq, 2) / 8.0;

      double strand_loss_dot =
          output.stack_length * M_PI * pow(output.radius, 4) *
          pow(2 * M_PI * output.freq, 2) / 8.0 * sigma_b2_dot;

      double num_strands =
          2 * output.strands_in_hand * output.num_turns * output.num_slots;

      // double loss = num_strands * strand_loss;
      double loss_dot = num_strands * strand_loss_dot;

      double volume = calcOutput(output.volume, output.inputs);
      // double volume_dot = jacobianVectorProduct(output.volume, wrt_dot, wrt);

      return loss_dot / volume;  // - loss / pow(volume, 2) * volume_dot;
   }
   // Adding in w/r/t temperature for the future. Untested.
   else if (wrt.rfind("temperature", 0) == 0)
   {
      // double sigma_b2 = calcOutput(output.output, output.inputs);
      /// TODO: Determine if the sigma_b2_dot defined below computes correctly.
      /// Should equal sigma_b2/(alpha*(T-Tref)).
      double sigma_b2_dot = jacobianVectorProduct(output.output, wrt_dot, wrt);

      // double strand_loss = sigma_b2 * output.stack_length * M_PI *
      //                      pow(output.radius, 4) *
      //                      pow(2 * M_PI * output.freq, 2) / 8.0;

      double strand_loss_dot =
          output.stack_length * M_PI * pow(output.radius, 4) *
          pow(2 * M_PI * output.freq, 2) / 8.0 * sigma_b2_dot;

      double num_strands =
          2 * output.strands_in_hand * output.num_turns * output.num_slots;

      // double loss = num_strands * strand_loss;
      double loss_dot = num_strands * strand_loss_dot;

      double volume = calcOutput(output.volume, output.inputs);
      // double volume_dot = jacobianVectorProduct(output.volume, wrt_dot, wrt);
      // // volume is independent of temperature

      return loss_dot / volume;
   }
   else
   {
      return 0.0;
   }
}

double vectorJacobianProduct(ACLossFunctional &output,
                             const mfem::Vector &out_bar,
                             const std::string &wrt)
{
   if (wrt.rfind("strand_radius", 0) == 0)
   {
      double sigma_b2 = calcOutput(output.output, output.inputs);

      // double strand_loss = sigma_b2 * output.stack_length * M_PI *
      //                      pow(output.radius, 4) *
      //                      pow(2 * M_PI * output.freq, 2) / 8.0;
      double num_strands =
          2 * output.strands_in_hand * output.num_turns * output.num_slots;

      // double loss = num_strands * strand_loss;

      double volume = calcOutput(output.volume, output.inputs);

      // double ac_loss = loss / volume;

      /// Start reverse pass...
      double ac_loss_bar = out_bar(0);

      /// double ac_loss = loss / volume;
      double loss_bar = ac_loss_bar / volume;
      // double volume_bar = -ac_loss_bar * loss / pow(volume, 2);

      /// double volume = calcOutput(output.volume, inputs);
      // volume does not depend on any of the inputs except mesh coords

      /// double loss = num_strands * strand_loss;
      // double num_strands_bar = loss_bar * strand_loss;
      double strand_loss_bar = loss_bar * num_strands;

      /// double num_strands =
      ///     2 * output.strands_in_hand * output.num_turns * output.num_slots;
      // double strands_in_hand_bar =
      //     num_strands_bar * 2 * output.num_turns * output.num_slots;
      // double num_turns_bar =
      //     num_strands_bar * 2 * output.strands_in_hand * output.num_slots;
      // double num_slots_bar =
      //     num_strands_bar * 2 * output.strands_in_hand * output.num_turns;

      /// double strand_loss = sigma_b2 * output.stack_length * M_PI *
      ///                      pow(output.radius, 4) *
      ///                      pow(2 * M_PI * output.freq, 2) / 8.0;
      // double sigma_b2_bar = strand_loss_bar * output.stack_length * M_PI *
      //                       pow(output.radius, 4) *
      //                       pow(2 * M_PI * output.freq, 2) / 8.0;
      // double stack_length_bar = strand_loss_bar * sigma_b2 * M_PI *
      //                           pow(output.radius, 4) *
      //                           pow(2 * M_PI * output.freq, 2) / 8.0;
      double strand_radius_bar =
          strand_loss_bar * sigma_b2 * output.stack_length * M_PI * 4 *
          pow(output.radius, 3) * pow(2 * M_PI * output.freq, 2) / 8.0;
      // double frequency_bar = strand_loss_bar * sigma_b2 * output.stack_length
      // *
      //                        M_PI * pow(output.radius, 4) * 2 * output.freq *
      //                        pow(2 * M_PI, 2) / 8.0;

      return strand_radius_bar;
   }
   else if (wrt.rfind("frequency", 0) == 0)
   {
      double sigma_b2 = calcOutput(output.output, output.inputs);

      // double strand_loss = sigma_b2 * output.stack_length * M_PI *
      //                      pow(output.radius, 4) *
      //                      pow(2 * M_PI * output.freq, 2) / 8.0;
      double num_strands =
          2 * output.strands_in_hand * output.num_turns * output.num_slots;

      // double loss = num_strands * strand_loss;

      double volume = calcOutput(output.volume, output.inputs);

      // double ac_loss = loss / volume;

      /// Start reverse pass...
      double ac_loss_bar = out_bar(0);

      /// double ac_loss = loss / volume;
      double loss_bar = ac_loss_bar / volume;
      // double volume_bar = -ac_loss_bar * loss / pow(volume, 2);

      /// double volume = calcOutput(output.volume, inputs);
      // volume does not depend on any of the inputs except mesh coords

      /// double loss = num_strands * strand_loss;
      // double num_strands_bar = loss_bar * strand_loss;
      double strand_loss_bar = loss_bar * num_strands;

      /// double num_strands =
      ///     2 * output.strands_in_hand * output.num_turns * output.num_slots;
      // double strands_in_hand_bar =
      //     num_strands_bar * 2 * output.num_turns * output.num_slots;
      // double num_turns_bar =
      //     num_strands_bar * 2 * output.strands_in_hand * output.num_slots;
      // double num_slots_bar =
      //     num_strands_bar * 2 * output.strands_in_hand * output.num_turns;

      /// double strand_loss = sigma_b2 * output.stack_length * M_PI *
      ///                      pow(output.radius, 4) *
      ///                      pow(2 * M_PI * output.freq, 2) / 8.0;
      // double sigma_b2_bar = strand_loss_bar * output.stack_length * M_PI *
      //                       pow(output.radius, 4) *
      //                       pow(2 * M_PI * output.freq, 2) / 8.0;
      // double stack_length_bar = strand_loss_bar * sigma_b2 * M_PI *
      //                           pow(output.radius, 4) *
      //                           pow(2 * M_PI * output.freq, 2) / 8.0;
      // double strand_radius_bar =
      //     strand_loss_bar * sigma_b2 * output.stack_length * M_PI * 4 *
      //     pow(output.radius, 3) * pow(2 * M_PI * output.freq, 2) / 8.0;
      double frequency_bar = strand_loss_bar * sigma_b2 * output.stack_length *
                             M_PI * pow(output.radius, 4) * 2 * output.freq *
                             pow(2 * M_PI, 2) / 8.0;

      return frequency_bar;
   }
   else if (wrt.rfind("stack_length", 0) == 0)
   {
      double sigma_b2 = calcOutput(output.output, output.inputs);

      // double strand_loss = sigma_b2 * output.stack_length * M_PI *
      //                      pow(output.radius, 4) *
      //                      pow(2 * M_PI * output.freq, 2) / 8.0;
      double num_strands =
          2 * output.strands_in_hand * output.num_turns * output.num_slots;

      // double loss = num_strands * strand_loss;

      double volume = calcOutput(output.volume, output.inputs);

      // double ac_loss = loss / volume;

      /// Start reverse pass...
      double ac_loss_bar = out_bar(0);

      /// double ac_loss = loss / volume;
      double loss_bar = ac_loss_bar / volume;
      // double volume_bar = -ac_loss_bar * loss / pow(volume, 2);

      /// double volume = calcOutput(output.volume, inputs);
      // volume does not depend on any of the inputs except mesh coords

      /// double loss = num_strands * strand_loss;
      // double num_strands_bar = loss_bar * strand_loss;
      double strand_loss_bar = loss_bar * num_strands;

      /// double num_strands =
      ///     2 * output.strands_in_hand * output.num_turns * output.num_slots;
      // double strands_in_hand_bar =
      //     num_strands_bar * 2 * output.num_turns * output.num_slots;
      // double num_turns_bar =
      //     num_strands_bar * 2 * output.strands_in_hand * output.num_slots;
      // double num_slots_bar =
      //     num_strands_bar * 2 * output.strands_in_hand * output.num_turns;

      /// double strand_loss = sigma_b2 * output.stack_length * M_PI *
      ///                      pow(output.radius, 4) *
      ///                      pow(2 * M_PI * output.freq, 2) / 8.0;
      // double sigma_b2_bar = strand_loss_bar * output.stack_length * M_PI *
      //                       pow(output.radius, 4) *
      //                       pow(2 * M_PI * output.freq, 2) / 8.0;
      double stack_length_bar = strand_loss_bar * sigma_b2 * M_PI *
                                pow(output.radius, 4) *
                                pow(2 * M_PI * output.freq, 2) / 8.0;
      // double strand_radius_bar =
      //     strand_loss_bar * sigma_b2 * output.stack_length * M_PI * 4 *
      //     pow(output.radius, 3) * pow(2 * M_PI * output.freq, 2) / 8.0;
      // double frequency_bar = strand_loss_bar * sigma_b2 * output.stack_length
      // *
      //                        M_PI * pow(output.radius, 4) * 2 * output.freq *
      //                        pow(2 * M_PI, 2) / 8.0;

      return stack_length_bar;
   }
   else if (wrt.rfind("strands_in_hand", 0) == 0)
   {
      double sigma_b2 = calcOutput(output.output, output.inputs);

      double strand_loss = sigma_b2 * output.stack_length * M_PI *
                           pow(output.radius, 4) *
                           pow(2 * M_PI * output.freq, 2) / 8.0;
      // double num_strands =
      //     2 * output.strands_in_hand * output.num_turns * output.num_slots;

      // double loss = num_strands * strand_loss;

      double volume = calcOutput(output.volume, output.inputs);

      // double ac_loss = loss / volume;

      /// Start reverse pass...
      double ac_loss_bar = out_bar(0);

      /// double ac_loss = loss / volume;
      double loss_bar = ac_loss_bar / volume;
      // double volume_bar = -ac_loss_bar * loss / pow(volume, 2);

      /// double volume = calcOutput(output.volume, inputs);
      // volume does not depend on any of the inputs except mesh coords

      /// double loss = num_strands * strand_loss;
      double num_strands_bar = loss_bar * strand_loss;
      // double strand_loss_bar = loss_bar * num_strands;

      /// double num_strands =
      ///     2 * output.strands_in_hand * output.num_turns * output.num_slots;
      double strands_in_hand_bar =
          num_strands_bar * 2 * output.num_turns * output.num_slots;
      // double num_turns_bar =
      //     num_strands_bar * 2 * output.strands_in_hand * output.num_slots;
      // double num_slots_bar =
      //     num_strands_bar * 2 * output.strands_in_hand * output.num_turns;

      /// double strand_loss = sigma_b2 * output.stack_length * M_PI *
      ///                      pow(output.radius, 4) *
      ///                      pow(2 * M_PI * output.freq, 2) / 8.0;
      // double sigma_b2_bar = strand_loss_bar * output.stack_length * M_PI *
      //                       pow(output.radius, 4) *
      //                       pow(2 * M_PI * output.freq, 2) / 8.0;
      // double stack_length_bar = strand_loss_bar * sigma_b2 * M_PI *
      //                           pow(output.radius, 4) *
      //                           pow(2 * M_PI * output.freq, 2) / 8.0;
      // double strand_radius_bar =
      //     strand_loss_bar * sigma_b2 * output.stack_length * M_PI * 4 *
      //     pow(output.radius, 3) * pow(2 * M_PI * output.freq, 2) / 8.0;
      // double frequency_bar = strand_loss_bar * sigma_b2 * output.stack_length
      // *
      //                        M_PI * pow(output.radius, 4) * 2 * output.freq *
      //                        pow(2 * M_PI, 2) / 8.0;
      return strands_in_hand_bar;
   }
   else if (wrt.rfind("num_turns", 0) == 0)
   {
      double sigma_b2 = calcOutput(output.output, output.inputs);

      double strand_loss = sigma_b2 * output.stack_length * M_PI *
                           pow(output.radius, 4) *
                           pow(2 * M_PI * output.freq, 2) / 8.0;
      // double num_strands =
      //     2 * output.strands_in_hand * output.num_turns * output.num_slots;

      // double loss = num_strands * strand_loss;

      double volume = calcOutput(output.volume, output.inputs);

      // double ac_loss = loss / volume;

      /// Start reverse pass...
      double ac_loss_bar = out_bar(0);

      /// double ac_loss = loss / volume;
      double loss_bar = ac_loss_bar / volume;
      // double volume_bar = -ac_loss_bar * loss / pow(volume, 2);

      /// double volume = calcOutput(output.volume, inputs);
      // volume does not depend on any of the inputs except mesh coords

      /// double loss = num_strands * strand_loss;
      double num_strands_bar = loss_bar * strand_loss;
      // double strand_loss_bar = loss_bar * num_strands;

      /// double num_strands =
      ///     2 * output.strands_in_hand * output.num_turns * output.num_slots;
      // double strands_in_hand_bar =
      //     num_strands_bar * 2 * output.num_turns * output.num_slots;
      double num_turns_bar =
          num_strands_bar * 2 * output.strands_in_hand * output.num_slots;
      // double num_slots_bar =
      //     num_strands_bar * 2 * output.strands_in_hand * output.num_turns;

      /// double strand_loss = sigma_b2 * output.stack_length * M_PI *
      ///                      pow(output.radius, 4) *
      ///                      pow(2 * M_PI * output.freq, 2) / 8.0;
      // double sigma_b2_bar = strand_loss_bar * output.stack_length * M_PI *
      //                       pow(output.radius, 4) *
      //                       pow(2 * M_PI * output.freq, 2) / 8.0;
      // double stack_length_bar = strand_loss_bar * sigma_b2 * M_PI *
      //                           pow(output.radius, 4) *
      //                           pow(2 * M_PI * output.freq, 2) / 8.0;
      // double strand_radius_bar =
      //     strand_loss_bar * sigma_b2 * output.stack_length * M_PI * 4 *
      //     pow(output.radius, 3) * pow(2 * M_PI * output.freq, 2) / 8.0;
      // double frequency_bar = strand_loss_bar * sigma_b2 * output.stack_length
      // *
      //                        M_PI * pow(output.radius, 4) * 2 * output.freq *
      //                        pow(2 * M_PI, 2) / 8.0;
      return num_turns_bar;
   }
   else if (wrt.rfind("num_slots", 0) == 0)
   {
      double sigma_b2 = calcOutput(output.output, output.inputs);

      double strand_loss = sigma_b2 * output.stack_length * M_PI *
                           pow(output.radius, 4) *
                           pow(2 * M_PI * output.freq, 2) / 8.0;
      // double num_strands =
      //     2 * output.strands_in_hand * output.num_turns * output.num_slots;

      // double loss = num_strands * strand_loss;

      double volume = calcOutput(output.volume, output.inputs);

      // double ac_loss = loss / volume;

      /// Start reverse pass...
      double ac_loss_bar = out_bar(0);

      /// double ac_loss = loss / volume;
      double loss_bar = ac_loss_bar / volume;
      // double volume_bar = -ac_loss_bar * loss / pow(volume, 2);

      /// double volume = calcOutput(output.volume, inputs);
      // volume does not depend on any of the inputs except mesh coords

      /// double loss = num_strands * strand_loss;
      double num_strands_bar = loss_bar * strand_loss;
      // double strand_loss_bar = loss_bar * num_strands;

      /// double num_strands =
      ///     2 * output.strands_in_hand * output.num_turns * output.num_slots;
      // double strands_in_hand_bar =
      //     num_strands_bar * 2 * output.num_turns * output.num_slots;
      // double num_turns_bar =
      //     num_strands_bar * 2 * output.strands_in_hand * output.num_slots;
      double num_slots_bar =
          num_strands_bar * 2 * output.strands_in_hand * output.num_turns;

      /// double strand_loss = sigma_b2 * output.stack_length * M_PI *
      ///                      pow(output.radius, 4) *
      ///                      pow(2 * M_PI * output.freq, 2) / 8.0;
      // double sigma_b2_bar = strand_loss_bar * output.stack_length * M_PI *
      //                       pow(output.radius, 4) *
      //                       pow(2 * M_PI * output.freq, 2) / 8.0;
      // double stack_length_bar = strand_loss_bar * sigma_b2 * M_PI *
      //                           pow(output.radius, 4) *
      //                           pow(2 * M_PI * output.freq, 2) / 8.0;
      // double strand_radius_bar =
      //     strand_loss_bar * sigma_b2 * output.stack_length * M_PI * 4 *
      //     pow(output.radius, 3) * pow(2 * M_PI * output.freq, 2) / 8.0;
      // double frequency_bar = strand_loss_bar * sigma_b2 * output.stack_length
      // *
      //                        M_PI * pow(output.radius, 4) * 2 * output.freq *
      //                        pow(2 * M_PI, 2) / 8.0;
      return num_slots_bar;
   }
   else
   {
      return 0.0;
   }
}

void vectorJacobianProduct(ACLossFunctional &output,
                           const mfem::Vector &out_bar,
                           const std::string &wrt,
                           mfem::Vector &wrt_bar)
{
   if (wrt.rfind("mesh_coords", 0) == 0)
   {
      double sigma_b2 = calcOutput(output.output, output.inputs);

      double strand_loss = sigma_b2 * output.stack_length * M_PI *
                           pow(output.radius, 4) *
                           pow(2 * M_PI * output.freq, 2) / 8.0;
      double num_strands =
          2 * output.strands_in_hand * output.num_turns * output.num_slots;

      double loss = num_strands * strand_loss;

      double volume = calcOutput(output.volume, output.inputs);

      // double ac_loss = loss / volume;

      /// Start reverse pass...
      double ac_loss_bar = out_bar(0);

      /// double ac_loss = loss / volume;
      double loss_bar = ac_loss_bar / volume;
      double volume_bar = -ac_loss_bar * loss / pow(volume, 2);

      /// double volume = calcOutput(output.volume, inputs);
      mfem::Vector vol_bar_vec(&volume_bar, 1);
      vectorJacobianProduct(output.volume, vol_bar_vec, wrt, wrt_bar);

      /// double loss = num_strands * strand_loss;
      // double num_strands_bar = loss_bar * strand_loss;
      double strand_loss_bar = loss_bar * num_strands;

      /// double num_strands =
      ///     2 * output.strands_in_hand * output.num_turns * output.num_slots;
      // double strands_in_hand_bar =
      //     num_strands_bar * 2 * output.num_turns * output.num_slots;
      // double num_turns_bar =
      //     num_strands_bar * 2 * output.strands_in_hand * output.num_slots;
      // double num_slots_bar =
      //     num_strands_bar * 2 * output.strands_in_hand * output.num_turns;

      /// double strand_loss = sigma_b2 * output.stack_length * M_PI *
      ///                      pow(output.radius, 4) *
      ///                      pow(2 * M_PI * output.freq, 2) / 8.0;
      double sigma_b2_bar = strand_loss_bar * output.stack_length * M_PI *
                            pow(output.radius, 4) *
                            pow(2 * M_PI * output.freq, 2) / 8.0;
      // double stack_length_bar = strand_loss_bar * sigma_b2 * M_PI *
      //                           pow(output.radius, 4) *
      //                           pow(2 * M_PI * output.freq, 2) / 8.0;
      // double strand_radius_bar =
      //     strand_loss_bar * sigma_b2 * output.stack_length * M_PI * 4 *
      //     pow(output.radius, 3) * pow(2 * M_PI * output.freq, 2) / 8.0;
      // double frequency_bar = strand_loss_bar * sigma_b2 * output.stack_length
      // *
      //                        M_PI * pow(output.radius, 4) * 2 * output.freq *
      //                        pow(2 * M_PI, 2) / 8.0;

      /// double sigma_b2 = calcOutput(output.output, output.inputs);
      mfem::Vector sigma_b2_bar_vec(&sigma_b2_bar, 1);
      vectorJacobianProduct(output.output, sigma_b2_bar_vec, wrt, wrt_bar);
   }
   else if (wrt.rfind("peak_flux", 0) == 0)
   {
      // double sigma_b2 = calcOutput(output.output, output.inputs);

      // double strand_loss = sigma_b2 * output.stack_length * M_PI *
      //                      pow(output.radius, 4) *
      //                      pow(2 * M_PI * output.freq, 2) / 8.0;
      double num_strands =
          2 * output.strands_in_hand * output.num_turns * output.num_slots;

      // double loss = num_strands * strand_loss;

      double volume = calcOutput(output.volume, output.inputs);

      // double ac_loss = loss / volume;

      /// Start reverse pass...
      double ac_loss_bar = out_bar(0);

      /// double ac_loss = loss / volume;
      double loss_bar = ac_loss_bar / volume;
      // double volume_bar = -ac_loss_bar * loss / pow(volume, 2);

      /// double volume = calcOutput(output.volume, inputs);
      // mfem::Vector vol_bar_vec(&volume_bar, 1);
      // vectorJacobianProduct(output.volume, vol_bar_vec, "state", wrt_bar);

      /// double loss = num_strands * strand_loss;
      // double num_strands_bar = loss_bar * strand_loss;
      double strand_loss_bar = loss_bar * num_strands;

      /// double num_strands =
      ///     2 * output.strands_in_hand * output.num_turns * output.num_slots;
      // double strands_in_hand_bar =
      //     num_strands_bar * 2 * output.num_turns * output.num_slots;
      // double num_turns_bar =
      //     num_strands_bar * 2 * output.strands_in_hand * output.num_slots;
      // double num_slots_bar =
      //     num_strands_bar * 2 * output.strands_in_hand * output.num_turns;

      /// double strand_loss = sigma_b2 * output.stack_length * M_PI *
      ///                      pow(output.radius, 4) *
      ///                      pow(2 * M_PI * output.freq, 2) / 8.0;
      double sigma_b2_bar = strand_loss_bar * output.stack_length * M_PI *
                            pow(output.radius, 4) *
                            pow(2 * M_PI * output.freq, 2) / 8.0;
      // double stack_length_bar = strand_loss_bar * sigma_b2 * M_PI *
      //                           pow(output.radius, 4) *
      //                           pow(2 * M_PI * output.freq, 2) / 8.0;
      // double strand_radius_bar =
      //     strand_loss_bar * sigma_b2 * output.stack_length * M_PI * 4 *
      //     pow(output.radius, 3) * pow(2 * M_PI * output.freq, 2) / 8.0;
      // double frequency_bar = strand_loss_bar * sigma_b2 * output.stack_length
      // *
      //                        M_PI * pow(output.radius, 4) * 2 * output.freq *
      //                        pow(2 * M_PI, 2) / 8.0;

      /// double sigma_b2 = calcOutput(output.output, output.inputs);
      mfem::Vector sigma_b2_bar_vec(&sigma_b2_bar, 1);
      vectorJacobianProduct(output.output, sigma_b2_bar_vec, wrt, wrt_bar);
   }
   // Adding in w/r/t temperature for the future. Untested.
   else if (wrt.rfind("temperature", 0) == 0)
   {
      // double sigma_b2 = calcOutput(output.output, output.inputs);

      // double strand_loss = sigma_b2 * output.stack_length * M_PI *
      //                      pow(output.radius, 4) *
      //                      pow(2 * M_PI * output.freq, 2) / 8.0;
      double num_strands =
          2 * output.strands_in_hand * output.num_turns * output.num_slots;

      // double loss = num_strands * strand_loss;

      double volume = calcOutput(output.volume, output.inputs);

      // double ac_loss = loss / volume;

      /// Start reverse pass...
      double ac_loss_bar = out_bar(0);

      /// double ac_loss = loss / volume;
      double loss_bar = ac_loss_bar / volume;
      // double volume_bar = -ac_loss_bar * loss / pow(volume, 2);

      /// double volume = calcOutput(output.volume, inputs);
      // mfem::Vector vol_bar_vec(&volume_bar, 1);
      // vectorJacobianProduct(output.volume, vol_bar_vec, "state", wrt_bar);

      /// double loss = num_strands * strand_loss;
      // double num_strands_bar = loss_bar * strand_loss;
      double strand_loss_bar = loss_bar * num_strands;

      /// double num_strands =
      ///     2 * output.strands_in_hand * output.num_turns * output.num_slots;
      // double strands_in_hand_bar =
      //     num_strands_bar * 2 * output.num_turns * output.num_slots;
      // double num_turns_bar =
      //     num_strands_bar * 2 * output.strands_in_hand * output.num_slots;
      // double num_slots_bar =
      //     num_strands_bar * 2 * output.strands_in_hand * output.num_turns;

      /// double strand_loss = sigma_b2 * output.stack_length * M_PI *
      ///                      pow(output.radius, 4) *
      ///                      pow(2 * M_PI * output.freq, 2) / 8.0;
      double sigma_b2_bar = strand_loss_bar * output.stack_length * M_PI *
                            pow(output.radius, 4) *
                            pow(2 * M_PI * output.freq, 2) / 8.0;
      // double stack_length_bar = strand_loss_bar * sigma_b2 * M_PI *
      //                           pow(output.radius, 4) *
      //                           pow(2 * M_PI * output.freq, 2) / 8.0;
      // double strand_radius_bar =
      //     strand_loss_bar * sigma_b2 * output.stack_length * M_PI * 4 *
      //     pow(output.radius, 3) * pow(2 * M_PI * output.freq, 2) / 8.0;
      // double frequency_bar = strand_loss_bar * sigma_b2 * output.stack_length
      // *
      //                        M_PI * pow(output.radius, 4) * 2 * output.freq *
      //                        pow(2 * M_PI, 2) / 8.0;

      /// TODO: Determine if the sigma_b2_bar defined below computes correctly.
      /// That is, is vectorJacobianProduct in functional_output.cpp correct?
      ///  double sigma_b2 = calcOutput(output.output, output.inputs);
      mfem::Vector sigma_b2_bar_vec(&sigma_b2_bar, 1);
      vectorJacobianProduct(output.output, sigma_b2_bar_vec, wrt, wrt_bar);
   }
}

// Made sigma a StateCoefficient (was formerly an mfem::coefficient)
/// Also made this functional see the temperature field
ACLossFunctional::ACLossFunctional(
    std::map<std::string, FiniteElementState> &fields,
    StateCoefficient &sigma,
    const nlohmann::json &options)
 : output(fields.at("peak_flux").space(), fields), volume(fields, options)
{
   // Making the integrator see the temperature field
   const auto &temp_field_iter =
       fields.find("temperature");  // find where temperature field is
   mfem::GridFunction *temperature_field =
       nullptr;  // default temperature field to null pointer
   if (temp_field_iter != fields.end())
   {
      // If temperature field exists, turn it into a grid function
      auto &temp_field = temp_field_iter->second;
      temperature_field = &temp_field.gridFunc();
   }

   // Assign the integrator used to compute the AC losses
   if (options.contains("attributes"))
   {
      auto attributes = options["attributes"].get<std::vector<int>>();
      output.addOutputDomainIntegrator(
          new ACLossFunctionalIntegrator(sigma, temperature_field), attributes);
   }
   else
   {
      output.addOutputDomainIntegrator(
          new ACLossFunctionalIntegrator(sigma, temperature_field));
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

double jacobianVectorProduct(CoreLossFunctional &output,
                             const mfem::Vector &wrt_dot,
                             const std::string &wrt)
{
   return jacobianVectorProduct(output.output, wrt_dot, wrt);
}

double vectorJacobianProduct(CoreLossFunctional &output,
                             const mfem::Vector &out_bar,
                             const std::string &wrt)
{
   return vectorJacobianProduct(output.output, out_bar, wrt);
}

void vectorJacobianProduct(CoreLossFunctional &output,
                           const mfem::Vector &out_bar,
                           const std::string &wrt,
                           mfem::Vector &wrt_bar)
{
   vectorJacobianProduct(output.output, out_bar, wrt, wrt_bar);
}

CoreLossFunctional::CoreLossFunctional(
    std::map<std::string, FiniteElementState> &fields,
    const nlohmann::json &components,
    const nlohmann::json &materials,
    const nlohmann::json &options)
 : output(fields.at("state").space(), fields),
   rho(constructMaterialCoefficient("rho", components, materials)),
   k_s(constructMaterialCoefficient("ks", components, materials)),
   alpha(constructMaterialCoefficient("alpha", components, materials)),
   beta(constructMaterialCoefficient("beta", components, materials)),
   CAL2_kh(std::make_unique<CAL2khCoefficient>(components, materials)),
   CAL2_ke(std::make_unique<CAL2keCoefficient>(components, materials))
{
   // Making the integrator see the peak flux field
   const auto &peak_flux_iter =
       fields.find("peak_flux");  // find where peak flux field is
   mfem::GridFunction *peak_flux =
       nullptr;  // default peak flux field to null pointer
   if (peak_flux_iter != fields.end())
   {
      // If peak flux field exists, turn it into a grid function
      /// TODO: Ultimately handle the case where there is no peak flux field
      auto &flux_field = peak_flux_iter->second;
      peak_flux = &flux_field.gridFunc();
   }

   // Making the integrator see the temperature field
   const auto &temp_field_iter =
       fields.find("temperature");  // find where temperature field is
   mfem::GridFunction *temperature_field =
       nullptr;  // default temperature field to null pointer
   if (temp_field_iter != fields.end())
   {
      // If temperature field exists, turn it into a grid function
      auto &temp_field = temp_field_iter->second;
      temperature_field = &temp_field.gridFunc();
   }

   if (options.contains("attributes"))
   {
      if (options.contains("UseCAL2forCoreLoss") &&
          options["UseCAL2forCoreLoss"].get<bool>())
      {
         auto attributes = options["attributes"].get<std::vector<int>>();
         output.addOutputDomainIntegrator(
             new CAL2CoreLossIntegrator(
                 *rho, *CAL2_kh, *CAL2_ke, *peak_flux, *temperature_field),
             attributes);
         std::cout << "CoreLossFunctional using CAL2\n";
      }
      else
      {
         auto attributes = options["attributes"].get<std::vector<int>>();
         output.addOutputDomainIntegrator(
             new SteinmetzLossIntegrator(*rho, *k_s, *alpha, *beta, "stator"),
             attributes);
         std::cout << "CoreLossFunctional using Steinmetz\n";
      }
   }
   else
   {
      if (options.contains("UseCAL2forCoreLoss") &&
          options["UseCAL2forCoreLoss"].get<bool>())
      {
         output.addOutputDomainIntegrator(new CAL2CoreLossIntegrator(
             *rho, *CAL2_kh, *CAL2_ke, *peak_flux, *temperature_field));
         std::cout << "CoreLossFunctional using CAL2\n";
      }
      else
      {
         output.addOutputDomainIntegrator(
             new SteinmetzLossIntegrator(*rho, *k_s, *alpha, *beta));
         std::cout << "CoreLossFunctional using Steinmetz\n";
      }
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

   std::cout << "calcOutput in EMHeatSourceOutput\n";

   out_vec = 0.0;
   addLoad(output.lf, out_vec);

   std::cout << "Load has been added to out_vec in calcOutput in "
                "EMHeatSourceOutput\n";
}

/// TODO: Ensure implementation is complete and correct
// Made sigma a StateCoefficient (was formerly an mfem::Coefficient)
EMHeatSourceOutput::EMHeatSourceOutput(
    std::map<std::string, FiniteElementState> &fields,
    mfem::Coefficient &rho,
    StateCoefficient &sigma,
    const nlohmann::json &components,
    const nlohmann::json &materials,
    const nlohmann::json &options)
 : lf(fields.at("temperature").space(), fields),
   rho(constructMaterialCoefficient("rho", components, materials)),
   k_s(constructMaterialCoefficient("ks", components, materials)),
   alpha(constructMaterialCoefficient("alpha", components, materials)),
   beta(constructMaterialCoefficient("beta", components, materials)),
   CAL2_kh(std::make_unique<CAL2khCoefficient>(components, materials)),
   CAL2_ke(std::make_unique<CAL2keCoefficient>(components, materials))
{
   // Making the integrator see the peak flux field
   const auto &peak_flux_iter =
       fields.find("peak_flux");  // find where peak flux field is
   mfem::GridFunction *peak_flux =
       nullptr;  // default peak flux field to null pointer
   if (peak_flux_iter != fields.end())
   {
      // If peak flux field exists, turn it into a grid function
      /// TODO: Ultimately handle the case where there is no peak flux field
      auto &flux_field = peak_flux_iter->second;
      peak_flux = &flux_field.gridFunc();
      /// TODO: Remove once done debugging
      // std::cout << "peak_flux seen by EMHeatSourceOutput\n";
      // std::cout << "peak_flux->Size() = " << peak_flux->Size() << "\n";
      // std::cout << "peak_flux->Min() = " << peak_flux->Min() << "\n";
      // std::cout << "peak_flux->Max() = " << peak_flux->Max() << "\n";
      // std::cout << "peak_flux->Sum() = " << peak_flux->Sum() << "\n";
   }

   // Making the integrator see the temperature field
   const auto &temp_field_iter =
       fields.find("temperature");  // find where temperature field is
   mfem::GridFunction *temperature_field =
       nullptr;  // default temperature field to null pointer
   if (temp_field_iter != fields.end())
   {
      // If temperature field exists, turn it into a grid function
      auto &temp_field = temp_field_iter->second;
      temperature_field = &temp_field.gridFunc();

      /// TODO: Remove once done debugging
      // std::cout << "Temperature field seen by EMHeatSourceOutput\n";
      // std::cout << "temperature_field->Size() = " <<
      // temperature_field->Size() << "\n"; std::cout <<
      // "temperature_field->Min() = " << temperature_field->Min() << "\n";
      // std::cout << "temperature_field->Max() = " << temperature_field->Max()
      // << "\n"; std::cout << "temperature_field->Sum() = " <<
      // temperature_field->Sum() << "\n";
   }

   std::vector<int> stator_attrs =
       components["stator"]["attrs"].get<std::vector<int>>();
   if (options.contains("UseCAL2forCoreLoss") &&
       options["UseCAL2forCoreLoss"].get<bool>())
   {
      lf.addDomainIntegrator(
          new CAL2CoreLossDistributionIntegrator(
              rho, *CAL2_kh, *CAL2_ke, *peak_flux, temperature_field),
          stator_attrs);
      std::cout << "(options.contains(\"UseCAL2forCoreLoss\") && "
                   "options[\"UseCAL2forCoreLoss\"].get<bool>()) = TRUE\n";
   }
   else
   {
      lf.addDomainIntegrator(new SteinmetzLossDistributionIntegrator(
                                 rho, *k_s, *alpha, *beta, "stator"),
                             stator_attrs);
      std::cout << "False, using Steinmetz\n";
   }

   std::vector<int> winding_attrs =
       components["windings"]["attrs"].get<std::vector<int>>();
   lf.addDomainIntegrator(
       new DCLossFunctionalDistributionIntegrator(sigma, temperature_field),
       winding_attrs);  // DCLFI WITH a temperature field
   lf.addDomainIntegrator(new ACLossFunctionalDistributionIntegrator(
                              *peak_flux, sigma, temperature_field),
                          winding_attrs);  // ACLFI WITH a temperature field

   // std::cout << "EMHeatSourceOutput::EMHeatSourceOutput has been
   // constructed\n";
}

void setOptions(PMDemagOutput &output, const nlohmann::json &options)
{
   // setOptions(output.lf, options);
   setOptions(output.output, options);
}

void setInputs(PMDemagOutput &output, const MachInputs &inputs)
{
   // setInputs(output.lf, inputs);

   output.inputs = inputs;
   output.inputs["state"] = inputs.at("peak_flux");
   // output.inputs["state"] = inputs.at("pm_demag_field"); // causes
   // temperature to be 1 exclusively

   setInputs(output.output, inputs);
}

double calcOutput(PMDemagOutput &output, const MachInputs &inputs)
{
   setInputs(output, inputs);

   // mfem::Vector flux_state;
   // setVectorFromInputs(inputs, "peak_flux", flux_state, false, true);
   // ///TODO: Remove once done debugging
   // std::cout << "flux_state.Size() = " << flux_state.Size() << "\n";
   // std::cout << "flux_state.Min() = " << flux_state.Min() << "\n";
   // std::cout << "flux_state.Max() = " << flux_state.Max() << "\n";
   // std::cout << "flux_state.Sum() = " << flux_state.Sum() << "\n";
   // std::cout << "flux_state=np.array([";
   // for (int j = 0; j < flux_state.Size(); j++) {std::cout <<
   // flux_state.Elem(j) << ", ";} std::cout << "])\n";

   // mfem::Vector temperature_vector;
   // setVectorFromInputs(inputs, "temperature", temperature_vector, false,
   // true);
   // ///TODO: Remove once done debugging
   // std::cout << "temperature_vector.Size() = " << temperature_vector.Size()
   // << "\n"; std::cout << "temperature_vector.Min() = " <<
   // temperature_vector.Min() << "\n"; std::cout << "temperature_vector.Max() =
   // " << temperature_vector.Max() << "\n"; std::cout <<
   // "temperature_vector.Sum() = " << temperature_vector.Sum() << "\n";

   return calcOutput(output.output, output.inputs);
}

/// TODO: Implement this method for the AssembleElementVector (or distribution
/// case) for demag rather than singular value
// void calcOutput(PMDemagOutput &output,
//                 const MachInputs &inputs,
//                 mfem::Vector &out_vec)
// {
//    setInputs(output, inputs);

//    out_vec = 0.0;
//    addLoad(output.lf, out_vec);
// }

PMDemagOutput::PMDemagOutput(std::map<std::string, FiniteElementState> &fields,
                             const nlohmann::json &components,
                             const nlohmann::json &materials,
                             const nlohmann::json &options)
 : output(fields.at("peak_flux").space(), fields),
   PMDemagConstraint(
       std::make_unique<PMDemagConstraintCoefficient>(components, materials))
{
   // /*
   // Making the integrator see the temperature field
   const auto &temp_field_iter =
       fields.find("temperature");  // find where temperature field is
   mfem::GridFunction *temperature_field =
       nullptr;  // default temperature field to null pointer
   if (temp_field_iter != fields.end())
   {
      // If temperature field exists, turn it into a grid function
      auto &temp_field = temp_field_iter->second;
      temperature_field = &temp_field.gridFunc();

      // std::cout << "temperature_field.Size() = " << temperature_field->Size()
      // << "\n"; std::cout << "temperature_field.Min() = " <<
      // temperature_field->Min() << "\n"; std::cout << "temperature_field.Max()
      // = " << temperature_field->Max() << "\n"; std::cout <<
      // "temperature_field.Sum() = " << temperature_field->Sum() << "\n";

      // std::cout << "PMDemagOutput, electromag_outputs.cpp, temperature field
      // seen\n";
   }

   // Assign the integrator used to compute the singular value for the PMDM
   // constraint coefficient
   if (options.contains("attributes"))
   {
      auto attributes = options["attributes"].get<std::vector<int>>();
      output.addOutputDomainIntegrator(
          new PMDemagIntegrator(*PMDemagConstraint, temperature_field),
          attributes);
   }
   else
   {
      output.addOutputDomainIntegrator(
          new PMDemagIntegrator(*PMDemagConstraint, temperature_field));
   }
   // */
}

}  // namespace mach
