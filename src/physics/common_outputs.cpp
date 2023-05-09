#include <string>
#include <unordered_map>

#include "mach_load.hpp"
#include "mfem.hpp"

#include "coefficient.hpp"
#include "mfem_common_integ.hpp"

#include "common_outputs.hpp"

// Needed for IEAgg when considering demagnetization
#include "demag_flux_coefficient.hpp"

namespace mach
{
VolumeFunctional::VolumeFunctional(
    std::map<std::string, FiniteElementState> &fields,
    const nlohmann::json &options)
 : output(fields.at("state").space(), fields)
{
   if (options.contains("attributes"))
   {
      auto attributes = options["attributes"].get<std::vector<int>>();
      output.addOutputDomainIntegrator(new VolumeIntegrator, attributes);
   }
   else
   {
      output.addOutputDomainIntegrator(new VolumeIntegrator);
   }
}

MassFunctional::MassFunctional(
    std::map<std::string, FiniteElementState> &fields,
    const nlohmann::json &components,
    const nlohmann::json &materials,
    const nlohmann::json &options)
 : output(fields.at("state").space(), fields),
   rho(constructMaterialCoefficient("rho", components, materials))
{
   if (options.contains("attributes"))
   {
      auto attributes = options["attributes"].get<std::vector<int>>();
      output.addOutputDomainIntegrator(new VolumeIntegrator(rho.get()),
                                       attributes);
   }
   else
   {
      output.addOutputDomainIntegrator(new VolumeIntegrator(rho.get()));
   }
}

StateAverageFunctional::StateAverageFunctional(
    mfem::ParFiniteElementSpace &fes,
    std::map<std::string, FiniteElementState> &fields)
 : StateAverageFunctional(fes, fields, {})
{ }

StateAverageFunctional::StateAverageFunctional(
    mfem::ParFiniteElementSpace &fes,
    std::map<std::string, FiniteElementState> &fields,
    const nlohmann::json &options)
 : state_integ(fes, fields), volume(fes, fields)
{
   if (options.contains("attributes"))
   {
      auto attributes = options["attributes"].get<std::vector<int>>();
      state_integ.addOutputDomainIntegrator(new StateIntegrator, attributes);
      volume.addOutputDomainIntegrator(new VolumeIntegrator, attributes);
   }
   else
   {
      state_integ.addOutputDomainIntegrator(new StateIntegrator);
      volume.addOutputDomainIntegrator(new VolumeIntegrator);
   }
}

double jacobianVectorProduct(AverageMagnitudeCurlState &output,
                             const mfem::Vector &wrt_dot,
                             const std::string &wrt)
{
   const MachInputs &inputs = *output.inputs;
   double state = calcOutput(output.state_integ, inputs);
   double volume = calcOutput(output.volume, inputs);

   auto out_dot =
       volume * jacobianVectorProduct(output.state_integ, wrt_dot, wrt);
   out_dot -= state * jacobianVectorProduct(output.volume, wrt_dot, wrt);
   out_dot /= pow(volume, 2);
   return out_dot;
}

void vectorJacobianProduct(AverageMagnitudeCurlState &output,
                           const mfem::Vector &out_bar,
                           const std::string &wrt,
                           mfem::Vector &wrt_bar)
{
   const MachInputs &inputs = *output.inputs;
   double state = calcOutput(output.state_integ, inputs);
   double volume = calcOutput(output.volume, inputs);

   output.scratch.SetSize(wrt_bar.Size());

   output.scratch = 0.0;
   vectorJacobianProduct(output.state_integ, out_bar, wrt, output.scratch);
   wrt_bar.Add(1 / volume, output.scratch);

   output.scratch = 0.0;
   vectorJacobianProduct(output.volume, out_bar, wrt, output.scratch);
   wrt_bar.Add(-state / pow(volume, 2), output.scratch);
}

AverageMagnitudeCurlState::AverageMagnitudeCurlState(
    mfem::ParFiniteElementSpace &fes,
    std::map<std::string, FiniteElementState> &fields,
    const nlohmann::json &options)
 : state_integ(fes, fields), volume(fes, fields)
{
   if (options.contains("attributes"))
   {
      auto attributes = options["attributes"].get<std::vector<int>>();
      state_integ.addOutputDomainIntegrator(new MagnitudeCurlStateIntegrator,
                                            attributes);
      volume.addOutputDomainIntegrator(new VolumeIntegrator, attributes);
   }
   else
   {
      state_integ.addOutputDomainIntegrator(new MagnitudeCurlStateIntegrator);
      volume.addOutputDomainIntegrator(new VolumeIntegrator);
   }
}

double jacobianVectorProduct(IEAggregateFunctional &output,
                             const mfem::Vector &wrt_dot,
                             const std::string &wrt)
{
   const MachInputs &inputs = *output.inputs;
   double num = calcOutput(output.numerator, inputs);
   double denom = calcOutput(output.denominator, inputs);

   auto out_dot = denom * jacobianVectorProduct(output.numerator, wrt_dot, wrt);
   out_dot -= num * jacobianVectorProduct(output.denominator, wrt_dot, wrt);
   out_dot /= pow(denom, 2);
   return out_dot;
}

void vectorJacobianProduct(IEAggregateFunctional &output,
                           const mfem::Vector &out_bar,
                           const std::string &wrt,
                           mfem::Vector &wrt_bar)
{
   const MachInputs &inputs = *output.inputs;
   double num = calcOutput(output.numerator, inputs);
   double denom = calcOutput(output.denominator, inputs);

   output.scratch.SetSize(wrt_bar.Size());

   output.scratch = 0.0;
   vectorJacobianProduct(output.numerator, out_bar, wrt, output.scratch);
   wrt_bar.Add(1 / denom, output.scratch);

   output.scratch = 0.0;
   vectorJacobianProduct(output.denominator, out_bar, wrt, output.scratch);
   wrt_bar.Add(-num / pow(denom, 2), output.scratch);
}

IEAggregateFunctional::IEAggregateFunctional(
    mfem::ParFiniteElementSpace &fes,
    std::map<std::string, FiniteElementState> &fields,
    const nlohmann::json &options)
 : numerator(fes, fields, options.value("state", "state")),
   denominator(fes, fields, options.value("state", "state"))
{
   auto rho = options.value("rho", 1.0);
   // auto state_name = options.value("state", "state");

   if (options.contains("attributes"))
   {
      auto attributes = options["attributes"].get<std::vector<int>>();
      numerator.addOutputDomainIntegrator(
          new IEAggregateIntegratorNumerator(rho), attributes);
      denominator.addOutputDomainIntegrator(
          new IEAggregateIntegratorDenominator(rho), attributes);
   }
   else
   {
      numerator.addOutputDomainIntegrator(
          new IEAggregateIntegratorNumerator(rho));
      denominator.addOutputDomainIntegrator(
          new IEAggregateIntegratorDenominator(rho));
   }
}

double jacobianVectorProduct(IECurlMagnitudeAggregateFunctional &output,
                             const mfem::Vector &wrt_dot,
                             const std::string &wrt)
{
   const MachInputs &inputs = *output.inputs;
   double num = calcOutput(output.numerator, inputs);
   double denom = calcOutput(output.denominator, inputs);

   auto out_dot = denom * jacobianVectorProduct(output.numerator, wrt_dot, wrt);
   out_dot -= num * jacobianVectorProduct(output.denominator, wrt_dot, wrt);
   out_dot /= pow(denom, 2);
   return out_dot;
}

void vectorJacobianProduct(IECurlMagnitudeAggregateFunctional &output,
                           const mfem::Vector &out_bar,
                           const std::string &wrt,
                           mfem::Vector &wrt_bar)
{
   const MachInputs &inputs = *output.inputs;
   double num = calcOutput(output.numerator, inputs);
   double denom = calcOutput(output.denominator, inputs);

   output.scratch.SetSize(wrt_bar.Size());

   output.scratch = 0.0;
   vectorJacobianProduct(output.numerator, out_bar, wrt, output.scratch);
   wrt_bar.Add(1 / denom, output.scratch);

   output.scratch = 0.0;
   vectorJacobianProduct(output.denominator, out_bar, wrt, output.scratch);
   wrt_bar.Add(-num / pow(denom, 2), output.scratch);
}

IECurlMagnitudeAggregateFunctional::IECurlMagnitudeAggregateFunctional(
    mfem::ParFiniteElementSpace &fes,
    std::map<std::string, FiniteElementState> &fields,
    const nlohmann::json &options)
 : numerator(fes, fields), denominator(fes, fields)
{
   auto rho = options["rho"].get<double>();

   if (options.contains("attributes"))
   {
      auto attributes = options["attributes"].get<std::vector<int>>();
      numerator.addOutputDomainIntegrator(
          new IECurlMagnitudeAggregateIntegratorNumerator(rho), attributes);
      denominator.addOutputDomainIntegrator(
          new IECurlMagnitudeAggregateIntegratorDenominator(rho), attributes);
   }
   else
   {
      numerator.addOutputDomainIntegrator(
          new IECurlMagnitudeAggregateIntegratorNumerator(rho));
      denominator.addOutputDomainIntegrator(
          new IECurlMagnitudeAggregateIntegratorDenominator(rho));
   }
}

}  // namespace mach
