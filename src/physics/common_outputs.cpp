#include <string>
#include <unordered_map>

#include "mach_load.hpp"
#include "mfem.hpp"

#include "coefficient.hpp"
#include "mfem_common_integ.hpp"

#include "common_outputs.hpp"

namespace mach
{
double calcOutput(VolumeFunctional &output, const MachInputs &inputs)
{
   setInputs(output, inputs);
   output.scratch.SetSize(output.output.ParFESpace()->GetTrueVSize());
   return output.output.GetEnergy(output.scratch);
}

VolumeFunctional::VolumeFunctional(
    std::map<std::string, FiniteElementState> &fields,
    const nlohmann::json &options)
 : FunctionalOutput(fields.at("state").space(), fields)
{
   if (options.contains("attributes"))
   {
      auto attributes = options["attributes"].get<std::vector<int>>();
      addOutputDomainIntegrator(new VolumeIntegrator, attributes);
   }
   else
   {
      addOutputDomainIntegrator(new VolumeIntegrator);
   }
}

double calcOutput(MassFunctional &output, const MachInputs &inputs)
{
   setInputs(output, inputs);
   output.scratch.SetSize(output.output.ParFESpace()->GetTrueVSize());
   return output.output.GetEnergy(output.scratch);
}

MassFunctional::MassFunctional(
    std::map<std::string, FiniteElementState> &fields,
    const nlohmann::json &components,
    const nlohmann::json &materials,
    const nlohmann::json &options)
 : FunctionalOutput(fields.at("state").space(), fields),
   rho(constructMaterialCoefficient("rho", components, materials))
{
   if (options.contains("attributes"))
   {
      auto attributes = options["attributes"].get<std::vector<int>>();
      addOutputDomainIntegrator(new VolumeIntegrator(rho.get()), attributes);
   }
   else
   {
      addOutputDomainIntegrator(new VolumeIntegrator(rho.get()));
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

IEAggregateFunctional::IEAggregateFunctional(
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
