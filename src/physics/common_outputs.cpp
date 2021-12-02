#include <string>
#include <unordered_map>

#include "mfem.hpp"

#include "mfem_common_integ.hpp"
#include "common_outputs.hpp"

namespace mach
{
StateAverageFunctional::StateAverageFunctional(
    mfem::ParFiniteElementSpace &fes,
    std::unordered_map<std::string, mfem::ParGridFunction> &fields)
 : StateAverageFunctional(fes, fields, {})
{ }

StateAverageFunctional::StateAverageFunctional(
    mfem::ParFiniteElementSpace &fes,
    std::unordered_map<std::string, mfem::ParGridFunction> &fields,
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

IEAggregateFunctional::IEAggregateFunctional(
    mfem::ParFiniteElementSpace &fes,
    std::unordered_map<std::string, mfem::ParGridFunction> &fields,
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

IECurlMagnitudeAggregateFunctional::IECurlMagnitudeAggregateFunctional(
    mfem::ParFiniteElementSpace &fes,
    std::unordered_map<std::string, mfem::ParGridFunction> &fields,
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
