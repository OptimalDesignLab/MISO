#include <string>
#include <unordered_map>

#include "mfem.hpp"

#include "mfem_common_integ.hpp"
#include "common_outputs.hpp"

namespace mach
{
IEAggregateFunctional::IEAggregateFunctional(
    mfem::ParFiniteElementSpace &fes,
    std::unordered_map<std::string, mfem::ParGridFunction> &fields,
    double rho)
 : numerator(fes, fields), denominator(fes, fields)
{
   numerator.addOutputDomainIntegrator(new IEAggregateIntegratorNumerator(rho));
   denominator.addOutputDomainIntegrator(
       new IEAggregateIntegratorDenominator(rho));
}

IECurlMagnitudeAggregateFunctional::IECurlMagnitudeAggregateFunctional(
    mfem::ParFiniteElementSpace &fes,
    std::unordered_map<std::string, mfem::ParGridFunction> &fields,
    double rho)
 : numerator(fes, fields), denominator(fes, fields)
{
   numerator.addOutputDomainIntegrator(
       new IECurlMagnitudeAggregateIntegratorNumerator(rho));
   denominator.addOutputDomainIntegrator(
       new IECurlMagnitudeAggregateIntegratorDenominator(rho));
}

}  // namespace mach
