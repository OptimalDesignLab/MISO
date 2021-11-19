#ifndef MACH_COMMON_OUTPUTS
#define MACH_COMMON_OUTPUTS

#include <string>
#include <unordered_map>

#include "mfem.hpp"

#include "functional_output.hpp"
#include "mach_input.hpp"

namespace mach
{
class IEAggregateFunctional
{
public:
   IEAggregateFunctional(
       mfem::ParFiniteElementSpace &fes,
       std::unordered_map<std::string, mfem::ParGridFunction> &fields,
       double rho);

   friend void setInputs(IEAggregateFunctional &output,
                         const MachInputs &inputs)
   {
      setInputs(output.numerator, inputs);
      setInputs(output.denominator, inputs);
   }

   friend double calcOutput(IEAggregateFunctional &output,
                            const MachInputs &inputs)
   {
      double num = calcOutput(output.numerator, inputs);
      double denom = calcOutput(output.denominator, inputs);
      return num / denom;
   }

private:
   FunctionalOutput numerator;
   FunctionalOutput denominator;
};

class IECurlMagnitudeAggregateFunctional
{
public:
   IECurlMagnitudeAggregateFunctional(
       mfem::ParFiniteElementSpace &fes,
       std::unordered_map<std::string, mfem::ParGridFunction> &fields,
       double rho);

   friend void setInputs(IECurlMagnitudeAggregateFunctional &output,
                         const MachInputs &inputs)
   {
      setInputs(output.numerator, inputs);
      setInputs(output.denominator, inputs);
   }

   friend double calcOutput(IECurlMagnitudeAggregateFunctional &output,
                            const MachInputs &inputs)
   {
      double num = calcOutput(output.numerator, inputs);
      double denom = calcOutput(output.denominator, inputs);
      return num / denom;
   }

private:
   FunctionalOutput numerator;
   FunctionalOutput denominator;
};

}  // namespace mach

#endif
