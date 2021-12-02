#ifndef MACH_COMMON_OUTPUTS
#define MACH_COMMON_OUTPUTS

#include <string>
#include <unordered_map>

#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "functional_output.hpp"
#include "mach_input.hpp"

namespace mach
{
class StateAverageFunctional
{
public:
   StateAverageFunctional(
       mfem::ParFiniteElementSpace &fes,
       std::unordered_map<std::string, mfem::ParGridFunction> &fields);

   StateAverageFunctional(
       mfem::ParFiniteElementSpace &fes,
       std::unordered_map<std::string, mfem::ParGridFunction> &fields,
       const nlohmann::json &options);

   friend void setOptions(StateAverageFunctional &output,
                          const nlohmann::json &options)
   {
      setOptions(output.state_integ, options);
      setOptions(output.volume, options);
   }

   friend void setInputs(StateAverageFunctional &output,
                         const MachInputs &inputs)
   {
      setInputs(output.state_integ, inputs);
      setInputs(output.volume, inputs);
   }

   friend double calcOutput(StateAverageFunctional &output,
                            const MachInputs &inputs)
   {
      double state = calcOutput(output.state_integ, inputs);
      double volume = calcOutput(output.volume, inputs);
      return state / volume;
   }

private:
   FunctionalOutput state_integ;
   FunctionalOutput volume;
};

class IEAggregateFunctional
{
public:
   IEAggregateFunctional(
       mfem::ParFiniteElementSpace &fes,
       std::unordered_map<std::string, mfem::ParGridFunction> &fields,
       const nlohmann::json &options);

   friend void setOptions(IEAggregateFunctional &output,
                          const nlohmann::json &options)
   {
      setOptions(output.numerator, options);
      setOptions(output.denominator, options);
   }

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
       const nlohmann::json &options);

   friend void setOptions(IECurlMagnitudeAggregateFunctional &output,
                          const nlohmann::json &options)
   {
      setOptions(output.numerator, options);
      setOptions(output.denominator, options);
   }

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
