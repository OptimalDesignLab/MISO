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
class VolumeFunctional final
{
public:
   friend inline int getSize(const VolumeFunctional &output)
   {
      return getSize(output.output);
   }

   friend void setOptions(VolumeFunctional &output,
                          const nlohmann::json &options)
   {
      setOptions(output.output, options);
   }

   friend void setInputs(VolumeFunctional &output, const MachInputs &inputs)
   {
      setInputs(output.output, inputs);
   }

   friend double calcOutput(VolumeFunctional &output, const MachInputs &inputs)
   {
      return calcOutput(output.output, inputs);
   }

   friend double jacobianVectorProduct(VolumeFunctional &output,
                                       const mfem::Vector &wrt_dot,
                                       const std::string &wrt)
   {
      return jacobianVectorProduct(output.output, wrt_dot, wrt);
   }

   friend void vectorJacobianProduct(VolumeFunctional &output,
                                     const mfem::Vector &out_bar,
                                     const std::string &wrt,
                                     mfem::Vector &wrt_bar)
   {
      vectorJacobianProduct(output.output, out_bar, wrt, wrt_bar);
   }

   VolumeFunctional(std::map<std::string, FiniteElementState> &fields,
                    const nlohmann::json &options);

private:
   FunctionalOutput output;
};

class MassFunctional final
{
public:
   friend inline int getSize(const MassFunctional &output)
   {
      return getSize(output.output);
   }

   friend void setOptions(MassFunctional &output, const nlohmann::json &options)
   {
      setOptions(output.output, options);
   }

   friend void setInputs(MassFunctional &output, const MachInputs &inputs)
   {
      setInputs(output.output, inputs);
   }

   friend double calcOutput(MassFunctional &output, const MachInputs &inputs)
   {
      return calcOutput(output.output, inputs);
   }

   MassFunctional(std::map<std::string, FiniteElementState> &fields,
                  const nlohmann::json &components,
                  const nlohmann::json &materials,
                  const nlohmann::json &options);

private:
   FunctionalOutput output;
   /// Density
   std::unique_ptr<mfem::Coefficient> rho;
};

class StateAverageFunctional
{
public:
   friend inline int getSize(const StateAverageFunctional &output)
   {
      return getSize(output.state_integ);
   }

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

   StateAverageFunctional(mfem::ParFiniteElementSpace &fes,
                          std::map<std::string, FiniteElementState> &fields);

   StateAverageFunctional(mfem::ParFiniteElementSpace &fes,
                          std::map<std::string, FiniteElementState> &fields,
                          const nlohmann::json &options);

private:
   FunctionalOutput state_integ;
   FunctionalOutput volume;
};

class AverageMagnitudeCurlState
{
public:
   friend inline int getSize(const AverageMagnitudeCurlState &output)
   {
      return getSize(output.state_integ);
   }

   friend void setOptions(AverageMagnitudeCurlState &output,
                          const nlohmann::json &options)
   {
      setOptions(output.state_integ, options);
      setOptions(output.volume, options);
   }

   friend void setInputs(AverageMagnitudeCurlState &output,
                         const MachInputs &inputs)
   {
      output.inputs = &inputs;
      setInputs(output.state_integ, inputs);
      setInputs(output.volume, inputs);
   }

   friend double calcOutput(AverageMagnitudeCurlState &output,
                            const MachInputs &inputs)
   {
      double state = calcOutput(output.state_integ, inputs);
      double volume = calcOutput(output.volume, inputs);
      return state / volume;
   }

   friend double jacobianVectorProduct(AverageMagnitudeCurlState &output,
                                       const mfem::Vector &wrt_dot,
                                       const std::string &wrt);

   friend void vectorJacobianProduct(AverageMagnitudeCurlState &output,
                                     const mfem::Vector &out_bar,
                                     const std::string &wrt,
                                     mfem::Vector &wrt_bar);

   AverageMagnitudeCurlState(mfem::ParFiniteElementSpace &fes,
                             std::map<std::string, FiniteElementState> &fields)
    : AverageMagnitudeCurlState(fes, fields, {})
   { }

   AverageMagnitudeCurlState(mfem::ParFiniteElementSpace &fes,
                             std::map<std::string, FiniteElementState> &fields,
                             const nlohmann::json &options);

private:
   FunctionalOutput state_integ;
   FunctionalOutput volume;
   MachInputs const *inputs = nullptr;
   mfem::Vector scratch;
};

class IEAggregateFunctional
{
public:
   friend inline int getSize(const IEAggregateFunctional &output)
   {
      return getSize(output.numerator);
   }

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

   IEAggregateFunctional(mfem::ParFiniteElementSpace &fes,
                         std::map<std::string, FiniteElementState> &fields,
                         const nlohmann::json &options);

private:
   FunctionalOutput numerator;
   FunctionalOutput denominator;
};

class IECurlMagnitudeAggregateFunctional
{
public:
   friend inline int getSize(const IECurlMagnitudeAggregateFunctional &output)
   {
      return getSize(output.numerator);
   }

   friend void setOptions(IECurlMagnitudeAggregateFunctional &output,
                          const nlohmann::json &options)
   {
      setOptions(output.numerator, options);
      setOptions(output.denominator, options);
   }

   friend void setInputs(IECurlMagnitudeAggregateFunctional &output,
                         const MachInputs &inputs)
   {
      output.inputs = &inputs;
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

   friend double jacobianVectorProduct(
       IECurlMagnitudeAggregateFunctional &output,
       const mfem::Vector &wrt_dot,
       const std::string &wrt);

   friend void vectorJacobianProduct(IECurlMagnitudeAggregateFunctional &output,
                                     const mfem::Vector &out_bar,
                                     const std::string &wrt,
                                     mfem::Vector &wrt_bar);

   IECurlMagnitudeAggregateFunctional(
       mfem::ParFiniteElementSpace &fes,
       std::map<std::string, FiniteElementState> &fields,
       const nlohmann::json &options);

private:
   FunctionalOutput numerator;
   FunctionalOutput denominator;
   MachInputs const *inputs = nullptr;
   mfem::Vector scratch;
};

}  // namespace mach

#endif
