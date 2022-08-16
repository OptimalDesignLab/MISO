#ifndef MACH_ELECTROMAG_OUTPUT
#define MACH_ELECTROMAG_OUTPUT

#include <unordered_set>
#include <vector>

#include "mach_linearform.hpp"
#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "common_outputs.hpp"
#include "electromag_integ.hpp"
#include "functional_output.hpp"
#include "mach_input.hpp"
#include "mfem_common_integ.hpp"

namespace mach
{
class ForceFunctional final
{
public:
   friend inline int getSize(const ForceFunctional &output)
   {
      return getSize(output.output);
   }

   friend inline void setInputs(ForceFunctional &output,
                                const MachInputs &inputs)
   {
      setInputs(output.output, inputs);
   }

   friend void setOptions(ForceFunctional &output,
                          const nlohmann::json &options);

   friend inline double calcOutput(ForceFunctional &output,
                                   const MachInputs &inputs)
   {
      return calcOutput(output.output, inputs);
   }

   friend inline double calcOutputPartial(ForceFunctional &output,
                                          const std::string &wrt,
                                          const MachInputs &inputs)
   {
      return calcOutputPartial(output.output, wrt, inputs);
   }

   friend inline void calcOutputPartial(ForceFunctional &output,
                                        const std::string &wrt,
                                        const MachInputs &inputs,
                                        mfem::Vector &partial)
   {
      calcOutputPartial(output.output, wrt, inputs, partial);
   }

   friend inline double jacobianVectorProduct(ForceFunctional &output,
                                              const mfem::Vector &wrt_dot,
                                              const std::string &wrt)
   {
      return jacobianVectorProduct(output.output, wrt_dot, wrt);
   }

   friend inline void vectorJacobianProduct(ForceFunctional &output,
                                            const mfem::Vector &out_bar,
                                            const std::string &wrt,
                                            mfem::Vector &wrt_bar)
   {
      vectorJacobianProduct(output.output, out_bar, wrt, wrt_bar);
   }

   ForceFunctional(mfem::ParFiniteElementSpace &fes,
                   std::map<std::string, FiniteElementState> &fields,
                   const nlohmann::json &options,
                   mach::StateCoefficient &nu)
    : output(fes, fields), fields(fields)
   {
      setOptions(*this, options);

      auto &&attrs = options["attributes"].get<std::unordered_set<int>>();
      output.addOutputDomainIntegrator(
          new ForceIntegrator3(nu, fields.at("vforce").gridFunc(), attrs));
         //  new ForceIntegrator(nu, fields.at("vforce").gridFunc(), attrs));
   }

private:
   FunctionalOutput output;
   std::map<std::string, FiniteElementState> &fields;
};

class TorqueFunctional final
{
public:
   friend inline int getSize(const TorqueFunctional &output)
   {
      return getSize(output.output);
   }

   friend inline void setInputs(TorqueFunctional &output,
                                const MachInputs &inputs)
   {
      setInputs(output.output, inputs);
   }

   friend void setOptions(TorqueFunctional &output,
                          const nlohmann::json &options);

   friend inline double calcOutput(TorqueFunctional &output,
                                   const MachInputs &inputs)
   {
      return calcOutput(output.output, inputs);
   }

   friend inline double calcOutputPartial(TorqueFunctional &output,
                                          const std::string &wrt,
                                          const MachInputs &inputs)
   {
      return calcOutputPartial(output.output, wrt, inputs);
   }

   friend inline void calcOutputPartial(TorqueFunctional &output,
                                        const std::string &wrt,
                                        const MachInputs &inputs,
                                        mfem::Vector &partial)
   {
      calcOutputPartial(output.output, wrt, inputs, partial);
   }

   friend inline double jacobianVectorProduct(TorqueFunctional &output,
                                              const mfem::Vector &wrt_dot,
                                              const std::string &wrt)
   {
      return jacobianVectorProduct(output.output, wrt_dot, wrt);
   }

   friend inline void vectorJacobianProduct(TorqueFunctional &output,
                                            const mfem::Vector &out_bar,
                                            const std::string &wrt,
                                            mfem::Vector &wrt_bar)
   {
      vectorJacobianProduct(output.output, out_bar, wrt, wrt_bar);
   }

   TorqueFunctional(mfem::ParFiniteElementSpace &fes,
                    std::map<std::string, FiniteElementState> &fields,
                    const nlohmann::json &options,
                    mach::StateCoefficient &nu)
    : output(fes, fields), fields(fields)
   {
      setOptions(*this, options);

      auto &&attrs = options["attributes"].get<std::unordered_set<int>>();
      if (options.contains("air_attributes"))
      {
         auto &&air_attrs = options["air_attributes"].get<std::vector<int>>();
         output.addOutputDomainIntegrator(
             new ForceIntegrator3(nu, fields.at("vtorque").gridFunc(), attrs),
             //  new ForceIntegrator(nu, fields.at("vtorque").gridFunc(),
             //  attrs),
             air_attrs);
      }
      else
      {
         output.addOutputDomainIntegrator(
             new ForceIntegrator3(nu, fields.at("vtorque").gridFunc(), attrs));
         //  new ForceIntegrator(nu, fields.at("vtorque").gridFunc(), attrs));
      }
   }

private:
   FunctionalOutput output;
   std::map<std::string, FiniteElementState> &fields;
};

class DCLossFunctional final : private FunctionalOutput
{
public:
   friend inline int getSize(const DCLossFunctional &output)
   {
      const auto &fun_output = dynamic_cast<const FunctionalOutput &>(output);
      return getSize(fun_output);
   }

   friend void setOptions(DCLossFunctional &output,
                          const nlohmann::json &options)
   {
      auto &fun_output = dynamic_cast<FunctionalOutput &>(output);
      setOptions(fun_output, options);
   }

   friend void setInputs(DCLossFunctional &output, const MachInputs &inputs)
   {
      setValueFromInputs(inputs, "wire_length", output.wire_length);
      setValueFromInputs(inputs, "rms_current", output.rms_current);
      setValueFromInputs(inputs, "strand_radius", output.strand_radius);
      setValueFromInputs(inputs, "strands_in_hand", output.strands_in_hand);

      auto &fun_output = dynamic_cast<FunctionalOutput &>(output);
      setInputs(fun_output, inputs);
   }

   friend double calcOutput(DCLossFunctional &output, const MachInputs &inputs);

   DCLossFunctional(std::map<std::string, FiniteElementState> &fields,
                    mfem::Coefficient &sigma,
                    const nlohmann::json &options);

private:
   VolumeFunctional volume;

   double wire_length = 1.0;
   double rms_current = 1.0;
   double strand_radius = 1.0;
   double strands_in_hand = 1.0;
};

class ACLossFunctional final
{
public:
   friend inline int getSize(const ACLossFunctional &output)
   {
      return getSize(output.output);
   }

   friend void setOptions(ACLossFunctional &output,
                          const nlohmann::json &options);

   friend void setInputs(ACLossFunctional &output, const MachInputs &inputs);

   friend double calcOutput(ACLossFunctional &output, const MachInputs &inputs);

   friend double calcOutputPartial(ACLossFunctional &output,
                                   const std::string &wrt,
                                   const MachInputs &inputs);

   friend void calcOutputPartial(ACLossFunctional &output,
                                 const std::string &wrt,
                                 const MachInputs &inputs,
                                 mfem::Vector &partial);

   ACLossFunctional(std::map<std::string, FiniteElementState> &fields,
                    mfem::Coefficient &sigma,
                    const nlohmann::json &options);

private:
   FunctionalOutput output;
   VolumeFunctional volume;

   std::map<std::string, FiniteElementState> &fields;

   double freq = 1.0;
   double radius = 1.0;
   double stack_length = 1.0;
   double strands_in_hand = 1.0;
   double num_turns = 1.0;
   double num_slots = 1.0;
};

class CoreLossFunctional final
{
public:
   friend inline int getSize(const CoreLossFunctional &output)
   {
      return getSize(output.output);
   }

   friend void setOptions(CoreLossFunctional &output,
                          const nlohmann::json &options);

   friend void setInputs(CoreLossFunctional &output, const MachInputs &inputs);

   friend double calcOutput(CoreLossFunctional &output,
                            const MachInputs &inputs);

   friend double calcOutputPartial(CoreLossFunctional &output,
                                   const std::string &wrt,
                                   const MachInputs &inputs);

   friend void calcOutputPartial(CoreLossFunctional &output,
                                 const std::string &wrt,
                                 const MachInputs &inputs,
                                 mfem::Vector &partial);

   CoreLossFunctional(std::map<std::string, FiniteElementState> &fields,
                      const nlohmann::json &components,
                      const nlohmann::json &materials,
                      const nlohmann::json &options);

private:
   FunctionalOutput output;
   /// Density
   std::unique_ptr<mfem::Coefficient> rho;
   /// Steinmetz coefficients
   std::unique_ptr<mfem::Coefficient> k_s;
   std::unique_ptr<mfem::Coefficient> alpha;
   std::unique_ptr<mfem::Coefficient> beta;
};

class EMHeatSourceOutput final
{
public:
   friend inline int getSize(const EMHeatSourceOutput &output)
   {
      return getSize(output.lf);
   }

   friend void setOptions(EMHeatSourceOutput &output,
                          const nlohmann::json &options);

   friend void setInputs(EMHeatSourceOutput &output, const MachInputs &inputs);

   friend double calcOutput(EMHeatSourceOutput &output,
                            const MachInputs &inputs)
   {
      return NAN;
   }

   friend void calcOutput(EMHeatSourceOutput &output,
                          const MachInputs &inputs,
                          mfem::Vector &out_vec);

   EMHeatSourceOutput(std::map<std::string, FiniteElementState> &fields,
                      mfem::Coefficient &rho,
                      mfem::Coefficient &sigma,
                      const nlohmann::json &components,
                      const nlohmann::json &materials,
                      const nlohmann::json &options);

private:
   MachLinearForm lf;
   /// Steinmetz coefficients
   std::unique_ptr<mfem::Coefficient> k_s;
   std::unique_ptr<mfem::Coefficient> alpha;
   std::unique_ptr<mfem::Coefficient> beta;
};

}  // namespace mach

#endif
