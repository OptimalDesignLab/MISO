#ifndef MACH_ELECTROMAG_OUTPUT
#define MACH_ELECTROMAG_OUTPUT

#include <memory>
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

// Needed for CoreLossFunctional or PMDemagOutput
#include "cal2_kh_coefficient.hpp"
#include "cal2_ke_coefficient.hpp"
#include "pm_demag_constraint_coeff.hpp"

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

// class ResistivityFunctional final
// {
// public:
//    friend inline int getSize(const ResistivityFunctional &output)
//    {
//       return getSize(output.output);
//    }

//    friend void setOptions(ResistivityFunctional &output,
//                           const nlohmann::json &options)
//    {
//       setOptions(output.output, options);
//    }

//    friend void setInputs(ResistivityFunctional &output, const MachInputs
//    &inputs)
//    {
//       setInputs(output.output, inputs);
//    }
//    friend double calcOutput(ResistivityFunctional &output, const MachInputs
//    &inputs);

//    friend double jacobianVectorProduct(ResistivityFunctional &output,
//                                        const mfem::Vector &wrt_dot,
//                                        const std::string &wrt);

//    friend void jacobianVectorProduct(ResistivityFunctional &output,
//                                      const mfem::Vector &wrt_dot,
//                                      const std::string &wrt,
//                                      mfem::Vector &out_dot);

//    friend double vectorJacobianProduct(ResistivityFunctional &output,
//                                        const mfem::Vector &out_bar,
//                                        const std::string &wrt);

//    friend void vectorJacobianProduct(ResistivityFunctional &output,
//                                      const mfem::Vector &out_bar,
//                                      const std::string &wrt,
//                                      mfem::Vector &wrt_bar);

//    ResistivityFunctional();

// private:
//    FunctionalOutput output;

// };

class DCLossFunctional final
{
public:
   friend inline int getSize(const DCLossFunctional &output) { return 1; }

   friend void setOptions(DCLossFunctional &output,
                          const nlohmann::json &options)
   {
      setOptions(output.resistivity, options);
      setOptions(output.volume, options);
   }

   friend void setInputs(DCLossFunctional &output, const MachInputs &inputs)
   {
      output.inputs = &inputs;
      setValueFromInputs(inputs, "wire_length", output.wire_length);
      setValueFromInputs(inputs, "rms_current", output.rms_current);
      setValueFromInputs(inputs, "strand_radius", output.strand_radius);
      setValueFromInputs(inputs, "strands_in_hand", output.strands_in_hand);

      setInputs(output.resistivity, inputs);
      setInputs(output.volume, inputs);
   }

   friend double calcOutput(DCLossFunctional &output, const MachInputs &inputs);

   friend double jacobianVectorProduct(DCLossFunctional &output,
                                       const mfem::Vector &wrt_dot,
                                       const std::string &wrt);

   // friend void jacobianVectorProduct(DCLossFunctional &output,
   //                                   const mfem::Vector &wrt_dot,
   //                                   const std::string &wrt,
   //                                   mfem::Vector &out_dot);

   friend double vectorJacobianProduct(DCLossFunctional &output,
                                       const mfem::Vector &out_bar,
                                       const std::string &wrt);

   friend void vectorJacobianProduct(DCLossFunctional &output,
                                     const mfem::Vector &out_bar,
                                     const std::string &wrt,
                                     mfem::Vector &wrt_bar);

   DCLossFunctional(std::map<std::string, FiniteElementState> &fields,
                    StateCoefficient &sigma,
                    const nlohmann::json &options);

private:
   FunctionalOutput resistivity;
   VolumeFunctional volume;

   double wire_length = 1.0;
   double rms_current = 1.0;
   double strand_radius = 1.0;
   double strands_in_hand = 1.0;

   MachInputs const *inputs = nullptr;
};

class DCLossDistribution final
{
public:
   friend inline int getSize(const DCLossDistribution &output)
   {
      return getSize(output.output);
   }

   friend inline void setOptions(DCLossDistribution &output,
                                 const nlohmann::json &options)
   {
      setOptions(output.output, options);
      setOptions(output.volume, options);
   }

   friend inline void setInputs(DCLossDistribution &output,
                                const MachInputs &inputs)
   {
      output.inputs = &inputs;
      setInputs(output.output, inputs);
      setInputs(output.volume, inputs);
   }

   friend inline double calcOutput(DCLossDistribution &output,
                                   const MachInputs &inputs)
   {
      return NAN;
   }

   friend void calcOutput(DCLossDistribution &output,
                          const MachInputs &inputs,
                          mfem::Vector &out_vec);

   friend void jacobianVectorProduct(DCLossDistribution &output,
                                     const mfem::Vector &wrt_dot,
                                     const std::string &wrt,
                                     mfem::Vector &out_dot);

   friend double vectorJacobianProduct(DCLossDistribution &output,
                                       const mfem::Vector &out_bar,
                                       const std::string &wrt);

   friend void vectorJacobianProduct(DCLossDistribution &output,
                                     const mfem::Vector &out_bar,
                                     const std::string &wrt,
                                     mfem::Vector &wrt_bar);

   DCLossDistribution(std::map<std::string, FiniteElementState> &fields,
                      StateCoefficient &sigma,
                      const nlohmann::json &options);

private:
   MachLinearForm output;
   VolumeFunctional volume;

   MachInputs const *inputs = nullptr;
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

   friend double jacobianVectorProduct(ACLossFunctional &output,
                                       const mfem::Vector &wrt_dot,
                                       const std::string &wrt);

   friend double vectorJacobianProduct(ACLossFunctional &output,
                                       const mfem::Vector &out_bar,
                                       const std::string &wrt);

   friend void vectorJacobianProduct(ACLossFunctional &output,
                                     const mfem::Vector &out_bar,
                                     const std::string &wrt,
                                     mfem::Vector &wrt_bar);

   // Made sigma a StateCoefficient (was formerly an mfem::coefficient)
   ACLossFunctional(std::map<std::string, FiniteElementState> &fields,
                    StateCoefficient &sigma,
                    const nlohmann::json &options);

private:
   FunctionalOutput output;
   VolumeFunctional volume;

   double freq = 1.0;
   double radius = 1.0;
   double stack_length = 1.0;
   double strands_in_hand = 1.0;
   double num_turns = 1.0;
   double num_slots = 1.0;

   MachInputs inputs;
};

class ACLossDistribution final
{
public:
   friend inline int getSize(const ACLossDistribution &output)
   {
      return getSize(output.output);
   }

   friend inline void setOptions(ACLossDistribution &output,
                                 const nlohmann::json &options)
   {
      setOptions(output.output, options);
      setOptions(output.volume, options);
   }

   friend inline void setInputs(ACLossDistribution &output,
                                const MachInputs &inputs)
   {
      output.inputs = &inputs;
      setInputs(output.output, inputs);
      setInputs(output.volume, inputs);
   }

   friend inline double calcOutput(ACLossDistribution &output,
                                   const MachInputs &inputs)
   {
      return NAN;
   }

   friend void calcOutput(ACLossDistribution &output,
                          const MachInputs &inputs,
                          mfem::Vector &out_vec);

   friend void jacobianVectorProduct(ACLossDistribution &output,
                                     const mfem::Vector &wrt_dot,
                                     const std::string &wrt,
                                     mfem::Vector &out_dot);

   friend double vectorJacobianProduct(ACLossDistribution &output,
                                       const mfem::Vector &out_bar,
                                       const std::string &wrt);

   friend void vectorJacobianProduct(ACLossDistribution &output,
                                     const mfem::Vector &out_bar,
                                     const std::string &wrt,
                                     mfem::Vector &wrt_bar);

   ACLossDistribution(std::map<std::string, FiniteElementState> &fields,
                      StateCoefficient &sigma,
                      const nlohmann::json &options);

private:
   MachLinearForm output;
   VolumeFunctional volume;

   MachInputs const *inputs = nullptr;
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

   friend double jacobianVectorProduct(CoreLossFunctional &output,
                                       const mfem::Vector &wrt_dot,
                                       const std::string &wrt);

   friend double vectorJacobianProduct(CoreLossFunctional &output,
                                       const mfem::Vector &out_bar,
                                       const std::string &wrt);

   friend void vectorJacobianProduct(CoreLossFunctional &output,
                                     const mfem::Vector &out_bar,
                                     const std::string &wrt,
                                     mfem::Vector &wrt_bar);

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

   /// CAL2 Coefficients
   std::unique_ptr<ThreeStateCoefficient> CAL2_kh;
   std::unique_ptr<ThreeStateCoefficient> CAL2_ke;

   MachInputs inputs;
};

class CAL2CoreLossDistribution final
{
public:
   friend inline int getSize(const CAL2CoreLossDistribution &output)
   {
      return getSize(output.output);
   }

   friend inline void setOptions(CAL2CoreLossDistribution &output,
                                 const nlohmann::json &options)
   {
      setOptions(output.output, options);
   }

   friend inline void setInputs(CAL2CoreLossDistribution &output,
                                const MachInputs &inputs)
   {
      output.inputs = &inputs;
      setInputs(output.output, inputs);
   }

   friend inline double calcOutput(CAL2CoreLossDistribution &output,
                                   const MachInputs &inputs)
   {
      return NAN;
   }

   friend void calcOutput(CAL2CoreLossDistribution &output,
                          const MachInputs &inputs,
                          mfem::Vector &out_vec);

   friend void jacobianVectorProduct(CAL2CoreLossDistribution &output,
                                     const mfem::Vector &wrt_dot,
                                     const std::string &wrt,
                                     mfem::Vector &out_dot);

   friend double vectorJacobianProduct(CAL2CoreLossDistribution &output,
                                       const mfem::Vector &out_bar,
                                       const std::string &wrt);

   friend void vectorJacobianProduct(CAL2CoreLossDistribution &output,
                                     const mfem::Vector &out_bar,
                                     const std::string &wrt,
                                     mfem::Vector &wrt_bar);

   CAL2CoreLossDistribution(std::map<std::string, FiniteElementState> &fields,
                            const nlohmann::json &components,
                            const nlohmann::json &materials,
                            const nlohmann::json &options);

private:
   MachLinearForm output;

   /// mass density
   std::unique_ptr<mfem::Coefficient> rho;

   /// CAL2 Coefficients
   std::unique_ptr<ThreeStateCoefficient> CAL2_kh;
   std::unique_ptr<ThreeStateCoefficient> CAL2_ke;

   MachInputs const *inputs = nullptr;
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

   friend void jacobianVectorProduct(EMHeatSourceOutput &output,
                                     const mfem::Vector &wrt_dot,
                                     const std::string &wrt,
                                     mfem::Vector &out_dot);

   friend double vectorJacobianProduct(EMHeatSourceOutput &output,
                                       const mfem::Vector &out_bar,
                                       const std::string &wrt);

   friend void vectorJacobianProduct(EMHeatSourceOutput &output,
                                     const mfem::Vector &out_bar,
                                     const std::string &wrt,
                                     mfem::Vector &wrt_bar);

   EMHeatSourceOutput(std::map<std::string, FiniteElementState> &fields,
                      mfem::Coefficient &rho,
                      StateCoefficient &sigma,
                      const nlohmann::json &components,
                      const nlohmann::json &materials,
                      const nlohmann::json &options);

private:
   MachLinearForm lf;

   /// Density
   std::unique_ptr<mfem::Coefficient> rho;

   /// Steinmetz coefficients
   std::unique_ptr<mfem::Coefficient> k_s;
   std::unique_ptr<mfem::Coefficient> alpha;
   std::unique_ptr<mfem::Coefficient> beta;

   /// CAL2 Coefficients
   std::unique_ptr<ThreeStateCoefficient> CAL2_kh;
   std::unique_ptr<ThreeStateCoefficient> CAL2_ke;
};

// Adding an output for the permanent magnet demagnetization constraint equation
class PMDemagOutput final
{
public:
   friend inline int getSize(const PMDemagOutput &output)
   {
      // return getSize(output.lf);
      return getSize(output.output);
   }

   friend void setOptions(PMDemagOutput &output, const nlohmann::json &options);

   friend void setInputs(PMDemagOutput &output, const MachInputs &inputs);

   friend double calcOutput(PMDemagOutput &output, const MachInputs &inputs);

   /// TODO: Implement this method for the AssembleElementVector (or
   /// distribution case) for demag rather than singular value
   // friend void calcOutput(PMDemagOutput &output,
   //                        const MachInputs &inputs,
   //                        mfem::Vector &out_vec);

   PMDemagOutput(std::map<std::string, FiniteElementState> &fields,
                 const nlohmann::json &components,
                 const nlohmann::json &materials,
                 const nlohmann::json &options);

private:
   MachInputs inputs;
   FunctionalOutput output;

   // MachLinearForm lf;

   std::unique_ptr<TwoStateCoefficient> PMDemagConstraint;
};

}  // namespace mach

#endif
