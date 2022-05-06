#ifndef MACH_ELECTROMAG_OUTPUT
#define MACH_ELECTROMAG_OUTPUT

#include <unordered_set>
#include <vector>

#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "electromag_integ.hpp"
#include "functional_output.hpp"
#include "mach_input.hpp"
#include "mfem_common_integ.hpp"

namespace mach
{
// class BNormSquaredAverageFunctional final
// {
// public:
//    friend void setOptions(BNormSquaredAverageFunctional &output,
//                           const nlohmann::json &options)
//    {
//       setOptions(output.bnormsquared, options);
//       setOptions(output.volume, options);
//    }

//    friend void setInputs(BNormSquaredAverageFunctional &output,
//                          const MachInputs &inputs)
//    {
//       setInputs(output.bnormsquared, inputs);
//       setInputs(output.volume, inputs);
//    }

//    friend double calcOutput(BNormSquaredAverageFunctional &output,
//                             const MachInputs &inputs)
//    {
//       double bnormsquared = calcOutput(output.bnormsquared, inputs);
//       double volume = calcOutput(output.volume, inputs);
//       return bnormsquared / volume;
//    }

//    BNormSquaredAverageFunctional(
//        mfem::ParFiniteElementSpace &fes,
//        std::unordered_map<std::string, mfem::ParGridFunction> &fields,
//        const nlohmann::json &options)
//     : bnormsquared(fes, fields), volume(fes, fields)
//    {
//       if (options.contains("attributes"))
//       {
//          auto attributes = options["attributes"].get<std::vector<int>>();
//          bnormsquared.addOutputDomainIntegrator(new BNormSquaredIntegrator,
//                                                 attributes);
//          volume.addOutputDomainIntegrator(new VolumeIntegrator, attributes);
//       }
//       else
//       {
//          bnormsquared.addOutputDomainIntegrator(new BNormSquaredIntegrator);
//          volume.addOutputDomainIntegrator(new VolumeIntegrator);
//       }
//    }

// private:
//    FunctionalOutput bnormsquared;
//    FunctionalOutput volume;
// };

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
          new ForceIntegrator(nu, fields.at("vforce").gridFunc(), attrs));
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
      output.addOutputDomainIntegrator(
          new ForceIntegrator(nu, fields.at("vtorque").gridFunc(), attrs));
   }

private:
   FunctionalOutput output;
   std::map<std::string, FiniteElementState> &fields;
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
   FunctionalOutput volume;
   // std::map<std::string, FiniteElementState> &fields;

   double freq = 1.0;
   double radius = 1.0;
   double stack_length = 1.0;
   double num_strands = 1.0;
};

// class SteinmetzValue final
// {
// public:
//    friend inline int getSize(const SteinmetzValue &output)
//    {
//       return 1.0;
//    }

// };

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

inline void setOptions(ForceFunctional &output, const nlohmann::json &options)
{
   auto &&attrs = options["attributes"].get<std::unordered_set<int>>();
   auto &&axis = options["axis"].get<std::vector<double>>();
   mfem::VectorConstantCoefficient axis_vector(
       mfem::Vector(&axis[0], axis.size()));

   auto &v = output.fields.at("vforce").gridFunc();
   v = 0.0;
   for (const auto &attr : attrs)
   {
      v.ProjectCoefficient(axis_vector, attr);
   }
}

inline void setOptions(TorqueFunctional &output, const nlohmann::json &options)
{
   auto &&attrs = options["attributes"].get<std::unordered_set<int>>();
   auto &&axis = options["axis"].get<std::vector<double>>();
   auto &&about = options["about"].get<std::vector<double>>();
   mfem::Vector axis_vector(&axis[0], axis.size());
   axis_vector /= axis_vector.Norml2();
   mfem::Vector about_vector(&about[0], axis.size());
   double r_data[3];
   mfem::Vector r(r_data, axis.size());
   mfem::VectorFunctionCoefficient v_vector(
       3,
       [&axis_vector, &about_vector, &r](const mfem::Vector &x, mfem::Vector &v)
       {
          subtract(x, about_vector, r);
          // r /= r.Norml2();
          v(0) = axis_vector(1) * r(2) - axis_vector(2) * r(1);
          v(1) = axis_vector(2) * r(0) - axis_vector(0) * r(2);
          v(2) = axis_vector(0) * r(1) - axis_vector(1) * r(0);
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

}  // namespace mach

#endif
