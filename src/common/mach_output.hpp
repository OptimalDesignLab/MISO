#ifndef MACH_OUTPUT
#define MACH_OUTPUT

#include <cmath>
#include <memory>
#include <utility>
#include <vector>
#include <string>
#include <algorithm>

#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "mach_input.hpp"
#include "mach_residual.hpp"
#include "utils.hpp"

/// TODO: add some compile time check that makes sure that the type used to
/// construct MachOutput has calcOutput for scalar or field outputs. Maybe take
/// inspiration from:
/// https://stackoverflow.com/questions/87372/check-if-a-class-has-a-member-function-of-a-given-signature

namespace mach
{
template <typename T>
double calcOutputPartial(T & /*unused*/,
                         const std::string & /*unused*/,
                         const MachInputs & /*unused*/)
{
   throw NotImplementedException("not specialized for concrete output type!\n");
}

template <typename T>
void calcOutputPartial(T & /*unused*/,
                       const std::string & /*unused*/,
                       const MachInputs & /*unused*/,
                       mfem::Vector & /*unused*/)
{
   throw NotImplementedException("not specialized for concrete output type!\n");
}

template <typename T>
void calcOutput(T & /*unused*/,
                const MachInputs & /*unused*/,
                mfem::Vector & /*unused*/)
{
   throw NotImplementedException("not specialized for concrete output type!\n");
}

/// Creates common interface for outputs computable by mach
/// A MachOutput can wrap any type `T` that has the interface of an output.
class MachOutput final
{
public:
   /// Gets the dimension of the output
   /// \param[inout] output - the output whose size is being queried
   /// \return the dimension of the output
   friend int getSize(const MachOutput &output);

   /// Used to set inputs in the underlying output type
   friend void setInputs(MachOutput &output, const MachInputs &inputs);

   /// Used to set options for the underlying output type
   friend void setOptions(MachOutput &output, const nlohmann::json &options);

   /// Compute the scalar output based on the inputs
   friend double calcOutput(MachOutput &output, const MachInputs &inputs);

   /// Compute the scalar output's sensitivity to a scalar
   friend double calcOutputPartial(MachOutput &output,
                                   const std::string &wrt,
                                   const MachInputs &inputs);

   /// Compute the scalar output's sensitivity to a field and store in @a
   /// partial
   friend void calcOutputPartial(MachOutput &output,
                                 const std::string &wrt,
                                 const MachInputs &inputs,
                                 mfem::Vector &partial);

   /// Compute the vector output based on the inputs
   friend void calcOutput(MachOutput &output,
                          const MachInputs &inputs,
                          mfem::Vector &out_vec);

   /// Compute the output's sensitivity to a scalar and contract it with
   /// wrt_dot
   /// \param[inout] output - the output whose sensitivity we want
   /// \param[in] wrt_dot - the "wrt"-sized vector to contract with the
   /// sensitivity
   /// \param[in] wrt - string denoting what variable to take the derivative
   /// with respect to
   /// \return the assembled/contracted sensitivity
   friend double jacobianVectorProduct(MachOutput &output,
                                       const mfem::Vector &wrt_dot,
                                       const std::string &wrt);

   /// Compute the output's sensitivity to a vector and contract it with
   /// wrt_dot
   /// \param[inout] output - the output whose sensitivity we want
   /// \param[in] wrt_dot - the "wrt"-sized vector to contract with the
   /// sensitivity
   /// \param[in] wrt - string denoting what variable to take the derivative
   /// with respect to
   /// \param[inout] out_dot - the assembled/contracted sensitivity is
   /// accumulated into out_dot
   friend void jacobianVectorProduct(MachOutput &output,
                                     const mfem::Vector &wrt_dot,
                                     const std::string &wrt,
                                     mfem::Vector &out_dot);

   /// Compute the output's sensitivity to a scalar and contract it with
   /// out_bar
   /// \param[inout] output - the output whose sensitivity we want
   /// \param[in] out_bar - the output-sized vector to contract with the
   /// sensitivity
   /// \param[in] wrt - string denoting what variable to take the derivative
   /// with respect to
   /// \return the assembled/contracted sensitivity
   friend double vectorJacobianProduct(MachOutput &output,
                                       const mfem::Vector &out_bar,
                                       const std::string &wrt);

   /// Compute the output's sensitivity to a vector and contract it with
   /// out_bar
   /// \param[inout] output - the output whose sensitivity we want
   /// \param[in] out_bar - the output-sized vector to contract with the
   /// sensitivity
   /// \param[in] wrt - string denoting what variable to take the derivative
   /// with respect to
   /// \param[inout] wrt_bar - the assembled/contracted sensitivity is
   /// accumulated into wrt_bar
   friend void vectorJacobianProduct(MachOutput &output,
                                     const mfem::Vector &out_bar,
                                     const std::string &wrt,
                                     mfem::Vector &wrt_bar);

   template <typename T>
   MachOutput(T x) : self_(new model<T>(std::move(x)))
   { }

private:
   class concept_t
   {
   public:
      virtual ~concept_t() = default;
      virtual int getSize_() const = 0;
      virtual void setInputs_(const MachInputs &inputs) = 0;
      virtual void setOptions_(const nlohmann::json &options) = 0;
      virtual double calcOutput_(const MachInputs &inputs) = 0;
      virtual double calcOutputPartial_(const std::string &wrt,
                                        const MachInputs &inputs) = 0;
      virtual void calcOutputPartial_(const std::string &wrt,
                                      const MachInputs &inputs,
                                      mfem::Vector &partial) = 0;
      virtual void calcOutput_(const MachInputs &inputs,
                               mfem::Vector &out_vec) = 0;
      virtual double jacobianVectorProduct_(const mfem::Vector &wrt_dot,
                                            const std::string &wrt) = 0;
      virtual void jacobianVectorProduct_(const mfem::Vector &wrt_dot,
                                          const std::string &wrt,
                                          mfem::Vector &out_dot) = 0;
      virtual double vectorJacobianProduct_(const mfem::Vector &out_bar,
                                            const std::string &wrt) = 0;
      virtual void vectorJacobianProduct_(const mfem::Vector &out_bar,
                                          const std::string &wrt,
                                          mfem::Vector &wrt_bar) = 0;
   };

   template <typename T>
   class model final : public concept_t
   {
   public:
      model(T x) : data_(std::move(x)) { }
      int getSize_() const override { return getSize(data_); }
      void setInputs_(const MachInputs &inputs) override
      {
         setInputs(data_, inputs);
      }
      void setOptions_(const nlohmann::json &options) override
      {
         setOptions(data_, options);
      }
      double calcOutput_(const MachInputs &inputs) override
      {
         return calcOutput(data_, inputs);
      }
      double calcOutputPartial_(const std::string &wrt,
                                const MachInputs &inputs) override
      {
         return calcOutputPartial(data_, wrt, inputs);
      }
      void calcOutputPartial_(const std::string &wrt,
                              const MachInputs &inputs,
                              mfem::Vector &partial) override
      {
         calcOutputPartial(data_, wrt, inputs, partial);
      }
      void calcOutput_(const MachInputs &inputs, mfem::Vector &out_vec) override
      {
         calcOutput(data_, inputs, out_vec);
      }
      double jacobianVectorProduct_(const mfem::Vector &wrt_dot,
                                    const std::string &wrt) override
      {
         return jacobianVectorProduct(data_, wrt_dot, wrt);
      }
      void jacobianVectorProduct_(const mfem::Vector &wrt_dot,
                                  const std::string &wrt,
                                  mfem::Vector &out_dot) override
      {
         jacobianVectorProduct(data_, wrt_dot, wrt, out_dot);
      }
      double vectorJacobianProduct_(const mfem::Vector &out_bar,
                                    const std::string &wrt) override
      {
         return vectorJacobianProduct(data_, out_bar, wrt);
      }
      void vectorJacobianProduct_(const mfem::Vector &out_bar,
                                  const std::string &wrt,
                                  mfem::Vector &wrt_bar) override
      {
         vectorJacobianProduct(data_, out_bar, wrt, wrt_bar);
      }

      T data_;
   };

   std::unique_ptr<concept_t> self_;
};

inline int getSize(const MachOutput &output)
{
   return output.self_->getSize_();
}

inline void setInputs(MachOutput &output, const MachInputs &inputs)
{
   output.self_->setInputs_(inputs);
}

inline void setOptions(MachOutput &output, const nlohmann::json &options)
{
   output.self_->setOptions_(options);
}

inline double calcOutput(MachOutput &output, const MachInputs &inputs)
{
   return output.self_->calcOutput_(inputs);
}

inline double calcOutputPartial(MachOutput &output,
                                const std::string &wrt,
                                const MachInputs &inputs)
{
   return output.self_->calcOutputPartial_(wrt, inputs);
}

inline void calcOutputPartial(MachOutput &output,
                              const std::string &wrt,
                              const MachInputs &inputs,
                              mfem::Vector &partial)
{
   output.self_->calcOutputPartial_(wrt, inputs, partial);
}

inline void calcOutput(MachOutput &output,
                       const MachInputs &inputs,
                       mfem::Vector &out_vec)
{
   output.self_->calcOutput_(inputs, out_vec);
}

inline double jacobianVectorProduct(MachOutput &output,
                                    const mfem::Vector &wrt_dot,
                                    const std::string &wrt)
{
   return output.self_->jacobianVectorProduct_(wrt_dot, wrt);
}

inline void jacobianVectorProduct(MachOutput &output,
                                  const mfem::Vector &wrt_dot,
                                  const std::string &wrt,
                                  mfem::Vector &out_dot)
{
   output.self_->jacobianVectorProduct_(wrt_dot, wrt, out_dot);
}

inline double vectorJacobianProduct(MachOutput &output,
                                    const mfem::Vector &out_bar,
                                    const std::string &wrt)
{
   return output.self_->vectorJacobianProduct_(out_bar, wrt);
}

inline void vectorJacobianProduct(MachOutput &output,
                                  const mfem::Vector &out_bar,
                                  const std::string &wrt,
                                  mfem::Vector &wrt_bar)
{
   output.self_->vectorJacobianProduct_(out_bar, wrt, wrt_bar);
}

/// Wrapper for residuals to access its calcEntropy function as a MachOutput
template <typename T>
class EntropyOutput final
{
public:
   friend int getSize(const EntropyOutput &output)
   {
      return getSize(output.res);
   }
   friend void setInputs(EntropyOutput &output, const MachInputs &inputs) { }
   friend void setOptions(EntropyOutput &output, const nlohmann::json &options)
   { }
   friend double calcOutput(EntropyOutput &output, const MachInputs &inputs)
   {
      return calcEntropy(output.res, inputs);
   }
   friend double calcOutputPartial(EntropyOutput &output,
                                   const std::string &wrt,
                                   const MachInputs &inputs)
   {
      return 0.0;
   }
   friend void calcOutputPartial(EntropyOutput &output,
                                 const std::string &wrt,
                                 const MachInputs &inputs,
                                 mfem::Vector &partial)
   { }

   EntropyOutput(T &res_) : res(res_) { }

private:
   T &res;
};

}  // namespace mach

#endif
