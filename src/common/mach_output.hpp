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

namespace mach
{
template <typename T>
void setInputs(T & /*unused*/, const MachInputs & /*unused*/)
{ }

template <typename T>
void setOptions(T & /*unused*/, const nlohmann::json & /*unused*/)
{ }

template <typename T>
double calcOutputPartial(T & /*unused*/,
                         const std::string & /*unused*/,
                         const MachInputs & /*unused*/)
{
   return NAN;
}

template <typename T>
void calcOutputPartial(T & /*unused*/,
                       const std::string & /*unused*/,
                       const MachInputs & /*unused*/,
                       mfem::HypreParVector & /*unused*/)
{ }

template <typename T>
void calcOutput(T & /*unused*/,
                           const MachInputs & /*unused*/,
                           mfem::Vector & /*unused*/)
{ }
template <typename T>
double vectorJacobianProduct(T & /*unused*/,
                                    const std::string & /*unused*/,
                                    const MachInputs & /*unused*/,
                                    const mfem::Vector & /*unused*/)
{
   return NAN;
}

template <typename T>
void vectorJacobianProduct(T &load,
                                    const std::string & /*unused*/,
                                    const MachInputs & /*unused*/,
                                    const mfem::Vector & /*unused*/,
                                    mfem::Vector & /*unused*/)
{ }

/// Creates common interface for outputs computable by mach
/// A MachOutput can wrap any type `T` that has the interface of an output.
class MachOutput final
{
public:
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

   /// Compute the scalar output's sensitivity to a field and store in @a partial
   friend void calcOutputPartial(MachOutput &output,
                                 const std::string &wrt,
                                 const MachInputs &inputs,
                                 mfem::HypreParVector &partial);

   /// Compute the vector output based on the inputs
   friend void calcOutput(MachOutput &output,
                            const MachInputs &inputs,
                            mfem::Vector &out_vec);
   
   /// Assemble the output vector's sensitivity to a scalar and contract it
   /// with out_bar
   friend double vectorJacobianProduct(MachOutput &load,
                                       std::string wrt,
                                       const MachInputs &inputs,
                                       const mfem::Vector &res_bar);

   /// Assemble the output vector's sensitivity to a field and contract it with
   /// out_bar
   friend void vectorJacobianProduct(MachOutput &load,
                                     std::string wrt,
                                     const MachInputs &inputs,
                                     const mfem::Vector &res_bar,
                                     mfem::Vector &wrt_bar);

   template <typename T>
   MachOutput(T x) : self_(new model<T>(std::move(x)))
   { }

private:
   class concept_t
   {
   public:
      virtual ~concept_t() = default;
      virtual void setInputs_(const MachInputs &inputs) = 0;
      virtual void setOptions_(const nlohmann::json &options) = 0;
      virtual double calcOutput_(const MachInputs &inputs) = 0;
      virtual double calcOutputPartial_(const std::string &wrt,
                                        const MachInputs &inputs) = 0;
      virtual void calcOutputPartial_(const std::string &wrt,
                                      const MachInputs &inputs,
                                      mfem::HypreParVector &partial) = 0;
   };

   template <typename T>
   class model final : public concept_t
   {
   public:
      model(T x) : data_(std::move(x)) { }
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
                              mfem::HypreParVector &partial) override
      {
         calcOutputPartial(data_, wrt, inputs, partial);
      }

      T data_;
   };

   std::unique_ptr<concept_t> self_;
};

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
                              mfem::HypreParVector &partial)
{
   output.self_->calcOutputPartial_(wrt, inputs, partial);
}

}  // namespace mach

#endif
