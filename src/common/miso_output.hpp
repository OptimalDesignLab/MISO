#ifndef MISO_OUTPUT
#define MISO_OUTPUT

#include <memory>
#include <utility>
#include <vector>
#include <string>
#include <algorithm>

#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "miso_input.hpp"

namespace miso
{
/// Creates common interface for outputs computable by miso
/// A MISOOutput can wrap any type `T` that has the interface of an output.
class MISOOutput final
{
public:
   /// Used to set inputs in the underlying output type
   friend void setInputs(MISOOutput &output, const MISOInputs &inputs);

   /// Used to set options for the underlying output type
   friend void setOptions(MISOOutput &output, const nlohmann::json &options);

   /// Compute the scalar output based on the inputs
   friend double calcOutput(MISOOutput &output, const MISOInputs &inputs);

   /// Compute the output's sensitivity to a scalar
   friend double calcOutputPartial(MISOOutput &output,
                                   const std::string &wrt,
                                   const MISOInputs &inputs);

   /// Compute the output's sensitivity to a field and store in @a partial
   friend void calcOutputPartial(MISOOutput &output,
                                 const std::string &wrt,
                                 const MISOInputs &inputs,
                                 mfem::HypreParVector &partial);

   template <typename T>
   MISOOutput(T x) : self_(new model<T>(std::move(x)))
   { }

private:
   class concept_t
   {
   public:
      virtual ~concept_t() = default;
      virtual void setInputs_(const MISOInputs &inputs) = 0;
      virtual void setOptions_(const nlohmann::json &options) = 0;
      virtual double calcOutput_(const MISOInputs &inputs) = 0;
      virtual double calcOutputPartial_(const std::string &wrt,
                                        const MISOInputs &inputs) = 0;
      virtual void calcOutputPartial_(const std::string &wrt,
                                      const MISOInputs &inputs,
                                      mfem::HypreParVector &partial) = 0;
   };

   template <typename T>
   class model final : public concept_t
   {
   public:
      model(T x) : data_(std::move(x)) { }
      void setInputs_(const MISOInputs &inputs) override
      {
         setInputs(data_, inputs);
      }
      void setOptions_(const nlohmann::json &options) override
      {
         setOptions(data_, options);
      }
      double calcOutput_(const MISOInputs &inputs) override
      {
         return calcOutput(data_, inputs);
      }
      double calcOutputPartial_(const std::string &wrt,
                                const MISOInputs &inputs) override
      {
         return calcOutputPartial(data_, wrt, inputs);
      }
      void calcOutputPartial_(const std::string &wrt,
                              const MISOInputs &inputs,
                              mfem::HypreParVector &partial) override
      {
         calcOutputPartial(data_, wrt, inputs, partial);
      }

      T data_;
   };

   std::unique_ptr<concept_t> self_;
};

inline void setInputs(MISOOutput &output, const MISOInputs &inputs)
{
   output.self_->setInputs_(inputs);
}

inline void setOptions(MISOOutput &output, const nlohmann::json &options)
{
   output.self_->setOptions_(options);
}

inline double calcOutput(MISOOutput &output, const MISOInputs &inputs)
{
   return output.self_->calcOutput_(inputs);
}

inline double calcOutputPartial(MISOOutput &output,
                                const std::string &wrt,
                                const MISOInputs &inputs)
{
   return output.self_->calcOutputPartial_(wrt, inputs);
}

inline void calcOutputPartial(MISOOutput &output,
                              const std::string &wrt,
                              const MISOInputs &inputs,
                              mfem::HypreParVector &partial)
{
   output.self_->calcOutputPartial_(wrt, inputs, partial);
}

}  // namespace miso

#endif
