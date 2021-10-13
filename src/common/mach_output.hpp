#ifndef MACH_OUTPUT
#define MACH_OUTPUT

#include <memory>
#include <utility>
#include <vector>
#include <string>
#include <algorithm>

#include "mfem.hpp"

#include "mach_input.hpp"

namespace mach
{
/// Creates common interface for outputs computable by mach
/// A MachOutput can wrap any type `T` that has the interface of an output.
class MachOutput final
{
public:
   /// Used to set inputs in the underlying output type
   friend void setInputs(MachOutput &output, const MachInputs &inputs);

   /// Compute the output vector on the true dofs and add it to tv
   friend double calcOutput(MachOutput &output, const MachInputs &inputs);

   /// Compute the output's sensitivity to a scalar
   friend double calcOutputPartial(MachOutput &output,
                                   const std::string &wrt,
                                   const MachInputs &inputs);

   /// Compute the output's sensitivity to a field and store in @a partial
   friend void calcOutputPartial(MachOutput &output,
                                 const std::string &wrt,
                                 const MachInputs &inputs,
                                 mfem::HypreParVector &partial);

   template <typename T>
   MachOutput(T &x) : self_(new model<T>(x))
   { }
   MachOutput(const MachOutput &x) : self_(x.self_->copy_()) { }
   MachOutput(MachOutput &&) noexcept = default;

   MachOutput &operator=(const MachOutput &x)
   {
      MachOutput tmp(x);
      *this = std::move(tmp);
      return *this;
   }
   MachOutput &operator=(MachOutput &&) noexcept = default;

   ~MachOutput() = default;

private:
   class concept_t
   {
   public:
      virtual ~concept_t() = default;
      virtual concept_t *copy_() const = 0;
      virtual void setInputs_(const MachInputs &inputs) const = 0;
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
      model(T &x) : data_(x) { }
      concept_t *copy_() const override { return new model(*this); }
      void setInputs_(const MachInputs &inputs) const override
      {
         setInputs(data_, inputs);
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

      T &data_;
   };

   std::unique_ptr<concept_t> self_;
};

inline void setInputs(MachOutput &output, const MachInputs &inputs)
{
   output.self_->setInputs_(inputs);
}

inline double calcOutput(MachOutput &output, const MachInputs &inputs)
{
   return output.self_->calcOutput(inputs);
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
