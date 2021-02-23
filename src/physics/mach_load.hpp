#ifndef MACH_LOAD
#define MACH_LOAD

#include <memory>
#include <vector>
#include <string>
#include <algorithm>

#include "mfem.hpp"

#include "mach_input.hpp"

namespace mach
{

/// Creates common interface for load vectors used by mach
/// A MachLoad can wrap any type `T` that has the interface of a load vector.
/// We have defined this function where `T` is a
/// `mfem::LinearForm` so that every linear form can be wrapped with a
/// MachLoad.

/// We use this class as a way to achieve polymorphism without needing to rely
/// on inheritance This approach is based on the example in Sean Parent's talk:
/// ``Inheritance is the base class of evil''
class MachLoad
{
public:
   template <typename T>
   MachLoad(T &x) : self_(new model<T>(x))
   { }
   MachLoad(const MachLoad &x) : self_(x.self_->copy_())
   { }
   MachLoad(MachLoad&&) noexcept = default;

   MachLoad& operator=(const MachLoad &x)
   { MachLoad tmp(x); *this = std::move(tmp); return *this; }
   MachLoad& operator=(MachLoad&&) noexcept = default;

   friend void setInputs(MachLoad &load,
                         const MachInputs &inputs);

   friend void assemble(MachLoad &load,
                        mfem::HypreParVector &tv);

private:
   class concept_t
   {
   public:
      virtual ~concept_t() = default;
      virtual concept_t* copy_() const = 0;
      virtual void setInputs_(const MachInputs &inputs) const = 0;
      virtual void assemble_(mfem::HypreParVector &tv) = 0;
   };

   template <typename T>
   class model : public concept_t
   {
   public:
      model(T &x) : data_(x) { }
      concept_t* copy_() const override { return new model(*this); }
      void setInputs_(const MachInputs &inputs) const override
      { setInputs(data_, inputs); }
      void assemble_(mfem::HypreParVector &tv) override
      { assemble(data_, tv); }

      T &data_;
   };

   std::unique_ptr<concept_t> self_;
};

/// Used to set scalar inputs in the underlying load type
/// Ends up calling `setInputs` on for either the `NonlinearFormloadrator` or
/// a specialized version for each particular loadrator.
void setInputs(MachLoad &load,
               const MachInputs &inputs);

/// Assemble the load vector
void assemble(MachLoad &load,
              mfem::HypreParVector &tv);

/// Default implementation of setInputs for a LinearForm that get the linear
/// form's integrators and sets their inputs
void setInputs(mfem::LinearForm &load,
               const MachInputs &inputs);

} // namespace mach

namespace mfem
{

/// Assemble the linear form
void assemble(LinearForm &load,
              HypreParVector &tv);

/// Assemble and then parallel assemble the parallel linear form
void assemble(ParLinearForm &load,
              HypreParVector &tv);

} // namespace mfem

#endif
