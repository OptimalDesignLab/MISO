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
/// We have defined this function where `T` is a `MachLinearForm` so that every
/// linear form can be wrapped with a MachLoad.

/// We use this class as a way to achieve polymorphism without needing to rely
/// on inheritance This approach is based on the example in Sean Parent's talk:
/// ``Inheritance is the base class of evil''
class MachLoad final
{
public:
   /// Used to set scalar inputs in the underlying load type
   /// Ends up calling `setInputs` on either the `MachLinearForm` or
   /// a specialized version for each particular load.
   friend void setInputs(MachLoad &load,
                         const MachInputs &inputs);

   /// Assemble the load vector on the true dofs and add it to tv
   friend void addLoad(MachLoad &load,
                       mfem::Vector &tv);
   
   /// Assemble the load vector's sensitivity to a scalar and contract it with
   /// res_bar
   friend double vectorJacobianProduct(MachLoad &load,
                                       const mfem::HypreParVector &res_bar,
                                       std::string wrt);

   /// Assemble the load vector's sensitivity to a field and contract it with
   /// res_bar
   friend void vectorJacobianProduct(MachLoad &load,
                                     const mfem::HypreParVector &res_bar,
                                     std::string wrt,
                                     mfem::HypreParVector &wrt_bar);

   template <typename T>
   MachLoad(T &x) : self_(new model<T>(x))
   { }
   MachLoad(const MachLoad &x) : self_(x.self_->copy_())
   { }
   MachLoad(MachLoad&&) noexcept = default;

   MachLoad& operator=(const MachLoad &x)
   { MachLoad tmp(x); *this = std::move(tmp); return *this; }
   MachLoad& operator=(MachLoad&&) noexcept = default;

private:
   class concept_t
   {
   public:
      virtual ~concept_t() = default;
      virtual concept_t* copy_() const = 0;
      virtual void setInputs_(const MachInputs &inputs) const = 0;
      virtual void addLoad_(mfem::Vector &tv) = 0;
      virtual double vectorJacobianProduct_(const mfem::HypreParVector &res_bar,
                                            std::string wrt) = 0;
      virtual void vectorJacobianProduct_(const mfem::HypreParVector &res_bar,
                                          std::string wrt,
                                          mfem::HypreParVector &wrt_bar) = 0;
   };

   template <typename T>
   class model : public concept_t
   {
   public:
      model(T &x) : data_(x) { }
      concept_t* copy_() const override { return new model(*this); }
      void setInputs_(const MachInputs &inputs) const override
      { setInputs(data_, inputs); }
      void addLoad_(mfem::Vector &tv) override
      { addLoad(data_, tv); }
      double vectorJacobianProduct_(const mfem::HypreParVector &res_bar,
                                    std::string wrt) override
      { return vectorJacobianProduct(data_, res_bar, wrt); }
      void vectorJacobianProduct_(const mfem::HypreParVector &res_bar,
                                  std::string wrt,
                                  mfem::HypreParVector &wrt_bar) override
      { vectorJacobianProduct(data_, res_bar, wrt, wrt_bar); }


      T &data_;
   };

   std::unique_ptr<concept_t> self_;
};

inline void setInputs(MachLoad &load,
                      const MachInputs &inputs)
{
   load.self_->setInputs_(inputs);
}

inline void addLoad(MachLoad &load,
                    mfem::Vector &tv)
{
   load.self_->addLoad_(tv);
}

inline double vectorJacobianProduct(MachLoad &load,
                                    const mfem::HypreParVector &res_bar,
                                    std::string wrt)
{
   return load.self_->vectorJacobianProduct_(res_bar, wrt);
}

inline void vectorJacobianProduct(MachLoad &load,
                                  const mfem::HypreParVector &res_bar,
                                  std::string wrt,
                                  mfem::HypreParVector &wrt_bar)
{
   load.self_->vectorJacobianProduct_(res_bar, wrt, wrt_bar);
}

} // namespace mach

#endif
