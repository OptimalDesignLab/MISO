#ifndef MISO_LOAD
#define MISO_LOAD

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
/// Creates common interface for load vectors used by miso
/// A MISOLoad can wrap any type `T` that has the interface of a load vector.
/// We have defined this function where `T` is a `MISOLinearForm` so that every
/// linear form can be wrapped with a MISOLoad.

/// We use this class as a way to achieve polymorphism without needing to rely
/// on inheritance This approach is based on the example in Sean Parent's talk:
/// ``Inheritance is the base class of evil''
class MISOLoad final
{
public:
   /// Used to set scalar inputs in the underlying load type
   /// Ends up calling `setInputs` on either the `MISOLinearForm` or
   /// a specialized version for each particular load.
   friend void setInputs(MISOLoad &load, const MISOInputs &inputs);

   /// Used to set options for the underlying load type
   friend void setOptions(MISOLoad &load, const nlohmann::json &options);

   /// Assemble the load vector on the true dofs and add it to tv
   friend void addLoad(MISOLoad &load, mfem::Vector &tv);

   /// Assemble the load vector's sensitivity to a scalar and contract it with
   /// res_bar
   friend double vectorJacobianProduct(MISOLoad &load,
                                       const mfem::HypreParVector &res_bar,
                                       std::string wrt);

   /// Assemble the load vector's sensitivity to a field and contract it with
   /// res_bar
   friend void vectorJacobianProduct(MISOLoad &load,
                                     const mfem::HypreParVector &res_bar,
                                     std::string wrt,
                                     mfem::HypreParVector &wrt_bar);

   template <typename T>
   MISOLoad(T &x) : self_(new model<T>(x))
   { }

private:
   class concept_t
   {
   public:
      virtual ~concept_t() = default;
      virtual void setInputs_(const MISOInputs &inputs) = 0;
      virtual void setOptions_(const nlohmann::json &options) = 0;
      virtual void addLoad_(mfem::Vector &tv) = 0;
      virtual double vectorJacobianProduct_(const mfem::HypreParVector &res_bar,
                                            std::string wrt) = 0;
      virtual void vectorJacobianProduct_(const mfem::HypreParVector &res_bar,
                                          std::string wrt,
                                          mfem::HypreParVector &wrt_bar) = 0;
   };

   template <typename T>
   class model final : public concept_t
   {
   public:
      model(T &x) : data_(x) { }
      void setInputs_(const MISOInputs &inputs) override
      {
         setInputs(data_, inputs);
      }
      void setOptions_(const nlohmann::json &options) override
      {
         setOptions(data_, options);
      }
      void addLoad_(mfem::Vector &tv) override { addLoad(data_, tv); }
      double vectorJacobianProduct_(const mfem::HypreParVector &res_bar,
                                    std::string wrt) override
      {
         return vectorJacobianProduct(data_, res_bar, wrt);
      }
      void vectorJacobianProduct_(const mfem::HypreParVector &res_bar,
                                  std::string wrt,
                                  mfem::HypreParVector &wrt_bar) override
      {
         vectorJacobianProduct(data_, res_bar, wrt, wrt_bar);
      }

      T &data_;
   };

   std::unique_ptr<concept_t> self_;
};

inline void setInputs(MISOLoad &load, const MISOInputs &inputs)
{
   load.self_->setInputs_(inputs);
}

inline void setOptions(MISOLoad &load, const nlohmann::json &options)
{
   load.self_->setOptions_(options);
}

inline void addLoad(MISOLoad &load, mfem::Vector &tv)
{
   load.self_->addLoad_(tv);
}

inline double vectorJacobianProduct(MISOLoad &load,
                                    const mfem::HypreParVector &res_bar,
                                    std::string wrt)
{
   return load.self_->vectorJacobianProduct_(res_bar, std::move(wrt));
}

inline void vectorJacobianProduct(MISOLoad &load,
                                  const mfem::HypreParVector &res_bar,
                                  std::string wrt,
                                  mfem::HypreParVector &wrt_bar)
{
   load.self_->vectorJacobianProduct_(res_bar, std::move(wrt), wrt_bar);
}

}  // namespace miso

#endif
