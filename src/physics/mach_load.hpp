#ifndef MACH_LOAD
#define MACH_LOAD

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
   /// Gets the size of the underlying load vector
   /// \param[in] load - the load vector whose size is being queried
   /// \returns the size of the load vector
   friend int getSize(const MachLoad &load);

   /// Used to set scalar inputs in the underlying load type
   /// Ends up calling `setInputs` on either the `MachLinearForm` or
   /// a specialized version for each particular load.
   friend void setInputs(MachLoad &load, const MachInputs &inputs);

   /// Used to set options for the underlying load type
   friend void setOptions(MachLoad &load, const nlohmann::json &options);

   /// Assemble the load vector on the true dofs and add it to tv
   friend void addLoad(MachLoad &load, mfem::Vector &tv);

   /// Compute the load's sensitivity to a scalar and contract it with wrt_dot
   /// \param[inout] load - the load whose sensitivity we want
   /// \param[in] wrt_dot - the "wrt"-sized vector to contract with the
   /// sensitivity
   /// \param[in] wrt - string denoting what variable to take the derivative
   /// with respect to
   /// \return the assembled/contracted sensitivity
   friend double jacobianVectorProduct(MachLoad &load,
                                       const mfem::Vector &wrt_dot,
                                       const std::string &wrt);

   /// Compute the load's sensitivity to a vector and contract it with wrt_dot
   /// \param[inout] load - the load whose sensitivity we want
   /// \param[in] wrt_dot - the "wrt"-sized vector to contract with the
   /// sensitivity
   /// \param[in] wrt - string denoting what variable to take the derivative
   /// with respect to
   /// \param[inout] res_dot - the assembled/contracted sensitivity is
   /// accumulated into res_dot
   friend void jacobianVectorProduct(MachLoad &load,
                                     const mfem::Vector &wrt_dot,
                                     const std::string &wrt,
                                     mfem::Vector &res_dot);

   /// Assemble the load vector's sensitivity to a scalar and contract it with
   /// res_bar
   friend double vectorJacobianProduct(MachLoad &load,
                                       const mfem::Vector &res_bar,
                                       const std::string &wrt);

   /// Assemble the load vector's sensitivity to a field and contract it with
   /// res_bar
   friend void vectorJacobianProduct(MachLoad &load,
                                     const mfem::Vector &res_bar,
                                     const std::string &wrt,
                                     mfem::Vector &wrt_bar);

   template <typename T>
   MachLoad(T &x) : self_(new model<T>(x))
   { }

private:
   class concept_t
   {
   public:
      virtual ~concept_t() = default;
      virtual void setInputs_(const MachInputs &inputs) = 0;
      virtual void setOptions_(const nlohmann::json &options) = 0;
      virtual void addLoad_(mfem::Vector &tv) = 0;
      virtual double jacobianVectorProduct_(const mfem::Vector &wrt_dot,
                                            const std::string &wrt) = 0;
      virtual void jacobianVectorProduct_(const mfem::Vector &wrt_dot,
                                          const std::string &wrt,
                                          mfem::Vector &res_dot) = 0;
      virtual double vectorJacobianProduct_(const mfem::Vector &res_bar,
                                            const std::string &wrt) = 0;
      virtual void vectorJacobianProduct_(const mfem::Vector &res_bar,
                                          const std::string &wrt,
                                          mfem::Vector &wrt_bar) = 0;
   };

   template <typename T>
   class model final : public concept_t
   {
   public:
      model(T &x) : data_(x) { }
      void setInputs_(const MachInputs &inputs) override
      {
         setInputs(data_, inputs);
      }
      void setOptions_(const nlohmann::json &options) override
      {
         setOptions(data_, options);
      }
      void addLoad_(mfem::Vector &tv) override { addLoad(data_, tv); }
      double jacobianVectorProduct_(const mfem::Vector &wrt_dot,
                                    const std::string &wrt) override
      {
         return jacobianVectorProduct(data_, wrt_dot, wrt);
      }
      void jacobianVectorProduct_(const mfem::Vector &wrt_dot,
                                  const std::string &wrt,
                                  mfem::Vector &res_dot) override
      {
         jacobianVectorProduct(data_, wrt_dot, wrt, res_dot);
      }
      double vectorJacobianProduct_(const mfem::Vector &res_bar,
                                    const std::string &wrt) override
      {
         return vectorJacobianProduct(data_, res_bar, wrt);
      }
      void vectorJacobianProduct_(const mfem::Vector &res_bar,
                                  const std::string &wrt,
                                  mfem::Vector &wrt_bar) override
      {
         vectorJacobianProduct(data_, res_bar, wrt, wrt_bar);
      }

      T &data_;
   };

   std::unique_ptr<concept_t> self_;
};

inline void setInputs(MachLoad &load, const MachInputs &inputs)
{
   load.self_->setInputs_(inputs);
}

inline void setOptions(MachLoad &load, const nlohmann::json &options)
{
   load.self_->setOptions_(options);
}

inline void addLoad(MachLoad &load, mfem::Vector &tv)
{
   load.self_->addLoad_(tv);
}

inline double jacobianVectorProduct(MachLoad &load,
                                    const mfem::Vector &wrt_dot,
                                    const std::string &wrt)
{
   return load.self_->jacobianVectorProduct_(wrt_dot, wrt);
}

inline void jacobianVectorProduct(MachLoad &load,
                                  const mfem::Vector &wrt_dot,
                                  const std::string &wrt,
                                  mfem::Vector &res_dot)
{
   load.self_->jacobianVectorProduct_(wrt_dot, wrt, res_dot);
}

inline double vectorJacobianProduct(MachLoad &load,
                                    const mfem::Vector &res_bar,
                                    const std::string &wrt)
{
   return load.self_->vectorJacobianProduct_(res_bar, wrt);
}

inline void vectorJacobianProduct(MachLoad &load,
                                  const mfem::Vector &res_bar,
                                  const std::string &wrt,
                                  mfem::Vector &wrt_bar)
{
   load.self_->vectorJacobianProduct_(res_bar, wrt, wrt_bar);
}

}  // namespace mach

#endif
