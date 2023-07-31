#ifndef MISO_RESIDUAL
#define MISO_RESIDUAL

#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "miso_input.hpp"

namespace miso
{
/// Defines a common interface for residual functions used by miso.
/// A MISOResidual can wrap any type `T` that has the interface of a residual
/// function.  For example, one instance of `T` is given by `MISONonlinearForm`,
/// so that every nonlinear form can be wrapped by a MISOResidual.
/// \note We use this approach to achieve polymorphism without inheritance.
/// This is important because we need to derive from `mfem` classes frequently,
/// but at the same time we need to build on their classes' functionality.
/// Without this approach, we would need to use frequent dynamic casts because
/// we have pointers to base classes.
/// \note This approach is based on the example in Sean Parent's talk:
/// ``Inheritance is the base class of evil''
class MISOResidual final
{
public:
   /// Gets the number of equations/unknowns of the underlying residual type
   /// \param[inout] residual - the residual whose size is being queried
   /// \returns the number of equations/unknowns
   /// \note Needed, e.g., by the ODESystemOperator constructor (see evolver.*)
   friend int getSize(const MISOResidual &residual);

   /// Set inputs in the underlying residual type
   /// \param[inout] residual - the residual being assigned the input
   /// \param[in] inputs - the inputs that are being assigned
   /// \note Ends up calling `setInputs` on either the `MISONonlinearForm` or
   /// a specialized version for each particular residual.
   friend void setInputs(MISOResidual &residual, const MISOInputs &inputs);

   /// Set options in the underlying residual type
   /// \param[inout] residual - the residual whose options are being set
   /// \param[in] options - the options that are being assigned
   friend void setOptions(MISOResidual &residual,
                          const nlohmann::json &options);

   /// Evaluate the residual function at given inputs and return as `res_vec`
   /// \param[inout] residual - the residual being evaluated
   /// \param[in] inputs - the independent variables at which to evaluate `res`
   /// \param[out] res_vec - the dependent variable, the output from `residual`
   friend void evaluate(MISOResidual &residual,
                        const MISOInputs &inputs,
                        mfem::Vector &res_vec);

   /// Compute the Jacobian of the given residual and return a reference to it
   /// \param[inout] residual - function whose Jacobian we want
   /// \param[in] inputs - the variables needed to evaluate the Jacobian
   /// \param[in] wrt - the input we are differentiating with respect to
   /// \returns a reference to the residuals Jacobian with respect to `wrt`
   /// \note the underlying `Operator` is owned by `residual`
   friend mfem::Operator &getJacobian(MISOResidual &residual,
                                      const MISOInputs &inputs,
                                      std::string wrt);

   // TODO: we will eventual want to add functions for Jacobian products

   // The following constructors, assignment operators, and destructors allow
   // the `MISOResidual` to wrap the generic type `T`.
   template <typename T>
   MISOResidual(T x) : self_(new model<T>(std::move(x)))
   { }

private:
   /// Abstract base class with common functions needed by all residuals
   class concept_t
   {
   public:
      virtual ~concept_t() = default;
      virtual int getSize_() const = 0;
      virtual void setInputs_(const MISOInputs &inputs) = 0;
      virtual void setOptions_(const nlohmann::json &options) = 0;
      virtual void eval_(const MISOInputs &inputs, mfem::Vector &res_vec) = 0;
      virtual mfem::Operator &getJac_(const MISOInputs &inputs,
                                      std::string wrt) = 0;
   };

   /// Concrete (templated) class for residuals
   /// \tparam T - a residual class
   template <typename T>
   class model final : public concept_t
   {
   public:
      model(T x) : data_(std::move(x)) { }
      int getSize_() const override { return getSize(data_); }
      void setInputs_(const MISOInputs &inputs) override
      {
         setInputs(data_, inputs);
      }
      void setOptions_(const nlohmann::json &options) override
      {
         setOptions(data_, options);
      }
      void eval_(const MISOInputs &inputs, mfem::Vector &res_vec) override
      {
         evaluate(data_, inputs, res_vec);
      }
      mfem::Operator &getJac_(const MISOInputs &inputs,
                              std::string wrt) override
      {
         return getJacobian(data_, inputs, std::move(wrt));
      }

      T data_;
   };

   /// Pointer to `model` via its abstract base class `concept_t`
   std::unique_ptr<concept_t> self_;
};

inline int getSize(const MISOResidual &residual)
{
   return residual.self_->getSize_();
}

inline void setInputs(MISOResidual &residual, const MISOInputs &inputs)
{
   // passes `inputs` on to the `setInputs` function for the concrete
   // residual type
   residual.self_->setInputs_(inputs);
}

inline void setOptions(MISOResidual &residual, const nlohmann::json &options)
{
   // passes `options` on to the `setOptions` function for the concrete
   // residual type
   residual.self_->setOptions_(options);
}

inline void evaluate(MISOResidual &residual,
                     const MISOInputs &inputs,
                     mfem::Vector &res_vec)
{
   // passes `inputs` and `res_vec` on to the `evaluate` function for the
   // concrete residual type
   residual.self_->eval_(inputs, res_vec);
}

inline mfem::Operator &getJacobian(MISOResidual &residual,
                                   const MISOInputs &inputs,
                                   std::string wrt)
{
   // passes `inputs` and `res_vec` on to the `getJacobian` function for the
   // concrete residual type
   return residual.self_->getJac_(inputs, std::move(wrt));
}

}  // namespace miso

#endif  // MISO_RESIDUAL
