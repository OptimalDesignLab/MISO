#ifndef MACH_RESIDUAL
#define MACH_RESIDUAL

#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "mach_input.hpp"

namespace mach
{
/// Defines a common interface for residual functions used by mach.
/// A MachResidual can wrap any type `T` that has the interface of a residual
/// function.  For example, one instance of `T` is given by `MachNonlinearForm`,
/// so that every nonlinear form can be wrapped by a MachResidual.
/// \note We use this approach to achieve polymorphism without inheritance.
/// This is important because we need to derive from `mfem` classes frequently,
/// but at the same time we need to build on their classes' functionality.
/// Without this approach, we would need to use frequent dynamic casts because
/// we have pointers to base classes.
/// \note This approach is based on the example in Sean Parent's talk:
/// ``Inheritance is the base class of evil''
class MachResidual final
{
public:
   /// Set inputs in the underlying residual type
   /// \param[inout] residual - the residual being assigned the input
   /// \param[in] inputs - the inputs that are being assigned
   /// \note Ends up calling `setInputs` on either the `MachNonlinearForm` or
   /// a specialized version for each particular residual.
   friend void setInputs(MachResidual &residual, const MachInputs &inputs);

   /// Set options in the underlying residual type
   /// \param[inout] residual - the residual being assigned the input
   /// \param[in] options - the options that are being assigned
   friend void setOptions(MachResidual &residual,
                          const nlohmann::json &options);

   /// Evaluate the residual function at given inputs and return as `res_vec`
   /// \param[inout] residual - the residual being evaluated
   /// \param[in] inputs - the independent variables at which to evaluate `res`
   /// \param[out] res_vec - the dependent variable, the output from `residual`
   friend void evaluate(MachResidual &residual,
                        const MachInputs &inputs,
                        mfem::Vector &res_vec);

   // TODO: we will eventual want to add functions for Jacobian products

   // The following constructors, assignment operators, and destructors allow
   // the `MachResidual` to wrap the generic type `T`.

   template <typename T>
   MachResidual(T x) : self_(new model<T>(std::move(x)))
   { }
   MachResidual(const MachResidual &) = delete;
   MachResidual &operator=(const MachResidual &) = delete;
   MachResidual(MachResidual &&) noexcept = default;
   MachResidual &operator=(MachResidual &&) noexcept = default;

   ~MachResidual() = default;

private:
   /// Abstract base class with common functions needed by all residuals
   class concept_t
   {
   public:
      virtual ~concept_t() = default;
      virtual void setInputs_(const MachInputs &inputs) = 0;
      virtual void setOptions_(const nlohmann::json &options) = 0;
      virtual void eval_(const MachInputs &inputs, mfem::Vector &res_vec) = 0;
   };

   /// Concrete (templated) class for residuals
   /// \tparam T - a residual class
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
      void eval_(const MachInputs &inputs, mfem::Vector &res_vec) override
      {
         evaluate(data_, inputs, res_vec);
      }

      T data_;
   };

   /// Pointer to `model` via its abstract base class `concept_t`
   std::unique_ptr<concept_t> self_;
};

inline void setInputs(MachResidual &residual, const MachInputs &inputs)
{
   // passes `inputs` on to the `setInputs` function for the concrete
   // residual type
   residual.self_->setInputs_(inputs);
}

inline void setOptions(MachResidual &residual, const nlohmann::json &options)
{
   // passes `options` on to the `setOptions` function for the concrete
   // residual type
   residual.self_->setOptions_(options);
}

inline void evaluate(MachResidual &residual,
                     const MachInputs &inputs,
                     mfem::Vector &res_vec)
{
   // passes `inputs` and `res_vec` on to the `evaluate` function for the
   // concrete residual type
   residual.self_->eval_(inputs, res_vec);
}

}  // namespace mach

#endif  // MACH_RESIDUAL
