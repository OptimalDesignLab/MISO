#ifndef MISO_INTEGRATOR
#define MISO_INTEGRATOR

#include <memory>
#include <vector>
#include <string>
#include <algorithm>

#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "miso_input.hpp"

namespace miso
{
/// Default implementation of setInputs for a NonlinearFormIntegrator that does
/// nothing but allows each child class of NonlinearFormIntegrator to be a
/// `MISOIntegrator`
inline void setInputs(mfem::NonlinearFormIntegrator &integ,
                      const MISOInputs &inputs)
{ }

/// Default implementation of setInputs for a LinearFormIntegrator that does
/// nothing but allows each child class of LinearFormIntegrator to be a
/// `MISOIntegrator`
inline void setInputs(mfem::LinearFormIntegrator &integ,
                      const MISOInputs &inputs)
{ }

/// Default implementation of setOptions for a NonlinearFormIntegrator that does
/// nothing but allows each child class of NonlinearFormIntegrator to be a
/// `MISOIntegrator`
inline void setOptions(mfem::NonlinearFormIntegrator &integ,
                       const nlohmann::json &options)
{ }

/// Default implementation of setOptions for a LinearFormIntegrator that does
/// nothing but allows each child class of LinearFormIntegrator to be a
/// `MISOIntegrator`
inline void setOptions(mfem::LinearFormIntegrator &integ,
                       const nlohmann::json &options)
{ }

/// Creates common interface for integrators used by miso
/// A MISOIntegrator can wrap any type `T` that has a function
/// `setInput(T &, const std::string &, const MISOInput &)` defined.
/// We have defined this function where `T` is a
/// `mfem::NonlinearFormIntegrator` so that every nonlinear form integrator
/// can be wrapped with a MISOIntegrator. This default implementation does
/// nothing, but `setInput` can be specialized for specific integrators that
/// depend on scalar or field inputs. For example, see `setInput` for
/// `HybridACLossFunctionalIntegrator`.

/// We use this class as a way to achieve polymorphism without having to modify
/// `mfem::NonlinearFormIntegrator`. This approach is based on the example in
/// Sean Parent's talk: ``Inheritance is the base class of evil''
class MISOIntegrator
{
public:
   friend void setInputs(MISOIntegrator &integ, const MISOInputs &inputs);
   friend void setOptions(MISOIntegrator &integ, const nlohmann::json &options);

   template <typename T>
   MISOIntegrator(T &x) : self_(new model<T>(x))
   { }
   MISOIntegrator(const MISOIntegrator &x) : self_(x.self_->copy_()) { }
   MISOIntegrator(MISOIntegrator &&) noexcept = default;

   MISOIntegrator &operator=(const MISOIntegrator &x)
   {
      MISOIntegrator tmp(x);
      *this = std::move(tmp);
      return *this;
   }
   MISOIntegrator &operator=(MISOIntegrator &&) noexcept = default;

   ~MISOIntegrator() = default;

private:
   class concept_t
   {
   public:
      virtual ~concept_t() = default;
      virtual concept_t *copy_() const = 0;
      virtual void setInputs_(const MISOInputs &inputs) const = 0;
      virtual void setOptions_(const nlohmann::json &options) const = 0;
   };

   template <typename T>
   class model final : public concept_t
   {
   public:
      model(T &x) : integ(x) { }
      concept_t *copy_() const override { return new model(*this); }
      void setInputs_(const MISOInputs &inputs) const override
      {
         setInputs(integ, inputs);
      }
      void setOptions_(const nlohmann::json &options) const override
      {
         setOptions(integ, options);
      }

      T &integ;
   };

   std::unique_ptr<concept_t> self_;
};

/// Used to set inputs in several integrators
void setInputs(std::vector<MISOIntegrator> &integrators,
               const MISOInputs &inputs);

/// Used to set inputs in the underlying integrator
void setInputs(MISOIntegrator &integ, const MISOInputs &inputs);

/// Used to set options in several integrators
void setOptions(std::vector<MISOIntegrator> &integrators,
                const nlohmann::json &options);

/// Used to set options in the underlying integrator
void setOptions(MISOIntegrator &integ, const nlohmann::json &options);

/// Function meant to be specialized to allow sensitivity integrators
/// to be associated with the forward version of the integrator
/// \param[in] primal_integ - integrator used in forward evaluation
/// \param[in] fields - map of fields solver depends on
/// \param[inout] sens - map of linear forms that will assemble the sensitivity
/// \param[inout] scalar_sens - map of nonlinear forms that will assemble the
///                             scalar sensitivity
template <typename T>
inline void addSensitivityIntegrator(
    T &primal_integ,
    std::unordered_map<std::string, mfem::ParGridFunction> &fields,
    std::map<std::string, mfem::ParLinearForm> &sens,
    std::map<std::string, mfem::ParNonlinearForm> &scalar_sens)
{ }

}  // namespace miso

#endif
