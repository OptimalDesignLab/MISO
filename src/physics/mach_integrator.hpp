#ifndef MACH_INTEGRATOR
#define MACH_INTEGRATOR

#include <memory>
#include <vector>
#include <string>
#include <algorithm>

#include "mfem.hpp"

#include "mach_input.hpp"

namespace mach
{
/// Creates common interface for integrators used by mach
/// A MachIntegrator can wrap any type `T` that has a function
/// `setInput(T &, const std::string &, const MachInput &)` defined.
/// We have defined this function where `T` is a
/// `mfem::NonlinearFormIntegrator` so that every nonlinear form integrator
/// can be wrapped with a MachIntegrator. This default implementation does
/// nothing, but `setInput` can be specialized for specific integrators that
/// depend on scalar or field inputs. For example, see `setInput` for
/// `HybridACLossFunctionalIntegrator`.

/// We use this class as a way to achieve polymorphism without having to modify
/// `mfem::NonlinearFormIntegrator`. This approach is based on the example in
/// Sean Parent's talk: ``Inheritance is the base class of evil''
class MachIntegrator
{
public:
   friend void setInputs(MachIntegrator &integ, const MachInputs &inputs);

   template <typename T>
   MachIntegrator(T &x) : self_(new model<T>(x))
   { }
   MachIntegrator(const MachIntegrator &x) : self_(x.self_->copy_()) { }
   MachIntegrator(MachIntegrator &&) noexcept = default;

   MachIntegrator &operator=(const MachIntegrator &x)
   {
      MachIntegrator tmp(x);
      *this = std::move(tmp);
      return *this;
   }
   MachIntegrator &operator=(MachIntegrator &&) noexcept = default;

   ~MachIntegrator() = default;

private:
   class concept_t
   {
   public:
      virtual ~concept_t() = default;
      virtual concept_t *copy_() const = 0;
      virtual void setInputs_(const MachInputs &inputs) const = 0;
   };

   template <typename T>
   class model : public concept_t
   {
   public:
      model(T &x) : integ(x) { }
      concept_t *copy_() const override { return new model(*this); }
      void setInputs_(const MachInputs &inputs) const override
      {
         setInputs(integ, inputs);
      }

      T &integ;
   };

   std::unique_ptr<concept_t> self_;
};

/// Used to set inputs in several integrators
void setInputs(std::vector<MachIntegrator> &integrators,
               const MachInputs &inputs);

/// Used to set inputs in the underlying integrator
void setInputs(MachIntegrator &integ, const MachInputs &inputs);

/// Default implementation of setInput for a NonlinearFormIntegrator that does
/// nothing but allows each child class of NonlinearFormIntegrator to be a
/// `MachIntegrator`
inline void setInputs(mfem::NonlinearFormIntegrator &integ,
                      const MachInputs &inputs)
{ }

/// Default implementation of setInput for a LinearFormIntegrator that does
/// nothing but allows each child class of LinearFormIntegrator to be a
/// `MachIntegrator`
inline void setInputs(mfem::LinearFormIntegrator &integ,
                      const MachInputs &inputs)
{ }

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

}  // namespace mach

#endif
