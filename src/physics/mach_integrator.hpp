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
   template <typename T>
   MachIntegrator(T &x) : self_(new model<T>(x))
   { }
   MachIntegrator(const MachIntegrator &x) : self_(x.self_->copy_())
   { }
   MachIntegrator(MachIntegrator&&) noexcept = default;

   MachIntegrator& operator=(const MachIntegrator &x)
   { MachIntegrator tmp(x); *this = std::move(tmp); return *this; }
   MachIntegrator& operator=(MachIntegrator&&) noexcept = default;

   friend void setInput(MachIntegrator &integ,
                        const std::string &name,
                        const MachInput &input);

private:
   class concept_t
   {
   public:
      virtual ~concept_t() = default;
      virtual concept_t* copy_() const = 0;
      virtual void setInput_(const std::string &name,
                             const MachInput &input) const = 0;
   };

   template <typename T>
   class model : public concept_t
   {
   public:
      model(T &x) : integ(x) { }
      concept_t* copy_() const override { return new model(*this); }
      void setInput_(const std::string &name,
                     const MachInput &input) const override
      { setInput(integ, name, input); }

      T &integ;
   };

   std::unique_ptr<concept_t> self_;
};

/// Used to set scalar inputs in several integrators
void setScalarInputs(std::vector<MachIntegrator> &integrators,
                    const MachInputs &inputs);

/// Used to set a scalar input in several integrators
void setScalarInput(std::vector<MachIntegrator> &integrators,
                    const std::string &name,
                    const MachInput &input);

/// Used to set scalar inputs in the underlying integrator
/// Ends up calling `setInput` on for either the `NonlinearFormIntegrator` or
/// a specialized version for each particular integrator.
void setInput(MachIntegrator &integ,
              const std::string &name,
              const MachInput &input);

/// Default implementation of setInput for a NonlinearFormIntegrator that does
/// nothing but allows each child class of NonlinearFormIntegrator to be a
/// `MachIntegrator`
inline void setInput(mfem::NonlinearFormIntegrator &integ,
                     const std::string &name,
                     const MachInput &input) { };

/// Default implementation of setInput for a LinearFormIntegrator that does
/// nothing but allows each child class of LinearFormIntegrator to be a
/// `MachIntegrator`
inline void setInput(mfem::LinearFormIntegrator &integ,
                     const std::string &name,
                     const MachInput &input) { };

/// Function meant to be specialized to allow residual sensitivity integrators
/// to be associated with the forward version of the integrator
/// \param[in] primal_integ - integrator used in residual evaluation
/// \param[in] res_fields - map of fields solver depends on
/// \param[inout] res_sens - map of sens to linear form that will assemble the sensitivity
template <typename T>
inline void addResidualSensitivityIntegrator(
   T &primal_integ,
   std::unordered_map<std::string, mfem::ParGridFunction> &res_fields,
   std::map<std::string, mfem::ParLinearForm> &res_sens,
   std::map<std::string, mfem::ParNonlinearForm> &res_scalar_sens) { };

/// Function meant to be specialized to allow a output sensitivity integrators
/// to be associated with the forward version of the integrator
/// \param[in] primal_integ - integrator used for functional evaluation
/// \param[in] res_fields - map of fields solver depends on
/// \param[inout] output_sens - map of sens to linear form that will assemble the sensitivity
template <typename T>
inline void addOutputSensitivityIntegrator(
   T &primal_integ,
   std::unordered_map<std::string, mfem::ParGridFunction> &res_fields,
   std::map<std::string, mfem::ParLinearForm> &output_sens) { };

} // namespace mach

#endif
