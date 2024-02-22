#ifndef MISO_INTEGRATOR
#define MISO_INTEGRATOR

#include <map>
#include <memory>
#include <vector>
#include <string>

#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "finite_element_state.hpp"
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

/// Function meant to be overloaded to allow residual sensitivity integrators
/// to be associated with the forward version of the integrator
/// \param[in] primal_integ - integrator used in forward evaluation
/// \param[in] fields - map of fields solver depends on
/// \param[inout] rev_sens - map of linear forms that will assemble the
/// reverse-mode sensitivity
/// \param[inout] rev_scalar_sens - map of nonlinear forms that will assemble
/// the reverse-mode scalar sensitivity
/// \param[inout] fwd_sens - map of linear forms that will assemble the
/// forward-mode sensitivity
/// \param[inout] fwd_scalar_sens - map of nonlinear forms that will assemble
/// the forward-mode scalar sensitivity
/// \param[in] attr_marker - optional list of element attributes the sensitivity
/// integrators should be used on
template <typename T>
inline void addDomainSensitivityIntegrator(
    T &primal_integ,
    std::map<std::string, FiniteElementState> &fields,
    std::map<std::string, mfem::ParLinearForm> &rev_sens,
    std::map<std::string, mfem::ParNonlinearForm> &rev_scalar_sens,
    std::map<std::string, mfem::ParLinearForm> &fwd_sens,
    std::map<std::string, mfem::ParNonlinearForm> &fwd_scalar_sens,
    mfem::Array<int> *attr_marker = nullptr,
    std::string adjoint_name = "adjoint")
{ }

/// Function meant to be overloaded to allow residual sensitivity integrators
/// to be associated with the forward version of the integrator
/// \param[in] primal_integ - integrator used in forward evaluation
/// \param[in] fields - map of fields solver depends on
/// \param[inout] rev_sens - map of linear forms that will assemble the
/// reverse-mode sensitivity
/// \param[inout] rev_scalar_sens - map of nonlinear forms that will assemble
/// the reverse-mode scalar sensitivity
/// \param[inout] fwd_sens - map of linear forms that will assemble the
/// forward-mode sensitivity
/// \param[inout] fwd_scalar_sens - map of nonlinear forms that will assemble
/// the forward-mode scalar sensitivity
/// \param[in] attr_marker - optional list of element attributes the sensitivity
/// integrators should be used on
template <typename T>
inline void addInteriorFaceSensitivityIntegrator(
    T &primal_integ,
    std::map<std::string, FiniteElementState> &fields,
    std::map<std::string, mfem::ParLinearForm> &rev_sens,
    std::map<std::string, mfem::ParNonlinearForm> &rev_scalar_sens,
    std::map<std::string, mfem::ParLinearForm> &fwd_sens,
    std::map<std::string, mfem::ParNonlinearForm> &fwd_scalar_sens,
    std::string adjoint_name = "adjoint")
{ }

/// Function meant to be overloaded to allow residual sensitivity integrators
/// to be associated with the forward version of the integrator
/// \param[in] primal_integ - integrator used in forward evaluation
/// \param[in] fields - map of fields solver depends on
/// \param[inout] rev_sens - map of linear forms that will assemble the
/// reverse-mode sensitivity
/// \param[inout] rev_scalar_sens - map of nonlinear forms that will assemble
/// the reverse-mode scalar sensitivity
/// \param[inout] fwd_sens - map of linear forms that will assemble the
/// forward-mode sensitivity
/// \param[inout] fwd_scalar_sens - map of nonlinear forms that will assemble
/// the forward-mode scalar sensitivity
/// \param[in] attr_marker - optional list of element attributes the sensitivity
/// integrators should be used on
template <typename T>
inline void addBdrSensitivityIntegrator(
    T &primal_integ,
    std::map<std::string, FiniteElementState> &fields,
    std::map<std::string, mfem::ParLinearForm> &rev_sens,
    std::map<std::string, mfem::ParNonlinearForm> &rev_scalar_sens,
    std::map<std::string, mfem::ParLinearForm> &fwd_sens,
    std::map<std::string, mfem::ParNonlinearForm> &fwd_scalar_sens,
    mfem::Array<int> *attr_marker = nullptr,
    std::string adjoint_name = "adjoint")
{ }

/// Function meant to be overloaded to allow residual sensitivity integrators
/// to be associated with the forward version of the integrator
/// \param[in] primal_integ - integrator used in forward evaluation
/// \param[in] fields - map of fields solver depends on
/// \param[inout] rev_sens - map of linear forms that will assemble the
/// reverse-mode sensitivity
/// \param[inout] rev_scalar_sens - map of nonlinear forms that will assemble
/// the reverse-mode scalar sensitivity
/// \param[inout] fwd_sens - map of linear forms that will assemble the
/// forward-mode sensitivity
/// \param[inout] fwd_scalar_sens - map of nonlinear forms that will assemble
/// the forward-mode scalar sensitivity
/// \param[in] attr_marker - optional list of element attributes the sensitivity
/// integrators should be used on
template <typename T>
inline void addInternalBoundarySensitivityIntegrator(
    T &primal_integ,
    std::map<std::string, FiniteElementState> &fields,
    std::map<std::string, mfem::ParLinearForm> &rev_sens,
    std::map<std::string, mfem::ParNonlinearForm> &rev_scalar_sens,
    std::map<std::string, mfem::ParLinearForm> &fwd_sens,
    std::map<std::string, mfem::ParNonlinearForm> &fwd_scalar_sens,
    mfem::Array<int> *attr_marker = nullptr,
    std::string adjoint_name = "adjoint")
{ }

/// Function meant to be overloaded to allow output sensitivity integrators
/// to be associated with the forward version of the integrator
/// \param[in] primal_integ - integrator used in forward evaluation
/// \param[in] fields - map of fields solver depends on
/// \param[inout] output_sens - map of linear forms that will assemble the
/// output partial derivatives wrt to fields
/// \param[inout] output_scalar_sens - map of nonlinear forms that will
/// assemble the output partial derivatives wrt to scalars
/// \param[in] attr_marker - optional list of element attributes the sensitivity
/// integrators should be used on
template <typename T>
inline void addDomainSensitivityIntegrator(
    T &primal_integ,
    std::map<std::string, FiniteElementState> &fields,
    std::map<std::string, mfem::ParLinearForm> &output_sens,
    std::map<std::string, mfem::ParNonlinearForm> &output_scalar_sens,
    mfem::Array<int> *attr_marker = nullptr,
    std::string state_name = "state")
{ }

/// Function meant to be overloaded to allow output sensitivity integrators
/// to be associated with the forward version of the integrator
/// \param[in] primal_integ - integrator used in forward evaluation
/// \param[in] fields - map of fields solver depends on
/// \param[inout] output_sens - map of linear forms that will assemble the
/// output partial derivatives wrt to fields
/// \param[inout] output_scalar_sens - map of nonlinear forms that will
/// assemble the output partial derivatives wrt to scalars
template <typename T>
inline void addInteriorFaceSensitivityIntegrator(
    T &primal_integ,
    std::map<std::string, FiniteElementState> &fields,
    std::map<std::string, mfem::ParLinearForm> &output_sens,
    std::map<std::string, mfem::ParNonlinearForm> &output_scalar_sens,
    std::string state_name = "state")
{ }

/// Function meant to be overloaded to allow output sensitivity integrators
/// to be associated with the forward version of the integrator
/// \param[in] primal_integ - integrator used in forward evaluation
/// \param[in] fields - map of fields solver depends on
/// \param[inout] output_sens - map of linear forms that will assemble the
/// output partial derivatives wrt to fields
/// \param[inout] output_scalar_sens - map of nonlinear forms that will
/// assemble the output partial derivatives wrt to scalars
/// \param[in] attr_marker - optional list of element attributes the sensitivity
/// integrators should be used on
template <typename T>
inline void addBdrSensitivityIntegrator(
    T &primal_integ,
    std::map<std::string, FiniteElementState> &fields,
    std::map<std::string, mfem::ParLinearForm> &output_sens,
    std::map<std::string, mfem::ParNonlinearForm> &output_scalar_sens,
    mfem::Array<int> *attr_marker,
    std::string state_name = "state")
{ }

/// Function meant to be overloaded to allow residual sensitivity integrators
/// to be associated with the forward version of the integrator
/// \param[in] primal_integ - integrator used in forward evaluation
/// \param[in] fields - map of fields solver depends on
/// \param[inout] output_sens - map of linear forms that will assemble the
/// output partial derivatives wrt to fields
/// \param[inout] output_scalar_sens - map of nonlinear forms that will
/// assemble the output partial derivatives wrt to scalars
/// \param[in] attr_marker - optional list of element attributes the sensitivity
/// integrators should be used on
template <typename T>
inline void addInternalBoundarySensitivityIntegrator(
    T &primal_integ,
    std::map<std::string, FiniteElementState> &fields,
    std::map<std::string, mfem::ParLinearForm> &output_sens,
    std::map<std::string, mfem::ParNonlinearForm> &output_scalar_sens,
    mfem::Array<int> *attr_marker = nullptr,
    std::string state_name = "state")
{ }

}  // namespace miso

#endif
