#ifndef MACH_NONLINEAR_FORM
#define MACH_NONLINEAR_FORM

#include <vector>

#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "mach_input.hpp"
#include "mach_integrator.hpp"

namespace mach
{
class MachNonlinearForm final
{
public:
   /// Set inputs in all integrators used by the nonlinear form
   friend void setInputs(MachNonlinearForm &form, const MachInputs &inputs);

   friend void setOptions(MachNonlinearForm &form,
                          const nlohmann::json &options);

   /// Evaluate the nonlinear form using `inputs` and return result in `res_vec`
   friend void evaluate(MachNonlinearForm &form,
                        const MachInputs &inputs,
                        mfem::Vector &res_vec);

   /// Adds the given domain integrator to the nonlinear form
   /// \param[in] integrator - nonlinear form integrator for domain
   /// \tparam T - type of integrator, used for constructing MachIntegrator
   /// \note Assumes ownership of integrator
   template <typename T>
   void addDomainIntegrator(T *integrator);

   /// Adds the given boundary face integrator to the nonlinear form
   /// \param[in] integrator - face nonlinear form integrator for boundary
   /// \tparam T - type of integrator, used for constructing MachIntegrator
   /// \note Assumes ownership of integrator
   template <typename T>
   void addBdrFaceIntegrator(T *integrator);

   /// Adds given boundary face integrator, restricted to the given boundary
   /// attributes.
   /// \param[in] integrator - face nonlinear form integrator for boundary
   /// \param[in] bdr_attr_marker - boundary attributes for integrator
   /// \tparam T - type of integrator, used for constructing MachIntegrator
   /// \note Assumes ownership of integrator
   /// \note The array bdr_attr_marker is copied
   template <typename T>
   void addBdrFaceIntegrator(T *integrator, std::vector<int> bdr_attr_marker);

   /// Adds the given interior face integrator to the nonlinear form
   /// \param[in] integrator - face nonlinear form integrator for interfaces
   /// \tparam T - type of integrator, used for constructing MachIntegrator
   /// \note Assumes ownership of integrator
   template <typename T>
   void addInteriorFaceIntegrator(T *integrator);

   /// Constructor for nonlinear form types
   /// \param[in] pfes - FEM space for the state (and possibly the adjoint)
   /// \param[in] fields - map of grid functions
   MachNonlinearForm(
       mfem::ParFiniteElementSpace &pfes,
       std::unordered_map<std::string, mfem::ParGridFunction> &fields)
    : nf(&pfes), scratch(&pfes), nf_fields(&fields)
   {
      if (nf_fields->count("adjoint") == 0)
      {
         nf_fields->emplace("adjoint", &pfes);
      }
   }

private:
   /// underlying nonlinear form object
   mfem::ParNonlinearForm nf;
   /// work vector (needed?)
   mfem::HypreParVector scratch;

   /// Collection of integrators to be applied.
   std::vector<MachIntegrator> integs;
   /// Collection of boundary markers for boundary integrators
   std::vector<mfem::Array<int>> bdr_markers;

   /// map of external fields that the nonlinear form depends on
   std::unordered_map<std::string, mfem::ParGridFunction> *nf_fields;

   /// map of linear forms that will compute d(psi^T F) / d(field)
   /// for each field the nonlinear form depends on
   // std::map<std::string, mfem::ParLinearForm> sens;
   /// map of nonlinear forms that will compute d(psi^T F) / d(scalar)
   /// for each scalar the nonlinear form depends on
   // std::map<std::string, mfem::ParNonlinearForm> scalar_sens;
};

template <typename T>
void MachNonlinearForm::addDomainIntegrator(T *integrator)
{
   integs.emplace_back(*integrator);
   nf.AddDomainIntegrator(integrator);
   // mach::addSensitivityIntegrator(*integrator, *nf_fields, sens,
   // scalar_sens);
}

template <typename T>
void MachNonlinearForm::addBdrFaceIntegrator(T *integrator)
{
   integs.emplace_back(*integrator);
   nf.AddBdrFaceIntegrator(integrator);
   // mach::addSensitivityIntegrator(*integrator, *nf_fields, sens,
   // scalar_sens);
}

template <typename T>
void MachNonlinearForm::addBdrFaceIntegrator(T *integrator,
                                             std::vector<int> bdr_attr_marker)
{
   integs.emplace_back(*integrator);
   bdr_markers.emplace_back(bdr_attr_marker.size());
   bdr_markers.back().Assign(bdr_attr_marker.data());
   nf.AddBdrFaceIntegrator(integrator, bdr_markers.back());
   // mach::addSensitivityIntegrator(*integrator, *nf_fields, sens,
   // scalar_sens);
}

template <typename T>
void MachNonlinearForm::addInteriorFaceIntegrator(T *integrator)
{
   integs.emplace_back(*integrator);
   nf.AddInteriorFaceIntegrator(integrator);
   // mach::addSensitivityIntegrator(*integrator, *nf_fields, sens,
   // scalar_sens);
}

}  // namespace mach

#endif  // MACH_NONLINEAR_FORM