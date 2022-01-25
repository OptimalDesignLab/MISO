#ifndef MACH_LINEAR_FORM
#define MACH_LINEAR_FORM

#include <map>
#include <unordered_map>
#include <vector>

#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "mach_input.hpp"
#include "mach_integrator.hpp"

namespace mach
{
class MachLinearForm final
{
public:
   /// Assemble the linear form on the true dofs and add it to tv
   friend void addLoad(MachLinearForm &load, mfem::Vector &tv);

   /// Set inputs in all integrators used by the linear form
   friend void setInputs(MachLinearForm &load, const MachInputs &inputs);

   /// Set options in all integrators used by the linear form
   friend void setOptions(MachLinearForm &load, const nlohmann::json &options);

   /// Assemble the linear form's sensitivity to a scalar and contract it with
   /// load_bar
   friend double vectorJacobianProduct(MachLinearForm &load,
                                       const mfem::Vector &load_bar,
                                       const std::string &wrt);

   friend void vectorJacobianProduct(MachLinearForm &load,
                                     const mfem::Vector &load_bar,
                                     const std::string &wrt,
                                     mfem::Vector &wrt_bar);

   /// Adds domain integrator to linear form
   /// \param[in] integrator - linear form integrator for domain
   /// \tparam T - type of integrator, used for constructing MachIntegrator
   /// \note Assumes ownership of integrator
   template <typename T>
   void addDomainIntegrator(T *integrator);

   /// Adds boundary integrator to linear form
   /// \param[in] integrator - linear form integrator for boundary
   /// \tparam T - type of integrator, used for constructing MachIntegrator
   /// \note Assumes ownership of integrator
   template <typename T>
   void addBoundaryIntegrator(T *integrator);

   /// Adds boundary integrator to linear form restricted to the given boundary
   /// attributes.
   /// \param[in] integrator - linear form integrator for boundary
   /// \param[in] bdr_attr_marker - boundary attributes for integrator
   /// \tparam T - type of integrator, used for constructing MachIntegrator
   /// \note Assumes ownership of integrator
   /// \note The array bdr_attr_marker is copied
   template <typename T>
   void addBoundaryIntegrator(T *integrator, mfem::Array<int> &bdr_attr_marker);

   /// Adds boundary face integrator to linear form
   /// \param[in] integrator - face linear form integrator for boundary
   /// \tparam T - type of integrator, used for constructing MachIntegrator
   /// \note Assumes ownership of integrator
   template <typename T>
   void addBdrFaceIntegrator(T *integrator);

   /// Adds boundary face integrator to linear form restricted to the given
   /// boundary attributes.
   /// \param[in] integrator - face linear form integrator for boundary
   /// \param[in] bdr_attr_marker - boundary attributes for integrator
   /// \tparam T - type of integrator, used for constructing MachIntegrator
   /// \note Assumes ownership of integrator
   /// \note The array bdr_attr_marker is copied
   template <typename T>
   void addBdrFaceIntegrator(T *integrator, mfem::Array<int> &bdr_attr_marker);

   MachLinearForm(
       mfem::ParFiniteElementSpace &pfes,
       std::unordered_map<std::string, mfem::ParGridFunction> &fields)
    : lf(&pfes), scratch(pfes.GetTrueVSize()), lf_fields(&fields)
   {
      if (lf_fields->count("adjoint") == 0)
      {
         lf_fields->emplace("adjoint", &pfes);
      }
   }

private:
   /// underlying linear form object
   mfem::ParLinearForm lf;
   /// essential tdofs
   mfem::Array<int> ess_tdof_list;

   /// work vector
   mfem::Vector scratch;

   /// Collection of integrators to be applied.
   std::vector<MachIntegrator> integs;
   /// Collection of boundary markers for boundary integrators
   std::vector<mfem::Array<int>> bdr_marker;

   /// map of external fields the linear form depends on
   std::unordered_map<std::string, mfem::ParGridFunction> *lf_fields;

   /// map of linear forms that will compute d(psi^T F) / d(field)
   /// for each field the linear form depends on
   std::map<std::string, mfem::ParLinearForm> sens;
   /// map of nonlinear forms that will compute d(psi^T F) / d(scalar)
   /// for each scalar the linear form depends on
   std::map<std::string, mfem::ParNonlinearForm> scalar_sens;
};

template <typename T>
void MachLinearForm::addDomainIntegrator(T *integrator)
{
   integs.emplace_back(*integrator);
   lf.AddDomainIntegrator(integrator);
   mach::addSensitivityIntegrator(*integrator, *lf_fields, sens, scalar_sens);
}

template <typename T>
void MachLinearForm::addBoundaryIntegrator(T *integrator)
{
   integs.emplace_back(*integrator);
   lf.AddBoundaryIntegrator(integrator);
   mach::addSensitivityIntegrator(*integrator, *lf_fields, sens, scalar_sens);
}

template <typename T>
void MachLinearForm::addBoundaryIntegrator(T *integrator,
                                           mfem::Array<int> &bdr_attr_marker)
{
   integs.emplace_back(*integrator);
   bdr_marker.emplace_back(bdr_attr_marker);
   lf.AddBoundaryIntegrator(integrator, bdr_marker.back());
   mach::addSensitivityIntegrator(*integrator, *lf_fields, sens, scalar_sens);
}

template <typename T>
void MachLinearForm::addBdrFaceIntegrator(T *integrator)
{
   integs.emplace_back(*integrator);
   lf.AddBdrFaceIntegrator(integrator);
   mach::addSensitivityIntegrator(*integrator, *lf_fields, sens, scalar_sens);
}

template <typename T>
void MachLinearForm::addBdrFaceIntegrator(T *integrator,
                                          mfem::Array<int> &bdr_attr_marker)
{
   integs.emplace_back(*integrator);
   bdr_marker.emplace_back(bdr_attr_marker);
   lf.AddBdrFaceIntegrator(integrator, bdr_marker.back());
   mach::addSensitivityIntegrator(*integrator, *lf_fields, sens, scalar_sens);
}

}  // namespace mach

#endif
