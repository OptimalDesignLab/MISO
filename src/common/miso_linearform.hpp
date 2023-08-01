#ifndef MISO_LINEAR_FORM
#define MISO_LINEAR_FORM

#include <map>
#include <unordered_map>
#include <vector>

#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "miso_input.hpp"
#include "miso_integrator.hpp"

namespace miso
{
class MISOLinearForm final
{
public:
   /// Assemble the linear form on the true dofs and add it to tv
   friend void addLoad(MISOLinearForm &load, mfem::Vector &tv);

   /// Set inputs in all integrators used by the linear form
   friend void setInputs(MISOLinearForm &load, const MISOInputs &inputs);

   /// Set options in all integrators used by the linear form
   friend void setOptions(MISOLinearForm &load, const nlohmann::json &options);

   /// Assemble the linear form's sensitivity to a scalar and contract it with
   /// load_bar
   friend double vectorJacobianProduct(MISOLinearForm &load,
                                       const mfem::HypreParVector &load_bar,
                                       const std::string &wrt);

   friend void vectorJacobianProduct(MISOLinearForm &load,
                                     const mfem::HypreParVector &load_bar,
                                     const std::string &wrt,
                                     mfem::HypreParVector &wrt_bar);

   /// Adds domain integrator to linear form
   /// \param[in] integrator - linear form integrator for domain
   /// \tparam T - type of integrator, used for constructing MISOIntegrator
   /// \note Assumes ownership of integrator
   template <typename T>
   void addDomainIntegrator(T *integrator);

   /// Adds boundary integrator to linear form
   /// \param[in] integrator - linear form integrator for boundary
   /// \tparam T - type of integrator, used for constructing MISOIntegrator
   /// \note Assumes ownership of integrator
   template <typename T>
   void addBoundaryIntegrator(T *integrator);

   /// Adds boundary integrator to linear form restricted to the given boundary
   /// attributes.
   /// \param[in] integrator - linear form integrator for boundary
   /// \param[in] bdr_attr_marker - boundary attributes for integrator
   /// \tparam T - type of integrator, used for constructing MISOIntegrator
   /// \note Assumes ownership of integrator
   /// \note The array bdr_attr_marker is copied
   template <typename T>
   void addBoundaryIntegrator(T *integrator, mfem::Array<int> &bdr_attr_marker);

   /// Adds boundary face integrator to linear form
   /// \param[in] integrator - face linear form integrator for boundary
   /// \tparam T - type of integrator, used for constructing MISOIntegrator
   /// \note Assumes ownership of integrator
   template <typename T>
   void addBdrFaceIntegrator(T *integrator);

   /// Adds boundary face integrator to linear form restricted to the given
   /// boundary attributes.
   /// \param[in] integrator - face linear form integrator for boundary
   /// \param[in] bdr_attr_marker - boundary attributes for integrator
   /// \tparam T - type of integrator, used for constructing MISOIntegrator
   /// \note Assumes ownership of integrator
   /// \note The array bdr_attr_marker is copied
   template <typename T>
   void addBdrFaceIntegrator(T *integrator, mfem::Array<int> &bdr_attr_marker);

   MISOLinearForm(
       mfem::ParFiniteElementSpace &pfes,
       std::unordered_map<std::string, mfem::ParGridFunction> &fields)
    : lf(&pfes), scratch(&pfes), lf_fields(&fields)
   {
      if (lf_fields->count("adjoint") == 0)
      {
         lf_fields->emplace("adjoint", &pfes);
      }
   }

private:
   /// underlying linear form object
   mfem::ParLinearForm lf;
   /// work vector
   mfem::HypreParVector scratch;

   /// Collection of integrators to be applied.
   std::vector<MISOIntegrator> integs;
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
void MISOLinearForm::addDomainIntegrator(T *integrator)
{
   integs.emplace_back(*integrator);
   lf.AddDomainIntegrator(integrator);
   miso::addSensitivityIntegrator(*integrator, *lf_fields, sens, scalar_sens);
}

template <typename T>
void MISOLinearForm::addBoundaryIntegrator(T *integrator)
{
   integs.emplace_back(*integrator);
   lf.AddBoundaryIntegrator(integrator);
   miso::addSensitivityIntegrator(*integrator, *lf_fields, sens, scalar_sens);
}

template <typename T>
void MISOLinearForm::addBoundaryIntegrator(T *integrator,
                                           mfem::Array<int> &bdr_attr_marker)
{
   integs.emplace_back(*integrator);
   bdr_marker.emplace_back(bdr_attr_marker);
   lf.AddBoundaryIntegrator(integrator, bdr_marker.back());
   miso::addSensitivityIntegrator(*integrator, *lf_fields, sens, scalar_sens);
}

template <typename T>
void MISOLinearForm::addBdrFaceIntegrator(T *integrator)
{
   integs.emplace_back(*integrator);
   lf.AddBdrFaceIntegrator(integrator);
   miso::addSensitivityIntegrator(*integrator, *lf_fields, sens, scalar_sens);
}

template <typename T>
void MISOLinearForm::addBdrFaceIntegrator(T *integrator,
                                          mfem::Array<int> &bdr_attr_marker)
{
   integs.emplace_back(*integrator);
   bdr_marker.emplace_back(bdr_attr_marker);
   lf.AddBdrFaceIntegrator(integrator, bdr_marker.back());
   miso::addSensitivityIntegrator(*integrator, *lf_fields, sens, scalar_sens);
}

}  // namespace miso

#endif
