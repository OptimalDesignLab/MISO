#ifndef MISO_LINEAR_FORM
#define MISO_LINEAR_FORM

#include <cstddef>
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
   friend int getSize(const MISOLinearForm &load);

   /// Assemble the linear form on the true dofs and add it to tv
   friend void addLoad(MISOLinearForm &load, mfem::Vector &tv);

   /// Set inputs in all integrators used by the linear form
   friend void setInputs(MISOLinearForm &load, const MISOInputs &inputs);

   /// Set options in all integrators used by the linear form
   friend void setOptions(MISOLinearForm &load, const nlohmann::json &options);

   friend double jacobianVectorProduct(MISOLinearForm &load,
                                       const mfem::Vector &wrt_dot,
                                       const std::string &wrt);

   friend void jacobianVectorProduct(MISOLinearForm &load,
                                     const mfem::Vector &wrt_dot,
                                     const std::string &wrt,
                                     mfem::Vector &res_dot);

   friend double vectorJacobianProduct(MISOLinearForm &load,
                                       const mfem::Vector &load_bar,
                                       const std::string &wrt);

   friend void vectorJacobianProduct(MISOLinearForm &load,
                                     const mfem::Vector &load_bar,
                                     const std::string &wrt,
                                     mfem::Vector &wrt_bar);

   /// Adds domain integrator to linear form
   /// \param[in] integrator - linear form integrator for domain
   /// \tparam T - type of integrator, used for constructing MISOIntegrator
   /// \note Assumes ownership of integrator
   template <typename T>
   void addDomainIntegrator(T *integrator);

   /// Adds domain integrator restricted to certain elements specified by the
   /// attributes listed in @a attr_marker to linear form
   /// \param[in] integrator - integrator to add to functional
   /// \param[in] bdr_attr_marker - lists element attributes this integrator
   /// should be used on
   /// \tparam T - type of integrator, used for constructing MISOIntegrator
   template <typename T>
   void addDomainIntegrator(T *integrator, const std::vector<int> &attr_marker);

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
   void addBoundaryIntegrator(T *integrator,
                              const std::vector<int> &bdr_attr_marker);

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
   void addBdrFaceIntegrator(T *integrator,
                             const std::vector<int> &bdr_attr_marker);

   /// Adds internal boundary face integrator to linear form
   /// \param[in] integrator - face linear form integrator for internal boundary
   /// \tparam T - type of integrator, used for constructing MISOIntegrator
   /// \note Assumes ownership of integrator
   template <typename T>
   void addInternalBoundaryFaceIntegrator(T *integrator);

   /// Adds internal boundary face integrator to linear form restricted to the
   /// given boundary attributes.
   /// \param[in] integrator - face linear form integrator for internal boundary
   /// \param[in] bdr_attr_marker - internal boundary attributes for integrator
   /// \tparam T - type of integrator, used for constructing MISOIntegrator
   /// \note Assumes ownership of integrator
   /// \note The array bdr_attr_marker is copied
   template <typename T>
   void addInternalBoundaryFaceIntegrator(
       T *integrator,
       const std::vector<int> &bdr_attr_marker);

   const mfem::Array<int> &getEssentialDofs() const { return ess_tdof_list; }

   MISOLinearForm(mfem::ParFiniteElementSpace &pfes,
                  std::map<std::string, FiniteElementState> &fields,
                  std::string adjoint_name = "adjoint")
    : lf(&pfes), scratch(0), lf_fields(&fields), adjoint_name(adjoint_name)
   {
      if (lf_fields->count(adjoint_name) == 0)
      {
         lf_fields->emplace(
             adjoint_name,
             FiniteElementState(*pfes.GetParMesh(), pfes, adjoint_name));
      }
   }

   MISOLinearForm(
       mfem::ParFiniteElementSpace &pfes,
       std::unordered_map<std::string, mfem::ParGridFunction> &fields)
    : lf(&pfes), scratch(pfes.GetTrueVSize()), lf_fields(nullptr)
   { }

private:
   /// underlying linear form object
   mfem::ParLinearForm lf;
   /// essential tdofs
   mfem::Array<int> ess_tdof_list;

   /// work vector
   mfem::Vector scratch;

   /// Collection of integrators to be applied.
   std::vector<MISOIntegrator> integs;

   /// Collection of element attribute markers for domain integrators
   std::list<mfem::Array<int>> domain_markers;

   /// Collection of boundary markers for boundary integrators
   std::list<mfem::Array<int>> bdr_markers;

   /// map of external fields the linear form depends on
   std::map<std::string, FiniteElementState> *lf_fields;

   /// name of the field that holds the adjoint for this linear form
   std::string adjoint_name;

   /// map of linear forms that will compute (dF / dfield) * field_dot
   /// for each field the nonlinear form depends on
   std::map<std::string, mfem::ParLinearForm> fwd_sens;
   /// map of nonlinear forms that will compute (dF / dscalar) * scalar_dot
   /// for each scalar the nonlinear form depends on
   std::map<std::string, mfem::ParNonlinearForm> fwd_scalar_sens;

   /// map of linear forms that will compute psi^T (dF / dfield)
   /// for each field the nonlinear form depends on
   std::map<std::string, mfem::ParLinearForm> rev_sens;
   /// map of nonlinear forms that will compute psi^T (dF / dscalar)
   /// for each scalar the nonlinear form depends on
   std::map<std::string, mfem::ParNonlinearForm> rev_scalar_sens;
};

template <typename T>
void MISOLinearForm::addDomainIntegrator(T *integrator)
{
   integs.emplace_back(*integrator);
   lf.AddDomainIntegrator(integrator);
   addDomainSensitivityIntegrator(*integrator,
                                  *lf_fields,
                                  rev_sens,
                                  rev_scalar_sens,
                                  fwd_sens,
                                  fwd_scalar_sens,
                                  nullptr,
                                  adjoint_name);
}

template <typename T>
void MISOLinearForm::addDomainIntegrator(T *integrator,
                                         const std::vector<int> &attr_marker)
{
   integs.emplace_back(*integrator);

   auto mesh_attr_size = lf.ParFESpace()->GetMesh()->attributes.Max();
   auto &marker = domain_markers.emplace_back(mesh_attr_size);
   attrVecToArray(attr_marker, marker);
   lf.AddDomainIntegrator(integrator, marker);
   addDomainSensitivityIntegrator(*integrator,
                                  *lf_fields,
                                  rev_sens,
                                  rev_scalar_sens,
                                  fwd_sens,
                                  fwd_scalar_sens,
                                  &marker,
                                  adjoint_name);
}

template <typename T>
void MISOLinearForm::addBoundaryIntegrator(T *integrator)
{
   integs.emplace_back(*integrator);
   lf.AddBoundaryIntegrator(integrator);
   addBdrSensitivityIntegrator(*integrator,
                               *lf_fields,
                               rev_sens,
                               rev_scalar_sens,
                               fwd_sens,
                               fwd_scalar_sens,
                               nullptr,
                               adjoint_name);
}

template <typename T>
void MISOLinearForm::addBoundaryIntegrator(
    T *integrator,
    const std::vector<int> &bdr_attr_marker)
{
   integs.emplace_back(*integrator);
   auto mesh_attr_size = lf.ParFESpace()->GetMesh()->bdr_attributes.Max();
   auto &marker = bdr_markers.emplace_back(mesh_attr_size);
   attrVecToArray(bdr_attr_marker, marker);

   lf.AddBoundaryIntegrator(integrator, marker);
   addBdrSensitivityIntegrator(*integrator,
                               *lf_fields,
                               rev_sens,
                               rev_scalar_sens,
                               fwd_sens,
                               fwd_scalar_sens,
                               &marker,
                               adjoint_name);
}

template <typename T>
void MISOLinearForm::addBdrFaceIntegrator(T *integrator)
{
   integs.emplace_back(*integrator);
   lf.AddBdrFaceIntegrator(integrator);
   addBdrSensitivityIntegrator(*integrator,
                               *lf_fields,
                               rev_sens,
                               rev_scalar_sens,
                               fwd_sens,
                               fwd_scalar_sens,
                               nullptr,
                               adjoint_name);
}

template <typename T>
void MISOLinearForm::addBdrFaceIntegrator(
    T *integrator,
    const std::vector<int> &bdr_attr_marker)
{
   integs.emplace_back(*integrator);
   auto mesh_attr_size = lf.ParFESpace()->GetMesh()->bdr_attributes.Max();
   auto &marker = bdr_markers.emplace_back(mesh_attr_size);
   attrVecToArray(bdr_attr_marker, marker);

   lf.AddBdrFaceIntegrator(integrator, marker);
   addBdrSensitivityIntegrator(*integrator,
                               *lf_fields,
                               rev_sens,
                               rev_scalar_sens,
                               fwd_sens,
                               fwd_scalar_sens,
                               &marker,
                               adjoint_name);
}

template <typename T>
void MISOLinearForm::addInternalBoundaryFaceIntegrator(T *integrator)
{
   integs.emplace_back(*integrator);
   lf.AddInternalBoundaryFaceIntegrator(integrator);
   addInternalBoundarySensitivityIntegrator(*integrator,
                                            *lf_fields,
                                            rev_sens,
                                            rev_scalar_sens,
                                            fwd_sens,
                                            fwd_scalar_sens,
                                            nullptr,
                                            adjoint_name);
}

template <typename T>
void MISOLinearForm::addInternalBoundaryFaceIntegrator(
    T *integrator,
    const std::vector<int> &bdr_attr_marker)
{
   integs.emplace_back(*integrator);
   auto mesh_attr_size = lf.ParFESpace()->GetMesh()->bdr_attributes.Max();
   auto &marker = bdr_markers.emplace_back(mesh_attr_size);
   attrVecToArray(bdr_attr_marker, marker);

   lf.AddInternalBoundaryFaceIntegrator(integrator, marker);
   addInternalBoundarySensitivityIntegrator(*integrator,
                                            *lf_fields,
                                            rev_sens,
                                            rev_scalar_sens,
                                            fwd_sens,
                                            fwd_scalar_sens,
                                            &marker,
                                            adjoint_name);
}

}  // namespace miso

#endif
