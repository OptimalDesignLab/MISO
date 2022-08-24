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
   friend int getSize(const MachLinearForm &load);

   /// Assemble the linear form on the true dofs and add it to tv
   friend void addLoad(MachLinearForm &load, mfem::Vector &tv);

   /// Set inputs in all integrators used by the linear form
   friend void setInputs(MachLinearForm &load, const MachInputs &inputs);

   /// Set options in all integrators used by the linear form
   friend void setOptions(MachLinearForm &load, const nlohmann::json &options);

   friend double jacobianVectorProduct(MachLinearForm &load,
                                       const mfem::Vector &wrt_dot,
                                       const std::string &wrt);

   friend void jacobianVectorProduct(MachLinearForm &load,
                                     const mfem::Vector &wrt_dot,
                                     const std::string &wrt,
                                     mfem::Vector &res_dot);

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

   /// Adds domain integrator restricted to certain elements specified by the
   /// attributes listed in @a bdr_attr_marker to linear form
   /// \param[in] integrator - integrator to add to functional
   /// \param[in] bdr_attr_marker - lists element attributes this integrator
   /// should be used on
   /// \tparam T - type of integrator, used for constructing MachIntegrator
   template <typename T>
   void addDomainIntegrator(T *integrator, const std::vector<int> &attr_marker);

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

   MachLinearForm(mfem::ParFiniteElementSpace &pfes,
                  std::map<std::string, FiniteElementState> &fields)
    : lf(&pfes), scratch(0), lf_fields(&fields)
   {
      if (lf_fields->count("adjoint") == 0)
      {
         lf_fields->emplace(std::piecewise_construct,
                            std::make_tuple("adjoint"),
                            std::forward_as_tuple(*pfes.GetParMesh(), pfes));
      }
   }

   MachLinearForm(
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
   std::vector<MachIntegrator> integs;

   /// Collection of element attribute markers for domain integrators
   std::list<mfem::Array<int>> domain_markers;

   /// Collection of boundary markers for boundary integrators
   std::vector<mfem::Array<int>> bdr_marker;

   /// map of external fields the linear form depends on
   std::map<std::string, FiniteElementState> *lf_fields;

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
void MachLinearForm::addDomainIntegrator(T *integrator)
{
   integs.emplace_back(*integrator);
   lf.AddDomainIntegrator(integrator);
   addSensitivityIntegrator(*integrator,
                            *lf_fields,
                            rev_sens,
                            rev_scalar_sens,
                            fwd_sens,
                            fwd_scalar_sens);
}

template <typename T>
void MachLinearForm::addDomainIntegrator(T *integrator,
                                         const std::vector<int> &attr_marker)
{
   integs.emplace_back(*integrator);

   auto mesh_attr_size = lf.ParFESpace()->GetMesh()->attributes.Max();
   auto &marker = domain_markers.emplace_back(mesh_attr_size);
   attrVecToArray(attr_marker, marker);
   lf.AddDomainIntegrator(integrator, marker);
   addSensitivityIntegrator(*integrator,
                            *lf_fields,
                            rev_sens,
                            rev_scalar_sens,
                            fwd_sens,
                            fwd_scalar_sens);
}

template <typename T>
void MachLinearForm::addBoundaryIntegrator(T *integrator)
{
   integs.emplace_back(*integrator);
   lf.AddBoundaryIntegrator(integrator);
   addSensitivityIntegrator(*integrator,
                            *lf_fields,
                            rev_sens,
                            rev_scalar_sens,
                            fwd_sens,
                            fwd_scalar_sens);
}

template <typename T>
void MachLinearForm::addBoundaryIntegrator(T *integrator,
                                           mfem::Array<int> &bdr_attr_marker)
{
   integs.emplace_back(*integrator);
   bdr_marker.emplace_back(bdr_attr_marker);
   lf.AddBoundaryIntegrator(integrator, bdr_marker.back());
   addSensitivityIntegrator(*integrator,
                            *lf_fields,
                            rev_sens,
                            rev_scalar_sens,
                            fwd_sens,
                            fwd_scalar_sens);
}

template <typename T>
void MachLinearForm::addBdrFaceIntegrator(T *integrator)
{
   integs.emplace_back(*integrator);
   lf.AddBdrFaceIntegrator(integrator);
   addSensitivityIntegrator(*integrator,
                            *lf_fields,
                            rev_sens,
                            rev_scalar_sens,
                            fwd_sens,
                            fwd_scalar_sens);
}

template <typename T>
void MachLinearForm::addBdrFaceIntegrator(T *integrator,
                                          mfem::Array<int> &bdr_attr_marker)
{
   integs.emplace_back(*integrator);
   bdr_marker.emplace_back(bdr_attr_marker);
   lf.AddBdrFaceIntegrator(integrator, bdr_marker.back());
   addSensitivityIntegrator(*integrator,
                            *lf_fields,
                            rev_sens,
                            rev_scalar_sens,
                            fwd_sens,
                            fwd_scalar_sens);
}

}  // namespace mach

#endif
