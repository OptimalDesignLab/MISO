#ifndef MISO_NONLINEAR_FORM
#define MISO_NONLINEAR_FORM

#include <memory>
#include <vector>
#include <list>

#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "miso_input.hpp"
#include "miso_integrator.hpp"

namespace miso
{
class MISONonlinearForm final
{
public:
   /// Get the size of the nonlinear form (number of equations/unknowns)
   friend int getSize(const MISONonlinearForm &form);

   /// Set inputs in all integrators used by the nonlinear form
   friend void setInputs(MISONonlinearForm &form, const MISOInputs &inputs);

   friend void setOptions(MISONonlinearForm &form,
                          const nlohmann::json &options);

   /// Calls GetEnergy() for the underlying form using "state" in inputs.
   friend double calcFormOutput(MISONonlinearForm &form,
                                const MISOInputs &inputs);

   /// Evaluate the nonlinear form using `inputs` and return result in `res_vec`
   friend void evaluate(MISONonlinearForm &form,
                        const MISOInputs &inputs,
                        mfem::Vector &res_vec);

   friend void linearize(MISONonlinearForm &form, const MISOInputs &inputs);

   /// Compute Jacobian of `form` with respect to `wrt` and return
   friend mfem::Operator &getJacobian(MISONonlinearForm &form,
                                      const MISOInputs &inputs,
                                      const std::string &wrt);

   /// Compute transpose of Jacobian of `form` with respect to `wrt` and return
   friend mfem::Operator &getJacobianTranspose(MISONonlinearForm &form,
                                               const MISOInputs &inputs,
                                               const std::string &wrt);

   friend void setUpAdjointSystem(MISONonlinearForm &form,
                                  mfem::Solver &adj_solver,
                                  const MISOInputs &inputs,
                                  mfem::Vector &state_bar,
                                  mfem::Vector &adjoint);

   friend void finalizeAdjointSystem(MISONonlinearForm &form,
                                     mfem::Solver &adj_solver,
                                     const MISOInputs &inputs,
                                     mfem::Vector &state_bar,
                                     mfem::Vector &adjoint);

   friend double jacobianVectorProduct(MISONonlinearForm &form,
                                       const mfem::Vector &wrt_dot,
                                       const std::string &wrt);

   friend void jacobianVectorProduct(MISONonlinearForm &form,
                                     const mfem::Vector &wrt_dot,
                                     const std::string &wrt,
                                     mfem::Vector &res_dot);

   friend double vectorJacobianProduct(MISONonlinearForm &form,
                                       const mfem::Vector &res_bar,
                                       const std::string &wrt);

   friend void vectorJacobianProduct(MISONonlinearForm &form,
                                     const mfem::Vector &res_bar,
                                     const std::string &wrt,
                                     mfem::Vector &wrt_bar);

   /// Adds the given domain integrator to the nonlinear form
   /// \param[in] integrator - nonlinear form integrator for domain
   /// \tparam T - type of integrator, used for constructing MISOIntegrator
   /// \note Assumes ownership of integrator
   template <typename T>
   void addDomainIntegrator(T *integrator);

   /// Adds the given domain integrator to the nonlinear form, restricted to
   /// the given domain attributes
   /// \param[in] integrator - nonlinear form integrator for domain
   /// \param[in] attr_marker - domain attributes for integrator
   /// \tparam T - type of integrator, used for constructing MISOIntegrator
   /// \note Assumes ownership of integrator
   template <typename T>
   void addDomainIntegrator(T *integrator, const std::vector<int> &attr_marker);

   /// Adds the given interior face integrator to the nonlinear form
   /// \param[in] integrator - face nonlinear form integrator for interfaces
   /// \tparam T - type of integrator, used for constructing MISOIntegrator
   /// \note Assumes ownership of integrator
   template <typename T>
   void addInteriorFaceIntegrator(T *integrator);

   /// Adds the given boundary face integrator to the nonlinear form
   /// \param[in] integrator - face nonlinear form integrator for boundary
   /// \tparam T - type of integrator, used for constructing MISOIntegrator
   /// \note Assumes ownership of integrator
   template <typename T>
   void addBdrFaceIntegrator(T *integrator);

   /// Adds given boundary face integrator, restricted to the given boundary
   /// attributes.
   /// \param[in] integrator - face nonlinear form integrator for boundary
   /// \param[in] bdr_attr_marker - boundary attributes for integrator
   /// \tparam T - type of integrator, used for constructing MISOIntegrator
   /// \note Assumes ownership of integrator
   /// \note The array bdr_attr_marker is copied
   template <typename T>
   void addBdrFaceIntegrator(T *integrator,
                             const std::vector<int> &bdr_attr_marker);

   /// Adds internal boundary face integrator to nonlinear form
   /// \param[in] integrator - integrator for internal boundary faces
   /// \tparam T - type of integrator, used for constructing MISOIntegrator
   /// \note Assumes ownership of integrator
   template <typename T>
   void addInternalBoundaryFaceIntegrator(T *integrator);

   /// Adds internal boundary face integrator to nonlinear form restricted to
   /// the given boundary attributes.
   /// \param[in] integrator - integrator for internal boundary faces
   /// \param[in] bdr_attr_marker - internal boundary attributes for integrator
   /// \tparam T - type of integrator, used for constructing MISOIntegrator
   /// \note Assumes ownership of integrator
   /// \note The array bdr_attr_marker is copied
   template <typename T>
   void addInternalBoundaryFaceIntegrator(
       T *integrator,
       const std::vector<int> &bdr_attr_marker);

   const mfem::Array<int> &getEssentialDofs() const
   {
      return nf.GetEssentialTrueDofs();
   }

   /// Constructor for nonlinear form types
   /// \param[in] pfes - FEM space for the state (and possibly the adjoint)
   /// \param[in] fields - map of grid functions
   MISONonlinearForm(mfem::ParFiniteElementSpace &pfes,
                     std::map<std::string, FiniteElementState> &fields,
                     std::string adjoint_name = "adjoint")
    : nf(&pfes),
      scratch(0),
      nf_fields(fields),
      adjoint_name(adjoint_name),
      jac(mfem::Operator::Hypre_ParCSR),
      jac_e(mfem::Operator::Hypre_ParCSR)
   {
      if (nf_fields.count(adjoint_name) == 0)
      {
         // nf_fields.emplace("adjoint", {*pfes.GetParMesh(), pfes});
         nf_fields.emplace(
             adjoint_name,
             FiniteElementState(*pfes.GetParMesh(), pfes, adjoint_name));
      }
   }

private:
   /// underlying nonlinear form object
   mfem::ParNonlinearForm nf;
   /// work vector
   mfem::Vector scratch;

   /// Essential boundary marker
   mfem::Array<int> ess_bdr;

   /// Collection of integrators to be applied.
   std::vector<MISOIntegrator> integs;
   /// Collection of  markers for domain integrators
   std::list<mfem::Array<int>> domain_markers;
   /// Collection of boundary markers for boundary integrators
   std::list<mfem::Array<int>> bdr_markers;
   /// map of external fields that the nonlinear form depends on
   std::map<std::string, FiniteElementState> &nf_fields;

   /// name of the field that holds the adjoint for this nonlinear form
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

   /// Holds reference to the Jacobian (owned elsewhere)
   mfem::OperatorHandle jac;
   /// Holds eliminated entries from the Jacobian
   mfem::OperatorHandle jac_e;

   /// Holds the transpose of the Jacobian, needed for solving for the adjoint
   std::unique_ptr<mfem::Operator> jac_trans;
   /// Holds the transpose of the eliminated entries from the Jacobian,
   /// needed for solving for the adjoint
   std::unique_ptr<mfem::Operator> jac_e_trans;

   /// Holds the sum of jac_trans and jac_e_trans
   /// This is given to the adjoint linear solver
   std::unique_ptr<mfem::Operator> adjoint_jac_trans;

   mfem::Vector adj_work1, adj_work2;
};

template <typename T>
void MISONonlinearForm::addDomainIntegrator(T *integrator)
{
   integs.emplace_back(*integrator);
   nf.AddDomainIntegrator(integrator);
   addDomainSensitivityIntegrator(*integrator,
                                  nf_fields,
                                  rev_sens,
                                  rev_scalar_sens,
                                  fwd_sens,
                                  fwd_scalar_sens,
                                  nullptr,
                                  adjoint_name);
}

template <typename T>
void MISONonlinearForm::addDomainIntegrator(
    T *integrator,
    const std::vector<int> &bdr_attr_marker)
{
   integs.emplace_back(*integrator);
   auto mesh_attr_size = nf.ParFESpace()->GetMesh()->bdr_attributes.Max();
   auto &marker = domain_markers.emplace_back(mesh_attr_size);
   attrVecToArray(bdr_attr_marker, marker);
   nf.AddDomainIntegrator(integrator);
   addDomainSensitivityIntegrator(*integrator,
                                  nf_fields,
                                  rev_sens,
                                  rev_scalar_sens,
                                  fwd_sens,
                                  fwd_scalar_sens,
                                  &marker,
                                  adjoint_name);
}

template <typename T>
void MISONonlinearForm::addInteriorFaceIntegrator(T *integrator)
{
   integs.emplace_back(*integrator);
   nf.AddInteriorFaceIntegrator(integrator);
   addInteriorFaceSensitivityIntegrator(*integrator,
                                        nf_fields,
                                        rev_sens,
                                        rev_scalar_sens,
                                        fwd_sens,
                                        fwd_scalar_sens,
                                        adjoint_name);
}

template <typename T>
void MISONonlinearForm::addBdrFaceIntegrator(T *integrator)
{
   integs.emplace_back(*integrator);
   nf.AddBdrFaceIntegrator(integrator);
   addBdrSensitivityIntegrator(*integrator,
                               nf_fields,
                               rev_sens,
                               rev_scalar_sens,
                               fwd_sens,
                               fwd_scalar_sens,
                               nullptr,
                               adjoint_name);
}

template <typename T>
void MISONonlinearForm::addBdrFaceIntegrator(
    T *integrator,
    const std::vector<int> &bdr_attr_marker)
{
   integs.emplace_back(*integrator);

   auto mesh_attr_size = nf.ParFESpace()->GetMesh()->bdr_attributes.Max();
   auto &marker = bdr_markers.emplace_back(mesh_attr_size);
   attrVecToArray(bdr_attr_marker, marker);

   nf.AddBdrFaceIntegrator(integrator, marker);
   addBdrSensitivityIntegrator(*integrator,
                               nf_fields,
                               rev_sens,
                               rev_scalar_sens,
                               fwd_sens,
                               fwd_scalar_sens,
                               &marker,
                               adjoint_name);
}

template <typename T>
void MachNonlinearForm::addInternalBoundaryFaceIntegrator(T *integrator)
{
   integs.emplace_back(*integrator);
   nf.AddInternalBoundaryFaceIntegrator(integrator);
   addInternalBoundarySensitivityIntegrator(*integrator,
                                            nf_fields,
                                            rev_sens,
                                            rev_scalar_sens,
                                            fwd_sens,
                                            fwd_scalar_sens,
                                            nullptr,
                                            adjoint_name);
}

template <typename T>
void MachNonlinearForm::addInternalBoundaryFaceIntegrator(
    T *integrator,
    const std::vector<int> &bdr_attr_marker)
{
   integs.emplace_back(*integrator);
   auto mesh_attr_size = nf.ParFESpace()->GetMesh()->bdr_attributes.Max();
   auto &marker = bdr_markers.emplace_back(mesh_attr_size);
   attrVecToArray(bdr_attr_marker, marker);

   nf.AddInternalBoundaryFaceIntegrator(integrator, marker);
   addInternalBoundarySensitivityIntegrator(*integrator,
                                            nf_fields,
                                            rev_sens,
                                            rev_scalar_sens,
                                            fwd_sens,
                                            fwd_scalar_sens,
                                            &marker,
                                            adjoint_name);
}

}  // namespace miso

#endif  // MISO_NONLINEAR_FORM