#ifndef MACH_NONLINEAR_FORM
#define MACH_NONLINEAR_FORM

#include <vector>
#include <list>

#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "mach_input.hpp"
#include "mach_integrator.hpp"

namespace mach
{
class MachNonlinearForm final
{
public:
   /// Get the size of the nonlinear form (number of equations/unknowns)
   friend int getSize(const MachNonlinearForm &form);

   /// Set inputs in all integrators used by the nonlinear form
   friend void setInputs(MachNonlinearForm &form, const MachInputs &inputs);

   friend void setOptions(MachNonlinearForm &form,
                          const nlohmann::json &options);

   /// Calls GetEnergy() for the underlying form using "state" in inputs.
   friend double calcFormOutput(MachNonlinearForm &form,
                                const MachInputs &inputs);

   /// Evaluate the nonlinear form using `inputs` and return result in `res_vec`
   friend void evaluate(MachNonlinearForm &form,
                        const MachInputs &inputs,
                        mfem::Vector &res_vec);

   friend void linearize(MachNonlinearForm &form, const MachInputs &inputs);

   /// Compute Jacobian of `form` with respect to `wrt` and return
   friend mfem::Operator &getJacobian(MachNonlinearForm &form,
                                      const MachInputs &inputs,
                                      const std::string &wrt);

   /// Compute transpose of Jacobian of `form` with respect to `wrt` and return
   friend mfem::Operator &getJacobianTranspose(MachNonlinearForm &form,
                                               const MachInputs &inputs,
                                               const std::string &wrt);

   friend void setUpAdjointSystem(MachNonlinearForm &form,
                                  mfem::Solver &adj_solver,
                                  const MachInputs &inputs,
                                  mfem::Vector &state_bar,
                                  mfem::Vector &adjoint);

   friend void finalizeAdjointSystem(MachNonlinearForm &form,
                                     mfem::Solver &adj_solver,
                                     const MachInputs &inputs,
                                     mfem::Vector &state_bar,
                                     mfem::Vector &adjoint);

   friend double jacobianVectorProduct(MachNonlinearForm &form,
                                       const mfem::Vector &wrt_dot,
                                       const std::string &wrt);

   friend void jacobianVectorProduct(MachNonlinearForm &form,
                                     const mfem::Vector &wrt_dot,
                                     const std::string &wrt,
                                     mfem::Vector &res_dot);

   friend double vectorJacobianProduct(MachNonlinearForm &form,
                                       const mfem::Vector &res_bar,
                                       const std::string &wrt);

   friend void vectorJacobianProduct(MachNonlinearForm &form,
                                     const mfem::Vector &res_bar,
                                     const std::string &wrt,
                                     mfem::Vector &wrt_bar);

   /// Adds the given domain integrator to the nonlinear form
   /// \param[in] integrator - nonlinear form integrator for domain
   /// \tparam T - type of integrator, used for constructing MachIntegrator
   /// \note Assumes ownership of integrator
   template <typename T>
   void addDomainIntegrator(T *integrator);

   /// Adds the given domain integrator to the nonlinear form, restricted to
   /// the given domain attributes
   /// \param[in] integrator - nonlinear form integrator for domain
   /// \param[in] attr_marker - domain attributes for integrator
   /// \tparam T - type of integrator, used for constructing MachIntegrator
   /// \note Assumes ownership of integrator
   template <typename T>
   void addDomainIntegrator(T *integrator, const std::vector<int> &attr_marker);

   /// Adds the given interior face integrator to the nonlinear form
   /// \param[in] integrator - face nonlinear form integrator for interfaces
   /// \tparam T - type of integrator, used for constructing MachIntegrator
   /// \note Assumes ownership of integrator
   template <typename T>
   void addInteriorFaceIntegrator(T *integrator);

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
   void addBdrFaceIntegrator(T *integrator,
                             const std::vector<int> &bdr_attr_marker);

   const mfem::Array<int> &getEssentialDofs() const
   {
      return nf.GetEssentialTrueDofs();
   }

   /// Constructor for nonlinear form types
   /// \param[in] pfes - FEM space for the state (and possibly the adjoint)
   /// \param[in] fields - map of grid functions
   MachNonlinearForm(mfem::ParFiniteElementSpace &pfes,
                     std::map<std::string, FiniteElementState> &fields)
    : nf(&pfes),
      scratch(0),
      nf_fields(fields),
      jac(mfem::Operator::Hypre_ParCSR),
      jac_e(mfem::Operator::Hypre_ParCSR)
   {
      if (nf_fields.count("adjoint") == 0)
      {
         // nf_fields.emplace("adjoint", {*pfes.GetParMesh(), pfes});
         nf_fields.emplace(std::piecewise_construct,
                           std::forward_as_tuple("adjoint"),
                           std::forward_as_tuple(*pfes.GetParMesh(), pfes));
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
   std::vector<MachIntegrator> integs;
   /// Collection of  markers for domain integrators
   std::list<mfem::Array<int>> domain_markers;
   /// Collection of boundary markers for boundary integrators
   std::list<mfem::Array<int>> bdr_markers;

   /// map of external fields that the nonlinear form depends on
   std::map<std::string, FiniteElementState> &nf_fields;

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

   mfem::Vector adj_work1, adj_work2;
};

template <typename T>
void MachNonlinearForm::addDomainIntegrator(T *integrator)
{
   integs.emplace_back(*integrator);
   nf.AddDomainIntegrator(integrator);
   addDomainSensitivityIntegrator(*integrator,
                                  nf_fields,
                                  rev_sens,
                                  rev_scalar_sens,
                                  fwd_sens,
                                  fwd_scalar_sens,
                                  nullptr);
}

template <typename T>
void MachNonlinearForm::addDomainIntegrator(
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
                                  &marker);
}

template <typename T>
void MachNonlinearForm::addInteriorFaceIntegrator(T *integrator)
{
   integs.emplace_back(*integrator);
   nf.AddInteriorFaceIntegrator(integrator);
   addInteriorFaceSensitivityIntegrator(*integrator,
                                        nf_fields,
                                        rev_sens,
                                        rev_scalar_sens,
                                        fwd_sens,
                                        fwd_scalar_sens);
}

template <typename T>
void MachNonlinearForm::addBdrFaceIntegrator(T *integrator)
{
   integs.emplace_back(*integrator);
   nf.AddBdrFaceIntegrator(integrator);
   addBdrSensitivityIntegrator(*integrator,
                               nf_fields,
                               rev_sens,
                               rev_scalar_sens,
                               fwd_sens,
                               fwd_scalar_sens,
                               nullptr);
}

template <typename T>
void MachNonlinearForm::addBdrFaceIntegrator(
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
                               &marker);
}

}  // namespace mach

#endif  // MACH_NONLINEAR_FORM