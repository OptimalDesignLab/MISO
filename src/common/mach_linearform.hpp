#ifndef MACH_LINEAR_FORM
#define MACH_LINEAR_FORM

#include <vector>

#include "mfem.hpp"

#include "mach_input.hpp"
#include "mach_integrator.hpp"

namespace mach
{

class MachLinearForm final
{
public:
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
   /// \tparam T - type of integrator, used for constructing MachIntegrator
   /// \param[in] bdr_attr_marker - boundary attributes for integrator
   /// \note Assumes ownership of integrator
   /// \note The array bdr_attr_marker is copied
   template <typename T>
   void addBoundaryIntegrator(T *integrator,
                              mfem::Array<int> &bdr_attr_marker);

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
   /// \note Assumes ownership of integrator
   /// \note The array bdr_attr_marker is copied
   template <typename T>
   void addBdrFaceIntegrator(T *integrator,
                             mfem::Array<int> &bdr_attr_marker);

   /// Assemble the linear form on the true dofs and add it to tv
   friend void addLoad(MachLinearForm &lf,
                       mfem::Vector &tv);
   
   /// Set scalar inputs in all integrators used by the linear form
   friend void setInputs(MachLinearForm &lf,
                         const MachInputs &inputs);

   /// Assemble the linear form's sensitivity to a scalar and contract it with
   /// res_bar
   friend double vectorJacobianProduct(MachLinearForm &load,
                                       const mfem::HypreParVector &res_bar,
                                       std::string wrt);

   friend void vectorJacobianProduct(MachLinearForm &load,
                                     const mfem::HypreParVector &res_bar,
                                     std::string wrt,
                                     mfem::HypreParVector &wrt_bar);

   MachLinearForm(mfem::ParFiniteElementSpace *pfes)
   : lf(pfes), scratch(pfes)
   { }

private:
   mfem::ParLinearForm lf;
   mfem::HypreParVector scratch;

   /// Set of Domain Integrators to be applied.
   std::vector<MachIntegrator> dlfi;

   /// Set of Boundary Integrators to be applied.
   std::vector<MachIntegrator> blfi;
   std::vector<mfem::Array<int>> blfi_marker; 

   /// Set of Boundary Face Integrators to be applied.
   std::vector<MachIntegrator> flfi;
   std::vector<mfem::Array<int>> flfi_marker;
};

template <typename T>
void MachLinearForm::addDomainIntegrator(T *integrator)
{
   lf.AddDomainIntegrator(integrator);
   dlfi.emplace_back(*integrator);
}

template <typename T>
void MachLinearForm::addBoundaryIntegrator(T *integrator)
{
   lf.AddBoundaryIntegrator(integrator);
   blfi.emplace_back(*integrator);
}

template <typename T>
void MachLinearForm::addBoundaryIntegrator(T *integrator,
                                           mfem::Array<int> &bdr_attr_marker)
{
   blfi.emplace_back(*integrator);
   blfi_marker.emplace_back(bdr_attr_marker);
   lf.AddBoundaryIntegrator(integrator, blfi_marker.back());
}

template <typename T>
void MachLinearForm::addBdrFaceIntegrator(T *integrator)
{
   lf.AddBoundaryIntegrator(integrator);
   flfi.emplace_back(*integrator);

}

template <typename T>
void MachLinearForm::addBdrFaceIntegrator(T *integrator,
                                          mfem::Array<int> &bdr_attr_marker)
{
   flfi.emplace_back(*integrator);
   flfi_marker.emplace_back(bdr_attr_marker);
   lf.AddBoundaryIntegrator(integrator, flfi_marker.back());
}

} // namespace mach

#endif
