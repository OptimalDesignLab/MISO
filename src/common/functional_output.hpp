#ifndef MACH_FUNCTIONAL_OUTPUT
#define MACH_FUNCTIONAL_OUTPUT

#include "mfem.hpp"

#include "mach_input.hpp"
#include "mach_integrator.hpp"

namespace mach
{
class FunctionalOutput final
{
public:
   /// Used to set inputs in the underlying output type
   friend void setInputs(FunctionalOutput &output, const MachInputs &inputs);

   /// Compute the output vector on the true dofs and add it to tv
   friend double calcOutput(FunctionalOutput &output, const MachInputs &inputs);

   /// Compute the output's sensitivity to a scalar
   friend double calcOutputPartial(FunctionalOutput &output,
                                   const std::string &wrt,
                                   const MachInputs &inputs);

   /// Compute the output's sensitivity to a field and store in @a partial
   friend void calcOutputPartial(FunctionalOutput &output,
                                 const std::string &wrt,
                                 const MachInputs &inputs,
                                 mfem::HypreParVector &partial);

   /// Adds domain integrator to the nonlinear form that backs this output,
   /// and adds a reference to it to in integs as a MachIntegrator
   /// \param[in] integrator - integrator to add to functional
   /// \tparam T - type of integrator, used for constructing MachIntegrator
   template <typename T>
   void addOutputDomainIntegrator(T *integrator);

   /// Adds interface integrator to the nonlinear form that backs this output,
   /// and adds a reference to it to in integs as a MachIntegrator
   /// \param[in] integrator - integrator to add to functional
   /// \tparam T - type of integrator, used for constructing MachIntegrator
   template <typename T>
   void addOutputInteriorFaceIntegrator(T *integrator);

   /// Adds boundary integrator to the nonlinear form that backs this output,
   /// and adds a reference to it to in integs as a MachIntegrator
   /// \param[in] integrator - integrator to add to functional
   /// \tparam T - type of integrator, used for constructing MachIntegrator
   template <typename T>
   void addOutputBdrFaceIntegrator(T *integrator,
                                   mfem::Array<int> &bdr_marker);

   FunctionalOutput(
       mfem::ParFiniteElementSpace &pfes,
       std::unordered_map<std::string, mfem::ParGridFunction> &fields)
    : output(&pfes), func_fields(fields)
   { }

private:
   /// underlying nonlinear form object
   mfem::ParNonlinearForm output;

   /// Collection of integrators to be applied.
   std::vector<MachIntegrator> integs;
   /// Collection of boundary markers for boundary integrators
   std::vector<mfem::Array<int>> bdr_marker;

   /// map of external fields the functional depends on
   std::unordered_map<std::string, mfem::ParGridFunction> &func_fields;

   /// map of linear forms that will compute \frac{\partial J}{\partial field}
   /// for each field the functional depends on
   std::map<std::string, mfem::ParLinearForm> output_sens;
   /// map of nonlinear forms that will compute \frac{\partial J}{\partial
   /// scalar} for each scalar the functional depends on
   std::map<std::string, mfem::ParNonlinearForm> output_scalar_sens;
};

template <typename T>
void FunctionalOutput::addOutputDomainIntegrator(T *integrator)
{
   output.AddDomainIntegrator(integrator);
   integs.emplace_back(*integrator);
   mach::addSensitivityIntegrator(
       *integrator, func_fields, output_sens, output_scalar_sens);
}

template <typename T>
void FunctionalOutput::addOutputInteriorFaceIntegrator(T *integrator)
{
   output.AddInteriorFaceIntegrator(integrator);
   integs.emplace_back(*integrator);
   mach::addSensitivityIntegrator(
       *integrator, func_fields, output_sens, output_scalar_sens);
}

template <typename T>
void FunctionalOutput::addOutputBdrFaceIntegrator(T *integrator,
                                                  mfem::Array<int> &bdr_marker)
{
   output.AddBdrFaceIntegrator(integrator, bdr_marker);
   integs.emplace_back(*integrator);
   mach::addSensitivityIntegrator(
       *integrator, func_fields, output_sens, output_scalar_sens);
}

}  // namespace mach

#endif
