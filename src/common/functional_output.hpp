#ifndef MACH_FUNCTIONAL_OUTPUT
#define MACH_FUNCTIONAL_OUTPUT

#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "mach_input.hpp"
#include "mach_integrator.hpp"

namespace mach
{
class FunctionalOutput final
{
public:
   friend void setInputs(FunctionalOutput &output, const MachInputs &inputs);

   friend void setOptions(FunctionalOutput &output,
                          const nlohmann::json &options)
   { }

   friend double calcOutput(FunctionalOutput &output, const MachInputs &inputs);

   friend double calcOutputPartial(FunctionalOutput &output,
                                   const std::string &wrt,
                                   const MachInputs &inputs);

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
   void addOutputBdrFaceIntegrator(T *integrator, mfem::Array<int> &bdr_marker);

   FunctionalOutput(
       mfem::ParFiniteElementSpace &fes,
       std::unordered_map<std::string, mfem::ParGridFunction> &fields)
    : output(&fes), func_fields(&fields)
   { }

   FunctionalOutput(const FunctionalOutput &) = delete;
   FunctionalOutput &operator=(const FunctionalOutput &) = delete;

   FunctionalOutput(FunctionalOutput &&) = default;
   FunctionalOutput &operator=(FunctionalOutput &&) = default;

   ~FunctionalOutput() = default;

private:
   /// underlying nonlinear form object
   mfem::ParNonlinearForm output;
   /// map of external fields the functional depends on
   std::unordered_map<std::string, mfem::ParGridFunction> *func_fields;

   /// Collection of integrators to be applied.
   std::vector<MachIntegrator> integs;
   /// Collection of boundary markers for boundary integrators
   std::vector<mfem::Array<int>> bdr_marker;


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
       *integrator, *func_fields, output_sens, output_scalar_sens);
}

template <typename T>
void FunctionalOutput::addOutputInteriorFaceIntegrator(T *integrator)
{
   output.AddInteriorFaceIntegrator(integrator);
   integs.emplace_back(*integrator);
   mach::addSensitivityIntegrator(
       *integrator, *func_fields, output_sens, output_scalar_sens);
}

template <typename T>
void FunctionalOutput::addOutputBdrFaceIntegrator(T *integrator,
                                                  mfem::Array<int> &bdr_marker)
{
   output.AddBdrFaceIntegrator(integrator, bdr_marker);
   integs.emplace_back(*integrator);
   mach::addSensitivityIntegrator(
       *integrator, *func_fields, output_sens, output_scalar_sens);
}

}  // namespace mach

#endif
