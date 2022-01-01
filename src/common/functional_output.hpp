#ifndef MACH_FUNCTIONAL_OUTPUT
#define MACH_FUNCTIONAL_OUTPUT

#include <list>

#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "finite_element_state.hpp"
#include "mach_input.hpp"
#include "mach_integrator.hpp"

namespace mach
{
class FunctionalOutput final
{
public:
   friend void setInputs(FunctionalOutput &output, const MachInputs &inputs);

   friend void setOptions(FunctionalOutput &output,
                          const nlohmann::json &options);

   friend double calcOutput(FunctionalOutput &output, const MachInputs &inputs);

   friend double calcOutputPartial(FunctionalOutput &output,
                                   const std::string &wrt,
                                   const MachInputs &inputs);

   friend void calcOutputPartial(FunctionalOutput &output,
                                 const std::string &wrt,
                                 const MachInputs &inputs,
                                 mfem::Vector &partial);

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
                                   std::vector<int> bdr_attr_marker);

   FunctionalOutput(
       mfem::ParFiniteElementSpace &fes,
       std::unordered_map<std::string, mfem::ParGridFunction> &fields)
    : output(&fes), func_fields(&fields)
   { }

   FunctionalOutput(
      mfem::ParFiniteElementSpace &fes,
      std::map<std::string, FiniteElementState> &fields)
    : output(&fes), fun_fields(&fields)
   { }

private:
   /// underlying nonlinear form object
   mfem::ParNonlinearForm output;
   /// map of external fields the functional depends on
   std::unordered_map<std::string, mfem::ParGridFunction> *func_fields =
       nullptr;
   /// map of external fields the functional depends on
   std::map<std::string, FiniteElementState> *fun_fields = nullptr;

   /// Collection of integrators to be applied.
   std::vector<MachIntegrator> integs;
   /// Collection of boundary markers for boundary integrators
   std::list<mfem::Array<int>> bdr_markers;

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
   integs.emplace_back(*integrator);
   output.AddDomainIntegrator(integrator);
   if (func_fields)
   {
      mach::addSensitivityIntegrator(
          *integrator, *func_fields, output_sens, output_scalar_sens);
   }
   else
   {
      std::cout << "WARNING: FunctionalOutput::addSensitivityIntegrator" <<
          "is not set up to work with FiniteElementState.\n";
   }
}

template <typename T>
void FunctionalOutput::addOutputInteriorFaceIntegrator(T *integrator)
{
   integs.emplace_back(*integrator);
   output.AddInteriorFaceIntegrator(integrator);
   if (func_fields)
   {
      mach::addSensitivityIntegrator(
          *integrator, *func_fields, output_sens, output_scalar_sens);
   }
   else
   {
      std::cout << "WARNING: FunctionalOutput::addSensitivityIntegrator" <<
          "is not set up to work with FiniteElementState.\n";
   }
}

template <typename T>
void FunctionalOutput::addOutputBdrFaceIntegrator(
    T *integrator,
    std::vector<int> bdr_attr_marker)
{
   integs.emplace_back(*integrator);
   bdr_markers.emplace_back(bdr_attr_marker.size());
   bdr_markers.back().Assign(bdr_attr_marker.data());
   output.AddBdrFaceIntegrator(integrator, bdr_markers.back());
   if (func_fields)
   {
      mach::addSensitivityIntegrator(
          *integrator, *func_fields, output_sens, output_scalar_sens);
   }
   else
   {
      std::cout << "WARNING: FunctionalOutput::addSensitivityIntegrator" <<
          "is not set up to work with FiniteElementState.\n";
   }

}

}  // namespace mach

#endif
