#ifndef MISO_FUNCTIONAL_OUTPUT
#define MISO_FUNCTIONAL_OUTPUT

#include <list>
#include <map>
#include <string>
#include <vector>

#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "miso_input.hpp"
#include "miso_integrator.hpp"

namespace miso
{
class FunctionalOutput final
{
public:
   friend void setInputs(FunctionalOutput &output, const MISOInputs &inputs);

   friend void setOptions(FunctionalOutput &output,
                          const nlohmann::json &options);

   friend double calcOutput(FunctionalOutput &output, const MISOInputs &inputs);

   friend double calcOutputPartial(FunctionalOutput &output,
                                   const std::string &wrt,
                                   const MISOInputs &inputs);

   friend void calcOutputPartial(FunctionalOutput &output,
                                 const std::string &wrt,
                                 const MISOInputs &inputs,
                                 mfem::HypreParVector &partial);

   /// Adds domain integrator to the nonlinear form that backs this output,
   /// and adds a reference to it to in integs as a MISOIntegrator
   /// \param[in] integrator - integrator to add to functional
   /// \tparam T - type of integrator, used for constructing MISOIntegrator
   template <typename T>
   void addOutputDomainIntegrator(T *integrator);

   /// Adds domain integrator restricted to certain elements specified by the
   /// attributes listed in @a bdr_attr_marker to the nonlinear form that backs
   /// this output, and adds a reference to it to in integs as a MISOIntegrator
   /// \param[in] integrator - integrator to add to functional
   /// \param[in] bdr_attr_marker - lists element attributes this integrator
   /// should be used on
   /// \tparam T - type of integrator, used for constructing MISOIntegrator
   template <typename T>
   void addOutputDomainIntegrator(T *integrator,
                                  std::vector<int> bdr_attr_marker);

   /// Adds interface integrator to the nonlinear form that backs this output,
   /// and adds a reference to it to in integs as a MISOIntegrator
   /// \param[in] integrator - integrator to add to functional
   /// \tparam T - type of integrator, used for constructing MISOIntegrator
   template <typename T>
   void addOutputInteriorFaceIntegrator(T *integrator);

   /// Adds boundary integrator to the nonlinear form that backs this output,
   /// and adds a reference to it to in integs as a MISOIntegrator
   /// \param[in] integrator - integrator to add to functional
   /// \tparam T - type of integrator, used for constructing MISOIntegrator
   template <typename T>
   void addOutputBdrFaceIntegrator(T *integrator);

   /// Adds boundary integrator restricted to certain boundaries specified by
   /// the attributes listed in @a bdr_attr_marker to the nonlinear form that
   /// backs this output, and adds a reference to it to in integs as a
   /// MISOIntegrator
   /// \param[in] integrator - integrator to add to functional
   /// \param[in] bdr_attr_marker - lists boundary attributes this integrator
   /// should be used on
   /// \tparam T - type of integrator, used for constructing MISOIntegrator
   template <typename T>
   void addOutputBdrFaceIntegrator(T *integrator,
                                   std::vector<int> bdr_attr_marker);

   FunctionalOutput(
       mfem::ParFiniteElementSpace &fes,
       std::unordered_map<std::string, mfem::ParGridFunction> &fields)
    : output(&fes), func_fields(&fields)
   { }

private:
   /// underlying nonlinear form object
   mfem::ParNonlinearForm output;
   /// map of external fields the functional depends on
   std::unordered_map<std::string, mfem::ParGridFunction> *func_fields;

   /// Collection of integrators to be applied.
   std::vector<MISOIntegrator> integs;

   /// Collection of element attribute markers for domain integrators
   std::list<mfem::Array<int>> domain_markers;

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
   miso::addSensitivityIntegrator(
       *integrator, *func_fields, output_sens, output_scalar_sens);
}

template <typename T>
void FunctionalOutput::addOutputDomainIntegrator(
    T *integrator,
    std::vector<int> bdr_attr_marker)
{
   integs.emplace_back(*integrator);
   auto &marker = domain_markers.emplace_back(bdr_attr_marker.size());
   marker.Assign(bdr_attr_marker.data());
   output.AddDomainIntegrator(integrator, marker);
   miso::addSensitivityIntegrator(
       *integrator, *func_fields, output_sens, output_scalar_sens);
}

template <typename T>
void FunctionalOutput::addOutputInteriorFaceIntegrator(T *integrator)
{
   integs.emplace_back(*integrator);
   output.AddInteriorFaceIntegrator(integrator);
   miso::addSensitivityIntegrator(
       *integrator, *func_fields, output_sens, output_scalar_sens);
}

template <typename T>
void FunctionalOutput::addOutputBdrFaceIntegrator(T *integrator)
{
   integs.emplace_back(*integrator);
   output.AddBdrFaceIntegrator(integrator);
   miso::addSensitivityIntegrator(
       *integrator, *func_fields, output_sens, output_scalar_sens);
}

template <typename T>
void FunctionalOutput::addOutputBdrFaceIntegrator(
    T *integrator,
    std::vector<int> bdr_attr_marker)
{
   integs.emplace_back(*integrator);
   auto &marker = bdr_markers.emplace_back(bdr_attr_marker.size());
   marker.Assign(bdr_attr_marker.data());
   output.AddBdrFaceIntegrator(integrator, marker);
   miso::addSensitivityIntegrator(
       *integrator, *func_fields, output_sens, output_scalar_sens);
}

}  // namespace miso

#endif
