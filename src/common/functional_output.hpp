#ifndef MACH_FUNCTIONAL_OUTPUT
#define MACH_FUNCTIONAL_OUTPUT

#include <list>
#include <map>
#include <string>
#include <vector>

#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "mach_input.hpp"
#include "mach_integrator.hpp"
#include "utils.hpp"

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

   /// Adds domain integrator restricted to certain elements specified by the
   /// attributes listed in @a bdr_attr_marker to the nonlinear form that backs
   /// this output, and adds a reference to it to in integs as a MachIntegrator
   /// \param[in] integrator - integrator to add to functional
   /// \param[in] bdr_attr_marker - lists element attributes this integrator
   /// should be used on
   /// \tparam T - type of integrator, used for constructing MachIntegrator
   template <typename T>
   void addOutputDomainIntegrator(T *integrator, std::vector<int> attr_marker);

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
   void addOutputBdrFaceIntegrator(T *integrator);

   /// Adds boundary integrator restricted to certain boundaries specified by
   /// the attributes listed in @a bdr_attr_marker to the nonlinear form that
   /// backs this output, and adds a reference to it to in integs as a
   /// MachIntegrator
   /// \param[in] integrator - integrator to add to functional
   /// \param[in] bdr_attr_marker - lists boundary attributes this integrator
   /// should be used on
   /// \tparam T - type of integrator, used for constructing MachIntegrator
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
   std::vector<MachIntegrator> integs;

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
   mach::addSensitivityIntegrator(
       *integrator, *func_fields, output_sens, output_scalar_sens);
}

template <typename T>
void FunctionalOutput::addOutputDomainIntegrator(T *integrator,
                                                 std::vector<int> attr_marker)
{
   integs.emplace_back(*integrator);
   // auto &marker = domain_markers.emplace_back(attr_marker.size());
   // marker.Assign(attr_marker.data());
   auto mesh_attr_size = output.ParFESpace()->GetMesh()->attributes.Size();
   auto &marker = domain_markers.emplace_back(mesh_attr_size);
   attrVecToArray(attr_marker, marker);
   output.AddDomainIntegrator(integrator, marker);
   mach::addSensitivityIntegrator(
       *integrator, *func_fields, output_sens, output_scalar_sens);
}

template <typename T>
void FunctionalOutput::addOutputInteriorFaceIntegrator(T *integrator)
{
   integs.emplace_back(*integrator);
   output.AddInteriorFaceIntegrator(integrator);
   mach::addSensitivityIntegrator(
       *integrator, *func_fields, output_sens, output_scalar_sens);
}

template <typename T>
void FunctionalOutput::addOutputBdrFaceIntegrator(T *integrator)
{
   integs.emplace_back(*integrator);
   output.AddBdrFaceIntegrator(integrator);
   mach::addSensitivityIntegrator(
       *integrator, *func_fields, output_sens, output_scalar_sens);
}

template <typename T>
void FunctionalOutput::addOutputBdrFaceIntegrator(
    T *integrator,
    std::vector<int> bdr_attr_marker)
{
   integs.emplace_back(*integrator);
   //    auto &marker = bdr_markers.emplace_back(bdr_attr_marker.size());
   //    marker.Assign(bdr_attr_marker.data());
   auto mesh_attr_size = output.ParFESpace()->GetMesh()->attributes.Size();
   auto &marker = bdr_markers.emplace_back(mesh_attr_size);
   attrVecToArray(bdr_attr_marker, marker);
   output.AddBdrFaceIntegrator(integrator, marker);
   mach::addSensitivityIntegrator(
       *integrator, *func_fields, output_sens, output_scalar_sens);
}

}  // namespace mach

#endif
