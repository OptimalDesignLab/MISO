#include <vector>

#include "mfem.hpp"

#include "mach_input.hpp"
#include "mach_integrator.hpp"
#include "mach_linearform.hpp"
#include "utils.hpp"

namespace mach
{
int getSize(const MachLinearForm &load)
{
   return load.lf.ParFESpace()->GetTrueVSize();
}

void setInputs(MachLinearForm &load, const MachInputs &inputs)
{
   // for (const auto &in : inputs)
   // {
   //    const auto &input = in.second;
   //    if (input.isField())
   //    {
   //       const auto &name = in.first;
   //       auto it = load.lf_fields->find(name);
   //       if (it != load.lf_fields->end())
   //       {
   //          auto &field = it->second;
   //          field.GetTrueVector().SetDataAndSize(
   //              input.getField(), field.ParFESpace()->GetTrueVSize());
   //          field.SetFromTrueVector();
   //       }
   //    }
   // }
   for (const auto &in : inputs)
   {
      const auto &input = in.second;
      if (std::holds_alternative<InputVector>(input))
      {
         const auto &name = in.first;
         auto it = load.lf_fields->find(name);
         if (it != load.lf_fields->end())
         {
            auto &field = it->second;
            mfem::Vector field_tv;
            setVectorFromInput(input, field_tv);

            field.distributeSharedDofs(field_tv);
         }
      }
   }
   setInputs(load.integs, inputs);
}

void setOptions(MachLinearForm &load, const nlohmann::json &options)
{
   setOptions(load.integs, options);

   if (options.contains("ess-bdr"))
   {
      auto fes = *load.lf.ParFESpace();
      mfem::Array<int> ess_bdr(fes.GetParMesh()->bdr_attributes.Max());
      getEssentialBoundaries(options, ess_bdr);
      fes.GetEssentialTrueDofs(ess_bdr, load.ess_tdof_list);
   }
}

void addLoad(MachLinearForm &load, mfem::Vector &tv)
{
   load.lf.Assemble();
   load.scratch.SetSize(tv.Size());
   load.lf.ParallelAssemble(load.scratch);
   load.scratch.SetSubVector(load.ess_tdof_list, 0.0);
   add(tv, load.scratch, tv);
}

double vectorJacobianProduct(MachLinearForm &load,
                             const mfem::Vector &load_bar,
                             const std::string &wrt)
{
   if (load.scalar_sens.count(wrt) != 0)
   {
      throw std::logic_error(
          "vectorJacobianProduct not implemented for MachLinearForm!\n");
      // auto &adjoint = load.lf_fields->at("adjoint");
      // adjoint = load_bar;
      // return load.scalar_sens.at(wrt).GetEnergy();
   }
   else
   {
      return 0.0;
   }
}

void vectorJacobianProduct(MachLinearForm &load,
                           const mfem::Vector &load_bar,
                           const std::string &wrt,
                           mfem::Vector &wrt_bar)
{
   if (load.sens.count(wrt) != 0)
   {
      auto &adjoint = load.lf_fields->at("adjoint");
      adjoint.distributeSharedDofs(load_bar);
      load.sens.at(wrt).Assemble();
      load.scratch.SetSize(wrt_bar.Size());
      load.sens.at(wrt).ParallelAssemble(load.scratch);
      wrt_bar += load.scratch;
   }
}

}  // namespace mach
