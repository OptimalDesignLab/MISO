#include <vector>

#include "mfem.hpp"

#include "miso_input.hpp"
#include "miso_integrator.hpp"
#include "miso_linearform.hpp"
#include "utils.hpp"

namespace miso
{
int getSize(const MISOLinearForm &load)
{
   return load.lf.ParFESpace()->GetTrueVSize();
}

void setInputs(MISOLinearForm &load, const MISOInputs &inputs)
{
   for (const auto &[name, input] : inputs)
   {
      if (std::holds_alternative<InputVector>(input))
      {
         if (load.lf_fields != nullptr)
         {
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
   }
   setInputs(load.integs, inputs);
}

void setOptions(MISOLinearForm &load, const nlohmann::json &options)
{
   setOptions(load.integs, options);

   if (options.contains("bcs"))
   {
      if (options["bcs"].contains("essential"))
      {
         auto &fes = *load.lf.ParFESpace();
         mfem::Array<int> ess_bdr(fes.GetParMesh()->bdr_attributes.Max());
         getMFEMBoundaryArray(options["bcs"]["essential"], ess_bdr);
         fes.GetEssentialTrueDofs(ess_bdr, load.ess_tdof_list);
      }
   }
}

void addLoad(MISOLinearForm &load, mfem::Vector &tv)
{
   load.lf.Assemble();
   load.scratch.SetSize(tv.Size());
   load.lf.ParallelAssemble(load.scratch);
   load.scratch.SetSubVector(load.ess_tdof_list, 0.0);
   add(tv, load.scratch, tv);
}

double jacobianVectorProduct(MISOLinearForm &load,
                             const mfem::Vector &wrt_dot,
                             const std::string &wrt)
{
   if (load.fwd_scalar_sens.count(wrt) != 0)
   {
      throw NotImplementedException(
          "not implemented for scalar sensitivities!\n");
   }
   return 0.0;
}

void jacobianVectorProduct(MISOLinearForm &load,
                           const mfem::Vector &wrt_dot,
                           const std::string &wrt,
                           mfem::Vector &res_dot)
{
   if (load.fwd_sens.count(wrt) != 0)
   {
      throw NotImplementedException(
          "not implemented for vector sensitivities!\n");
   }
}

double vectorJacobianProduct(MISOLinearForm &load,
                             const mfem::Vector &load_bar,
                             const std::string &wrt)
{
   if (load.rev_scalar_sens.count(wrt) != 0)
   {
      /// Integrators added to rev_scalar_sens will reference the adjoint grid
      /// func so we update it here
      auto &adjoint = load.lf_fields->at(load.adjoint_name);
      adjoint.distributeSharedDofs(load_bar);

      /// The state must have previously been distributed before calling this
      /// function
      auto &state = load.lf_fields->at("state").gridFunc();
      return load.rev_scalar_sens.at(wrt).GetGridFunctionEnergy(state);
   }
   return 0.0;
}

void vectorJacobianProduct(MISOLinearForm &load,
                           const mfem::Vector &load_bar,
                           const std::string &wrt,
                           mfem::Vector &wrt_bar)
{
   if (load.rev_sens.count(wrt) != 0)
   {
      load.scratch.SetSize(load_bar.Size());
      load.scratch = load_bar;
      const auto &ess_tdof_list = load.getEssentialDofs();
      load.scratch.SetSubVector(ess_tdof_list, 0.0);

      /// Integrators added to rev_sens will reference the adjoint, grid func
      /// so we update it here
      auto &adjoint = load.lf_fields->at(load.adjoint_name);
      adjoint.distributeSharedDofs(load.scratch);

      auto &wrt_rev_sens = load.rev_sens.at(wrt);

      wrt_rev_sens.Assemble();
      load.scratch.SetSize(wrt_bar.Size());
      load.scratch = 0.0;
      wrt_rev_sens.ParallelAssemble(load.scratch);

      wrt_bar += load.scratch;
   }
}

}  // namespace miso
