#include <vector>

#include "mfem.hpp"

#include "utils.hpp"
#include "mach_input.hpp"
#include "mach_integrator.hpp"
#include "mach_nonlinearform.hpp"
#include "utils.hpp"

namespace mach
{
int getSize(const MachNonlinearForm &form)
{
   return form.nf.ParFESpace()->GetTrueVSize();
}

void setInputs(MachNonlinearForm &form, const MachInputs &inputs)
{
   for (const auto &[name, input] : inputs)
   {
      if (std::holds_alternative<InputVector>(input))
      {
         auto it = form.nf_fields.find(name);
         if (it != form.nf_fields.end())
         {
            auto &field = it->second;
            mfem::Vector field_tv;
            setVectorFromInput(input, field_tv);

            field.distributeSharedDofs(field_tv);
         }
      }
   }
   setInputs(form.integs, inputs);
}

void setOptions(MachNonlinearForm &form, const nlohmann::json &options)
{
   setOptions(form.integs, options);

   if (options.contains("bcs"))
   {
      if (options["bcs"].contains("essential"))
      {
         auto &fes = *form.nf.ParFESpace();
         mfem::Array<int> ess_bdr(fes.GetParMesh()->bdr_attributes.Max());
         getEssentialBoundaries(options["bcs"], ess_bdr);
         mfem::Array<int> ess_tdof_list;
         fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
         if (ess_tdof_list != nullptr)
         {
            form.nf.SetEssentialTrueDofs(ess_tdof_list);
         }
      }
   }
}

double calcFormOutput(MachNonlinearForm &form, const MachInputs &inputs)
{
   mfem::Vector state;
   setVectorFromInputs(inputs, "state", state, false, true);
   setInputs(form.integs, inputs);
   return form.nf.GetEnergy(state);
}

void evaluate(MachNonlinearForm &form,
              const MachInputs &inputs,
              mfem::Vector &res_vec)
{
   mfem::Vector state;
   setVectorFromInputs(inputs, "state", state, false, true);
   form.nf.Mult(state, res_vec);
}

void linearize(MachNonlinearForm &form, const MachInputs &inputs)
{
   setInputs(form, inputs);
   // getJacobianTranspose also gets the regular Jacobian
   getJacobianTranspose(form, inputs, "state");
}

mfem::Operator &getJacobian(MachNonlinearForm &form,
                            const MachInputs &inputs,
                            const std::string &wrt)
{
   mfem::Vector state;
   setVectorFromInputs(inputs, "state", state, false, true);
   form.jac = &form.nf.GetGradient(state);
   return *form.jac;
}

mfem::Operator &getJacobianTranspose(MachNonlinearForm &form,
                                     const MachInputs &inputs,
                                     const std::string &wrt)
{
   mfem::Vector state;
   setVectorFromInputs(inputs, "state", state, false, true);
   form.jac = &form.nf.GetGradient(state);

   auto *hypre_jac = dynamic_cast<mfem::HypreParMatrix *>(form.jac);
   if (hypre_jac == nullptr)
   {
      throw MachException(
          "getJacobianTranspose (MachNonlinearForm) currently only supports "
          "Jacobian matrices assembled to a HypreParMatrix!\n");
   }

   form.jac_trans = std::unique_ptr<mfem::Operator>(hypre_jac->Transpose());
   return *form.jac_trans;
}

double jacobianVectorProduct(MachNonlinearForm &form,
                             const mfem::Vector &wrt_dot,
                             const std::string &wrt)
{
   if (form.fwd_scalar_sens.count(wrt) != 0)
   {
      throw NotImplementedException(
          "not implemented for scalar sensitivities!\n");
   }
   return 0.0;
}

void jacobianVectorProduct(MachNonlinearForm &form,
                           const mfem::Vector &wrt_dot,
                           const std::string &wrt,
                           mfem::Vector &res_dot)
{
   if (form.fwd_sens.count(wrt) != 0)
   {
      if (wrt != "state")
      {
         throw NotImplementedException(
             "not implemented for vector sensitivities (except for state)!\n");
      }

      /// Integrators added to fwd_sens will reference wrt_dot grid func so
      /// we update it here
      std::string wrt_dot_name = wrt + "_dot";
      auto &wrt_dot_field = form.nf_fields.at(wrt_dot_name);
      wrt_dot_field.distributeSharedDofs(wrt_dot);

      /// Integrators added to fwd_sens will also reference the state grid func
      /// so that must have been distributed before calling this function
      form.fwd_sens.at(wrt).Assemble();
      form.scratch.SetSize(res_dot.Size());
      form.fwd_sens.at(wrt).ParallelAssemble(form.scratch);

      if (wrt == "state")
      {
         const auto &ess_tdof_list = form.nf.GetEssentialTrueDofs();
         for (int i = 0; i < ess_tdof_list.Size(); ++i)
         {
            form.scratch(ess_tdof_list[i]) = wrt_dot(ess_tdof_list[i]);
         }
      }
      res_dot += form.scratch;
   }
}

double vectorJacobianProduct(MachNonlinearForm &form,
                             const mfem::Vector &res_bar,
                             const std::string &wrt)
{
   if (form.rev_scalar_sens.count(wrt) != 0)
   {
      /// Integrators added to rev_scalar_sens will reference the adjoint grid
      /// func so we update it here
      auto &adjoint = form.nf_fields.at("adjoint");
      adjoint.distributeSharedDofs(res_bar);

      /// The state must have previously been distributed before calling this
      /// function
      auto &state = form.nf_fields.at("state").gridFunc();
      return form.rev_scalar_sens.at(wrt).GetGridFunctionEnergy(state);
   }
   return 0.0;
}

void vectorJacobianProduct(MachNonlinearForm &form,
                           const mfem::Vector &res_bar,
                           const std::string &wrt,
                           mfem::Vector &wrt_bar)
{
   if (form.rev_sens.count(wrt) != 0)
   {
      if (wrt == "state")
      {
         const auto &ess_tdof_list = form.nf.GetEssentialTrueDofs();
         form.scratch.SetSize(res_bar.Size());
         form.scratch = res_bar;

         form.scratch2.SetSize(res_bar.Size());
         form.scratch2 = 0.0;
         for (int i = 0; i < ess_tdof_list.Size(); ++i)
         {
            form.scratch2(ess_tdof_list[i]) = res_bar(ess_tdof_list[i]);
            form.scratch(ess_tdof_list[i]) = 0.0;
         }
         /// Integrators added to rev_sens will reference the adjoint grid func
         /// so we update it here
         auto &adjoint = form.nf_fields.at("adjoint");
         adjoint.distributeSharedDofs(form.scratch);
      }
      else
      {
         /// Integrators added to rev_sens will reference the adjoint, grid func
         /// so we update it here
         auto &adjoint = form.nf_fields.at("adjoint");
         adjoint.distributeSharedDofs(res_bar);
      }

      /// Integrators added to rev_sens will also reference the state grid func,
      /// so that must have been distributed before calling this function
      form.rev_sens.at(wrt).Assemble();
      form.scratch.SetSize(wrt_bar.Size());
      form.rev_sens.at(wrt).ParallelAssemble(form.scratch);
      wrt_bar += form.scratch;

      if (wrt == "state")
      {
         const auto &ess_tdof_list = form.nf.GetEssentialTrueDofs();
         for (int i = 0; i < ess_tdof_list.Size(); ++i)
         {
            wrt_bar(ess_tdof_list[i]) += form.scratch2(ess_tdof_list[i]);
         }
      }
   }
}

}  // namespace mach
