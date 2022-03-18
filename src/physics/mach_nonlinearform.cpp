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
   // auto *pfes = form.nf.ParFESpace();
   // auto state = bufferToHypreParVector(inputs.at("state").getField(), *pfes);
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

   auto *hypre_jac = dynamic_cast<const mfem::HypreParMatrix *>(form.jac);
   if (hypre_jac == nullptr)
   {
      throw MachException(
          "getJacobianTranspose (MachNonlinearForm) currently only supports "
          "Jacobian matrices assembled to a HypreParMatrix!\n");
   }

   form.jac_trans = std::unique_ptr<mfem::Operator>(hypre_jac->Transpose());
   return *form.jac_trans;
}

double vectorJacobianProduct(MachNonlinearForm &form,
                             const mfem::Vector &res_bar,
                             const std::string &wrt)
{
   if (form.scalar_sens.count(wrt) != 0)
   {
      /// Integrators added to scalar_sens will reference the adjoint, grid
      /// func so we update it here
      auto &adjoint = form.nf_fields.at("adjoint");
      adjoint.distributeSharedDofs(res_bar);

      /// The state must have previously been distributed before calling this
      /// function
      auto &state = form.nf_fields.at("state").gridFunc();
      return form.scalar_sens.at(wrt).GetParGridFunctionEnergy(state);
   }
   return 0.0;
}

void vectorJacobianProduct(MachNonlinearForm &form,
                           const mfem::Vector &res_bar,
                           const std::string &wrt,
                           mfem::Vector &wrt_bar)
{
   if (wrt == "state")
   {
      form.scratch.SetSize(wrt_bar.Size());
      form.jac_trans->Mult(res_bar, form.scratch);

      wrt_bar += form.scratch;
   }
   else
   {
      if (form.sens.count(wrt) != 0)
      {
         /// Integrators added to sens will reference the adjoint, grid func so
         /// we update it here
         auto &adjoint = form.nf_fields.at("adjoint");
         adjoint.distributeSharedDofs(res_bar);

         /// Integrators added to sens will also reference the state grid func,
         /// so that must have been distributed before calling this function
         form.sens.at(wrt).Assemble();
         form.scratch.SetSize(wrt_bar.Size());
         form.sens.at(wrt).ParallelAssemble(form.scratch);

         wrt_bar += form.scratch;
      }
   }
}

}  // namespace mach
