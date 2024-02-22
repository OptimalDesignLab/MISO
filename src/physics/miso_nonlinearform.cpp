#include <vector>

#include "mfem.hpp"

#include "utils.hpp"
#include "miso_input.hpp"
#include "miso_integrator.hpp"
#include "miso_nonlinearform.hpp"

namespace miso
{
int getSize(const MISONonlinearForm &form)
{
   return form.nf.ParFESpace()->GetTrueVSize();
}

void setInputs(MISONonlinearForm &form, const MISOInputs &inputs)
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

void setOptions(MISONonlinearForm &form, const nlohmann::json &options)
{
   setOptions(form.integs, options);

   if (options.contains("bcs"))
   {
      if (options["bcs"].contains("essential"))
      {
         auto &fes = *form.nf.ParFESpace();
         form.ess_bdr.SetSize(fes.GetParMesh()->bdr_attributes.Max());
         getMFEMBoundaryArray(options["bcs"]["essential"], form.ess_bdr);
         mfem::Array<int> ess_tdof_list;
         fes.GetEssentialTrueDofs(form.ess_bdr, ess_tdof_list);
         form.nf.SetEssentialTrueDofs(ess_tdof_list);
      }
   }
}

double calcFormOutput(MISONonlinearForm &form, const MISOInputs &inputs)
{
   mfem::Vector state;
   setVectorFromInputs(inputs, "state", state, false, true);
   setInputs(form.integs, inputs);
   return form.nf.GetEnergy(state);
}

void evaluate(MISONonlinearForm &form,
              const MISOInputs &inputs,
              mfem::Vector &res_vec)
{
   mfem::Vector state;
   setVectorFromInputs(inputs, "state", state, false, true);
   form.nf.Mult(state, res_vec);

   const auto &ess_tdof_list = form.getEssentialDofs();
   if (ess_tdof_list.Size() == 0)
   {
      return;
   }

   if (auto bc_iter = form.nf_fields.find("dirichlet_bc");
       bc_iter != form.nf_fields.end())
   {
      auto &dirichlet_bc = bc_iter->second;
      dirichlet_bc.setTrueVec(form.scratch);

      for (int i = 0; i < ess_tdof_list.Size(); ++i)
      {
         res_vec(ess_tdof_list[i]) =
             state(ess_tdof_list[i]) - form.scratch(ess_tdof_list[i]);
      }
   }
}

void linearize(MISONonlinearForm &form, const MISOInputs &inputs)
{
   std::cout << "In linearize!\n";
   setInputs(form, inputs);
   if (form.jac.Ptr() == nullptr)
   {
      getJacobian(form, inputs, "state");
   }
   if (form.jac_trans == nullptr)
   {
      getJacobianTranspose(form, inputs, "state");
   }
}

mfem::Operator &getJacobian(MISONonlinearForm &form,
                            const MISOInputs &inputs,
                            const std::string &wrt)
{
   std::cout << "Re-assembling Jacobian!\n";

   mfem::Vector state;
   setVectorFromInputs(inputs, "state", state, false, true);

   mfem::Array<int> ess_tdof_list(form.nf.GetEssentialTrueDofs());
   mfem::Array<int> zeros;
   // Setting our essential true dofs to zero to full Jacobian is preserved
   form.nf.SetEssentialTrueDofs(zeros);

   // get our gradient with everything preserved
   auto *hypre_jac =
       dynamic_cast<mfem::HypreParMatrix *>(&form.nf.GetGradient(state));
   form.jac.Reset(hypre_jac, false);

   // Impose boundary conditions on pGrad
   form.jac_e.Clear();
   form.jac_e.EliminateRowsCols(form.jac, ess_tdof_list);
   form.jac_e.EliminateRows(ess_tdof_list);

   // reset our essential BCs to what they used to be
   form.nf.SetEssentialTrueDofs(ess_tdof_list);

   // reset transposed Jacobian to null (to indicate we should re-transpose it)
   form.jac_trans = nullptr;
   return *form.jac;
}

mfem::Operator &getJacobianTranspose(MISONonlinearForm &form,
                                     const MISOInputs &inputs,
                                     const std::string &wrt)
{
   if (form.jac_trans == nullptr)
   {
      std::cout << "Re-transposing Jacobian!\n";

      auto *hypre_jac = form.jac.As<mfem::HypreParMatrix>();
      if (hypre_jac == nullptr)
      {
         throw MISOException(
             "getJacobianTranspose (MISONonlinearForm) only supports "
             "Jacobian matrices assembled to a HypreParMatrix!\n");
      }

      form.jac_trans = std::unique_ptr<mfem::Operator>(hypre_jac->Transpose());
   }
   return *form.jac_trans;
}

void setUpAdjointSystem(MISONonlinearForm &form,
                        mfem::Solver &adj_solver,
                        const MISOInputs &inputs,
                        mfem::Vector &state_bar,
                        mfem::Vector &adjoint)
{
   // std::cout << "Setting up adjoint system!\n";

   auto &jac_trans = getJacobianTranspose(form, inputs, "state");
   adj_solver.SetOperator(jac_trans);

   const auto &ess_tdof_list = form.getEssentialDofs();
   if (ess_tdof_list.Size() == 0)
   {
      return;
   }

   /// New approach
   state_bar.GetSubVector(ess_tdof_list, form.scratch);
   state_bar.SetSubVector(ess_tdof_list, 0.0);
   adjoint.SetSubVector(ess_tdof_list, 0.0);
}

void finalizeAdjointSystem(MISONonlinearForm &form,
                           mfem::Solver &adj_solver,
                           const MISOInputs &inputs,
                           mfem::Vector &state_bar,
                           mfem::Vector &adjoint)
{
   const auto &ess_tdof_list = form.getEssentialDofs();
   if (ess_tdof_list.Size() == 0)
   {
      return;
   }

   /// New approach
   adjoint.SetSubVector(ess_tdof_list, form.scratch);

   auto *hypre_jac_e = form.jac_e.As<mfem::HypreParMatrix>();
   if (hypre_jac_e == nullptr)
   {
      throw MISOException(
          "setUpAdjointSystem (MISONonlinearForm) only supports "
          "Jacobian matrices assembled to a HypreParMatrix!\n");
   }

   hypre_jac_e->MultTranspose(1.0, adjoint, -1.0, adjoint);
}

double jacobianVectorProduct(MISONonlinearForm &form,
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

void jacobianVectorProduct(MISONonlinearForm &form,
                           const mfem::Vector &wrt_dot,
                           const std::string &wrt,
                           mfem::Vector &res_dot)
{
   if (wrt == "state")
   {
      form.scratch = wrt_dot;
      const auto &ess_tdof_list = form.getEssentialDofs();
      form.scratch.SetSubVector(ess_tdof_list, 0.0);

      auto *hypre_jac = form.jac.As<mfem::HypreParMatrix>();
      if (hypre_jac == nullptr)
      {
         throw MISOException(
             "getJacobianTranspose (MISONonlinearForm) only supports "
             "Jacobian matrices assembled to a HypreParMatrix!\n");
      }
      hypre_jac->Mult(1.0, form.scratch, 1.0, res_dot);

      auto *hypre_jac_e = form.jac_e.As<mfem::HypreParMatrix>();
      if (hypre_jac_e == nullptr)
      {
         throw MISOException(
             "setUpAdjointSystem (MISONonlinearForm) only supports "
             "Jacobian matrices assembled to a HypreParMatrix!\n");
      }
      hypre_jac_e->Mult(1.0, wrt_dot, 1.0, res_dot);
   }
   else if (form.fwd_sens.count(wrt) != 0)
   {
      throw NotImplementedException(
          "not implemented for vector sensitivities (except for state)!\n");
   }
}

double vectorJacobianProduct(MISONonlinearForm &form,
                             const mfem::Vector &res_bar,
                             const std::string &wrt)
{
   if (form.rev_scalar_sens.count(wrt) != 0)
   {
      /// Integrators added to rev_scalar_sens will reference the adjoint grid
      /// func so we update it here
      auto &adjoint = form.nf_fields.at(form.adjoint_name);
      adjoint.distributeSharedDofs(res_bar);

      /// The state must have previously been distributed before calling this
      /// function
      auto &state = form.nf_fields.at("state").gridFunc();
      return form.rev_scalar_sens.at(wrt).GetGridFunctionEnergy(state);
   }
   return 0.0;
}

void vectorJacobianProduct(MISONonlinearForm &form,
                           const mfem::Vector &res_bar,
                           const std::string &wrt,
                           mfem::Vector &wrt_bar)
{
   if (wrt == "state")
   {
      form.scratch = res_bar;
      const auto &ess_tdof_list = form.getEssentialDofs();
      form.scratch.SetSubVector(ess_tdof_list, 0.0);

      auto *hypre_jac = form.jac.As<mfem::HypreParMatrix>();
      if (hypre_jac == nullptr)
      {
         throw MISOException(
             "getJacobianTranspose (MISONonlinearForm) only supports "
             "Jacobian matrices assembled to a HypreParMatrix!\n");
      }
      // hypre_jac->MultTranspose(1.0, res_bar, 1.0, wrt_bar);
      hypre_jac->MultTranspose(1.0, form.scratch, 1.0, wrt_bar);
      // form.scratch.SetSize(res_bar.Size());
      // hypre_jac->MultTranspose(1.0, res_bar, 1.0, form.scratch);
      // form.scratch.SetSubVector(ess_tdof_list, 0.0);

      auto *hypre_jac_e = form.jac_e.As<mfem::HypreParMatrix>();
      if (hypre_jac_e == nullptr)
      {
         throw MISOException(
             "setUpAdjointSystem (MISONonlinearForm) only supports "
             "Jacobian matrices assembled to a HypreParMatrix!\n");
      }
      hypre_jac_e->MultTranspose(1.0, res_bar, 1.0, wrt_bar);

      // form.adjoint_jac_trans->Mult(res_bar, wrt_bar);
   }
   else if (form.rev_sens.count(wrt) != 0)
   {
      form.scratch.SetSize(res_bar.Size());
      form.scratch = res_bar;
      const auto &ess_tdof_list = form.getEssentialDofs();
      form.scratch.SetSubVector(ess_tdof_list, 0.0);

      /// Integrators added to rev_sens will reference the adjoint, grid func
      /// so we update it here
      auto &adjoint = form.nf_fields.at(form.adjoint_name);
      adjoint.distributeSharedDofs(form.scratch);

      /// Integrators added to rev_sens will also reference the state grid func,
      /// so that must have been distributed before calling this function
      auto &wrt_rev_sens = form.rev_sens.at(wrt);
      wrt_rev_sens.Assemble();
      form.scratch.SetSize(wrt_bar.Size());
      form.scratch = 0.0;
      wrt_rev_sens.ParallelAssemble(form.scratch);

      // mfem::Array<int> wrt_ess_tdof_list;
      // wrt_rev_sens.ParFESpace()->GetEssentialTrueDofs(form.ess_bdr,
      // wrt_ess_tdof_list); for (int i = 0; i < wrt_ess_tdof_list.Size(); ++i)
      // {
      //    form.scratch(wrt_ess_tdof_list[i]) = 0.0;
      // }
      wrt_bar += form.scratch;
   }
}

}  // namespace miso
