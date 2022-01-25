#include "mach_input.hpp"

#include "mixed_nonlinear_operator.hpp"

namespace mach
{

void MixedNonlinearOperator::apply(const MachInputs &inputs,
                                   mfem::Vector &out_vec)
{
   out_vec = 0.0;
   if (!operation)
   {
      return;
   }

   mfem::Vector state;
   setVectorFromInputs(inputs, "state", state, false, true);

   domain.distributeSharedDofs(state);

   const auto &domain_fes = domain.space();
   const auto &range_fes = range.space();
   mfem::Array<int> domain_vdofs;
   mfem::Array<int> range_vdofs;
   mfem::Vector el_domain;
   mfem::Vector el_range;

   for (int i = 0; i < range_fes.GetNE(); ++i)
   {
      const auto &domain_fe = *domain_fes.GetFE(i);
      const auto &range_fe = *range_fes.GetFE(i);
      auto &trans = *range_fes.GetElementTransformation(i);

      auto *domain_dof_trans = domain_fes.GetElementVDofs(i, domain_vdofs);
      el_domain.SetSize(domain_vdofs.Size());
      auto *range_dof_trans = range_fes.GetElementVDofs(i, range_vdofs);
      el_range.SetSize(range_vdofs.Size());

      domain.gridFunc().GetSubVector(domain_vdofs, el_domain);
      if (domain_dof_trans)
      {
         domain_dof_trans->InvTransformPrimal(el_domain);
      }

      /// apply the operation
      operation(domain_fe, range_fe, trans, el_domain, el_range);

      if (range_dof_trans)
      {
         range_dof_trans->TransformPrimal(el_range);
      }
      range.gridFunc().AddElementVector(range_vdofs, el_range);
   }

   range.setTrueVec(out_vec);
}

}  // namespace mach
