#include "mfem.hpp"

#include "mesh_move_integ.hpp"

using namespace mfem;

namespace miso
{
void ElasticityPositionIntegrator::AssembleElementVector(
    const FiniteElement &el,
    ElementTransformation &trans,
    const Vector &elfun,
    Vector &elvect)
{
   DenseMatrix elmat;
   AssembleElementMatrix(el, trans, elmat);

   Vector disp(elfun);

   auto &iso_trans = *dynamic_cast<IsoparametricTransformation *>(&trans);
   auto point_mat = iso_trans.GetPointMat();
   point_mat.Transpose();
   Vector point_vec(point_mat.GetData(),
                    point_mat.Height() * point_mat.Width());

   disp -= point_vec;

   elvect.SetSize(elmat.Height());
   elmat.Mult(disp, elvect);
}

void ElasticityPositionIntegratorStateRevSens::AssembleRHSElementVect(
    const mfem::FiniteElement &el,
    mfem::ElementTransformation &trans,
    mfem::Vector &state_bar)
{
   /// get the proper element, transformation, and adjoint vector
   int element = trans.ElementNo;
   auto *dof_tr = adjoint.FESpace()->GetElementVDofs(element, vdofs);
   adjoint.GetSubVector(vdofs, psi);
   if (dof_tr != nullptr)
   {
      dof_tr->InvTransformPrimal(psi);
   }

   DenseMatrix elmat;
   integ.AssembleElementMatrix(el, trans, elmat);

   state_bar.SetSize(psi.Size());
   elmat.MultTranspose(psi, state_bar);
}

void ElasticityPositionIntegratorStateFwdSens::AssembleRHSElementVect(
    const mfem::FiniteElement &el,
    mfem::ElementTransformation &trans,
    mfem::Vector &res_dot)
{
   /// get the proper element, transformation, and state_dot vector
   int element = trans.ElementNo;
   auto *dof_tr = state_dot.FESpace()->GetElementVDofs(element, vdofs);
   state_dot.GetSubVector(vdofs, elfun_dot);
   if (dof_tr != nullptr)
   {
      dof_tr->InvTransformPrimal(elfun_dot);
   }

   DenseMatrix elmat;
   integ.AssembleElementMatrix(el, trans, elmat);

   res_dot.SetSize(elfun_dot.Size());
   elmat.Mult(elfun_dot, res_dot);
}

}  // namespace miso
