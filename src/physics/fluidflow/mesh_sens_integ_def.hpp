#ifndef MACH_MESH_SENS_INTEG_DEF
#define MACH_MESH_SENS_INTEG_DEF

#include "mfem.hpp"

#include "sbp_fe.hpp"

namespace mach
{
template <typename Derived>
void DyadicMeshSensIntegrator<Derived>::AssembleRHSElementVect(
    const mfem::FiniteElement &el,
    mfem::ElementTransformation &trans,
    mfem::Vector &elvect)
{
   using namespace mfem;

   // reverse-diff functions we need are only defined for IsoparametricTrans
   IsoparametricTransformation &isotrans =
       dynamic_cast<IsoparametricTransformation &>(trans);
   // extract the relevant sbp operator for this element
   const FiniteElementSpace *fes =
       state.FESpace();  // Should check that fes match with adjoint
   const FiniteElement *fe = fes->GetFE(trans.ElementNo);
   const SBPFiniteElement &sbp = dynamic_cast<const SBPFiniteElement &>(*fe);
   // extract the state and adjoint values for this element
   const IntegrationRule &ir = sbp.GetNodes();
   DenseMatrix u, psi;
   state.GetVectorValues(isotrans, ir, u);
   adjoint.GetVectorValues(isotrans, ir, psi);

   int num_nodes = sbp.GetDof();  // number of state dofs
   int ndof = el.GetDof();        // number of coord node dofs != num_nodes
   int dim = el.GetDim();
   Vector u_i, u_j, psi_i, psi_j;
#ifdef MFEM_THREAD_SAFE
   Vector fluxij;
   DenseMatrix adjJ_i_bar, adjJ_j_bar, PointMat_bar;
#endif
   fluxij.SetSize(num_states);
   adjJ_i_bar.SetSize(dim);
   adjJ_j_bar.SetSize(dim);
   elvect.SetSize(dim * ndof);
   PointMat_bar.SetSize(dim, ndof);  // PointMat_bar = dfdx
   PointMat_bar = 0.0;
   for (int i = 0; i < num_nodes; ++i)
   {
      u.GetColumnReference(i, u_i);
      psi.GetColumnReference(i, psi_i);
      adjJ_i_bar = 0.0;  // prepare adjJ_i_bar for accumulation
      for (int j = i + 1; j < num_nodes; ++j)
      {
         u.GetColumnReference(j, u_j);
         psi.GetColumnReference(j, psi_j);
         adjJ_j_bar = 0.0;  // prepare adjJ_j_bar for accumulation
         for (int di = 0; di < dim; ++di)
         {
            flux(di, u_i, u_j, fluxij);
            // begin reverse sweep...
            double Sij_bar = 0.0;
            for (int n = 0; n < num_states; ++n)
            {
               // res(i,n) += Sij*fluxij(n);
               Sij_bar += fluxij(n) * psi_i(n);
               // res(j,n) -= Sij*fluxij(n);
               Sij_bar -= fluxij(n) * psi_j(n);
            }
            Sij_bar *= alpha;
            // double Sij = sbp.getSkewEntry(di, i, j, adjJ_i, adjJ_j);
            sbp.getSkewEntryRevDiff(di, i, j, Sij_bar, adjJ_i_bar, adjJ_j_bar);
         }  // di loop
         isotrans.SetIntPoint(&ir.IntPoint(j));
         // adjJ_j = isotrans.AdjugateJacobian();
         isotrans.AdjugateJacobianRevDiff(adjJ_j_bar, PointMat_bar);
      }  // j node loop
      isotrans.SetIntPoint(&ir.IntPoint(i));
      // adjJ_i = isotrans.AdjugateJacobian();
      isotrans.AdjugateJacobianRevDiff(adjJ_i_bar, PointMat_bar);
   }  // i node loop

   // Insert PointMat_bar = dfdx into elvect
   for (int i = 0; i < ndof; ++i)
   {
      for (int d = 0; d < dim; ++d)
      {
         elvect(d * ndof + i) = PointMat_bar(d, i);
      }
   }
}

template <typename Derived>
void BoundaryMeshSensIntegrator<Derived>::AssembleRHSElementVect(
    const mfem::FiniteElement &el,
    mfem::ElementTransformation &trans,
    mfem::Vector &elvect)
{
   throw MachException(
       "BoundaryMeshSensIntegrator::AssembleRHSElementVect()\n"
       "\tUse AddBdrFaceIntegrator not AddBoundaryIntegrator");

   // TODO: here I was trying to use GetBdrElementAdjacentElement to
   // find the element and then its element transformation...the problem is the
   // member of the face transformation... we don't have this here.

   //    using namespace mfem;

   //    // reverse-diff functions we need are only defined for
   //    IsoparametricTrans IsoparametricTransformation &isotrans =
   //      dynamic_cast<IsoparametricTransformation&>(*trans.Elem1);
   //    // extract the relevant sbp operator for this element
   //    const FiniteElementSpace *fes = state.FESpace(); // Should check that
   //    fes match with adjoint int elem, info;
   //    fes->GetMesh()->GetBdrElementAdjacentElement(trans.ElementNo, elem,
   //    info); const FiniteElement *fe = fes->GetFE(elem); const
   //    SBPFiniteElement &sbp = dynamic_cast<const SBPFiniteElement&>(*fe);
   //    ElementTransformation *elem_trans =
   //    fes->GetElementTransformation(elem);
   //    // extract the state and adjoint values for this element
   //    const IntegrationRule& ir = sbp.GetNodes();
   //    DenseMatrix u, psi;
   //    state.GetVectorValues(isotrans, ir, u);
   //    adjoint.GetVectorValues(isotrans, ir, psi);

   //    int ndof = el.GetDof(); // number mesh dofs != num sbp nodes, in
   //    general int dim = trans.GetDimension(); int space_dim =
   //    trans.GetSpaceDim(); Vector u_face, psi_face; // references only, no
   //    allocation
   // #ifdef MFEM_THREAD_SAFE
   //    Vector x, nrm, nrm_bar;
   //    DenseMatrix Jac_map, Jac_bar, Jac_face_bar;
   //    DenseMatrix PointMat_bar;
   // #endif
   //    x.SetSize(space_dim);
   //    nrm.SetSize(space_dim);
   //    nrm_bar.SetSize(space_dim);
   // 	elvect.SetSize(space_dim*ndof);
   //    Jac_map.SetSize(space_dim, dim);
   //    Jac_bar.SetSize(space_dim);
   //    Jac_face_bar.SetSize(space_dim, dim);
   //    PointMat_bar.SetSize(space_dim, ndof); // PointMat_bar = dfdx
   //    PointMat_bar = 0.0;

   //    const mfem::FiniteElementCollection *fec = fes->FEColl();
   //    const FiniteElement *sbp_face;
   //    switch (space_dim)
   //    {
   //       case 1: sbp_face = fec->FiniteElementForGeometry(Geometry::POINT);
   //               break;
   //       case 2: sbp_face = fec->FiniteElementForGeometry(Geometry::SEGMENT);
   //               break;
   //       default: throw mach::MachException(
   //          "BoundaryMeshSensIntegrator::AssembleFaceVector())\n"
   //          "\tcannot handle given dimension");
   //    }
   //    IntegrationPoint el_ip;
   //    for (int i = 0; i < sbp_face->GetDof(); ++i)
   //    {
   //       // get the face and element integration points
   //       const IntegrationPoint &face_ip = sbp_face->GetNodes().IntPoint(i);
   //       //trans.Loc1.Transform(face_ip, el_ip);
   //       trans.SetIntPoint(&face_ip);
   //       //trans.Elem1->SetIntPoint(&el_ip);

   //       // Find Jac_map, the linear map from the element to the face
   //       Jacobian
   //       //CalcInverse(trans.Elem1->Jacobian(), Jac_bar); // use Jac_bar for
   //       inv
   //       //Mult(Jac_bar, trans.Face->Jacobian(), Jac_map);

   //       // Get the state and adjoint for this face node
   //       int j = sbp.getIntegrationPointIndex(el_ip);
   //       u.GetColumnReference(j, u_face);
   //       psi.GetColumnReference(j, psi_face);

   //       // get the physical coordinate and normal vector
   //       trans.Elem1->Transform(el_ip, x);
   //       CalcOrtho(trans.Face->Jacobian(), nrm);

   //       // start reverse sweep
   //       // flux(x, nrm, u_face, flux_face);
   //       fluxBar(x, nrm, u_face, psi_face, nrm_bar);

   //       // flux_face *= face_ip.weight;
   //       nrm_bar *= face_ip.weight*alpha;
   //       // CalcOrtho(trans.Face->Jacobian(), nrm);
   //       CalcOrthoRevDiff(trans.Face->Jacobian(), nrm_bar, Jac_face_bar);
   //       MultABt(Jac_face_bar, Jac_map, Jac_bar);
   //       isotrans.JacobianRevDiff(Jac_bar, PointMat_bar);
   //    }
   //    // Insert PointMat_bar = dfdx into elvect
   //    for (int i = 0; i < ndof; ++i)
   //    {
   //       for (int d = 0; d < space_dim; ++d)
   //       {
   //          elvect(d*ndof + i) = PointMat_bar(d, i);
   //       }
   //    }
}

template <typename Derived>
void BoundaryMeshSensIntegrator<Derived>::AssembleRHSElementVect(
    const mfem::FiniteElement &el_bnd,
    mfem::FaceElementTransformations &trans,
    mfem::Vector &elvect)
{
   using namespace mfem;

   // reverse-diff functions we need are only defined for IsoparametricTrans
   IsoparametricTransformation &isotrans =
       dynamic_cast<IsoparametricTransformation &>(*trans.Elem1);
   // extract the relevant sbp operator for this element
   const FiniteElementSpace *fes =
       state.FESpace();  // Should check that fes match with adjoint
   const FiniteElement *fe = fes->GetFE(trans.Elem1No);
   const SBPFiniteElement &sbp = dynamic_cast<const SBPFiniteElement &>(*fe);
   // extract the state and adjoint values for this element
   const IntegrationRule &ir = sbp.GetNodes();
   DenseMatrix u, psi;
   state.GetVectorValues(isotrans, ir, u);
   adjoint.GetVectorValues(isotrans, ir, psi);

   int ndof = el_bnd.GetDof();  // number mesh dofs != num sbp nodes, in general
   int dim = trans.Face->GetDimension();
   int space_dim = trans.Face->GetSpaceDim();
   Vector u_face, psi_face;  // references only, no allocation
#ifdef MFEM_THREAD_SAFE
   Vector x, nrm, nrm_bar;
   DenseMatrix Jac_map, Jac_bar, Jac_face_bar;
   DenseMatrix PointMat_bar;
#endif
   x.SetSize(space_dim);
   nrm.SetSize(space_dim);
   nrm_bar.SetSize(space_dim);
   elvect.SetSize(space_dim * ndof);
   Jac_map.SetSize(space_dim, dim);
   Jac_bar.SetSize(space_dim);
   Jac_face_bar.SetSize(space_dim, dim);
   PointMat_bar.SetSize(space_dim, ndof);  // PointMat_bar = dfdx
   PointMat_bar = 0.0;

   const mfem::FiniteElementCollection *fec = fes->FEColl();
   const FiniteElement *sbp_face;
   switch (space_dim)
   {
   case 1:
      sbp_face = fec->FiniteElementForGeometry(Geometry::POINT);
      break;
   case 2:
      sbp_face = fec->FiniteElementForGeometry(Geometry::SEGMENT);
      break;
   default:
      throw mach::MachException(
          "BoundaryMeshSensIntegrator::AssembleFaceVector())\n"
          "\tcannot handle given dimension");
   }
   IntegrationPoint el_ip;
   for (int i = 0; i < sbp_face->GetDof(); ++i)
   {
      // get the face and element integration points
      const IntegrationPoint &face_ip = sbp_face->GetNodes().IntPoint(i);
      trans.Loc1.Transform(face_ip, el_ip);
      trans.Face->SetIntPoint(&face_ip);
      trans.Elem1->SetIntPoint(&el_ip);

      // Find Jac_map, the linear map from the element to the face Jacobian
      CalcInverse(trans.Elem1->Jacobian(), Jac_bar);  // use Jac_bar for inv
      Mult(Jac_bar, trans.Face->Jacobian(), Jac_map);

      // Get the state and adjoint for this face node
      int j = sbp.getIntegrationPointIndex(el_ip);
      u.GetColumnReference(j, u_face);
      psi.GetColumnReference(j, psi_face);

      // get the physical coordinate and normal vector
      trans.Elem1->Transform(el_ip, x);
      CalcOrtho(trans.Face->Jacobian(), nrm);

      // start reverse sweep
      // flux(x, nrm, u_face, flux_face);
      fluxBar(x, nrm, u_face, psi_face, nrm_bar);

      // flux_face *= face_ip.weight;
      nrm_bar *= face_ip.weight * alpha;
      // CalcOrtho(trans.Face->Jacobian(), nrm);
      CalcOrthoRevDiff(trans.Face->Jacobian(), nrm_bar, Jac_face_bar);
      MultABt(Jac_face_bar, Jac_map, Jac_bar);
      isotrans.JacobianRevDiff(Jac_bar, PointMat_bar);
   }
   // Insert PointMat_bar = dfdx into elvect
   for (int i = 0; i < ndof; ++i)
   {
      for (int d = 0; d < space_dim; ++d)
      {
         elvect(d * ndof + i) = PointMat_bar(d, i);
      }
   }
}

}  // namespace mach

#endif
