template <typename Derived>
void DyadicMeshSensIntegrator<Derived>::AssembleRHSElementVect(
    const mfem::FiniteElement &el,
    mfem::ElementTransformation &trans,
    mfem::Vector &elvect)
{
   using namespace mfem;

   // reverse-diff functions we need are only defined for IsoparametricTrans
   IsoparametricTransformation &isotrans = 
     dynamic_cast<IsoparametricTransformation&>(trans);
   // extract the relevant sbp operator for this element
   const FiniteElementSpace *fes = state.FESpace(); // Should check that fes match with adjoint
   const FiniteElement *fe = fes->GetFE(trans.ElementNo);
   const SBPFiniteElement &sbp = dynamic_cast<const SBPFiniteElement&>(*fe);
   // extract the state and adjoint values for this element
   const IntegrationRule& ir = sbp.GetNodes();
   DenseMatrix u, psi;
   state.GetVectorValues(isotrans, ir, u);
   adjoint.GetVectorValues(isotrans, ir, psi);

   int num_nodes = sbp.GetDof(); // number of state dofs
   int ndof = el.GetDof(); // number of coord node dofs != num_nodes
   int dim = el.GetDim();
   Vector u_i, u_j, psi_i, psi_j;
#ifdef MFEM_THREAD_SAFE
   Vector fluxij;
   DenseMatrix adjJ_i_bar, adjJ_j_bar, PointMat_bar;
#endif
   fluxij.SetSize(num_states);
   adjJ_i_bar.SetSize(dim);
   adjJ_j_bar.SetSize(dim);
	elvect.SetSize(dim*ndof);
   PointMat_bar.SetSize(dim, ndof); // PointMat_bar = dfdx
   PointMat_bar = 0.0;
   for (int i = 0; i < num_nodes; ++i)
   {
      u.GetColumnReference(i, u_i);
      psi.GetColumnReference(i, psi_i);
      adjJ_i_bar = 0.0; // prepare adjJ_i_bar for accumulation
		for (int j = i+1; j < num_nodes; ++j)
		{
         u.GetColumnReference(j, u_j);
         psi.GetColumnReference(j, psi_j);
         adjJ_j_bar = 0.0; // prepare adjJ_j_bar for accumulation         
			for (int di = 0; di < dim; ++di)
			{
            flux(di, u_i, u_j, fluxij);
            // begin reverse sweep...
            double Sij_bar = 0.0;
            for (int n = 0; n < num_states; ++n)
            {
               // res(i,n) += Sij*fluxij(n);
               Sij_bar += fluxij(n)*psi_i(n);
               // res(j,n) -= Sij*fluxij(n);
               Sij_bar -= fluxij(n)*psi_j(n);
            }
            Sij_bar *= alpha;
            // double Sij = sbp.getSkewEntry(di, i, j, adjJ_i, adjJ_j);
            sbp.getSkewEntryRevDiff(di, i, j, Sij_bar, adjJ_i_bar, adjJ_j_bar);
			} // di loop
         isotrans.SetIntPoint(&ir.IntPoint(j));
         // adjJ_j = isotrans.AdjugateJacobian();
         isotrans.AdjugateJacobianRevDiff(adjJ_j_bar, PointMat_bar);
      } // j node loop
      isotrans.SetIntPoint(&ir.IntPoint(i));
      // adjJ_i = isotrans.AdjugateJacobian();
      isotrans.AdjugateJacobianRevDiff(adjJ_i_bar, PointMat_bar);
   } // i node loop

   // Insert PointMat_bar = dfdx into elvect
   for (int i = 0; i < ndof; ++i)
   {
      for (int d = 0; d < dim; ++d)
      {
         elvect(d*ndof + i) = PointMat_bar(d, i);
      }
   }
}


template <typename Derived>
void SymmetricViscousMeshSensIntegrator<Derived>::AssembleRHSElementVect(
    const mfem::FiniteElement &el,
    mfem::ElementTransformation &trans,
    mfem::Vector &elvect)
{
   using namespace mfem;

   // reverse-diff functions we need are only defined for IsoparametricTrans
   IsoparametricTransformation &isotrans = 
     dynamic_cast<IsoparametricTransformation&>(trans);
   // extract the relevant sbp operator for this element
   const FiniteElementSpace *fes = state.FESpace(); // Should check that fes match with adjoint
   const FiniteElement *fe = fes->GetFE(trans.ElementNo);
   const SBPFiniteElement &sbp = dynamic_cast<const SBPFiniteElement&>(*fe);
   // extract the state and adjoint values for this element
   const IntegrationRule& ir = sbp.GetNodes();
   DenseMatrix u, psi, Dwi, Dpsi;
   state.GetVectorValues(isotrans, ir, u);
   adjoint.GetVectorValues(isotrans, ir, psi);
   u.Transpose(); psi.Transpose();

   int num_nodes = sbp.GetDof(); // number of state dofs
   int ndof = el.GetDof(); // number of coord node dofs != num_nodes
   int dim = el.GetDim();
   Vector xi, ui, uj, wi, wj, psii, psj, CDwi, CDpsi;
#ifdef MFEM_THREAD_SAFE
   Vector fluxij;
   DenseMatrix adjJ_i_bar, adjJ_k_bar, PointMat_bar;
#endif
   fluxij.SetSize(num_states);
   xi.SetSize(dim);
   wj.SetSize(num_states);
   ui.SetSize(num_states);
   uj.SetSize(num_states);
   psii.SetSize(num_states);
   psj.SetSize(num_states);
   Dwi.SetSize(num_states,dim);
   Dpsi.SetSize(num_states,dim);
   CDwi.SetSize(num_states);
   CDpsi.SetSize(num_states);
   adjJ_i.SetSize(dim);
   adjJ_j.SetSize(dim);
   adjJ_k.SetSize(dim);
   adjJ_i_bar.SetSize(dim);
   adjJ_k_bar.SetSize(dim);
	elvect.SetSize(dim*ndof);
   PointMat_bar.SetSize(dim, ndof); // PointMat_bar = dfdx
   PointMat_bar = 0.0;
   for (int i = 0; i < num_nodes; ++i)
   {
      // get the Jacobian (trans.Weight) and cubature weight (node.weight)
      const IntegrationPoint &node = sbp.GetNodes().IntPoint(i);
      isotrans.SetIntPoint(&node);
      trans.SetIntPoint(&node);
      double Hinv = 1.0 / (sbp.getDiagNormEntry(i) * trans.Weight());
      CalcAdjugate(trans.Jacobian(), adjJ_i);
      u.GetRow(i, ui);
      psi.GetRow(i, psii);
      trans.Transform(node, xi);

      // compute Dwi and Dpsi derivatives at node i
      Dwi = 0.0; Dpsi = 0.0;
      for (int j = 0; j < num_nodes; ++j)
      {
         // Get mapping Jacobian adjugate and transform state to entropy vars
         trans.SetIntPoint(&sbp.GetNodes().IntPoint(j));
         CalcAdjugate(trans.Jacobian(), adjJ_j);
         u.GetRow(j, uj);  psi.GetRow(j, psj);
         convert(uj, wj);
         for (int d = 0; d < dim; ++d)
         {
            double Qij = sbp.getQEntry(d, i, j, adjJ_i, adjJ_j);
            for (int s = 0; s < num_states; ++s)
            {
               Dwi(s,d) += Qij * wj(s);
               Dpsi(s,d) += Qij * psj(s);
            }
         } // loop over space dimensions d
      } // loop over element nodes j
      Dwi *= Hinv;
      Dpsi *= Hinv;
      double dHinv = -1.0 / (trans.Weight()); // multiply Dwi by this afterward

      // next, scale the derivatives (e.g. using \hat{C} matrices), and then 
      // apply adjugate jacobian sensitivities
      adjJ_i_bar = 0.0; // prepare adjJ_i_bar for accumulation      
      for (int d = 0; d < dim; ++d) 
      {
         scale(d, xi, ui, Dwi, CDwi);
         scale(d, xi, ui, Dpsi, CDpsi);

         for (int k = 0; k < num_nodes; ++k) 
         {
            adjJ_k_bar = 0.0;
            trans.SetIntPoint(&sbp.GetNodes().IntPoint(k));
            CalcAdjugate(trans.Jacobian(), adjJ_k);
            //double Qik = sbp.getQEntry(d, i, k, adjJ_i, adjJ_k);
            // apply transposed derivative in `d` direction
            // this evaluates Q_d'*(C_d,1 D_1 + C_d,2 D_2 + ...) w
            // begin reverse sweep...
            double Sij_u_bar = 0.0;
            double Sij_psi_bar = 0.0;
            for (int s = 0; s < num_states; ++s)
            {
               // res(k, s) += alpha * Qik * CDwi(s);
               Sij_u_bar += alpha*CDwi(s)*psii(s);
               Sij_psi_bar += alpha*CDpsi(s)*ui(s);
            }
            sbp.getQEntryRevDiff(d, i, k, Sij_u_bar, adjJ_i_bar, adjJ_k_bar);
            sbp.getQEntryRevDiff(d, i, k, Sij_psi_bar, adjJ_i_bar, adjJ_k_bar);
            isotrans.SetIntPoint(&ir.IntPoint(k));
            // adjJ_i = isotrans.AdjugateJacobian();
            isotrans.AdjugateJacobianRevDiff(adjJ_k_bar, PointMat_bar);
         }
      } // loop over space dimensions d
      isotrans.SetIntPoint(&ir.IntPoint(i));
      // adjJ_i = isotrans.AdjugateJacobian();
      isotrans.AdjugateJacobianRevDiff(adjJ_i_bar, PointMat_bar);

      // Hinv sensitivity: compute dHinv, and accumulate energy in H_bar
      Dwi *= dHinv;
      double H_bar = 0.0;
      for (int d = 0; d < dim; ++d) {
         scale(d, xi, ui, Dwi, CDwi);
         for (int k = 0; k < num_nodes; ++k) 
         {
            psi.GetRow(k, psj);
            // Get mapping Jacobian adjugate
            trans.SetIntPoint(&sbp.GetNodes().IntPoint(k));
            CalcAdjugate(trans.Jacobian(), adjJ_k);
            double Qik = sbp.getQEntry(d, i, k, adjJ_i, adjJ_k);
            // apply transposed derivative in `d` direction
            for (int s = 0; s < num_states; ++s)
            {
               H_bar += alpha * Qik * CDwi(s)*psj(s); 
            }
         } // loop over element nodes k
      } // loop over space dimensions d
      DenseMatrix PointMat_bar_w(dim, ndof);
      PointMat_bar_w = 0.0;
      isotrans.WeightRevDiff(PointMat_bar_w);

      // add to total sensitivity
      PointMat_bar.Add(H_bar, PointMat_bar_w);
   } // i node loop



   // Insert PointMat_bar = dfdx into elvect
   for (int i = 0; i < ndof; ++i)
   {
      for (int d = 0; d < dim; ++d)
      {
         elvect(d*ndof + i) = PointMat_bar(d, i);
      }
   }
}



template <typename Derived>
void BoundaryMeshSensIntegrator<Derived>::AssembleRHSElementVect(
   const mfem::FiniteElement &el, mfem::ElementTransformation &trans,
   mfem::Vector &elvect)
{
   throw MachException("BoundaryMeshSensIntegrator::AssembleRHSElementVect()\n"
                       "\tUse AddBdrFaceIntegrator not AddBoundaryIntegrator");

   // TODO: here I was trying to use GetBdrElementAdjacentElement to
   // find the element and then its element transformation...the problem is the
   // member of the face transformation... we don't have this here.   

//    using namespace mfem;

//    // reverse-diff functions we need are only defined for IsoparametricTrans
//    IsoparametricTransformation &isotrans = 
//      dynamic_cast<IsoparametricTransformation&>(*trans.Elem1);
//    // extract the relevant sbp operator for this element
//    const FiniteElementSpace *fes = state.FESpace(); // Should check that fes match with adjoint
//    int elem, info;
//    fes->GetMesh()->GetBdrElementAdjacentElement(trans.ElementNo, elem, info);
//    const FiniteElement *fe = fes->GetFE(elem);
//    const SBPFiniteElement &sbp = dynamic_cast<const SBPFiniteElement&>(*fe);
//    ElementTransformation *elem_trans = fes->GetElementTransformation(elem);
//    // extract the state and adjoint values for this element
//    const IntegrationRule& ir = sbp.GetNodes();
//    DenseMatrix u, psi;
//    state.GetVectorValues(isotrans, ir, u);
//    adjoint.GetVectorValues(isotrans, ir, psi);

//    int ndof = el.GetDof(); // number mesh dofs != num sbp nodes, in general
//    int dim = trans.GetDimension();
//    int space_dim = trans.GetSpaceDim();
//    Vector u_face, psi_face; // references only, no allocation
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

//       // Find Jac_map, the linear map from the element to the face Jacobian
//       //CalcInverse(trans.Elem1->Jacobian(), Jac_bar); // use Jac_bar for inv
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
   const mfem::FiniteElement &el_bnd, mfem::FaceElementTransformations &trans,
   mfem::Vector &elvect)
{
   using namespace mfem;

   // reverse-diff functions we need are only defined for IsoparametricTrans
   IsoparametricTransformation &isotrans = 
     dynamic_cast<IsoparametricTransformation&>(*trans.Elem1);
   // extract the relevant sbp operator for this element
   const FiniteElementSpace *fes = state.FESpace(); // Should check that fes match with adjoint
   const FiniteElement *fe = fes->GetFE(trans.Elem1No);
   const SBPFiniteElement &sbp = dynamic_cast<const SBPFiniteElement&>(*fe);
   // extract the state and adjoint values for this element
   const IntegrationRule& ir = sbp.GetNodes();
   DenseMatrix u, psi;
   state.GetVectorValues(isotrans, ir, u);
   adjoint.GetVectorValues(isotrans, ir, psi);

   int ndof = el_bnd.GetDof(); // number mesh dofs != num sbp nodes, in general
   int dim = trans.Face->GetDimension();
   int space_dim = trans.Face->GetSpaceDim();
   Vector u_face, psi_face; // references only, no allocation
#ifdef MFEM_THREAD_SAFE
   Vector x, nrm, nrm_bar;
   DenseMatrix Jac_map, Jac_bar, Jac_face_bar;
   DenseMatrix PointMat_bar;
#endif
   x.SetSize(space_dim);
   nrm.SetSize(space_dim);
   nrm_bar.SetSize(space_dim);
	elvect.SetSize(space_dim*ndof);
   Jac_map.SetSize(space_dim, dim);
   Jac_bar.SetSize(space_dim);
   Jac_face_bar.SetSize(space_dim, dim);
   PointMat_bar.SetSize(space_dim, ndof); // PointMat_bar = dfdx
   PointMat_bar = 0.0;

   const mfem::FiniteElementCollection *fec = fes->FEColl();
   const FiniteElement *sbp_face;
   switch (space_dim)
   {
      case 1: sbp_face = fec->FiniteElementForGeometry(Geometry::POINT);
              break;
      case 2: sbp_face = fec->FiniteElementForGeometry(Geometry::SEGMENT);
              break;
      default: throw mach::MachException(
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
      CalcInverse(trans.Elem1->Jacobian(), Jac_bar); // use Jac_bar for inv
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
      nrm_bar *= face_ip.weight*alpha;
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
         elvect(d*ndof + i) = PointMat_bar(d, i);
      }
   }
}
