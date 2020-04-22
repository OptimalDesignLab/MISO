template <typename Derived>
void DyadicMeshSensIntegrator<Derived>::AssembleElementVector(
    const mfem::FiniteElement &el,
    mfem::ElementTransformation &trans,
    const mfem::Vector &elfun,
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

#if 0
template <typename Derived>
void InviscidBoundarySensitivity<Derived>::AssembleFaceVector(
   const mfem::FiniteElement &el_bnd,
   const mfem::FiniteElement &el_unused,
   mfem::FaceElementTransformations &trans,
   const mfem::Vector &elfun,
   mfem::Vector &elvect)
{
   using namespace mfem;
   const SBPFiniteElement &sbp = dynamic_cast<const SBPFiniteElement&>(el_bnd);
   const int num_nodes = el_bnd.GetDof();
   const int dim = sbp.GetDim();
#ifdef MFEM_THREAD_SAFE
   Vector u_face, x, nrm, flux_face;
#endif
   u_face.SetSize(num_states);
   x.SetSize(dim);
   nrm.SetSize(dim);
   flux_face.SetSize(num_states);
   elvect.SetSize(num_states*num_nodes);
   elvect = 0.0;

   DenseMatrix u(elfun.GetData(), num_nodes, num_states);
   DenseMatrix res(elvect.GetData(), num_nodes, num_states);

   const FiniteElement *sbp_face;
   switch (dim)
   {
      case 1: sbp_face = fec->FiniteElementForGeometry(Geometry::POINT);
              break;
      case 2: sbp_face = fec->FiniteElementForGeometry(Geometry::SEGMENT);
              break;
      default: throw mach::MachException(
         "InviscidBoundaryIntegrator::AssembleFaceVector())\n"
         "\tcannot handle given dimension");
   }
   IntegrationPoint el_ip;
   for (int i = 0; i < sbp_face->GetDof(); ++i)
   {
      const IntegrationPoint &face_ip = sbp_face->GetNodes().IntPoint(i);
      trans.Loc1.Transform(face_ip, el_ip);
      trans.Elem1->Transform(el_ip, x);
      int j = sbp.getIntegrationPointIndex(el_ip);
      u.GetRow(j, u_face);

      // get the normal vector and the flux on the face
      trans.Face->SetIntPoint(&face_ip);
      CalcOrtho(trans.Face->Jacobian(), nrm);
      flux(x, nrm, u_face, flux_face);
      flux_face *= face_ip.weight;

      // multiply by test function
      for (int n = 0; n < num_states; ++n)
      {
         res(j, n) += alpha*flux_face(n);
      }
   }
}
#endif