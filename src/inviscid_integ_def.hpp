
template <typename Derived>
void InviscidIntegrator<Derived>::AssembleElementVector(
    const mfem::FiniteElement &el, mfem::ElementTransformation &Trans,
    const mfem::Vector &elfun, mfem::Vector &elvect)
{
   using namespace mfem;
   // This should be in a try/catch, but that creates other issues
   const SBPFiniteElement &sbp = dynamic_cast<const SBPFiniteElement&>(el);
   int num_nodes = sbp.GetDof();
   int dim = sbp.GetDim();
#ifdef MFEM_THREAD_SAFE
   Vector ui, fluxi, dxidx;
   DenseMatrix adjJ_i, elflux, elres;
#endif
	elvect.SetSize(num_states*num_nodes);
   ui.SetSize(num_states);
   adjJ_i.SetSize(dim);
   dxidx.SetSize(dim);
   elflux.SetSize(num_states, num_nodes);
   elres.SetSize(num_states, num_nodes);
   DenseMatrix u(elfun.GetData(), num_nodes, num_states);
   DenseMatrix res(elvect.GetData(), num_nodes, num_states);
   
   elres = 0.0;
   for (int di = 0; di < dim; ++di)
   {
      // get the flux at all the nodes      
      for (int i = 0; i < num_nodes; ++i)
      {
         Trans.SetIntPoint(&el.GetNodes().IntPoint(i));
         CalcAdjugate(Trans.Jacobian(), adjJ_i);
         adjJ_i.GetRow(di, dxidx);
         u.GetRow(i, ui);
         elflux.GetColumnReference(i, fluxi);
         flux(dxidx, ui, fluxi);
      }
      sbp.multWeakOperator(di, elflux, elres, true);
   }
   // This is necessary because data in elvect is expected to be ordered `byNODES`
   res.Transpose(elres);
   res *= alpha;
}

template <typename Derived>
void InviscidIntegrator<Derived>::AssembleElementGrad(
   const mfem::FiniteElement &el, mfem::ElementTransformation &Trans,
   const mfem::Vector &elfun, mfem::DenseMatrix &elmat)
{
   using namespace mfem;
   // This should be in a try/catch, but that creates other issues
   const SBPFiniteElement &sbp = dynamic_cast<const SBPFiniteElement &>(el);
   int num_nodes = sbp.GetDof();
   int dim = sbp.GetDim();
#ifdef MFEM_THREAD_SAFE
   Vector ui, dxidx;
   DenseMatrix adjJ_i, flux_jaci;
#endif
   elmat.SetSize(num_states*num_nodes);
   elmat = 0.0;
   ui.SetSize(num_states);
   adjJ_i.SetSize(dim);
   dxidx.SetSize(dim);
   flux_jaci.SetSize(num_states);
   DenseMatrix u(elfun.GetData(), num_nodes, num_states);
   for (int di = 0; di < dim; ++di)
   {  
      for (int i = 0; i < num_nodes; ++i)
      {
         // get the flux Jacobian at node i
         Trans.SetIntPoint(&el.GetNodes().IntPoint(i));
         CalcAdjugate(Trans.Jacobian(), adjJ_i);
         adjJ_i.GetRow(di, dxidx);
         u.GetRow(i, ui);
         fluxJacState(dxidx, ui, flux_jaci);

         // loop over rows j for contribution (Q^T)_{i,j} * Jac_i
         for (int j = 0; j < num_nodes; ++j)
         {
            // get the entry of (Q^T)_{j,i} = Q_{i,j}
            double Q = alpha*sbp.getQ(di, i, j);
            for (int n = 0; n < dim+2; ++n)
            {
               for (int m = 0; m < dim+2; ++m)
               {
                  elmat(m*num_nodes+j, n*num_nodes+i) -= Q*flux_jaci(m,n);
               }
            }
         }
      }
   }
}

template <typename Derived>
void DyadicFluxIntegrator<Derived>::AssembleElementGrad(
   const mfem::FiniteElement &el, mfem::ElementTransformation &Trans,
   const mfem::Vector &elfun, mfem::DenseMatrix &elmat)
{
   using namespace mfem;
   // This should be in a try/catch, but that creates other issues
   const SBPFiniteElement &sbp = dynamic_cast<const SBPFiniteElement &>(el);
   int num_nodes = sbp.GetDof();
   int dim = sbp.GetDim();
#ifdef MFEM_THREAD_SAFE
   Vector ui, uj, dxidx;
   DenseMatrix adjJ_i, adjJ_j, flux_jaci, flux_jacj;
#endif
   elmat.SetSize(num_states*num_nodes);
   elmat = 0.0;
   adjJ_i.SetSize(dim);
   adjJ_j.SetSize(dim);
   dxidx.SetSize(dim);
   flux_jaci.SetSize(num_states);
   flux_jacj.SetSize(num_states);
   DenseMatrix u(elfun.GetData(), num_nodes, num_states);
    for (int di = 0; di < dim; ++di)
   {  
      for (int i = 0; i < num_nodes; ++i)
      {
         // get the flux Jacobian at node i
         Trans.SetIntPoint(&el.GetNodes().IntPoint(i)); 
         CalcAdjugate(Trans.Jacobian(), adjJ_i);
         adjJ_i.GetRow(di, dxidx);
         u.GetRow(i, ui);
         // loop over rows j for contribution (Q^T)_{i,j} * Jac_i
         for (int j = i+1; j < num_nodes; ++j)
         {
            // get the flux Jacobian at node i
            Trans.SetIntPoint(&el.GetNodes().IntPoint(j));
            CalcAdjugate(Trans.Jacobian(), adjJ_j);
            adjJ_j.GetRow(di, dxidx);
            u.GetRow(j, uj);
            fluxJacStates(di, ui, uj, flux_jaci, flux_jacj);
            double Sij = sbp.getSkewEntry(di, i, j, adjJ_i, adjJ_j);
            Sij *= alpha;
            for (int n = 0; n < num_states; ++n)
            {
               for (int m = 0; m < num_states; ++m)
               {
                  // res(i,n) += Sij*fluxij(n);
                  elmat(n*num_nodes+i, m*num_nodes+i) += Sij*flux_jaci(n,m);
                  elmat(n*num_nodes+i, m*num_nodes+j) += Sij*flux_jacj(n,m);
                  // res(j,n) -= Sij*fluxij(n);
                  elmat(n*num_nodes+j, m*num_nodes+i) -= Sij*flux_jaci(n,m);
                  elmat(n*num_nodes+j, m*num_nodes+j) -= Sij*flux_jacj(n,m);
               }
            } 
         }
      }
   }
}


template <typename Derived>
void DyadicFluxIntegrator<Derived>::AssembleElementVector(
    const mfem::FiniteElement &el,
    mfem::ElementTransformation &Trans,
    const mfem::Vector &elfun, mfem::Vector &elvect)
{
   using namespace mfem;
   // This should be in a try/catch, but that creates other issues
   const SBPFiniteElement &sbp = dynamic_cast<const SBPFiniteElement&>(el);
   int num_nodes = sbp.GetDof();
   int dim = sbp.GetDim();
#ifdef MFEM_THREAD_SAFE
   Vector ui, uj, fluxij;
   DenseMatrix adjJ_i, adjJ_j;
#endif
	elvect.SetSize(num_states*num_nodes);
   fluxij.SetSize(num_states);
   adjJ_i.SetSize(dim);
   adjJ_j.SetSize(dim);
   DenseMatrix u(elfun.GetData(), num_nodes, num_states);
   DenseMatrix res(elvect.GetData(), num_nodes, num_states);

	elvect = 0.0;
   for (int i = 0; i < num_nodes; ++i)
   {
      Trans.SetIntPoint(&el.GetNodes().IntPoint(i));
      CalcAdjugate(Trans.Jacobian(), adjJ_i);
      u.GetRow(i,ui);
		for (int j = i+1; j < num_nodes; ++j)
		{
         Trans.SetIntPoint(&el.GetNodes().IntPoint(j));
         CalcAdjugate(Trans.Jacobian(), adjJ_j);
         u.GetRow(j, uj);
			for (int di = 0; di < dim; ++di)
			{
            // TODO: we should add state_offset to ui and uj, and eqn_offset to
            // fluxij because the flux function may not know about other states 
            // and equations.
				flux(di, ui, uj, fluxij);
            double Sij = sbp.getSkewEntry(di, i, j, adjJ_i, adjJ_j);
            Sij *= alpha;
            for (int n = 0; n < num_states; ++n)
            {
               res(i,n) += Sij*fluxij(n);
               res(j,n) -= Sij*fluxij(n);
            }
			} // di loop
      } // j node loop
   } // i node loop
}

template <typename Derived>
void LPSIntegrator<Derived>::AssembleElementVector(
    const mfem::FiniteElement &el, mfem::ElementTransformation &Trans,
    const mfem::Vector &elfun, mfem::Vector &elvect)
{
   using namespace mfem;
   const SBPFiniteElement &sbp = dynamic_cast<const SBPFiniteElement&>(el);
   int num_nodes = sbp.GetDof();
   int dim = sbp.GetDim();
#ifdef MFEM_THREAD_SAFE
   Vector ui;
   DenseMatrix adjJt, w, Pw;
#endif
	elvect.SetSize(num_states*num_nodes);
   ui.SetSize(num_states);
   adjJt.SetSize(dim);
   w.SetSize(num_states, num_nodes);
   Pw.SetSize(num_states, num_nodes);
   Vector wi, Pwi;
   DenseMatrix u(elfun.GetData(), num_nodes, num_states);
   DenseMatrix res(elvect.GetData(), num_nodes, num_states);

   // Step 1: convert from working variables (this may be the identity)
   for (int i = 0; i < num_nodes; ++i)
   {
      u.GetRow(i,ui);
      w.GetColumnReference(i, wi);
      convert(ui, wi);
   }
   // Step 2: apply the projection operator to w
   sbp.multProjOperator(w, Pw, false);
   // Step 3: apply scaling matrix at each node and diagonal norm
   for (int i = 0; i < num_nodes; ++i)
   {
      Trans.SetIntPoint(&el.GetNodes().IntPoint(i));
      //CalcAdjugateTranspose(Trans.Jacobian(), adjJt);
      CalcAdjugate(Trans.Jacobian(), adjJt);
      u.GetRow(i,ui);
      Pw.GetColumnReference(i, Pwi);
      w.GetColumnReference(i, wi);
      scale(adjJt, ui, Pwi, wi);
      wi *= lps_coeff;
   }
   sbp.multNormMatrix(w, w);
   // Step 4: apply the transposed projection operator to H*A*P*w
   sbp.multProjOperator(w, Pw, true);
   // This is necessary because data in elvect is expected to be ordered `byNODES`
   res.Transpose(Pw);
   res *= alpha;
}

template <typename Derived>
void InviscidBoundaryIntegrator<Derived>::AssembleFaceVector(
   const mfem::FiniteElement &el_bnd,
   const mfem::FiniteElement &el_unused,
   mfem::FaceElementTransformations &trans,
   const mfem::Vector &elfun,
   mfem::Vector &elvect)
{
   //cout << "bnd_marker = " << bnd_marker << endl;
   //cout.flush();
   using namespace mfem;
   const SBPFiniteElement &sbp = dynamic_cast<const SBPFiniteElement&>(el_bnd);
   const int num_nodes = el_bnd.GetDof();
   const int dim = sbp.GetDim();
#ifdef MFEM_THREAD_SAFE
   Vector u_face, x, nrm, flux_face;
#endif
	elvect.SetSize(num_states*num_nodes);
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
      //cout << "face node " << face_ip.x << ": nrm = " << nrm[0] << ", " << nrm[1] << endl;
      flux(x, nrm, u_face, flux_face);

      // cout << "face node " << face_ip.x << ": flux = ";
      // for (int n = 0; n < num_states; ++n)
      // {
      //    cout << flux_face[n] << ", ";
      // }
      // cout << endl;


      flux_face *= face_ip.weight;


      // multiply by test function
      for (int n = 0; n < num_states; ++n)
      {
         res(j, n) += alpha*flux_face(n);
      }
   }
}

template <typename Derived>
void InviscidBoundaryIntegrator<Derived>::AssembleFaceGrad(
   const mfem::FiniteElement &el_bnd,
   const mfem::FiniteElement &el_unused,
   mfem::FaceElementTransformations &trans,
   const mfem::Vector &elfun,
   mfem::DenseMatrix &elmat)
{
   using namespace mfem;
   const SBPFiniteElement &sbp = dynamic_cast<const SBPFiniteElement&>(el_bnd);
   const int num_nodes = el_bnd.GetDof();
   const int dim = sbp.GetDim();
#ifdef MFEM_THREAD_SAFE
   Vector u_face, x, nrm; // flux_face;
   DenseMatrix flux_jac_face;
#endif
	// elvect.SetSize(num_states*num_nodes);
   u_face.SetSize(num_states);
   x.SetSize(dim);
   nrm.SetSize(dim);
   // flux_face.SetSize(num_states);
   flux_jac_face.SetSize(num_states);
   elmat.SetSize(num_states*num_nodes);
   elmat = 0.0;

   DenseMatrix u(elfun.GetData(), num_nodes, num_states);

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
      // flux(x, nrm, u_face, flux_face);
      fluxJacState(x, nrm, u_face, flux_jac_face);

      // flux_face *= face_ip.weight;
      flux_jac_face *= face_ip.weight;

      // multiply by test function
      for (int n = 0; n < num_states; ++n)
      {
         for (int m = 0; m < num_states; ++m)
         {
            // res(j, n) += alpha*flux_face(n);
            elmat(m*num_nodes+j, n*num_nodes+j) += alpha*flux_jac_face(m,n);
         }
      }
   }
}