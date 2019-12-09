
template <typename Derived>
void SymmetricViscousIntegrator<Derived>::AssembleElementVector(
    const mfem::FiniteElement &el, mfem::ElementTransformation &Trans,
    const mfem::Vector &elfun, mfem::Vector &elvect)
{
   using namespace mfem;
   const SBPFiniteElement &sbp = dynamic_cast<const SBPFiniteElement &>(el);
   int num_nodes = sbp.GetDof();
   int dim = sbp.GetDim();
#ifdef MFEM_THREAD_SAFE
   Vector ui, wj, uj, Dwi, CDwi;
   DenseMatrix adjJ_i, adjJ_j, adjJ_k;
#endif
   elvect.SetSize(num_states * num_nodes);
   ui.SetSize(num_states);
   wj.SetSize(num_states);
   uj.SetSize(num_states);
   Dwi.SetSize(num_states);
   CDw.SetSize(num_states);
   adjJ_i.SetSize(dim);
   adjJ_j.SetSize(dim);
   adjJ_k.SetSize(dim);
   DenseMatrix u(elfun.GetData(), num_nodes, num_states);
   DenseMatrix res(elvect.GetData(), num_nodes, num_states);
   elvect = 0.0;
   for (int i = 0; i < num_nodes; ++i)
   {
      // get the Jacobian (Trans.Weight) and cubature weight (node.weight)
      const IntegrationRule &ir = el.GetNodes();
      const IntegrationPoint &node = ir.IntPoint(i);
      Trans.SetIntPoint(&node);
      double norm = sbp.getDiagNormEntry(i) * Trans.Weight();
      double Hinv = 1 / norm;
      CalcAdjugate(Trans.Jacobian(), adjJ_i);
      u.GetRow(i, ui);
      for (int di2 = 0; di2 < dim; ++di2)
      {
         Dwi = 0.0;
         for (int j = 0; j < num_nodes; ++j)
         {
            Trans.SetIntPoint(&el.GetNodes().IntPoint(j));
            CalcAdjugate(Trans.Jacobian(), adjJ_j);
            double Qij = sbp.getQEntry(di2, i, j, adjJ_i, adjJ_j);
            u.GetRow(j, uj);
            // Step 1: convert to entropy variables
            convert(uj, wj);
            // Step 2: find the derivative in `di2` direction
            for (int s = 0; s < num_states; ++s)
            {
               Dwi(s) += Qij * wj(s);
            }
         } // j node loop
         Dwi *= Hinv;
         for (int di1 = 0; di1 < dim; ++di1)
         {
            // Step 3: apply the viscous coefficients' scaling
            scale(di1, di2, ui, Dwi, CDwi);
            for (int k = 0; k < num_nodes; ++k)
            {
               Trans.SetIntPoint(&el.GetNodes().IntPoint(k));
               CalcAdjugate(Trans.Jacobian(), adjJ_k);
               double Qik = sbp.getQEntry(di1, i, k, adjJ_i, adjJ_k);
               // Step 4: apply derivative in `di1` direction
               // this evaluates Qd1'*C*(H^-1)*Qd2
               for (int s = 0; s < num_states; ++s)
               {
                  res(k, s) += alpha * Qik * CDwi(s);
               }
            } // k loop
         }    // di1 loop
      }       // di2 loop
   }          // i node loop
}

template <typename Derived>
void ViscousBoundaryIntegrator<Derived>::AssembleFaceVector(
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
   Vector u_face, uj, wj, x, nrm, flux_face;
   DenseMatrix adjJ_i, adjJ_j, Dwi;
#endif
   u_face.SetSize(num_states);
   uj.SetSize(num_states);
   wj.SetSize(num_states);
   x.SetSize(dim);
   nrm.SetSize(dim);
   flux_face.SetSize(num_states);
   adjJ_i.SetSize(dim);
   adjJ_j.SetSize(dim);
   Dwi.SetSize(num_states, dim);

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
         "ViscousBoundaryIntegrator::AssembleFaceVector())\n"
         "\tcannot handle given dimension");
   }
   IntegrationPoint el_ip;
   for (int k = 0; k < sbp_face->GetDof(); ++k)
   {
      // convert face index k to element index i, and get node location
      const IntegrationPoint &face_ip = sbp_face->GetNodes().IntPoint(k);
      trans.Loc1.Transform(face_ip, el_ip);
      trans.Elem1->Transform(el_ip, x);
      int i = sbp.getIntegrationPointIndex(el_ip);

      // get the state at element node i, as well as the mapping adjugate
      u.GetRow(i, u_face);
      Trans.SetIntPoint(&el_ip);
      double Hinv = 1.0 /(sbp.getDiagNormEntry(i) * Trans.Weight());
      CalcAdjugate(Trans.Jacobian(), adjJ_i);

      // compute the (physcial space) derivatives at node i
      Dwi = 0.0;
      for (int j = 0; j < num_nodes; ++j)
      {
         // Get mapping Jacobian adjugate and transform state to entropy vars
         Trans.SetIntPoint(&el_bnd.GetNodes().IntPoint(j));
         CalcAdjugate(Trans.Jacobian(), adjJ_j);
         u.GetRow(j, uj);
         convert(uj, wj);
         for (int d = 0; d < dim; ++d)
         {
            double Qij = sbp.getQEntry(d, i, j, adjJ_i, adjJ_j);
            for (int s = 0; s < num_states; ++s)
            {
               Dwi(s,d) += Qij * wj(s);
            }
         } // d loop
      } // j loop
      Dwi *= Hinv;

      // get the normal vector to the face, and then compute the flux
      trans.Face->SetIntPoint(&face_ip);
      CalcOrtho(trans.Face->Jacobian(), nrm);
      flux(x, nrm, u_face, Dwi, flux_face);
      flux_face *= face_ip.weight;

      // multiply by test function
      for (int s = 0; s < num_states; ++s)
      {
         res(i, s) += alpha*flux_face(s);
      }
   } // k/i loop
}