
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
   Vector ui, wj, uj, Qwi, CQwd1d2;
   DenseMatrix w, adjJ_i, adjJ_j, adjJ_k;
#endif
   elvect.SetSize(num_states * num_nodes);
   ui.SetSize(num_states);
   wj.SetSize(num_states);
   uj.SetSize(num_states);
   Qwi.SetSize(num_states);
   CQwd1d2.SetSize(num_states);
   w.SetSize(num_states, num_nodes);
   adjJ_i.SetSize(dim);
   adjJ_j.SetSize(dim);
   adjJ_k.SetSize(dim);
   DenseMatrix u(elfun.GetData(), num_nodes, num_states);
   DenseMatrix res(elvect.GetData(), num_nodes, num_states);
   elvect = 0.0;
   for (int i = 0; i < num_nodes; ++i)
   {
      Qwi = 0;
      // get the Jacobian (Trans.Weight) and cubature weight (node.weight)
      const IntegrationRule &ir = el.GetNodes();
      const IntegrationPoint &node = ir.IntPoint(i);
      Trans.SetIntPoint(&node);
      double norm = node.weight * Trans.Weight();
      double H = 1 / norm;
      CalcAdjugate(Trans.Jacobian(), adjJ_i);
      for (int d2 = 0; d2 < dim; ++d2)
      {
         for (int j = 0; j < num_nodes; ++j)
         {
            Trans.SetIntPoint(&el.GetNodes().IntPoint(j));
            CalcAdjugate(Trans.Jacobian(), adjJ_j);
            double Qij = sbp.getQEntry(d2, i, j, adjJ_i, adjJ_j);
            u.GetRow(j, uj);
            w.GetColumnReference(j, wj);
            // Step 1: convert to entropy variables
            convert(uj, wj);
            // Step 2: find the derivative in `d2` direction
            for (int s = 0; s < num_states; ++s)
            {
               Qwi(s) += Qij * wj(s);
            }
         } // j node loop
         u.GetRow(i, ui);
         for (int d1 = 0; d1 < dim; ++d1)
         {
            // Step 3: apply the viscous coefficients' scaling
            scale(d1, d2, ui, Qwi, CQwd1d2);
            for (int k = 0; k < num_nodes; ++k)
            {
               Trans.SetIntPoint(&el.GetNodes().IntPoint(k));
               CalcAdjugate(Trans.Jacobian(), adjJ_k);
               double Qik = sbp.getQEntry(d1, i, k, adjJ_i, adjJ_k);
               // Step 4: apply derivative in `d1` direction
               // this evaluates Qd1'*C*(H^-1)*Qd2
               for (int s = 0; s < num_states; ++s)
               {
                  res(k, s) += alpha * Qik * H * CQwd1d2(s);
               }
            } // k loop
         }    // d1 loop
      }       //d2 loop
   }          // i node loop
}
