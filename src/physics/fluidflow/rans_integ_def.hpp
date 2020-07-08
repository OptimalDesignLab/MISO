//==============================================================================
// ESViscousSAIntegrator methods


//==============================================================================
// SASourceIntegrator methods

template <typename Derived>
void SASourceIntegrator<Derived>::AssembleElementVector(
    const mfem::FiniteElement &el, mfem::ElementTransformation &Trans,
    const mfem::Vector &elfun, mfem::Vector &elvect)
{
   using namespace mfem;
   const SBPFiniteElement &sbp = dynamic_cast<const SBPFiniteElement &>(el);
   int num_nodes = sbp.GetDof();
   int dim = sbp.GetDim();
#ifdef MFEM_THREAD_SAFE
   Vector ui, xi, wj, uj, CDwi;
   DenseMatrix adjJ_i, adjJ_j, adjJ_k, Dwi;
#endif
   elvect.SetSize(num_states * num_nodes);
   ui.SetSize(num_states);
   xi.SetSize(dim);
   wj.SetSize(num_states);
   uj.SetSize(num_states);
   Dwi.SetSize(num_states,dim);
   CDwi.SetSize(num_states);
   adjJ_i.SetSize(dim);
   adjJ_j.SetSize(dim);
   adjJ_k.SetSize(dim);
   DenseMatrix u(elfun.GetData(), num_nodes, num_states);
   DenseMatrix PDS(num_nodes, num_states);
   DenseMatrix res(elvect.GetData(), num_nodes, num_states);
   elvect = 0.0; PDS = 0.0;
   for (int i = 0; i < num_nodes; ++i)
   {
      // get the Jacobian (Trans.Weight) and cubature weight (node.weight)
      const IntegrationPoint &node = el.GetNodes().IntPoint(i);
      Trans.SetIntPoint(&node);
      CalcAdjugate(Trans.Jacobian(), adjJ_i);
      u.GetRow(i, ui);
      Trans.Transform(node, xi);

      // compute vorticity at node and take magnitude
      double S;

      // compute gradient of turbulent viscosity at node
      Vector dir; 

      // get distance function value at node
      double d;

      // accumulate source terms
      double src = calcSASource(ui.GetData(), dir.GetData(), sacs.GetData());
      // use negative model if turbulent viscosity is negative
      if (ui(dim+2) < 0)
      {
         src += calcSANegativeProduction(ui.GetData(), S, sacs.GetData());
         src += calcSANegativeDestruction(ui.GetData(), d, sacs.GetData());
      }
      else
      {
         src += calcSAProduction(ui.GetData(), mu, d, S, sacs.GetData());
         src += calcSADestruction(ui.GetData(), mu, d, S, sacs.GetData());
      }

      PDS(i, dim+2) += alpha *  src;
   } // loop over element nodes i

   // NOTE: how to multiply cubature and transformation weights?

   // multiply by projection
   sbp.multProjOperator(PDS, res);
}

template <typename Derived>
void SAViscousIntegrator<Derived>::AssembleElementGrad(
    const mfem::FiniteElement &el, mfem::ElementTransformation &Trans,
    const mfem::Vector &elfun, mfem::DenseMatrix &elmat)
{
   using namespace mfem;
   using namespace std;
   const SBPFiniteElement &sbp = dynamic_cast<const SBPFiniteElement &>(el);
   int num_nodes = sbp.GetDof();
   int dim = sbp.GetDim();
#ifdef MFEM_THREAD_SAFE
   Vector ui, xi, wj, uj;
   DenseMatrix adjJ_i, adjJ_j, adjJ_k, Dwi, jac_term1, jac_term2, dwduj;
   vector<DenseMatrix> CDw_jac(dim);
#endif
   ui.SetSize(num_states);
   xi.SetSize(dim);
   wj.SetSize(num_states);
   uj.SetSize(num_states);
   Dwi.SetSize(num_states,dim);
   adjJ_i.SetSize(dim);
   adjJ_j.SetSize(dim);
   adjJ_k.SetSize(dim);
   jac_term1.SetSize(num_states);
   jac_term2.SetSize(num_states);
   dwduj.SetSize(num_states);
   CDw_jac.resize(dim);
   for (int d = 0; d < dim; ++d)
   {
      CDw_jac[d].SetSize(num_states);
   }
   DenseMatrix u(elfun.GetData(), num_nodes, num_states);
   elmat.SetSize(num_states*num_nodes);
   elmat = 0.0;

   for (int i = 0; i < num_nodes; ++i)
   {
      // get the Jacobian (Trans.Weight) and cubature weight (node.weight)
      const IntegrationPoint &node = el.GetNodes().IntPoint(i);
      Trans.SetIntPoint(&node);
      double Hinv = 1.0 / (sbp.getDiagNormEntry(i) * Trans.Weight());
      CalcAdjugate(Trans.Jacobian(), adjJ_i);
      u.GetRow(i, ui);
      Trans.Transform(node, xi);

      // compute the (physcial space) derivatives at node i
      Dwi = 0.0;
      for (int j = 0; j < num_nodes; ++j)
      {
         // Get mapping Jacobian adjugate and transform state to entropy vars
         Trans.SetIntPoint(&el.GetNodes().IntPoint(j));
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
         } // loop over space dimensions d
      } // loop over element nodes j
      Dwi *= Hinv;

      for (int d = 0; d < dim; ++d) {
         // scale(d, xi, ui, Dwi, CDwi);
         scaleJacState(d, xi, ui, Dwi, jac_term1);
         scaleJacDw(d, xi, ui, Dwi, CDw_jac);
         for (int k = 0; k < num_nodes; ++k) 
         {
            // Node k mapping Jacobian adjugate and 
            Trans.SetIntPoint(&el.GetNodes().IntPoint(k));
            CalcAdjugate(Trans.Jacobian(), adjJ_k);
            double Qik = sbp.getQEntry(d, i, k, adjJ_i, adjJ_k);

            // Contribution to Jacobian due to scaling operation
            for (int sk = 0; sk < num_states; ++sk)
            {
               for (int si = 0; si < num_states; ++si)
               {
                  // res(k, s) += alpha * Qik * CDwi(s);
                  elmat(sk*num_nodes+k, si*num_nodes+i) += Qik*jac_term1(sk,si);
               }
            }

            // Contribution to Jacobian assuming constant scaling operation
            for (int j = 0; j < num_nodes; ++j)
            {
               // Node j mapping Jacobian adjugate and get Jacobian dw/du_j
               Trans.SetIntPoint(&el.GetNodes().IntPoint(j));
               CalcAdjugate(Trans.Jacobian(), adjJ_j);
               u.GetRow(j, uj);
               convertJacState(uj, dwduj);
               for (int d2 = 0; d2 < dim; ++d2)
               {
                  // Following computes
                  // (Q_d1)_{ik}*H^{-1}_{ii}(C_{d,d2}(u_i)*(Q_d2)_{ij}*dw/du_j
                  // where C_{d,d2}(u_i) is a (state x state) matrix
                  // (Q_d2)_ij, (Q_d1)_ij, and H^{-1} are scalars, and 
                  // (dw/du_j) is a (state x state) matrix
                  double Qij = sbp.getQEntry(d2, i, j, adjJ_i, adjJ_j);
                  Mult(CDw_jac[d2], dwduj, jac_term2);
                  jac_term2 *= (Qik*Hinv*Qij);
                  // Add to the Jacobian
                  for (int sk = 0; sk < num_states; ++sk)
                  {
                     for (int sj = 0; sj < num_states; ++sj)
                     {
                        elmat(sk * num_nodes + k, sj * num_nodes + j) +=
                            jac_term2(sk, sj);
                     }
                  }

               } // loop over space dimension d2
            } // loop over the element nodes j
         } // loop over element nodes k
      } // loop over space dimensions d
   } // loop over element nodes i
   elmat *= alpha;
}