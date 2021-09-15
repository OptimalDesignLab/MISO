template <typename Derived>
void SymmetricViscousIntegrator<Derived>::AssembleElementVector(
    const mfem::FiniteElement &el,
    mfem::ElementTransformation &Trans,
    const mfem::Vector &elfun,
    mfem::Vector &elvect)
{
   using namespace mfem;
   const auto &sbp = dynamic_cast<const SBPFiniteElement &>(el);
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
   Dwi.SetSize(num_states, dim);
   CDwi.SetSize(num_states);
   adjJ_i.SetSize(dim);
   adjJ_j.SetSize(dim);
   adjJ_k.SetSize(dim);
   DenseMatrix u(elfun.GetData(), num_nodes, num_states);
   DenseMatrix res(elvect.GetData(), num_nodes, num_states);
   elvect = 0.0;
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
               Dwi(s, d) += Qij * wj(s);
            }
         }  // loop over space dimensions d
      }     // loop over element nodes j
      Dwi *= Hinv;

      // next, scale the derivatives (e.g. using \hat{C} matrices), and then
      // apply derivative of test function
      for (int d = 0; d < dim; ++d)
      {
         scale(d, xi, ui, Dwi, CDwi);
         for (int k = 0; k < num_nodes; ++k)
         {
            // Get mapping Jacobian adjugate
            Trans.SetIntPoint(&el.GetNodes().IntPoint(k));
            CalcAdjugate(Trans.Jacobian(), adjJ_k);
            double Qik = sbp.getQEntry(d, i, k, adjJ_i, adjJ_k);
            // apply transposed derivative in `d` direction
            // this evaluates Q_d'*(C_d,1 D_1 + C_d,2 D_2 + ...) w
            for (int s = 0; s < num_states; ++s)
            {
               res(k, s) += alpha * Qik * CDwi(s);
            }
         }  // loop over element nodes k
      }     // loop over space dimensions d
   }        // loop over element nodes i
}

template <typename Derived>
void SymmetricViscousIntegrator<Derived>::AssembleElementGrad(
    const mfem::FiniteElement &el,
    mfem::ElementTransformation &Trans,
    const mfem::Vector &elfun,
    mfem::DenseMatrix &elmat)
{
   using namespace mfem;
   using namespace std;
   const auto &sbp = dynamic_cast<const SBPFiniteElement &>(el);
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
   Dwi.SetSize(num_states, dim);
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
   elmat.SetSize(num_states * num_nodes);
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
               Dwi(s, d) += Qij * wj(s);
            }
         }  // loop over space dimensions d
      }     // loop over element nodes j
      Dwi *= Hinv;

      for (int d = 0; d < dim; ++d)
      {
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
                  elmat(sk * num_nodes + k, si * num_nodes + i) +=
                      Qik * jac_term1(sk, si);
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
                  jac_term2 *= (Qik * Hinv * Qij);
                  // Add to the Jacobian
                  for (int sk = 0; sk < num_states; ++sk)
                  {
                     for (int sj = 0; sj < num_states; ++sj)
                     {
                        elmat(sk * num_nodes + k, sj * num_nodes + j) +=
                            jac_term2(sk, sj);
                     }
                  }

               }  // loop over space dimension d2
            }     // loop over the element nodes j
         }        // loop over element nodes k
      }           // loop over space dimensions d
   }              // loop over element nodes i
   elmat *= alpha;
}

template <typename Derived>
double ViscousBoundaryIntegrator<Derived>::GetFaceEnergy(
    const mfem::FiniteElement &el_bnd,
    const mfem::FiniteElement &el_unused,
    mfem::FaceElementTransformations &trans,
    const mfem::Vector &elfun)
{
   using namespace mfem;
   const auto &sbp = dynamic_cast<const SBPFiniteElement &>(el_bnd);
   const int num_nodes = sbp.GetDof();
   const int dim = sbp.GetDim();
#ifdef MFEM_THREAD_SAFE
   Vector u_face, uj, wj, x, nrm;
   DenseMatrix adjJ_i, adjJ_j, Dwi;
#endif
   u_face.SetSize(num_states);
   uj.SetSize(num_states);
   wj.SetSize(num_states);
   x.SetSize(dim);
   nrm.SetSize(dim);
   adjJ_i.SetSize(dim);
   adjJ_j.SetSize(dim);
   Dwi.SetSize(num_states, dim);
   DenseMatrix u(elfun.GetData(), num_nodes, num_states);

   const FiniteElement *sbp_face = nullptr;
   switch (dim)
   {
   case 1:
      sbp_face = fec->FiniteElementForGeometry(Geometry::POINT);
      break;
   case 2:
      sbp_face = fec->FiniteElementForGeometry(Geometry::SEGMENT);
      break;
   default:
      throw mach::MachException(
          "ViscousBoundaryIntegrator::AssembleFaceVector())\n"
          "\tcannot handle given dimension");
   }
   IntegrationPoint el_ip;
   double fun = 0.0;  // initialize the functional value
   for (int k = 0; k < sbp_face->GetDof(); ++k)
   {
      // convert face index k to element index i, and get node location
      const IntegrationPoint &face_ip = sbp_face->GetNodes().IntPoint(k);
      trans.Loc1.Transform(face_ip, el_ip);
      trans.Elem1->Transform(el_ip, x);
      int i = sbp.getIntegrationPointIndex(el_ip);

      // get the state at element node i, as well as the mapping adjugate
      u.GetRow(i, u_face);
      trans.Elem1->SetIntPoint(&el_ip);
      double jac_i = trans.Elem1->Weight();
      double Hinv = 1.0 / (sbp.getDiagNormEntry(i) * jac_i);
      CalcAdjugate(trans.Elem1->Jacobian(), adjJ_i);

      // compute the (physcial space) derivatives at node i
      Dwi = 0.0;
      for (int j = 0; j < num_nodes; ++j)
      {
         // Get mapping Jacobian adjugate and transform state to entropy vars
         trans.Elem1->SetIntPoint(&el_bnd.GetNodes().IntPoint(j));
         CalcAdjugate(trans.Elem1->Jacobian(), adjJ_j);
         u.GetRow(j, uj);
         convert(uj, wj);
         for (int d = 0; d < dim; ++d)
         {
            double Qij = sbp.getQEntry(d, i, j, adjJ_i, adjJ_j);
            for (int s = 0; s < num_states; ++s)
            {
               Dwi(s, d) += Qij * wj(s);
            }
         }  // d loop
      }     // j loop
      Dwi *= Hinv;

      // get the normal vector to the face, and then compute the contribution
      trans.Face->SetIntPoint(&face_ip);
      CalcOrtho(trans.Face->Jacobian(), nrm);
      fun += bndryFun(x, nrm, jac_i, u_face, Dwi) * face_ip.weight * alpha;
   }  // k/i loop
   return fun;
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
   const auto &sbp = dynamic_cast<const SBPFiniteElement &>(el_bnd);
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

   elvect.SetSize(num_states * num_nodes);
   elvect = 0.0;

   DenseMatrix u(elfun.GetData(), num_nodes, num_states);
   DenseMatrix res(elvect.GetData(), num_nodes, num_states);

   const FiniteElement *sbp_face = nullptr;
   switch (dim)
   {
   case 1:
      sbp_face = fec->FiniteElementForGeometry(Geometry::POINT);
      break;
   case 2:
      sbp_face = fec->FiniteElementForGeometry(Geometry::SEGMENT);
      break;
   default:
      throw mach::MachException(
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
      trans.Elem1->SetIntPoint(&el_ip);
      double jac_i = trans.Elem1->Weight();
      double Hinv = 1.0 / (sbp.getDiagNormEntry(i) * jac_i);
      CalcAdjugate(trans.Elem1->Jacobian(), adjJ_i);

      // compute the (physcial space) derivatives at node i
      Dwi = 0.0;
      for (int j = 0; j < num_nodes; ++j)
      {
         // Get mapping Jacobian adjugate and transform state to entropy vars
         trans.Elem1->SetIntPoint(&el_bnd.GetNodes().IntPoint(j));
         CalcAdjugate(trans.Elem1->Jacobian(), adjJ_j);
         u.GetRow(j, uj);
         convert(uj, wj);
         for (int d = 0; d < dim; ++d)
         {
            double Qij = sbp.getQEntry(d, i, j, adjJ_i, adjJ_j);
            for (int s = 0; s < num_states; ++s)
            {
               Dwi(s, d) += Qij * wj(s);
            }
         }  // d loop
      }     // j loop
      Dwi *= Hinv;

      // get the normal vector to the face, and then compute the flux
      trans.Face->SetIntPoint(&face_ip);
      CalcOrtho(trans.Face->Jacobian(), nrm);
      flux(x, nrm, jac_i, u_face, Dwi, flux_face);
      flux_face *= face_ip.weight;

      // multiply by test function
      for (int s = 0; s < num_states; ++s)
      {
         res(i, s) += alpha * flux_face(s);
      }

      // get the flux terms that are scaled by the test-function derivative
      // (reuse Dwi here)
      fluxDv(x, nrm, u_face, Dwi);
      Dwi *= face_ip.weight;

      // multiply flux_Dvi by the test-function derivative
      Dwi *= Hinv;
      for (int j = 0; j < num_nodes; ++j)
      {
         // Get mapping Jacobian adjugate
         trans.Elem1->SetIntPoint(&el_bnd.GetNodes().IntPoint(j));
         CalcAdjugate(trans.Elem1->Jacobian(), adjJ_j);
         for (int d = 0; d < dim; ++d)
         {
            double Qij = sbp.getQEntry(d, i, j, adjJ_i, adjJ_j);
            for (int s = 0; s < num_states; ++s)
            {
               // Dwi(s,d) += Qij * wj(s);
               res(j, s) += Qij * Dwi(s, d);
            }
         }  // d loop
      }     // j loop
   }        // k/i loop
}

template <typename Derived>
void ViscousBoundaryIntegrator<Derived>::AssembleFaceGrad(
    const mfem::FiniteElement &el_bnd,
    const mfem::FiniteElement &el_unused,
    mfem::FaceElementTransformations &trans,
    const mfem::Vector &elfun,
    mfem::DenseMatrix &elmat)
{
   using namespace mfem;
   const auto &sbp = dynamic_cast<const SBPFiniteElement &>(el_bnd);
   const int num_nodes = el_bnd.GetDof();
   const int dim = sbp.GetDim();
#ifdef MFEM_THREAD_SAFE
   Vector u_face, uj, wj, x, nrm, flux_face;
   DenseMatrix adjJ_i, adjJ_j, Dwi, jac_term, dwduj;
   vector<DenseMatrix> fluxDw_jac(dim);
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
   jac_term.SetSize(num_states);
   dwduj.SetSize(num_states);
   fluxDw_jac.resize(dim);
   for (int d = 0; d < dim; ++d)
   {
      fluxDw_jac[d].SetSize(num_states);
   }
   DenseMatrix u(elfun.GetData(), num_nodes, num_states);
   elmat.SetSize(num_states * num_nodes);
   elmat = 0.0;

   const FiniteElement *sbp_face = nullptr;
   switch (dim)
   {
   case 1:
      sbp_face = fec->FiniteElementForGeometry(Geometry::POINT);
      break;
   case 2:
      sbp_face = fec->FiniteElementForGeometry(Geometry::SEGMENT);
      break;
   default:
      throw mach::MachException(
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
      trans.Elem1->SetIntPoint(&el_ip);
      double jac_i = trans.Elem1->Weight();
      double Hinv = 1.0 / (sbp.getDiagNormEntry(i) * jac_i);
      CalcAdjugate(trans.Elem1->Jacobian(), adjJ_i);

      // compute the (physcial space) derivatives at node i
      Dwi = 0.0;
      for (int j = 0; j < num_nodes; ++j)
      {
         // Get mapping Jacobian adjugate and transform state to entropy vars
         trans.Elem1->SetIntPoint(&el_bnd.GetNodes().IntPoint(j));
         CalcAdjugate(trans.Elem1->Jacobian(), adjJ_j);
         u.GetRow(j, uj);
         convert(uj, wj);
         for (int d = 0; d < dim; ++d)
         {
            double Qij = sbp.getQEntry(d, i, j, adjJ_i, adjJ_j);
            for (int s = 0; s < num_states; ++s)
            {
               Dwi(s, d) += Qij * wj(s);
            }
         }  // d loop
      }     // j loop
      Dwi *= Hinv;

      // get the normal vector to the face, and then compute the flux
      trans.Face->SetIntPoint(&face_ip);
      CalcOrtho(trans.Face->Jacobian(), nrm);
      // flux(x, nrm, jac_i, u_face, Dwi, flux_face, flux_Dvi);
      fluxJacState(x, nrm, jac_i, u_face, Dwi, jac_term);
      jac_term *= face_ip.weight;

      // Add contribution due to dependence on state
      for (int s1 = 0; s1 < num_states; ++s1)
      {
         for (int s2 = 0; s2 < num_states; ++s2)
         {
            // res(i, s) += alpha*flux_face(s);
            elmat(s1 * num_nodes + i, s2 * num_nodes + i) += jac_term(s1, s2);
         }
      }

      // flux(x, nrm, jac_i, u_face, Dwi, flux_face);
      fluxJacDw(x, nrm, jac_i, u_face, Dwi, fluxDw_jac);
      // flux_jac *= face_ip.weight;

      // Add contribution due to dependence on state derivative
      for (int j = 0; j < num_nodes; ++j)
      {
         // Get mapping Jacobian adjugate and transform state to entropy vars
         trans.Elem1->SetIntPoint(&el_bnd.GetNodes().IntPoint(j));
         CalcAdjugate(trans.Elem1->Jacobian(), adjJ_j);
         u.GetRow(j, uj);
         // convert(uj, wj);
         convertJacState(uj, dwduj);
         for (int d = 0; d < dim; ++d)
         {
            double Qij = sbp.getQEntry(d, i, j, adjJ_i, adjJ_j);
            Mult(fluxDw_jac[d], dwduj, jac_term);
            jac_term *= (Hinv * Qij * face_ip.weight);
            for (int si = 0; si < num_states; ++si)
            {
               for (int sj = 0; sj < num_states; ++sj)
               {
                  // Dwi(s,d) += Qij * wj(s);
                  elmat(si * num_nodes + i, sj * num_nodes + j) +=
                      jac_term(si, sj);
               }
            }
         }  // d loop
      }     // j loop

      // fluxDv(x, nrm, u_face, Dwi);
      fluxDvJacState(x, nrm, u_face, fluxDw_jac);
      // Dwi *= face_ip.weight;

      // Add contribution due to flux that is scaled by test-function derivative
      for (int j = 0; j < num_nodes; ++j)
      {
         // Get mapping Jacobian adjugate
         trans.Elem1->SetIntPoint(&el_bnd.GetNodes().IntPoint(j));
         CalcAdjugate(trans.Elem1->Jacobian(), adjJ_j);
         for (int d = 0; d < dim; ++d)
         {
            double Dij =
                face_ip.weight * Hinv * sbp.getQEntry(d, i, j, adjJ_i, adjJ_j);
            for (int si = 0; si < num_states; ++si)
            {
               for (int sj = 0; sj < num_states; ++sj)
               {
                  // res(j, s) += Qij * Dwi(s, d);
                  elmat(sj * num_nodes + j, si * num_nodes + i) +=
                      Dij * fluxDw_jac[d](sj, si);
               }
            }
         }  // d loop
      }     // j loop
   }        // k/i loop
}
