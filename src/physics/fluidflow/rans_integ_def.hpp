//==============================================================================
// ESViscousSAIntegrator methods


//==============================================================================
// SASourceIntegrator methods
#if 0
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
   Vector ui, xi, uj, grad_i, curl_i;
   DenseMatrix grad, curl;
#endif
   elvect.SetSize(num_states * num_nodes);
   ui.SetSize(num_states);
   xi.SetSize(dim);
   grad.SetSize(num_nodes, dim);
   curl.SetSize(num_nodes, 3);
   grad_i.SetSize(dim);
   curl_i.SetSize(3);
   DenseMatrix u(elfun.GetData(), num_nodes, num_states); // send u into function to compute curl, gradient of nu, send transformation as well
   DenseMatrix res(elvect.GetData(), num_nodes, num_states);
   // precompute certain values
   calcVorticity<dim>(u, sbp, Trans, curl);
   calcGrad(u, sbp, Trans, grad);
   elvect = 0.0; PDS = 0.0;
   for (int i = 0; i < num_nodes; ++i)
   {
      // get the Jacobian (Trans.Weight) and cubature weight (node.weight)
      const IntegrationPoint &node = el.GetNodes().IntPoint(i);
      Trans.SetIntPoint(&node);
      u.GetRow(i, ui);
      Trans.Transform(node, xi);

      // compute vorticity at node and take magnitude
      curl.GetRow(i, curl_i);
      double S = curl_i.Norml2();

      // compute gradient of turbulent viscosity at node
      grad.GetRow(i, grad_i); 

      // get distance function value at node
      double d;

      // accumulate source terms
      double src = calcSASource(ui.GetData(), grad_i.GetData(), sacs.GetData());
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

      res(i, dim+2) += alpha * Trans.Weight() * node.weight * src;
   } // loop over element nodes i
}

template <typename Derived>
void SASourceIntegrator<Derived>::AssembleElementGrad(
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

      for (int d = 0; d < dim; ++d) 
      {
         
      }
   elmat *= alpha;
}

template <typename Derived>
void calcVorticitySBP(const DenseMatrix &q, const SBPFiniteElement sbp, 
                      const ElementTransformation &Trans, DenseMatrix curl)
{
   DenseMatrix dq(q.Height(), q.Width()); //contains state derivatives in reference space
   DenseMatrix dxi(dim*q.Height(), dim); //contains velocity derivatives in reference space
   dq = 0.0
   for(int di = 0; di < dim; di++)
   {
      sbp.multWeakOperator(di, q, dq);

      dxi.CopyMN(dq, q.Height(), dim, 1, dim+1, di*q.Height, 0);
   }

   DenseMatrix dx(dim, dim); //contains velocity derivatives in absolute space
   DenseMatrix dxin(dim, dim); //contains velocity derivatives in reference space for a node
   Vector dxrow(dim); Vector curln(dim);
   for (int i = 0; i < num_nodes; ++i)
   {
      // get the Jacobian
      const IntegrationPoint &node = el.GetNodes().IntPoint(i);
      Trans.SetIntPoint(&node);

      // store nodal derivatives in a single matrix
      for(int di = 0; di < dim; di++)
      {
         dxi.GetRow(di*q.Height() + i, dxrow);
         dxin.SetRow(di, dxrow);
      }

      // compute absolute derivatives
      MultAtB(Trans.InverseJacobian(), dxin, dx);
   
      // compute curl at node and append
      if(dim == 2)
      {
         curl(i, 0) = 0;
         curl(i, 1) = 0;
         curl(i, 2) = dx(1,0) - dx(0,1);
      }
      if(dim == 3)
      {
         curl(i, 0) = dx(2,1) - dx(1,2);
         curl(i, 1) = dx(0,2) - dx(2,0);
         curl(i, 2) = dx(1,0) - dx(0,1);
      }
   }
}

template <typename Derived>
void calcGradSBP(const DenseMatrix &q, const SBPFiniteElement sbp, 
                      const ElementTransformation &Trans, DenseMatrix grad)
{
   DenseMatrix dq(q.Height(), q.Width()); //contains state derivatives in reference space
   DenseMatrix dnu(q.Height(), dim); //contains turb variable derivatives in reference space
   dq = 0.0
   for(int di = 0; di < dim; di++)
   {
      sbp.multWeakOperator(int di, const DenseMatrix &u, DenseMatrix &dq);

      dnu.SetCol(di, dq.GetColumn(dim+2));
   }

   Vector dnurow(dim); Vector gradn(dim);
   for (int i = 0; i < num_nodes; ++i)
   {
      // get the Jacobian
      const IntegrationPoint &node = el.GetNodes().IntPoint(i);
      Trans.SetIntPoint(&node);

      // store nodal grad in a vector
      dnu.GetRow(i, dnurow);

      // compute absolute derivatives
      Trans.InverseJacobian().MultTranspose(dnurow, gradn);
   
      // append result
      grad.SetRow(i, gradn);
   }
}

//====================================================================================
//SA Inviscid Integrator Methods

template <int dim, bool entvar>
void SAInviscidIntegrator<dim, entvar>::calcFlux(int di, const mfem::Vector &qL,
                                                const mfem::Vector &qR,
                                                mfem::Vector &flux)
{
   //call base class, operate on original state variables
   IsmailRoeIntegrator<dim, entvar>::calcFlux(di, qL, qR, flux);
   //add flux term for SA
   flux(dim+2) = 0.5*(qL(di+1)/qL(0) + qR(di+1)/qR(0))*0.5*(qL(dim+2) + qR(dim+2));
}

template <int dim, bool entvar>
void SAInviscidIntegrator<dim, entvar>::calcFluxJacStates(
    int di, const mfem::Vector &qL, const mfem::Vector &qR,
    mfem::DenseMatrix &jacL, mfem::DenseMatrix &jacR)
{
   // store the full jacobian in jac
   mfem::DenseMatrix jac(dim + 2, 2 * (dim + 2));
   // vector of active input variables
   std::vector<adouble> qL_a(qL.Size());
   std::vector<adouble> qR_a(qR.Size());
   // initialize adouble inputs
   adept::set_values(qL_a.data(), qL.Size(), qL.GetData());
   adept::set_values(qR_a.data(), qR.Size(), qR.GetData());
   // start recording
   this->stack.new_recording();
   // create vector of active output variables
   std::vector<adouble> flux_a(qL.Size());
   // run algorithm
   if (entvar)
   {
      mach::calcIsmailRoeFluxUsingEntVars<adouble, dim>(di, qL_a.data(),
                                                        qR_a.data(),
                                                        flux_a.data());
   }
   else
   {
      mach::calcIsmailRoeFlux<adouble, dim>(di, qL_a.data(),
                                            qR_a.data(), flux_a.data());
   }
   // identify independent and dependent variables
   this->stack.independent(qL_a.data(), qL.Size());
   this->stack.independent(qR_a.data(), qR.Size());
   this->stack.dependent(flux_a.data(), qL.Size());
   // compute and store jacobian in jac
   this->stack.jacobian_reverse(jac.GetData());
   // retrieve the jacobian w.r.t left state
   jacL.CopyCols(jac, 0, dim + 1);
   // retrieve the jacobian w.r.t right state
   jacR.CopyCols(jac, dim + 2, 2 * (dim + 2) - 1);
}


#endif