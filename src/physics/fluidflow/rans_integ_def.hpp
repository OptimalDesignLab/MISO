//==============================================================================
// SASourceIntegrator methods
template <int dim>
void SASourceIntegrator<dim>::AssembleElementVector(
    const mfem::FiniteElement &fe, mfem::ElementTransformation &Trans,
    const mfem::Vector &elfun, mfem::Vector &elvect)
{
   using namespace mfem;
   const SBPFiniteElement &sbp = dynamic_cast<const SBPFiniteElement &>(fe);
   int num_nodes = sbp.GetDof();
#ifdef MFEM_THREAD_SAFE
   Vector ui, uci, xi, uj, grad_nu_i, curl_i;
   DenseMatrix grad, curl;
#endif
   elvect.SetSize(num_states * num_nodes);
   ui.SetSize(num_states);
   uci.SetSize(num_states);
   xi.SetSize(dim);
   grad_nu_i.SetSize(dim);
   grad_rho_i.SetSize(dim);
   curl_i.SetSize(3);
   DenseMatrix uc(elfun.GetData(), num_nodes, num_states);
   DenseMatrix u; u = uc;
   u.Transpose(); //NOTE: Do not manipulate elfun directly
   DenseMatrix Dui(num_states, dim);
   Vector Duidi(num_states);

   // convert momentum to velocity
   for (int nn = 0; nn < num_nodes; nn++)
   {
      for (int di = 0; di < dim; di++)
      {
         u(di+1, nn) *= 1.0/u(0, nn);
      }
      //u(dim+2, nn) *= 1.0/u(0, nn);
   }

   // precompute certain values
   elvect = 0.0; 
   for (int i = 0; i < num_nodes; ++i)
   {
      // get the Jacobian (Trans.Weight) and cubature weight (node.weight)
      const IntegrationPoint &node = fe.GetNodes().IntPoint(i);
      Trans.SetIntPoint(&node);
      u.GetColumn(i, ui);
      //uc.GetRow(i, uci);
      Trans.Transform(node, xi);

      // get Dui
      Duidi = 0.0; Dui = 0.0;
      for (int di = 0; di < dim; di++)
      {
         sbp.multStrongOperator(di, i, u, Duidi);
         Dui.SetCol(di, Duidi);
      }

      // compute vorticity at node and take magnitude
      calcVorticity<double, dim>(Dui.GetData(), Trans.InverseJacobian().GetData(), curl_i.GetData()); 
      double S = sqrt((curl_i(0)*curl_i(0) + curl_i(1)*curl_i(1) +curl_i(2)*curl_i(2)));
      //S = abs(curl_i(2));
      // compute gradient of turbulent viscosity at node
      calcGrad<double, dim>(dim+2, Dui.GetData(), Trans.InverseJacobian().GetData(), grad_nu_i.GetData()); 
      // compute gradient of density at node
      calcGrad<double, dim>(0, Dui.GetData(), Trans.InverseJacobian().GetData(), grad_rho_i.GetData()); 

      // get distance function value at node
      double d = dist.GetValue(Trans.ElementNo, node); //temp solution, y distance from wall

      // accumulate source terms
      double src = calcSAFullSource<double, dim>(ui.GetData(), 
                                 mu, d, S, Re, d0, grad_nu_i.GetData(), 
                                 grad_rho_i.GetData(), sacs.GetData(),
                                 prod, dest);
      

      //byNODES
      elvect(i + num_nodes*(num_states-1)) = -alpha * Trans.Weight() * node.weight * src;
      //elvect((dim+2) + i*(num_states)) = alpha * Trans.Weight() * node.weight * src;
   } // loop over element nodes i
}

template <int dim>
void SASourceIntegrator<dim>::AssembleElementGrad(
    const mfem::FiniteElement &fe, mfem::ElementTransformation &Trans,
    const mfem::Vector &elfun, mfem::DenseMatrix &elmat)
{
   using namespace mfem;
   const SBPFiniteElement &sbp = dynamic_cast<const SBPFiniteElement &>(fe);
   int num_nodes = sbp.GetDof();
#ifdef MFEM_THREAD_SAFE
   Vector ui, uci, xi, uj, grad_nu_i, curl_i;
#endif
   ui.SetSize(num_states);
   uci.SetSize(num_states);
   xi.SetSize(dim);
   grad_nu_i.SetSize(dim);
   grad_rho_i.SetSize(dim);
   curl_i.SetSize(3);
   DenseMatrix uc(elfun.GetData(), num_nodes, num_states);
   DenseMatrix u; u = uc;
   u.Transpose();
   DenseMatrix Dui(num_states, dim);
   Vector Duidi(num_states);
   vector<DenseMatrix> jac_curl(dim);
   vector<DenseMatrix> jac_grad_nu(dim);
   vector<DenseMatrix> jac_grad_rho(dim);
   std::vector<adouble> sacs_a(sacs.Size());
   adept::set_values(sacs_a.data(), sacs.Size(), sacs.GetData());
   
   // each matrix represents d/dx, d/dy, d/dz
   for (int d = 0; d < dim; ++d)
   {
      jac_curl[d].SetSize(3, num_states);
      jac_grad_nu[d].SetSize(dim, num_states);
      jac_grad_rho[d].SetSize(dim, num_states);
   }

   // partial derivative terms
   Vector dSdc(3); //partial vorticity mag (S) w.r.t. curl
   Vector dSrcdu(num_states); //partial src w.r.t. u
   Vector dSrcdS(1); //partial src w.r.t. S
   Vector dSrcdgradnu(dim); //partial src w.r.t grad
   Vector dSrcdgradrho(dim); //partial src w.r.t grad
   Vector work1(num_nodes*num_states); 
   Vector work2s(num_states);
   Vector work2gn(num_states);
   Vector work2gr(num_states);
   Vector work3(num_nodes);
   Vector dnu(num_nodes*num_states); //total derivative

   // convert momentum to velocity
   for (int nn = 0; nn < num_nodes; nn++)
   {
      for (int di = 0; di < dim; di++)
      {
         u(di+1, nn) *= 1.0/u(0, nn);
      }
      //u(dim+2, nn) *= 1.0/u(0, nn);
   }

   elmat.SetSize(num_states*num_nodes);
   elmat = 0.0;
   for (int i = 0; i < num_nodes; ++i)
   {
      dnu = 0.0;
      // get the Jacobian (Trans.Weight) and cubature weight (node.weight)
      const IntegrationPoint &node = fe.GetNodes().IntPoint(i);
      Trans.SetIntPoint(&node);
      u.GetColumn(i, ui);
      //uc.GetRow(i, uci);
      Trans.Transform(node, xi);

      // get Dui, insert each spatial derivative into column matrix
      Duidi = 0.0; Dui = 0.0;
      for (int di = 0; di < dim; di++)
      {
         sbp.multStrongOperator(di, i, u, Duidi);
         Dui.SetCol(di, Duidi);
      }
      
      int Dui_size = Dui.Height() * Dui.Width();
      // set adept inputs
      std::vector<adouble> ui_a(ui.Size());
      //std::vector<adouble> Dui_a(Dui_size);
      std::vector<adouble> curl_i_a(curl_i.Size());
      std::vector<adouble> grad_nu_i_a(grad_nu_i.Size());
      std::vector<adouble> grad_rho_i_a(grad_rho_i.Size());
      adouble mu_a = mu;
      adouble Re_a = Re;
      adept::set_values(ui_a.data(), ui.Size(), ui.GetData());

      // compute vorticity at node
      calcVorticity<double, dim>(Dui.GetData(), 
         Trans.InverseJacobian().GetData(), curl_i.GetData()); 
      calcVorticityJacDw<dim>(this->stack, Dui.GetData(), 
         Trans.InverseJacobian().GetData(), jac_curl);

      // compute gradient of turbulent viscosity at node
      calcGrad<double, dim>(dim+2, Dui.GetData(), 
         Trans.InverseJacobian().GetData(), grad_nu_i.GetData()); 
      calcGradJacDw<dim>(this->stack, dim+2, Dui.GetData(), 
         Trans.InverseJacobian().GetData(), jac_grad_nu);
      // compute gradient of density at node
      calcGrad<double, dim>(0, Dui.GetData(), 
         Trans.InverseJacobian().GetData(), grad_rho_i.GetData()); //curl.GetRow(i, curl_i);
      calcGradJacDw<dim>(this->stack, 0, Dui.GetData(), 
         Trans.InverseJacobian().GetData(), jac_grad_rho);

      // set adept outputs
      adept::set_values(curl_i_a.data(), curl_i.Size(), curl_i.GetData());
      adept::set_values(grad_nu_i_a.data(), grad_nu_i.Size(), grad_nu_i.GetData());
      adept::set_values(grad_rho_i_a.data(), grad_rho_i.Size(), grad_rho_i.GetData());

      // get distance function value at node
      adouble d = dist.GetValue(Trans.ElementNo, node); //evaluate the distance gridfunction at the node
      adouble d0_a = d0; 
      // if (d < 1e-12)
      //    d = d0;

      // vorticity magnitude deriv
      this->stack.new_recording();
      adouble S = sqrt((curl_i_a[0]*curl_i_a[0] + curl_i_a[1]*curl_i_a[1] +curl_i_a[2]*curl_i_a[2]));
      //S = abs(curl_i_a[2]);
      this->stack.independent(curl_i_a.data(), curl_i.Size());
      this->stack.dependent(S);
      this->stack.jacobian(dSdc.GetData());
      adouble src;

      dSrcdu = 0.0;
      // ui differentiation
      this->stack.new_recording();
      src = calcSAFullSource<adouble, dim>(ui_a.data(), 
                                 mu_a, d, S, Re_a, d0_a, grad_nu_i_a.data(), 
                                 grad_rho_i_a.data(), sacs_a.data(),
                                 prod, dest);
      //src = prod*(uci_a[dim+2]-3)*d; ///!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      this->stack.independent(ui_a.data(), ui.Size());
      this->stack.dependent(src);
      this->stack.jacobian(dSrcdu.GetData());

      dSrcdS = 0.0;
      // S differentiation
      this->stack.new_recording();
      src = calcSAFullSource<adouble, dim>(ui_a.data(), 
                                 mu_a, d, S, Re_a, d0_a, grad_nu_i_a.data(), 
                                 grad_rho_i_a.data(), sacs_a.data(),
                                 prod, dest);
      //src = prod*(uci_a[dim+2]-3)*d; ///!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      this->stack.independent(S);
      this->stack.dependent(src);
      this->stack.jacobian(dSrcdS.GetData());
      
      dSrcdgradnu = 0.0;
      // grad nu differentiation
      this->stack.new_recording();
      src = calcSAFullSource<adouble, dim>(ui_a.data(), 
                                 mu_a, d, S, Re_a, d0_a, grad_nu_i_a.data(), 
                                 grad_rho_i_a.data(), sacs_a.data(),
                                 prod, dest);
      //src = prod*(uci_a[dim+2]-3)*d; ///!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      this->stack.independent(grad_nu_i_a.data(), grad_nu_i_a.size());
      this->stack.dependent(src);
      this->stack.jacobian(dSrcdgradnu.GetData());

      dSrcdgradrho = 0.0;
      // grad rho differentiation
      this->stack.new_recording();
      src = calcSAFullSource<adouble, dim>(ui_a.data(), 
                                 mu_a, d, S, Re_a, d0_a, grad_nu_i_a.data(), 
                                 grad_rho_i_a.data(), sacs_a.data(),
                                 prod, dest);
      //src = prod*(uci_a[dim+2]-3)*d;  ///!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      this->stack.independent(grad_rho_i_a.data(), grad_rho_i_a.size());
      this->stack.dependent(src);
      this->stack.jacobian(dSrcdgradrho.GetData());

      // Assemble nu derivative

      // Dui sensitivities
      for (int di = 0; di < dim; di++)
      {
         work1 = 0.0; work2s = 0.0; work2gn = 0.0; work2gr = 0.0; work3 = 0.0;
         sbp.getStrongOperator(di, i, work3);
         jac_curl[di].MultTranspose(dSdc, work2s);
         work2s *= dSrcdS(0);
         jac_grad_nu[di].MultTranspose(dSrcdgradnu, work2gn);
         jac_grad_rho[di].MultTranspose(dSrcdgradrho, work2gr);
         for (int nn = 0; nn < num_nodes; nn++)
         {
            // assuming ordering BY NODES
            for (int ns = 0; ns < num_states; ns++)
            {
               work1(nn + ns*num_nodes) = work3(nn)*(work2s(ns) + work2gn(ns) + work2gr(ns));
            }
         }
         dnu += work1;
      }

      // Direct ui sensitivities
      for(int ns = 0; ns < num_states; ns ++)
      {
         dnu(i + ns*num_nodes) += dSrcdu(ns);
      }      

      // node weighting
      dnu *= (Trans.Weight() * node.weight);

      // account for momentum conversion
      for (int nn = 0; nn < num_nodes; nn++)
      {
         for (int di = 0; di < dim; di++)
         {
            dnu(nn) -= dnu(nn + (di+1)*num_nodes)*uc(nn, di+1)/(uc(nn, 0)*uc(nn, 0));
            dnu(nn + (di+1)*num_nodes) *= 1.0/uc(nn, 0);
         }
         //dnu(nn) -= dnu(nn + (dim+2)*num_nodes)*uc(nn, dim+2)/(uc(nn, 0)*uc(nn, 0));
         //dnu(nn + (dim+2)*num_nodes) *= 1.0/uc(nn, 0);
      }

      // Set elmat entry

      //byNODES
      //elmat.SetRow((num_states-1)+i*num_states, dnu);
      elmat.SetRow(i+num_nodes*(num_states-1), dnu);
   } // loop over element nodes i
   elmat *= -alpha;
}

//====================================================================================
//SA Inviscid Integrator Methods

template <int dim, bool entvar>
void SAInviscidIntegrator<dim, entvar>::calcFlux(int di, const mfem::Vector &qL,
                                                const mfem::Vector &qR,
                                                mfem::Vector &flux)
{
   if (entvar)
   {
      calcIsmailRoeFluxUsingEntVars<double, dim>(di, qL.GetData(), qR.GetData(),
                                                 flux.GetData());
   }
   else
   {
      calcIsmailRoeFlux<double, dim>(di, qL.GetData(), qR.GetData(),
                                     flux.GetData());
   }
   //add flux term for SA
   //flux(dim+2) = 0.5*(qfs(di+1)/qfs(0) + qfs(di+1)/qfs(0))*0.5*(qL(dim+2) + qR(dim+2));
   flux(dim+2) = 0.5*(qL(di+1)/qL(0) + qR(di+1)/qR(0))*0.5*(qL(dim+2) + qR(dim+2));
}

template <int dim, bool entvar>
void SAInviscidIntegrator<dim, entvar>::calcFluxJacStates(
    int di, const mfem::Vector &qL, const mfem::Vector &qR,
    mfem::DenseMatrix &jacL, mfem::DenseMatrix &jacR)
{
   // store the full jacobian in jac
   mfem::DenseMatrix jac(dim + 3, 2 * (dim + 3));
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
   jacL.CopyCols(jac, 0, dim + 2);
   // retrieve the jacobian w.r.t right state
   jacR.CopyCols(jac, dim + 3, 2 * (dim + 3) - 1);
   // add SA variable contributions ///NOTE: no real need to do it this way
   jacL(dim+2,0) = -0.25*(qL(di+1)/(qL(0)*qL(0)))*(qL(dim+2) + qR(dim+2));
   jacR(dim+2,0) = -0.25*(qR(di+1)/(qR(0)*qR(0)))*(qL(dim+2) + qR(dim+2));
   jacL(dim+2,di+1) = 0.25*(1.0/qL(0))*(qL(dim+2) + qR(dim+2));
   jacR(dim+2,di+1) = 0.25*(1.0/qR(0))*(qL(dim+2) + qR(dim+2));   
   jacL(dim+2,dim+2) = 0.25*(qL(di+1)/qL(0) + qR(di+1)/qR(0));
   jacR(dim+2,dim+2) = 0.25*(qL(di+1)/qL(0) + qR(di+1)/qR(0));
   // jacL(dim+2,dim+2) = 0.5*(qfs(di+1)/qfs(0) + qfs(di+1)/qfs(0));
   // jacR(dim+2,dim+2) = 0.5*(qfs(di+1)/qfs(0) + qfs(di+1)/qfs(0));
}

//====================================================================================
//SA Boundary Integrator Methods

//==============================================================================
//SA No-Slip Adiabatic Wall Integrator methods
template <int dim>
void SANoSlipAdiabaticWallBC<dim>::calcFlux(const mfem::Vector &x,
                                          const mfem::Vector &dir, double jac,
                                          const mfem::Vector &q,
                                          const mfem::DenseMatrix &Dw,
                                          mfem::Vector &flux_vec)
{
   // Step 1: apply the EC slip wall flux
   calcSlipWallFlux<double, dim>(x.GetData(), dir.GetData(), q.GetData(),
                                 flux_vec.GetData());
   flux_vec(dim+2) = 0.0;
   // Step 2: evaluate the adiabatic flux
   double mu_Re = mu;
   if (mu < 0.0)
   {
      mu_Re = calcSutherlandViscosity<double, dim>(q.GetData());
   }
   double fv1 = calcSACoefficient<double, dim>(q.GetData(), mu, 
                                                      sacs.GetData());
   if (q(dim+2)<0)
      fv1 = 0.0;
   double mu_Re_SA = (mu_Re + q(0)*q(dim+2)*fv1)/Re;
   calcAdiabaticWallFlux<double, dim>(dir.GetData(), mu_Re_SA, Pr, q.GetData(),
                                      Dw.GetData(), work_vec.GetData());
   work_vec(dim+2) = 0.0;
   flux_vec -= work_vec; // note the minus sign!!!
   // evaluate wall normal eddy viscosity flux
   double grad[dim];
   for (int di = 0; di < dim; di++)
      grad[di] = Dw(dim+2, di);
   double SAflux = dot<double, dim>(dir.GetData(), grad);
   double fn = calcSANegativeCoefficient<double, dim>(q.GetData(), mu, 
                                                      sacs.GetData());
   flux_vec(dim+2) -= (mu/q(0) + fn*q(dim+2))*SAflux/(sacs(2)*Re);
   // Step 3: evaluate the no-slip penalty
   calcNoSlipPenaltyFlux<double, dim>(dir.GetData(), jac, mu_Re_SA, Pr, qfs.GetData(),
                                      q.GetData(), work_vec.GetData());
   work_vec(dim+2) = 0.0;
   flux_vec += work_vec;
   double dnu = q(dim+2);
   double dnuflux = (mu/q(0) + fn*q(dim+2))*dnu/(sacs(2)*Re);
   double fac = 10*sqrt(dot<double,dim>(dir, dir))/jac;
   flux_vec(dim+2) += dnuflux*fac;
}

template <int dim>
void SANoSlipAdiabaticWallBC<dim>::calcFluxJacState(
    const mfem::Vector &x, const mfem::Vector &dir, double jac,
    const mfem::Vector &q, const mfem::DenseMatrix &Dw,
    mfem::DenseMatrix &flux_jac)
{
   // create containers for active double objects for each input
   int Dw_size = Dw.Height() * Dw.Width();
   std::vector<adouble> x_a(x.Size());
   std::vector<adouble> q_a(q.Size());
   std::vector<adouble> Dw_a(Dw_size);
   std::vector<adouble> work_vec_a(work_vec.Size());
   std::vector<adouble> dir_a(dir.Size());
   std::vector<adouble> qfs_a(qfs.Size());
   std::vector<adouble> sacs_a(sacs.Size());
   // initialize active double containers with data from inputs
   adept::set_values(q_a.data(), q.Size(), q.GetData());
   adept::set_values(dir_a.data(), dir.Size(), dir.GetData());
   adept::set_values(Dw_a.data(), Dw_size, Dw.GetData());
   adept::set_values(qfs_a.data(), qfs.Size(), qfs.GetData());
   adept::set_values(x_a.data(), x.Size(), x.GetData());
   adept::set_values(sacs_a.data(), sacs.Size(), sacs.GetData());
   // start new stack recording
   this->stack.new_recording();
   // create container for active double flux output
   std::vector<adouble> flux_a(q.Size());
   // Step 1: apply the EC slip wall flux
   mach::calcSlipWallFlux<adouble, dim>(x_a.data(), dir_a.data(), q_a.data(),
                                        flux_a.data());
   flux_a[dim+2] = 0.0;
   // Step 2: evaluate the adiabatic flux
   adouble mu_Re = mu;
   if (mu < 0.0)
   {
      mu_Re = mach::calcSutherlandViscosity<adouble, dim>(q_a.data());
   }
   //mu_Re /= Re;
   adouble fv1 = calcSACoefficient<adouble, dim>(q_a.data(), mu, 
                                                      sacs_a.data());
   if (q_a[dim+2]<0)
      fv1 = 0.0;
   adouble mu_Re_SA = (mu_Re + q_a[0]*q_a[dim+2]*fv1)/Re;
   mach::calcAdiabaticWallFlux<adouble, dim>(dir_a.data(), mu_Re_SA, Pr, q_a.data(),
                                             Dw_a.data(), work_vec_a.data());
   work_vec_a[dim+2] = 0.0;
   for (int i = 0; i < flux_a.size(); ++i)
   {
      flux_a[i] -= work_vec_a[i]; // note the minus sign!!!
   }
   // evaluate wall normal eddy viscosity flux
   adouble grad[dim];
   for (int di = 0; di < dim; di++)
      grad[di] = Dw_a[dim+2 + di*(dim+3)];
   adouble SAflux = dot<adouble, dim>(dir_a.data(), grad);
   adouble fn = calcSANegativeCoefficient<adouble, dim>(q_a.data(), mu, 
                                                      sacs_a.data());
   flux_a[dim+2] -= (mu/q_a[0] + fn*q_a[dim+2])*SAflux/(sacs_a[2]*Re);
   // Step 3: evaluate the no-slip penalty
   mach::calcNoSlipPenaltyFlux<adouble, dim>(dir_a.data(), jac, mu_Re_SA, Pr, qfs_a.data(),
                                             q_a.data(), work_vec_a.data());
   work_vec_a[dim+2] = 0.0;
   for (int i = 0; i < flux_a.size(); ++i)
   {
      flux_a[i] += work_vec_a[i];
   }
   adouble dnu = q_a[dim+2];
   adouble dnuflux = (mu/q_a[0] + fn*q_a[dim+2])*dnu/(sacs_a[2]*Re);
   adouble fac = 10*sqrt(dot<adouble,dim>(dir_a.data(), dir_a.data()))/jac;
   flux_a[dim+2] += dnuflux*fac;

   this->stack.independent(q_a.data(), q.Size());
   this->stack.dependent(flux_a.data(), q.Size());
   this->stack.jacobian(flux_jac.GetData());
}

template <int dim>
void SANoSlipAdiabaticWallBC<dim>::calcFluxJacDw(const mfem::Vector &x, const mfem::Vector &dir, double jac,
                                               const mfem::Vector &q, const mfem::DenseMatrix &Dw,
                                               vector<mfem::DenseMatrix> &flux_jac)
{
   // create containers for active double objects for each input
   int Dw_size = Dw.Height() * Dw.Width();
   std::vector<adouble> x_a(x.Size());
   std::vector<adouble> q_a(q.Size());
   std::vector<adouble> Dw_a(Dw_size);
   std::vector<adouble> work_vec_a(work_vec.Size());
   std::vector<adouble> dir_a(dir.Size());
   std::vector<adouble> qfs_a(qfs.Size());
   std::vector<adouble> sacs_a(sacs.Size());
   // initialize active double containers with data from inputs
   adept::set_values(q_a.data(), q.Size(), q.GetData());
   adept::set_values(dir_a.data(), dir.Size(), dir.GetData());
   adept::set_values(Dw_a.data(), Dw_size, Dw.GetData());
   adept::set_values(qfs_a.data(), qfs.Size(), qfs.GetData());
   adept::set_values(sacs_a.data(), sacs.Size(), sacs.GetData());
   // start new stack recording
   this->stack.new_recording();
   // create container for active double flux output
   std::vector<adouble> flux_a(q.Size());
   // Step 1: apply the EC slip wall flux
   mach::calcSlipWallFlux<adouble, dim>(x_a.data(), dir_a.data(), q_a.data(),
                                        flux_a.data());
   flux_a[dim+2] = 0.0;
   // Step 2: evaluate the adiabatic flux
   adouble mu_Re = mu;
   if (mu < 0.0)
   {
      mu_Re = mach::calcSutherlandViscosity<adouble, dim>(q_a.data());
   }
   //mu_Re /= Re;
   adouble fv1 = calcSACoefficient<adouble, dim>(q_a.data(), mu, 
                                                      sacs_a.data());
   if (q_a[dim+2]<0)
      fv1 = 0.0;
   adouble mu_Re_SA = (mu_Re + q_a[0]*q_a[dim+2]*fv1)/Re;
   mach::calcAdiabaticWallFlux<adouble, dim>(dir_a.data(), mu_Re_SA, Pr, q_a.data(),
                                             Dw_a.data(), work_vec_a.data());
   work_vec_a[dim+2] = 0.0;
   for (int i = 0; i < flux_a.size(); ++i)
   {
      flux_a[i] -= work_vec_a[i]; // note the minus sign!!!
   }
   // evaluate wall normal eddy viscosity flux
   adouble grad[dim];
   for (int di = 0; di < dim; di++)
      grad[di] = Dw_a[dim+2 + di*(dim+3)];
   adouble SAflux = dot<adouble, dim>(dir_a.data(), grad);
   adouble fn = calcSANegativeCoefficient<adouble, dim>(q_a.data(), mu, 
                                                      sacs_a.data());
   flux_a[dim+2] -= (mu/q_a[0] + fn*q_a[dim+2])*SAflux/(sacs_a[2]*Re);
   // Step 3: evaluate the no-slip penalty
   mach::calcNoSlipPenaltyFlux<adouble, dim>(dir_a.data(), jac, mu_Re_SA, Pr, qfs_a.data(),
                                             q_a.data(), work_vec_a.data());
   work_vec_a[dim+2] = 0.0;
   for (int i = 0; i < flux_a.size(); ++i)
   {
      flux_a[i] += work_vec_a[i];
   }
   adouble dnu = q_a[dim+2];
   adouble dnuflux = (mu/q_a[0] + fn*q_a[dim+2])*dnu/(sacs_a[2]*Re);
   adouble fac = 10*sqrt(dot<adouble,dim>(dir_a.data(), dir_a.data()))/jac;
   flux_a[dim+2] += dnuflux*fac;

   this->stack.independent(Dw_a.data(),Dw_size);
   this->stack.dependent(flux_a.data(), q.Size());
   // compute and store jacobian in CDw_jac
   mfem::Vector work(dim*this->num_states*this->num_states);
   this->stack.jacobian(work.GetData());
   for (int i = 0; i < dim; ++i)
   {
      flux_jac[i] = (work.GetData() + i*this->num_states*this->num_states);
   }
}

template <int dim>
void SANoSlipAdiabaticWallBC<dim>::calcFluxDv(const mfem::Vector &x,
                                              const mfem::Vector &dir,
                                              const mfem::Vector &q,
                                              mfem::DenseMatrix &flux_mat)
{
   double mu_Re = mu;
   if (mu < 0.0)
      mu_Re = calcSutherlandViscosity<double, dim>(q.GetData());
   //mu_Re /= Re;
   double fv1 = calcSACoefficient<double, dim>(q.GetData(), mu, 
                                                      sacs.GetData());
   if (q(dim+2)<0)
      fv1 = 0.0;
   double mu_Re_SA = (mu_Re + q(0)*q(dim+2)*fv1)/Re;
   double fn = calcSANegativeCoefficient<double, dim>(q.GetData(), mu, 
                                                      sacs.GetData());
   double mu_2 = (mu_Re/q(0) + q(dim+2)*fn)/Re;
   calcNoSlipDualFluxSA<double, dim>(dir.GetData(), mu_Re_SA, mu_2, Pr, q.GetData(),
                                   flux_mat.GetData());
}

template <int dim>
void SANoSlipAdiabaticWallBC<dim>::calcFluxDvJacState(
   const mfem::Vector &x, const mfem::Vector dir, const mfem::Vector &q,
   std::vector<mfem::DenseMatrix> &flux_jac)
{
   // create containers for active double objects for each input
   int flux_size = dim*(dim+3);
   std::vector<adouble> q_a(q.Size());
   std::vector<adouble> dir_a(dir.Size());
   std::vector<adouble> sacs_a(sacs.Size());
   // initialize active double containers with data from inputs
   adept::set_values(q_a.data(), q.Size(), q.GetData());
   adept::set_values(dir_a.data(), dir.Size(), dir.GetData());
   adept::set_values(sacs_a.data(), sacs.Size(), sacs.GetData());
   // start new stack recording
   this->stack.new_recording();
   // create container for active double flux output
   std::vector<adouble> fluxes_a(flux_size);
   // evaluate the fluxes
   adouble mu_Re = mu;
   if (mu < 0.0)
      mu_Re = calcSutherlandViscosity<adouble, dim>(q_a.data());
   //mu_Re /= Re;
   adouble fv1 = calcSACoefficient<adouble, dim>(q_a.data(), mu, 
                                                      sacs_a.data());
   if (q_a[dim+2]<0)
      fv1 = 0.0;
   adouble mu_Re_SA = (mu_Re + q_a[0]*q_a[dim+2]*fv1)/Re;
   adouble fn = calcSANegativeCoefficient<adouble, dim>(q_a.data(), mu, 
                                                      sacs_a.data());
   adouble mu_2 = (mu_Re/q_a[0] + q_a[dim+2]*fn)/Re;
   calcNoSlipDualFluxSA<adouble, dim>(dir_a.data(), mu_Re_SA, mu_2, Pr, q_a.data(),
                                    fluxes_a.data());

   this->stack.independent(q_a.data(), q.Size());
   this->stack.dependent(fluxes_a.data(), flux_size);
   // compute and store jacobian in flux_jac
   mfem::Vector work(flux_size*(dim+3));
   this->stack.jacobian(work.GetData());
   for (int s = 0; s < dim+3; ++s)
   {
      for (int i = 0; i < dim; ++i)
      {
         flux_jac[i].SetCol(s, work.GetData() + (s*dim + i)*(dim+3));
      }
   }
}


//==============================================================================
//SA Viscous Slip-Wall Integrator methods
template <int dim>
void SAViscousSlipWallBC<dim>::calcFlux(const mfem::Vector &x,
                                          const mfem::Vector &dir, double jac,
                                          const mfem::Vector &q,
                                          const mfem::DenseMatrix &Dw,
                                          mfem::Vector &flux_vec)
{
   // Part 1: apply the inviscid slip wall BCs
   calcSlipWallFlux<double, dim>(x.GetData(), dir.GetData(), q.GetData(),
                                 flux_vec.GetData());
   flux_vec(dim+2) = 0.0;
   int Dw_size = Dw.Height() * Dw.Width();
   mfem::Vector Dw_work(Dw_size);
   setZeroNormalDeriv<double, dim>(dir.GetData(), Dw.GetData(),
                                   Dw_work.GetData());
   // SA treatment
   double nrm[dim];
   double Dw_nrm = 0.0;
   double fac = 1.0 / sqrt(dot<double, dim>(dir, dir));
   for (int i = 0; i < dim; ++i)
   {
      nrm[i] = dir(i) * fac;
      Dw_nrm += Dw(dim+2, i)*nrm[i];
   }
   for (int i = 0; i < dim; ++i)
   {
      Dw_work(dim+2 + i*(dim+3)) = Dw(dim+2, i) - nrm[i]*Dw_nrm;
   }

   // Part 2: viscous BCs
   double mu_Re = mu;
   if (mu < 0.0)
   {
      mu_Re = calcSutherlandViscosity<double, dim>(q.GetData());
   }
   //mu_Re /= Re;
   double fv1 = calcSACoefficient<double, dim>(q.GetData(), mu, 
                                                      sacs.GetData());
   if (q(dim+2)<0)
      fv1 = 0.0;
   double mu_Re_SA = (mu_Re + q(0)*q(dim+2)*fv1)/Re;
   for (int d = 0; d < dim; ++d)
   {
      work_vec = 0.0;
      applyViscousScalingSA<double, dim>(d, mu_Re_SA, Pr, q.GetData(),
                                       Dw_work.GetData(), work_vec.GetData());
      work_vec *= dir(d);
      flux_vec -= work_vec;
      double fn = calcSANegativeCoefficient<double, dim>(q.GetData(), mu, 
                                                      sacs.GetData());
      flux_vec(dim+2) -= (mu/q(0) + fn*q(dim+2))*Dw_work(dim+2 + d*(dim+3))/(sacs(2)*Re);
   }
}

template <int dim>
void SAViscousSlipWallBC<dim>::calcFluxJacState(
    const mfem::Vector &x, const mfem::Vector &dir, double jac,
    const mfem::Vector &q, const mfem::DenseMatrix &Dw,
    mfem::DenseMatrix &flux_jac)
{
   // create containers for active double objects for each input
   int Dw_size = Dw.Height() * Dw.Width();
   std::vector<adouble> x_a(x.Size());
   std::vector<adouble> q_a(q.Size());
   std::vector<adouble> Dw_a(Dw_size);
   std::vector<adouble> work_vec_a(work_vec.Size());
   std::vector<adouble> dir_a(dir.Size());
   std::vector<adouble> sacs_a(sacs.Size());
   // initialize active double containers with data from inputs
   adept::set_values(q_a.data(), q.Size(), q.GetData());
   adept::set_values(dir_a.data(), dir.Size(), dir.GetData());
   adept::set_values(Dw_a.data(), Dw_size, Dw.GetData());
   adept::set_values(x_a.data(), x.Size(), x.GetData());
   adept::set_values(sacs_a.data(), sacs.Size(), sacs.GetData());
   // start new stack recording
   this->stack.new_recording();
   // create container for active double flux output
   std::vector<adouble> flux_a(q.Size());
      // Part 1: apply the inviscid slip wall BCs
   calcSlipWallFlux<adouble, dim>(x_a.data(), dir_a.data(), q_a.data(),
                                 flux_a.data());
   flux_a[dim+2] = 0.0;
   std::vector<adouble> Dw_work_a(Dw_size);
   setZeroNormalDeriv<adouble, dim>(dir_a.data(), Dw_a.data(),
                                   Dw_work_a.data());
   // SA treatment
   adouble nrm[dim];
   adouble Dw_nrm = 0.0;
   adouble fac = 1.0 / sqrt(dot<adouble, dim>(dir_a.data(), dir_a.data()));
   for (int i = 0; i < dim; ++i)
   {
      nrm[i] = dir_a[i] * fac;
      Dw_nrm += Dw_a[dim+2 + i*(dim+3)]*nrm[i];
   }
   for (int i = 0; i < dim; ++i)
   {
      Dw_work_a[dim+2 + i*(dim+3)] = Dw_a[dim+2 + i*(dim+3)] - nrm[i]*Dw_nrm;
   }

   // Part 2: viscous BCs
   adouble mu_Re = mu;
   if (mu < 0.0)
   {
      mu_Re = calcSutherlandViscosity<adouble, dim>(q_a.data());
   }
   //mu_Re /= Re;
   adouble fv1 = calcSACoefficient<adouble, dim>(q_a.data(), mu, 
                                                      sacs_a.data());
   if (q_a[dim+2]<0)
      fv1 = 0.0;
   adouble mu_Re_SA = (mu_Re + q_a[0]*q_a[dim+2]*fv1)/Re;
   for (int d = 0; d < dim; ++d)
   {
      work_vec_a[dim+2] = 0.0;
      applyViscousScalingSA<adouble, dim>(d, mu_Re_SA, Pr, q_a.data(),
                                       Dw_work_a.data(), work_vec_a.data());
      for (int k = 0; k < dim+2; k++)
      {
         work_vec_a[k] *= dir_a[d];
         flux_a[k] -= work_vec_a[k];
      }
      adouble fn = calcSANegativeCoefficient<adouble, dim>(q_a.data(), mu, 
                                                      sacs_a.data());
      flux_a[dim+2] -= (mu/q_a[0] + fn*q_a[dim+2])*Dw_work_a[dim+2 + d*(dim+3)]/(sacs_a[2]*Re);
   }
   this->stack.independent(q_a.data(), q.Size());
   this->stack.dependent(flux_a.data(), q.Size());
   this->stack.jacobian(flux_jac.GetData());
}

template <int dim>
void SAViscousSlipWallBC<dim>::calcFluxJacDw(const mfem::Vector &x, const mfem::Vector &dir, double jac,
                                               const mfem::Vector &q, const mfem::DenseMatrix &Dw,
                                               vector<mfem::DenseMatrix> &flux_jac)
{
   // create containers for active double objects for each input
   int Dw_size = Dw.Height() * Dw.Width();
   std::vector<adouble> x_a(x.Size());
   std::vector<adouble> q_a(q.Size());
   std::vector<adouble> Dw_a(Dw_size);
   std::vector<adouble> work_vec_a(work_vec.Size());
   std::vector<adouble> dir_a(dir.Size());
   std::vector<adouble> sacs_a(sacs.Size());
   // initialize active double containers with data from inputs
   adept::set_values(q_a.data(), q.Size(), q.GetData());
   adept::set_values(dir_a.data(), dir.Size(), dir.GetData());
   adept::set_values(Dw_a.data(), Dw_size, Dw.GetData());
   adept::set_values(sacs_a.data(), sacs.Size(), sacs.GetData());
   // start new stack recording
   this->stack.new_recording();
   // create container for active double flux output
   std::vector<adouble> flux_a(q.Size());
   // Part 1: apply the inviscid slip wall BCs
   calcSlipWallFlux<adouble, dim>(x_a.data(), dir_a.data(), q_a.data(),
                                 flux_a.data());
   flux_a[dim+2] = 0.0;
   std::vector<adouble> Dw_work_a(Dw_size);
   setZeroNormalDeriv<adouble, dim>(dir_a.data(), Dw_a.data(),
                                   Dw_work_a.data());
   // SA treatment
   adouble nrm[dim];
   adouble Dw_nrm = 0.0;
   adouble fac = 1.0 / sqrt(dot<adouble, dim>(dir_a.data(), dir_a.data()));
   for (int i = 0; i < dim; ++i)
   {
      nrm[i] = dir_a[i] * fac;
      Dw_nrm += Dw_a[dim+2 + i*(dim+3)]*nrm[i];
   }
   for (int i = 0; i < dim; ++i)
   {
      Dw_work_a[dim+2 + i*(dim+3)] = Dw_a[dim+2 + i*(dim+3)] - nrm[i]*Dw_nrm;
   }

   // Part 2: viscous BCs
   adouble mu_Re = mu;
   if (mu < 0.0)
   {
      mu_Re = calcSutherlandViscosity<adouble, dim>(q_a.data());
   }
   //mu_Re /= Re;
   adouble fv1 = calcSACoefficient<adouble, dim>(q_a.data(), mu, 
                                                      sacs_a.data());
   if (q_a[dim+2]<0)
      fv1 = 0.0;
   adouble mu_Re_SA = (mu_Re + q_a[0]*q_a[dim+2]*fv1)/Re;
   for (int d = 0; d < dim; ++d)
   {
      work_vec_a[dim+2] = 0.0;
      applyViscousScalingSA<adouble, dim>(d, mu_Re_SA, Pr, q_a.data(),
                                       Dw_work_a.data(), work_vec_a.data());
      for (int k = 0; k < dim+2; k++)
      {
         work_vec_a[k] *= dir_a[d];
         flux_a[k] -= work_vec_a[k];
      }
      adouble fn = calcSANegativeCoefficient<adouble, dim>(q_a.data(), mu, 
                                                      sacs_a.data());
      flux_a[dim+2] -= (mu/q_a[0] + fn*q_a[dim+2])*Dw_work_a[dim+2 + d*(dim+3)]/(sacs_a[2]*Re);
   }

   this->stack.independent(Dw_a.data(),Dw_size);
   this->stack.dependent(flux_a.data(), q.Size());
   // compute and store jacobian in CDw_jac
   mfem::Vector work(dim*this->num_states*this->num_states);
   this->stack.jacobian(work.GetData());
   for (int i = 0; i < dim; ++i)
   {
      flux_jac[i] = (work.GetData() + i*this->num_states*this->num_states);
   }
}

//====================================================================================
//SA Far Field Integrator Methods
template <int dim>
void SAFarFieldBC<dim>::calcFlux(const mfem::Vector &x, const mfem::Vector &dir, double jac,
                 const mfem::Vector &q, const mfem::DenseMatrix &Dw,
                 mfem::Vector &flux_vec)
{
   calcBoundaryFlux<double, dim>(dir.GetData(), qfs.GetData(), q.GetData(),
                                    work_vec.GetData(), flux_vec.GetData());

#if 0   
   // Part 2: evaluate the adiabatic flux
   double mu_Re = mu;
   if (mu < 0.0)
   {
      mu_Re = calcSutherlandViscosity<double, dim>(q.GetData());
   }
   //mu_Re /= Re;
   adouble mu_Re_SA = (mu_Re + q_a[dim+2]*fv1)/Re;
   for (int d = 0; d < dim; ++d)
   {
      work_vec = 0.0;
      applyViscousScalingSA<double, dim>(d, mu_Re_SA, Pr, q.GetData(), Dw.GetData(),
                                       work_vec.GetData());
      work_vec *= dir[d];
      flux_vec -= work_vec;      
   }
#endif    

   //flux_vec(dim+2) = 0;
   // handle SA variable
   double Unrm = dot<double, dim>(dir.GetData(), qfs.GetData()+1); 
   double fac = 1.0;//sqrt(dot<double, dim>(dir, dir));
   if (Unrm > 0.0)
   {
      
      flux_vec(dim+2) = Unrm*q(dim+2)/(q(0)*fac);
   }
   else
   {
      //double dq = q(dim+2) - qfs(dim+2);
      flux_vec(dim+2) = Unrm*qfs(dim+2)/(qfs(0)*fac);//Unrm*dq/fac;
   }
}


template <int dim>
void SAFarFieldBC<dim>::calcFluxJacState(
    const mfem::Vector &x, const mfem::Vector &dir, double jac,
    const mfem::Vector &q, const mfem::DenseMatrix &Dw,
    mfem::DenseMatrix &flux_jac)
{
   flux_jac = 0.0;
//   mach::calcFluxJacState<dim>(x, dir, jac, q, Dw, qfs, work_vec, this->stack, flux_jac);
   
   int Dw_size = Dw.Height() * Dw.Width();
   // create containers for active double objects for each input
   std::vector<adouble> q_a(q.Size());
   std::vector<adouble> dir_a(dir.Size());
   std::vector<adouble> x_a(x.Size());
   std::vector<adouble> Dw_a(Dw_size);
   std::vector<adouble> qfs_a(q.Size());
   std::vector<adouble> work_vec_a(work_vec.Size());
   // initialize active double containers with data from inputs
   adept::set_values(q_a.data(), q.Size(), q.GetData());
   adept::set_values(dir_a.data(), dir.Size(), dir.GetData());
   adept::set_values(x_a.data(), x.Size(), x.GetData());
   adept::set_values(Dw_a.data(), Dw_size, Dw.GetData());
   adept::set_values(qfs_a.data(), qfs.Size(), qfs.GetData());
   // start new stack recording
   this->stack.new_recording();
   // create container for active double flux output
   std::vector<adouble> flux_a(q.Size());
   // Part 1: apply the inviscid inflow boundary condition
   mach::calcBoundaryFlux<adouble, dim>(dir_a.data(), qfs_a.data(), q_a.data(),
                                        work_vec_a.data(), flux_a.data());
#if 0
   adouble mu_Re = mu;
   if (mu < 0.0)
   {
      mu_Re = mach::calcSutherlandViscosity<adouble, dim>(q_a.data());
   }
   mu_Re /= Re;
   adouble mu_Re_SA = mu_Re + q_a[dim+2]*fv1/Re;
   for (int d = 0; d < dim; ++d)
   {
      for (int i = 0; i < work_vec_a.size(); ++i)
      {
         work_vec_a[i] = 0.0;
      }
      applyViscousScalingSA<adouble, dim>(d, mu_Re_SA, Pr, q_a.data(), Dw_a.data(),
                                       work_vec_a.data());
      for (int i = 0; i < flux_a.size(); ++i)
      {
         work_vec_a[i] *= dir_a[d];
         flux_a[i] -= work_vec_a[i]; // note the minus sign!!!
      }
   }
#endif

   adouble Unrm = dot<adouble, dim>(dir_a.data(), qfs_a.data()+1); 
   adouble fac = 1.0;//sqrt(dot<adouble, dim>(dir_a.data(), dir_a.data()));
   if (Unrm > 0.0)
   {
      flux_a[dim+2] = Unrm*q_a[dim+2]/(q_a[0]*fac);
   }
   else
   {
      //adouble dq = q_a[dim+2] - qfs_a[dim+2];
      flux_a[dim+2] = Unrm*qfs_a[dim+2]/(qfs_a[0]*fac); //Unrm*dq/fac;
   }

   this->stack.independent(q_a.data(), q.Size());
   this->stack.dependent(flux_a.data(), q.Size());
   this->stack.jacobian(flux_jac.GetData());   
   
   // handle SA variable
   // double Unrm = dot<double, dim>(dir.GetData(), qfs.GetData()+1); 
   // double fac = sqrt(dot<double, dim>(dir, dir));
   // if (Unrm > 0.0)
   // {
   //    flux_jac(dim+2, 0) = -Unrm*q(dim+2)/(q(0)*q(0)*fac);
   //    for(int di = 0; di < dim; di++)
   //    {
   //       flux_jac(dim+2, di+1) = dir(di)*q(dim+2)/(q(0)*fac);
   //    }
   //    flux_jac(dim+2, dim+2) = Unrm/(q(0)*fac);
   // }
   // else
   // {
   //    // for(int di = 0; di < dim; di++)
   //    // {
   //    //    flux_jac(dim+2, di+1) = dir(di)*qfs(dim+2)/qfs(0);
   //    // }
   // }
}

template <int dim>
void SAFarFieldBC<dim>::calcFluxJacDw(
   const mfem::Vector &x, const mfem::Vector &dir, double jac,
   const mfem::Vector &q, const mfem::DenseMatrix &Dw,
   vector<mfem::DenseMatrix> &flux_jac)
{
   // Presently, this BC has no dependence on the derivative
   for (int i = 0; i < dim; ++i)
   {
      flux_jac[i] = 0.0;
   }
}

//====================================================================================
//SA LPS Integrator Methods

template <int dim, bool entvar>
void SALPSIntegrator<dim, entvar>::convertVars(
    const mfem::Vector &q, mfem::Vector &w)
{
   // This conditional should have no overhead, if the compiler is good
   if (entvar)
   {
      w = q;
   }
   else
   {
      calcEntropyVars<double, dim>(q.GetData(), w.GetData());
      // SA variable copy
      w(dim+2) = q(dim+2);
   }
}

template <int dim, bool entvar>
void SALPSIntegrator<dim, entvar>::convertVarsJacState(
   const mfem::Vector &q, mfem::DenseMatrix &dwdu)
{
   if (entvar)
   {
      dwdu = 0.0;
      for (int i = 0; i < dim+3; ++i)
      {
         dwdu(i,i) = 1.0;
      }
   }
   else
   {
      dwdu = 0.0;
      // vector of active input variables
      std::vector<adouble> q_a(q.Size());
      // initialize adouble inputs
      adept::set_values(q_a.data(), q.Size(), q.GetData());
      // start recording
      this->stack.new_recording();
      // create vector of active output variables
      std::vector<adouble> w_a(q.Size());
      // run algorithm
      calcEntropyVars<adouble, dim>(q_a.data(), w_a.data());
      // identify independent and dependent variables
      this->stack.independent(q_a.data(), q.Size());
      this->stack.dependent(w_a.data(), q.Size());
      // compute and store jacobian in dwdu
      this->stack.jacobian(dwdu.GetData());
      // SA Variable Derivative
      dwdu(dim+2, dim+2) = 1.0;
   }
}

template <int dim, bool entvar>
void SALPSIntegrator<dim, entvar>::applyScaling(
   const mfem::DenseMatrix &adjJ, const mfem::Vector &q,
   const mfem::Vector &vec, mfem::Vector &mat_vec)
{
   if (entvar)
   {
      applyLPSScalingUsingEntVars<double, dim>(adjJ.GetData(), q.GetData(),
                                               vec.GetData(), mat_vec.GetData());
      throw MachException("Entropy variables not yet supported");
   }
   else
   {
      applyLPSScaling<double,dim>(adjJ.GetData(), q.GetData(), vec.GetData(),
                                  mat_vec.GetData());
      // SA Variable
      double U = sqrt(dot<double, dim>(q.GetData()+1, q.GetData()+1))/q(0);
      mat_vec(dim+2) = U*vec(dim+2);
   }
}

template <int dim, bool entvar>
void SALPSIntegrator<dim, entvar>::applyScalingJacState(
    const mfem::DenseMatrix &adjJ, const mfem::Vector &q,
    const mfem::Vector &vec, mfem::DenseMatrix &mat_vec_jac)
{
   // declare vectors of active input variables
   int adjJ_a_size = adjJ.Height() * adjJ.Width();
   std::vector<adouble> adjJ_a(adjJ_a_size);
   std::vector<adouble> q_a(q.Size());
   std::vector<adouble> vec_a(vec.Size());
   // copy data from mfem::Vector
   adept::set_values(adjJ_a.data(), adjJ_a_size, adjJ.GetData());
   adept::set_values(q_a.data(), q.Size(), q.GetData());
   adept::set_values(vec_a.data(), vec.Size(), vec.GetData());
   // start recording
   this->stack.new_recording();
   // the dependent variable must be declared after the recording
   std::vector<adouble> mat_vec_a(q.Size());
   if (entvar)
   {
      applyLPSScalingUsingEntVars<adouble, dim>(adjJ_a.data(), q_a.data(),
                                                vec_a.data(), mat_vec_a.data());
   }
   else
   {
      applyLPSScaling<adouble, dim>(adjJ_a.data(), q_a.data(),
                                    vec_a.data(), mat_vec_a.data());
      adouble U = sqrt(dot<adouble, dim>(q_a.data()+1, q_a.data()+1))/q_a[0];
      mat_vec_a[dim+2] = U*vec_a[dim+2];   
   }
   // set the independent and dependent variable
   this->stack.independent(q_a.data(), q.Size());
   this->stack.dependent(mat_vec_a.data(), q.Size());
   // Calculate the jabobian
   this->stack.jacobian(mat_vec_jac.GetData());
}

template <int dim, bool entvar>
void SALPSIntegrator<dim, entvar>::applyScalingJacV(
    const mfem::DenseMatrix &adjJ, const mfem::Vector &q,
    mfem::DenseMatrix &mat_vec_jac)
{
   // declare vectors of active input variables
   int adjJ_a_size = adjJ.Height() * adjJ.Width();
   std::vector<adouble> adjJ_a(adjJ_a_size);
   std::vector<adouble> q_a(q.Size());
   std::vector<adouble> vec_a(q.Size());
   // copy data from mfem::Vector
   adept::set_values(adjJ_a.data(), adjJ_a_size, adjJ.GetData());
   adept::set_values(q_a.data(), q.Size(), q.GetData());
   // dependence on vec is linear, so any value is ok; use q
   adept::set_values(vec_a.data(), q.Size(), q.GetData());
   // start recording
   this->stack.new_recording();
   // the dependent variable must be declared after the recording
   std::vector<adouble> mat_vec_a(q.Size());
   if (entvar)
   {
      // applyLPSScalingUsingEntVars<adouble, dim>(adjJ_a.data(), q_a.data(),
      //                                           vec_a.data(), mat_vec_a.data());
      throw MachException("Entropy variables not yet supported");
   }
   else
   {
      applyLPSScaling<adouble, dim>(adjJ_a.data(), q_a.data(),
                                    vec_a.data(), mat_vec_a.data());
      adouble U = sqrt(dot<adouble, dim>(q_a.data()+1, q_a.data()+1))/q_a[0];
      mat_vec_a[dim+2] = U*vec_a[dim+2];                              
   }
   // set the independent and dependent variable
   this->stack.independent(vec_a.data(), q.Size());
   this->stack.dependent(mat_vec_a.data(), q.Size());
   // Calculate the jabobian
   this->stack.jacobian(mat_vec_jac.GetData());
}

//==============================================================================
// SAViscousIntegrator methods
template <int dim>
void SAViscousIntegrator<dim>::applyScaling(int d, const mfem::Vector &x, 
                     const mfem::Vector &q, const mfem::DenseMatrix &Dw, 
                     mfem::Vector &CDw)
{
   double mu_Re = mu;
   if (mu < 0.0)
   {
      mu_Re = calcSutherlandViscosity<double, dim>(q.GetData());
   }
   //mu_Re /= Re;
   double fv1 = calcSACoefficient<double, dim>(q.GetData(), mu, 
                                                      sacs.GetData());
   if (q(dim+2)<0)
      fv1 = 0.0;
   double mu_Re_SA = (mu_Re + q(0)*q(dim+2)*fv1)/Re;
   applyViscousScalingSA<double, dim>(d, mu_Re_SA, Pr, q.GetData(), Dw.GetData(),
                                    CDw.GetData());
   double fn = calcSANegativeCoefficient<double, dim>(q.GetData(), mu, 
                                                      sacs.GetData());
   CDw(dim+2) = (mu/q(0) + fn*q(dim+2))*Dw(dim+2, d)/(sacs(2)*Re);
   // if (mu + fn*q(dim+2) < 0)
   //    cout << "Negative diffusion!: " <<mu + fn*q(dim+2) <<endl;
}

template <int dim>
void SAViscousIntegrator<dim>::applyScalingJacState(int d, const mfem::Vector &x, 
                              const mfem::Vector &q, const mfem::DenseMatrix &Dw, 
                              mfem::DenseMatrix &CDw_jac)
{
   // vector of active input variables
   int Dw_size = Dw.Height() * Dw.Width();
   std::vector<adouble> q_a(q.Size());
   std::vector<adouble> Dw_a(Dw_size);
   std::vector<adouble> sacs_a(sacs.Size());
   // initialize adouble inputs
   adept::set_values(q_a.data(), q.Size(), q.GetData());
   adept::set_values(Dw_a.data(), Dw_size, Dw.GetData());
   adept::set_values(sacs_a.data(), sacs.Size(), sacs.GetData());
   // start recording
   this->stack.new_recording();
   // create vector of active output variables
   std::vector<adouble> CDw_a(q.Size());
   // run algorithm
   adouble mu_Re = mu;
   if (mu < 0.0)
   {
      mu_Re = calcSutherlandViscosity<adouble, dim>(q_a.data());
   }
   //mu_Re /= Re;
   adouble fv1 = calcSACoefficient<adouble, dim>(q_a.data(), mu, 
                                                      sacs_a.data());
   if (q_a[dim+2]<0)
      fv1 = 0.0;
   adouble mu_Re_SA = (mu_Re + q_a[0]*q_a[dim+2]*fv1)/Re;
   applyViscousScalingSA<adouble, dim>(d, mu_Re_SA, Pr, q_a.data(), Dw_a.data(),
                                    CDw_a.data());
   adouble fn = calcSANegativeCoefficient<adouble, dim>(q_a.data(), mu, 
                                                      sacs_a.data());
   CDw_a[dim+2] = (mu/q_a[0] + fn*q_a[dim+2])*Dw_a[dim+2 + d*(dim+3)]/(sacs_a[2]*Re);
   // identify independent and dependent variables
   this->stack.independent(q_a.data(), q.Size());
   this->stack.dependent(CDw_a.data(), q.Size());
   // compute and store jacobian in CDw_jac
   this->stack.jacobian(CDw_jac.GetData());
}

template <int dim>
void SAViscousIntegrator<dim>::applyScalingJacDw(
    int d, const mfem::Vector &x, const mfem::Vector &q,
    const mfem::DenseMatrix &Dw, vector<mfem::DenseMatrix> &CDw_jac)
{
   // vector of active input variables
   int Dw_size = Dw.Height() * Dw.Width();
   std::vector<adouble> q_a(q.Size());
   std::vector<adouble> Dw_a(Dw_size);
   std::vector<adouble> sacs_a(sacs.Size());
   // initialize adouble inputs
   adept::set_values(q_a.data(), q.Size(), q.GetData());
   adept::set_values(Dw_a.data(), Dw_size, Dw.GetData());
   adept::set_values(sacs_a.data(), sacs.Size(), sacs.GetData());
   // start recording
   this->stack.new_recording();
   // create vector of active output variables
   std::vector<adouble> CDw_a(q.Size());
   // run algorithm
   adouble mu_Re = mu;
   if (mu < 0.0)
   {
      mu_Re = calcSutherlandViscosity<adouble, dim>(q_a.data());
   }
   //mu_Re /= Re;
   adouble fv1 = calcSACoefficient<adouble, dim>(q_a.data(), mu, 
                                                      sacs_a.data());
   if (q_a[dim+2]<0)
      fv1 = 0.0;
   adouble mu_Re_SA = (mu_Re + q_a[0]*q_a[dim+2]*fv1)/Re;
   applyViscousScalingSA<adouble, dim>(d, mu_Re_SA, Pr, q_a.data(), Dw_a.data(),
                                    CDw_a.data());
   adouble fn = calcSANegativeCoefficient<adouble, dim>(q_a.data(), mu, 
                                                      sacs_a.data());
   CDw_a[dim+2] = (mu/q_a[0] + fn*q_a[dim+2])*Dw_a[dim+2 + d*(dim+3)]/(sacs_a[2]*Re);
   // identify independent and dependent variables
   this->stack.independent(Dw_a.data(), Dw_size);
   this->stack.dependent(CDw_a.data(), q.Size());
   // compute and store jacobian in CDw_jac
   mfem::Vector work(dim*this->num_states*this->num_states);
   this->stack.jacobian(work.GetData());
   for (int i = 0; i < dim; ++i)
   {
      CDw_jac[i] = (work.GetData() + i*this->num_states*this->num_states);
   }
}