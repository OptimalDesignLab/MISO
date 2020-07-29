//==============================================================================
// ESViscousSAIntegrator methods


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
   calcVorticitySBP(u, sbp, Trans, curl);
   calcGradSBP(u, sbp, Trans, grad);
   elvect = 0.0; 
   for (int i = 0; i < num_nodes; ++i)
   {
      // get the Jacobian (Trans.Weight) and cubature weight (node.weight)
      IntegrationPoint &node = fe.GetNodes().IntPoint(i);
      Trans.SetIntPoint(&node);
      u.GetRow(i, ui);
      Trans.Transform(node, xi);

      // compute vorticity at node and take magnitude
      curl.GetRow(i, curl_i);
      double S = curl_i.Norml2();

      // compute gradient of turbulent viscosity at node
      grad.GetRow(i, grad_i); 

      // get distance function value at node
      double d = 1.0;

      // accumulate source terms
      double src = calcSASource<adouble,dim>(
         ui.GetData(), grad_i.GetData(), sacs.GetData());
      // use negative model if turbulent viscosity is negative
      if (ui(dim+2) < 0)
      {
         src += calcSANegativeProduction<adouble,dim>(
            ui.GetData(), S, sacs.GetData());
         src += calcSANegativeDestruction<adouble,dim>(
            ui.GetData(), d, sacs.GetData());
      }
      else
      {
         src += calcSAProduction<adouble,dim>(
            ui.GetData(), mu, d, S, sacs.GetData());
         src += calcSADestruction<adouble,dim>(
            ui.GetData(), mu, d, S, sacs.GetData());
      }

      res(i, dim+2) += alpha * Trans.Weight() * node.weight * src;
   } // loop over element nodes i
}

template <int dim>
void SASourceIntegrator<dim>::AssembleElementGrad(
    const mfem::FiniteElement &fe, mfem::ElementTransformation &Trans,
    const mfem::Vector &elfun, mfem::DenseMatrix &elmat)
{
   using namespace mfem;
   using namespace std;
   const SBPFiniteElement &sbp = dynamic_cast<const SBPFiniteElement &>(fe);
   int num_nodes = sbp.GetDof();
#ifdef MFEM_THREAD_SAFE
   Vector ui, xi, uj, grad_i, curl_i;
   DenseMatrix grad, curl;
#endif
   ui.SetSize(num_states);
   xi.SetSize(dim);
   grad.SetSize(num_nodes, dim);
   curl.SetSize(num_nodes, 3);
   grad_i.SetSize(dim);
   curl_i.SetSize(3);
   DenseMatrix u(elfun.GetData(), num_nodes, num_states); 

   //precompute vorticity and derivatives
   calcVorticitySBP(u, sbp, Trans, curl);

   //precompute gradient and derivatives
   calcGradSBP(u, sbp, Trans, grad);

   elmat = 0.0; 
   for (int i = 0; i < num_nodes; ++i)
   {
      // get the Jacobian (Trans.Weight) and cubature weight (node.weight)
      IntegrationPoint &node = fe.GetNodes().IntPoint(i);
      Trans.SetIntPoint(&node);
      u.GetRow(i, ui);
      Trans.Transform(node, xi);

      // compute vorticity at node and take magnitude
      curl.GetRow(i, curl_i);
      double S = curl_i.Norml2();

      // compute gradient of turbulent viscosity at node
      grad.GetRow(i, grad_i); 

      // get distance function value at node
      double d = 1.0;

      // accumulate source terms
      double src = calcSASource<adouble,dim>(
         ui.GetData(), grad_i.GetData(), sacs.GetData());
      double P; double D; 
      // use negative model if turbulent viscosity is negative
      if (ui(dim+2) < 0)
      {
         P = calcSANegativeProduction<adouble,dim>(
            ui.GetData(), S, sacs.GetData());
         D = calcSANegativeDestruction<adouble,dim>(
            ui.GetData(), d, sacs.GetData());
      }
      else
      {
         P = calcSAProduction<adouble,dim>(
            ui.GetData(), mu, d, S, sacs.GetData());
         D = calcSADestruction<adouble,dim>(
            ui.GetData(), mu, d, S, sacs.GetData());
      }

      //res(i, dim+2) += alpha * Trans.Weight() * node.weight * src;
   } // loop over element nodes i
}

template <int dim>
void SASourceIntegrator<dim>::calcVorticitySBP(const mfem::DenseMatrix &q, const mfem::FiniteElement &fe, 
                      mfem::ElementTransformation &Trans, mfem::DenseMatrix curl)
{
   using namespace mfem;
   const SBPFiniteElement &sbp = dynamic_cast<const SBPFiniteElement &>(fe);
   int num_nodes = sbp.GetDof();
   DenseMatrix dq(q.Height(), q.Width()); //contains state derivatives in reference space
   DenseMatrix dxi(dim*q.Height(), dim); //contains velocity derivatives in reference space
   dq = 0.0;
   for(int di = 0; di < dim; di++)
   {
      sbp.multWeakOperator(di, q, dq);
      //need to scale with 1/H, probably when looping over nodes
      dxi.CopyMN(dq, q.Height(), dim, 1, dim+1, di*q.Height(), 0);
   }

   DenseMatrix dx(dim, dim); //contains velocity derivatives in absolute space
   DenseMatrix dxin(dim, dim); //contains velocity derivatives in reference space for a node
   Vector dxrow(dim); Vector curln(dim);
   for (int i = 0; i < num_nodes; ++i)
   {
      // get the Jacobian
      IntegrationPoint &node = sbp.GetNodes().IntPoint(i);
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

template <int dim>
void SASourceIntegrator<dim>::calcGradSBP(const mfem::DenseMatrix &q, const mfem::FiniteElement &fe, 
                      mfem::ElementTransformation &Trans, mfem::DenseMatrix grad)
{
   using namespace mfem;
   const SBPFiniteElement &sbp = dynamic_cast<const SBPFiniteElement &>(fe);
   int num_nodes = sbp.GetDof();
   DenseMatrix dq(q.Height(), q.Width()); //contains state derivatives in reference space
   DenseMatrix dnu(q.Height(), dim); //contains turb variable derivatives in reference space
   dq = 0.0;
   for(int di = 0; di < dim; di++)
   {
      sbp.multWeakOperator(di, q, dq);
      //need to scale with 1/H
      dnu.SetCol(di, dq.GetColumn(dim+2));
   }

   Vector dnurow(dim); Vector gradn(dim);
   for (int i = 0; i < num_nodes; ++i)
   {
      // get the Jacobian
      IntegrationPoint &node = sbp.GetNodes().IntPoint(i);
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
   ///NOTE: Differentiate this function with adept directly, check euler/navier stokes integrators
   //operate on original state variables
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
   // add SA variable contributions
   jacL(dim+2,0) = -0.25*(qL(di+1)/(qL(0)*qL(0)))*(qL(dim+2) + qR(dim+2));
   jacR(dim+2,0) = -0.25*(qR(di+1)/(qR(0)*qR(0)))*(qL(dim+2) + qR(dim+2));
   jacL(dim+2,di+1) = 0.25*(1.0/qL(0))*(qL(dim+2) + qR(dim+2));
   jacR(dim+2,di+1) = 0.25*(1.0/qR(0))*(qL(dim+2) + qR(dim+2));
   jacL(dim+2,dim+2) = 0.25*(qL(di+1)/qL(0) + qR(di+1)/qR(0));
   jacR(dim+2,dim+2) = 0.25*(qL(di+1)/qL(0) + qR(di+1)/qR(0));

}

//====================================================================================
//SA Boundary Integrator Methods

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
   mu_Re /= Re;

   for (int d = 0; d < dim; ++d)
   {
      work_vec = 0.0;
      applyViscousScaling<double, dim>(d, mu_Re, Pr, q.GetData(), Dw.GetData(),
                                       work_vec.GetData());
      work_vec *= dir[d];
      flux_vec -= work_vec;      
   }
#endif    

   //flux_vec(dim+2) = 0;
   // handle SA variable
   double Unrm = dot<double, dim>(dir.GetData(), qfs.GetData()+1); 
   if (Unrm > 0.0)
   {
      flux_vec(dim+2) = Unrm*q(dim+2)/q(0);
   }
   else
   {
      flux_vec(dim+2) = Unrm*qfs(dim+2)/qfs(0);
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
   for (int d = 0; d < dim; ++d)
   {
      for (int i = 0; i < work_vec_a.size(); ++i)
      {
         work_vec_a[i] = 0.0;
      }
      applyViscousScaling<adouble, dim>(d, mu_Re, Pr, q_a.data(), Dw_a.data(),
                                       work_vec_a.data());
      for (int i = 0; i < flux_a.size(); ++i)
      {
         work_vec_a[i] *= dir_a[d];
         flux_a[i] -= work_vec_a[i]; // note the minus sign!!!
      }
   }
#endif

   this->stack.independent(q_a.data(), q.Size());
   this->stack.dependent(flux_a.data(), q.Size());
   this->stack.jacobian(flux_jac.GetData());   
   
   // handle SA variable
   double Unrm = dot<double, dim>(dir.GetData(), qfs.GetData()+1); 
   if (Unrm > 0.0)
   {
      flux_jac(dim+2, 0) = -Unrm*q(dim+2)/(q(0)*q(0));
      // for(int di = 0; di < dim; di++)
      // {
      //    flux_jac(dim+2, di+1) = dir(di)*q(dim+2)/q(0);
      // }
      flux_jac(dim+2, dim+2) = Unrm/q(0);
   }
   else
   {
      // for(int di = 0; di < dim; di++)
      // {
      //    flux_jac(dim+2, di+1) = dir(di)*qfs(dim+2)/qfs(0);
      // }
   }
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