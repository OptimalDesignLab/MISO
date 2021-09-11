//==============================================================================
// ESViscousIntegrator methods

template <int dim>
void ESViscousIntegrator<dim>::applyScalingJacState(int d,
                                                    const mfem::Vector &x,
                                                    const mfem::Vector &q,
                                                    const mfem::DenseMatrix &Dw,
                                                    mfem::DenseMatrix &CDw_jac)
{
   // vector of active input variables
   int Dw_size = Dw.Height() * Dw.Width();
   std::vector<adouble> q_a(q.Size());
   std::vector<adouble> Dw_a(Dw_size);
   // initialize adouble inputs
   adept::set_values(q_a.data(), q.Size(), q.GetData());
   adept::set_values(Dw_a.data(), Dw_size, Dw.GetData());
   // start recording
   this->stack.new_recording();
   // create vector of active output variables
   std::vector<adouble> CDw_a(q.Size());
   // run algorithm
   adouble mu_Re = mu;
   if (mu < 0.0)
   {
      mu_Re = mach::calcSutherlandViscosity<adouble, dim>(q_a.data());
   }
   mu_Re /= Re;
   applyViscousScaling<adouble, dim>(
       d, mu_Re, Pr, q_a.data(), Dw_a.data(), CDw_a.data());
   // identify independent and dependent variables
   this->stack.independent(q_a.data(), q.Size());
   this->stack.dependent(CDw_a.data(), q.Size());
   // compute and store jacobian in CDw_jac
   this->stack.jacobian(CDw_jac.GetData());
}

template <int dim>
void ESViscousIntegrator<dim>::applyScalingJacDw(
    int d,
    const mfem::Vector &x,
    const mfem::Vector &q,
    const mfem::DenseMatrix &Dw,
    vector<mfem::DenseMatrix> &CDw_jac)
{
   // vector of active input variables
   int Dw_size = Dw.Height() * Dw.Width();
   std::vector<adouble> q_a(q.Size());
   std::vector<adouble> Dw_a(Dw_size);
   // initialize adouble inputs
   adept::set_values(q_a.data(), q.Size(), q.GetData());
   adept::set_values(Dw_a.data(), Dw_size, Dw.GetData());
   // start recording
   this->stack.new_recording();
   // create vector of active output variables
   std::vector<adouble> CDw_a(q.Size());
   // run algorithm
   adouble mu_Re = mu;
   if (mu < 0.0)
   {
      mu_Re = mach::calcSutherlandViscosity<adouble, dim>(q_a.data());
   }
   mu_Re /= Re;
   applyViscousScaling<adouble, dim>(
       d, mu_Re, Pr, q_a.data(), Dw_a.data(), CDw_a.data());
   // identify independent and dependent variables
   this->stack.independent(Dw_a.data(), Dw_size);
   this->stack.dependent(CDw_a.data(), q.Size());
   // compute and store jacobian in CDw_jac
   mfem::Vector work(dim * this->num_states * this->num_states);
   this->stack.jacobian(work.GetData());
   for (int i = 0; i < dim; ++i)
   {
      CDw_jac[i] = (work.GetData() + i * this->num_states * this->num_states);
   }
}

//==============================================================================
// NoSlipAdiabaticWallBC methods

template <int dim>
double NoSlipAdiabaticWallBC<dim>::calcBndryFun(const mfem::Vector &x,
                                                const mfem::Vector &dir,
                                                double jac,
                                                const mfem::Vector &u,
                                                const mfem::DenseMatrix &Dw)
{
   mfem::Vector flux_vec(u.Size());
   calcFlux(x, dir, jac, u, Dw, flux_vec);
   mfem::Vector w(u.Size());
   calcEntropyVars<double, dim>(u.GetData(), w.GetData());
   return w * flux_vec;
}

template <int dim>
void NoSlipAdiabaticWallBC<dim>::calcFlux(const mfem::Vector &x,
                                          const mfem::Vector &dir,
                                          double jac,
                                          const mfem::Vector &q,
                                          const mfem::DenseMatrix &Dw,
                                          mfem::Vector &flux_vec)
{
   // Step 1: apply the EC slip wall flux
   calcSlipWallFlux<double, dim>(
       x.GetData(), dir.GetData(), q.GetData(), flux_vec.GetData());
   // Step 2: evaluate the adiabatic flux
   double mu_Re = mu;
   if (mu < 0.0) mu_Re = calcSutherlandViscosity<double, dim>(q.GetData());
   mu_Re /= Re;
   calcAdiabaticWallFlux<double, dim>(
       dir.GetData(), mu_Re, Pr, q.GetData(), Dw.GetData(), work_vec.GetData());
   flux_vec -= work_vec;  // note the minus sign!!!
   // Step 3: evaluate the no-slip penalty
   calcNoSlipPenaltyFlux<double, dim>(dir.GetData(),
                                      jac,
                                      mu_Re,
                                      Pr,
                                      qfs.GetData(),
                                      q.GetData(),
                                      work_vec.GetData());
   flux_vec += work_vec;
}

template <int dim>
void NoSlipAdiabaticWallBC<dim>::calcFluxDv(const mfem::Vector &x,
                                            const mfem::Vector &dir,
                                            const mfem::Vector &q,
                                            mfem::DenseMatrix &flux_mat)
{
   double mu_Re = mu;
   if (mu < 0.0) mu_Re = calcSutherlandViscosity<double, dim>(q.GetData());
   mu_Re /= Re;
   calcNoSlipDualFlux<double, dim>(
       dir.GetData(), mu_Re, Pr, q.GetData(), flux_mat.GetData());
}

template <int dim>
void NoSlipAdiabaticWallBC<dim>::calcFluxJacState(const mfem::Vector &x,
                                                  const mfem::Vector &dir,
                                                  double jac,
                                                  const mfem::Vector &q,
                                                  const mfem::DenseMatrix &Dw,
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
   // initialize active double containers with data from inputs
   adept::set_values(q_a.data(), q.Size(), q.GetData());
   adept::set_values(dir_a.data(), dir.Size(), dir.GetData());
   adept::set_values(Dw_a.data(), Dw_size, Dw.GetData());
   adept::set_values(qfs_a.data(), qfs.Size(), qfs.GetData());
   adept::set_values(x_a.data(), x.Size(), x.GetData());
   // start new stack recording
   this->stack.new_recording();
   // create container for active double flux output
   std::vector<adouble> flux_a(q.Size());
   // Step 1: apply the EC slip wall flux
   mach::calcSlipWallFlux<adouble, dim>(
       x_a.data(), dir_a.data(), q_a.data(), flux_a.data());
   // Step 2: evaluate the adiabatic flux
   adouble mu_Re = mu;
   if (mu < 0.0)
      mu_Re = mach::calcSutherlandViscosity<adouble, dim>(q_a.data());
   mu_Re /= Re;
   mach::calcAdiabaticWallFlux<adouble, dim>(
       dir_a.data(), mu_Re, Pr, q_a.data(), Dw_a.data(), work_vec_a.data());
   for (int i = 0; i < flux_a.size(); ++i)
   {
      flux_a[i] -= work_vec_a[i];  // note the minus sign!!!
   }
   // Step 3: evaluate the no-slip penalty
   mach::calcNoSlipPenaltyFlux<adouble, dim>(dir_a.data(),
                                             jac,
                                             mu_Re,
                                             Pr,
                                             qfs_a.data(),
                                             q_a.data(),
                                             work_vec_a.data());

   for (int i = 0; i < flux_a.size(); ++i)
   {
      flux_a[i] += work_vec_a[i];
   }
   this->stack.independent(q_a.data(), q.Size());
   this->stack.dependent(flux_a.data(), q.Size());
   this->stack.jacobian(flux_jac.GetData());
}

template <int dim>
void NoSlipAdiabaticWallBC<dim>::calcFluxJacDw(
    const mfem::Vector &x,
    const mfem::Vector &dir,
    double jac,
    const mfem::Vector &q,
    const mfem::DenseMatrix &Dw,
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
   // initialize active double containers with data from inputs
   adept::set_values(q_a.data(), q.Size(), q.GetData());
   adept::set_values(dir_a.data(), dir.Size(), dir.GetData());
   adept::set_values(Dw_a.data(), Dw_size, Dw.GetData());
   adept::set_values(qfs_a.data(), qfs.Size(), qfs.GetData());
   // start new stack recording
   this->stack.new_recording();
   // create container for active double flux output
   std::vector<adouble> flux_a(q.Size());
   // Step 1: apply the EC slip wall flux
   mach::calcSlipWallFlux<adouble, dim>(
       x_a.data(), dir_a.data(), q_a.data(), flux_a.data());
   // Step 2: evaluate the adiabatic flux
   adouble mu_Re = mu;
   if (mu < 0.0)
   {
      mu_Re = mach::calcSutherlandViscosity<adouble, dim>(q_a.data());
   }
   mu_Re /= Re;
   mach::calcAdiabaticWallFlux<adouble, dim>(
       dir_a.data(), mu_Re, Pr, q_a.data(), Dw_a.data(), work_vec_a.data());
   for (int i = 0; i < flux_a.size(); ++i)
   {
      flux_a[i] -= work_vec_a[i];  // note the minus sign!!!
   }
   // Step 3: evaluate the no-slip penalty
   mach::calcNoSlipPenaltyFlux<adouble, dim>(dir_a.data(),
                                             jac,
                                             mu_Re,
                                             Pr,
                                             qfs_a.data(),
                                             q_a.data(),
                                             work_vec_a.data());
   for (int i = 0; i < flux_a.size(); ++i)
   {
      flux_a[i] += work_vec_a[i];
   }
   this->stack.independent(Dw_a.data(), Dw_size);
   this->stack.dependent(flux_a.data(), q.Size());
   // compute and store jacobian in CDw_jac
   mfem::Vector work(dim * this->num_states * this->num_states);
   this->stack.jacobian(work.GetData());
   for (int i = 0; i < dim; ++i)
   {
      flux_jac[i] = (work.GetData() + i * this->num_states * this->num_states);
   }
}

template <int dim>
void NoSlipAdiabaticWallBC<dim>::calcFluxDvJacState(
    const mfem::Vector &x,
    const mfem::Vector dir,
    const mfem::Vector &q,
    std::vector<mfem::DenseMatrix> &flux_jac)
{
   // create containers for active double objects for each input
   int flux_size = dim * (dim + 2);
   std::vector<adouble> q_a(q.Size());
   std::vector<adouble> dir_a(dir.Size());
   // initialize active double containers with data from inputs
   adept::set_values(q_a.data(), q.Size(), q.GetData());
   adept::set_values(dir_a.data(), dir.Size(), dir.GetData());
   // start new stack recording
   this->stack.new_recording();
   // create container for active double flux output
   std::vector<adouble> fluxes_a(flux_size);
   // evaluate the fluxes
   adouble mu_Re = mu;
   if (mu < 0.0) mu_Re = calcSutherlandViscosity<adouble, dim>(q_a.data());
   mu_Re /= Re;
   calcNoSlipDualFlux<adouble, dim>(
       dir_a.data(), mu_Re, Pr, q_a.data(), fluxes_a.data());
   this->stack.independent(q_a.data(), q.Size());
   this->stack.dependent(fluxes_a.data(), flux_size);
   // compute and store jacobian in flux_jac
   mfem::Vector work(flux_size * (dim + 2));
   this->stack.jacobian(work.GetData());
   for (int s = 0; s < dim + 2; ++s)
   {
      for (int i = 0; i < dim; ++i)
      {
         flux_jac[i].SetCol(s, work.GetData() + (s * dim + i) * (dim + 2));
      }
   }
}

//==============================================================================
// ViscousSlipWallBC methods

template <int dim>
double ViscousSlipWallBC<dim>::calcBndryFun(const mfem::Vector &x,
                                            const mfem::Vector &dir,
                                            double jac,
                                            const mfem::Vector &u,
                                            const mfem::DenseMatrix &Dw)
{
   mfem::Vector flux_vec(u.Size());
   calcFlux(x, dir, jac, u, Dw, flux_vec);
   mfem::Vector w(u.Size());
   calcEntropyVars<double, dim>(u.GetData(), w.GetData());
   return w * flux_vec;
}

template <int dim>
void ViscousSlipWallBC<dim>::calcFlux(const mfem::Vector &x,
                                      const mfem::Vector &dir,
                                      double jac,
                                      const mfem::Vector &q,
                                      const mfem::DenseMatrix &Dw,
                                      mfem::Vector &flux_vec)
{
   // Part 1: apply the inviscid slip wall BCs
   calcSlipWallFlux<double, dim>(
       x.GetData(), dir.GetData(), q.GetData(), flux_vec.GetData());
#if 0
   // Part 2: supply the viscous flux (based on numercial solution)
   double mu_Re = mu;
   if (mu < 0.0)
   {
      mu_Re = calcSutherlandViscosity<double, dim>(q.GetData());
   }
   mu_Re /= Re;

   calcAdiabaticWallFlux<double, dim>(dir.GetData(), mu_Re, Pr, q.GetData(),
                                      Dw.GetData(), work_vec.GetData());
#endif
#if 1
   int Dw_size = Dw.Height() * Dw.Width();
   mfem::Vector Dw_work(Dw_size);
   setZeroNormalDeriv<double, dim>(
       dir.GetData(), Dw.GetData(), Dw_work.GetData());
   double mu_Re = mu;
   if (mu < 0.0)
   {
      mu_Re = calcSutherlandViscosity<double, dim>(q.GetData());
   }
   mu_Re /= Re;
   for (int d = 0; d < dim; ++d)
   {
      work_vec = 0.0;
      applyViscousScaling<double, dim>(
          d, mu_Re, Pr, q.GetData(), Dw_work.GetData(), work_vec.GetData());
      work_vec *= dir(d);
      flux_vec -= work_vec;
   }
#endif
}

template <int dim>
void ViscousSlipWallBC<dim>::calcFluxJacState(const mfem::Vector &x,
                                              const mfem::Vector &dir,
                                              double jac,
                                              const mfem::Vector &q,
                                              const mfem::DenseMatrix &Dw,
                                              mfem::DenseMatrix &flux_jac)
{
   // create containers for active double objects for each input
   int Dw_size = Dw.Height() * Dw.Width();
   std::vector<adouble> q_a(q.Size());
   std::vector<adouble> dir_a(dir.Size());
   std::vector<adouble> x_a(x.Size());
   std::vector<adouble> Dw_a(Dw_size);
   // initialize active double containers with data from inputs
   adept::set_values(q_a.data(), q.Size(), q.GetData());
   adept::set_values(dir_a.data(), dir.Size(), dir.GetData());
   adept::set_values(x_a.data(), x.Size(), x.GetData());
   adept::set_values(Dw_a.data(), Dw_size, Dw.GetData());
   // start new stack recording
   this->stack.new_recording();
   // create container for active double flux output
   std::vector<adouble> flux_a(q.Size());
   mach::calcSlipWallFlux<adouble, dim>(
       x_a.data(), dir_a.data(), q_a.data(), flux_a.data());
#if 1
   std::vector<adouble> Dw_work(Dw_size);
   setZeroNormalDeriv<adouble, dim>(dir_a.data(), Dw_a.data(), Dw_work.data());
   adouble mu_Re_a = mu;
   if (mu < 0.0)
   {
      mu_Re_a = calcSutherlandViscosity<adouble, dim>(q_a.data());
   }
   mu_Re_a /= Re;
   std::vector<adouble> work_a(q.Size());
   for (int d = 0; d < dim; ++d)
   {
      for (int i = 0; i < q.Size(); ++i)
      {
         work_a[i] = 0.0;
      }
      applyViscousScaling<adouble, dim>(
          d, mu_Re_a, Pr, q_a.data(), Dw_work.data(), work_a.data());
      for (int i = 0; i < q.Size(); ++i)
      {
         flux_a[i] -= dir_a[d] * work_a[i];
      }
   }
#endif
   this->stack.independent(q_a.data(), q.Size());
   this->stack.dependent(flux_a.data(), q.Size());
   this->stack.jacobian(flux_jac.GetData());
}

template <int dim>
void ViscousSlipWallBC<dim>::calcFluxJacDw(const mfem::Vector &x,
                                           const mfem::Vector &dir,
                                           double jac,
                                           const mfem::Vector &q,
                                           const mfem::DenseMatrix &Dw,
                                           vector<mfem::DenseMatrix> &flux_jac)
{
#if 0
   // Presently, this BC has no dependence on the derivative
   for (int i = 0; i < dim; ++i)
   {
      flux_jac[i] = 0.0;
   }
#endif
#if 1
   // create containers for active double objects for each input
   int Dw_size = Dw.Height() * Dw.Width();
   std::vector<adouble> x_a(x.Size());
   std::vector<adouble> q_a(q.Size());
   std::vector<adouble> Dw_a(Dw_size);
   std::vector<adouble> work_vec_a(work_vec.Size());
   std::vector<adouble> dir_a(dir.Size());
   // initialize active double containers with data from inputs
   adept::set_values(q_a.data(), q.Size(), q.GetData());
   adept::set_values(dir_a.data(), dir.Size(), dir.GetData());
   adept::set_values(Dw_a.data(), Dw_size, Dw.GetData());
   // start new stack recording
   this->stack.new_recording();
   // create container for active double flux output
   std::vector<adouble> flux_a(q.Size());
   // Step 1: apply the EC slip wall flux
   mach::calcSlipWallFlux<adouble, dim>(
       x_a.data(), dir_a.data(), q_a.data(), flux_a.data());
   // Step 2: evaluate the derivative flux
   std::vector<adouble> Dw_work(Dw_size);
   setZeroNormalDeriv<adouble, dim>(dir_a.data(), Dw_a.data(), Dw_work.data());
   adouble mu_Re = mu;
   if (mu < 0.0)
   {
      mu_Re = mach::calcSutherlandViscosity<adouble, dim>(q_a.data());
   }
   mu_Re /= Re;
   std::vector<adouble> work_a(q.Size());
   for (int d = 0; d < dim; ++d)
   {
      for (int i = 0; i < q.Size(); ++i)
      {
         work_a[i] = 0.0;
      }
      applyViscousScaling<adouble, dim>(
          d, mu_Re, Pr, q_a.data(), Dw_work.data(), work_a.data());
      for (int i = 0; i < q.Size(); ++i)
      {
         flux_a[i] -= dir_a[d] * work_a[i];
      }
   }
   this->stack.independent(Dw_a.data(), Dw_size);
   this->stack.dependent(flux_a.data(), q.Size());
   // compute and store jacobian in CDw_jac
   mfem::Vector work(dim * this->num_states * this->num_states);
   this->stack.jacobian(work.GetData());
   for (int i = 0; i < dim; ++i)
   {
      flux_jac[i] = (work.GetData() + i * this->num_states * this->num_states);
   }
#endif
}

//==============================================================================
// ViscousInflowBC methods

template <int dim>
double ViscousInflowBC<dim>::calcBndryFun(const mfem::Vector &x,
                                          const mfem::Vector &dir,
                                          double jac,
                                          const mfem::Vector &u,
                                          const mfem::DenseMatrix &Dw)
{
   mfem::Vector flux_vec(u.Size());
   calcFlux(x, dir, jac, u, Dw, flux_vec);
   mfem::Vector w(u.Size());
   calcEntropyVars<double, dim>(u.GetData(), w.GetData());
   return w * flux_vec;
}

template <int dim>
void ViscousInflowBC<dim>::calcFlux(const mfem::Vector &x,
                                    const mfem::Vector &dir,
                                    double jac,
                                    const mfem::Vector &q,
                                    const mfem::DenseMatrix &Dw,
                                    mfem::Vector &flux_vec)
{
   // Part 1: apply the inviscid inflow boundary condition
   calcBoundaryFlux<double, dim>(dir.GetData(),
                                 q_in.GetData(),
                                 q.GetData(),
                                 work_vec.GetData(),
                                 flux_vec.GetData());
#if 0
   // Part 2: evaluate the adiabatic flux
   double mu_Re = mu;
   if (mu < 0.0)
   {
      mu_Re = calcSutherlandViscosity<double, dim>(q.GetData());
   }
   mu_Re /= Re;

   calcAdiabaticWallFlux<double, dim>(dir.GetData(), mu_Re, Pr, q.GetData(),
                                      Dw.GetData(), work_vec.GetData());
   flux_vec -= work_vec; // note the minus sign!!!
#endif
#if 0
   for (int d = 0; d < dim; ++d)
   {
      work_vec = 0.0;
      applyViscousScaling<double, dim>(d, mu_Re, Pr, q.GetData(), Dw.GetData(),
                                       work_vec.GetData());
      work_vec *= dir[d];
      flux_vec -= work_vec;      
   }
#endif
}

template <int dim>
void ViscousInflowBC<dim>::calcFluxJacState(const mfem::Vector &x,
                                            const mfem::Vector &dir,
                                            double jac,
                                            const mfem::Vector &q,
                                            const mfem::DenseMatrix &Dw,
                                            mfem::DenseMatrix &flux_jac)
{
   // function defined in euler_fluxes.hpp
   mach::calcFluxJacState<dim>(
       x, dir, jac, q, Dw, q_in, work_vec, this->stack, flux_jac);
#if 0
   // create containers for active double objects for each input
   int Dw_size = Dw.Height() * Dw.Width();
   std::vector<adouble> q_a(q.Size());
   std::vector<adouble> dir_a(dir.Size());
   std::vector<adouble> x_a(x.Size());
   std::vector<adouble> Dw_a(Dw_size);
   std::vector<adouble> q_in_a(q_in.Size());
   std::vector<adouble> work_vec_a(work_vec.Size());
   // initialize active double containers with data from inputs
   adept::set_values(q_a.data(), q.Size(), q.GetData());
   adept::set_values(dir_a.data(), dir.Size(), dir.GetData());
   adept::set_values(x_a.data(), x.Size(), x.GetData());
   adept::set_values(Dw_a.data(),Dw_size, Dw.GetData());
   adept::set_values(q_in_a.data(), q_in.Size(), q_in.GetData());
   // start new stack recording
   this->stack.new_recording();
   // create container for active double flux output
   std::vector<adouble> flux_a(q.Size());
   mach::calcBoundaryFlux<adouble, dim>(dir_a.data(), q_in_a.data(), q_a.data(),
                                        work_vec_a.data(), flux_a.data());
   this->stack.independent(q_a.data(), q.Size());
   this->stack.dependent(flux_a.data(), q.Size());
   this->stack.jacobian(flux_jac.GetData());
#endif
}

template <int dim>
void ViscousInflowBC<dim>::calcFluxJacDw(const mfem::Vector &x,
                                         const mfem::Vector &dir,
                                         double jac,
                                         const mfem::Vector &q,
                                         const mfem::DenseMatrix &Dw,
                                         vector<mfem::DenseMatrix> &flux_jac)
{
   // Presently, this BC has no dependence on the derivative
   for (int i = 0; i < dim; ++i)
   {
      flux_jac[i] = 0.0;
   }
}

//==============================================================================
// ViscousOutflowBC methods

template <int dim>
double ViscousOutflowBC<dim>::calcBndryFun(const mfem::Vector &x,
                                           const mfem::Vector &dir,
                                           double jac,
                                           const mfem::Vector &u,
                                           const mfem::DenseMatrix &Dw)
{
   mfem::Vector flux_vec(u.Size());
   calcFlux(x, dir, jac, u, Dw, flux_vec);
   mfem::Vector w(u.Size());
   calcEntropyVars<double, dim>(u.GetData(), w.GetData());
   return w * flux_vec;
}

template <int dim>
void ViscousOutflowBC<dim>::calcFlux(const mfem::Vector &x,
                                     const mfem::Vector &dir,
                                     double jac,
                                     const mfem::Vector &q,
                                     const mfem::DenseMatrix &Dw,
                                     mfem::Vector &flux_vec)
{
   // Part 1: apply the inviscid inflow boundary condition
   calcBoundaryFlux<double, dim>(dir.GetData(),
                                 q_out.GetData(),
                                 q.GetData(),
                                 work_vec.GetData(),
                                 flux_vec.GetData());
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
      flux_vec -= work_vec; // note the minus sign!!!
   }
#endif
}

template <int dim>
void ViscousOutflowBC<dim>::calcFluxJacState(const mfem::Vector &x,
                                             const mfem::Vector &dir,
                                             double jac,
                                             const mfem::Vector &q,
                                             const mfem::DenseMatrix &Dw,
                                             mfem::DenseMatrix &flux_jac)
{
   mach::calcFluxJacState<dim>(
       x, dir, jac, q, Dw, q_out, work_vec, this->stack, flux_jac);
#if 0
   int Dw_size = Dw.Height() * Dw.Width();
   // create containers for active double objects for each input
   std::vector<adouble> q_a(q.Size());
   std::vector<adouble> dir_a(dir.Size());
   std::vector<adouble> x_a(x.Size());
   std::vector<adouble> Dw_a(Dw_size);
   std::vector<adouble> q_out_a(q_out.Size());
   std::vector<adouble> work_vec_a(work_vec.Size());
   // initialize active double containers with data from inputs
   adept::set_values(q_a.data(), q.Size(), q.GetData());
   adept::set_values(dir_a.data(), dir.Size(), dir.GetData());
   adept::set_values(x_a.data(), x.Size(), x.GetData());
   adept::set_values(Dw_a.data(), Dw_size, Dw.GetData());
   adept::set_values(q_out_a.data(), q_out.Size(), q_out.GetData());
   // start new stack recording
   this->stack.new_recording();
   // create container for active double flux output
   std::vector<adouble> flux_a(q.Size());
   // Part 1: apply the inviscid inflow boundary condition
   mach::calcBoundaryFlux<adouble, dim>(dir_a.data(), q_out_a.data(), q_a.data(),
                                        work_vec_a.data(), flux_a.data());
   // Part 2: evaluate the adiabatic flux
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
   this->stack.independent(q_a.data(), q.Size());
   this->stack.dependent(flux_a.data(), q.Size());
   this->stack.jacobian(flux_jac.GetData());
#endif
}

template <int dim>
void ViscousOutflowBC<dim>::calcFluxJacDw(const mfem::Vector &x,
                                          const mfem::Vector &dir,
                                          double jac,
                                          const mfem::Vector &q,
                                          const mfem::DenseMatrix &Dw,
                                          vector<mfem::DenseMatrix> &flux_jac)
{
   // Presently, this BC has no dependence on the derivative
   for (int i = 0; i < dim; ++i)
   {
      flux_jac[i] = 0.0;
   }
}

//==============================================================================
// ViscouFarFieldBC methods

template <int dim>
double ViscousFarFieldBC<dim>::calcBndryFun(const mfem::Vector &x,
                                            const mfem::Vector &dir,
                                            double jac,
                                            const mfem::Vector &u,
                                            const mfem::DenseMatrix &Dw)
{
   mfem::Vector flux_vec(u.Size());
   calcFlux(x, dir, jac, u, Dw, flux_vec);
   mfem::Vector w(u.Size());
   calcEntropyVars<double, dim>(u.GetData(), w.GetData());
   return w * flux_vec;
}

template <int dim>
void ViscousFarFieldBC<dim>::calcFluxJacState(const mfem::Vector &x,
                                              const mfem::Vector &dir,
                                              double jac,
                                              const mfem::Vector &q,
                                              const mfem::DenseMatrix &Dw,
                                              mfem::DenseMatrix &flux_jac)
{
   mach::calcFluxJacState<dim>(
       x, dir, jac, q, Dw, qfs, work_vec, this->stack, flux_jac);
#if 0
   // create containers for active double objects for each input
   std::vector<adouble> q_a(q.Size());
   std::vector<adouble> dir_a(dir.Size());
   std::vector<adouble> qfs_a(qfs.Size());
   std::vector<adouble> work_vec_a(work_vec.Size());
   // initialize active double containers with data from inputs
   adept::set_values(q_a.data(), q.Size(), q.GetData());
   adept::set_values(dir_a.data(), dir.Size(), dir.GetData());
   adept::set_values(qfs_a.data(), qfs.Size(), qfs.GetData());
   // start new stack recording
   this->stack.new_recording();
   // create container for active double flux output
   std::vector<adouble> flux_a(q.Size());
   mach::calcBoundaryFlux<adouble, dim>(dir_a.data(), qfs_a.data(), q_a.data(),
                                        work_vec_a.data(), flux_a.data());
   this->stack.independent(q_a.data(), q.Size());
   this->stack.dependent(flux_a.data(), q.Size());
   this->stack.jacobian(flux_jac.GetData());
#endif
}

template <int dim>
void ViscousFarFieldBC<dim>::calcFluxJacDw(const mfem::Vector &x,
                                           const mfem::Vector &dir,
                                           double jac,
                                           const mfem::Vector &q,
                                           const mfem::DenseMatrix &Dw,
                                           vector<mfem::DenseMatrix> &flux_jac)
{
   // Presently, this BC has no dependence on the derivative
   for (int i = 0; i < dim; ++i)
   {
      flux_jac[i] = 0.0;
   }
}

//==============================================================================
// ViscousExactBC methods

template <int dim>
double ViscousExactBC<dim>::calcBndryFun(const mfem::Vector &x,
                                         const mfem::Vector &dir,
                                         double jac,
                                         const mfem::Vector &u,
                                         const mfem::DenseMatrix &Dw)
{
   mfem::Vector flux_vec(u.Size());
   calcFlux(x, dir, jac, u, Dw, flux_vec);
   mfem::Vector w(u.Size());
   calcEntropyVars<double, dim>(u.GetData(), w.GetData());
   return w * flux_vec;
}

template <int dim>
void ViscousExactBC<dim>::calcFlux(const mfem::Vector &x,
                                   const mfem::Vector &dir,
                                   double jac,
                                   const mfem::Vector &q,
                                   const mfem::DenseMatrix &Dw,
                                   mfem::Vector &flux_vec)
{
   // Part 1: apply the characteristic, invsicid BCs
   exactSolution(x, qexact);
   calcBoundaryFlux<double, dim>(dir.GetData(),
                                 qexact.GetData(),
                                 q.GetData(),
                                 work_vec.GetData(),
                                 flux_vec.GetData());
   // Step 2: evaluate the derivative flux
   double mu_Re = mu;
   if (mu < 0.0) mu_Re = calcSutherlandViscosity<double, dim>(q.GetData());
   mu_Re /= Re;
   int Dw_size = Dw.Height() * Dw.Width();
   mfem::Vector Dw_work(Dw_size);
   setZeroNormalDeriv<double, dim>(
       dir.GetData(), Dw.GetData(), Dw_work.GetData());
   for (int d = 0; d < dim; ++d)
   {
      work_vec = 0.0;
      applyViscousScaling<double, dim>(
          d, mu_Re, Pr, q.GetData(), Dw_work.GetData(), work_vec.GetData());
      work_vec *= dir(d);
      flux_vec -= work_vec;
   }
}

template <int dim>
void ViscousExactBC<dim>::calcFluxJacState(const mfem::Vector &x,
                                           const mfem::Vector &dir,
                                           double jac,
                                           const mfem::Vector &q,
                                           const mfem::DenseMatrix &Dw,
                                           mfem::DenseMatrix &flux_jac)
{
   exactSolution(x, qexact);
   // create containers for active double objects for each input
   int Dw_size = Dw.Height() * Dw.Width();
   std::vector<adouble> qexact_a(qexact.Size());
   std::vector<adouble> q_a(q.Size());
   std::vector<adouble> dir_a(dir.Size());
   std::vector<adouble> Dw_a(Dw_size);
   std::vector<adouble> work_vec_a(work_vec.Size());
   // initialize active double containers with data from inputs
   adept::set_values(qexact_a.data(), qexact.Size(), qexact.GetData());
   adept::set_values(q_a.data(), q.Size(), q.GetData());
   adept::set_values(dir_a.data(), dir.Size(), dir.GetData());
   adept::set_values(Dw_a.data(), Dw_size, Dw.GetData());
   // start new stack recording
   this->stack.new_recording();
   // create container for active double flux output
   std::vector<adouble> flux_a(q.Size());
   // Part 1: apply the characteristic, invsicid BCs
   calcBoundaryFlux<adouble, dim>(dir_a.data(),
                                  qexact_a.data(),
                                  q_a.data(),
                                  work_vec_a.data(),
                                  flux_a.data());
   // Step 2: evaluate the derivative flux
   adouble mu_Re = mu;
   if (mu < 0.0) mu_Re = calcSutherlandViscosity<adouble, dim>(q_a.data());
   mu_Re /= Re;
   std::vector<adouble> Dw_work(Dw_size);
   setZeroNormalDeriv<adouble, dim>(dir_a.data(), Dw_a.data(), Dw_work.data());
   std::vector<adouble> work_a(q.Size());
   for (int d = 0; d < dim; ++d)
   {
      for (int i = 0; i < q.Size(); ++i)
      {
         work_a[i] = 0.0;
      }
      applyViscousScaling<adouble, dim>(
          d, mu_Re, Pr, q_a.data(), Dw_work.data(), work_a.data());
      for (int i = 0; i < q.Size(); ++i)
      {
         flux_a[i] -= dir_a[d] * work_a[i];
      }
   }
   this->stack.independent(q_a.data(), q.Size());
   this->stack.dependent(flux_a.data(), q.Size());
   this->stack.jacobian(flux_jac.GetData());
}

template <int dim>
void ViscousExactBC<dim>::calcFluxJacDw(const mfem::Vector &x,
                                        const mfem::Vector &dir,
                                        double jac,
                                        const mfem::Vector &q,
                                        const mfem::DenseMatrix &Dw,
                                        vector<mfem::DenseMatrix> &flux_jac)
{
   exactSolution(x, qexact);
   // create containers for active double objects for each input
   int Dw_size = Dw.Height() * Dw.Width();
   std::vector<adouble> qexact_a(qexact.Size());
   std::vector<adouble> q_a(q.Size());
   std::vector<adouble> Dw_a(Dw_size);
   std::vector<adouble> work_vec_a(work_vec.Size());
   std::vector<adouble> dir_a(dir.Size());
   // initialize active double containers with data from inputs
   adept::set_values(qexact_a.data(), qexact.Size(), qexact.GetData());
   adept::set_values(q_a.data(), q.Size(), q.GetData());
   adept::set_values(dir_a.data(), dir.Size(), dir.GetData());
   adept::set_values(Dw_a.data(), Dw_size, Dw.GetData());
   // start new stack recording
   this->stack.new_recording();
   // create container for active double flux output
   std::vector<adouble> flux_a(q.Size());
   // Step 1: apply the characteristic BCs
   calcBoundaryFlux<adouble, dim>(dir_a.data(),
                                  qexact_a.data(),
                                  q_a.data(),
                                  work_vec_a.data(),
                                  flux_a.data());
   // Step 2: evaluate the adiabatic flux
   adouble mu_Re = mu;
   if (mu < 0.0) mu_Re = calcSutherlandViscosity<adouble, dim>(q_a.data());
   mu_Re /= Re;
   std::vector<adouble> Dw_work(Dw_size);
   setZeroNormalDeriv<adouble, dim>(dir_a.data(), Dw_a.data(), Dw_work.data());
   std::vector<adouble> work_a(q.Size());
   for (int d = 0; d < dim; ++d)
   {
      for (int i = 0; i < q.Size(); ++i)
      {
         work_a[i] = 0.0;
      }
      applyViscousScaling<adouble, dim>(
          d, mu_Re, Pr, q_a.data(), Dw_work.data(), work_a.data());
      for (int i = 0; i < q.Size(); ++i)
      {
         flux_a[i] -= dir_a[d] * work_a[i];
      }
   }
   this->stack.independent(Dw_a.data(), Dw_size);
   this->stack.dependent(flux_a.data(), q.Size());
   // compute and store jacobian in CDw_jac
   mfem::Vector work(dim * this->num_states * this->num_states);
   this->stack.jacobian(work.GetData());
   for (int i = 0; i < dim; ++i)
   {
      flux_jac[i] = (work.GetData() + i * this->num_states * this->num_states);
   }
}

//==============================================================================
// SurfaceForce methods

template <int dim>
double SurfaceForce<dim>::calcBndryFun(const mfem::Vector &x,
                                       const mfem::Vector &dir,
                                       double jac,
                                       const mfem::Vector &q,
                                       const mfem::DenseMatrix &Dw)
{
   mfem::Vector flux_vec(q.Size());
   // Step 1: apply the EC slip wall flux
   calcSlipWallFlux<double, dim>(
       x.GetData(), dir.GetData(), q.GetData(), flux_vec.GetData());
   // Step 2: evaluate the adiabatic flux
   double mu_Re = mu;
   if (mu < 0.0)
   {
      mu_Re = calcSutherlandViscosity<double, dim>(q.GetData());
   }
   mu_Re /= Re;
   calcAdiabaticWallFlux<double, dim>(
       dir.GetData(), mu_Re, Pr, q.GetData(), Dw.GetData(), work_vec.GetData());
   flux_vec -= work_vec;  // note the minus sign!!!
   // Step 3: evaluate the no-slip penalty
   calcNoSlipPenaltyFlux<double, dim>(dir.GetData(),
                                      jac,
                                      mu_Re,
                                      Pr,
                                      qfs.GetData(),
                                      q.GetData(),
                                      work_vec.GetData());
   flux_vec += work_vec;
   return dot<double, dim>(force_nrm.GetData(), flux_vec.GetData() + 1);
}

template <int dim>
void SurfaceForce<dim>::calcBndryFunJacState(const mfem::Vector &x,
                                             const mfem::Vector &dir,
                                             double jac,
                                             const mfem::Vector &q,
                                             const mfem::DenseMatrix &Dw,
                                             mfem::Vector &flux_vec)
{
   // create containers for active double objects for each input
   int Dw_size = Dw.Height() * Dw.Width();
   std::vector<adouble> x_a(x.Size());
   std::vector<adouble> q_a(q.Size());
   std::vector<adouble> Dw_a(Dw_size);
   std::vector<adouble> dir_a(dir.Size());
   std::vector<adouble> qfs_a(qfs.Size());
   std::vector<adouble> force_nrm_a(force_nrm.Size());
   std::vector<adouble> work_vec_a(work_vec.Size());
   // initialize active double containers with data from inputs
   adept::set_values(x_a.data(), x.Size(), x.GetData());
   adept::set_values(q_a.data(), q.Size(), q.GetData());
   adept::set_values(Dw_a.data(), Dw_size, Dw.GetData());
   adept::set_values(dir_a.data(), dir.Size(), dir.GetData());
   adept::set_values(qfs_a.data(), qfs.Size(), qfs.GetData());
   adept::set_values(force_nrm_a.data(), force_nrm.Size(), force_nrm.GetData());
   // start new stack recording
   this->stack.new_recording();
   // create container for active double flux output
   std::vector<adouble> flux_a(q.Size());
   // Step 1: apply the EC slip wall flux
   mach::calcSlipWallFlux<adouble, dim>(
       x_a.data(), dir_a.data(), q_a.data(), flux_a.data());
   // Step 2: evaluate the adiabatic flux
   adouble mu_Re = mu;
   if (mu < 0.0)
   {
      mu_Re = mach::calcSutherlandViscosity<adouble, dim>(q_a.data());
   }
   mu_Re /= Re;
   mach::calcAdiabaticWallFlux<adouble, dim>(
       dir_a.data(), mu_Re, Pr, q_a.data(), Dw_a.data(), work_vec_a.data());
   for (int i = 0; i < flux_a.size(); ++i)
   {
      flux_a[i] -= work_vec_a[i];  // note the minus sign!!!
   }
   // Step 3: evaluate the no-slip penalty
   mach::calcNoSlipPenaltyFlux<adouble, dim>(dir_a.data(),
                                             jac,
                                             mu_Re,
                                             Pr,
                                             qfs_a.data(),
                                             q_a.data(),
                                             work_vec_a.data());
   for (int i = 0; i < flux_a.size(); ++i)
   {
      flux_a[i] += work_vec_a[i];
   }
   adouble fun_a = dot<adouble, dim>(force_nrm_a.data(), flux_a.data() + 1);
   fun_a.set_gradient(1.0);
   this->stack.compute_adjoint();
   adept::get_gradients(q_a.data(), q.Size(), flux_vec.GetData());
}

template <int dim>
void SurfaceForce<dim>::calcBndryFunJacDw(const mfem::Vector &x,
                                          const mfem::Vector &dir,
                                          double jac,
                                          const mfem::Vector &q,
                                          const mfem::DenseMatrix &Dw,
                                          mfem::DenseMatrix &flux_mat)
{
   // create containers for active double objects for each input
   int Dw_size = Dw.Height() * Dw.Width();
   std::vector<adouble> x_a(x.Size());
   std::vector<adouble> q_a(q.Size());
   std::vector<adouble> Dw_a(Dw_size);
   std::vector<adouble> dir_a(dir.Size());
   std::vector<adouble> qfs_a(qfs.Size());
   std::vector<adouble> force_nrm_a(force_nrm.Size());
   std::vector<adouble> work_vec_a(work_vec.Size());
   // initialize active double containers with data from inputs
   adept::set_values(x_a.data(), x.Size(), x.GetData());
   adept::set_values(q_a.data(), q.Size(), q.GetData());
   adept::set_values(Dw_a.data(), Dw_size, Dw.GetData());
   adept::set_values(dir_a.data(), dir.Size(), dir.GetData());
   adept::set_values(qfs_a.data(), qfs.Size(), qfs.GetData());
   adept::set_values(force_nrm_a.data(), force_nrm.Size(), force_nrm.GetData());
   // start new stack recording
   this->stack.new_recording();
   // create container for active double flux output
   std::vector<adouble> flux_a(q.Size());
   // Step 1: apply the EC slip wall flux
   mach::calcSlipWallFlux<adouble, dim>(
       x_a.data(), dir_a.data(), q_a.data(), flux_a.data());
   // Step 2: evaluate the adiabatic flux
   adouble mu_Re = mu;
   if (mu < 0.0)
   {
      mu_Re = mach::calcSutherlandViscosity<adouble, dim>(q_a.data());
   }
   mu_Re /= Re;
   mach::calcAdiabaticWallFlux<adouble, dim>(
       dir_a.data(), mu_Re, Pr, q_a.data(), Dw_a.data(), work_vec_a.data());
   for (int i = 0; i < flux_a.size(); ++i)
   {
      flux_a[i] -= work_vec_a[i];  // note the minus sign!!!
   }
   // Step 3: evaluate the no-slip penalty
   mach::calcNoSlipPenaltyFlux<adouble, dim>(dir_a.data(),
                                             jac,
                                             mu_Re,
                                             Pr,
                                             qfs_a.data(),
                                             q_a.data(),
                                             work_vec_a.data());
   for (int i = 0; i < flux_a.size(); ++i)
   {
      flux_a[i] += work_vec_a[i];
   }
   adouble fun_a = dot<adouble, dim>(force_nrm_a.data(), flux_a.data() + 1);
   fun_a.set_gradient(1.0);
   this->stack.compute_adjoint();
   mfem::Vector work(Dw_size);
   adept::get_gradients(Dw_a.data(), Dw_size, work.GetData());
   for (int d = 0; d < dim; ++d)
   {
      for (int s = 0; s < dim + 2; ++s)
      {
         flux_mat(s, d) = work(d * (dim + 2) + s);
      }
   }
}

template <int dim>
double SurfaceForce<dim>::GetFaceEnergy(const mfem::FiniteElement &el_bnd,
                                        const mfem::FiniteElement &el_unused,
                                        mfem::FaceElementTransformations &trans,
                                        const mfem::Vector &elfun)
{
   using namespace mfem;
   const SBPFiniteElement &sbp = dynamic_cast<const SBPFiniteElement &>(el_bnd);
   const int num_nodes = sbp.GetDof();
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

   const FiniteElement *sbp_face;
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
         convertVars(uj, wj);
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
      fun += calcBndryFun(x, nrm, jac_i, u_face, Dwi) * face_ip.weight * alpha;
   }  // k/i loop
   return fun;
}

template <int dim>
void SurfaceForce<dim>::AssembleFaceVector(
    const mfem::FiniteElement &el_bnd,
    const mfem::FiniteElement &el_unused,
    mfem::FaceElementTransformations &trans,
    const mfem::Vector &elfun,
    mfem::Vector &elvect)
{
   using namespace mfem;
   const SBPFiniteElement &sbp = dynamic_cast<const SBPFiniteElement &>(el_bnd);
   const int num_nodes = el_bnd.GetDof();
#ifdef MFEM_THREAD_SAFE
   Vector u_face, uj, wj, x, nrm, flux_face;
   DenseMatrix adjJ_i, adjJ_j, dwduj, Dwi, flux_mat;
#endif
   u_face.SetSize(num_states);
   uj.SetSize(num_states);
   wj.SetSize(num_states);
   x.SetSize(dim);
   nrm.SetSize(dim);
   flux_face.SetSize(num_states);
   adjJ_i.SetSize(dim);
   adjJ_j.SetSize(dim);
   dwduj.SetSize(num_states);
   Dwi.SetSize(num_states, dim);
   flux_mat.SetSize(num_states, dim);

   elvect.SetSize(num_states * num_nodes);
   elvect = 0.0;

   DenseMatrix u(elfun.GetData(), num_nodes, num_states);
   DenseMatrix res(elvect.GetData(), num_nodes, num_states);

   const FiniteElement *sbp_face;
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
         convertVars(uj, wj);
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
      calcBndryFunJacState(x, nrm, jac_i, u_face, Dwi, flux_face);
      flux_face *= face_ip.weight;

      // multiply by test function
      for (int s = 0; s < num_states; ++s)
      {
         res(i, s) += alpha * flux_face(s);
      }

      // get the flux terms that are scaled by the test-function derivative
      calcBndryFunJacDw(x, nrm, jac_i, u_face, Dwi, flux_mat);
      flux_mat *= face_ip.weight;

      // multiply flux_Dvi by the test-function derivative
      flux_mat *= Hinv;
      for (int j = 0; j < num_nodes; ++j)
      {
         // Get mapping Jacobian adjugate
         trans.Elem1->SetIntPoint(&el_bnd.GetNodes().IntPoint(j));
         CalcAdjugate(trans.Elem1->Jacobian(), adjJ_j);
         u.GetRow(j, uj);
         convertVarsJacState(uj, dwduj);
         for (int d = 0; d < dim; ++d)
         {
            double Qij = sbp.getQEntry(d, i, j, adjJ_i, adjJ_j);
            for (int s1 = 0; s1 < num_states; ++s1)
            {
               for (int s2 = 0; s2 < num_states; ++s2)
               {
                  // Dwi(s,d) += Qij * wj(s);
                  res(j, s2) += dwduj(s1, s2) * Qij * flux_mat(s1, d);
               }
            }
         }  // d loop
      }     // j loop
   }        // k/i loop
}