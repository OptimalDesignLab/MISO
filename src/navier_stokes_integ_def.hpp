
template <int dim>
void NoSlipAdiabaticWallBC<dim>::calcFlux(const mfem::Vector &x,
                                          const mfem::Vector &dir, double jac,
                                          const mfem::Vector &q,
                                          const mfem::DenseMatrix &Dw,
                                          mfem::Vector &flux_vec)
{
   // Step 1: apply the EC slip wall flux
   calcSlipWallFlux<double, dim>(x.GetData(), dir.GetData(), q.GetData(),
                                 flux_vec.GetData());
   // Step 2: evaluate the adiabatic flux
   double mu_Re = mu;
   if (mu < 0.0)
   {
      mu_Re = calcSutherlandViscosity<double, dim>(q.GetData());
   }
   mu_Re /= Re;
   calcAdiabaticWallFlux<double, dim>(dir.GetData(), mu_Re, Pr, q.GetData(),
                                      Dw.GetData(), work_vec.GetData());
   flux_vec -= work_vec; // note the minus sign!!!
   // Step 3: evaluate the no-slip penalty
   calcNoSlipPenaltyFlux<double, dim>(dir.GetData(), jac, mu_Re, Pr, qfs.GetData(),
                                      q.GetData(), work_vec.GetData());
   flux_vec += work_vec;
}
//To do: do we need `jac` here ?
template <int dim>
void ViscousSlipWallBC<dim>::calcFlux(const mfem::Vector &x,
                                      const mfem::Vector &dir, double jac,
                                      const mfem::Vector &q,
                                      const mfem::DenseMatrix &Dw,
                                      mfem::Vector &flux_vec)
{
   // Part 1: apply the inviscid slip wall BCs
   calcSlipWallFlux<double, dim>(x.GetData(), dir.GetData(), q.GetData(),
                                 flux_vec.GetData());
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
#if 0
   for (int d = 0; d < dim; ++d)
   {
      work_vec = 0.0;
      applyViscousScaling<double, dim>(d, mu_Re, Pr, q.GetData(), Dw.GetData(),
                                       work_vec.GetData());
      work_vec *= dir(d);
      flux_vec -= work_vec;
   }
#endif
}
//To do: do we need `jac` here ?
template <int dim>
void ViscousInflowBC<dim>::calcFlux(const mfem::Vector &x,
                                    const mfem::Vector &dir, double jac,
                                    const mfem::Vector &q,
                                    const mfem::DenseMatrix &Dw,
                                    mfem::Vector &flux_vec)
{
   // Part 1: apply the inviscid inflow boundary condition
   calcBoundaryFlux<double, dim>(dir.GetData(), q_in.GetData(), q.GetData(),
                                 work_vec.GetData(), flux_vec.GetData());
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
//To do: do we need `jac` here ?
template <int dim>
void ViscousOutflowBC<dim>::calcFlux(const mfem::Vector &x,
                                     const mfem::Vector &dir, double jac,
                                     const mfem::Vector &q,
                                     const mfem::DenseMatrix &Dw,
                                     mfem::Vector &flux_vec)
{
   // Part 1: apply the inviscid inflow boundary condition
   calcBoundaryFlux<double, dim>(dir.GetData(), q_out.GetData(), q.GetData(),
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
      flux_vec -= work_vec; // note the minus sign!!!
   }
#endif
}

template <int dim>
void ESViscousIntegrator<dim>::convertVarsJacState(const mfem::Vector &q,
                                                   mfem::DenseMatrix &dwdu)
{
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
}

template <int dim>
void ESViscousIntegrator<dim>::applyScalingJacState(int d, const mfem::Vector &q,
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
   applyViscousScaling<adouble, dim>(d, mu_Re, Pr, q_a.data(), Dw_a.data(),
                                     CDw_a.data());
   //std::cout << "In applyScalingJacState " << std::endl;
   for (int j = 0; j < dim + 2; j++)
   {
      //std::cout <<  CDw_a[j] << std::endl;
   }
   // identify independent and dependent variables
   this->stack.independent(q_a.data(), q.Size());
   // test with only one independent variable
   //this->stack.independent(&q_a[0], 1);
   this->stack.dependent(CDw_a.data(), q.Size());
   // compute and store jacobian in CDw_jac
   this->stack.jacobian(CDw_jac.GetData());
}

template <int dim>
void ESViscousIntegrator<dim>::applyScalingJacV(int d, const mfem::Vector &q,
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
   applyViscousScaling<adouble, dim>(d, mu_Re, Pr, q_a.data(), Dw_a.data(),
                                     CDw_a.data());
   // identify independent and dependent variables
   this->stack.independent(Dw_a.data(), Dw_size);
   this->stack.dependent(CDw_a.data(), q.Size());
   // compute and store jacobian in CDw_jac
   this->stack.jacobian(CDw_jac.GetData());
}

template <int dim>
void NoSlipAdiabaticWallBC<dim>::calcFluxJacState(const mfem::Vector &x, const mfem::Vector &dir, double jac,
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
   mach::calcSlipWallFlux<adouble, dim>(x_a.data(), dir_a.data(), q_a.data(),
                                        flux_a.data());
   // Step 2: evaluate the adiabatic flux
   adouble mu_Re = mu;
   if (mu < 0.0)
   {
      mu_Re = mach::calcSutherlandViscosity<adouble, dim>(q_a.data());
   }
   mu_Re /= Re;
   mach::calcAdiabaticWallFlux<adouble, dim>(dir_a.data(), mu_Re, Pr, q_a.data(),
                                             Dw_a.data(), work_vec_a.data());
   for (int i = 0; i < flux_a.size(); ++i)
   {
      flux_a[i] -= work_vec_a[i]; // note the minus sign!!!
   }
   // Step 3: evaluate the no-slip penalty
   mach::calcNoSlipPenaltyFlux<adouble, dim>(dir_a.data(), jac, mu_Re, Pr, qfs_a.data(),
                                             q_a.data(), work_vec_a.data());

   for (int i = 0; i < flux_a.size(); ++i)
   {
      flux_a[i] += work_vec_a[i];
   }
   this->stack.independent(q_a.data(), q.Size());
   this->stack.dependent(flux_a.data(), q.Size());
   this->stack.jacobian(flux_jac.GetData());
}

template <int dim>
void NoSlipAdiabaticWallBC<dim>::calcFluxJacV(const mfem::Vector &x, const mfem::Vector &dir, double jac,
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
   mach::calcSlipWallFlux<adouble, dim>(x_a.data(), dir_a.data(), q_a.data(),
                                        flux_a.data());
   // Step 2: evaluate the adiabatic flux
   adouble mu_Re = mu;
   if (mu < 0.0)
   {
      mu_Re = mach::calcSutherlandViscosity<adouble, dim>(q_a.data());
   }
   mu_Re /= Re;
   mach::calcAdiabaticWallFlux<adouble, dim>(dir_a.data(), mu_Re, Pr, q_a.data(),
                                             Dw_a.data(), work_vec_a.data());
   for (int i = 0; i < flux_a.size(); ++i)
   {
      flux_a[i] -= work_vec_a[i]; // note the minus sign!!!
   }
   // Step 3: evaluate the no-slip penalty
   mach::calcNoSlipPenaltyFlux<adouble, dim>(dir_a.data(), jac, mu_Re, Pr, qfs_a.data(),
                                             q_a.data(), work_vec_a.data());

   for (int i = 0; i < flux_a.size(); ++i)
   {
      flux_a[i] += work_vec_a[i];
   }
   this->stack.independent(Dw_a.data(),Dw_size);
   this->stack.dependent(flux_a.data(), q.Size());
   this->stack.jacobian(flux_jac.GetData());
}

template <int dim>
void ViscousSlipWallBC<dim>::calcFluxJacState(const mfem::Vector &x, const mfem::Vector &dir,
                                              const mfem::Vector &q, const mfem::DenseMatrix &Dw,
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
   adept::set_values(Dw_a.data(),Dw_size, Dw.GetData());
   // start new stack recording
   this->stack.new_recording();
   // create container for active double flux output
   std::vector<adouble> flux_a(q.Size());
   mach::calcSlipWallFlux<adouble, dim>(x_a.data(), dir_a.data(), q_a.data(),
                                        flux_a.data());
   this->stack.independent(q_a.data(), q.Size());
   this->stack.dependent(flux_a.data(), q.Size());
   this->stack.jacobian(flux_jac.GetData());
}
template <int dim>
void ViscousInflowBC<dim>::calcFluxJacState(const mfem::Vector &x, const mfem::Vector &dir,
                                            const mfem::Vector &q, const mfem::DenseMatrix &Dw,
                                            mfem::DenseMatrix &flux_jac)
{
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
}

template <int dim>
void ViscousOutflowBC<dim>::calcFluxJacState(const mfem::Vector &x, const mfem::Vector &dir,
                                             const mfem::Vector &q, const mfem::DenseMatrix &Dw,
                                             mfem::DenseMatrix &flux_jac)
{
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
   mach::calcBoundaryFlux<adouble, dim>(dir_a.data(), q_out_a.data(), q_a.data(),
                                        work_vec_a.data(), flux_a.data());
   this->stack.independent(q_a.data(), q.Size());
   this->stack.dependent(flux_a.data(), q.Size());
   this->stack.jacobian(flux_jac.GetData());
}

template <int dim>
void ViscousFarFieldBC<dim>::calcFluxJacState(const mfem::Vector &dir,
                                              const mfem::Vector &q,
                                              mfem::DenseMatrix &flux_jac)
{
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
}
