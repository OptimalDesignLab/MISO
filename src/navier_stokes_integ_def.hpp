
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
   calcNoSlipPenaltyFlux<double, dim>(dir, jac, mu_Re, Pr, qfs.GetData(),
                                      q.GetData(), work_vec.GetData());
   flux_vec += work_vec;
}

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
   // Part 2: supply the viscous flux (based on numercial solution)
   double mu_Re = mu;
   if (mu < 0.0)
   {
      mu_Re = calcSutherlandViscosity<double, dim>(q.GetData());
   }
   mu_Re /= Re;
#if 0
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
   // Part 2: evaluate the adiabatic flux
   double mu_Re = mu;
   if (mu < 0.0)
   {
      mu_Re = calcSutherlandViscosity<double, dim>(q.GetData());
   }
   mu_Re /= Re;
#if 0
   calcAdiabaticWallFlux<double, dim>(dir.GetData(), mu_Re, Pr, q.GetData(),
                                      Dw.GetData(), work_vec.GetData());
   flux_vec -= work_vec; // note the minus sign!!!
#endif
#if 1
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
void ViscousOutflowBC<dim>::calcFlux(const mfem::Vector &x,
                                     const mfem::Vector &dir, double jac,
                                     const mfem::Vector &q,
                                     const mfem::DenseMatrix &Dw,
                                     mfem::Vector &flux_vec)
{
   // Part 1: apply the inviscid inflow boundary condition
   calcBoundaryFlux<double, dim>(dir.GetData(), q_out.GetData(), q.GetData(),
                                 work_vec.GetData(), flux_vec.GetData());
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
   //flux_vec -= work_vec; // note the minus sign!!!
}