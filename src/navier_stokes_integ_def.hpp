
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
    calcAdiabaticWallFlux<double, dim>(dir.GetData(), Re, Pr, q.GetData(),
                                       Dw.GetData(), work_vec.GetData());
    flux_vec += work_vec;
    // Step 3: evaluate the no-slip penalty
    calcNoSlipPenaltyFlux<double, dim>(dir, jac, Re, Pr, qfs.GetData(),
                                       q.GetData(), work_vec.GetData());
    flux_vec += work_vec;
}
