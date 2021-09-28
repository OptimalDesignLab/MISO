template <int dim, bool entvar>
void IsmailRoeMeshSensIntegrator<dim, entvar>::calcFlux(int di,
                                                        const mfem::Vector &qL,
                                                        const mfem::Vector &qR,
                                                        mfem::Vector &flux)
{
   if (entvar)
   {
      calcIsmailRoeFluxUsingEntVars<double, dim>(
          di, qL.GetData(), qR.GetData(), flux.GetData());
   }
   else
   {
      calcIsmailRoeFlux<double, dim>(
          di, qL.GetData(), qR.GetData(), flux.GetData());
   }
}

template <int dim, bool entvar>
void SlipWallBCMeshSens<dim, entvar>::calcFluxBar(const mfem::Vector &x,
                                                  const mfem::Vector &dir,
                                                  const mfem::Vector &u,
                                                  const mfem::Vector &flux_bar,
                                                  mfem::Vector &dir_bar)
{
   // create containers for active double objects for each input
   std::vector<adouble> x_a(x.Size());
   std::vector<adouble> dir_a(dir.Size());
   std::vector<adouble> u_a(u.Size());
   // initialize active double containers with data from inputs
   adept::set_values(x_a.data(), x.Size(), x.GetData());
   adept::set_values(dir_a.data(), dir.Size(), dir.GetData());
   adept::set_values(u_a.data(), u.Size(), u.GetData());
   // start new stack recording
   this->stack.new_recording();
   // create container for active double flux output
   std::vector<adouble> flux_a(u.Size());
   mach::calcSlipWallFlux<adouble, dim, entvar>(
       x_a.data(), dir_a.data(), u_a.data(), flux_a.data());
   adept::set_gradients(flux_a.data(), flux_bar.Size(), flux_bar.GetData());
   this->stack.compute_adjoint();
   adept::get_gradients(dir_a.data(), dir.Size(), dir_bar.GetData());
}