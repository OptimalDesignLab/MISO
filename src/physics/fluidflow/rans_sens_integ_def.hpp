template <int dim, bool entvar>
void SAInviscidMeshSensIntegrator<dim, entvar>::calcFlux(
    int di, const mfem::Vector &qL, const mfem::Vector &qR, mfem::Vector &flux)
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