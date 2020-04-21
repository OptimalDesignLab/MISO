template <int dim, bool entvar>
void IsmailRoeMeshSensIntegrator<dim, entvar>::calcFlux(
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
}