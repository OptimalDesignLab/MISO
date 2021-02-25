#include "mfem.hpp"

#include "coefficient.hpp"
#include "mach_input.hpp"
#include "magnetic_load.hpp"

using namespace mfem;

namespace mach
{

/// set inputs should include fields, so things can check it they're "dirty"
void setInputs(MagneticLoad &load,
               const MachInputs &inputs)
{
   load.weakCurlMuInv.Update();
   load.assembleLoad();
}

void assemble(MagneticLoad &load,
              HypreParVector &tv)
{
   add(tv, load.load, tv);
}

MagneticLoad::MagneticLoad(ParFiniteElementSpace &pfes,
                           VectorCoefficient &mag_coeff,
                           Coefficient &nu)
   : fes(pfes), rt_coll(fes.GetFE(0)->GetOrder(), fes.GetMesh()->Dimension()),
   rt_fes(fes.GetParMesh(), &rt_coll), mag_coeff(mag_coeff),
   load(&fes), weakCurlMuInv(&rt_fes, &fes), M(&rt_fes), scratch(&fes)
{
   /// Create a H(curl) mass matrix for integrating grid functions
   weakCurlMuInv.AddDomainIntegrator(new VectorFECurlIntegrator(nu));
}

void MagneticLoad::assembleLoad()
{
   weakCurlMuInv.Assemble();
   weakCurlMuInv.Finalize();

   M.ProjectCoefficient(mag_coeff);
   weakCurlMuInv.Mult(M, scratch);
   scratch.ParallelAssemble(load);
}

} // namespace mach
