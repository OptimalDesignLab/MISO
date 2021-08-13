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
   auto it = inputs.find("mesh_coords");
   if (it != inputs.end())
   {
      load.dirty = true;
   }
}

void addLoad(MagneticLoad &load,
             Vector &tv)
{
   if (load.dirty)
   {
      load.assembleLoad();
      load.dirty = false;
   }
   add(tv, -1.0, load.load, tv);
}

double vectorJacobianProduct(MagneticLoad &load,
                             const mfem::HypreParVector &res_bar,
                             std::string wrt)
{
   return 0.0;
}

void vectorJacobianProduct(MagneticLoad &load,
                           const mfem::HypreParVector &res_bar,
                           std::string wrt,
                           mfem::HypreParVector &wrt_bar)
{
   throw std::logic_error("vectorJacobianProduct not implemented for MagneticLoad!\n");
}

MagneticLoad::MagneticLoad(ParFiniteElementSpace &pfes,
                           VectorCoefficient &mag_coeff,
                           Coefficient &nu)
   : fes(pfes), rt_coll(fes.GetFE(0)->GetOrder(), fes.GetMesh()->Dimension()),
   rt_fes(fes.GetParMesh(), &rt_coll), mag_coeff(mag_coeff),
   load(&fes), weakCurlMuInv(&rt_fes, &fes), M(&rt_fes), scratch(&fes),
   dirty(true)
{
   /// Create a H(curl) mass matrix for integrating grid functions
   weakCurlMuInv.AddDomainIntegrator(new VectorFECurlIntegrator(nu));
}

void MagneticLoad::assembleLoad()
{
   weakCurlMuInv.Update();
   weakCurlMuInv.Assemble();
   weakCurlMuInv.Finalize();

   M.ProjectCoefficient(mag_coeff);
   weakCurlMuInv.Mult(M, scratch);
   scratch.ParallelAssemble(load);
}

} // namespace mach
