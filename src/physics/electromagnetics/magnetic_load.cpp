#include "mfem.hpp"

#include "coefficient.hpp"
#include "mach_input.hpp"
#include "mach_linearform.hpp"
#include "mfem_common_integ.hpp"
#include "magnetic_load.hpp"

namespace mach
{
MagneticLoad::MagneticLoad(adept::Stack &diff_stack,
                           mfem::ParFiniteElementSpace &fes,
                           std::map<std::string, FiniteElementState> &fields,
                           const nlohmann::json &options,
                           const nlohmann::json &materials,
                           mfem::Coefficient &nu)
 : lf(fes, fields),
   mag_coeff(std::make_unique<MagnetizationCoefficient>(diff_stack,
                                                        options["magnets"],
                                                        materials)),
   nuM(std::make_unique<mfem::ScalarVectorProductCoefficient>(nu, *mag_coeff))
{
   // auto &mesh_gf =
   //     dynamic_cast<mfem::ParGridFunction &>(*fes.GetMesh()->GetNodes());
   // auto *mesh_fes = mesh_gf.ParFESpace();
   // mag_load_fields.emplace(std::piecewise_construct,
   //                         std::make_tuple("mesh_coords"),
   //                         std::forward_as_tuple(mesh_fes,
   //                         mesh_gf.GetData()));

   lf.addDomainIntegrator(new mach::VectorFEDomainLFCurlIntegrator(*nuM, -1.0));
   /// only needed if magnets are on the boundary and not normal to boundary
   // lf.addBdrFaceIntegrator(new mach::VectorFEBoundaryTangentLFIntegrator(nuM,
   // -1.0));
}

/// set inputs should include fields, so things can check it they're "dirty"
void setInputs(LegacyMagneticLoad &load, const MachInputs &inputs)
{
   auto it = inputs.find("mesh_coords");
   if (it != inputs.end())
   {
      load.dirty = true;
   }
}

void addLoad(LegacyMagneticLoad &load, mfem::Vector &tv)
{
   if (load.dirty)
   {
      load.assembleLoad();
      load.dirty = false;
   }
   add(tv, -1.0, load.load, tv);
}

double vectorJacobianProduct(LegacyMagneticLoad &load,
                             const mfem::Vector &res_bar,
                             const std::string &wrt)
{
   return 0.0;
}

void vectorJacobianProduct(LegacyMagneticLoad &load,
                           const mfem::Vector &res_bar,
                           const std::string &wrt,
                           mfem::Vector &wrt_bar)
{
   throw std::logic_error(
       "vectorJacobianProduct not implemented for LegacyMagneticLoad!\n");
}

LegacyMagneticLoad::LegacyMagneticLoad(mfem::ParFiniteElementSpace &pfes,
                                       mfem::VectorCoefficient &mag_coeff,
                                       mfem::Coefficient &nu)
 : fes(pfes),
   rt_coll(fes.GetFE(0)->GetOrder(), fes.GetMesh()->Dimension()),
   rt_fes(fes.GetParMesh(), &rt_coll),
   mag_coeff(mag_coeff),
   load(fes.GetTrueVSize()),
   weakCurlMuInv(&rt_fes, &fes),
   M(&rt_fes),
   scratch(&fes),
   dirty(true)
{
   /// Create a H(curl) mass matrix for integrating grid functions
   weakCurlMuInv.AddDomainIntegrator(new mfem::VectorFECurlIntegrator(nu));
}

void LegacyMagneticLoad::assembleLoad()
{
   weakCurlMuInv.Update();
   weakCurlMuInv.Assemble();
   weakCurlMuInv.Finalize();

   M.ProjectCoefficient(mag_coeff);
   weakCurlMuInv.Mult(M, scratch);
   scratch.ParallelAssemble(load);
}

}  // namespace mach
