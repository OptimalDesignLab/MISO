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

}  // namespace mach
