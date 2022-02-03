#include <memory>

#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "coefficient.hpp"
#include "current_source_functions.hpp"
#include "mach_input.hpp"
#include "mfem_common_integ.hpp"

#include "current_load.hpp"

using namespace mfem;

namespace mach
{
/// set inputs should include fields, so things can check if they're "dirty"
void setInputs(CurrentLoad &load, const MachInputs &inputs)
{
   setInputs(load.div_free_proj, inputs);
   auto dirty_coeff = setInputs(load.current, inputs);
   load.dirty = load.dirty || dirty_coeff;

   auto it = inputs.find("mesh_coords");
   if (it != inputs.end())
   {
      load.dirty = true;
   }
}

void setOptions(CurrentLoad &load, const nlohmann::json &options)
{
   if (options.contains("ess-bdr"))
   {
      auto fes = load.fes;
      mfem::Array<int> ess_bdr(fes.GetParMesh()->bdr_attributes.Max());
      getEssentialBoundaries(options, ess_bdr);
      fes.GetEssentialTrueDofs(ess_bdr, load.ess_tdof_list);
   }
}

void addLoad(CurrentLoad &load, Vector &tv)
{
   if (load.dirty)
   {
      load.assembleLoad();
      load.dirty = false;
   }
   load.load.SetSubVector(load.ess_tdof_list, 0.0);
   subtract(tv, load.load, tv);
}

double vectorJacobianProduct(CurrentLoad &load,
                             const mfem::Vector &load_bar,
                             const std::string &wrt)
{
   if (wrt == "current_density")
   {
      // load.current.SetAConst(1.0);
      // load.nd_mass.Update();
      // load.assembleLoad();
      // load.current.SetAConst(load.current_density);
      // load.dirty = true;

      // return -(load.load * load_bar);
      return NAN;
   }
   return 0.0;
}

void vectorJacobianProduct(CurrentLoad &load,
                           const mfem::Vector &load_bar,
                           const std::string &wrt,
                           mfem::Vector &wrt_bar)
{
   if (wrt == "mesh_coords")
   {
      if (load.dirty)
      {
         load.assembleLoad();
         load.dirty = false;
      }

      /// begin reverse pass
      ParGridFunction psi_l(&load.fes);
      psi_l = load_bar;

      load.nd_mass.Update();
      load.nd_mass.Assemble();
      load.nd_mass.Finalize();

      ParGridFunction psi_k(&load.fes);
      psi_k = 0.0;
      load.nd_mass.MultTranspose(psi_l, psi_k);

      ParGridFunction rhs(&load.fes);
      rhs = 0.0;
      load.div_free_proj.vectorJacobianProduct(load.j, psi_k, "in", rhs);
      rhs *= -1.0;

      ParGridFunction psi_j(&load.fes);
      psi_j = 0.0;
      // psi_j = load_bar;

      HypreParMatrix M;
      Vector X;
      Vector RHS;
      Array<int> ess_tdof_list;
      load.nd_mass.FormLinearSystem(ess_tdof_list, psi_j, rhs, M, X, RHS);
      auto M_matT = std::unique_ptr<HypreParMatrix>(M.Transpose());
      HypreBoomerAMG amg(*M_matT);
      amg.SetPrintLevel(-1);

      HyprePCG pcg(*M_matT);
      pcg.SetTol(1e-12);
      pcg.SetMaxIter(500);
      pcg.SetPrintLevel(2);
      pcg.SetPreconditioner(amg);
      pcg.Mult(RHS, X);

      load.nd_mass.RecoverFEMSolution(X, rhs, psi_j);

      load.m_j_mesh_sens->setState(load.j);
      load.m_j_mesh_sens->setAdjoint(psi_j);
      load.J_mesh_sens->setAdjoint(psi_j);

      load.m_l_mesh_sens->setState(load.div_free_current_vec);
      load.m_l_mesh_sens->setAdjoint(psi_l);

      auto &mesh = *load.h1_fes.GetParMesh();
      auto &x_nodes = dynamic_cast<ParGridFunction &>(*mesh.GetNodes());
      auto &mesh_fes = *x_nodes.ParFESpace();
      HypreParVector scratch_tv(&mesh_fes);
      scratch_tv = 0.0;

      load.mesh_sens.Assemble();
      load.mesh_sens.ParallelAssemble(scratch_tv);
      wrt_bar -= scratch_tv;

      ParGridFunction scratch(&mesh_fes);
      scratch = 0.0;
      load.div_free_proj.vectorJacobianProduct(load.j, psi_k, wrt, scratch);
      scratch.ParallelAssemble(scratch_tv);
      wrt_bar -= scratch_tv;
   }
}

CurrentLoad::CurrentLoad(adept::Stack &diff_stack,
                         mfem::ParFiniteElementSpace &fes,
                         std::map<std::string, FiniteElementState> &fields,
                         const nlohmann::json &options)
 : current(diff_stack, options["current"]),
   fes(fes),
   h1_coll(fes.GetFE(0)->GetOrder(), fes.GetMesh()->Dimension()),
   h1_fes(fes.GetParMesh(), &h1_coll),
   rt_coll(fes.GetFE(0)->GetOrder(), fes.GetMesh()->Dimension()),
   rt_fes(fes.GetParMesh(), &rt_coll),
   nd_mass(&fes),
   J(&fes),
   j(&fes),
   div_free_current_vec(&fes),
   scratch(&fes),
   load(fes.GetTrueVSize()),
   div_free_proj(h1_fes,
                 fes,
                 h1_fes.GetElementTransformation(0)->OrderW() +
                     2 * fes.GetFE(0)->GetOrder()),
   m_j_mesh_sens(new VectorFEMassIntegratorMeshSens),
   J_mesh_sens(new VectorFEDomainLFIntegratorMeshSens(current, -1.0)),
   m_l_mesh_sens(new VectorFEMassIntegratorMeshSens),
   dirty(true)
{
   /// Create a H(curl) mass matrix for integrating grid functions
   nd_mass.AddDomainIntegrator(new VectorFEMassIntegrator);

   J.AddDomainIntegrator(new VectorFEDomainLFIntegrator(current));

   auto &mesh = *h1_fes.GetParMesh();
   auto &x_nodes = dynamic_cast<ParGridFunction &>(*mesh.GetNodes());
   auto &mesh_fes = *x_nodes.ParFESpace();
   mesh_sens.Update(&mesh_fes);

   mesh_sens.AddDomainIntegrator(m_j_mesh_sens);

   mesh_sens.AddDomainIntegrator(J_mesh_sens);

   mesh_sens.AddDomainIntegrator(m_l_mesh_sens);

   setOptions(*this, options);
}

void CurrentLoad::assembleLoad()
{
   /// assemble mass matrix
   nd_mass.Update();
   nd_mass.Assemble();
   nd_mass.Finalize();

   /// assemble linear form
   J.Assemble();

   /// project current coeff as initial guess for iterative solve
   j.ProjectCoefficient(current);
   mfem::ParaViewDataCollection pv("CurrentDensity",
                                   j.ParFESpace()->GetParMesh());
   pv.SetPrefixPath("ParaView");
   pv.SetLevelsOfDetail(3);
   pv.SetDataFormat(mfem::VTKFormat::ASCII);
   pv.SetHighOrderOutput(true);
   pv.RegisterField("CurrentDensity", &j);
   pv.Save();

   HypreParMatrix M;
   Vector X;
   Vector RHS;
   Array<int> ess_tdof_list;
   nd_mass.FormLinearSystem(ess_tdof_list, j, J, M, X, RHS);

   HypreBoomerAMG amg(M);
   amg.SetPrintLevel(-1);

   HyprePCG pcg(M);
   pcg.SetTol(1e-12);
   pcg.SetMaxIter(500);
   pcg.SetPrintLevel(2);
   pcg.SetPreconditioner(amg);
   pcg.Mult(RHS, X);

   nd_mass.RecoverFEMSolution(X, J, j);

   /// Compute the discretely divergence-free portion of j
   div_free_current_vec = 0.0;
   div_free_proj.Mult(j, div_free_current_vec);

   /** alternative approaches for computing dual */
   /// Compute the dual of div_free_current_vec
   // div_free_current_vec.ParallelAssemble(scratch);
   // M.Mult(scratch, load);

   /// Compute the dual of div_free_current_vec
   nd_mass.Update();
   nd_mass.Assemble();
   nd_mass.Finalize();
   nd_mass.Mult(div_free_current_vec, scratch);
   scratch.ParallelAssemble(load);
}

}  // namespace mach
