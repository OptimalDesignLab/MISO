#include <random>

#include "mfem.hpp"

#include "coefficient.hpp"
#include "mach_input.hpp"
#include "mfem_common_integ.hpp"
#include "current_load.hpp"

using namespace mfem;

namespace mach
{

/// set inputs should include fields, so things can check if they're "dirty"
void setInputs(CurrentLoad &load,
               const MachInputs &inputs)
{
   setInputs(load.div_free_proj, inputs);

   auto it = inputs.find("current_density");
   if (it != inputs.end())
   {
      load.current_density = it->second.getValue();
      load.current.SetAConst(load.current_density);
      load.dirty = true;
   }

   it = inputs.find("mesh_coords");
   if (it != inputs.end())
   {
      load.dirty = true;
   }
}

void addLoad(CurrentLoad &load,
             Vector &tv)
{
   if (load.dirty)
   {
      load.assembleLoad();
      load.dirty = false;
   }
   subtract(tv, load.load, tv);
}

double vectorJacobianProduct(CurrentLoad &load,
                             const mfem::HypreParVector &load_bar,
                             std::string wrt)
{
   if (wrt == "current_density")
   {
      load.current.SetAConst(1.0);
      load.nd_mass.Update();
      load.assembleLoad();
      load.current.SetAConst(load.current_density);
      load.dirty = true;

      return -(load.load * load_bar);
   }
   return 0.0;
}

void vectorJacobianProduct(CurrentLoad &load,
                           const mfem::HypreParVector &load_bar,
                           std::string wrt,
                           mfem::HypreParVector &wrt_bar)
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

      ParGridFunction psi_k(&load.fes); psi_k = 0.0;
      load.nd_mass.MultTranspose(psi_l, psi_k);
      // psi_k = load_bar;

      ParGridFunction rhs(&load.fes); rhs = 0.0;
      // load.nd_mass.MultTranspose(psi_l, rhs);
      // rhs = load_bar;
      load.div_free_proj.vectorJacobianProduct(load.j, psi_k, "in", rhs);
      rhs *= -1.0;

      ParGridFunction psi_j(&load.fes); psi_j = 0.0;
      // psi_j = load_bar;

      HypreParMatrix M;
      Vector X, RHS;
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
      auto &x_nodes = dynamic_cast<ParGridFunction&>(*mesh.GetNodes());
      auto &mesh_fes = *x_nodes.ParFESpace();
      HypreParVector scratch_tv(&mesh_fes); scratch_tv = 0.0;

      load.mesh_sens.Assemble();
      load.mesh_sens.ParallelAssemble(scratch_tv);
      wrt_bar -= scratch_tv;

      ParGridFunction scratch(&mesh_fes); scratch = 0.0;
      load.div_free_proj.vectorJacobianProduct(load.j, psi_k, wrt, scratch);
      scratch.ParallelAssemble(scratch_tv);
      wrt_bar -= scratch_tv;
   }
}

CurrentLoad::CurrentLoad(ParFiniteElementSpace &pfes,
                         VectorCoefficient &current_coeff)
   : current_density(1.0), current(1.0, current_coeff),
   fes(pfes), h1_coll(fes.GetFE(0)->GetOrder(), fes.GetMesh()->Dimension()),
   h1_fes(fes.GetParMesh(), &h1_coll),
   rt_coll(fes.GetFE(0)->GetOrder(), fes.GetMesh()->Dimension()), 
   rt_fes(fes.GetParMesh(), &rt_coll), nd_mass(&fes), J(&fes), j(&fes),
   div_free_current_vec(&fes), scratch(&fes), load(&fes),
   div_free_proj(h1_fes, fes, h1_fes.GetElementTransformation(0)->OrderW()
                                 + 2 * fes.GetFE(0)->GetOrder()), dirty(true)
{
   /// Create a H(curl) mass matrix for integrating grid functions
   nd_mass.AddDomainIntegrator(new VectorFEMassIntegrator);

   J.AddDomainIntegrator(new VectorFEDomainLFIntegrator(current));

   auto &mesh = *h1_fes.GetParMesh();
   auto &x_nodes = dynamic_cast<ParGridFunction&>(*mesh.GetNodes());
   auto &mesh_fes = *x_nodes.ParFESpace();
   mesh_sens.Update(&mesh_fes);

   m_j_mesh_sens = new VectorFEMassIntegratorMeshSens(1.0);
   mesh_sens.AddDomainIntegrator(m_j_mesh_sens);
   J_mesh_sens = new VectorFEDomainLFIntegratorMeshSens(current, -1.0);
   mesh_sens.AddDomainIntegrator(J_mesh_sens);
   m_l_mesh_sens = new VectorFEMassIntegratorMeshSens(1.0);
   mesh_sens.AddDomainIntegrator(m_l_mesh_sens);

   std::default_random_engine generator; generator.seed(1);
   std::uniform_real_distribution<double> distribution(-1.0,1.0);
   for (int i = 0; i < div_free_current_vec.Size(); ++i)
   {
      auto val = distribution(generator);
      // std::cout << "val: " << val << "i: " << i << "\n";
      // div_free_current_vec(i) = val;
      // dynamic_cast<Vector&>(J)(i) = val;
   }
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
   // j.ProjectCoefficient(current);

   HypreParMatrix M;
   Vector X, RHS;
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
   /** sensitivity debugging */
   // j.ParallelAssemble(load);
   // J.ParallelAssemble(load);

   /// Compute the discretely divergence-free portion of j
   div_free_current_vec = 0.0;
   div_free_proj.Mult(j, div_free_current_vec);
   // div_free_current_vec = j;

   // /** sensitivity debugging */
   // div_free_current_vec.ParallelAssemble(load);

   // /// Compute the dual of div_free_current_vec
   // // div_free_current_vec.ParallelAssemble(scratch);
   // // M.Mult(scratch, load);
   
   /// Compute the dual of div_free_current_vec
   nd_mass.Update();
   nd_mass.Assemble();
   nd_mass.Finalize();
   nd_mass.Mult(div_free_current_vec, scratch);
   scratch.ParallelAssemble(load);
}

} // namespace mach
