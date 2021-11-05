#include "mfem.hpp"

#include "mfem_common_integ.hpp"
#include "irrotational_projector.hpp"

using namespace mfem;

namespace mach
{
IrrotationalProjector::IrrotationalProjector(ParFiniteElementSpace &h1_fes,
                                             ParFiniteElementSpace &nd_fes,
                                             const int &ir_order)
 : dirty(true),
   h1_fes(h1_fes),
   diffusion(&h1_fes),
   weak_div(&nd_fes, &h1_fes),
   grad(&h1_fes, &nd_fes),
   diff_mesh_sens(new DiffusionIntegratorMeshSens),
   div_mesh_sens(new VectorFEWeakDivergenceIntegratorMeshSens),
   psi(&h1_fes),
   div_x(&h1_fes),
   pcg(h1_fes.GetComm())
{
   /// not sure if theres a better way to handle this
   ess_bdr.SetSize(h1_fes.GetParMesh()->bdr_attributes.Max());
   ess_bdr = 1;
   h1_fes.GetEssentialTrueDofs(ess_bdr, ess_bdr_tdofs);

   int geom = h1_fes.GetFE(0)->GetGeomType();
   const IntegrationRule *ir = &IntRules.Get(geom, ir_order);

   BilinearFormIntegrator *diffInteg = new DiffusionIntegrator;
   diffInteg->SetIntRule(ir);
   diffusion.AddDomainIntegrator(diffInteg);

   BilinearFormIntegrator *wdivInteg = new VectorFEWeakDivergenceIntegrator;
   wdivInteg->SetIntRule(ir);
   weak_div.AddDomainIntegrator(wdivInteg);

   auto &mesh = *h1_fes.GetParMesh();
   auto &x_nodes = dynamic_cast<ParGridFunction &>(*mesh.GetNodes());
   auto &mesh_fes = *x_nodes.ParFESpace();
   mesh_sens.Update(&mesh_fes);

   mesh_sens.AddDomainIntegrator(diff_mesh_sens);

   mesh_sens.AddDomainIntegrator(div_mesh_sens);
}

void setInputs(IrrotationalProjector &op, const MachInputs &inputs)
{
   auto it = inputs.find("mesh_coords");
   if (it != inputs.end())
   {
      op.dirty = true;
   }
}

void IrrotationalProjector::Mult(const Vector &x, Vector &y) const
{
   if (dirty)
   {
      update();
      dirty = false;
   }

   // Compute the divergence of x
   weak_div.Mult(x, div_x);
   div_x *= -1.0;

   // Apply essential BC and form linear system
   psi = 0.0;
   HypreParMatrix D_mat;
   diffusion.FormLinearSystem(ess_bdr_tdofs, psi, div_x, D_mat, Psi, RHS);

   amg.SetOperator(D_mat);
   amg.SetPrintLevel(0);

   pcg.SetOperator(D_mat);
   pcg.SetTol(1e-14);
   pcg.SetMaxIter(200);
   pcg.SetPrintLevel(0);
   pcg.SetPreconditioner(amg);

   // Solve the linear system for Psi
   pcg.Mult(RHS, Psi);

   // Compute the parallel grid function correspoinding to Psi
   diffusion.RecoverFEMSolution(Psi, div_x, psi);

   // Compute the irrotational portion of x
   grad.Mult(psi, y);
}

void IrrotationalProjector::vectorJacobianProduct(const mfem::Vector &x,
                                                  const mfem::Vector &proj_bar,
                                                  const std::string &wrt,
                                                  mfem::Vector &wrt_bar)
{
   if (wrt == "in")
   {
      if (dirty)
      {
         update();
         dirty = false;
      }

      ParGridFunction GTproj_bar(&h1_fes);
      grad.MultTranspose(proj_bar, GTproj_bar);
      GTproj_bar *= -1.0;

      // Apply essential BC and form linear system
      ParGridFunction psi_bar(psi);
      psi_bar = 0.0;
      // auto D_mat = std::unique_ptr<HypreParMatrix>(new HypreParMatrix);
      HypreParMatrix D_mat;
      diffusion.FormLinearSystem(
          ess_bdr_tdofs, psi_bar, GTproj_bar, D_mat, Psi, RHS);
      auto D_matT = std::unique_ptr<HypreParMatrix>(D_mat.Transpose());
      amg.SetOperator(*D_matT);
      amg.SetPrintLevel(0);

      pcg.SetOperator(*D_matT);
      pcg.SetTol(1e-14);
      pcg.SetMaxIter(200);
      pcg.SetPrintLevel(0);
      pcg.SetPreconditioner(amg);

      // Solve the linear system for Psi
      pcg.Mult(RHS, Psi);

      // Compute the parallel grid function correspoinding to Psi
      diffusion.RecoverFEMSolution(Psi, GTproj_bar, psi_bar);

      weak_div.AddMultTranspose(psi_bar, wrt_bar);
   }
   else if (wrt == "mesh_coords")
   {
      if (dirty)
      {
         update();
         dirty = false;
      }
      // Compute the divergence of x
      weak_div.Mult(x, div_x);
      div_x *= -1.0;

      // Apply essential BC and form linear system
      psi = 0.0;
      HypreParMatrix D_mat;
      diffusion.FormLinearSystem(ess_bdr_tdofs, psi, div_x, D_mat, Psi, RHS);

      amg.SetOperator(D_mat);
      amg.SetPrintLevel(0);

      pcg.SetOperator(D_mat);
      pcg.SetTol(1e-14);
      pcg.SetMaxIter(200);
      pcg.SetPrintLevel(0);
      pcg.SetPreconditioner(amg);

      // Solve the linear system for Psi
      pcg.Mult(RHS, Psi);

      // Compute the parallel grid function correspoinding to Psi
      diffusion.RecoverFEMSolution(Psi, div_x, psi);

      /// start reverse pass
      ParGridFunction GTproj_bar(&h1_fes);
      grad.MultTranspose(proj_bar, GTproj_bar);
      GTproj_bar *= -1.0;

      // Apply essential BC and form linear system
      ParGridFunction psi_bar(psi);
      psi_bar = 0.0;
      diffusion.FormLinearSystem(
          ess_bdr_tdofs, psi_bar, GTproj_bar, D_mat, Psi, RHS);
      auto D_matT = std::unique_ptr<HypreParMatrix>(D_mat.Transpose());
      amg.SetOperator(*D_matT);
      amg.SetPrintLevel(0);

      pcg.SetOperator(*D_matT);
      pcg.SetTol(1e-14);
      pcg.SetMaxIter(200);
      pcg.SetPrintLevel(0);
      pcg.SetPreconditioner(amg);

      // Solve the linear system for Psi
      pcg.Mult(RHS, Psi);

      // Compute the parallel grid function correspoinding to Psi
      diffusion.RecoverFEMSolution(Psi, GTproj_bar, psi_bar);

      const auto &x_gf = dynamic_cast<const mfem::GridFunction &>(x);
      diff_mesh_sens->setState(psi);
      diff_mesh_sens->setAdjoint(psi_bar);
      div_mesh_sens->setState(x_gf);
      div_mesh_sens->setAdjoint(psi_bar);
      mesh_sens.Assemble();
      wrt_bar += mesh_sens;
   }
}

void IrrotationalProjector::update() const
{
   diffusion.Update();
   diffusion.Assemble();
   diffusion.Finalize();

   weak_div.Update();
   weak_div.Assemble();
   weak_div.Finalize();

   grad.Update();
   grad.Assemble();
   grad.Finalize();
}

}  // namespace mach
