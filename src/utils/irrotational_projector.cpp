#include "irrotational_projector.hpp"

using namespace mfem;

namespace mach
{

IrrotationalProjector::IrrotationalProjector(ParFiniteElementSpace &h1_fes,
                                             ParFiniteElementSpace &nd_fes,
                                             const int &ir_order)
   : h1_fes(h1_fes), nd_fes(nd_fes), diffusion(&h1_fes),
   weak_div(&nd_fes, &h1_fes), grad(&h1_fes, &nd_fes), psi(&h1_fes),
   div_x(&h1_fes), pcg(h1_fes.GetComm()), dirty(true)
{
   /// not sure if theres a better way to handle this
   ess_bdr.SetSize(h1_fes.GetParMesh()->bdr_attributes.Max());
   ess_bdr = 1;
   h1_fes.GetEssentialTrueDofs(ess_bdr, ess_bdr_tdofs);

   int geom = h1_fes.GetFE(0)->GetGeomType();
   const IntegrationRule * ir = &IntRules.Get(geom, ir_order);

   BilinearFormIntegrator *diffInteg = new DiffusionIntegrator;
   diffInteg->SetIntRule(ir);
   diffusion.AddDomainIntegrator(diffInteg);

   BilinearFormIntegrator *wdivInteg = new VectorFEWeakDivergenceIntegrator;
   wdivInteg->SetIntRule(ir);
   weak_div.AddDomainIntegrator(wdivInteg);
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
   weak_div.Mult(x, div_x); div_x *= -1.0;

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

void IrrotationalProjector::vectorJacobianProduct(
   const mfem::ParGridFunction &out_bar,
   std::string wrt,
   mfem::ParGridFunction &wrt_bar)
{
   if (wrt == "in")
   {
      if (dirty)
      {
         update();
         dirty = false;
      }

      ParGridFunction GTout_bar(&h1_fes);
      grad.MultTranspose(out_bar, GTout_bar);
      GTout_bar *= -1.0;

      // Apply essential BC and form linear system
      ParGridFunction psi_bar(psi); psi_bar = 0.0;
      // auto D_mat = std::unique_ptr<HypreParMatrix>(new HypreParMatrix);
      HypreParMatrix D_mat;
      diffusion.FormLinearSystem(ess_bdr_tdofs, psi_bar, GTout_bar, D_mat, Psi, RHS);
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
      diffusion.RecoverFEMSolution(Psi, GTout_bar, psi_bar);

      weak_div.AddMultTranspose(psi_bar, wrt_bar);
   }
   else if (wrt == "mesh_coords")
   {

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

} // namespace mach
