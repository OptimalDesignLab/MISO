#include "mfem.hpp"

#include "coefficient.hpp"
#include "mach_input.hpp"
#include "current_load.hpp"

using namespace mfem;

namespace mach
{

/// set inputs should include fields, so things can check if they're "dirty"
void setInputs(CurrentLoad &load,
               const MachInputs &inputs)
{
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
                             const mfem::HypreParVector &res_bar,
                             std::string wrt)
{
   if (wrt == "current_density")
   {
      load.current.SetAConst(1.0);
      load.nd_mass.Update();
      load.assembleLoad();
      load.current.SetAConst(load.current_density);
      load.dirty = true;

      return -(load.load * res_bar);
   }
   return 0.0;
}

void vectorJacobianProduct(CurrentLoad &load,
                           const mfem::HypreParVector &res_bar,
                           std::string wrt,
                           mfem::HypreParVector &wrt_bar)
{
   if (wrt == "mesh_coords")
   {
      ParGridFunction psi_a(&load.fes); psi_a = res_bar;
      ParGridFunction psi_L(psi_a); psi_L *= -1.0;
      ParGridFunction psi_k(&load.fes);

      ParBilinearForm M(&load.fes);
      M.AddDomainIntegrator(new VectorFEMassIntegrator);
      M.Assemble();
      M.Finalize();
      M.MultTranspose(psi_L, psi_k);

      ParGridFunction psi_y(psi_k); psi_y *= -1.0;

      mfem::common::ParDiscreteGradOperator grad(&load.h1_fes, &load.fes);
      grad.Assemble();
      grad.Finalize();

      ParGridFunction GTpsi_y(&load.h1_fes);
      grad.MultTranspose(psi_y, GTpsi_y);

      ParBilinearForm D(&load.h1_fes);
      D.AddDomainIntegrator(new DiffusionIntegrator);
      D.Assemble();
      D.Finalize();

      /// compute psi_psi
      /// D^T \psi_psi = G^T \psi_y
      ParGridFunction psi_psi(&load.h1_fes);
      psi_psi = 0.0;
      HypreParMatrix Dmat;
      {
         Array<int> ess_bdr, ess_bdr_tdofs;
         ess_bdr.SetSize(load.h1_fes.GetParMesh()->bdr_attributes.Max());
         ess_bdr = 1;
         load.h1_fes.GetEssentialTrueDofs(ess_bdr, ess_bdr_tdofs);

         Vector PSI_PSI;
         Vector RHS;
         D.FormLinearSystem(ess_bdr_tdofs, psi_psi, GTpsi_y, Dmat, PSI_PSI, RHS);
         auto *DmatT = Dmat.Transpose();
         HypreBoomerAMG amg(*DmatT);
         amg.SetPrintLevel(0);
         HyprePCG pcg(*DmatT);
         pcg.SetTol(1e-14);
         pcg.SetMaxIter(200);
         pcg.SetPrintLevel(-1);
         pcg.SetPreconditioner(amg);
         pcg.Mult(RHS, PSI_PSI);

         D.RecoverFEMSolution(PSI_PSI, GTpsi_y, psi_psi);
         delete DmatT;
      }

      /// compute psi_j
      /// M^T \psi_j = -W^T \psi_psi + \psi_k
      ParMixedBilinearForm weakDiv(&load.fes, &load.h1_fes);
      weakDiv.AddDomainIntegrator(new VectorFEWeakDivergenceIntegrator);
      weakDiv.Assemble();
      weakDiv.Finalize();

      ParGridFunction WTpsipsi(&load.fes);
      weakDiv.MultTranspose(psi_psi, WTpsipsi); WTpsipsi *= -1.0;
      WTpsipsi += psi_k;

      ParGridFunction psi_j(&load.fes);
      {
         Array<int> ess_tdof_list;
         HypreParMatrix Mmat;
         Vector PSIJ;
         Vector RHS;
         M.FormLinearSystem(ess_tdof_list, psi_j, WTpsipsi,
                            Mmat, PSIJ, RHS);
         auto *MmatT = Mmat.Transpose();

         HypreBoomerAMG amg(*MmatT);
         amg.SetPrintLevel(0);
         HyprePCG pcg(*MmatT);
         pcg.SetTol(1e-14);
         pcg.SetMaxIter(200);
         pcg.SetPrintLevel(-1);
         pcg.SetPreconditioner(amg);
         pcg.Mult(RHS, PSIJ);

         M.RecoverFEMSolution(PSIJ, WTpsipsi, psi_j);
         delete MmatT;
      }

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
   // div_free_proj(h1_fes, fes, h1_fes.GetElementTransformation(0)->OrderW()
   //                               + 2 * fes.GetFE(0)->GetOrder(),
   //               NULL, NULL, NULL), dirty(true)
{
   /// Create a H(curl) mass matrix for integrating grid functions
   nd_mass.AddDomainIntegrator(new VectorFEMassIntegrator);

   J.AddDomainIntegrator(new VectorFEDomainLFIntegrator(current));
}

void CurrentLoad::assembleLoad()
{
   // assemble mass matrix
   nd_mass.Update();
   nd_mass.Assemble();
   nd_mass.Finalize();

   // assemble linear form
   J.Assemble();

   // project current coeff as initial guess for iterative solve
   j.ProjectCoefficient(current);
   // std::cout << "j load: " << j.Norml2() << "\n";

   // {
   //    ParaViewDataCollection paraview_dc("current_raw", j.FESpace()->GetMesh());
   //    paraview_dc.SetPrefixPath("ParaView");
   //    paraview_dc.SetLevelsOfDetail(fes.GetElementOrder(0));
   //    paraview_dc.SetCycle(0);
   //    paraview_dc.SetDataFormat(VTKFormat::BINARY);
   //    paraview_dc.SetHighOrderOutput(true);
   //    paraview_dc.SetTime(0.0); // set the time
   //    paraview_dc.RegisterField("CurrentDensity", &j);
   //    paraview_dc.Save();
   // }

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

   /// Compute the discretely divergence-free portion of j
   // ParGridFunction div_free_current_vec(&fes);
   div_free_current_vec = 0.0;
   div_free_proj.Mult(j, div_free_current_vec);

   // std::cout << "div free norm new load: " << div_free_current_vec.Norml2() << "\n";
   
   /// Save divergence free current in the ParaView format
   // ParaViewDataCollection paraview_dc("current", div_free_current_vec.FESpace()->GetMesh());
   // paraview_dc.SetPrefixPath("ParaView");
   // paraview_dc.SetLevelsOfDetail(fes.GetElementOrder(0));
   // paraview_dc.SetCycle(0);
   // paraview_dc.SetDataFormat(VTKFormat::BINARY);
   // paraview_dc.SetHighOrderOutput(true);
   // paraview_dc.SetTime(0.0); // set the time
   // paraview_dc.RegisterField("CurrentDensity",&div_free_current_vec);
   // paraview_dc.Save();

   /// Compute the dual of div_free_current_vec
   div_free_current_vec.ParallelAssemble(scratch);
   M.Mult(scratch, load);

   // /// Compute the dual of div_free_current_vec
   // nd_mass.Assemble();
   // nd_mass.Finalize();
   // {
   //    ParGridFunction test(&fes);
   //    test = 1.0;
   //    scratch = 0.0;
   //    nd_mass.AddMult(test, scratch);
   //    std::cout << "one's norm: " << scratch.Norml2() << "\n\n";
   // }
   // scratch = 0.0;
   // nd_mass.AddMult(div_free_current_vec, scratch);
   // // {
   // //    /// Save divergence free current in the ParaView format
   // //    ParaViewDataCollection paraview_dc("scratch", scratch.FESpace()->GetMesh());
   // //    paraview_dc.SetPrefixPath("ParaView");
   // //    paraview_dc.SetLevelsOfDetail(fes.GetElementOrder(0));
   // //    paraview_dc.SetCycle(0);
   // //    paraview_dc.SetDataFormat(VTKFormat::BINARY);
   // //    paraview_dc.SetHighOrderOutput(true);
   // //    paraview_dc.SetTime(0.0); // set the time
   // //    paraview_dc.RegisterField("CurrentDensity",&scratch);
   // //    paraview_dc.Save();
   // // }   
   // std::cout << "scratch norm: " << scratch.Norml2() << "\n\n";
   // scratch.GetTrueDofs(load);

   // std::cout << "load norm: " << load.Norml2() << "\n\n";

}


} // namespace mach
