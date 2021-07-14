#include "mfem.hpp"

#include "coefficient.hpp"
#include "mach_input.hpp"
#include "current_load.hpp"

using namespace mfem;

namespace mach
{

/// set inputs should include fields, so things can check it they're "dirty"
void setInputs(CurrentLoad &load,
               const MachInputs &inputs)
{
   for (auto &input : inputs)
   {
      if (input.first == "current_density")
      {
         load.current_density = input.second.getValue();
         load.dirty = true;
      }
      else if (input.first == "mesh_coords")
      {
         load.nd_mass.Update();
         load.dirty = true;
      }
   }
   load.current.SetAConst(load.current_density);
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

CurrentLoad::CurrentLoad(ParFiniteElementSpace &pfes,
                         VectorCoefficient &current_coeff)
   : current_density(1.0), current(1.0, current_coeff),
   fes(pfes), h1_coll(fes.GetFE(0)->GetOrder(), fes.GetMesh()->Dimension()),
   h1_fes(fes.GetParMesh(), &h1_coll),
   rt_coll(fes.GetFE(0)->GetOrder(), fes.GetMesh()->Dimension()), 
   rt_fes(fes.GetParMesh(), &rt_coll), nd_mass(&fes), J(&fes), j(&fes),
   div_free_current_vec(&fes), scratch(&fes), load(&fes), 
   div_free_proj(h1_fes, fes, h1_fes.GetElementTransformation(0)->OrderW()
                                 + 2 * fes.GetFE(0)->GetOrder(),
                 NULL, NULL, NULL), dirty(true)
{
   /// Create a H(curl) mass matrix for integrating grid functions
   nd_mass.AddDomainIntegrator(new VectorFEMassIntegrator);

   J.AddDomainIntegrator(new VectorFEDomainLFIntegrator(current));
}

void CurrentLoad::assembleLoad()
{
   // assemble mass matrix
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
   ParaViewDataCollection paraview_dc("current", div_free_current_vec.FESpace()->GetMesh());
   paraview_dc.SetPrefixPath("ParaView");
   paraview_dc.SetLevelsOfDetail(fes.GetElementOrder(0));
   paraview_dc.SetCycle(0);
   paraview_dc.SetDataFormat(VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(true);
   paraview_dc.SetTime(0.0); // set the time
   paraview_dc.RegisterField("CurrentDensity",&div_free_current_vec);
   paraview_dc.Save();

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
