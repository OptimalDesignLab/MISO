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
      if (input.first == "current-density")
      {
         load.current_density = input.second.getValue();
      }
      else if (input.first == "fill-factor")
      {
         load.fill_factor = input.second.getValue();
      }
   }
   load.nd_mass.Update();
   load.dirty = true;
}

void addLoad(CurrentLoad &load,
             Vector &tv)
{
   if (load.dirty)
   {
      load.dirty = false;
      load.assembleLoad();
   }
   const auto effective_cd = load.current_density * load.fill_factor;
   add(tv, -effective_cd, load.load, tv);
}

CurrentLoad::CurrentLoad(ParFiniteElementSpace &pfes,
                         VectorCoefficient &current_coeff)
   : fes(pfes), h1_coll(fes.GetFE(0)->GetOrder(), fes.GetMesh()->Dimension()),
   h1_fes(fes.GetParMesh(), &h1_coll),
   rt_coll(fes.GetFE(0)->GetOrder(), fes.GetMesh()->Dimension()), 
   rt_fes(fes.GetParMesh(), &rt_coll), current_density(1.0), fill_factor(1.0),
   load(&fes), scratch(&fes), nd_mass(&fes), J(&fes), j(&fes),
   div_free_current_vec(&fes),
   div_free_proj(h1_fes, fes, h1_fes.GetElementTransformation(0)->OrderW()
                                 + 2 * fes.GetFE(0)->GetOrder(),
                 NULL, NULL, NULL), dirty(true)
{
   /// Create a H(curl) mass matrix for integrating grid functions
   nd_mass.AddDomainIntegrator(new VectorFEMassIntegrator);

   J.AddDomainIntegrator(new VectorFEDomainLFIntegrator(current_coeff));

   // project current_coeff as initial guess for iterative solve
   j.ProjectCoefficient(current_coeff);
}

void CurrentLoad::assembleLoad()
{
   // assemble mass matrix
   nd_mass.Assemble();
   nd_mass.Finalize();

   // assemble linear form
   J.Assemble();


   HypreParMatrix M;
   Vector X, RHS;
   Array<int> ess_tdof_list;
   nd_mass.FormLinearSystem(ess_tdof_list, j, J, M, X, RHS);

   HypreBoomerAMG amg(M);
   amg.SetPrintLevel(-1);

   HyprePCG pcg(M);
   pcg.SetTol(1e-12);
   pcg.SetMaxIter(200);
   pcg.SetPrintLevel(2);
   pcg.SetPreconditioner(amg);
   pcg.Mult(RHS, X);

   nd_mass.RecoverFEMSolution(X, J, j);

   /// Compute the discretely divergence-free portion of j
   ParGridFunction div_free_current_vec(&fes);
   div_free_proj.Mult(j, div_free_current_vec);

   std::cout << "div free norm new load: " << div_free_current_vec.Norml2() << "\n\n";

   /// get the div_free_current_vec's true dofs
   div_free_current_vec.ParallelAssemble(scratch);

   /// integrate the divergence free current vec
   M.Mult(scratch, load);
}


} // namespace mach
