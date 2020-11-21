#ifndef MACH_TYPES
#define MACH_TYPES

#include "mfem.hpp"

namespace mach
{

// Aliases for various mfem types.
// Originally used to distinguish between serial and parallel types
#ifdef MFEM_USE_PUMI
   using MeshType = mfem::ParMesh;
#else
   using MeshType = mfem::ParMesh;
#endif
   using SpaceType = mfem::ParFiniteElementSpace;
   using LinearFormType = mfem::ParLinearForm;
   using BilinearFormType = mfem::ParBilinearForm;
   using NonlinearFormType = mfem::ParNonlinearForm;
   using GridFunType = mfem::ParGridFunction;
   using MatrixType = mfem::HypreParMatrix;
   using SmootherType = mfem::HypreSmoother;
   using DiscLinOperatorType = mfem::ParDiscreteLinearOperator;
   using MixedBilinearFormType = mfem::ParMixedBilinearForm;
   using CGType = mfem::HyprePCG;
   using EMPrecType = mfem::HypreAMS;
   using EMPrecType2 = mfem::HypreBoomerAMG;
   using SolverType = CGType;

} // namespace mach

#endif
