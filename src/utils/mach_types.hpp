#ifndef MACH_TYPES
#define MACH_TYPES

#include "mfem.hpp"

namespace mach
{

// Aliases that distinguish between serial and parallel types
// Todo: Is this ok "polluting" the mach namespace with these aliases?
// Todo: Should this go in a separate header file (e.g. "mach_types.hpp")?
#ifdef MFEM_USE_MPI
#ifdef MFEM_USE_PUMI
   using MeshType = mfem::ParPumiMesh;
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
#else
   using MeshType = mfem::Mesh;
   using SpaceType = mfem::FiniteElementSpace;
   using LinearFormType = mfem::LinearForm;
   using BilinearFormType = mfem::BilinearForm;
   using NonlinearFormType = mfem::NonlinearForm;
   using GridFunType = mfem::GridFunction;
   using MatrixType = mfem::SparseMatrix;
   using SmootherType = mfem::DSmoother;
   using DiscLinOperatorType = mfem::DiscreteLinearOperator;
   using MixedBilinearFormType = mfem::MixedBilinearForm;
   using CGType = mfem::CGSolver;
   using EMPrecType = mfem::GSSmoother;
   using SolverType = mfem::CGSolver;
#endif

} // namespace mach

#endif
