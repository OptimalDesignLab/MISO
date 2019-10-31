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
   using BilinearFormType = mfem::ParBilinearForm;
   using NonlinearFormType = mfem::ParNonlinearForm;
   using GridFunType = mfem::ParGridFunction;
   using MatrixType = mfem::HypreParMatrix;
   using SmootherType = mfem::HypreSmoother;
   using DiscLinOperatorType = mfem::ParDiscreteLinearOperator;
   using MixedBilinearFormType = mfem::ParMixedBilinearForm;
   using CGType = mfem::Solver;
   using EMPrecType = mfem::HypreBoomerAMG;
#else
   using MeshType = mfem::Mesh;
   using SpaceType = mfem::FiniteElementSpace;
   using BilinearFormType = mfem::BilinearForm;
   using NonlinearFormType = mfem::NonlinearForm;
   using GridFunType = mfem::GridFunction;
   using MatrixType = mfem::SparseMatrix;
   using SmootherType = mfem::DSmoother;
   using DiscLinOperatorType = mfem::DiscreteLinearOperator;
   using MixedBilinearFormType = mfem::MixedBilinearForm;
   using CGType = mfem::Solver;
   using EMPrecType = mfem::GSSmoother;
#endif

} // namespace mach

#endif