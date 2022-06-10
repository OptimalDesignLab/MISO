#ifndef MFEM_EXTENSIONS
#define MFEM_EXTENSIONS

#include "mfem.hpp"
#include "nlohmann/json.hpp"

namespace mach
{
/// steady ode solver
class SteadyODESolver : public mfem::ODESolver
{
public:
   SteadyODESolver(std::ostream *out_stream = nullptr) : out(out_stream) { }

   void Step(mfem::Vector &x, double &t, double &dt) override;

protected:
   mfem::Vector k;
   std::ostream *out;
};

/// Backward Euler pseudo-transient continuation solver
class PseudoTransientSolver : public mfem::ODESolver
{
public:
   PseudoTransientSolver(std::ostream *out_stream = nullptr) : out(out_stream)
   { }

   void Init(mfem::TimeDependentOperator &_f) override;

   void Step(mfem::Vector &x, double &t, double &dt) override;

protected:
   mfem::Vector k;
   std::ostream *out;
};

/// Relaxation version of implicit midpoint method
class RRKImplicitMidpointSolver : public mfem::ODESolver
{
public:
   RRKImplicitMidpointSolver(std::ostream *out_stream = nullptr)
    : out(out_stream)
   { }

   void Init(mfem::TimeDependentOperator &_f) override;

   void Step(mfem::Vector &x, double &t, double &dt) override;

protected:
   mfem::Vector k;
   std::ostream *out;
};

/// Generic explicit relaxation Runge-Kutta (RRK) solver (base class)
/// \note This just modifies mfem's `ExplicitRKSolver` step method
class ExplicitRRKSolver : public mfem::ODESolver
{
public:
   ExplicitRRKSolver(int s_,
                     const double *a_,
                     const double *b_,
                     const double *c_,
                     std::ostream *out_stream = nullptr);

   void Init(mfem::TimeDependentOperator &f_) override;

   void Step(mfem::Vector &x, double &t, double &dt) override;

   virtual ~ExplicitRRKSolver();

protected:
   int s;
   const double *a, *b, *c;
   mfem::Vector y;
   mfem::Vector x_new;
   mfem::Vector *k;
   std::ostream *out;
};

/// An 8-stage, 6th order RK method (Verner's "efficient" 9-stage 6(5) pair).
/// \note This is effectively a copy paste of mfem's RK6Solver.
class RRK6Solver : public ExplicitRRKSolver
{
public:
   RRK6Solver(std::ostream *out_stream = nullptr)
    : ExplicitRRKSolver(8, a, b, c, out_stream)
   { }

protected:
   static const double a[28], b[8], c[7];
};

/// For block-Jacobian preconditioning of block operator systems
/// \note This class is almost identical to mfem::BlockDiagonalPreconditioner;
/// however, unlike MFEM's solver, this one does not check for consistency
/// between the offsets and the given block operator when using
/// SetDiagonalBlock.  This change was needed to permit the use of HYPRE's
/// preconditioners, which do not define Width().  Furthermore, SetOperator is
/// called recursively on the block entries.
class BlockJacobiPreconditioner : public mfem::Solver
{
public:
   /// Constructor that specifies the block structure
   /// \param[in] offsets - mark the start of each row/column block
   BlockJacobiPreconditioner(const mfem::Array<int> &offsets);

   /// Add a square block op in the block-entry (iblock, iblock)
   /// \param[in] iblock - the index of row-column block entry being set
   /// \param[in] op - the solver used to define the (iblock, iblock) entry
   void SetDiagonalBlock(int iblock, mfem::Solver *op);

   /// Calls SetOperator on the diagonal block operators
   /// \param[in] op - a BlockOperator whose diagonal entries are used
   virtual void SetOperator(const mfem::Operator &op) override;

   /// Return the number of blocks
   /// \returns the number of row/column blocks in the preconditioner
   int NumBlocks() const { return nBlocks; }

   /// Get a reference to block `iblock`,`iblock`
   /// \param[in] iblock - index of the desired diagonal entry
   /// \returns a reference to block `iblock`,`iblock`
   Operator &GetDiagonalBlock(int iblock)
   {
      MFEM_VERIFY(op[iblock], "");
      return *op[iblock];
   }

   /// Get a reference to block `iblock`,`iblock` (const version)
   /// \param[in] iblock - index of the desired diagonal entry
   /// \returns a reference to block `iblock`,`iblock`
   const Operator &GetDiagonalBlock(int iblock) const
   {
      MFEM_VERIFY(op[iblock], "");
      return *op[iblock];
   }

   /// Return the offsets for block starts
   mfem::Array<int> &Offsets() { return offsets; }

   /// Read only access to the offsets for block starts
   const mfem::Array<int> &Offsets() const { return offsets; }

   /// Operator application
   /// \param[in] x - the vector being preconditioned
   /// \param[in] y - the preconditioned vector
   virtual void Mult(const mfem::Vector &x, mfem::Vector &y) const override;

   /// Action of the transpose operator
   /// \param[in] x - the vector being preconditioned
   /// \param[in] y - the preconditioned vector
   virtual void MultTranspose(const mfem::Vector &x,
                              mfem::Vector &y) const override;

   /// Preconditioner destructor
   ~BlockJacobiPreconditioner();

   /// Controls the ownership of the blocks
   /// \note if nonzero, BlockJacobiPreconditioner will delete all blocks that
   /// are set (non-NULL); the default value is zero.
   int owns_blocks;

private:
   /// Number of Blocks
   int nBlocks;
   /// Offsets for the starting position of each block
   mfem::Array<int> offsets;
   /// 1D array that stores each block of the operator.
   mfem::Array<Solver *> op;
   /// Temporary Vectors used to efficiently apply the Mult and MultTranspose
   mutable mfem::BlockVector xblock;
   mutable mfem::BlockVector yblock;
};

/// Constuct a linear system solver based on the given options
/// \param[in] comm - MPI communicator used by linear solver
/// \param[in] lin_options - options structure that determines the solver
/// \param[in] prec - non-owning pointer to preconditioner for linear solvers
/// \return unique pointer to the linear solver object
std::unique_ptr<mfem::Solver> constructLinearSolver(
    MPI_Comm comm,
    const nlohmann::json &lin_options,
    mfem::Solver *prec = nullptr);

/// Constructs the nonlinear solver object
/// \param[in] comm - MPI communicator used by non-linear solver
/// \param[in] nonlin_options - options structure that determines the solver
/// \param[in] lin_solver - linear solver for the Newton steps
/// \return unique pointer to the Newton solver object
std::unique_ptr<mfem::NewtonSolver> constructNonlinearSolver(
    MPI_Comm comm,
    const nlohmann::json &nonlin_options,
    mfem::Solver &lin_solver);

}  // namespace mach

#endif
