#ifndef MACH_EULER
#define MACH_EULER

#include "mfem.hpp"
#include "solver.hpp"

namespace mach
{

/// Solver for linear advection problems
class EulerSolver : public AbstractSolver
{
public:
   /// Class constructor.
   /// \param[in] opt_file_name - file where options are stored
   /// \param[in] smesh - if provided, defines the mesh for the problem
   /// \param[in] dim - number of dimensions
   /// \todo Can we infer dim some other way without using a template param?
   EulerSolver(const std::string &opt_file_name, 
               std::unique_ptr<mfem::Mesh> smesh = nullptr,
               int dim = 1);
   
   /// Find the gobal step size for the given CFL number
   /// \param[in] cfl - target CFL number for the domain
   /// \returns dt_min - the largest step size for the given CFL
   /// This uses the average spectral radius to estimate the largest wave speed,
   /// and uses the minimum distance between nodes for the length in the CFL
   /// number.
   virtual double calcStepSize(double cfl) const;

   /// Compute the residual norm based on the current solution in `u`
   /// \returns the l2 (discrete) norm of the residual evaluated at `u`
   double calcResidualNorm();

   /// Calculate the Euler flux jacobian respect to state variables.
   /// \param[in] dir - direction in which flux is calculated.
   /// \param[in] q - the conservative variables
   /// \param[in] jac - a pointer to the jacobian.
   template<int dim>
   static void calcEulerFluxJacQ(const mfem::Vector& dir, const mfem::Vector& q,
                                 mfem::DenseMatrix* jac);
   
   /// Calculate the Euler flux jacobian respect to direction.
   /// \param[in] dir - direction in which flux is calculated.
   /// \param[in] q - the conservative variables
   /// \param[in] jac - a pointer to the jacobian.
   template<int dim>
   static void calcEulerFluxJacDir(const mfem::Vector& dir, const mfem::Vector& q,
                                 mfem::DenseMatrix* jac);

protected:
   /// `bndry_marker[i]` lists the boundaries associated with a particular BC
   std::vector<mfem::Array<int>> bndry_marker;
   /// the mass matrix bilinear form
   std::unique_ptr<BilinearFormType> mass;
   /// the spatial residual (a semilinear form)
   std::unique_ptr<NonlinearFormType> res;
   /// mass matrix (move to AbstractSolver?)
   std::unique_ptr<MatrixType> mass_matrix;

   /// Add boundary-face integrators to `res` based on `options`
   /// \param[in] alpha - scales the data; used to move terms to rhs or lhs
   /// \param[in] dim - number of dimensions
   void addBoundaryIntegrators(double alpha, int dim = 1);
};

} // namespace mach

#endif