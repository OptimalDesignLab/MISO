#ifndef MACH_EULER
#define MACH_EULER

#include "mfem.hpp"
#include "solver.hpp"
#include "euler_fluxes.hpp"
#include "adept.h"

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

   /// Compute the Jacobian of the Euler flux with respect to state variables
   /// \param[in] dir - direction in which euler flux is calculated
   /// \param[in] q - conservative state variables
   /// \param[out] Jac - the Jacobian of the euler flux with respect to q
   /// \tparam dim - number of spatial dimensions (1, 2, or 3)
   template<int dim>
   static void calcEulerFluxJacQ(const mfem::Vector &dir, const mfem::Vector &q,
                                 mfem::DenseMatrix &jac);

   /// Compute the Jacobian of the Euler flux with respect to dir
   /// \param[in] dir - direction in which euler flux is calculated
   /// \param[in] q - conservative state variables
   /// \param[out] Jac - the Jacobian of the euler flux with respect to dir
   /// \tparam dim - number of spatial dimensions (1, 2, or 3)
   template<int dim>
   static void calcEulerFluxJacDir(const mfem::Vector &dir, const mfem::Vector &q,
                                    mfem::DenseMatrix &jac);

   /// Compute the Jacobian of the slip wall flux with respect to Q
   /// \param[in] x - not used
   /// \param[in] dir - desired (scaled) normal vector to the wall
   /// \param[in] q - conservative state variable on the boundary
   /// \param[out] Jac - the Jacobian of the boundary flux in the direction 
   ///                   `dir` with respect to Q
   /// \tparam dim - number of spatial dimensions (1, 2, or 3)
   template <int dim>
   static void calcSlipWallFluxJacQ(const mfem::Vector &x, const mfem::Vector &dir,
                                    const mfem::Vector &q, mfem::DenseMatrix Jac);

   /// Compute the Jacobian of the slip wall flux with respect to Dir
   /// \param[in] x - not used
   /// \param[in] dir - desired (scaled) normal vector to the wall
   /// \param[in] q - conservative state variable on the boundary
   /// \param[out] Jac - the Jacobian of the boundary flux in the direction 
   ///                   `dir` with respect to dir
   /// \tparam dim - number of spatial dimensions (1, 2, or 3)
   template <int dim>
   static void calcSlipWallFluxJacDir(const mfem::Vector &x, const mfem::Vector &dir,
                                      const mfem::Vector &q, mfem::DenseMatrix Jac);


   /// Below are flux functions in `euler_fluxes.hpp` wrapped using 
   /// mfem::Vector as inputs

   /// Euler flux function in a given (scaled) direction
   /// \param[in] dir - direction in which the flux is desired
   /// \param[in] q - conservative variables
   /// \param[out] flux - fluxes in the direction `dir`
   /// \tparam dim - number of spatial dimensions (1, 2, or 3)
   template <int dim>
   static void calcEulerFlux(const mfem::Vector &dir, const mfem::Vector &q,
                             mfem::Vector &flux)
   {
      calcEulerFlux<double, dim>(dir.GetData(), q.GetData(), flux.GetData());
   }

   /// Ismail-Roe two-point (dyadic) entropy conservative flux function
   /// \param[in] di - physical coordinate direction in which flux is wanted
   /// \param[in] qL - conservative variables at "left" state
   /// \param[in] qR - conservative variables at "right" state
   /// \param[out] flux - fluxes in the direction `di`
   /// \tparam xdouble - typically `double` or `adept::adouble`
   /// \tparam dim - number of spatial dimensions (1, 2, or 3)

   template<int dim>
   static void calcIsmailRoeFlux(int di, const mfem::Vector &qL,
                                 const mfem::Vector &qR, mfem::Vector &flux)
   {
      calcIsmailRoeFlux<double, dim>(di, qL.GetData(), qR.GetData(),
                                     flux.GetData());
   }
   

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