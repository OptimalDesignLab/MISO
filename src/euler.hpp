#ifndef MACH_EULER
#define MACH_EULER

#include "mfem.hpp"
#include "solver.hpp"
#include "adept.h"
#include "inviscid_integ.hpp"
#include "euler_fluxes.hpp"

namespace mach
{

/// Integrator for the Euler flux over an element
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \note This derived class uses the CRTP
template <int dim>
class EulerIntegrator : public InviscidIntegrator<EulerIntegrator<dim>>
{
public:
   /// Construct an integrator for the Euler flux over elements
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] a - factor, usually used to move terms to rhs
   EulerIntegrator(adept::Stack &diff_stack, double a = 1.0)
       : InviscidIntegrator<EulerIntegrator<dim>>(diff_stack, dim + 2, a) {}

   /// Euler flux function in a given (scaled) direction
   /// \param[in] dir - direction in which the flux is desired
   /// \param[in] q - conservative variables
   /// \param[out] flux - fluxes in the direction `dir`
   /// \note wrapper for the relevant function in `euler_fluxes.hpp`
   void calcFlux(const mfem::Vector &dir, const mfem::Vector &q,
                 mfem::Vector &flux);

   /// Compute the Jacobian of the Euler flux w.r.t. `q`
   /// \parma[in] dir - desired direction for the flux 
   /// \param[in] q - state at which to evaluate the flux Jacobian
   /// \param[out] flux_jac - Jacobian of the flux function w.r.t. `q`
   void calcFluxJacState(const mfem::Vector &dir, const mfem::Vector &q,
                         mfem::DenseMatrix &flux_jac);

   /// Compute the Jacobian of the flux function `flux` w.r.t. `dir`
   /// \parma[in] dir - desired direction for the flux 
   /// \param[in] q - state at which to evaluate the flux Jacobian
   /// \param[out] flux_jac - Jacobian of the flux function w.r.t. `dir`
   void calcFluxJacDir(const mfem::Vector &dir, const mfem::Vector &q,
                       mfem::DenseMatrix &flux_jac);
};

/// Integrator for the two-point entropy conservative Ismail-Roe flux
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \note This derived class uses the CRTP
template <int dim>
class IsmailRoeIntegrator : public DyadicFluxIntegrator<IsmailRoeIntegrator<dim>>
{
public:
   /// Construct an integrator for the Ismail-Roe flux over domains
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] a - factor, usually used to move terms to rhs
   IsmailRoeIntegrator(adept::Stack &diff_stack, double a = 1.0)
       : DyadicFluxIntegrator<IsmailRoeIntegrator<dim>>(
             diff_stack, dim+2, a) {}

   /// Ismail-Roe two-point (dyadic) entropy conservative flux function
   /// \param[in] di - physical coordinate direction in which flux is wanted
   /// \param[in] qL - conservative variables at "left" state
   /// \param[in] qR - conservative variables at "right" state
   /// \param[out] flux - fluxes in the direction `di`
   /// \note This is simply a wrapper for the function in `euler_fluxes.hpp`
   void calcFlux(int di, const mfem::Vector &qL,
                 const mfem::Vector &qR, mfem::Vector &flux)
 
   {
   
      calcIsmailRoeFlux<double,dim>(di, qL.GetData(), qR.GetData(),
                                 flux.GetData());
   }
   /// Compute the Jacobians of `flux` with respect to `u_left` and `u_right`
   /// \param[in] di - desired coordinate direction for flux 
   /// \param[in] qL - the "left" state
   /// \param[in] qR - the "right" state
   /// \param[out] jacL - Jacobian of `flux` w.r.t. `qL`
   /// \param[out] jacR - Jacobian of `flux` w.r.t. `qR`   
   void calcFluxJacStates(int di, const mfem::Vector &qL,
                          const mfem::Vector &qR,
                          mfem::DenseMatrix &jacL,
                          mfem::DenseMatrix &jacR)

   {

    // import stack and adouble from adept
     using adept::adouble;
     // vector of active input variables
     mfem::DenseMatrix Jac(dim+2,2*(dim+2));
     std::vector<adouble> qL_a(qL.Size());
     std::vector<adouble> qR_a(qR.Size());
     // initialize adouble inputs
     adept::set_values(qL_a.data(),qL.Size(),qL.GetData());
     adept::set_values(qR_a.data(),qR.Size(),qR.GetData());
     // start recording
     this->stack.new_recording();
     // create vector of active output variables
     std::vector<adouble> flux_a(qL.Size());
     // run algorithm 
     mach::calcIsmailRoeFlux<adouble,dim>(di,qL_a.data(),qR_a.data(),flux_a.data());
     // identify independent and dependent variables
     this->stack.independent(qL_a.data(),qL.Size());
     this->stack.independent(qR_a.data(),qR.Size());
     this->stack.dependent(flux_a.data(),qL.Size());
     // compute and store jacobian in jac ?
     this->stack.jacobian_reverse(Jac.GetData());
     // retrieve the jacobian w.r.t left state
     jacL.CopyCols(Jac,0,dim+1);
     // retrieve the jacobian w.r.t right state
     jacR.CopyCols(Jac,dim+2,2*(dim+2)-1);

  }
};

/// Integrator for entropy stable local-projection stabilization
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \note This derived class uses the CRTP
template <int dim>
class EntStableLPSIntegrator : public LPSIntegrator<EntStableLPSIntegrator<dim>>
{
public:
   /// Construct an entropy-stable LPS integrator
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   /// \param[in] coeff - the LPS coefficient
   EntStableLPSIntegrator(adept::Stack &diff_stack, double a = 1.0,
                          double coeff = 1.0)
       : LPSIntegrator<EntStableLPSIntegrator<dim>>(
             diff_stack, dim + 2, a, coeff) {}

   /// converts conservative variables to entropy variables
   /// \param[in] q - conservative variables that are to be converted
   /// \param[out] w - entropy variables corresponding to `q`
   /// \note a wrapper for the relevant function in `euler_fluxes.hpp`
   void convertVars(const mfem::Vector &q, mfem::Vector &w);

   /// Compute the Jacobian of the mapping `convert` w.r.t. `u`
   /// \param[in] q - conservative variables that are to be converted
   /// \param[out] dwdu - Jacobian of entropy variables w.r.t. `u`
   void convertVarsJacState(const mfem::Vector &q, mfem::DenseMatrix &dwdu);

   /// Applies the matrix `dQ/dW` to `vec`, and scales by the avg. spectral radius
   /// \param[in] adjJ - the adjugate of the mapping Jacobian
   /// \param[in] q - the state at which `dQ/dW` and radius are to be evaluated
   /// \param[in] vec - the vector being multiplied
   /// \param[out] mat_vec - the result of the operation
   /// \warning adjJ must be supplied transposed from its `mfem` storage format,
   /// so we can use pointer arithmetic to access its rows.
   /// \note a wrapper for the relevant function in `euler_fluxes.hpp`
   void applyScaling(const mfem::DenseMatrix &adjJ, const mfem::Vector &q,
                     const mfem::Vector &vec, mfem::Vector &mat_vec);

   /// Computes the Jacobian of the product `A(adjJ,q)*v` w.r.t. `q`
   /// \param[in] adjJ - adjugate of the mapping Jacobian
   /// \param[in] q - state at which `dQ/dW` and radius are evaluated
   /// \param[in] vec - vector that is being multiplied
   /// \param[out] mat_vec_jac - Jacobian of product w.r.t. `q`
   /// \warning adjJ must be supplied transposed from its `mfem` storage format,
   /// so we can use pointer arithmetic to access its rows.
   void applyScalingJacState(const mfem::DenseMatrix &adjJ,
                             const mfem::Vector &q,
                             const mfem::Vector &vec,
                             mfem::DenseMatrix &mat_vec_jac);

   /// Computes the Jacobian of the product `A(adjJ,u)*v` w.r.t. `adjJ`
   /// \param[in] adjJ - adjugate of the mapping Jacobian
   /// \param[in] q - state at which the symmetric matrix `A` is evaluated
   /// \param[in] vec - vector that is being multiplied
   /// \param[out] mat_vec_jac - Jacobian of product w.r.t. `adjJ`
   /// \note `mat_vec_jac` stores derivatives treating `adjJ` is a 1d array.
   void applyScalingJacAdjJ(const mfem::DenseMatrix &adjJ,
                            const mfem::Vector &q,
                            const mfem::Vector &vec,
                            mfem::DenseMatrix &mat_vec_jac);
};

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


   /// Compute the Jacobian of the Euler flux with respect to state variables
   /// \param[in] dir - direction in which euler flux is calculated
   /// \param[in] q - conservative state variables
   /// \param[out] Jac - the Jacobian of the euler flux with respect to q
   /// \tparam dim - number of spatial dimensions (1, 2, or 3)
   template <int dim>
   static void calcEulerFluxJacQ(const mfem::Vector &dir, const mfem::Vector &q,
                                 mfem::DenseMatrix &jac);

   /// Compute the Jacobian of the Euler flux with respect to dir
   /// \param[in] dir - direction in which euler flux is calculated
   /// \param[in] q - conservative state variables
   /// \param[out] Jac - the Jacobian of the euler flux with respect to dir
   /// \tparam dim - number of spatial dimensions (1, 2, or 3)
   template <int dim>
   static void calcEulerFluxJacDir(const mfem::Vector &dir,
                                   const mfem::Vector &q,
                                   mfem::DenseMatrix &jac);

   /// Compute the Jacobian of the slip wall flux with respect to Q
   /// \param[in] x - not used
   /// \param[in] dir - desired (scaled) normal vector to the wall
   /// \param[in] q - conservative state variable on the boundary
   /// \param[out] Jac - the Jacobian of the boundary flux in the direction 
   ///                   `dir` with respect to Q
   /// \tparam dim - number of spatial dimensions (1, 2, or 3)
   template <int dim>
   static void calcSlipWallFluxJacQ(const mfem::Vector &x,
                                    const mfem::Vector &dir,
                                    const mfem::Vector &q,
                                    mfem::DenseMatrix &Jac);

   /// Compute the Jacobian of the slip wall flux with respect to Dir
   /// \param[in] x - not used
   /// \param[in] dir - desired (scaled) normal vector to the wall
   /// \param[in] q - conservative state variable on the boundary
   /// \param[out] Jac - the Jacobian of the boundary flux in the direction 
   ///                   `dir` with respect to dir
   /// \tparam dim - number of spatial dimensions (1, 2, or 3)
   template <int dim>
   static void calcSlipWallFluxJacDir(const mfem::Vector &x,
                                      const mfem::Vector &dir,
                                      const mfem::Vector &q,
                                      mfem::DenseMatrix &Jac);
                                      
   // template<int dim>
   // static void calcIsmailRoeJacQ(int di, const mfem::Vector &qL, 
   //                               const mfem::Vector &qR,
   //                               mfem::DenseMatrix &jac);

};

} // namespace mach

#endif