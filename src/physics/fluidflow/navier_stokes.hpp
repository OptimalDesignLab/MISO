#ifndef MACH_NAVIER_STOKES
#define MACH_NAVIER_STOKES

#include "mfem.hpp"

#include "euler.hpp"

namespace mach
{

/// Solver for Navier-Stokes flows
/// dim - number of spatial dimensions (1, 2, or 3)
/// entvar - if true, the entropy variables are used in the integrators
template <int dim, bool entvar = false>
class NavierStokesSolver : public EulerSolver<dim, entvar>
{
protected:
   /// free-stream Reynolds number
   double re_fs;
   /// Prandtl number
   double pr_fs;

   /// Class constructor.
   /// \param[in] json_options - json object containing the options
   /// \param[in] smesh - if provided, defines the mesh for the problem
   NavierStokesSolver(const nlohmann::json &json_options,
                      std::unique_ptr<mfem::Mesh> smesh,
                      MPI_Comm comm);

   /// Add volume/domain integrators to `res` based on `options`
   /// \param[in] alpha - scales the data; used to move terms to rhs or lhs
   /// \note This function calls EulerSolver::addVolumeIntegrators() first
   virtual void addResVolumeIntegrators(double alpha);

   /// Add boundary-face integrators to `res` based on `options`
   /// \param[in] alpha - scales the data; used to move terms to rhs or lhs
   /// \note This function calls EulerSolver::addBoundaryIntegrators() first
   virtual void addResBoundaryIntegrators(double alpha);

   /// Add interior-face integrators to `res` based on `options`
   /// \param[in] alpha - scales the data; used to move terms to rhs or lhs
   /// \note This function calls EulerSolver::addInterfaceIntegrators() first
   virtual void addResInterfaceIntegrators(double alpha);

   /// Create `output` based on `options` and add approporiate integrators
   virtual void addOutputs() override;

   /// Set the state corresponding to the inflow boundary
   /// \param[in] q_in - state corresponding to the inflow
   void getViscousInflowState(mfem::Vector &q_in);

   /// Set the state corresponding to the outflow boundary
   /// \param[in] q_out - state corresponding to the outflow
   void getViscousOutflowState(mfem::Vector &q_out);

   friend SolverPtr createSolver<NavierStokesSolver<dim, entvar>>(
       const nlohmann::json &json_options,
       std::unique_ptr<mfem::Mesh> smesh,
       MPI_Comm comm);    
};

/// Defines the right-hand side of Equation (7.5) in "Entropy stable spectral
/// collocation schemes for the Navier-Stokes questions: discontinuous
/// interfaces."  See also Fisher's thesis in the appendix, but note that the 
/// value of alpha listed there is incorrect!!!
/// \param[in] Re - Reynolds number
/// \param[in] Ma - Mach number
/// \param[in] v - velocity ration u/u_L
/// \returns the right hand side of Equation (7.5)
double shockEquation(double Re, double Ma, double v);

/// Defines the exact solution for the steady viscous shock.
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - conservative variables stored as a 4-vector
void shockExact(const mfem::Vector &x, mfem::Vector& u);

/// Defines the exact solution for the viscous MMS verification
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - conservative variables stored as a 4-vector
void viscousMMSExact(const mfem::Vector &x, mfem::Vector& u);

} // namespace mach

#endif