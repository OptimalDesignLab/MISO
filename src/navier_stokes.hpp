#ifndef MACH_NAVIER_STOKES
#define MACH_NAVIER_STOKES
#include "mfem.hpp"
#include "euler.hpp"

namespace mach
{

/// Solver for Navier-Stokes flows
template <int dim>
class NavierStokesSolver : public EulerSolver<dim>
{
public:
   /// Class constructor.
   /// \param[in] opt_file_name - file where options are stored
   /// \param[in] smesh - if provided, defines the mesh for the problem
   /// \param[in] dim - number of dimensions
   /// \todo Can we infer dim some other way without using a template param?
   NavierStokesSolver(const std::string &opt_file_name,
                      std::unique_ptr<mfem::Mesh> smesh = nullptr);

protected:
   /// free-stream Reynolds number
   double re_fs;
   /// Prandtl number
   double pr_fs;

   /// Add volume/domain integrators to `res` based on `options`
   /// \param[in] alpha - scales the data; used to move terms to rhs or lhs
   /// \note This function calls EulerSolver::addVolumeIntegrators() first
   virtual void addVolumeIntegrators(double alpha);

   /// Add boundary-face integrators to `res` based on `options`
   /// \param[in] alpha - scales the data; used to move terms to rhs or lhs
   /// \note This function calls EulerSolver::addBoundaryIntegrators() first
   virtual void addBoundaryIntegrators(double alpha);

   /// Add interior-face integrators to `res` based on `options`
   /// \param[in] alpha - scales the data; used to move terms to rhs or lhs
   /// \note This function calls EulerSolver::addInterfaceIntegrators() first
   virtual void addInterfaceIntegrators(double alpha);

   /// Set the state corresponding to the inflow boundary
   /// \param[in] q_in - state corresponding to the inflow
   void getViscousInflowState(mfem::Vector &q_in);

   /// Set the state corresponding to the outflow boundary
   /// \param[in] q_out - state corresponding to the outflow
   void getViscousOutflowState(mfem::Vector &q_out);

   /// Create `output` based on `options` and add approporiate integrators
   ///void addOutputs();
};

} // namespace mach

#endif