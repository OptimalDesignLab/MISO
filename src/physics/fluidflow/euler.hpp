#ifndef MACH_EULER
#define MACH_EULER

//#include <fstream>

#include "mfem.hpp"

#include "solver.hpp"
#include "euler_integ.hpp"

//using adept::adouble;

namespace mach
{

/// Solver for inviscid flow problems
/// dim - number of spatial dimensions (1, 2, or 3)
template <int dim, bool entvar = false>
class EulerSolver : public AbstractSolver
{
public:
   /// Class constructor.
   /// \param[in] opt_file_name - file where options are stored
   /// \param[in] smesh - if provided, defines the mesh for the problem
   /// \param[in] dim - number of dimensions
   /// \todo Can we infer dim some other way without using a template param?
   EulerSolver(const std::string &opt_file_name,
               std::unique_ptr<mfem::Mesh> smesh = nullptr);

   /// Find the global time step size
   /// \param[in] iter - the current iteration
   /// \param[in] t - the current time (before the step)
   /// \param[in] t_final - the final time
   /// \param[in] dt_old - the step size that was just taken
   /// \returns dt - appropriate step size
   /// \note If "const-cfl" option is invoked, this uses the average spectral
   /// radius to estimate the largest wave speed, and uses the minimum distance
   /// between nodes for the length in the CFL number.
   /// \note If "steady" option is involved, the time step will increase based
   /// on the baseline value of "dt" and the inverse residual norm.
   virtual double calcStepSize(int iter, double t, double t_final,
                               double dt_old) const override;

   /// Sets `q_ref` to the free-stream conservative variables
   void getFreeStreamState(mfem::Vector &q_ref);

   /// Returns the L2 error between the discrete and exact conservative vars.
   /// \param[in] u_exact - function that defines the exact **state** 
   /// \param[in] entry - if >= 0, the L2 error of state `entry` is returned
   /// \returns L2 error
   /// \note The solution given by `u_exact` is for the state, conservative or
   /// entropy variables.  **Do not give the exact solution for the conservative
   /// variables if using entropy variables**.   The conversion to conservative
   /// variables is done by this function.
   double calcConservativeVarsL2Error(void (*u_exact)(const mfem::Vector &,
                                                      mfem::Vector &),
                                      int entry = -1);
   /// convert conservative variables to entropy variables
   /// \param[in/out] state - the conservative/entropy variables
   virtual void convertToEntvar(mfem::Vector &state);

   /// get sensitivity of output quantity with respect to a scalar flow parameter
   /// TODO: Need to generalize this
   double getParamSens();

   /// verify parameter sensitivity using a finite difference approximation
   void verifyParamSens();

   /// compute mean and stdev of the output w.r.t. uncertain mach number
   virtual void calcStatistics() override;
protected:
   /// free-stream Mach number
   double mach_fs;
   /// free-stream angle of attack
   double aoa_fs;
   /// index of dimension corresponding to nose to tail axis
   int iroll;
   /// index of "vertical" dimension in body frame
   int ipitch;
   /// used to record the entropy
   std::ofstream entropylog;
   /// used to store the initial residual norm for PTC and convergence checks
   double res_norm0 = -1.0;

   /// Initialize `res` and either `mass` or `nonlinear_mass`
   virtual void constructForms() override;

   /// Add domain integrators to `mass`
   /// \param[in] alpha - scales the data; used to move terms to rhs or lhs
   virtual void addMassIntegrators(double alpha) override;

   /// Add domain integrator to the nonlinear mass operator
   /// \param[in] alpha - scales the data; used to ove terems to rhs or lhs
   virtual void addNonlinearMassIntegrators(double alpha) override;

   /// Add volume integrators to `res` based on `options`
   /// \param[in] alpha - scales the data; used to move terms to rhs or lhs
   virtual void addResVolumeIntegrators(double alpha) override;

   /// Add boundary-face integrators to `res` based on `options`
   /// \param[in] alpha - scales the data; used to move terms to rhs or lhs
   virtual void addResBoundaryIntegrators(double alpha) override;

   /// Add interior-face integrators to `res` based on `options`
   /// \param[in] alpha - scales the data; used to move terms to rhs or lhs
   virtual void addResInterfaceIntegrators(double alpha) override;

   virtual void addEntVolumeIntegrators() override;

   /// Create `output` based on `options` and add approporiate integrators
   virtual void addOutputs() override;

   /// Return the number of state variables
   virtual int getNumState() override {return dim+2; }

   /// For code that should be executed before the time stepping begins
   virtual void initialHook() override;

   /// For code that should be executed before `ode_solver->Step`
   /// \param[in] iter - the current iteration
   /// \param[in] t - the current time (before the step)
   /// \param[in] dt - the step size that will be taken
   virtual void iterationHook(int iter, double t, double dt) override;

   /// Determines when to exit the time stepping loop
   /// \param[in] iter - the current iteration
   /// \param[in] t - the current time (after the step)
   /// \param[in] t_final - the final time
   /// \param[in] dt - the step size that was just taken
   virtual bool iterationExit(int iter, double t, double t_final, 
                              double dt) override;

   /// For code that should be executed after the time stepping ends
   /// \param[in] iter - the terminal iteration
   /// \param[in] t_final - the final time
   virtual void terminalHook(int iter, double t_final) override;
};

} // namespace mach

#endif
