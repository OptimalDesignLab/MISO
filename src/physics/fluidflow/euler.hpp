#ifndef MACH_EULER
#define MACH_EULER

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

   /// Find the gobal step size for the given CFL number
   /// \param[in] cfl - target CFL number for the domain
   /// \returns dt_min - the largest step size for the given CFL
   /// This uses the average spectral radius to estimate the largest wave speed,
   /// and uses the minimum distance between nodes for the length in the CFL
   /// number.
   virtual double calcStepSize(double cfl) const;

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
protected:
   /// free-stream Mach number
   double mach_fs;
   /// free-stream angle of attack
   double aoa_fs;
   /// index of dimension corresponding to nose to tail axis
   int iroll;
   /// index of "vertical" dimension in body frame
   int ipitch;

   /// Initialize `res` and either `mass` or `nonlinear_mass`
   virtual void constructForms() override;

   /// Add Domain Integrator to the mass operators
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

   /// Create `output` based on `options` and add approporiate integrators
   virtual void addOutputs() override;

   /// Return the number of state variables
   virtual int getNumState() override {return dim+2; }
};

} // namespace mach

#endif
