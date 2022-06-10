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

   /// Sets `q_ref` to the boundar condition for the sod-shock problem
   //void getSodShockBC(mfem::Vector &q_ref);

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
   /// A virtual function convert the conservative variable to entropy variables
   /// \param[in/out] state - state vector
   virtual void convertToEntvar(mfem::Vector &state);
   virtual void convertToConserv(mfem::Vector &state);
   virtual void convertToConservCent(mfem::Vector &state);
   virtual void PrintSodShock(const std::string &file_name);
   virtual void PrintSodShockCenter(const std::string &file_name);
   virtual double computeMass();
   virtual void checkConversion(void (*u_exact)(const mfem::Vector &, mfem::Vector &));
   virtual void conToEntropyVars(const mfem::Vector &entropy, mfem::Vector &conserv);
   virtual void conToConservVars(const mfem::Vector &conserv, mfem::Vector &entropy);

protected:
   /// free-stream Mach number
   double mach_fs;
   /// free-stream angle of attack
   double aoa_fs;
   /// index of dimension corresponding to nose to tail axis
   int iroll;
   /// index of "vertical" dimension in body frame
   int ipitch;
   /// The nonlinaer integrator that computs the nonlinear form mass matrix
   std::unique_ptr<mfem::NonlinearFormIntegrator> mass_integ;

   /// Add volume integrators to `res` based on `options`
   /// \param[in] alpha - scales the data; used to move terms to rhs or lhs
   virtual void addVolumeIntegrators(double alpha);

   /// Add boundary-face integrators to `res` based on `options`
   /// \param[in] alpha - scales the data; used to move terms to rhs or lhs
   virtual void addBoundaryIntegrators(double alpha);

   /// Add interior-face integrators to `res` based on `options`
   /// \param[in] alpha - scales the data; used to move terms to rhs or lhs
   virtual void addInterfaceIntegrators(double alpha);

   /// Create `output` based on `options` and add approporiate integrators
   virtual void addOutputs();

   /// Return the number of state variables
   virtual int getNumState() {return dim+2; }

   /// Add Domain Integrator to the nonlinear form mass matrix
   /// \param[in] alpha - scales the data; used to ove terems to rhs or lhs
   virtual void addMassIntegrator(double alpha);

   /// Update the mass integrator
   /// \param[in] dt - the numerical time step
   //virtual void updateNonlinearMass(int ti, double dt, double alpha);

   virtual void addMassIntegrators(double alpha);
};

} // namespace mach

#endif
