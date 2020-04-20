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

protected:
   /// free-stream Mach number
   double mach_fs;
   /// free-stream angle of attack
   double aoa_fs;
   /// index of dimension corresponding to nose to tail axis
   int iroll;
   /// index of "vertical" dimension in body frame
   int ipitch;

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
};

} // namespace mach

#endif
