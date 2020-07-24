#ifndef MACH_RANS
#define MACH_RANS

#include "mfem.hpp"

#include "navier_stokes.hpp"

namespace mach
{

/// Solver for Reynolds-Averaged Navier-Stokes flows
/// dim - number of spatial dimensions (1, 2, or 3)
/// entvar - if true, the entropy variables are used in the integrators
template <int dim, bool entvar = false>

class RANavierStokesSolver : public  NavierStokesSolver<dim, entvar>
{
public:
    /// Sets `q_in` to the inflow conservative + SA variables
    void getRANSInflowState(mfem::Vector &q_in);

    /// Sets `q_out` to the outflow conservative + SA variables
    void getRANSOutflowState(mfem::Vector &q_out);

    /// Sets `q_ref` to the free-stream conservative + SA variables
    void getFreeStreamState(mfem::Vector &q_ref);

    /// convert conservative variables to entropy variables
    /// \param[in/out] state - the conservative/entropy variables
    virtual void convertToEntvar(mfem::Vector &state) override
    {
        throw MachException("Entropy variables not implemented!!!");
    }

protected:
    /// Class constructor.
    /// \param[in] opt_file_name - file where options are stored
    /// \param[in] smesh - if provided, defines the mesh for the problem
    /// \param[in] dim - number of dimensions
    RANavierStokesSolver(const std::string &opt_file_name,
                         std::unique_ptr<mfem::Mesh> smesh = nullptr);

    /// Add volume/domain integrators to `res` based on `options`
    /// \param[in] alpha - scales the data; used to move terms to rhs or lhs
    /// \note This function calls NavierStokes::addVolumeIntegrators() first
    virtual void addResVolumeIntegrators(double alpha);

    /// Add boundary-face integrators to `res` based on `options`
    /// \param[in] alpha - scales the data; used to move terms to rhs or lhs
    /// \note This function calls EulerSolver::addBoundaryIntegrators() first
    virtual void addResBoundaryIntegrators(double alpha);

    /// Return the number of state variables
    virtual int getNumState() override {return dim+3; }

    friend SolverPtr createSolver<RANavierStokesSolver<dim, entvar>>(
       const std::string &opt_file_name,
       std::unique_ptr<mfem::Mesh> smesh);

    /// SA constants vector
    mfem::Vector sacs;
    /// free-stream SA viscosity ratio (nu_tilde/nu_material)
    double chi_fs;
    /// material dynamic viscosity
    double mu;
};

} //namespace mach

#endif