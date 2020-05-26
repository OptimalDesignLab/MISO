/// Defines data that can be used by test_euler_fluxes.hpp and test_euler.hpp
#ifndef EULER_TEST_DATA
#define EULER_TEST_DATA

#include <limits>
#include <random>
#include "mfem.hpp"

namespace euler_data
{

/// Used for floating point checks when the benchmark value is zero
const double abs_tol = std::numeric_limits<double>::epsilon()*100;

// Define a random (but physical) state for the following tests
const double rho = 0.9856566615165173;
const double rhoe = 2.061597236955558;
const double rhou[3] = {0.09595562550099601, -0.030658751626551423, -0.13471469906596886};

// Define a second ("right") state for the tests
const double rho2 = 0.8575252486261279;
const double rhoe2 = 2.266357718749846;
const double rhou2[3] = {0.020099729730903737, -0.2702434209304979, -0.004256150573245826};

// Define a random direction vector
const double dir[3] = {0.6541305612927484, -0.0016604759052086802, -0.21763228465741322}; 

const double press_check[3] = {0.8227706007961364, 0.8225798733170867, 0.8188974449720051};
const double spect_check[3] = {0.7708202616595441, 0.7707922224516813, 0.8369733021138251};
const double entropy_check[3] = {0.4308595411133724, 0.4314308230805257, 0.44248676664229974};

// Define the Euler flux values for checking; The first 3 entries are for the
// 1D flux, the next 4 for the 2D flux, and the last 5 for the 3D flux
extern double flux_check[12];

// Define the Ismail-Roe flux values for checking; note that direction dim has 
// dim fluxes to check, each with dim+2 values (so these arrays have dim*(dim+2)
// entries)
extern double fluxIR_1D_check[3];
extern double fluxIR_2D_check[8];
extern double fluxIR_3D_check[15];

// Define the flux returned by calcBoundaryFlux; note, only the 2d version is
// tested so far
extern const double flux_bnd_check[4];

// Define the entropy variables for checking; The first 3 entries are for the 
// 1D variables, the next 4 for the 2D variables, and the last 5 for the 3D 
// variables
extern double entvar_check[12];

// Define products between dq/dw, evaluated at q, with vector qR.  The first 3
// entries are for the 1D product, the next 4 for the 2D product, and the last
// 5 for the 3D
extern double dqdw_prod_check[12];

// Use this for finite-difference direction-derivative checks
extern double vec_pert[9];

// Use this for LPS apply scaling jacobian checks
extern double adjJ_data[9];

// Use this for spatial derivatives of entropy-variables
extern double delw_data[15];

/// Returns a perturbed version of the baseline flow state
/// \param[in] x - coordinates (not used)
/// \param[out] u - pertrubed state variable
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \tparam entvar - if true, returns entropy variables
template <int dim, bool entvar = false>
void randBaselinePert(const mfem::Vector &x, mfem::Vector &u);

/// Returns a random state with entries uniformly distributed in [-1,1]
/// \param[in] x - coordinates (not used)
/// \param[out] u - rand state variable
void randState(const mfem::Vector &x, mfem::Vector &u);

<<<<<<< HEAD
void FreeStreamState2D(mfem::Vector &q_ref, double mach_fs, double aoa_fs);
=======
/// Returns a perturbed version of the baseline temperature state
/// \param[in] x - coordinates (not used)
double randBaselinePert(const mfem::Vector &x);

/// Returns a random state with entries uniformly distributed in [-1,1]
/// \param[in] x - coordinates (not used)
double randState(const mfem::Vector &x);
>>>>>>> euler

} // euler_data namespace

#endif