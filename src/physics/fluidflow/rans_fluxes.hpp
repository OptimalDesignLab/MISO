/// Functions related to RANS/SA equations
#ifndef MACH_RANS_FLUXES
#define MACH_RANS_FLUXES

#include <algorithm> // std::max

#include "utils.hpp"
#include "navier_stokes_fluxes.hpp"

/// Reference
/// sacs[0] = cb1
/// sacs[1] = cb2
/// sacs[2] = sigma
/// sacs[3] = kappa
/// sacs[4] = cw2
/// sacs[5] = cw3
/// sacs[6] = cv1

/// Not including trip terms for now

/// sacs[7] = ct3
/// sacs[8] = ct4
/// sacs[9] = rlim
/// sacs[10] = cn1

namespace mach
{

/// For constants related to the RANS/SA equations
namespace rans
{

} // namespace rans

/// Returns the laminar suppression term in SA
/// \param[in] q - state used to define the laminar suppression
/// \param[in] mu - **nondimensionalized** viscosity
/// \param[in] sacs - Spalart-Allmaras constants
/// \returns ft2 - laminar suppression term
/// \tparam xdouble - typically `double` or `adept::adouble`
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
template <typename xdouble, int dim>
xdouble calcSALaminarSuppression(const xdouble *q, const xdouble mu, 
                                    const xdouble *sacs)
{
    xdouble ct3 = sacs[7];
    xdouble ct4 = sacs[8];
    xdouble nu_tilde = q[dim+3];
    xdouble nu_mat = mu/q[0];
    xdouble chi = nu_tilde/nu_mat;
    xdouble ft2 = ct3*exp(-ct4*chi*chi);
    return ft2;
}



/// Returns the modified vorticity in SA
/// \param[in] q - state used to define the destruction
/// \param[in] mu - **nondimensionalized** viscosity
/// \param[in] sacs - Spalart-Allmaras constants
/// \returns St - modified vorticity
/// \tparam xdouble - typically `double` or `adept::adouble`
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
template <typename xdouble, int dim>
xdouble calcSAModifiedVorticity(const xdouble *q, const xdouble S, 
                                 const xdouble mu, const xdouble d, 
                                const xdouble *sacs)
{
    xdouble nu_tilde = q[dim+3];
    xdouble kappa = sacs[3];
    xdouble fv2 = calcSAProductionCoefficient(q, mu, sacs);
    xdouble St = S + nu_tilde*fv2/(kappa*kappa*d*d);
    return St;
}

/// Returns the turbulent viscosity coefficient in SA
/// \param[in] q - state used to define the destruction
/// \param[in] mu - **nondimensionalized** viscosity
/// \param[in] sacs - Spalart-Allmaras constants
/// \returns fn - turbulent viscosity coefficient
/// \tparam xdouble - typically `double` or `adept::adouble`
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
template <typename xdouble, int dim>
xdouble calcSACoefficient(const xdouble *q, const xdouble mu,
                                const xdouble *sacs)
{
    xdouble cv1 = sacs[6];
    xdouble nu_tilde = q[dim+3];
    xdouble nu_mat = mu/q[0];
    xdouble chi = nu_tilde/nu_mat;
    xdouble fv1 = chi*chi*chi/(cv1*cv1*cv1 + chi*chi*chi);
    return fv1;
}

/// Returns the production coefficient in SA
/// \param[in] q - state used to define the destruction
/// \param[in] mu - **nondimensionalized** viscosity
/// \param[in] sacs - Spalart-Allmaras constants
/// \returns fv2 - production coefficient
/// \tparam xdouble - typically `double` or `adept::adouble`
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
template <typename xdouble, int dim>
xdouble calcSAProductionCoefficient(const xdouble *q, const xdouble mu,
                                const xdouble *sacs)
{
    xdouble nu_tilde = q[dim+3];
    xdouble nu_mat = mu/q[0];
    xdouble chi = nu_tilde/nu_mat;
    xdouble fv1 = calcSACoefficient(q, mu, sacs);
    xdouble fv2 = 1 - chi/(1 + chi*fv1);
    return fv2;
}

/// Returns the destruction coefficient in SA
/// \param[in] q - state used to define the destruction
/// \param[in] mu - **nondimensionalized** viscosity
/// \param[in] sacs - Spalart-Allmaras constants
/// \returns fw - destruction coefficient
/// \tparam xdouble - typically `double` or `adept::adouble`
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
template <typename xdouble, int dim>
xdouble calcSADestructionCoefficient(const xdouble *q, 
                                 const xdouble mu, const xdouble d, 
                                 const xdouble S, const xdouble *sacs)
{
    xdouble nu_tilde = q[dim+3];
    xdouble kappa = sacs[3];
    xdouble cw2 = sacs[4];
    xdouble cw3 = sacs[5];
    xdouble rlim = sacs[9];

    xdouble St = calcSAModifiedVorticity(q, S, mu, d, sacs);
    xdouble work = nu_tilde/(St*kappa*kappa*d*d);
    xdouble r = std::min(work, rlim);
    xdouble g = r + cw2*(std::pow(r, 6) - r);
    xdouble work2 = (1 + std::pow(cw3, 6))/
                     (std::pow(g, 6) + std::pow(cw3, 6));
    xdouble fw = g*std::pow(work2, (1/6));
    return fw;
}

/// Returns the production term in SA
/// \param[in] q - state used to define the production
/// \param[in] S - vorticity magnitude (how to compute in h_grad space?)
/// \param[in] sacs - Spalart-Allmaras constants
/// \returns P - production term
/// \tparam xdouble - typically `double` or `adept::adouble`
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
template <typename xdouble, int dim>
xdouble calcSAProduction(const xdouble *q,
                                 const xdouble mu, const xdouble d, 
                                 const xdouble S, const xdouble *sacs)
{
    xdouble cb1 = sacs[0];
    xdouble nu_tilde = q[dim+3];
    xdouble St = calcSAModifiedVorticity(q, S, mu, d, sacs);
    xdouble ft2 = calcSALaminarSuppression(q, mu, sacs);
    xdouble P = cb1*(1-ft2)*St*nu_tilde;
    return P;
}

/// Returns the destruction term in SA
/// \param[in] q - state used to define the destruction
/// \param[in] d - wall distance (precomputed from gridfunction)
/// \param[in] sacs - Spalart-Allmaras constants
/// \returns D - destruction term
/// \tparam xdouble - typically `double` or `adept::adouble`
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
template <typename xdouble, int dim>
xdouble calcSADestruction(const xdouble *q,
                                 const xdouble mu, const xdouble d, 
                                 const xdouble S, const xdouble *sacs)
{
    xdouble cb1 = sacs[0];
    xdouble cb2 = sacs[1];
    xdouble sigma = sacs[2];
    xdouble kappa = sacs[3];
    xdouble chi_d = q[dim+3]/d;
    xdouble cw1 = cb1/(kappa*kappa) + (1+cb2)/sigma;
    xdouble fw = calcSADestructionCoefficient(q, mu, d, S, sacs);
    xdouble ft2 = calcSALaminarSuppression(q, mu, sacs);
    xdouble D = (cw1*fw - (cb1/(kappa*kappa))*ft2)*chi_d*chi_d;
    return D;
}

/// Returns the production term in negative SA
/// \param[in] q - state used to define the production
/// \param[in] S - vorticity magnitude (how to compute in h_grad space?)
/// \param[in] sacs - Spalart-Allmaras constants
/// \returns Pn - production term
/// \tparam xdouble - typically `double` or `adept::adouble`
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
template <typename xdouble, int dim>
xdouble calcSANegativeProduction(const xdouble *q, const xdouble S,
                             const xdouble *sacs)
{
    xdouble cb1 = sacs[0];
    xdouble ct3 = sacs[7];
    xdouble nu_tilde = q[dim+3];
    xdouble Pn = cb1*(1-ct3)*S*nu_tilde;
    return Pn;
}

/// Returns the destruction term in negative SA
/// \param[in] q - state used to define the destruction
/// \param[in] d - wall distance (precomputed from gridfunction)
/// \param[in] sacs - Spalart-Allmaras constants
/// \returns Dn - destruction term
/// \tparam xdouble - typically `double` or `adept::adouble`
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
template <typename xdouble, int dim>
xdouble calcSANegativeDestruction(const xdouble *q, const xdouble d,
                             const xdouble *sacs)
{
    xdouble cb1 = sacs[0];
    xdouble cb2 = sacs[1];
    xdouble sigma = sacs[2];
    xdouble kappa = sacs[3];
    xdouble chi_d = q[dim+3]/d;
    xdouble cw1 = cb1/(kappa*kappa) + (1+cb2)/sigma;
    xdouble Dn = -cw1*chi_d*chi_d;
    return Dn;
}

/// Returns the turbulent viscosity coefficient in negative SA
/// \param[in] q - state used to define the destruction
/// \param[in] mu - **nondimensionalized** viscosity
/// \param[in] sacs - Spalart-Allmaras constants
/// \returns fn - negative turbulent viscosity coefficient
/// \tparam xdouble - typically `double` or `adept::adouble`
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
template <typename xdouble, int dim>
xdouble calcSANegativeCoefficient(const xdouble *q, const xdouble mu,
                                const xdouble *sacs)
{
    xdouble cn1 = sacs[10];
    xdouble nu_tilde = q[dim+3];
    xdouble nu_mat = mu/q[0];
    xdouble chi = nu_tilde/nu_mat;
    xdouble fn = (cn1 + chi*chi*chi)/(cn1 - chi*chi*chi);
    return fn;
}

/// Returns the "source" term in SA
/// \param[in] q - state used to define the destruction
/// \param[in] dir - turbulent viscosity gradient
/// \param[in] sacs - Spalart-Allmaras constants
/// \returns Sr - "source" term
/// \tparam xdouble - typically `double` or `adept::adouble`
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
template <typename xdouble, int dim>
xdouble calcSASource(const xdouble *q, const xdouble *dir,
                             const xdouble *sacs)
{
    xdouble cb2 = sacs[1];
    xdouble sigma = sacs[2];
    xdouble Sr = (cb2/sigma)*dot<xdouble, dim>(dir, dir);
    return Sr;
}

} // namespace mach

#endif
