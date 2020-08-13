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
    xdouble nu_tilde = q[dim+2];
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
    xdouble nu_tilde = q[dim+2];
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
    xdouble nu_tilde = q[dim+2];
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
    xdouble nu_tilde = q[dim+2];
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
    xdouble nu_tilde = q[dim+2];
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
    xdouble nu_tilde = q[dim+2];
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
    xdouble chi_d = q[dim+2]/d;
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
    xdouble nu_tilde = q[dim+2];
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
    xdouble chi_d = q[dim+2]/d;
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
    xdouble nu_tilde = q[dim+2];
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

#if 0
// Compute vorticity on an SBP element, needed for SA model terms
/// \param[in] q - the state over the element
/// \param[in] sbp - the sbp element whose shape functions we want
/// \param[in] Trans - defines the reference to physical element mapping
/// \param[out] curl - the curl of the velocity field at each node/int point
/// \tparam xdouble - typically `double` or `adept::adouble`
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
template <int dim>
void calcVorticitySBP(const mfem::DenseMatrix &q, const mfem::FiniteElement &fe, 
                      mfem::ElementTransformation &Trans, mfem::DenseMatrix curl)
{
   using namespace mfem;
   const SBPFiniteElement &sbp = dynamic_cast<const SBPFiniteElement &>(fe);
   int num_nodes = sbp.GetDof();
   DenseMatrix dq(q.Height(), q.Width()); //contains state derivatives in reference space
   DenseMatrix dxi(dim*q.Height(), dim); //contains velocity derivatives in reference space
   dq = 0.0;
   for(int di = 0; di < dim; di++)
   {
      sbp.multWeakOperator(di, q, dq);
      //need to scale with 1/H, probably when looping over nodes
      dxi.CopyMN(dq, q.Height(), dim, 1, dim+1, di*q.Height(), 0);
   }

   DenseMatrix dx(dim, dim); //contains velocity derivatives in absolute space
   DenseMatrix dxin(dim, dim); //contains velocity derivatives in reference space for a node
   Vector dxrow(dim); Vector curln(dim);
   for (int i = 0; i < num_nodes; ++i)
   {
      // get the Jacobian
      IntegrationPoint &node = sbp.GetNodes().IntPoint(i);
      Trans.SetIntPoint(&node);

      // store nodal derivatives in a single matrix
      for(int di = 0; di < dim; di++)
      {
         dxi.GetRow(di*q.Height() + i, dxrow);
         dxin.SetRow(di, dxrow);
      }

      // compute absolute derivatives
      MultAtB(Trans.InverseJacobian(), dxin, dx);
   
      // compute curl at node and append
      if(dim == 2)
      {
         curl(i, 0) = 0;
         curl(i, 1) = 0;
         curl(i, 2) = dx(1,0) - dx(0,1);
      }
      if(dim == 3)
      {
         curl(i, 0) = dx(2,1) - dx(1,2);
         curl(i, 1) = dx(0,2) - dx(2,0);
         curl(i, 2) = dx(1,0) - dx(0,1);
      }
   }
}

// Compute gradient for the turbulence variable on an SBP element, 
// needed for SA model terms
/// \param[in] q - the state over the element
/// \param[in] sbp - the sbp element whose shape functions we want
/// \param[in] Trans - defines the reference to physical element mapping
/// \param[out] grad - the gradient of the turbulence variable at each node
/// \tparam xdouble - typically `double` or `adept::adouble`
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
template <typename xdouble, int dim>
void calcGradSBP(const mfem::DenseMatrix &q, const mfem::FiniteElement &fe, 
                      mfem::ElementTransformation &Trans, mfem::DenseMatrix grad)
{
   using namespace mfem;
   const SBPFiniteElement &sbp = dynamic_cast<const SBPFiniteElement &>(fe);
   int num_nodes = sbp.GetDof();
   DenseMatrix dq(q.Height(), q.Width()); //contains state derivatives in reference space
   DenseMatrix dnu(q.Height(), dim); //contains turb variable derivatives in reference space
   dq = 0.0;
   for(int di = 0; di < dim; di++)
   {
      sbp.multWeakOperator(di, q, dq);
      //need to scale with 1/H
      dnu.SetCol(di, dq.GetColumn(dim+2));
   }

   Vector dnurow(dim); Vector gradn(dim);
   for (int i = 0; i < num_nodes; ++i)
   {
      // get the Jacobian
      IntegrationPoint &node = sbp.GetNodes().IntPoint(i);
      Trans.SetIntPoint(&node);

      // store nodal grad in a vector
      dnu.GetRow(i, dnurow);

      // compute absolute derivatives
      Trans.InverseJacobian().MultTranspose(dnurow, gradn);
   
      // append result
      grad.SetRow(i, gradn);
   }
}
#endif

// Compute vorticity at a point on an element, needed for SA model terms
/// \param[in] Dq - the state gradient Dq
/// \param[in] jac_inv - inverse jacobian at node
/// \param[out] curl - the curl of the velocity field at the node
/// \tparam xdouble - typically `double` or `adept::adouble`
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// NOTE: Need inverse jacobian! Check EvalJacobian
template <typename xdouble, int dim>
void calcVorticity(const xdouble *Dq, const xdouble *jac_inv, 
                        xdouble *curl)
{
    xdouble DqJ[dim*(dim+3)]; 

    // compute absolute derivatives
    for(int di = 0; di < dim+3; di++)
    {
        for(int d = 0; d < dim; d++)
        {
            xdouble work = 0;
            for(int k = 0; k < dim; k++)
            {
                work += Dq[di+k*(dim+3)]*jac_inv[k+d*dim];
            }
            DqJ[di+d*(dim+3)] = work;
        }
    }

    // compute curl at node and append
    if(dim == 2)
    {
        curl[0] = 0;
        curl[1] = 0;
        curl[2] = DqJ[2] - DqJ[1 + dim+3];
    }
    if(dim == 3)
    {
        curl[0] = DqJ[3 + dim+3] - DqJ[2 + 2*(dim+3)];
        curl[1] = DqJ[1 + 2*(dim+3)] - DqJ[3];
        curl[2] = DqJ[2] - DqJ[1 + dim+3];
    }
}

// Compute vorticity jacobian w.r.t. state on an SBP element, needed for SA model terms
/// \param[in] stack - adept stack
/// \param[in] q - the state over the element
/// \param[in] sbp - the sbp element whose shape functions we want
/// \param[in] Trans - defines the reference to physical element mapping
/// \param[out] curl - the curl of the velocity field at each node/int point
/// \tparam xdouble - typically `double` or `adept::adouble`
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
template <int dim>
void calcVorticityJacDw(adept::Stack &stack, const double *Dq, const double *jac_inv, 
                        std::vector<mfem::DenseMatrix> &jac_curl)
{
    // vector of active input variables
    std::vector<adouble> Dq_a(dim*(dim+3));
    std::vector<adouble> jac_a(dim*(dim+3));
    // initialize adouble inputs
    adept::set_values(Dq_a.data(), dim*(dim+3), Dq);
    adept::set_values(jac_a.data(), dim*dim, jac_inv);
    // start recording
    stack.new_recording();
    // create vector of active output variables
    std::vector<adouble> curl_a(3);
    // run algorithm
    calcVorticity<adouble, dim>(Dq_a.data(), jac_a.data(), curl_a.data());
    // identify independent and dependent variables
    stack.independent(Dq_a.data(), Dq_a.size());
    stack.dependent(curl_a.data(), curl_a.size());
    // compute and store jacobian in jac_inv
    mfem::Vector work(dim*(dim+3)*3);
    stack.jacobian_reverse(work.GetData());
    for (int i = 0; i < dim; ++i)
    {
        jac_curl[i] = (work.GetData() + i*(dim+3)*3);
    }
}

// Compute gradient for the turbulence variable on an element node, 
// needed for SA model terms
/// \param[in] q - the derivative of the state at the node
/// \param[in] jac_inv - transformation jacobian at node
/// \param[out] grad - the gradient of the turbulence variable at each node
/// \tparam xdouble - typically `double` or `adept::adouble`
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
template <typename xdouble, int dim>
void calcGrad(const xdouble *Dq, const xdouble *jac_inv, 
                        xdouble *grad)
{
   xdouble DqJ[dim*(dim+3)]; 

    // compute absolute derivatives
    for(int di = 0; di < dim+3; di++)
    {
        for(int d = 0; d < dim; d++)
        {
            xdouble work = 0;
            for(int k = 0; k < dim; k++)
            {
                work += Dq[di+k*(dim+3)]*jac_inv[k+d*dim];
            }
            DqJ[di+d*(dim+3)] = work;
        }
    }

    for (int i = 0; i < dim; ++i)
    {
       grad[i] = DqJ[dim+2 + i*(dim+3)];
    }
}

// Compute grad jacobian w.r.t. state on an SBP element, needed for SA model terms
/// \param[in] stack - adept stack
/// \param[in] q - the state over the element
/// \param[in] sbp - the sbp element whose shape functions we want
/// \param[in] Trans - defines the reference to physical element mapping
/// \param[out] curl - the curl of the velocity field at each node/int point
/// \tparam xdouble - typically `double` or `adept::adouble`
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
template <int dim>
void calcGradJacDw(adept::Stack &stack, const double *Dq, const double *jac_inv, 
                        std::vector<mfem::DenseMatrix> &jac_grad)
{
    // vector of active input variables
   std::vector<adouble> Dq_a(dim*(dim+3));
    std::vector<adouble> jac_a(dim*(dim+3));
   // initialize adouble inputs
   adept::set_values(Dq_a.data(), dim*(dim+3), Dq);
    adept::set_values(jac_a.data(), dim*dim, jac_inv);
   // start recording
   stack.new_recording();
   // create vector of active output variables
   std::vector<adouble> grad_a(dim);
   // run algorithm
   calcGrad<adouble, dim>(Dq_a.data(), jac_a.data(), grad_a.data());
   // identify independent and dependent variables
   stack.independent(Dq_a.data(), Dq_a.size());
   stack.dependent(grad_a.data(), grad_a.size());
   // compute and store jacobian in jac_inv
    mfem::Vector work(dim*(dim+3)*dim);
   stack.jacobian_reverse(work.GetData());
    for (int i = 0; i < dim; ++i)
    {
        jac_grad[i] = (work.GetData() + i*(dim+3)*dim);
    }
}

} // namespace mach

#endif
