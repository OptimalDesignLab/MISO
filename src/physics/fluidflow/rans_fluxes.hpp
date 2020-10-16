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

///NOTE: These are related only to preventing negative St
/// sacs[11] = cv2
/// sacs[12] = cv3

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
    xdouble nu_tilde = q[dim+2]/q[0];
    xdouble nu_mat = mu/q[0];
    xdouble chi = nu_tilde/nu_mat;
    xdouble ft2 = ct3*exp(-ct4*chi*chi);
    return ft2;
}

/// Returns the turbulent viscosity coefficient in SA
/// \param[in] q - state used to define the destruction
/// \param[in] mu - **nondimensionalized** viscosity
/// \param[in] sacs - Spalart-Allmaras constants
/// \returns fv1 - turbulent viscosity coefficient
/// \tparam xdouble - typically `double` or `adept::adouble`
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
template <typename xdouble, int dim>
xdouble calcSACoefficient(const xdouble *q, const xdouble mu,
                                const xdouble *sacs)
{
    xdouble cv1 = sacs[6];
    xdouble nu_tilde = q[dim+2]/q[0];
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
    xdouble nu_tilde = q[dim+2]/q[0];
    xdouble nu_mat = mu/q[0];
    xdouble chi = nu_tilde/nu_mat;
    xdouble fv1 = calcSACoefficient<xdouble, dim>(q, mu, sacs);
    xdouble fv2 = 1.0 - chi/(1.0 + chi*fv1);
    return fv2;
}

/// Returns the modified vorticity in SA
/// \param[in] q - state used to define the destruction
/// \param[in] S - vorticity magnitude
/// \param[in] mu - **nondimensionalized** viscosity
/// \param[in] d - wall distance value
/// \param[in] Re - Reynold's number, if needed
/// \param[in] sacs - Spalart-Allmaras constants
/// \returns St - modified vorticity
/// \tparam xdouble - typically `double` or `adept::adouble`
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
template <typename xdouble, int dim>
xdouble calcSAModifiedVorticity(const xdouble *q, const xdouble S, 
                                 const xdouble mu, const xdouble d, 
                                const xdouble Re, const xdouble *sacs)
{
    xdouble nu_tilde = q[dim+2]/q[0];
    xdouble kappa = sacs[3];
    xdouble cv2 = sacs[11];
    xdouble cv3 = sacs[12];
    xdouble fv2 = calcSAProductionCoefficient<xdouble, dim>(q, mu, sacs);
    xdouble work = nu_tilde*fv2/(kappa*kappa*d*d);
    xdouble St;
    if (work < -cv2*S)
        St = S + (S*(cv2*cv2*S + cv3*work))/((cv3-2*cv2)*S - work); 
    else 
        St = S + work;
    return St;
}

/// Returns the destruction coefficient in SA
/// \param[in] q - state used to define the destruction
/// \param[in] mu - **nondimensionalized** viscosity
/// \param[in] d - wall distance value
/// \param[in] S - vorticity magnitude
/// \param[in] Re - Reynold's number, if needed
/// \param[in] sacs - Spalart-Allmaras constants
/// \returns fw - destruction coefficient
/// \tparam xdouble - typically `double` or `adept::adouble`
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
template <typename xdouble, int dim>
xdouble calcSADestructionCoefficient(const xdouble *q, 
                                 const xdouble mu, const xdouble d, 
                                 const xdouble S, const xdouble Re,
                                 const xdouble *sacs)
{
    using std::min;
    using adept::min;
    using std::pow;
    using adept::pow;
    xdouble nu_tilde = q[dim+2]/q[0];
    xdouble kappa = sacs[3];
    xdouble cw2 = sacs[4];
    xdouble cw3 = sacs[5];
    xdouble rlim = sacs[9];

    xdouble St = calcSAModifiedVorticity<xdouble, dim>(q, S, mu, d, Re, sacs);
    //xdouble fv2 = calcSAProductionCoefficient<xdouble, dim>(q, mu, sacs);

    xdouble work = nu_tilde/(Re*St*kappa*kappa*d*d);
    
    xdouble r = min(work, rlim);
    xdouble r6 = r*r*r*r*r*r;
    xdouble g = r + cw2*(/*pow(r, 6.0)*/ r6 - r);
    xdouble g6 = g*g*g*g*g*g;
    xdouble cw36 = cw3*cw3*cw3*cw3*cw3*cw3;
    xdouble work2 = (1.0 + /*pow(cw3, 6.0)*/ cw36)/
                     (/*pow(g, 6.0)*/ g6 + /*pow(cw3, 6.0)*/ cw36);
    xdouble fw = g*pow(work2, (1.0/6.0));
    return fw;
}

/// Returns the production term in SA
/// \param[in] q - state used to define the production
/// \param[in] d - wall distance value
/// \param[in] S - vorticity magnitude (how to compute in h_grad space?)
/// \param[in] Re - Reynold's number, if needed
/// \param[in] sacs - Spalart-Allmaras constants
/// \returns P - production term
/// \tparam xdouble - typically `double` or `adept::adouble`
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
template <typename xdouble, int dim>
xdouble calcSAProduction(const xdouble *q,
                                 const xdouble mu, const xdouble d, 
                                 const xdouble S, const xdouble Re,
                                 const xdouble *sacs)
{
    xdouble cb1 = sacs[0];
    xdouble nu_tilde = q[dim+2]/q[0];
    xdouble St = calcSAModifiedVorticity<xdouble, dim>(q, S, mu, d, Re, sacs);
    xdouble ft2 = calcSALaminarSuppression<xdouble, dim>(q, mu, sacs);
    xdouble P = cb1*(1.0-ft2)*St*nu_tilde;
    return q[0]*P;
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
                                 const xdouble S, const xdouble Re,
                                 const xdouble *sacs)
{
    xdouble cb1 = sacs[0];
    xdouble cb2 = sacs[1];
    xdouble sigma = sacs[2];
    xdouble kappa = sacs[3];
    xdouble chi_d = q[dim+2]/(d*q[0]);
    xdouble cw1 = cb1/(kappa*kappa) + (1+cb2)/sigma;
    xdouble fw = calcSADestructionCoefficient<xdouble, dim>(q, mu, d, S, Re, sacs);
    xdouble ft2 = calcSALaminarSuppression<xdouble, dim>(q, mu, sacs);
    xdouble D = (cw1*fw - (cb1/(kappa*kappa))*ft2)*chi_d*chi_d;
    return -q[0]*D;
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
    xdouble nu_tilde = q[dim+2]/q[0];
    xdouble Pn = cb1*(1.0-ct3)*S*nu_tilde;
    return q[0]*Pn;
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
    xdouble chi_d = q[dim+2]/(d*q[0]);
    xdouble cw1 = cb1/(kappa*kappa) + (1+cb2)/sigma;
    xdouble Dn = -cw1*chi_d*chi_d;
    return -q[0]*Dn;
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
    xdouble nu_tilde = q[dim+2]/q[0];
    xdouble nu_mat = mu/q[0];
    xdouble chi = nu_tilde/nu_mat;
    xdouble fn = (cn1 + chi*chi*chi)/(cn1 - chi*chi*chi);
    if(q[dim+2] >= 0)
        fn = 1.0;
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
    return q[0]*Sr;
}

/// Returns the viscous "source" term in SA
/// \param[in] q - state used to define the destruction
/// \param[in] dir - turbulent viscosity gradient
/// \param[in] sacs - Spalart-Allmaras constants
/// \returns Sr - "source" term
/// \tparam xdouble - typically `double` or `adept::adouble`
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
template <typename xdouble, int dim>
xdouble calcSASource2(const xdouble *q, const xdouble mu, const xdouble *dir, 
                         const xdouble *dir2, const xdouble *sacs)
{
    xdouble nu_tilde = q[dim+2]/q[0];
    xdouble sigma = sacs[2];
    xdouble fn = calcSANegativeCoefficient<xdouble, dim>(q, mu, sacs);
    xdouble Sr = (1.0/sigma)*(mu/q[0] + fn*nu_tilde)*dot<xdouble, dim>(dir, dir2);
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
                work += (Dq[di+k*(dim+3)])*jac_inv[k+d*dim];
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
    ////!!!!!!!
    //curl[2] = 1.0;// min(abs(curl[2]), 0.0);
    ////!!!!!!!
    if(dim == 3)
    {
        curl[0] = DqJ[3 + dim+3] - DqJ[2 + 2*(dim+3)];
        curl[1] = DqJ[1 + 2*(dim+3)] - DqJ[3];
        curl[2] = DqJ[2] - DqJ[1 + dim+3];
    }
}

// Compute vorticity jacobian w.r.t. state on an SBP element, needed for SA model terms
/// \param[in] stack - adept stack
/// \param[in] Dq - the state gradient Dq
/// \param[in] jac_inv - inverse jacobian at node
/// \param[out] jac_curl - the curl jacobian
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
/// \param[in] i - the state variable to take
/// \param[in] Dq - the state gradient Dq
/// \param[in] jac_inv - transformation jacobian at node
/// \param[out] grad - the gradient of the turbulence variable at each node
/// \tparam xdouble - typically `double` or `adept::adouble`
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
template <typename xdouble, int dim>
void calcGrad(const int i, const xdouble *Dq, const xdouble *jac_inv, 
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

    for (int j = 0; j < dim; ++j)
    {
       grad[j] = DqJ[i + j*(dim+3)];
    }
}

// Compute grad jacobian w.r.t. state on an SBP element, needed for SA model terms
/// \param[in] stack - adept stack
/// \param[in] i - the state variable to take
/// \param[in] Dq - the state gradient Dq
/// \param[in] jac_inv - transformation jacobian at node
/// \param[out] jac_grad - the gradient jacobian
/// \tparam xdouble - typically `double` or `adept::adouble`
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
template <int dim>
void calcGradJacDw(adept::Stack &stack, const int i, const double *Dq, const double *jac_inv, 
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
   calcGrad<adouble, dim>(i, Dq_a.data(), jac_a.data(), grad_a.data());
   // identify independent and dependent variables
   stack.independent(Dq_a.data(), Dq_a.size());
   stack.dependent(grad_a.data(), grad_a.size());
   // compute and store jacobian in jac_grad
    mfem::Vector work(dim*(dim+3)*dim);
   stack.jacobian_reverse(work.GetData());
    for (int j = 0; j < dim; ++j)
    {
        jac_grad[j] = (work.GetData() + j*(dim+3)*dim);
    }
}

/// Computes the dual-consistent term for the no-slip wall penalty (temporary SA version)
/// \param[in] dir - desired (scaled) normal vector to the wall
/// \param[in] mu - nondimensionalized dynamic viscosity 
/// \param[in] mu2 - SA transport viscosity
/// \param[in] Pr - Prandtl number
/// \param[in] q - state at the wall location
/// \param[out] fluxes - fluxes to be scaled by Dw (column major) 
/// \tparam xdouble - typically `double` or `adept::adouble`
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
template <typename xdouble, int dim>
void calcNoSlipDualFluxSA(const xdouble *dir, xdouble mu, xdouble mu2,
                    double Pr, const xdouble *q, xdouble *fluxes)
{
   int num_state = dim + 3;
   // zero out the fluxes, since applyCijMatrix accummulates 
   for (int i = 0; i < num_state*dim; ++i)
      fluxes[i] = 0.0;
   // evaluate the difference w - w_bc, where w_bc = [w[0], 0, 0, ...,w[dim+1], 0]
   xdouble dw[num_state];
   dw[0] = 0.0;
   dw[dim+1] = 0.0;
   xdouble p = pressure<xdouble,dim>(q);
   for (int d = 0; d < dim; ++d)
   {
      dw[d+1] = q[d+1]/p;
   }
   dw[dim+2] = q[dim+2];
   // loop over the normal components
   for (int d = 0; d < dim; ++d)
   {
      // loop over the derivative directions
      for (int d2 = 0; d2 < dim; ++d2)
      {
         // we "sneak" dir[d] into the computation via mu
         // !!!! we also sneak the sign into the computation here
         applyCijMatrix<xdouble, dim>(d2, d, -mu * dir[d], Pr, q, dw,
                                      fluxes + (d2 * num_state));
         fluxes[dim+2 + d2*num_state] = -mu2 * dir[d] * dw[dim+2];
      }
   }
#if 1
   // The following appears to be redundant (as far as the error goes)
   // Project out the normal component from the last variable
   xdouble nrm[dim];
   xdouble fac = 1.0 / sqrt(dot<xdouble, dim>(dir, dir));
   xdouble flux_nrm = 0.0;
   for (int d = 0; d < dim; ++d)
   {
      nrm[d] = dir[d] * fac;
      flux_nrm += fluxes[d*num_state + dim+1]*nrm[d];
   }
   for (int d = 0; d < dim; ++d)
      fluxes[d*num_state + dim+1] -= flux_nrm*nrm[d];
#endif
}

/// Compute the product \f$ \sum_{d'=0}^{dim} C_{d,d'} D_{d'} w \f$
/// Applies over conservative vars only
/// \param[in] d - desired space component of the flux
/// \param[in] mu - nondimensionalized dynamic viscosity 
/// \param[in] Pr - Prandtl number
/// \param[in] q - state used to evaluate the \f$ C_d,d'} \f$ matrices
/// \param[in] Dw - derivatives of entropy varaibles, stored column-major
/// \param[out] mat_vec - stores the resulting matrix vector product
/// \tparam xdouble - typically `double` or `adept::adouble`
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
template <typename xdouble, int dim>
void applyViscousScalingSA(int d, xdouble mu, double Pr, const xdouble *q,
                         const xdouble *Dw, xdouble *mat_vec)
{
   for (int k = 0; k < dim+3; ++k)
   {
      mat_vec[k] = 0.0;
   }
   for (int d2 = 0; d2 < dim; ++d2) {
      applyCijMatrix<xdouble, dim>(d, d2, mu, Pr, q, Dw+(d2*(dim+3)), mat_vec);
   }
}

} // namespace mach

#endif
