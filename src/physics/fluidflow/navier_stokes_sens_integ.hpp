#ifndef MACH_NAVIER_STOKES_SENS_INTEG
#define MACH_NAVIER_STOKES_SENS_INTEG

#include "adept.h"
#include "mfem.hpp"

#include "mesh_sens_integ.hpp"
#include "navier_stokes_fluxes.hpp"

using adept::adouble;

namespace mach
{

/// Integrator for the mesh sensitivity of the Ismail-Roe domain integrator
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \tparam entvar - if true, the state variables are the entropy variables
/// \note This derived class uses the CRTP
template <int dim, bool entvar = false>
class ESViscousMeshSensIntegrator : public SymmetricViscousMeshSensIntegrator<
                                ESViscousMeshSensIntegrator<dim, entvar>>
{
public:
    /// Construct an integrator for the Ismail-Roe flux over domains
    /// \param[in] a - factor, usually used to move terms to rhs
    ESViscousMeshSensIntegrator(const mfem::GridFunction &state_vec,
                               const mfem::GridFunction &adjoint_vec,
                                double Re_num, double Pr_num,
                                double vis = -1.0,  double a = 1.0)
       : SymmetricViscousMeshSensIntegrator<ESViscousMeshSensIntegrator<dim, entvar>>(
             state_vec, adjoint_vec, dim+2, a),
         Re(Re_num), Pr(Pr_num), mu(vis) {}

    void convertVars(const mfem::Vector &q, mfem::Vector &w)
    {
        calcEntropyVars<double, dim>(q.GetData(), w.GetData());
    }

    void applyScaling(int d, const mfem::Vector &x, const mfem::Vector &q,
               const mfem::DenseMatrix &Dw, mfem::Vector &CDw)
    {
        double mu_Re = mu;
        if (mu < 0.0)
        {
            mu_Re = calcSutherlandViscosity<double, dim>(q.GetData());
        }
        mu_Re /= Re;
        // applyViscousScaling<double, dim>(d, mu_Re, Pr, q.GetData(), Dw.GetData(),
        //                                CDw.GetData());
        for(int i = 0; i < CDw.Size(); i++)
        {
            CDw(i) = mu_Re*Dw(i,d);
        }
    }
private:
   /// Reynolds number
   double Re;
   /// Prandtl number
   double Pr;
   /// nondimensional dynamic viscosity
   double mu;
};

#include "navier_stokes_sens_integ_def.hpp"

} // namespace mach

#endif