#ifndef MACH_EULER_INTEG_DEF_DG
#define MACH_EULER_INTEG_DEF_DG

#include "adept.h"
#include "mfem.hpp"
#include "inviscid_integ_dg.hpp"
#include "euler_fluxes.hpp"
#include "euler_integ_dg.hpp"

namespace mach
{
using adept::adouble;

template <int dim>
void EulerDGIntegrator<dim>::calcFluxJacState(const mfem::Vector &dir,
                                              const mfem::Vector &q,
                                              mfem::DenseMatrix &flux_jac)
{
   // declare vectors of active input variables
   std::vector<adouble> dir_a(dir.Size());
   std::vector<adouble> q_a(q.Size());
   // copy data from mfem::Vector
   adept::set_values(dir_a.data(), dir.Size(), dir.GetData());
   adept::set_values(q_a.data(), q.Size(), q.GetData());
   // start recording
   this->stack.new_recording();
   // the depedent variable must be declared after the recording
   std::vector<adouble> flux_a(q.Size());
   mach::calcEulerFlux<adouble, dim>(dir_a.data(), q_a.data(), flux_a.data());
   // set the independent and dependent variable
   this->stack.independent(q_a.data(), q.Size());
   this->stack.dependent(flux_a.data(), q.Size());
   // calculate the jacobian w.r.t state vaiables
   this->stack.jacobian(flux_jac.GetData());
}

template <int dim>
void EulerDGIntegrator<dim>::calcFluxJacDir(const mfem::Vector &dir,
                                            const mfem::Vector &q,
                                            mfem::DenseMatrix &flux_jac)
{
   // declare vectors of active input variables
   std::vector<adouble> dir_a(dir.Size());
   std::vector<adouble> q_a(q.Size());
   // copy data from mfem::Vector
   adept::set_values(dir_a.data(), dir.Size(), dir.GetData());
   adept::set_values(q_a.data(), q.Size(), q.GetData());
   // start recording
   this->stack.new_recording();
   // the depedent variable must be declared after the recording
   std::vector<adouble> flux_a(q.Size());
   mach::calcEulerFlux<adouble, dim>(dir_a.data(), q_a.data(), flux_a.data());
   this->stack.independent(dir_a.data(), dir.Size());
   this->stack.dependent(flux_a.data(), q.Size());
   // calculate the jacobian w.r.t state vaiables
   this->stack.jacobian(flux_jac.GetData());
}

template <int dim, bool entvar>
double DGIsentropicVortexBC<dim, entvar>::calcBndryFun(const mfem::Vector &x,
                                                       const mfem::Vector &dir,
                                                       const mfem::Vector &q)
{
   mfem::Vector flux_vec(q.Size());
   calcFlux(x, dir, q, flux_vec);
   mfem::Vector w(q.Size());
   if (entvar)
   {
      w = q;
   }
   else
   {
      calcEntropyVars<double, dim>(q.GetData(), w.GetData());
   }
   return w * flux_vec;
}

template <int dim, bool entvar>
void DGIsentropicVortexBC<dim, entvar>::calcFlux(const mfem::Vector &x,
                                                 const mfem::Vector &dir,
                                                 const mfem::Vector &q,
                                                 mfem::Vector &flux_vec)
{
   calcIsentropicVortexFlux<double, entvar>(
       x.GetData(), dir.GetData(), q.GetData(), flux_vec.GetData());
}

template <int dim, bool entvar>
void DGIsentropicVortexBC<dim, entvar>::calcFluxJacState(
    const mfem::Vector &x,
    const mfem::Vector &dir,
    const mfem::Vector &q,
    mfem::DenseMatrix &flux_jac)
{
   // create containers for active double objects for each input
   std::vector<adouble> x_a(x.Size());
   std::vector<adouble> dir_a(dir.Size());
   std::vector<adouble> q_a(q.Size());
   // initialize active double containers with data from inputs
   adept::set_values(x_a.data(), x.Size(), x.GetData());
   adept::set_values(dir_a.data(), dir.Size(), dir.GetData());
   adept::set_values(q_a.data(), q.Size(), q.GetData());
   // start new stack recording
   this->stack.new_recording();
   // create container for active double flux output
   std::vector<adouble> flux_a(q.Size());
   mach::calcIsentropicVortexFlux<adouble, entvar>(
       x_a.data(), dir_a.data(), q_a.data(), flux_a.data());
   this->stack.independent(q_a.data(), q.Size());
   this->stack.dependent(flux_a.data(), q.Size());
   this->stack.jacobian(flux_jac.GetData());
}

template <int dim, bool entvar>
void DGIsentropicVortexBC<dim, entvar>::calcFluxJacDir(
    const mfem::Vector &x,
    const mfem::Vector &dir,
    const mfem::Vector &q,
    mfem::DenseMatrix &flux_jac)
{
   // create containers for active double objects for each input
   std::vector<adouble> x_a(x.Size());
   std::vector<adouble> dir_a(dir.Size());
   std::vector<adouble> q_a(q.Size());
   // initialize active double containers with data from inputs
   adept::set_values(x_a.data(), x.Size(), x.GetData());
   adept::set_values(dir_a.data(), dir.Size(), dir.GetData());
   adept::set_values(q_a.data(), q.Size(), q.GetData());
   // start new stack recording
   this->stack.new_recording();
   // create container for active double flux output
   std::vector<adouble> flux_a(q.Size());
   mach::calcIsentropicVortexFlux<adouble, entvar>(
       x_a.data(), dir_a.data(), q_a.data(), flux_a.data());
   this->stack.independent(dir_a.data(), dir.Size());
   this->stack.dependent(flux_a.data(), q.Size());
   this->stack.jacobian(flux_jac.GetData());
}

template <int dim, bool entvar>
double DGSlipWallBC<dim, entvar>::calcBndryFun(const mfem::Vector &x,
                                               const mfem::Vector &dir,
                                               const mfem::Vector &q)
{
   mfem::Vector flux_vec(q.Size());
   calcFlux(x, dir, q, flux_vec);
   mfem::Vector w(q.Size());
   if (entvar)
   {
      w = q;
   }
   else
   {
      calcEntropyVars<double, dim>(q.GetData(), w.GetData());
   }
   return w * flux_vec;
}

template <int dim, bool entvar>
void DGSlipWallBC<dim, entvar>::calcFlux(const mfem::Vector &x,
                                         const mfem::Vector &dir,
                                         const mfem::Vector &q,
                                         mfem::Vector &flux_vec)
{
   calcSlipWallFlux<double, dim, entvar>(
       x.GetData(), dir.GetData(), q.GetData(), flux_vec.GetData());
}

template <int dim, bool entvar>
void DGSlipWallBC<dim, entvar>::calcFluxJacState(const mfem::Vector &x,
                                                 const mfem::Vector &dir,
                                                 const mfem::Vector &q,
                                                 mfem::DenseMatrix &flux_jac)
{
   // create containers for active double objects for each input
   std::vector<adouble> x_a(x.Size());
   std::vector<adouble> dir_a(dir.Size());
   std::vector<adouble> q_a(q.Size());
   // initialize active double containers with data from inputs
   adept::set_values(x_a.data(), x.Size(), x.GetData());
   adept::set_values(dir_a.data(), dir.Size(), dir.GetData());
   adept::set_values(q_a.data(), q.Size(), q.GetData());
   // start new stack recording
   this->stack.new_recording();
   // create container for active double flux output
   std::vector<adouble> flux_a(q.Size());
   mach::calcSlipWallFlux<adouble, dim, entvar>(
       x_a.data(), dir_a.data(), q_a.data(), flux_a.data());
   this->stack.independent(q_a.data(), q.Size());
   this->stack.dependent(flux_a.data(), q.Size());
   this->stack.jacobian(flux_jac.GetData());
}

template <int dim, bool entvar>
void DGSlipWallBC<dim, entvar>::calcFluxJacDir(const mfem::Vector &x,
                                               const mfem::Vector &dir,
                                               const mfem::Vector &q,
                                               mfem::DenseMatrix &flux_jac)
{
   // create containers for active double objects for each input
   std::vector<adouble> x_a(x.Size());
   std::vector<adouble> dir_a(dir.Size());
   std::vector<adouble> q_a(q.Size());
   // initialize active double containers with data from inputs
   adept::set_values(x_a.data(), x.Size(), x.GetData());
   adept::set_values(dir_a.data(), dir.Size(), dir.GetData());
   adept::set_values(q_a.data(), q.Size(), q.GetData());
   // start new stack recording
   this->stack.new_recording();
   // create container for active double flux output
   std::vector<adouble> flux_a(q.Size());
   mach::calcSlipWallFlux<adouble, dim, entvar>(
       x_a.data(), dir_a.data(), q_a.data(), flux_a.data());
   this->stack.independent(dir_a.data(), dir.Size());
   this->stack.dependent(flux_a.data(), q.Size());
   this->stack.jacobian(flux_jac.GetData());
}

template <int dim, bool entvar>
double DGFarFieldBC<dim, entvar>::calcBndryFun(const mfem::Vector &x,
                                               const mfem::Vector &dir,
                                               const mfem::Vector &q)
{
   mfem::Vector flux_vec(q.Size());
   calcFlux(x, dir, q, flux_vec);
   mfem::Vector w(q.Size());
   if (entvar)
   {
      w = q;
   }
   else
   {
      calcEntropyVars<double, dim>(q.GetData(), w.GetData());
   }
   return w * flux_vec;
}

template <int dim, bool entvar>
void DGFarFieldBC<dim, entvar>::calcFlux(const mfem::Vector &x,
                                         const mfem::Vector &dir,
                                         const mfem::Vector &q,
                                         mfem::Vector &flux_vec)
{
   calcFarFieldFlux<double, dim, entvar>(dir.GetData(),
                                         qfs.GetData(),
                                         q.GetData(),
                                         work_vec.GetData(),
                                         flux_vec.GetData());
}

template <int dim, bool entvar>
void DGFarFieldBC<dim, entvar>::calcFluxJacState(const mfem::Vector &x,
                                                 const mfem::Vector &dir,
                                                 const mfem::Vector &q,
                                                 mfem::DenseMatrix &flux_jac)
{
   // create containers for active double objects for each input
   std::vector<adouble> qfs_a(qfs.Size());
   std::vector<adouble> work_vec_a(work_vec.Size());
   std::vector<adouble> dir_a(dir.Size());
   std::vector<adouble> q_a(q.Size());
   // initialize active double containers with data from inputs
   adept::set_values(qfs_a.data(), qfs.Size(), qfs.GetData());
   adept::set_values(dir_a.data(), dir.Size(), dir.GetData());
   adept::set_values(q_a.data(), q.Size(), q.GetData());
   // start new stack recording
   this->stack.new_recording();
   // create container for active double flux output
   std::vector<adouble> flux_a(q.Size());
   mach::calcFarFieldFlux<adouble, dim, entvar>(dir_a.data(),
                                                qfs_a.data(),
                                                q_a.data(),
                                                work_vec_a.data(),
                                                flux_a.data());
   this->stack.independent(q_a.data(), q.Size());
   this->stack.dependent(flux_a.data(), q.Size());
   this->stack.jacobian(flux_jac.GetData());
}

template <int dim, bool entvar>
void DGFarFieldBC<dim, entvar>::calcFluxJacDir(const mfem::Vector &x,
                                               const mfem::Vector &dir,
                                               const mfem::Vector &q,
                                               mfem::DenseMatrix &flux_jac)
{
   // create containers for active double objects for each input
   std::vector<adouble> qfs_a(qfs.Size());
   std::vector<adouble> work_vec_a(work_vec.Size());
   std::vector<adouble> dir_a(dir.Size());
   std::vector<adouble> q_a(q.Size());
   // initialize active double containers with data from inputs
   adept::set_values(qfs_a.data(), qfs.Size(), qfs.GetData());
   adept::set_values(dir_a.data(), dir.Size(), dir.GetData());
   adept::set_values(q_a.data(), q.Size(), q.GetData());
   // start new stack recording
   this->stack.new_recording();
   // create container for active double flux output
   std::vector<adouble> flux_a(q.Size());
   mach::calcFarFieldFlux<adouble, dim, entvar>(dir_a.data(),
                                                qfs_a.data(),
                                                q_a.data(),
                                                work_vec_a.data(),
                                                flux_a.data());
   this->stack.independent(dir_a.data(), dir.Size());
   this->stack.dependent(flux_a.data(), q.Size());
   this->stack.jacobian(flux_jac.GetData());
}

template <int dim, bool entvar>
DGInterfaceIntegrator<dim, entvar>::DGInterfaceIntegrator(
    adept::Stack &diff_stack,
    double coeff,
    const mfem::FiniteElementCollection *fe_coll,
    double a)
 : DGInviscidFaceIntegrator<DGInterfaceIntegrator<dim, entvar>>(diff_stack,
                                                                fe_coll,
                                                                dim + 2,
                                                                a)
{
   MFEM_ASSERT(coeff >= 0.0,
               "DGInterfaceIntegrator: "
               "dissipation coefficient must be >= 0.0");
   diss_coeff = coeff;
}

template <int dim, bool entvar>
double DGInterfaceIntegrator<dim, entvar>::calcIFaceFun(const mfem::Vector &dir,
                                                        const mfem::Vector &qL,
                                                        const mfem::Vector &qR)
{
   std::cout << "yes, I got callled " << std::endl;
   mfem::Vector flux(qL.Size());
   calcFlux(dir, qL, qR, flux);
   mfem::Vector wL(qL.Size());
   mfem::Vector wR(qR.Size());
   if (entvar)
   {
      wL = qL;
      wR = qR;
   }
   else
   {
      calcEntropyVars<double, dim>(qL.GetData(), wL.GetData());
      calcEntropyVars<double, dim>(qR.GetData(), wR.GetData());
   }
   wL -= wR;
   return wL * flux;
}

template <int dim, bool entvar>
void DGInterfaceIntegrator<dim, entvar>::calcFlux(const mfem::Vector &dir,
                                                  const mfem::Vector &qL,
                                                  const mfem::Vector &qR,
                                                  mfem::Vector &flux)
{
   if (entvar)
   {
      calcIsmailRoeFaceFluxWithDissUsingEntVars<double, dim>(dir.GetData(),
                                                             diss_coeff,
                                                             qL.GetData(),
                                                             qR.GetData(),
                                                             flux.GetData());
   }
   else
   {
      // mach::calcRoeFaceFlux<double, dim>(
      //     dir.GetData(), qL.GetData(), qR.GetData(), flux.GetData());
      mach::calcLaxFriedrichsFlux<double, dim>(dir.GetData(),qL.GetData(),
      qR.GetData(),
                                               flux.GetData());
   }
}

template <int dim, bool entvar>
void DGInterfaceIntegrator<dim, entvar>::calcFluxJacState(
    const mfem::Vector &dir,
    const mfem::Vector &qL,
    const mfem::Vector &qR,
    mfem::DenseMatrix &jacL,
    mfem::DenseMatrix &jacR)
{
   // full size jacobian stores both left the right jac state
   mfem::DenseMatrix jac(qL.Size(), 2 * qL.Size());
   // vector of active input variables
   std::vector<adouble> dir_a(dir.Size());
   std::vector<adouble> qR_a(qR.Size());
   std::vector<adouble> qL_a(qL.Size());
   // initialize the values
   adouble diss_coeff_a = diss_coeff;
   adept::set_values(dir_a.data(), dir.Size(), dir.GetData());
   adept::set_values(qL_a.data(), qL.Size(), qL.GetData());
   adept::set_values(qR_a.data(), qR.Size(), qR.GetData());
   // start new recording
   this->stack.new_recording();
   // create vector of active output variables
   std::vector<adouble> flux_a(qL.Size());
   if (entvar)
   {
      mach::calcIsmailRoeFaceFluxWithDissUsingEntVars<adouble, dim>(
          dir_a.data(), diss_coeff_a, qL_a.data(), qR_a.data(), flux_a.data());
   }
   else
   {
      // mach::calcRoeFaceFlux<adouble, dim>(
      //     dir_a.data(), qL_a.data(), qR_a.data(), flux_a.data());
      mach::calcLaxFriedrichsFlux<adouble, dim>(
          dir_a.data(), qL_a.data(), qR_a.data(), flux_a.data());
   }
   // set the independent and dependent variables
   this->stack.independent(qL_a.data(), qL.Size());
   this->stack.independent(qR_a.data(), qR.Size());
   this->stack.dependent(flux_a.data(), qL.Size());
   // compute the jacobian
   this->stack.jacobian_reverse(jac.GetData());
   // retrieve the left the right jacobians
   jacL.CopyCols(jac, 0, qL.Size() - 1);
   jacR.CopyCols(jac, qL.Size(), 2 * qL.Size() - 1);
}

template <int dim, bool entvar>
void DGInterfaceIntegrator<dim, entvar>::calcFluxJacDir(
    const mfem::Vector &dir,
    const mfem::Vector &qL,
    const mfem::Vector &qR,
    mfem::DenseMatrix &jac_dir)
{
   // vector of active input variables
   std::vector<adouble> dir_a(dir.Size());
   std::vector<adouble> qR_a(qR.Size());
   std::vector<adouble> qL_a(qL.Size());
   // initialize the values
   adouble diss_coeff_a = diss_coeff;
   adept::set_values(dir_a.data(), dir.Size(), dir.GetData());
   adept::set_values(qL_a.data(), qL.Size(), qL.GetData());
   adept::set_values(qR_a.data(), qR.Size(), qR.GetData());
   // start new recording
   this->stack.new_recording();
   // create vector of active output variables
   std::vector<adouble> flux_a(qL.Size());
   if (entvar)
   {
      mach::calcIsmailRoeFaceFluxWithDissUsingEntVars<adouble, dim>(
          dir_a.data(), diss_coeff_a, qL_a.data(), qR_a.data(), flux_a.data());
   }
   else
   {
      // mach::calcRoeFaceFlux<adouble, dim>(
      //     dir_a.data(), qL_a.data(), qR_a.data(), flux_a.data());
      mach::calcLaxFriedrichsFlux<adouble, dim>(
          dir_a.data(), qL_a.data(), qR_a.data(), flux_a.data());
   }
   // set the independent and dependent variables
   this->stack.independent(dir_a.data(), dir.Size());
   this->stack.dependent(flux_a.data(), qL.Size());
   // compute the jacobian w.r.t dir
   this->stack.jacobian(jac_dir.GetData());
}

template <int dim, bool entvar>
double DGPressureForce<dim, entvar>::calcBndryFun(const mfem::Vector &x,
                                                  const mfem::Vector &dir,
                                                  const mfem::Vector &q)
{
   calcSlipWallFlux<double, dim, entvar>(
       x.GetData(), dir.GetData(), q.GetData(), work_vec.GetData());
   return dot<double, dim>(force_nrm.GetData(), work_vec.GetData() + 1);
}

template <int dim, bool entvar>
void DGPressureForce<dim, entvar>::calcFlux(const mfem::Vector &x,
                                            const mfem::Vector &dir,
                                            const mfem::Vector &q,
                                            mfem::Vector &flux_vec)
{
   // create containers for active double objects for each input
   std::vector<adouble> x_a(x.Size());
   std::vector<adouble> dir_a(dir.Size());
   std::vector<adouble> q_a(q.Size());
   std::vector<adouble> force_nrm_a(force_nrm.Size());
   // initialize active double containers with data from inputs
   adept::set_values(x_a.data(), x.Size(), x.GetData());
   adept::set_values(dir_a.data(), dir.Size(), dir.GetData());
   adept::set_values(q_a.data(), q.Size(), q.GetData());
   adept::set_values(force_nrm_a.data(), force_nrm.Size(), force_nrm.GetData());
   // start new stack recording
   this->stack.new_recording();
   // create container for active double flux output
   std::vector<adouble> flux_a(q.Size());
   mach::calcSlipWallFlux<adouble, dim, entvar>(
       x_a.data(), dir_a.data(), q_a.data(), flux_a.data());
   adouble fun_a = dot<adouble, dim>(force_nrm_a.data(), flux_a.data() + 1);
   fun_a.set_gradient(1.0);
   this->stack.compute_adjoint();
   adept::get_gradients(q_a.data(), q.Size(), flux_vec.GetData());
}

}  // namespace mach

#endif
