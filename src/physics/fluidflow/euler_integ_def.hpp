#ifndef MISO_EULER_INTEG_DEF
#define MISO_EULER_INTEG_DEF

#include "adept.h"
#include "mfem.hpp"

#include "inviscid_integ.hpp"
#include "euler_fluxes.hpp"
#include "euler_integ.hpp"

namespace miso
{
using adept::adouble;

template <int dim>
void EulerIntegrator<dim>::calcFluxJacState(const mfem::Vector &dir,
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
   miso::calcEulerFlux<adouble, dim>(dir_a.data(), q_a.data(), flux_a.data());
   // set the independent and dependent variable
   this->stack.independent(q_a.data(), q.Size());
   this->stack.dependent(flux_a.data(), q.Size());
   // calculate the jacobian w.r.t state vaiables
   this->stack.jacobian(flux_jac.GetData());
}

template <int dim>
void EulerIntegrator<dim>::calcFluxJacDir(const mfem::Vector &dir,
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
   miso::calcEulerFlux<adouble, dim>(dir_a.data(), q_a.data(), flux_a.data());
   this->stack.independent(dir_a.data(), dir.Size());
   this->stack.dependent(flux_a.data(), q.Size());
   // calculate the jacobian w.r.t state vaiables
   this->stack.jacobian(flux_jac.GetData());
}

template <int dim, bool entvar>
double IsmailRoeIntegrator<dim, entvar>::GetElementEnergy(
    const mfem::FiniteElement &el,
    mfem::ElementTransformation &trans,
    const mfem::Vector &elfun)
{
   int num_states = this->num_states;
   int num_nodes = el.GetDof();
   mfem::Vector u_i(num_states);
   mfem::Vector w_i(num_states);
   mfem::Vector res_i(num_states);
   mfem::Vector elres;
   this->AssembleElementVector(el, trans, elfun, elres);
   mfem::DenseMatrix u(elfun.GetData(), num_nodes, num_states);
   mfem::DenseMatrix res(elres.GetData(), num_nodes, num_states);
   double ent_change = 0.0;
   for (int i = 0; i < el.GetDof(); ++i)
   {
      u.GetRow(i, u_i);
      res.GetRow(i, res_i);
      calcEntropyVars<double, dim, entvar>(u_i.GetData(), w_i.GetData());
      ent_change += w_i * res_i;
   }
   return ent_change;
}

template <int dim, bool entvar>
void IsmailRoeIntegrator<dim, entvar>::calcFlux(int di,
                                                const mfem::Vector &qL,
                                                const mfem::Vector &qR,
                                                mfem::Vector &flux)
{
   if (entvar)
   {
      calcIsmailRoeFluxUsingEntVars<double, dim>(
          di, qL.GetData(), qR.GetData(), flux.GetData());
   }
   else
   {
      calcIsmailRoeFlux<double, dim>(
          di, qL.GetData(), qR.GetData(), flux.GetData());
   }
}

template <int dim, bool entvar>
void IsmailRoeIntegrator<dim, entvar>::calcFluxJacStates(
    int di,
    const mfem::Vector &qL,
    const mfem::Vector &qR,
    mfem::DenseMatrix &jacL,
    mfem::DenseMatrix &jacR)
{
   // store the full jacobian in jac
   mfem::DenseMatrix jac(dim + 2, 2 * (dim + 2));
   // vector of active input variables
   std::vector<adouble> qL_a(qL.Size());
   std::vector<adouble> qR_a(qR.Size());
   // initialize adouble inputs
   adept::set_values(qL_a.data(), qL.Size(), qL.GetData());
   adept::set_values(qR_a.data(), qR.Size(), qR.GetData());
   // start recording
   this->stack.new_recording();
   // create vector of active output variables
   std::vector<adouble> flux_a(qL.Size());
   // run algorithm
   if constexpr (entvar)
   {
      miso::calcIsmailRoeFluxUsingEntVars<adouble, dim>(
          di, qL_a.data(), qR_a.data(), flux_a.data());
   }
   else
   {
      miso::calcIsmailRoeFlux<adouble, dim>(
          di, qL_a.data(), qR_a.data(), flux_a.data());
   }
   // identify independent and dependent variables
   this->stack.independent(qL_a.data(), qL.Size());
   this->stack.independent(qR_a.data(), qR.Size());
   this->stack.dependent(flux_a.data(), qL.Size());
   // compute and store jacobian in jac
   this->stack.jacobian_reverse(jac.GetData());
   // retrieve the jacobian w.r.t left state
   jacL.CopyCols(jac, 0, dim + 1);
   // retrieve the jacobian w.r.t right state
   jacR.CopyCols(jac, dim + 2, 2 * (dim + 2) - 1);
}

template <int dim, bool entvar>
double EntStableLPSIntegrator<dim, entvar>::GetElementEnergy(
    const mfem::FiniteElement &el,
    mfem::ElementTransformation &trans,
    const mfem::Vector &elfun)
{
   int num_states = this->num_states;
   int num_nodes = el.GetDof();
   mfem::Vector u_i(num_states);
   mfem::Vector w_i(num_states);
   mfem::Vector res_i(num_states);
   mfem::Vector elres;
   this->AssembleElementVector(el, trans, elfun, elres);
   mfem::DenseMatrix u(elfun.GetData(), num_nodes, num_states);
   mfem::DenseMatrix res(elres.GetData(), num_nodes, num_states);
   double ent_change = 0.0;
   for (int i = 0; i < el.GetDof(); ++i)
   {
      u.GetRow(i, u_i);
      res.GetRow(i, res_i);
      calcEntropyVars<double, dim, entvar>(u_i.GetData(), w_i.GetData());
      ent_change += w_i * res_i;
   }
   return ent_change;
}

template <int dim, bool entvar>
void EntStableLPSIntegrator<dim, entvar>::convertVars(const mfem::Vector &q,
                                                      mfem::Vector &w)
{
   calcEntropyVars<double, dim, entvar>(q.GetData(), w.GetData());
}

template <int dim, bool entvar>
void EntStableLPSIntegrator<dim, entvar>::convertVarsJacState(
    const mfem::Vector &q,
    mfem::DenseMatrix &dwdu)
{
   if constexpr (entvar)
   {
      dwdu = 0.0;
      for (int i = 0; i < dim + 2; ++i)
      {
         dwdu(i, i) = 1.0;
      }
   }
   else
   {
      // vector of active input variables
      std::vector<adouble> q_a(q.Size());
      // initialize adouble inputs
      adept::set_values(q_a.data(), q.Size(), q.GetData());
      // start recording
      this->stack.new_recording();
      // create vector of active output variables
      std::vector<adouble> w_a(q.Size());
      // run algorithm
      calcEntropyVars<adouble, dim, entvar>(q_a.data(), w_a.data());
      // identify independent and dependent variables
      this->stack.independent(q_a.data(), q.Size());
      this->stack.dependent(w_a.data(), q.Size());
      // compute and store jacobian in dwdu
      this->stack.jacobian(dwdu.GetData());
   }
}

template <int dim, bool entvar>
void EntStableLPSIntegrator<dim, entvar>::applyScaling(
    const mfem::DenseMatrix &adjJ,
    const mfem::Vector &q,
    const mfem::Vector &vec,
    mfem::Vector &mat_vec)
{
   if (entvar)
   {
      applyLPSScalingUsingEntVars<double, dim>(
          adjJ.GetData(), q.GetData(), vec.GetData(), mat_vec.GetData());
   }
   else
   {
      applyLPSScaling<double, dim>(
          adjJ.GetData(), q.GetData(), vec.GetData(), mat_vec.GetData());
   }
}

template <int dim, bool entvar>
void EntStableLPSIntegrator<dim, entvar>::applyScalingJacState(
    const mfem::DenseMatrix &adjJ,
    const mfem::Vector &q,
    const mfem::Vector &vec,
    mfem::DenseMatrix &mat_vec_jac)
{
   // declare vectors of active input variables
   int adjJ_a_size = adjJ.Height() * adjJ.Width();
   std::vector<adouble> adjJ_a(adjJ_a_size);
   std::vector<adouble> q_a(q.Size());
   std::vector<adouble> vec_a(vec.Size());
   // copy data from mfem::Vector
   adept::set_values(adjJ_a.data(), adjJ_a_size, adjJ.GetData());
   adept::set_values(q_a.data(), q.Size(), q.GetData());
   adept::set_values(vec_a.data(), vec.Size(), vec.GetData());
   // start recording
   this->stack.new_recording();
   // the dependent variable must be declared after the recording
   std::vector<adouble> mat_vec_a(q.Size());
   if (entvar)
   {
      applyLPSScalingUsingEntVars<adouble, dim>(
          adjJ_a.data(), q_a.data(), vec_a.data(), mat_vec_a.data());
   }
   else
   {
      applyLPSScaling<adouble, dim>(
          adjJ_a.data(), q_a.data(), vec_a.data(), mat_vec_a.data());
   }
   // set the independent and dependent variable
   this->stack.independent(q_a.data(), q.Size());
   this->stack.dependent(mat_vec_a.data(), q.Size());
   // Calculate the jabobian
   this->stack.jacobian(mat_vec_jac.GetData());
}

template <int dim, bool entvar>
void EntStableLPSIntegrator<dim, entvar>::applyScalingJacAdjJ(
    const mfem::DenseMatrix &adjJ,
    const mfem::Vector &q,
    const mfem::Vector &vec,
    mfem::DenseMatrix &mat_vec_jac)
{
   // create containers for active double objects
   std::vector<adouble> adjJ_a(adjJ.Height() * adjJ.Width());
   std::vector<adouble> q_a(q.Size());
   std::vector<adouble> vec_a(vec.Size());
   // initialize active double containers with input data
   adept::set_values(
       adjJ_a.data(), adjJ.Height() * adjJ.Width(), adjJ.GetData());
   adept::set_values(q_a.data(), q.Size(), q.GetData());
   adept::set_values(vec_a.data(), vec.Size(), vec.GetData());
   // start new stack recording
   this->stack.new_recording();
   // create container for active double mat_vec output
   std::vector<adouble> mat_vec_a(q.Size());
   if (entvar)
   {
      applyLPSScalingUsingEntVars<adouble, dim>(
          adjJ_a.data(), q_a.data(), vec_a.data(), mat_vec_a.data());
   }
   else
   {
      applyLPSScaling<adouble, dim>(
          adjJ_a.data(), q_a.data(), vec_a.data(), mat_vec_a.data());
   }
   this->stack.independent(adjJ_a.data(), adjJ.Height() * adjJ.Width());
   this->stack.dependent(mat_vec_a.data(), q.Size());
   this->stack.jacobian(mat_vec_jac.GetData());
}

template <int dim, bool entvar>
void EntStableLPSIntegrator<dim, entvar>::applyScalingJacV(
    const mfem::DenseMatrix &adjJ,
    const mfem::Vector &q,
    mfem::DenseMatrix &mat_vec_jac)
{
   // declare vectors of active input variables
   int adjJ_a_size = adjJ.Height() * adjJ.Width();
   std::vector<adouble> adjJ_a(adjJ_a_size);
   std::vector<adouble> q_a(q.Size());
   std::vector<adouble> vec_a(q.Size());
   // copy data from mfem::Vector
   adept::set_values(adjJ_a.data(), adjJ_a_size, adjJ.GetData());
   adept::set_values(q_a.data(), q.Size(), q.GetData());
   // dependence on vec is linear, so any value is ok; use q
   adept::set_values(vec_a.data(), q.Size(), q.GetData());
   // start recording
   this->stack.new_recording();
   // the dependent variable must be declared after the recording
   std::vector<adouble> mat_vec_a(q.Size());
   if (entvar)
   {
      applyLPSScalingUsingEntVars<adouble, dim>(
          adjJ_a.data(), q_a.data(), vec_a.data(), mat_vec_a.data());
   }
   else
   {
      applyLPSScaling<adouble, dim>(
          adjJ_a.data(), q_a.data(), vec_a.data(), mat_vec_a.data());
   }
   // set the independent and dependent variable
   this->stack.independent(vec_a.data(), q.Size());
   this->stack.dependent(mat_vec_a.data(), q.Size());
   // Calculate the jabobian
   this->stack.jacobian(mat_vec_jac.GetData());
}

template <int dim, bool entvar>
void MassIntegrator<dim, entvar>::convertVars(const mfem::Vector &u,
                                              mfem::Vector &q)
{
   calcConservativeVars<double, dim, entvar>(u.GetData(), q.GetData());
}

template <int dim, bool entvar>
void MassIntegrator<dim, entvar>::convertVarsJacState(const mfem::Vector &u,
                                                      mfem::DenseMatrix &dqdu)
{
   if constexpr (entvar)
   {
      // vector of active input variables
      std::vector<adouble> u_a(u.Size());
      // initialize adouble inputs
      adept::set_values(u_a.data(), u.Size(), u.GetData());
      // start recording
      this->stack.new_recording();
      // create vector of active output variables
      std::vector<adouble> q_a(u.Size());
      // run algorithm
      calcConservativeVars<adouble, dim, true>(u_a.data(), q_a.data());
      // identify independent and dependent variables
      this->stack.independent(u_a.data(), u.Size());
      this->stack.dependent(q_a.data(), u.Size());
      // compute and store jacobian in dwdu
      this->stack.jacobian(dqdu.GetData());
   }
   else
   {
      dqdu = 0.0;
      for (int i = 0; i < dim + 2; ++i)
      {
         dqdu(i, i) = 1.0;
      }
   }
}

template <int dim, bool entvar>
double IsentropicVortexBC<dim, entvar>::calcBndryFun(const mfem::Vector &x,
                                                     const mfem::Vector &dir,
                                                     const mfem::Vector &q)
{
   mfem::Vector flux_vec(q.Size());
   calcFlux(x, dir, q, flux_vec);
   mfem::Vector w(q.Size());
   calcEntropyVars<double, dim, entvar>(q.GetData(), w.GetData());
   return w * flux_vec;
}

template <int dim, bool entvar>
void IsentropicVortexBC<dim, entvar>::calcFlux(const mfem::Vector &x,
                                               const mfem::Vector &dir,
                                               const mfem::Vector &q,
                                               mfem::Vector &flux_vec)
{
   calcIsentropicVortexFlux<double, entvar>(
       x.GetData(), dir.GetData(), q.GetData(), flux_vec.GetData());
}

template <int dim, bool entvar>
void IsentropicVortexBC<dim, entvar>::calcFluxJacState(
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
   miso::calcIsentropicVortexFlux<adouble, entvar>(
       x_a.data(), dir_a.data(), q_a.data(), flux_a.data());
   this->stack.independent(q_a.data(), q.Size());
   this->stack.dependent(flux_a.data(), q.Size());
   this->stack.jacobian(flux_jac.GetData());
}

template <int dim, bool entvar>
void IsentropicVortexBC<dim, entvar>::calcFluxJacDir(
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
   miso::calcIsentropicVortexFlux<adouble, entvar>(
       x_a.data(), dir_a.data(), q_a.data(), flux_a.data());
   this->stack.independent(dir_a.data(), dir.Size());
   this->stack.dependent(flux_a.data(), q.Size());
   this->stack.jacobian(flux_jac.GetData());
}

template <int dim, bool entvar>
double SlipWallBC<dim, entvar>::calcBndryFun(const mfem::Vector &x,
                                             const mfem::Vector &dir,
                                             const mfem::Vector &q)
{
   mfem::Vector flux_vec(q.Size());
   calcFlux(x, dir, q, flux_vec);
   mfem::Vector w(q.Size());
   calcEntropyVars<double, dim, entvar>(q.GetData(), w.GetData());
   return w * flux_vec;
}

template <int dim, bool entvar>
void SlipWallBC<dim, entvar>::calcFlux(const mfem::Vector &x,
                                       const mfem::Vector &dir,
                                       const mfem::Vector &q,
                                       mfem::Vector &flux_vec)
{
   calcSlipWallFlux<double, dim, entvar>(
       x.GetData(), dir.GetData(), q.GetData(), flux_vec.GetData());
}

template <int dim, bool entvar>
void SlipWallBC<dim, entvar>::calcFluxJacState(const mfem::Vector &x,
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
   miso::calcSlipWallFlux<adouble, dim, entvar>(
       x_a.data(), dir_a.data(), q_a.data(), flux_a.data());
   this->stack.independent(q_a.data(), q.Size());
   this->stack.dependent(flux_a.data(), q.Size());
   this->stack.jacobian(flux_jac.GetData());
}

template <int dim, bool entvar>
void SlipWallBC<dim, entvar>::calcFluxJacDir(const mfem::Vector &x,
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
   miso::calcSlipWallFlux<adouble, dim, entvar>(
       x_a.data(), dir_a.data(), q_a.data(), flux_a.data());
   this->stack.independent(dir_a.data(), dir.Size());
   this->stack.dependent(flux_a.data(), q.Size());
   this->stack.jacobian(flux_jac.GetData());
}

template <int dim, bool entvar>
double FarFieldBC<dim, entvar>::calcBndryFun(const mfem::Vector &x,
                                             const mfem::Vector &dir,
                                             const mfem::Vector &q)
{
   mfem::Vector flux_vec(q.Size());
   calcFlux(x, dir, q, flux_vec);
   mfem::Vector w(q.Size());
   calcEntropyVars<double, dim, entvar>(q.GetData(), w.GetData());
   return w * flux_vec;
}

template <int dim, bool entvar>
void FarFieldBC<dim, entvar>::calcFlux(const mfem::Vector &x,
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
void FarFieldBC<dim, entvar>::calcFluxJacState(const mfem::Vector &x,
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
   miso::calcFarFieldFlux<adouble, dim, entvar>(dir_a.data(),
                                                qfs_a.data(),
                                                q_a.data(),
                                                work_vec_a.data(),
                                                flux_a.data());
   this->stack.independent(q_a.data(), q.Size());
   this->stack.dependent(flux_a.data(), q.Size());
   this->stack.jacobian(flux_jac.GetData());
}

template <int dim, bool entvar>
void FarFieldBC<dim, entvar>::calcFluxJacDir(const mfem::Vector &x,
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
   miso::calcFarFieldFlux<adouble, dim, entvar>(dir_a.data(),
                                                qfs_a.data(),
                                                q_a.data(),
                                                work_vec_a.data(),
                                                flux_a.data());
   this->stack.independent(dir_a.data(), dir.Size());
   this->stack.dependent(flux_a.data(), q.Size());
   this->stack.jacobian(flux_jac.GetData());
}

template <int dim, bool entvar>
double EntropyConserveBC<dim, entvar>::calcBndryFun(const mfem::Vector &x,
                                                    const mfem::Vector &dir,
                                                    const mfem::Vector &q)
{
   calcFlux(x, dir, q, work1);
   calcEntropyVars<double, dim, entvar>(q.GetData(), work2.GetData());
   return work2 * work1;
}

template <int dim, bool entvar>
void EntropyConserveBC<dim, entvar>::calcFlux(const mfem::Vector &x,
                                              const mfem::Vector &dir,
                                              const mfem::Vector &q,
                                              mfem::Vector &flux_vec)
{
   // first, evaluate the boundary state, and store the ent vars in work2.
   bc_fun(t, x, work1);
   calcConservativeVars<double, dim, entvar>(q.GetData(), work2.GetData());
   // Next, compute the target entropy flux
   double Un = dot<double, dim>(work1.GetData() + 1, dir.GetData()) / work1(0);
   double entflux = Un * entropy<double, dim>(work1.GetData());
   // Evaluate the boundary flux
   calcBoundaryFluxEC<double, dim>(dir.GetData(),
                                   work1.GetData(),
                                   work2.GetData(),
                                   entflux,
                                   flux_vec.GetData());
}

template <int dim, bool entvar>
void EntropyConserveBC<dim, entvar>::calcFluxJacState(
    const mfem::Vector &x,
    const mfem::Vector &dir,
    const mfem::Vector &q,
    mfem::DenseMatrix &flux_jac)
{
   // evaluate the boundary state, which does not depend on state
   bc_fun(t, x, work1);
   // create containers for active double objects for each input
   std::vector<adouble> work1_a(work1.Size());
   std::vector<adouble> work2_a(work2.Size());
   std::vector<adouble> dir_a(dir.Size());
   std::vector<adouble> q_a(q.Size());
   // initialize active double containers with data from inputs
   adept::set_values(work1_a.data(), work1.Size(), work1.GetData());
   adept::set_values(dir_a.data(), dir.Size(), dir.GetData());
   adept::set_values(q_a.data(), q.Size(), q.GetData());
   // start new stack recording
   this->stack.new_recording();
   // create container for active double flux output
   std::vector<adouble> flux_a(q.Size());
   calcConservativeVars<adouble, dim, entvar>(q_a.data(), work2_a.data());
   adouble Un_a =
       dot<adouble, dim>(work1_a.data() + 1, dir_a.data()) / work1_a[0];
   adouble entflux_a = Un_a * entropy<adouble, dim>(work1_a.data());
   calcBoundaryFluxEC<adouble, dim>(
       dir_a.data(), work1_a.data(), work2_a.data(), entflux_a, flux_a.data());
   this->stack.independent(q_a.data(), q.Size());
   this->stack.dependent(flux_a.data(), q.Size());
   this->stack.jacobian(flux_jac.GetData());
}

template <int dim, bool entvar>
void EntropyConserveBC<dim, entvar>::calcFluxJacDir(const mfem::Vector &x,
                                                    const mfem::Vector &dir,
                                                    const mfem::Vector &q,
                                                    mfem::DenseMatrix &flux_jac)
{
   // evaluate the boundary state, which does not depend on state
   bc_fun(t, x, work1);
   // create containers for active double objects for each input
   std::vector<adouble> work1_a(work1.Size());
   std::vector<adouble> work2_a(work2.Size());
   std::vector<adouble> dir_a(dir.Size());
   std::vector<adouble> q_a(q.Size());
   // initialize active double containers with data from inputs
   adept::set_values(work1_a.data(), work1.Size(), work1.GetData());
   adept::set_values(dir_a.data(), dir.Size(), dir.GetData());
   adept::set_values(q_a.data(), q.Size(), q.GetData());
   // start new stack recording
   this->stack.new_recording();
   // create container for active double flux output
   std::vector<adouble> flux_a(q.Size());
   calcConservativeVars<adouble, dim, entvar>(q_a.data(), work2_a.data());
   adouble Un_a =
       dot<adouble, dim>(work1_a.data() + 1, dir_a.data()) / work1_a[0];
   adouble entflux_a = Un_a * entropy<adouble, dim>(work1_a.data());
   calcBoundaryFluxEC<adouble, dim>(
       dir_a.data(), work1_a.data(), work2_a.data(), entflux_a, flux_a.data());
   this->stack.independent(dir_a.data(), dir.Size());
   this->stack.dependent(flux_a.data(), q.Size());
   this->stack.jacobian(flux_jac.GetData());
}

template <int dim, bool entvar>
double ControlBC<dim, entvar>::calcBndryFun(const mfem::Vector &x,
                                            const mfem::Vector &dir,
                                            const mfem::Vector &q)
{
   // don't use work1 for flux here, since it is used in calcFlux
   calcFlux(x, dir, q, work2);
   calcEntropyVars<double, dim, entvar>(q.GetData(), work1.GetData());
   return work2 * work1;
}

template <int dim, bool entvar>
void ControlBC<dim, entvar>::calcFlux(const mfem::Vector &x,
                                      const mfem::Vector &dir,
                                      const mfem::Vector &q,
                                      mfem::Vector &flux_vec)
{
   // use x to determine how we scale the control locally
   double uc = control * control_scale(len_scale, x_actuator, x);
   // convert variable to conservative, if necessary
   calcConservativeVars<double, dim, entvar>(q.GetData(), work1.GetData());
   // compute the flux
   calcControlFlux<double, dim>(
       dir.GetData(), work1.GetData(), uc, flux_vec.GetData());
}

template <int dim, bool entvar>
void ControlBC<dim, entvar>::calcFluxJacState(const mfem::Vector &x,
                                              const mfem::Vector &dir,
                                              const mfem::Vector &q,
                                              mfem::DenseMatrix &flux_jac)
{
   // evaluate the scaled control, which does not depend on the flow state
   adouble uc_a = control * control_scale(len_scale, x_actuator, x);
   // create containers for active double objects for each input
   std::vector<adouble> work1_a(work1.Size());
   std::vector<adouble> dir_a(dir.Size());
   std::vector<adouble> q_a(q.Size());
   // initialize active double containers with data from inputs
   adept::set_values(work1_a.data(), work1.Size(), work1.GetData());
   adept::set_values(dir_a.data(), dir.Size(), dir.GetData());
   adept::set_values(q_a.data(), q.Size(), q.GetData());
   // start new stack recording
   this->stack.new_recording();
   // create container for active double flux output
   std::vector<adouble> flux_a(q.Size());
   calcConservativeVars<adouble, dim, entvar>(q_a.data(), work1_a.data());
   calcControlFlux<adouble, dim>(
       dir_a.data(), work1_a.data(), uc_a, flux_a.data());
   this->stack.independent(q_a.data(), q.Size());
   this->stack.dependent(flux_a.data(), q.Size());
   this->stack.jacobian(flux_jac.GetData());
}

template <int dim, bool entvar>
void ControlBC<dim, entvar>::calcFluxJacDir(const mfem::Vector &x,
                                            const mfem::Vector &dir,
                                            const mfem::Vector &q,
                                            mfem::DenseMatrix &flux_jac)
{
   // evaluate the scaled control, which does not depend on dir
   adouble uc_a = control * control_scale(len_scale, x_actuator, x);
   // create containers for active double objects for each input
   std::vector<adouble> work1_a(work1.Size());
   std::vector<adouble> dir_a(dir.Size());
   std::vector<adouble> q_a(q.Size());
   // initialize active double containers with data from inputs
   adept::set_values(work1_a.data(), work1.Size(), work1.GetData());
   adept::set_values(dir_a.data(), dir.Size(), dir.GetData());
   adept::set_values(q_a.data(), q.Size(), q.GetData());
   // start new stack recording
   this->stack.new_recording();
   // create container for active double flux output
   std::vector<adouble> flux_a(q.Size());
   calcConservativeVars<adouble, dim, entvar>(q_a.data(), work1_a.data());
   calcControlFlux<adouble, dim>(
       dir_a.data(), work1_a.data(), uc_a, flux_a.data());
   this->stack.independent(dir_a.data(), dir.Size());
   this->stack.dependent(flux_a.data(), q.Size());
   this->stack.jacobian(flux_jac.GetData());
}

template <int dim, bool entvar>
InterfaceIntegrator<dim, entvar>::InterfaceIntegrator(
    adept::Stack &diff_stack,
    double coeff,
    const mfem::FiniteElementCollection *fe_coll,
    double a)
 : InviscidFaceIntegrator<InterfaceIntegrator<dim, entvar>>(diff_stack,
                                                            fe_coll,
                                                            dim + 2,
                                                            a)
{
   MFEM_ASSERT(coeff >= 0.0,
               "InterfaceIntegrator: "
               "dissipation coefficient must be >= 0.0");
   diss_coeff = coeff;
}

template <int dim, bool entvar>
double InterfaceIntegrator<dim, entvar>::calcIFaceFun(const mfem::Vector &dir,
                                                      const mfem::Vector &qL,
                                                      const mfem::Vector &qR)
{
   mfem::Vector flux(qL.Size());
   calcFlux(dir, qL, qR, flux);
   mfem::Vector wL(qL.Size());
   mfem::Vector wR(qR.Size());
   calcEntropyVars<double, dim, entvar>(qL.GetData(), wL.GetData());
   calcEntropyVars<double, dim, entvar>(qR.GetData(), wR.GetData());
   wL -= wR;
   return wL * flux;
}

template <int dim, bool entvar>
void InterfaceIntegrator<dim, entvar>::calcFlux(const mfem::Vector &dir,
                                                const mfem::Vector &qL,
                                                const mfem::Vector &qR,
                                                mfem::Vector &flux)
{
   if constexpr (entvar)
   {
      calcIsmailRoeFaceFluxWithDissUsingEntVars<double, dim>(dir.GetData(),
                                                             diss_coeff,
                                                             qL.GetData(),
                                                             qR.GetData(),
                                                             flux.GetData());
   }
   else
   {
      calcIsmailRoeFaceFluxWithDiss<double, dim>(dir.GetData(),
                                                 diss_coeff,
                                                 qL.GetData(),
                                                 qR.GetData(),
                                                 flux.GetData());
   }
}

template <int dim, bool entvar>
void InterfaceIntegrator<dim, entvar>::calcFluxJacState(const mfem::Vector &dir,
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
      miso::calcIsmailRoeFaceFluxWithDissUsingEntVars<adouble, dim>(
          dir_a.data(), diss_coeff_a, qL_a.data(), qR_a.data(), flux_a.data());
   }
   else
   {
      miso::calcIsmailRoeFaceFluxWithDiss<adouble, dim>(
          dir_a.data(), diss_coeff_a, qL_a.data(), qR_a.data(), flux_a.data());
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
void InterfaceIntegrator<dim, entvar>::calcFluxJacDir(
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
      miso::calcIsmailRoeFaceFluxWithDissUsingEntVars<adouble, dim>(
          dir_a.data(), diss_coeff_a, qL_a.data(), qR_a.data(), flux_a.data());
   }
   else
   {
      miso::calcIsmailRoeFaceFluxWithDiss<adouble, dim>(
          dir_a.data(), diss_coeff_a, qL_a.data(), qR_a.data(), flux_a.data());
   }
   // set the independent and dependent variables
   this->stack.independent(dir_a.data(), dir.Size());
   this->stack.dependent(flux_a.data(), qL.Size());
   // compute the jacobian w.r.t dir
   this->stack.jacobian(jac_dir.GetData());
}

template <int dim, bool entvar>
double PressureForce<dim, entvar>::calcBndryFun(const mfem::Vector &x,
                                                const mfem::Vector &dir,
                                                const mfem::Vector &q)
{
   calcSlipWallFlux<double, dim, entvar>(
       x.GetData(), dir.GetData(), q.GetData(), work_vec.GetData());
   return dot<double, dim>(force_nrm.GetData(), work_vec.GetData() + 1);
}

template <int dim, bool entvar>
void PressureForce<dim, entvar>::calcFlux(const mfem::Vector &x,
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
   miso::calcSlipWallFlux<adouble, dim, entvar>(
       x_a.data(), dir_a.data(), q_a.data(), flux_a.data());
   auto fun_a = dot<adouble, dim>(force_nrm_a.data(), flux_a.data() + 1);
   fun_a.set_gradient(1.0);
   this->stack.compute_adjoint();
   adept::get_gradients(q_a.data(), q.Size(), flux_vec.GetData());
}

template <int dim, bool entvar>
double EntropyIntegrator<dim, entvar>::calcVolFun(const mfem::Vector &x,
                                                  const mfem::Vector &u)
{
   return entropy<double, dim, entvar>(u.GetData());
}

template <int dim, bool entvar>
double BoundaryEntropy<dim, entvar>::calcBndryFun(const mfem::Vector &x,
                                                  const mfem::Vector &dir,
                                                  const mfem::Vector &q)
{
   // evaluate the entropy, then return the scaled value
   double S = entropy<double, dim, entvar>(q.GetData());
   double press = pressure<double, dim>(q.GetData());
   S += euler::rho_ref*(press - euler::press_ref)/euler::press_ref;
   double scale = control_scale(len_scale, x_actuator, x);
   double dA = sqrt(dot<double, dim>(dir.GetData(), dir.GetData()));
   return scale * S * dA;
}

template <int dim, bool entvar>
double SupplyRate<dim, entvar>::calcBndryFun(const mfem::Vector &x,
                                             const mfem::Vector &dir,
                                             const mfem::Vector &q)
{
   mfem::Vector flux_vec(q.Size());
   calcFarFieldFlux2<double, dim, entvar>(dir.GetData(),
                                          qfs.GetData(),
                                          q.GetData(),
                                          work_vec.GetData(),
                                          flux_vec.GetData());
   mfem::Vector w(q.Size());
   calcEntropyVars<double, dim, entvar>(q.GetData(), w.GetData());
   return dot<double, dim>(dir.GetData(), q.GetData()+1) - w * flux_vec;
}

}  // namespace miso

#endif
