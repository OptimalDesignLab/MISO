#ifndef MACH_MESH_MOVE_INTEG
#define MACH_MESH_MOVE_INTEG

#include "mfem.hpp"

#include "finite_element_state.hpp"

namespace mach
{
/** Integrator for the linear elasticity form:
    a(u,v) = (lambda div(u), div(v)) + (2 mu e(u), e(v)),
    where e(v) = (1/2) (grad(v) + grad(v)^T).
    This is a 'Vector' integrator, i.e. defined for FE spaces
    using multiple copies of a scalar FE space. */
class ElasticityPositionIntegrator : public mfem::ElasticityIntegrator
{
public:
   ElasticityPositionIntegrator(mfem::Coefficient &l, mfem::Coefficient &m)
    : ElasticityIntegrator(l, m)
   { }

   // /** With this constructor lambda = q_l * m and mu = q_m * m;
   //     if dim * q_l + 2 * q_m = 0 then trace(sigma) = 0. */
   // ElasticityPositionIntegrator(mfem::Coefficient &m, double q_l, double q_m)
   // : ElasticityIntegrator(m, q_l, q_m)
   // { }

   void AssembleElementVector(const mfem::FiniteElement &el,
                              mfem::ElementTransformation &trans,
                              const mfem::Vector &elfun,
                              mfem::Vector &elvect) override;

private:
   friend class ElasticityPositionIntegratorStateRevSens;
   friend class ElasticityPositionIntegratorStateFwdSens;
};

class ElasticityPositionIntegratorStateRevSens
 : public mfem::LinearFormIntegrator
{
public:
   ElasticityPositionIntegratorStateRevSens(mfem::GridFunction &adjoint,
                                            ElasticityPositionIntegrator &integ)
    : adjoint(adjoint), integ(integ)
   { }

   void AssembleRHSElementVect(const mfem::FiniteElement &el,
                               mfem::ElementTransformation &trans,
                               mfem::Vector &state_bar) override;

private:
   /// the adjoint to use when evaluating d(psi^T R)/du
   mfem::GridFunction &adjoint;
   /// reference to primal integrator
   ElasticityPositionIntegrator &integ;

   mfem::Array<int> vdofs;
   mfem::Vector psi;
};

class ElasticityPositionIntegratorStateFwdSens
 : public mfem::LinearFormIntegrator
{
public:
   ElasticityPositionIntegratorStateFwdSens(mfem::GridFunction &state_dot,
                                            ElasticityPositionIntegrator &integ)
    : state_dot(state_dot), integ(integ)
   { }

   void AssembleRHSElementVect(const mfem::FiniteElement &el,
                               mfem::ElementTransformation &trans,
                               mfem::Vector &state_bar) override;

private:
   /// the state_dot to use when evaluating (dR/du) * state_dot
   mfem::GridFunction &state_dot;
   /// reference to primal integrator
   ElasticityPositionIntegrator &integ;

   mfem::Array<int> vdofs;
   mfem::Vector elfun_dot;
};

inline void addSensitivityIntegrator(
    ElasticityPositionIntegrator &primal_integ,
    std::map<std::string, FiniteElementState> &fields,
    std::map<std::string, mfem::ParLinearForm> &rev_sens,
    std::map<std::string, mfem::ParNonlinearForm> &rev_scalar_sens,
    std::map<std::string, mfem::ParLinearForm> &fwd_sens,
    std::map<std::string, mfem::ParNonlinearForm> &fwd_scalar_sens)
{
   auto &state = fields.at("state");
   auto &state_fes = fields.at("state").space();
   rev_sens.emplace("state", &state_fes);
   rev_sens.at("state").AddDomainIntegrator(
       new ElasticityPositionIntegratorStateRevSens(
           fields.at("adjoint").gridFunc(), primal_integ));

   fields.emplace(std::piecewise_construct,
                  std::forward_as_tuple("state_dot"),
                  std::forward_as_tuple(state.mesh(), state_fes));

   fwd_sens.emplace("state", &state_fes);
   fwd_sens.at("state").AddDomainIntegrator(
       new ElasticityPositionIntegratorStateFwdSens(
           fields.at("state_dot").gridFunc(), primal_integ));
}

}  // namespace mach

#endif
