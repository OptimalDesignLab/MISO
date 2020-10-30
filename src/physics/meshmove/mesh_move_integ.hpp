#ifndef MACH_MESH_MOVE_INTEG
#define MACH_MESH_MOVE_INTEG

#include "mfem.hpp"

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

};

} // namespace mach

#endif
