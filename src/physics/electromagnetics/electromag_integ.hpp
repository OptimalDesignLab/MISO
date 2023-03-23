#ifndef MACH_ELECTROMAG_INTEG
#define MACH_ELECTROMAG_INTEG

#include <unordered_set>
#include <utility>

#include "mfem.hpp"

#include "mach_types.hpp"
#include "mach_input.hpp"
#include "mach_integrator.hpp"

namespace mach
{
class AbstractSolver;
class StateCoefficient;
class TwoStateCoefficient;
class ThreeStateCoefficient;
class VectorStateCoefficient;
class ScalarVectorProductCoefficient;

/// Compute the integral of HdB from 0 to B
/// \param[in] trans - element transformation for where to evaluate `nu`
/// \param[in] ip - integration point for where to evaluate `nu`
/// \param[in] nu - material dependent model describing reluctivity
/// \param[in] B - upper bound for integration
/// \return the magnetic energy
double calcMagneticEnergy(mfem::ElementTransformation &trans,
                          const mfem::IntegrationPoint &ip,
                          StateCoefficient &nu,
                          double B);

/// Compute the derivative of the magnetic energy with respect to B
/// \param[in] trans - element transformation for where to evaluate `nu`
/// \param[in] ip - integration point for where to evaluate `nu`
/// \param[in] nu - material dependent model describing reluctivity
/// \param[in] B - upper bound for integration
/// \return the derivative of the magnetic energy with respect to B
double calcMagneticEnergyDot(mfem::ElementTransformation &trans,
                             const mfem::IntegrationPoint &ip,
                             StateCoefficient &nu,
                             double B);

/// Compute the second derivative of the magnetic energy with respect to B
/// \param[in] trans - element transformation for where to evaluate `nu`
/// \param[in] ip - integration point for where to evaluate `nu`
/// \param[in] nu - material dependent model describing reluctivity
/// \param[in] B - upper bound for integration
/// \return the second derivative of the magnetic energy with respect to B
double calcMagneticEnergyDoubleDot(mfem::ElementTransformation &trans,
                                   const mfem::IntegrationPoint &ip,
                                   StateCoefficient &nu,
                                   double B);

/// Integrator for (m(u) grad u, grad v)
class NonlinearDiffusionIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   NonlinearDiffusionIntegrator(StateCoefficient &m, double a = 1.0)
    : model(m), alpha(a)
   { }

   /// Construct the element local residual
   /// \param[in] el - the finite element whose residual we want
   /// \param[in] trans - defines the reference to physical element mapping
   /// \param[in] elfun - element local state vector
   /// \param[out] elvect - element local residual
   void AssembleElementVector(const mfem::FiniteElement &el,
                              mfem::ElementTransformation &trans,
                              const mfem::Vector &elfun,
                              mfem::Vector &elvect) override;

   /// Construct the element local Jacobian
   /// \param[in] el - the finite element whose Jacobian we want
   /// \param[in] trans - defines the reference to physical element mapping
   /// \param[in] elfun - element local state vector
   /// \param[out] elmat - element local Jacobian
   void AssembleElementGrad(const mfem::FiniteElement &el,
                            mfem::ElementTransformation &trans,
                            const mfem::Vector &elfun,
                            mfem::DenseMatrix &elmat) override;

private:
   /// material (thus mesh) dependent model describing electromagnetic behavior
   StateCoefficient &model;
   /// scales the terms; can be used to move to rhs/lhs
   double alpha;

#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix dshape, dshapedxt, point_flux_2_dot;
   mfem::Vector pointflux_norm_dot;
#endif
   friend class NonlinearDiffusionIntegratorMeshRevSens;
};

/// Integrator to assemble d(psi^T R)/dX for the NonlinearDiffusionIntegrator
class NonlinearDiffusionIntegratorMeshRevSens
 : public mfem::LinearFormIntegrator
{
public:
   /// \param[in] state - the state to use when evaluating d(psi^T R)/dX
   /// \param[in] adjoint - the adjoint to use when evaluating d(psi^T R)/dX
   /// \param[in] integ - reference to primal integrator
   NonlinearDiffusionIntegratorMeshRevSens(mfem::GridFunction &state,
                                           mfem::GridFunction &adjoint,
                                           NonlinearDiffusionIntegrator &integ)
    : state(state), adjoint(adjoint), integ(integ)
   { }

   /// \brief - assemble an element's contribution to d(psi^T R)/dX
   /// \param[in] el - the finite element that describes the mesh element
   /// \param[in] trans - the transformation between reference and physical
   /// space
   /// \param[out] mesh_coords_bar - d(psi^T R)/dX for the element
   /// \note the LinearForm that assembles this integrator's FiniteElementSpace
   /// MUST be the mesh's nodal finite element space
   void AssembleRHSElementVect(const mfem::FiniteElement &el,
                               mfem::ElementTransformation &trans,
                               mfem::Vector &mesh_coords_bar) override;

private:
   /// the state to use when evaluating d(psi^T R)/dX
   mfem::GridFunction &state;
   /// the adjoint to use when evaluating d(psi^T R)/dX
   mfem::GridFunction &adjoint;
   /// reference to primal integrator
   NonlinearDiffusionIntegrator &integ;
#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix dshapedxt_bar, PointMat_bar;
   mfem::Array<int> vdofs;
   mfem::Vector elfun, psi;
#endif
};

inline void addSensitivityIntegrator(
    NonlinearDiffusionIntegrator &primal_integ,
    std::map<std::string, FiniteElementState> &fields,
    std::map<std::string, mfem::ParLinearForm> &rev_sens,
    std::map<std::string, mfem::ParNonlinearForm> &rev_scalar_sens,
    std::map<std::string, mfem::ParLinearForm> &fwd_sens,
    std::map<std::string, mfem::ParNonlinearForm> &fwd_scalar_sens)
{
   auto &mesh_fes = fields.at("mesh_coords").space();
   rev_sens.emplace("mesh_coords", &mesh_fes);
   rev_sens.at("mesh_coords")
       .AddDomainIntegrator(new NonlinearDiffusionIntegratorMeshRevSens(
           fields.at("state").gridFunc(),
           fields.at("adjoint").gridFunc(),
           primal_integ));
}

class MagnetizationSource2DIntegrator : public mfem::LinearFormIntegrator
{
public:
   ///TODO: mfem::VectorCoefficient &M -> ScalarVectorProductCoefficient &M (mach)
   MagnetizationSource2DIntegrator(ScalarVectorProductCoefficient &M, // will need to be formerly a mfem::VectorCoefficient &M
                                   double alpha = 1.0,
                                   mfem::GridFunction *temperature_field=nullptr)
    : M(M), alpha(alpha), temperature_field(temperature_field)
   { }

   void AssembleRHSElementVect(const mfem::FiniteElement &el,
                               mfem::ElementTransformation &trans,
                               mfem::Vector &elvect) override;

private:
   /// vector coefficient from linear form
   ///TODO: mfem::VectorCoefficient &M -> ScalarVectorProductCoefficient &M (mach)
   // mfem::VectorCoefficient &M;
   ScalarVectorProductCoefficient &M;

   /// scaling term if the linear form has a negative sign in the residual
   const double alpha;

   mfem::GridFunction *temperature_field; // pointer to the temperature field

#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix dshape, dshapedxt;
   mfem::Vector scratch;
   mfem::Vector temp_shape;
   mfem::Array<int> vdofs;
   mfem::Vector temp_elfun;
#endif
   /// class that implements mesh sensitivities for MagnetizationSource2DIntegrator
   friend class MagnetizationSource2DIntegratorMeshRevSens;
   /// class that implements temperature sensitivities for MagnetizationSource2DIntegrator
   friend class MagnetizationSource2DIntegratorTemperatureSens;
};

/// Integrator to assemble d(psi^T R)/dX for the MagnetizationSource2DIntegrator, updated to reflect the fact there is now (possibly) a temperature field
class MagnetizationSource2DIntegratorMeshRevSens
 : public mfem::LinearFormIntegrator
{
public:
   /// \param[in] adjoint - the adjoint to use when evaluating d(psi^T R)/dX
   /// \param[in] integ - reference to primal integrator
   /// \param[in] temperature_field - pointer to the temperature field to use when evaluating d(psi^T R)/dX
   MagnetizationSource2DIntegratorMeshRevSens(
       mfem::GridFunction &adjoint,
       MagnetizationSource2DIntegrator &integ,
       mfem::GridFunction *temperature_field=nullptr)
    : adjoint(adjoint), integ(integ), temperature_field(temperature_field)
   { }

   /// \brief - assemble an element's contribution to d(psi^T R)/dX
   /// \param[in] el - the finite element that describes the mesh element
   /// \param[in] trans - the transformation between reference and physical
   /// space
   /// \param[out] mesh_coords_bar - d(psi^T R)/dX for the element
   /// \note the LinearForm that assembles this integrator's FiniteElementSpace
   /// MUST be the mesh's nodal finite element space
   void AssembleRHSElementVect(const mfem::FiniteElement &el,
                               mfem::ElementTransformation &trans,
                               mfem::Vector &mesh_coords_bar) override;

private:
   /// the adjoint to use when evaluating d(psi^T R)/dX
   mfem::GridFunction &adjoint;
   /// reference to primal integrator
   MagnetizationSource2DIntegrator &integ;

   mfem::GridFunction *temperature_field; // pointer to the temperature field to use when evaluating d(psi^T R)/dX

#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix dshapedxt_bar, PointMat_bar;
   mfem::Vector scratch_bar;
   mfem::Array<int> vdofs;
   mfem::Vector psi;
   mfem::Vector temp_shape;
   mfem::Vector temp_elfun;
#endif
};

/// Integrator to assemble d(psi^T R)/dT for the MagnetizationSource2DIntegrator
class MagnetizationSource2DIntegratorTemperatureSens
 : public mfem::LinearFormIntegrator
{
public:
   /// \param[in] adjoint - the adjoint to use when evaluating d(psi^T R)/dT
   /// \param[in] integ - reference to primal integrator
   /// \param[in] temperature_field - pointer to the temperature field to use when evaluating d(psi^T R)/dT
   MagnetizationSource2DIntegratorTemperatureSens(
       mfem::GridFunction &adjoint,
       MagnetizationSource2DIntegrator &integ,
       mfem::GridFunction *temperature_field=nullptr)
    : adjoint(adjoint), integ(integ), temperature_field(temperature_field)
   { }

   /// \brief - assemble an element's contribution to d(psi^T R)/dT
   /// \param[in] el - the finite element that describes the mesh element
   /// \param[in] trans - the transformation between reference and physical
   /// space
   /// \param[out] temperature_bar - d(psi^T R)/dT for the element
   /// \note the LinearForm that assembles this integrator's FiniteElementSpace
   /// MUST be the mesh's nodal finite element space
   void AssembleRHSElementVect(const mfem::FiniteElement &el,
                               mfem::ElementTransformation &trans,
                               mfem::Vector &mesh_coords_bar) override;

private:
   /// the adjoint to use when evaluating d(psi^T R)/dT
   mfem::GridFunction &adjoint;
   /// reference to primal integrator
   MagnetizationSource2DIntegrator &integ;

   mfem::GridFunction *temperature_field; // pointer to the temperature field to use when evaluating d(psi^T R)/dT

#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix dshapedxt_bar, PointMat_bar;
   mfem::Vector scratch_bar;
   mfem::Array<int> vdofs;
   mfem::Vector psi;
   mfem::Vector temp_shape;
   mfem::Vector temp_elfun;
#endif
};

inline void addSensitivityIntegrator(
    MagnetizationSource2DIntegrator &primal_integ,
    std::map<std::string, FiniteElementState> &fields,
    std::map<std::string, mfem::ParLinearForm> &rev_sens,
    std::map<std::string, mfem::ParNonlinearForm> &rev_scalar_sens,
    std::map<std::string, mfem::ParLinearForm> &fwd_sens,
    std::map<std::string, mfem::ParNonlinearForm> &fwd_scalar_sens)
{
   ///TODO: Determine if need to emplace temperature field
      
   auto &mesh_fes = fields.at("mesh_coords").space();
   rev_sens.emplace("mesh_coords", &mesh_fes);
   rev_sens.at("mesh_coords")
       .AddDomainIntegrator(new MagnetizationSource2DIntegratorMeshRevSens(
           fields.at("adjoint").gridFunc(),
           primal_integ,
           &fields.at("temperature").gridFunc()));

   ///TODO: Dermine if the below is correct for the temperature sensitivity
   auto &temperature_fes = fields.at("temperature").space();
   rev_sens.emplace("temperature", &temperature_fes);
   rev_sens.at("temperature").AddDomainIntegrator(
       new MagnetizationSource2DIntegratorTemperatureSens(fields.at("adjoint").gridFunc(),
                                                         primal_integ,
                                                         &fields.at("temperature").gridFunc()));   
}

/// Integrator for (\nu(u)*curl u, curl v) for Nedelec elements
class CurlCurlNLFIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   /// Construct a curl curl nonlinear form integrator for Nedelec elements
   /// \param[in] m - model describing nonlinear material parameter
   /// \param[in] a - used to move to lhs or rhs
   CurlCurlNLFIntegrator(StateCoefficient &m, double a = 1.0)
    : model(m), alpha(a)
   { }

   /// Construct the element local residual
   /// \param[in] el - the finite element whose residual we want
   /// \param[in] trans - defines the reference to physical element mapping
   /// \param[in] elfun - element local state vector
   /// \param[out] elvect - element local residual
   void AssembleElementVector(const mfem::FiniteElement &el,
                              mfem::ElementTransformation &trans,
                              const mfem::Vector &elfun,
                              mfem::Vector &elvect) override;

   /// Construct the element local Jacobian
   /// \param[in] el - the finite element whose Jacobian we want
   /// \param[in] trans - defines the reference to physical element mapping
   /// \param[in] elfun - element local state vector
   /// \param[out] elmat - element local Jacobian
   void AssembleElementGrad(const mfem::FiniteElement &el,
                            mfem::ElementTransformation &trans,
                            const mfem::Vector &elfun,
                            mfem::DenseMatrix &elmat) override;

private:
   /// material (thus mesh) dependent model describing electromagnetic behavior
   StateCoefficient &model;
   /// scales the terms; can be used to move to rhs/lhs
   double alpha;

#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix curlshape, curlshape_dFt;
   // mfem::Vector b_vec, b_hat, temp_vec;
   mfem::Vector scratch;
#endif
   friend class CurlCurlNLFIntegratorMeshRevSens;
};

class CurlCurlNLFIntegratorStateRevSens : public mfem::LinearFormIntegrator
{
public:
   CurlCurlNLFIntegratorStateRevSens(mfem::GridFunction &state,
                                     mfem::GridFunction &adjoint,
                                     CurlCurlNLFIntegrator &integ)
    : state(state), adjoint(adjoint), integ(integ)
   { }

   void AssembleRHSElementVect(const mfem::FiniteElement &el,
                               mfem::ElementTransformation &trans,
                               mfem::Vector &state_bar) override;

private:
   /// the state to use when evaluating d(psi^T R)/du
   mfem::GridFunction &state;
   /// the adjoint to use when evaluating d(psi^T R)/du
   mfem::GridFunction &adjoint;
   /// reference to primal integrator
   CurlCurlNLFIntegrator &integ;

   mfem::Array<int> vdofs;
   mfem::Vector elfun, psi;
};

class CurlCurlNLFIntegratorStateFwdSens : public mfem::LinearFormIntegrator
{
public:
   CurlCurlNLFIntegratorStateFwdSens(mfem::GridFunction &state,
                                     mfem::GridFunction &state_dot,
                                     CurlCurlNLFIntegrator &integ)
    : state(state), state_dot(state_dot), integ(integ)
   { }

   void AssembleRHSElementVect(const mfem::FiniteElement &el,
                               mfem::ElementTransformation &trans,
                               mfem::Vector &res_dot) override;

private:
   /// the state to use when evaluating (dR/du) * state_dot
   mfem::GridFunction &state;
   /// the state_dot to use when evaluating (dR/du) * state_dot
   mfem::GridFunction &state_dot;
   /// reference to primal integrator
   CurlCurlNLFIntegrator &integ;

   mfem::Array<int> vdofs;
   mfem::Vector elfun, elfun_dot;
};

/// Integrator to assemble d(psi^T R)/dX for the CurlCurlNLFIntegrator
class CurlCurlNLFIntegratorMeshRevSens : public mfem::LinearFormIntegrator
{
public:
   /// \param[in] state - the state to use when evaluating d(psi^T R)/dX
   /// \param[in] adjoint - the adjoint to use when evaluating d(psi^T R)/dX
   /// \param[in] integ - reference to primal integrator
   CurlCurlNLFIntegratorMeshRevSens(mfem::GridFunction &state,
                                    mfem::GridFunction &adjoint,
                                    CurlCurlNLFIntegrator &integ)
    : state(state), adjoint(adjoint), integ(integ)
   { }

   /// \brief - assemble an element's contribution to d(psi^T R)/dX
   /// \param[in] el - the finite element that describes the mesh element
   /// \param[in] trans - the transformation between reference and physical
   /// space
   /// \param[out] mesh_coords_bar - d(psi^T R)/dX for the element
   /// \note the LinearForm that assembles this integrator's FiniteElementSpace
   /// MUST be the mesh's nodal finite element space
   void AssembleRHSElementVect(const mfem::FiniteElement &el,
                               mfem::ElementTransformation &trans,
                               mfem::Vector &mesh_coords_bar) override;

private:
   /// the state to use when evaluating d(psi^T R)/dX
   mfem::GridFunction &state;
   /// the adjoint to use when evaluating d(psi^T R)/dX
   mfem::GridFunction &adjoint;
   /// reference to primal integrator
   CurlCurlNLFIntegrator &integ;
#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix curlshape_dFt_bar;
   mfem::DenseMatrix PointMat_bar;
   mfem::Array<int> vdofs;
   mfem::Vector elfun, psi;
#endif
};

inline void addSensitivityIntegrator(
    CurlCurlNLFIntegrator &primal_integ,
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
       new CurlCurlNLFIntegratorStateRevSens(fields.at("state").gridFunc(),
                                             fields.at("adjoint").gridFunc(),
                                             primal_integ));

   fields.emplace(std::piecewise_construct,
                  std::forward_as_tuple("state_dot"),
                  std::forward_as_tuple(state.mesh(), state_fes));

   fwd_sens.emplace("state", &state_fes);
   fwd_sens.at("state").AddDomainIntegrator(
       new CurlCurlNLFIntegratorStateFwdSens(fields.at("state").gridFunc(),
                                             fields.at("state_dot").gridFunc(),
                                             primal_integ));

   auto &mesh_fes = fields.at("mesh_coords").space();
   rev_sens.emplace("mesh_coords", &mesh_fes);
   rev_sens.at("mesh_coords")
       .AddDomainIntegrator(
           new CurlCurlNLFIntegratorMeshRevSens(fields.at("state").gridFunc(),
                                                fields.at("adjoint").gridFunc(),
                                                primal_integ));
}

/// Integrator for (\nu(u) M, curl v) for Nedelec Elements
class MagnetizationIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   /// Construct a curl curl nonlinear form integrator for Nedelec elements
   /// \param[in] m - model describing nonlinear material parameter
   /// \param[in] a - used to move to lhs or rhs
   MagnetizationIntegrator(StateCoefficient *nu,
                           mfem::VectorCoefficient *M,
                           double a = 1.0)
    : nu(nu), mag(M), alpha(a)
   { }

   /// Construct the element local residual
   /// \param[in] el - the finite element whose residual we want
   /// \param[in] trans - defines the reference to physical element mapping
   /// \param[in] elfun - element local state vector
   /// \param[out] elvect - element local residual
   void AssembleElementVector(const mfem::FiniteElement &el,
                              mfem::ElementTransformation &trans,
                              const mfem::Vector &elfun,
                              mfem::Vector &elvect) override;

   /// Construct the element local Jacobian
   /// \param[in] el - the finite element whose Jacobian we want
   /// \param[in] trans - defines the reference to physical element mapping
   /// \param[in] elfun - element local state vector
   /// \param[out] elmat - element local Jacobian
   void AssembleElementGrad(const mfem::FiniteElement &el,
                            mfem::ElementTransformation &trans,
                            const mfem::Vector &elfun,
                            mfem::DenseMatrix &elmat) override;

private:
   /// material (thus mesh) dependent model for reluvtivity
   StateCoefficient *nu;
   /// material thus mesh dependent model for Magnetization
   mfem::VectorCoefficient *mag;
   /// scales the terms; can be used to move to rhs/lhs
   double alpha;

#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix curlshape, curlshape_dFt;
   mfem::Vector b_vec, mag_vec, temp_vec, temp_vec2;
#endif
};

/** moved/replaced in mfem_common_integ.xpp
class VectorFECurldJdXIntegerator : public mfem::LinearFormIntegrator
{
public:
   /// Construct a curl curl nonlinear form integrator for Nedelec elements
   /// \param[in] nu - coefficient describing reluctivity
   /// \param[in] state - the state to use when evaluating
   ///                    \frac{\partial psi^T R}{\partial X}
   /// \param[in] adjoint - the adjoint to use when evaluating
   ///                      \frac{\partial psi^T R}{\partial X}
   /// \param[in] alpha - used to move the terms to the LHS/RHS
   /// \note it is assumed that `state` is an a H(div) finite element space and
   ///       that adjoint is in a H(curl) finite element space
   VectorFECurldJdXIntegerator(mfem::Coefficient *_nu,
                               const mfem::GridFunction *_state,
                               const mfem::GridFunction *_adjoint,
                               mfem::VectorCoefficient *_vec_coeff = nullptr,
                               const double _alpha = 1.0)
      : nu(_nu), state(_state), adjoint(_adjoint), vec_coeff(_vec_coeff),
        alpha(_alpha) {};

   /// \brief - assemble an element's contribution to
   ///          \frac{\partial psi^T R}{\partial X}, needed for finding the
total
   ///          derivative of a functional with respect to the mesh nodes
   /// \param[in] el - the finite element that describes the mesh element
   /// \param[in] trans - the transformation between reference and physical
space
   /// \param[out] elvect - \frac{\partial J}{\partial X} for the element
   /// \note this is the `LinearFormIntegrator` component, the LinearForm that
   ///       assembles this integrator's FiniteElementSpace MUST be the mesh's
   ///       nodal finite element space
   void AssembleRHSElementVect(const mfem::FiniteElement &el,
                               mfem::ElementTransformation &trans,
                               mfem::Vector &elvect) override;

private:
   /// material (thus mesh) dependent model describing electromagnetic behavior
   mfem::Coefficient *nu;
   /// the state to use when evaluating \frac{\partial psi^T R}{\partial X}
   const mfem::GridFunction *state;
   /// the adjoint to use when evaluating \frac{\partial psi^T R}{\partial X}
   const mfem::GridFunction *adjoint;
   /// the coefficient that was projected to the GridFunction that is state
   mfem::VectorCoefficient *vec_coeff;
   /// to move the terms to the LHS or RHS
   const double alpha;

#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix curlshape, curlshape_dFt;
   mfem::DenseMatrix vshape, vshape_dFt;
   mfem::Vector m_vec, m_hat, curl_psi, curl_psi_hat, temp_vec;
#endif

};
*/

/** moved/replaced in mfem_common_integ.xpp
class VectorFEMassdJdXIntegerator : public mfem::LinearFormIntegrator
{
public:
   /// Construct a curl curl nonlinear form integrator for Nedelec elements
   /// \param[in] state - the state to use when evaluating
   ///                    \frac{\partial psi^T R}{\partial X}
   /// \param[in] adjoint - the adjoint to use when evaluating
   ///                      \frac{\partial psi^T R}{\partial X}
   /// \param[in] alpha - used to move the terms to the LHS/RHS
   /// \note it is assumed that both `state` and `adjoint` are in an H(curl)
   ///       finite element space
   VectorFEMassdJdXIntegerator(const mfem::GridFunction *_state,
                               const mfem::GridFunction *_adjoint,
                               mfem::VectorCoefficient *_vec_coeff = nullptr,
                               const double _alpha = 1.0)
      : state(_state), adjoint(_adjoint), vec_coeff(_vec_coeff), alpha(_alpha)
{};

   /// \brief - assemble an element's contribution to
   ///          \frac{\partial psi^T R}{\partial X}, needed for finding the
total
   ///          derivative of a functional with respect to the mesh nodes
   /// \param[in] el - the finite element that describes the mesh element
   /// \param[in] trans - the transformation between reference and physical
space
   /// \param[out] elvect - \frac{\partial J}{\partial X} for the element
   /// \note this is the `LinearFormIntegrator` component, the LinearForm that
   ///       assembles this integrator's FiniteElementSpace MUST be the mesh's
   ///       nodal finite element space
   void AssembleRHSElementVect(const mfem::FiniteElement &el,
                               mfem::ElementTransformation &trans,
                               mfem::Vector &elvect) override;

private:
   /// the state to use when evaluating \frac{\partial psi^T R}{\partial X}
   const mfem::GridFunction *state;
   /// the adjoint to use when evaluating \frac{\partial psi^T R}{\partial X}
   const mfem::GridFunction *adjoint;
   /// the coefficient that was projected to the GridFunction that is state
   mfem::VectorCoefficient *vec_coeff;
   /// to move the terms to the LHS or RHS
   const double alpha;

#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix vshape, vshape_dFt;
   mfem::Vector v_j_hat, v_j_vec, v_psi_hat, v_psi_vec;
#endif

};
*/

/** moved/replaced in mfem_common_integ.xpp
class VectorFEWeakDivergencedJdXIntegrator : public mfem::LinearFormIntegrator
{
public:
   /// Construct a curl curl nonlinear form integrator for Nedelec elements
   /// \param[in] state - the state to use when evaluating
   ///                    \frac{\partial psi^T R}{\partial X}
   /// \param[in] adjoint - the adjoint to use when evaluating
   ///                      \frac{\partial psi^T R}{\partial X}
   /// \note it is assumed that `state` is an a H(curl) finite element space
   ///       and that adjoint is in a H1 finite element space
   VectorFEWeakDivergencedJdXIntegrator(const mfem::GridFunction *_state,
                                        const mfem::GridFunction *_adjoint,
                                        mfem::VectorCoefficient *_vec_coeff =
nullptr, const double _alpha = 1.0) : state(_state), adjoint(_adjoint),
vec_coeff(_vec_coeff), alpha(_alpha) {};

   /// \brief - assemble an element's contribution to
   ///          \frac{\partial psi^T R}{\partial X}, needed for finding the
total
   ///          derivative of a functional with respect to the mesh nodes
   /// \param[in] el - the finite element that describes the mesh element
   /// \param[in] trans - the transformation between reference and physical
space
   /// \param[out] elvect - \frac{\partial J}{\partial X} for the element
   /// \note this is the `LinearFormIntegrator` component, the LinearForm that
   ///       assembles this integrator's FiniteElementSpace MUST be the mesh's
   ///       nodal finite element space
   void AssembleRHSElementVect(const mfem::FiniteElement &el,
                               mfem::ElementTransformation &trans,
                               mfem::Vector &elvect) override;

private:
   /// the state to use when evaluating \frac{\partial psi^T R}{\partial X}
   const mfem::GridFunction *state;
   /// the adjoint to use when evaluating \frac{\partial psi^T R}{\partial X}
   const mfem::GridFunction *adjoint;
   /// the coefficient that was projected to the GridFunction that is state
   mfem::VectorCoefficient *vec_coeff;
   /// used to move terms to LHS or RHS
   const double alpha;

#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix dshape, dshape_dFt;
   mfem::DenseMatrix vshape, vshape_dFt;
   mfem::Vector v_vec, v_hat, d_psi, d_psi_hat;
#endif

};
*/

/** moved/replaced in mfem_common_integ.xpp
class VectorFEDomainLFMeshSensInteg : public mfem::LinearFormIntegrator
{
public:
   VectorFEDomainLFMeshSensInteg(const mfem::GridFunction *_adjoint,
                                 mfem::VectorCoefficient &vc,
                                 double _alpha = 1.0)
   : adjoint(_adjoint), vec_coeff(vc), alpha(_alpha) {}

   /// \brief - assemble an element's contribution to
   ///          \frac{\partial psi^T LF}{\partial X}, needed for finding the
   ///          total derivative of a functional with respect to the mesh nodes
   /// \param[in] el - the finite element that describes the mesh element
   /// \param[in] trans - the transformation between reference and physical
space
   /// \param[out] elvect - \frac{\partial psi^T LF}{\partial X} for the element
   /// \note the LinearForm that assembles this integrator's FiniteElementSpace
   ///       MUST be the mesh's nodal finite element space
   void AssembleRHSElementVect(const mfem::FiniteElement &el,
                               mfem::ElementTransformation &trans,
                               mfem::Vector &elvect) override;
private:
   /// the adjoint to use when evaluating \frac{\partial psi^T R}{\partial X}
   const mfem::GridFunction *adjoint;
   /// the coefficient that was used for the linear form
   mfem::VectorCoefficient &vec_coeff;
   /// to move the terms to the LHS or RHS
   const double alpha;
#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix vshape, vshape_dFt;
   mfem::Vector v_psi_vec, v_psi_hat;
#endif

};
*/

/** moved/replaced in mfem_common_integ.xpp
/// TODO: Move this somewhere else to a common integrators spot
class GridFuncMeshSensIntegrator : public mfem::LinearFormIntegrator
{
public:
   /// An integrator to compute the mesh sensitivity of \psi^T GF
   /// \param[in] adjoint - the adjoint to use when evaluating
   ///                      \frac{\partial \psi^T GF}{\partial X}
   /// \note it is assumed that the adjoint and grid function are in the same
   /// finite element space
   GridFuncMeshSensIntegrator(mfem::GridFunction *_adjoint,
                              mfem::VectorCoefficient *_vec_coeff,
                              double _alpha = 1.0)
      : adjoint(_adjoint), vec_coeff(_vec_coeff), alpha(_alpha) {};

   /// \brief - assemble an element's contribution to
   ///          \frac{\partial psi^T R}{\partial X}, needed for finding the
total
   ///          derivative of a functional with respect to the mesh nodes
   /// \param[in] el - the finite element that describes the mesh element
   /// \param[in] trans - the transformation between reference and physical
space
   /// \param[out] elvect - \frac{\partial J}{\partial X} for the element
   /// \note this is the `LinearFormIntegrator` component, the LinearForm that
   ///       assembles this integrator's FiniteElementSpace MUST be the mesh's
   ///       nodal finite element space
   void AssembleRHSElementVect(const mfem::FiniteElement &el,
                               mfem::ElementTransformation &trans,
                               mfem::Vector &elvect) override;

private:
   /// the adjoint to use when evaluating \frac{\partial psi^T R}{\partial X}
   mfem::GridFunction *adjoint;
   /// the coefficient that was projected to the GridFunction that is state
   mfem::VectorCoefficient *vec_coeff;
   /// used to move terms to LHS or RHS
   double alpha;

#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix dshape, dshape_dFt;
   mfem::DenseMatrix vshape, vshape_dFt;
   mfem::Vector v_vec, v_hat, d_psi, d_psi_hat;
#endif

};
*/

/// Integrator to compute the magnetic energy
class MagneticEnergyIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   /// \param[in] nu - model describing reluctivity
   MagneticEnergyIntegrator(StateCoefficient &nu) : nu(nu) { }

   /// \param[in] el - the finite element
   /// \param[in] trans - defines the reference to physical element mapping
   /// \param[in] elfun - state vector of the element
   double GetElementEnergy(const mfem::FiniteElement &el,
                           mfem::ElementTransformation &trans,
                           const mfem::Vector &elfun) override;

   /// \brief - assemble an element's contribution to
   ///          \frac{\partial J}{\partial u}, needed to solve for the adjoint
   /// \param[in] el - the finite element
   /// \param[in] trans - defines the reference to physical element mapping
   /// \param[in] elfun - state vector of the element
   /// \param[out] elvect - \partial J \partial u for this functional
   void AssembleElementVector(const mfem::FiniteElement &el,
                              mfem::ElementTransformation &trans,
                              const mfem::Vector &elfun,
                              mfem::Vector &elfun_bar) override;

private:
   /// material (thus mesh) dependent model describing reluctivity
   StateCoefficient &nu;
#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix curlshape, curlshape_dFt;
   mfem::Vector b_vec;
#endif
   /// class that implements mesh sensitivities for MagneticEnergyIntegrator
   friend class MagneticEnergyIntegratorMeshSens;
};

class MagneticEnergyIntegratorMeshSens : public mfem::LinearFormIntegrator
{
public:
   /// \brief - Compute energy stored in magnetic field mesh sensitivity
   /// \param[in] state - the state vector to evaluate force at
   /// \param[in] integ - reference to primal integrator that holds inputs for
   /// integrators
   MagneticEnergyIntegratorMeshSens(mfem::GridFunction &state,
                                    MagneticEnergyIntegrator &integ)
    : state(state), integ(integ)
   { }

   /// \brief - assemble an element's contribution to dJdX
   /// \param[in] mesh_el - the finite element that describes the mesh element
   /// \param[in] mesh_trans - the transformation between reference and physical
   /// space \param[out] mesh_coords_bar - dJdX for the element
   void AssembleRHSElementVect(const mfem::FiniteElement &mesh_el,
                               mfem::ElementTransformation &mesh_trans,
                               mfem::Vector &mesh_coords_bar) override;

private:
   /// state vector for evaluating energy
   mfem::GridFunction &state;
   /// reference to primal integrator
   MagneticEnergyIntegrator &integ;
#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix curlshape_dFt_bar;
   mfem::DenseMatrix PointMat_bar;
   mfem::Array<int> vdofs;
   mfem::Vector elfun;
#endif
};

inline void addDomainSensitivityIntegrator(
    MagneticEnergyIntegrator &primal_integ,
    std::map<std::string, FiniteElementState> &fields,
    std::map<std::string, mfem::ParLinearForm> &output_sens,
    std::map<std::string, mfem::ParNonlinearForm> &output_scalar_sens,
    mfem::Array<int> *attr_marker)
{
   auto &mesh_fes = fields.at("mesh_coords").space();
   output_sens.emplace("mesh_coords", &mesh_fes);

   if (attr_marker == nullptr)
   {
      output_sens.at("mesh_coords")
          .AddDomainIntegrator(new MagneticEnergyIntegratorMeshSens(
              fields.at("state").gridFunc(), primal_integ));
   }
   else
   {
      output_sens.at("mesh_coords")
          .AddDomainIntegrator(new MagneticEnergyIntegratorMeshSens(
                                   fields.at("state").gridFunc(), primal_integ),
                               *attr_marker);
   }
}

/** commenting out co-energy stuff since I'm stopping maintaining it
/// Integrator to compute the magnetic co-energy
class MagneticCoenergyIntegrator : public mfem::NonlinearFormIntegrator,
                                   public mfem::LinearFormIntegrator
{
public:
   /// \param[in] state - the current state (A)
   /// \param[in] nu - model describing reluctivity
   MagneticCoenergyIntegrator(mfem::GridFunction &_state, StateCoefficient *_nu)
      : state(_state), nu(_nu) {};

   /// \param[in] el - the finite element
   /// \param[in] trans - defines the reference to physical element mapping
   /// \param[in] elfun - state vector of the element
   /// \returns the magnetic co-energy calculated over an element
   double GetElementEnergy(const mfem::FiniteElement &el,
                           mfem::ElementTransformation &trans,
                           const mfem::Vector &elfun) override;

   /// \brief - assemble an element's contribution to
   ///          \frac{\partial J}{\partial u}, needed to solve for the adjoint
   /// \param[in] el - the finite element
   /// \param[in] trans - defines the reference to physical element mapping
   /// \param[in] elfun - state vector of the element
   /// \param[out] elvect - \partial J \partial u for this functional
   void AssembleElementVector(const mfem::FiniteElement &el,
                              mfem::ElementTransformation &trans,
                              const mfem::Vector &elfun,
                              mfem::Vector &elvect) override;

   /// \brief - assemble an element's contribution to
   ///          \frac{\partial J}{\partial X}, needed for finding the total
   ///          derivative of the functional with respect to the mesh nodes
   /// \param[in] el - the finite element that describes the mesh element
   /// \param[in] trans - the transformation between reference and physical
space
   /// \param[out] elvect - \frac{\partial J}{\partial X} for the element
   /// \note this is the `LinearFormIntegrator` component, the LinearForm that
   ///       assembles this integrator's FiniteElementSpace MUST be the mesh's
   ///       nodal finite element space
   void AssembleRHSElementVect(const mfem::FiniteElement &el,
                               mfem::ElementTransformation &trans,
                               mfem::Vector &elvect) override;

   friend void setInput(MagneticCoenergyIntegrator &integ,
                        const std::string &name,
                        const MachInput &input);

private:
   /// the current state to use when evaluating \frac{\partial J}{\partial X}
   mfem::GridFunction &state;
   /// material (thus mesh) dependent model describing reluctivity
   StateCoefficient *nu;
#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix curlshape, curlshape_dFt, M;
   mfem::Vector b_vec, b_hat, temp_vec;
#endif

   double integrateBH(const mfem::IntegrationRule *ir,
                      mfem::ElementTransformation &trans,
                      const mfem::IntegrationPoint &old_ip,
                      double lower_bound,
                      double upper_bound);

   double FDintegrateBH(const mfem::IntegrationRule *ir,
                        mfem::ElementTransformation &trans,
                        const mfem::IntegrationPoint &old_ip,
                        double lower_bound,
                        double upper_bound);

   double RevADintegrateBH(const mfem::IntegrationRule *ir,
                           mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &old_ip,
                           double lower_bound,
                           double upper_bound);
};

inline void setInput(MagneticCoenergyIntegrator &integ,
         const std::string &name,
         const MachInput &input)
{
   // do nothing yet
}
*/

/// Integrator to compute the norm of the magnetic field
class BNormIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   BNormIntegrator() = default;

   /// \param[in] el - the finite element
   /// \param[in] trans - defines the reference to physical element mapping
   /// \param[in] elfun - state vector of the element
   /// \returns the norm of the magnetic field calculated over an element
   double GetElementEnergy(const mfem::FiniteElement &el,
                           mfem::ElementTransformation &trans,
                           const mfem::Vector &elfun) override;

   /// \brief - Computes dJdu, for solving for the adjoint
   /// \param[in] el - the finite element
   /// \param[in] trans - defines the reference to physical element mapping
   /// \param[in] elfun - state vector of the element
   /// \param[out] elvect - \partial J \partial u for this functional
   void AssembleElementVector(const mfem::FiniteElement &el,
                              mfem::ElementTransformation &trans,
                              const mfem::Vector &elfun,
                              mfem::Vector &elvect) override;

private:
#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix curlshape, curlshape_dFt, M;
   mfem::Vector b_vec, temp_vec;
#endif
};

/// Integrator to compute the norm of the magnetic field
class BNormSquaredIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   /// \param[in] el - the finite element
   /// \param[in] trans - defines the reference to physical element mapping
   /// \param[in] elfun - state vector of the element
   /// \returns the norm of the magnetic field calculated over an element
   double GetElementEnergy(const mfem::FiniteElement &el,
                           mfem::ElementTransformation &trans,
                           const mfem::Vector &elfun) override;

private:
#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix curlshape, curlshape_dFt;
#endif
};

class BNormdJdx : public mfem::LinearFormIntegrator
{
public:
   /// \brief - linear form integrator to assemble the vector
   ///          \frac{\partial J}{\partial X}
   /// \param[in] state - the current state (A)
   /// \note the finite element space used to by the linear form that assembles
   ///       this integrator will use the mesh's nodal finite element space
   BNormdJdx(mfem::GridFunction &_state) : state(_state) { }

   /// \brief - assemble an element's contribution to \frac{\partial J}{\partial
   /// X} \param[in] el - the finite element that describes the mesh element
   /// \param[in] trans - the transformation between reference and physical
   /// space \param[out] elvect - \frac{\partial J}{\partial X} for the element
   void AssembleRHSElementVect(const mfem::FiniteElement &el,
                               mfem::ElementTransformation &trans,
                               mfem::Vector &elvect) override;

private:
   /// the current state to use when evaluating \frac{\partial J}{\partial X}
   mfem::GridFunction &state;
#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix curlshape, curlshape_dFt, M;
   mfem::Vector b_vec, temp_vec;
#endif
};

/// Integrator to compute the magnetic co-energy
class nuBNormIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   /// \param[in] nu - model describing reluctivity
   nuBNormIntegrator(StateCoefficient *_nu) : nu(_nu) { }

   /// \param[in] el - the finite element
   /// \param[in] trans - defines the reference to physical element mapping
   /// \param[in] elfun - state vector of the element
   /// \returns the magnetic co-energy calculated over an element
   double GetElementEnergy(const mfem::FiniteElement &el,
                           mfem::ElementTransformation &trans,
                           const mfem::Vector &elfun) override;

   /// \brief - Computes dJdu, for solving for the adjoint
   /// \param[in] el - the finite element
   /// \param[in] trans - defines the reference to physical element mapping
   /// \param[in] elfun - state vector of the element
   /// \param[out] elvect - \partial J \partial u for this functional
   void AssembleElementVector(const mfem::FiniteElement &el,
                              mfem::ElementTransformation &trans,
                              const mfem::Vector &elfun,
                              mfem::Vector &elvect) override;

private:
   /// material (thus mesh) dependent model describing reluctivity
   StateCoefficient *nu;
#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix curlshape, curlshape_dFt, M;
   mfem::Vector b_vec, temp_vec;
#endif
};

class nuBNormdJdx : public mfem::LinearFormIntegrator
{
public:
   /// \brief - linear form integrator to assemble the vector
   ///          \frac{\partial J}{\partial X}
   /// \param[in] state - the current state (A)
   /// \note the finite element space used to by the linear form that assembles
   ///       this integrator will use the mesh's nodal finite element space
   nuBNormdJdx(mfem::GridFunction &_state, StateCoefficient *_nu)
    : state(_state), nu(_nu)
   { }

   /// \brief - assemble an element's contribution to \frac{\partial J}{\partial
   /// X} \param[in] el - the finite element that describes the mesh element
   /// \param[in] trans - the transformation between reference and physical
   /// space \param[out] elvect - \frac{\partial J}{\partial X} for the element
   void AssembleRHSElementVect(const mfem::FiniteElement &el,
                               mfem::ElementTransformation &trans,
                               mfem::Vector &elvect) override;

private:
   /// the current state to use when evaluating \frac{\partial J}{\partial X}
   mfem::GridFunction &state;
   /// material (thus mesh) dependent model describing reluctivity
   StateCoefficient *nu;
#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix curlshape, curlshape_dFt, M;
   mfem::Vector b_vec, temp_vec;
#endif
};

/// Integrator to compute the functional `\nu` over the domain
class nuFuncIntegrator : public mfem::NonlinearFormIntegrator,
                         public mfem::LinearFormIntegrator
{
public:
   /// \param[in] nu - model describing reluctivity
   nuFuncIntegrator(StateCoefficient *_nu) : state(nullptr), nu(_nu) { }

   /// \brief - linear form integrator to assemble the vector
   ///          \frac{\partial J}{\partial X}
   /// \param[in] state - the current state (A)
   /// \note the finite element space used to by the linear form that assembles
   ///       this integrator will use the mesh's nodal finite element space
   nuFuncIntegrator(mfem::GridFunction *_state, StateCoefficient *_nu)
    : state(_state), nu(_nu)
   { }

   /// \param[in] el - the finite element
   /// \param[in] trans - defines the reference to physical element mapping
   /// \param[in] elfun - state vector of the element
   /// \returns the magnetic co-energy calculated over an element
   double GetElementEnergy(const mfem::FiniteElement &el,
                           mfem::ElementTransformation &trans,
                           const mfem::Vector &elfun) override;

   /// \brief - Computes dJdu, for solving for the adjoint
   /// \param[in] el - the finite element
   /// \param[in] trans - defines the reference to physical element mapping
   /// \param[in] elfun - state vector of the element
   /// \param[out] elvect - \partial J \partial u for this functional
   /// \note - not implemented
   // void AssembleElementVector(const mfem::FiniteElement &el,
   //                            mfem::ElementTransformation &trans,
   //                            const mfem::Vector &elfun,
   //                            mfem::Vector &elvect) override;

   /// \brief - assemble an element's contribution to \frac{\partial J}{\partial
   /// X} \param[in] el - the finite element that describes the mesh element
   /// \param[in] trans - the transformation between reference and physical
   /// space \param[out] elvect - \frac{\partial J}{\partial X} for the element
   void AssembleRHSElementVect(const mfem::FiniteElement &el,
                               mfem::ElementTransformation &trans,
                               mfem::Vector &elvect) override;

private:
   /// the current state to use when evaluating \frac{\partial J}{\partial X}
   mfem::GridFunction *state;
   /// material (thus mesh) dependent model describing reluctivity
   StateCoefficient *nu;
#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix curlshape, curlshape_dFt, M;
   mfem::Vector b_vec, temp_vec;
#endif
};

/// Integrator to compute the sensitivity of the thermal residual to the vector
/// potential
class ThermalSensIntegrator : public mfem::NonlinearFormIntegrator,
                              public mfem::LinearFormIntegrator
{
public:
   /// Constructor, expected coefficient is SteinmetzVectorDiffCoefficient
   ThermalSensIntegrator(mfem::VectorCoefficient &Q,
                         mfem::GridFunction *_adjoint,
                         int a = 2,
                         int b = 0)
    : Q(Q), oa(a), ob(b), adjoint(_adjoint)
   { }

   /// \brief - assemble an element's contribution to
   ///          \frac{\partial psi^T R}{\partial A}
   /// \param[in] el - the finite element that describes the Nedelec element
   /// \param[in] trans - the transformation between reference and physical
   /// \param[out] elvect - \frac{\partial psi^T R}{\partial A} for the element
   /// \note LinearForm that assembles this integrator's FiniteElementSpace
   ///       MUST be the magnetic vector potential's finite element space
   void AssembleRHSElementVect(const mfem::FiniteElement &nd_el,
                               mfem::ElementTransformation &nd_trans,
                               mfem::Vector &elvect) override;

private:
   /// vector coefficient that evaluates dQ/dA
   mfem::VectorCoefficient &Q;
   int oa, ob;
   // thermal adjoint
   mfem::GridFunction *adjoint;
#ifndef MFEM_THREAD_SAFE
   mfem::Vector shape;
#endif
};

/// Functional integrator to compute DC copper losses
class DCLossFunctionalIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   /// \brief - Compute DC copper losses in the domain
   /// \param[in] sigma - the temperature dependent electrical conductivity coefficient // Made sigma a StateCoefficient (formerly mfem::Coefficient)
   /// \param[in] temperature_field - a pointer to the temperature field (default is null)
   DCLossFunctionalIntegrator(StateCoefficient &sigma, mfem::GridFunction *temperature_field=nullptr) 
   : sigma(sigma), temperature_field(temperature_field) { }

   /// \brief - Compute DC copper losses in the domain
   /// \param[in] el - the finite element
   /// \param[in] trans - defines the reference to physical element mapping
   /// \param[in] elfun - state vector of the element
   /// \returns the DC losses calculated over an element
   double GetElementEnergy(const mfem::FiniteElement &el,
                           mfem::ElementTransformation &trans,
                           const mfem::Vector &elfun) override;

private:
   // Made sigma a StateCoefficient (formerly mfem::Coefficient)
   StateCoefficient &sigma;

   mfem::GridFunction *temperature_field; // pointer to the temperature field

   ///TODO: Look into making code thread-safe
#ifndef MFEM_THREAD_SAFE
   mfem::Vector shape;
   mfem::Array<int> vdofs;
   mfem::Vector temp_elfun;
#endif

   // double rms_current;
   // double strand_radius;
   // double num_strands_in_hand;
   // double num_turns;
   friend class DCLossFunctionalIntegratorMeshSens;
};

class DCLossFunctionalIntegratorMeshSens : public mfem::LinearFormIntegrator
{
public:
   /// \param[in] state - the state grid function
   /// \param[in] integ - reference to primal integrator that holds inputs for
   /// integrator
   DCLossFunctionalIntegratorMeshSens(mfem::GridFunction &state,
                                      DCLossFunctionalIntegrator &integ)
    : state(state), integ(integ)
   { }

   /// \brief - assemble an element's contribution to dJdX
   /// \param[in] mesh_el - the finite element that describes the mesh element
   /// \param[in] mesh_trans - the transformation between reference and physical
   /// space
   /// \param[out] mesh_coords_bar - dJdX for the element
   void AssembleRHSElementVect(const mfem::FiniteElement &mesh_el,
                               mfem::ElementTransformation &mesh_trans,
                               mfem::Vector &mesh_coords_bar) override;

private:
   /// State GridFunction, needed to get integration order for each element
   mfem::GridFunction &state;
   /// reference to primal integrator
   DCLossFunctionalIntegrator &integ;

#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix PointMat_bar;
   mfem::Vector shape;
   mfem::Array<int> vdofs;
   mfem::Vector temp_elfun;
#endif
};

inline void addDomainSensitivityIntegrator(
    DCLossFunctionalIntegrator &primal_integ,
    std::map<std::string, FiniteElementState> &fields,
    std::map<std::string, mfem::ParLinearForm> &output_sens,
    std::map<std::string, mfem::ParNonlinearForm> &output_scalar_sens,
    mfem::Array<int> *attr_marker)
{
   auto &mesh_fes = fields.at("mesh_coords").space();
   output_sens.emplace("mesh_coords", &mesh_fes);

   if (attr_marker == nullptr)
   {
      output_sens.at("mesh_coords")
          .AddDomainIntegrator(new DCLossFunctionalIntegratorMeshSens(
              fields.at("state").gridFunc(), primal_integ));
   }
   else
   {
      output_sens.at("mesh_coords")
          .AddDomainIntegrator(new DCLossFunctionalIntegratorMeshSens(
                                   fields.at("state").gridFunc(), primal_integ),
                               *attr_marker);
   }
}

class DCLossFunctionalDistributionIntegrator : public mfem::LinearFormIntegrator
{
public:
   friend void setInputs(DCLossFunctionalDistributionIntegrator &integ,
                         const MachInputs &inputs);

   /// \param[in] el - the finite element
   /// \param[in] trans - defines the reference to physical element mapping
   /// \param[out] elvect - element local heat source distribution
   void AssembleRHSElementVect(const mfem::FiniteElement &el,
                               mfem::ElementTransformation &trans,
                               mfem::Vector &elvect) override;

   // Made sigma a StateCoefficient (formerly mfem::Coefficient)
   // Added a grid function pointer for the temperature field to signature
   DCLossFunctionalDistributionIntegrator(StateCoefficient &sigma,
                                          mfem::GridFunction *temperature_field=nullptr,
                                          std::string name = "")
    : sigma(sigma), temperature_field(temperature_field), name(std::move(name))
   { }

private:
   /// Temperature dependent Electrical conductivity // Made sigma a StateCoefficient (formerly mfem::Coefficient)
   StateCoefficient &sigma;
   
   mfem::GridFunction *temperature_field; // pointer to temperature field (can be null)

   // optional integrator name to differentiate setting inputs
   std::string name;

   /// Litz wire strand radius
   double strand_radius = 1.0;
   /// Total wire length
   double wire_length = 1.0;
   /// Number of strands in hand for litz wire
   double strands_in_hand = 1.0;
   /// RMS current
   double rms_current = 1.0;
   /// Stack length
   double stack_length = 1.0;

   ///TODO: Look into making code thread-safe
#ifndef MFEM_THREAD_SAFE
   mfem::Vector shape;
   mfem::Array<int> vdofs;
   mfem::Vector temp_elfun;
#endif

   /// class that implements mesh sensitivities for
   /// DCLossFunctionalDistributionIntegrator
   ///TODO: If necessary, define and implement this class
   friend class DCLossFunctionalDistributionIntegratorMeshSens;
};

/// Functional integrator to compute AC copper losses based on hybrid approach
/// (new)
class ACLossFunctionalIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   /// \brief - Compute AC copper losses in the domain based on a hybrid
   ///          analytical-FEM approach
   /// \param[in] sigma - the temperature dependent electrical conductivity coefficient // Made sigma a StateCoefficient (formerly an MFEM coefficient)
   /// \param[in] temperature_field - a pointer to the temperature field (default is null)
   ACLossFunctionalIntegrator(StateCoefficient &sigma, mfem::GridFunction *temperature_field=nullptr) 
   : sigma(sigma), temperature_field(temperature_field) { }


   /// \brief - Compute AC copper losses in the domain based on a hybrid
   ///          analytical-FEM approach
   /// \param[in] el - the finite element
   /// \param[in] trans - defines the reference to physical element mapping
   /// \param[in] elfun - state vector of the element
   /// \returns the AC losses calculated over an element
   double GetElementEnergy(const mfem::FiniteElement &el,
                           mfem::ElementTransformation &trans,
                           const mfem::Vector &elfun) override;

private:
   StateCoefficient &sigma; // Made sigma a StateCoefficient (formerly an MFEM coefficient)

   mfem::GridFunction *temperature_field; // temperature field, if it exists

#ifndef MFEM_THREAD_SAFE
   mfem::Vector shape;
   mfem::Vector temp_shape;
   mfem::Array<int> vdofs;
   mfem::Vector temp_elfun;
#endif

   friend class ACLossFunctionalIntegratorMeshSens;
   friend class ACLossFunctionalIntegratorPeakFluxSens;
};

class ACLossFunctionalIntegratorMeshSens : public mfem::LinearFormIntegrator
{
public:
   /// \param[in] peak_flux - the peak_flux grid function
   /// \param[in] integ - reference to primal integrator that holds inputs for
   /// integrator
   ACLossFunctionalIntegratorMeshSens(mfem::GridFunction &peak_flux,
                                      ACLossFunctionalIntegrator &integ)
    : peak_flux(peak_flux), integ(integ)
   { }

   /// \brief - assemble an element's contribution to dJdX
   /// \param[in] mesh_el - the finite element that describes the mesh element
   /// \param[in] mesh_trans - the transformation between reference and physical
   /// space
   /// \param[out] mesh_coords_bar - dJdX for the element
   void AssembleRHSElementVect(const mfem::FiniteElement &mesh_el,
                               mfem::ElementTransformation &mesh_trans,
                               mfem::Vector &mesh_coords_bar) override;

private:
   /// peak_flux GridFunction
   mfem::GridFunction &peak_flux;
   /// reference to primal integrator
   ACLossFunctionalIntegrator &integ;

#ifndef MFEM_THREAD_SAFE
   mfem::Array<int> vdofs;
   mfem::Vector elfun;
   mfem::Vector shape_bar;
   mfem::DenseMatrix PointMat_bar;
   mfem::Vector shape;
   mfem::Vector temp_shape;
   mfem::Vector temp_elfun;
#else
   auto &shape = integ.shape;
#endif
};

class ACLossFunctionalIntegratorPeakFluxSens : public mfem::LinearFormIntegrator
{
public:
   /// \param[in] peak_flux - the peak_flux grid function
   /// \param[in] integ - reference to primal integrator that holds inputs for
   /// integrator
   ACLossFunctionalIntegratorPeakFluxSens(mfem::GridFunction &peak_flux,
                                          ACLossFunctionalIntegrator &integ)
    : peak_flux(peak_flux), integ(integ)
   { }

   /// \brief - assemble an element's contribution to dJdX
   /// \param[in] pf_el - the finite element to integrate over
   /// \param[in] pf_trans - the transformation between reference and physical
   /// space
   /// \param[out] peak_flux_bar - dJdB for the element
   void AssembleRHSElementVect(const mfem::FiniteElement &pf_el,
                               mfem::ElementTransformation &pf_trans,
                               mfem::Vector &peak_flux_bar) override;

private:
   /// peak_flux GridFunction
   mfem::GridFunction &peak_flux;
   /// reference to primal integrator
   ACLossFunctionalIntegrator &integ;

#ifndef MFEM_THREAD_SAFE
   mfem::Array<int> vdofs;
   mfem::Vector elfun;
   mfem::Vector temp_elfun;
#endif
};

inline void addDomainSensitivityIntegrator(
    ACLossFunctionalIntegrator &primal_integ,
    std::map<std::string, FiniteElementState> &fields,
    std::map<std::string, mfem::ParLinearForm> &output_sens,
    std::map<std::string, mfem::ParNonlinearForm> &output_scalar_sens,
    mfem::Array<int> *attr_marker)
{
   auto &mesh_fes = fields.at("mesh_coords").space();
   output_sens.emplace("mesh_coords", &mesh_fes);

   auto &peak_flux_fes = fields.at("peak_flux").space();
   output_sens.emplace("peak_flux", &peak_flux_fes);

   if (attr_marker == nullptr)
   {
      output_sens.at("mesh_coords")
          .AddDomainIntegrator(new ACLossFunctionalIntegratorMeshSens(
              fields.at("peak_flux").gridFunc(), primal_integ));

      output_sens.at("peak_flux")
          .AddDomainIntegrator(new ACLossFunctionalIntegratorPeakFluxSens(
              fields.at("peak_flux").gridFunc(), primal_integ));
   }
   else
   {
      output_sens.at("mesh_coords")
          .AddDomainIntegrator(
              new ACLossFunctionalIntegratorMeshSens(
                  fields.at("peak_flux").gridFunc(), primal_integ),
              *attr_marker);
      output_sens.at("peak_flux")
          .AddDomainIntegrator(
              new ACLossFunctionalIntegratorPeakFluxSens(
                  fields.at("peak_flux").gridFunc(), primal_integ),
              *attr_marker);
   }
}

class ACLossFunctionalDistributionIntegrator : public mfem::LinearFormIntegrator
{
public:
   friend void setInputs(ACLossFunctionalDistributionIntegrator &integ,
                         const MachInputs &inputs);

   /// \param[in] el - the finite element
   /// \param[in] trans - defines the reference to physical element mapping
   /// \param[out] elvect - element local heat source distribution
   void AssembleRHSElementVect(const mfem::FiniteElement &el,
                               mfem::ElementTransformation &trans,
                               mfem::Vector &elvect) override;

   // Made sigma a StateCoefficient (formerly mfem::Coefficient)
   // Added a grid function pointer for the temperature field to signature
   ACLossFunctionalDistributionIntegrator(mfem::GridFunction &peak_flux,
                                          StateCoefficient &sigma,
                                          mfem::GridFunction *temperature_field=nullptr,
                                          std::string name = "")
    : peak_flux(peak_flux), sigma(sigma), temperature_field(temperature_field), name(std::move(name))
   { }

private:
   mfem::GridFunction &peak_flux;
   /// Temperature dependent Electrical conductivity // Made sigma a StateCoefficient (formerly mfem::Coefficient)
   StateCoefficient &sigma;

   mfem::GridFunction *temperature_field; // pointer to temperature field (can be null)

   // optional integrator name to differentiate setting inputs
   std::string name;

   /// Electrical excitation frequency
   double freq = 1.0;
   /// Litz wire strand radius
   double radius = 1.0;
   /// into the page length
   double stack_length = 1.0;
   /// Number of strands in hand for litz wire
   double strands_in_hand = 1.0;
   /// Number of turns of litz wire
   double num_turns = 1.0;
   /// Number of slots in motor
   double num_slots = 1.0;
#ifndef MFEM_THREAD_SAFE
   mfem::Vector shape;
   mfem::Vector flux_shape;
   mfem::Array<int> vdofs;
   mfem::Vector elfun;
   mfem::Vector temp_elfun;
#endif
   /// class that implements mesh sensitivities for
   /// ACLossFunctionalDistributionIntegrator
   ///TODO: If necessary, define and implement this class
   friend class ACLossFunctionalDistributionIntegratorMeshSens;
};

///NOTE: HybridACLossFunctionalIntegrator is a Dead class. Not putting any more time into
/// In cpp and hpp, Have only the immediate below HybridACLossFunctionalIntegrator uncommented for StateCoefficient logic for test_acloss_functional (this will be the one that remains)
/// Functional integrator to compute AC copper losses based on hybrid approach
// class HybridACLossFunctionalIntegrator : public mfem::NonlinearFormIntegrator
// {
// public:
//    /// \brief allows changing the frequency and diameter of the strands for AC
//    /// loss calculation
//    friend void setInputs(HybridACLossFunctionalIntegrator &integ,
//                          const MachInputs &inputs);

//    /// \brief - Compute AC copper losses in the domain based on a hybrid
//    ///          analytical-FEM approach
//    /// \param[in] sigma - the temperature dependent electrical conductivity coefficient
//    /// \param[in] freq - the electrical excitation frequency
//    /// \param[in] diam - the diameter of a strand in the bundle
//    /// \param[in] fill_factor - the density of strands in the bundle
//    /// \param[in] temperature_field - the temperature field that informs the conductivity
//    // Made sigma a StateCoefficient (was formerly an mfem::coefficient)
//    HybridACLossFunctionalIntegrator(StateCoefficient &sigma,
//                                     const double freq,
//                                     const double diam,
//                                     const double fill_factor,
//                                     mfem::GridFunction *temperature_field=nullptr)
//     : sigma(sigma), freq(freq), diam(diam), fill_factor(fill_factor), temperature_field(temperature_field)
//    { }

//    /// \brief - Compute AC copper losses in the domain based on a hybrid
//    ///          analytical-FEM approach
//    /// \param[in] el - the finite element
//    /// \param[in] trans - defines the reference to physical element mapping
//    /// \param[in] elfun - state vector of the element
//    /// \returns the AC losses calculated over an element
//    double GetElementEnergy(const mfem::FiniteElement &el,
//                            mfem::ElementTransformation &trans,
//                            const mfem::Vector &elfun) override;

// private:
//    // Made sigma a StateCoefficient (was formerly an mfem::coefficient)
//    StateCoefficient &sigma;
   
//    double freq;
//    double diam;
//    double fill_factor;

//    mfem::GridFunction *temperature_field; // pointer to temperature field (can be null)
   
// #ifndef MFEM_THREAD_SAFE
//    mfem::DenseMatrix curlshape, curlshape_dFt;
//    mfem::Vector b_vec;
//    mfem::Vector shape;
//    mfem::Array<int> vdofs;
//    mfem::Vector temp_elfun;
// #endif
// };

class ForceIntegrator3 : public mfem::NonlinearFormIntegrator
{
public:
   ForceIntegrator3(StateCoefficient &nu, mfem::GridFunction &v) : nu(nu), v(v)
   { }

   /// \brief - Compute forces/torques based on the virtual work method
   /// \param[in] nu - model describing reluctivity
   /// \param[in] v - the grid function containing virtual displacements for
   ///                each mesh node
   /// \param[in] attrs - the regions the force is acting on
   ForceIntegrator3(StateCoefficient &nu,
                    mfem::GridFunction &v,
                    std::unordered_set<int> attrs)
    : nu(nu), v(v), attrs(std::move(attrs))
   { }

   /// \brief - Compute element contribution to global force/torque
   /// \param[in] el - the finite element
   /// \param[in] trans - defines the reference to physical element mapping
   /// \param[in] elfun - state vector of the element
   /// \returns the element contribution to global force/torque
   double GetElementEnergy(const mfem::FiniteElement &el,
                           mfem::ElementTransformation &trans,
                           const mfem::Vector &elfun) override;

   /// \brief - Computes dJdu, for solving for the adjoint
   /// \param[in] el - the finite element
   /// \param[in] trans - defines the reference to physical element mapping
   /// \param[in] elfun - state vector of the element
   /// \param[out] elfun_bar - \partial J \partial u for this functional
   void AssembleElementVector(const mfem::FiniteElement &el,
                              mfem::ElementTransformation &trans,
                              const mfem::Vector &elfun,
                              mfem::Vector &elfun_bar) override;

private:
   /// material dependent model describing reluctivity
   StateCoefficient &nu;
   /// grid function containing virtual displacements for each mesh node
   mfem::GridFunction &v;
   /// set of attributes the force is acting on
   std::unordered_set<int> attrs;
#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix dshape;
   mfem::DenseMatrix curlshape, curlshape_dFt, curlshape_dFt_bar;
   mfem::DenseMatrix dBdX;
   mfem::Array<int> vdofs;
   mfem::Vector vfun;
#endif
   /// class that implements mesh sensitivities for ForceIntegrator
   friend class ForceIntegratorMeshSens3;
};

/// Linear form integrator to assemble the vector dJdX for the ForceIntegrator
class ForceIntegratorMeshSens3 : public mfem::LinearFormIntegrator
{
public:
   /// \brief - Compute forces/torques based on the virtual work method
   /// \param[in] state - the state vector to evaluate force at
   /// \param[in] integ - reference to primal integrator that holds inputs for
   /// integrators
   ForceIntegratorMeshSens3(mfem::GridFunction &state, ForceIntegrator3 &integ)
    : state(state), force_integ(integ)
   { }

   /// \brief - assemble an element's contribution to dJdX
   /// \param[in] el - the finite element that describes the mesh element
   /// \param[in] trans - the transformation between reference and physical
   /// space
   /// \param[out] mesh_coords_bar - dJdX for the element
   void AssembleRHSElementVect(const mfem::FiniteElement &el,
                               mfem::ElementTransformation &trans,
                               mfem::Vector &mesh_coords_bar) override;

private:
   /// state vector for evaluating force
   mfem::GridFunction &state;
   /// reference to primal integrator
   ForceIntegrator3 &force_integ;

#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix PointMat_bar;
   mfem::Vector elfun;
#endif
};

inline void addDomainSensitivityIntegrator(
    ForceIntegrator3 &primal_integ,
    std::map<std::string, FiniteElementState> &fields,
    std::map<std::string, mfem::ParLinearForm> &output_sens,
    std::map<std::string, mfem::ParNonlinearForm> &output_scalar_sens,
    mfem::Array<int> *attr_marker)
{
   auto &mesh_fes = fields.at("mesh_coords").space();
   output_sens.emplace("mesh_coords", &mesh_fes);

   if (attr_marker == nullptr)
   {
      output_sens.at("mesh_coords")
          .AddDomainIntegrator(new ForceIntegratorMeshSens3(
              fields.at("state").gridFunc(), primal_integ));
   }
   else
   {
      output_sens.at("mesh_coords")
          .AddDomainIntegrator(new ForceIntegratorMeshSens3(
                                   fields.at("state").gridFunc(), primal_integ),
                               *attr_marker);
   }
}

/// Functional integrator to compute forces/torques based on the virtual work
/// method
class ForceIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   /// \brief - Compute forces/torques based on the virtual work method
   /// \param[in] nu - model describing reluctivity
   /// \param[in] v - the grid function containing virtual displacements for
   ///                each mesh node
   ForceIntegrator(StateCoefficient &nu, mfem::GridFunction &v) : nu(nu), v(v)
   { }

   /// \brief - Compute forces/torques based on the virtual work method
   /// \param[in] nu - model describing reluctivity
   /// \param[in] v - the grid function containing virtual displacements for
   ///                each mesh node
   /// \param[in] attrs - the regions the force is acting on
   ForceIntegrator(StateCoefficient &nu,
                   mfem::GridFunction &v,
                   std::unordered_set<int> attrs)
    : nu(nu), v(v), attrs(std::move(attrs))
   { }

   /// \brief - Compute element contribution to global force/torque
   /// \param[in] el - the finite element
   /// \param[in] trans - defines the reference to physical element mapping
   /// \param[in] elfun - state vector of the element
   /// \returns the element contribution to global force/torque
   double GetElementEnergy(const mfem::FiniteElement &el,
                           mfem::ElementTransformation &trans,
                           const mfem::Vector &elfun) override;

   /// \brief - Computes dJdu, for solving for the adjoint
   /// \param[in] el - the finite element
   /// \param[in] trans - defines the reference to physical element mapping
   /// \param[in] elfun - state vector of the element
   /// \param[out] elfun_bar - \partial J \partial u for this functional
   void AssembleElementVector(const mfem::FiniteElement &el,
                              mfem::ElementTransformation &trans,
                              const mfem::Vector &elfun,
                              mfem::Vector &elfun_bar) override;

private:
   /// material dependent model describing reluctivity
   StateCoefficient &nu;
   /// grid function containing virtual displacements for each mesh node
   mfem::GridFunction &v;
   /// set of attributes the force is acting on
   std::unordered_set<int> attrs;

#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix dshape, curlshape, curlshape_dFt, dBdX;
   mfem::Vector b_vec, b_hat;
   mfem::Array<int> vdofs;
   mfem::Vector vfun;
#endif
   /// class that implements mesh sensitivities for ForceIntegrator
   friend class ForceIntegratorMeshSens;
};

/// Linear form integrator to assemble the vector dJdX for the ForceIntegrator
class ForceIntegratorMeshSens : public mfem::LinearFormIntegrator
{
public:
   /// \brief - Compute forces/torques based on the virtual work method
   /// \param[in] state - the state vector to evaluate force at
   /// \param[in] integ - reference to primal integrator that holds inputs for
   /// integrators
   ForceIntegratorMeshSens(mfem::GridFunction &state, ForceIntegrator &integ)
    : state(state), force_integ(integ)
   { }

   /// \brief - assemble an element's contribution to dJdX
   /// \param[in] el - the finite element that describes the mesh element
   /// \param[in] trans - the transformation between reference and physical
   /// space
   /// \param[out] mesh_coords_bar - dJdX for the element
   void AssembleRHSElementVect(const mfem::FiniteElement &el,
                               mfem::ElementTransformation &trans,
                               mfem::Vector &mesh_coords_bar) override;

private:
   /// state vector for evaluating force
   mfem::GridFunction &state;
   /// reference to primal integrator
   ForceIntegrator &force_integ;

#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix PointMat_bar;
   mfem::Vector elfun;
#endif
};

inline void addSensitivityIntegrator(
    ForceIntegrator &primal_integ,
    std::map<std::string, FiniteElementState> &fields,
    std::map<std::string, mfem::ParLinearForm> &output_sens,
    std::map<std::string, mfem::ParNonlinearForm> &output_scalar_sens)
{
   auto &mesh_fes = fields.at("mesh_coords").space();
   output_sens.emplace("mesh_coords", &mesh_fes);
   output_sens.at("mesh_coords")
       .AddDomainIntegrator(new ForceIntegratorMeshSens(
           fields.at("state").gridFunc(), primal_integ));
}

/// Functional integrator to compute core losses based on the Steinmetz
/// equations
class SteinmetzLossIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   friend void setInputs(SteinmetzLossIntegrator &integ,
                         const MachInputs &inputs);

   /// \brief - Compute element contribution to global force/torque
   /// \param[in] el - the finite element
   /// \param[in] trans - defines the reference to physical element mapping
   /// \param[in] elfun - state vector of the element
   /// \returns the element contribution to global force/torque
   double GetElementEnergy(const mfem::FiniteElement &el,
                           mfem::ElementTransformation &trans,
                           const mfem::Vector &elfun) override;

   // /// \brief - Computes dJdu, for solving for the adjoint
   // /// \param[in] el - the finite element
   // /// \param[in] trans - defines the reference to physical element mapping
   // /// \param[in] elfun - state vector of the element
   // /// \param[out] elfun_bar - \partial J \partial u for this functional
   // void AssembleElementVector(const mfem::FiniteElement &el,
   //                            mfem::ElementTransformation &trans,
   //                            const mfem::Vector &elfun,
   //                            mfem::Vector &elfun_bar) override;

   SteinmetzLossIntegrator(mfem::Coefficient &rho,
                           mfem::Coefficient &k_s,
                           mfem::Coefficient &alpha,
                           mfem::Coefficient &beta,
                           std::string name = "")
    : rho(rho), k_s(k_s), alpha(alpha), beta(beta), name(std::move(name))
   { }

private:
   /// Density
   mfem::Coefficient &rho;
   /// Steinmetz coefficients
   mfem::Coefficient &k_s;
   mfem::Coefficient &alpha;
   mfem::Coefficient &beta;

   // optional integrator name to differentiate setting inputs
   std::string name;

   /// Electrical excitation frequency
   double freq = 1.0;
   /// Maximum flux density magnitude
   double max_flux_mag = 1.0;

   /// class that implements frequency sensitivities for SteinmetzLossIntegrator
   friend class SteinmetzLossIntegratorFreqSens;

   /// class that implements max flux sensitivities for SteinmetzLossIntegrator
   friend class SteinmetzLossIntegratorMaxFluxSens;

   /// class that implements mesh sensitivities for SteinmetzLossIntegrator
   friend class SteinmetzLossIntegratorMeshSens;
};

class SteinmetzLossIntegratorFreqSens : public mfem::NonlinearFormIntegrator
{
public:
   SteinmetzLossIntegratorFreqSens(SteinmetzLossIntegrator &integ)
    : integ(integ)
   { }

   /// \brief - Compute element contribution to global sensitivity
   /// \param[in] el - the finite element
   /// \param[in] trans - defines the reference to physical element mapping
   /// \param[in] elfun - state vector of the element
   /// \returns the element contribution to global sensitvity
   double GetElementEnergy(const mfem::FiniteElement &el,
                           mfem::ElementTransformation &trans,
                           const mfem::Vector &elfun) override;

private:
   /// reference to primal integrator
   SteinmetzLossIntegrator &integ;
};

class SteinmetzLossIntegratorMaxFluxSens : public mfem::NonlinearFormIntegrator
{
public:
   SteinmetzLossIntegratorMaxFluxSens(SteinmetzLossIntegrator &integ)
    : integ(integ)
   { }

   /// \brief - Compute element contribution to global sensitivity
   /// \param[in] el - the finite element
   /// \param[in] trans - defines the reference to physical element mapping
   /// \param[in] elfun - state vector of the element
   /// \returns the element contribution to global sensitvity
   double GetElementEnergy(const mfem::FiniteElement &el,
                           mfem::ElementTransformation &trans,
                           const mfem::Vector &elfun) override;

private:
   /// reference to primal integrator
   SteinmetzLossIntegrator &integ;
};

class SteinmetzLossIntegratorMeshSens : public mfem::LinearFormIntegrator
{
public:
   SteinmetzLossIntegratorMeshSens(mfem::GridFunction &state,
                                   SteinmetzLossIntegrator &integ)
    : state(state), integ(integ)
   { }

   /// \brief - assemble an element's contribution to dJdX
   /// \param[in] mesh_el - the finite element that describes the mesh element
   /// \param[in] mesh_trans - the transformation between reference and physical
   /// space \param[out] mesh_coords_bar - dJdX for the element
   void AssembleRHSElementVect(const mfem::FiniteElement &mesh_el,
                               mfem::ElementTransformation &mesh_trans,
                               mfem::Vector &mesh_coords_bar) override;

private:
   /// state vector for evaluating loss
   mfem::GridFunction &state;
   /// reference to primal integrator
   SteinmetzLossIntegrator &integ;

#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix PointMat_bar;
#endif
};

inline void addDomainSensitivityIntegrator(
    SteinmetzLossIntegrator &primal_integ,
    std::map<std::string, FiniteElementState> &fields,
    std::map<std::string, mfem::ParLinearForm> &output_sens,
    std::map<std::string, mfem::ParNonlinearForm> &output_scalar_sens,
    mfem::Array<int> *attr_marker)
{
   auto &state_fes = fields.at("state").space();
   output_scalar_sens.emplace("frequency", &state_fes);
   output_scalar_sens.emplace("max_flux_magnitude", &state_fes);

   auto &mesh_fes = fields.at("mesh_coords").space();
   output_sens.emplace("mesh_coords", &mesh_fes);

   if (attr_marker == nullptr)
   {
      output_scalar_sens.at("frequency")
          .AddDomainIntegrator(
              new SteinmetzLossIntegratorFreqSens(primal_integ));

      output_scalar_sens.at("max_flux_magnitude")
          .AddDomainIntegrator(
              new SteinmetzLossIntegratorMaxFluxSens(primal_integ));

      output_sens.at("mesh_coords")
          .AddDomainIntegrator(new SteinmetzLossIntegratorMeshSens(
              fields.at("state").gridFunc(), primal_integ));
   }
   else
   {
      output_scalar_sens.at("frequency")
          .AddDomainIntegrator(
              new SteinmetzLossIntegratorFreqSens(primal_integ), *attr_marker);

      output_scalar_sens.at("max_flux_magnitude")
          .AddDomainIntegrator(
              new SteinmetzLossIntegratorMaxFluxSens(primal_integ),
              *attr_marker);

      output_sens.at("mesh_coords")
          .AddDomainIntegrator(new SteinmetzLossIntegratorMeshSens(
                                   fields.at("state").gridFunc(), primal_integ),
                               *attr_marker);
   }
}

class SteinmetzLossDistributionIntegrator : public mfem::LinearFormIntegrator
{
public:
   friend void setInputs(SteinmetzLossDistributionIntegrator &integ,
                         const MachInputs &inputs);

   /// \param[in] el - the finite element
   /// \param[in] trans - defines the reference to physical element mapping
   /// \param[out] elvect - element local heat source distribution
   void AssembleRHSElementVect(const mfem::FiniteElement &el,
                               mfem::ElementTransformation &trans,
                               mfem::Vector &elvect) override;

   SteinmetzLossDistributionIntegrator(mfem::Coefficient &rho,
                                       mfem::Coefficient &k_s,
                                       mfem::Coefficient &alpha,
                                       mfem::Coefficient &beta,
                                       std::string name = "")
    : rho(rho), k_s(k_s), alpha(alpha), beta(beta), name(std::move(name))
   { }

private:
   /// Density
   mfem::Coefficient &rho;
   /// Steinmetz coefficients
   mfem::Coefficient &k_s;
   mfem::Coefficient &alpha;
   mfem::Coefficient &beta;
   // optional integrator name to differentiate setting inputs
   std::string name;

   /// Electrical excitation frequency
   double freq = 1.0;
   /// Maximum flux density magnitude
   double max_flux_mag = 1.0;
   /// Stack length
   double stack_length = 1.0;
#ifndef MFEM_THREAD_SAFE
   mfem::Vector shape;
#endif
   /// class that implements mesh sensitivities for
   /// SteinmetzLossDistributionIntegrator
   ///TODO: If necessary, define and implement this class
   friend class SteinmetzLossDistributionIntegratorMeshSens;
};

/// Functional integrator to compute core losses based on the CAL2 core loss model,
/// a two term loss separation model consisting of a term for hysteresis losses and a term for eddy current losses 
/// that assumes the hysteresis exponent for B to be constant and equal to 2, like for eddy currents
class CAL2CoreLossIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   friend void setInputs(CAL2CoreLossIntegrator &integ,
                         const MachInputs &inputs);

   /// \brief - Compute element contribution to global force/torque
   /// \param[in] el - the finite element
   /// \param[in] trans - defines the reference to physical element mapping
   /// \param[in] elfun - state vector of the element
   /// \returns the element contribution to global force/torque
   double GetElementEnergy(const mfem::FiniteElement &el,
                           mfem::ElementTransformation &trans,
                           const mfem::Vector &elfun) override;

   CAL2CoreLossIntegrator(mfem::Coefficient &rho,
                           ThreeStateCoefficient &CAL2_kh,
                           ThreeStateCoefficient &CAL2_ke,
                           mfem::GridFunction &peak_flux,
                           mfem::GridFunction *temperature_field=nullptr,
                           std::string name = "")
    : rho(rho), CAL2_kh(CAL2_kh), CAL2_ke(CAL2_ke), peak_flux(peak_flux), 
    temperature_field(temperature_field), name(std::move(name))
   { }

private:
   // Density
   mfem::Coefficient &rho;
   /// CAL2 Coefficients
   ThreeStateCoefficient &CAL2_kh;
   ThreeStateCoefficient &CAL2_ke;

   mfem::GridFunction &peak_flux; // peak flux field

   mfem::GridFunction *temperature_field; // pointer to temperature field (can be null)

   // optional integrator name to differentiate setting inputs
   std::string name;

   /// Electrical excitation frequency
   double freq = 1.0;
   /// Maximum alternating flux density magnitude (assuming sinusoidal excitation)
   double max_flux_mag = 1.0;

   ///TODO: Move this temporary logic to higher level or remove entirely once determine whether max flux value or peak flux field should be used
   bool UseMaxFluxValueAndNotPeakFluxField = false;

   ///TODO: Look into making code thread-safe
#ifndef MFEM_THREAD_SAFE
   mfem::Vector shape;
   mfem::Array<int> vdofs;
   mfem::Vector temp_elfun;
   mfem::Vector flux_shape;
   mfem::Vector flux_elfun;
#endif

   /// class that implements frequency sensitivities for CAL2CoreLossIntegrator
   friend class CAL2CoreLossIntegratorFreqSens;

   /// class that implements max flux sensitivities for CAL2CoreLossIntegrator
   friend class CAL2CoreLossIntegratorMaxFluxSens;

   /// class that implements temperature sensitivities for CAL2CoreLossIntegrator
   friend class CAL2CoreLossIntegratorTemperatureSens;

   /// class that implements mesh sensitivities for CAL2CoreLossIntegrator
   friend class CAL2CoreLossIntegratorMeshSens;

   ///TODO: Determine if need to add CAL2CoreLossTemperatureSens
};

class CAL2CoreLossIntegratorFreqSens : public mfem::NonlinearFormIntegrator
{
public:
   CAL2CoreLossIntegratorFreqSens(CAL2CoreLossIntegrator &integ)
    : integ(integ)
   { }

   /// \brief - Compute element contribution to global sensitivity
   /// \param[in] el - the finite element
   /// \param[in] trans - defines the reference to physical element mapping
   /// \param[in] elfun - state vector of the element
   /// \returns the element contribution to global s6ensitvity
   double GetElementEnergy(const mfem::FiniteElement &el,
                           mfem::ElementTransformation &trans,
                           const mfem::Vector &elfun) override;

private:
   /// reference to primal integrator
   CAL2CoreLossIntegrator &integ;

   ///TODO: Move this temporary logic to higher level or remove entirely once determine whether max flux value or peak flux field should be used
   bool UseMaxFluxValueAndNotPeakFluxField = false;

   ///TODO: Look into making code thread-safe
#ifndef MFEM_THREAD_SAFE
   mfem::Vector shape;
   mfem::Array<int> vdofs;
   mfem::Vector temp_elfun;
   mfem::Vector flux_shape;
   mfem::Vector flux_elfun;
#endif

};

class CAL2CoreLossIntegratorMaxFluxSens : public mfem::NonlinearFormIntegrator
{
public:
   CAL2CoreLossIntegratorMaxFluxSens(CAL2CoreLossIntegrator &integ)
    : integ(integ)
   { }

   /// \brief - Compute element contribution to global sensitivity
   /// \param[in] el - the finite element
   /// \param[in] trans - defines the reference to physical element mapping
   /// \param[in] elfun - state vector of the element
   /// \returns the element contribution to global s6ensitvity
   double GetElementEnergy(const mfem::FiniteElement &el,
                           mfem::ElementTransformation &trans,
                           const mfem::Vector &elfun) override;

private:
   /// reference to primal integrator
   CAL2CoreLossIntegrator &integ;

   ///TODO: Move this temporary logic to higher level or remove entirely once determine whether max flux value or peak flux field should be used
   bool UseMaxFluxValueAndNotPeakFluxField = false;

   ///TODO: Look into making code thread-safe
#ifndef MFEM_THREAD_SAFE
   mfem::Vector shape;
   mfem::Array<int> vdofs;
   mfem::Vector temp_elfun;
   mfem::Vector flux_shape;
   mfem::Vector flux_elfun;
#endif

};

class CAL2CoreLossIntegratorTemperatureSens : public mfem::NonlinearFormIntegrator
{
public:
   CAL2CoreLossIntegratorTemperatureSens(CAL2CoreLossIntegrator &integ)
    : integ(integ)
   { }

   /// \brief - Compute element contribution to global sensitivity
   /// \param[in] el - the finite element
   /// \param[in] trans - defines the reference to physical element mapping
   /// \param[in] elfun - state vector of the element
   /// \returns the element contribution to global s6ensitvity
   double GetElementEnergy(const mfem::FiniteElement &el,
                           mfem::ElementTransformation &trans,
                           const mfem::Vector &elfun) override;

private:
   /// reference to primal integrator
   CAL2CoreLossIntegrator &integ;

   ///TODO: Move this temporary logic to higher level or remove entirely once determine whether max flux value or peak flux field should be used
   bool UseMaxFluxValueAndNotPeakFluxField = false;

   ///TODO: Look into making code thread-safe
#ifndef MFEM_THREAD_SAFE
   mfem::Vector shape;
   mfem::Array<int> vdofs;
   mfem::Vector temp_elfun;
   mfem::Vector flux_shape;
   mfem::Vector flux_elfun;
#endif

};

class CAL2CoreLossIntegratorMeshSens : public mfem::LinearFormIntegrator
{
public:
   CAL2CoreLossIntegratorMeshSens(mfem::GridFunction &state,
                                   CAL2CoreLossIntegrator &integ)
    : state(state), integ(integ)
   { }

   /// \brief - assemble an element's contribution to dJdX
   /// \param[in] mesh_el - the finite element that describes the mesh element
   /// \param[in] mesh_trans - the transformation between reference and physical
   /// space \param[out] mesh_coords_bar - dJdX for the element
   void AssembleRHSElementVect(const mfem::FiniteElement &mesh_el,
                               mfem::ElementTransformation &mesh_trans,
                               mfem::Vector &mesh_coords_bar) override;

private:
   /// state vector for evaluating loss
   mfem::GridFunction &state;
   /// reference to primal integrator
   CAL2CoreLossIntegrator &integ;

   ///TODO: Move this temporary logic to higher level or remove entirely once determine whether max flux value or peak flux field should be used
   bool UseMaxFluxValueAndNotPeakFluxField = false;

   ///TODO: Look into making code thread-safe
#ifndef MFEM_THREAD_SAFE
   mfem::Vector shape;
   mfem::Array<int> vdofs;
   mfem::Vector temp_elfun;
   mfem::Vector flux_shape;
   mfem::Vector flux_elfun;
   mfem::DenseMatrix PointMat_bar;
#endif
};

/// Functional integrator to compute the spacial distribution of the heat flux for core losses based on the CAL2 core loss model,
/// a two term loss separation model consisting of a term for hysteresis losses and a term for eddy current losses 
/// that assumes the hysteresis exponent for B to be constant and equal to 2, like for eddy currents
class CAL2CoreLossDistributionIntegrator : public mfem::LinearFormIntegrator
{
public:
   friend void setInputs(CAL2CoreLossDistributionIntegrator &integ,
                         const MachInputs &inputs);

   /// \brief - Compute element contribution to global force/torque
   /// \param[in] el - the finite element
   /// \param[in] trans - defines the reference to physical element mapping
   /// \param[out] elvect - element local heat source distribution
   void AssembleRHSElementVect(const mfem::FiniteElement &el,
                           mfem::ElementTransformation &trans,
                           mfem::Vector &elvect) override;

   CAL2CoreLossDistributionIntegrator(mfem::Coefficient &rho,
                           ThreeStateCoefficient &CAL2_kh,
                           ThreeStateCoefficient &CAL2_ke,
                           mfem::GridFunction &peak_flux,
                           mfem::GridFunction *temperature_field=nullptr,
                           std::string name = "")
    : rho(rho), CAL2_kh(CAL2_kh), CAL2_ke(CAL2_ke), peak_flux(peak_flux), 
    temperature_field(temperature_field), name(std::move(name))
   { }

private:
   // Density
   mfem::Coefficient &rho;
   /// CAL2 Coefficients
   ThreeStateCoefficient &CAL2_kh;
   ThreeStateCoefficient &CAL2_ke;

   mfem::GridFunction &peak_flux; // peak flux field

   mfem::GridFunction *temperature_field; // pointer to temperature field (can be null)

   // optional integrator name to differentiate setting inputs
   std::string name;

   /// Electrical excitation frequency
   double freq = 1.0;
   /// Maximum alternating flux density magnitude (assuming sinusoidal excitation)
   double max_flux_mag = 1.0;
   /// Stack length
   double stack_length = 1.0;

   ///TODO: Move this temporary logic to higher level or remove entirely once determine whether max flux value or peak flux field should be used
   bool UseMaxFluxValueAndNotPeakFluxField = false;

   ///TODO: Make code thread safe
#ifndef MFEM_THREAD_SAFE
   mfem::Vector shape;
   mfem::Array<int> vdofs;
   mfem::Vector temp_elfun;
   mfem::Vector flux_shape;
   mfem::Vector flux_elfun;
#endif

   /// class that implements mesh sensitivities for CAL2CoreLossDistributionIntegrator
   ///TODO: Will this needed to be added? The equivalent for Steinmetz never was made.
   friend class CAL2CoreLossDistributionIntegratorMeshSens;
};

/// Functional integrator to determine values or distribution 
/// of Permanent Magnet Demagnetization constraint equation
class PMDemagIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   /// \brief - determine values or distribution 
   ///         of Permanent Magnet Demagnetization constraint equation
   /// \param[in] PMDemagConstraint - the permanent magnet demagnetization constraint equation coefficient
   /// \param[in] temperature_field - a pointer to the temperature field (default is null)
   /// \param[in] name - component name to differentiate setting inputs (default is blank)
   PMDemagIntegrator(TwoStateCoefficient &PMDemagConstraint,
                     mfem::GridFunction *temperature_field=nullptr,
                     std::string name = "") 
   : PMDemagConstraint(PMDemagConstraint), temperature_field(temperature_field), name(std::move(name)) { }

   /// \brief - determine value 
   ///         of Permanent Magnet Demagnetization constraint equation
   /// \param[in] el - the finite element
   /// \param[in] trans - defines the reference to physical element mapping
   /// \param[in] elfun - state vector of the element
   /// \returns the value of the Permanent Magnet Demagnetization constraint equation calculated over an element
   double GetElementEnergy(const mfem::FiniteElement &el,
                           mfem::ElementTransformation &trans,
                           const mfem::Vector &elfun) override;

   /// \brief - determine distribution 
   ///         of Permanent Magnet Demagnetization constraint equation
   /// \param[in] el - the finite element
   /// \param[in] trans - defines the reference to physical element mapping
   /// \param[in] elfun - state vector of the element
   /// \param[out] elvect - element Permanent Magnet Demagnetization constraint equation distribution
   void AssembleElementVector(const mfem::FiniteElement &el, 
                              mfem::ElementTransformation &trans, 
                              const mfem::Vector &elfun, 
                              mfem::Vector &elvect) override;

private:
   TwoStateCoefficient &PMDemagConstraint; // the permanent magnet demagnetization constraint equation coefficient
   mfem::GridFunction *temperature_field; // pointer to temperature field (can be null)

   // optional integrator name to differentiate setting inputs
   std::string name;

#ifndef MFEM_THREAD_SAFE
   mfem::Vector shape;
   mfem::Vector temp_shape;
   mfem::Array<int> vdofs;
   mfem::Vector temp_elfun;
#endif

   ///TODO: Add in sens classes, and denote them as friends here

};

}  // namespace mach

#endif
