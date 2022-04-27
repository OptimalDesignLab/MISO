#ifndef MACH_ELECTROMAG_INTEG
#define MACH_ELECTROMAG_INTEG

#include <unordered_set>

#include "mfem.hpp"

#include "mach_types.hpp"
#include "mach_input.hpp"
#include "mach_integrator.hpp"

namespace mach
{
class AbstractSolver;
class StateCoefficient;

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
                               mfem::Vector &state_bar) override;

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
                              mfem::Vector &elvect) override;

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
   /// \param[in] el - the finite element that describes the mesh element
   /// \param[in] trans - the transformation between reference and physical
   /// space \param[out] mesh_coords_bar - dJdX for the element
   void AssembleRHSElementVect(const mfem::FiniteElement &el,
                               mfem::ElementTransformation &trans,
                               mfem::Vector &mesh_coords_bar) override;

private:
   /// state vector for evaluating force
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

inline void addSensitivityIntegrator(
    MagneticEnergyIntegrator &primal_integ,
    std::map<std::string, FiniteElementState> &fields,
    std::map<std::string, mfem::ParLinearForm> &output_sens,
    std::map<std::string, mfem::ParNonlinearForm> &output_scalar_sens)
{
   auto &mesh_fes = fields.at("mesh_coords").space();
   output_sens.emplace("mesh_coords", &mesh_fes);
   output_sens.at("mesh_coords")
       .AddDomainIntegrator(new MagneticEnergyIntegratorMeshSens(
           fields.at("state").gridFunc(), primal_integ));
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
   /// \brief allows changing the frequency and diameter of the strands for AC
   /// loss calculation
   friend void setInputs(DCLossFunctionalIntegrator &integ,
                         const MachInputs &inputs);

   /// \brief - Compute DC copper losses in the domain
   /// \param[in] sigma - the electrical conductivity coefficient
   /// \param[in] current - the current density vector coefficient
   /// \param[in] current_density - the current density magnitude
   DCLossFunctionalIntegrator(mfem::Coefficient &sigma,
                              mfem::VectorCoefficient &current,
                              double current_density)
    : sigma(sigma), current(current), current_density(current_density)
   { }

   /// \brief - Compute DC copper losses in the domain
   /// \param[in] el - the finite element
   /// \param[in] trans - defines the reference to physical element mapping
   /// \param[in] elfun - state vector of the element
   /// \returns the DC losses calculated over an element
   double GetElementEnergy(const mfem::FiniteElement &el,
                           mfem::ElementTransformation &trans,
                           const mfem::Vector &elfun) override;

private:
   mfem::Coefficient &sigma;
   mfem::VectorCoefficient &current;
   double current_density;
#ifndef MFEM_THREAD_SAFE
   mfem::Vector current_vec;
#endif
};

/// Functional integrator to compute AC copper losses based on hybrid approach
/// (new)
class ACLossFunctionalIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   friend void setOptions(ACLossFunctionalIntegrator &integ,
                          const nlohmann::json &options);

   friend void setInputs(ACLossFunctionalIntegrator &integ,
                         const MachInputs &inputs);

   /// \brief - Compute AC copper losses in the domain based on a hybrid
   ///          analytical-FEM approach
   /// \param[in] sigma - the electrical conductivity coefficient
   ACLossFunctionalIntegrator(mfem::Coefficient &sigma) : sigma(sigma) { }

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
   mfem::Coefficient &sigma;
   double freq = 1.0;
   double radius = 1.0;
   double stack_length = 1.0;
   double model_depth = 1.0;
   double num_strands = 1.0;
   double slot_area = 1.0;

#ifndef MFEM_THREAD_SAFE
   mfem::Vector shape;
#endif
};

/// Functional integrator to compute AC copper losses based on hybrid approach
class HybridACLossFunctionalIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   /// \brief allows changing the frequency and diameter of the strands for AC
   /// loss calculation
   friend void setInputs(HybridACLossFunctionalIntegrator &integ,
                         const MachInputs &inputs);

   /// \brief - Compute AC copper losses in the domain based on a hybrid
   ///          analytical-FEM approach
   /// \param[in] sigma - the electrical conductivity coefficient
   /// \param[in] freq - the electrical excitation frequency
   /// \param[in] diam - the diameter of a strand in the bundle
   /// \param[in] fill_factor - the density of strands in the bundle
   HybridACLossFunctionalIntegrator(mfem::Coefficient &sigma,
                                    const double freq,
                                    const double diam,
                                    const double fill_factor)
    : sigma(sigma), freq(freq), diam(diam), fill_factor(fill_factor)
   { }

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
   mfem::Coefficient &sigma;
   double freq;
   double diam;
   double fill_factor;

#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix curlshape, curlshape_dFt;
   mfem::Vector b_vec;
#endif
};

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
   /// space \param[out] mesh_coords_bar - dJdX for the element
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

// /** Class for constructing the (local) discrete curl matrix which can be used
//     as an integrator in a DiscreteLinearOperator object to assemble the
//     global discrete curl matrix. */
// class CurlInterpolator : public mfem::DiscreteInterpolator
// {
// public:
//    /// Construct the element local residual
//    /// \param[in] nd_el - the ND finite element used to take the curl
//    /// \param[in] rt_el - the RT finite element used to represent the curl
//    field
//    /// \param[in] trans - defines the reference to physical element mapping
//    /// \param[out] elmat - local discrete curl matrix
//    void AssembleElementMatrix2(const FiniteElement &nd_fe,
//                                const FiniteElement &rt_fe,
//                                ElementTransformation &trans,
//                                DenseMatrix &elmat) override;
// };

}  // namespace mach

#endif
