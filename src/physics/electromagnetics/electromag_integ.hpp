#ifndef MACH_ELECTROMAG_INTEG
#define MACH_ELECTROMAG_INTEG

#include <unordered_set>

#include "mfem.hpp"

#include "mach_types.hpp"

namespace mach
{

class AbstractSolver;
class StateCoefficient;

/// Integrator for (\nu(u)*curl u, curl v) for Nedelec elements
class CurlCurlNLFIntegrator : public mfem::NonlinearFormIntegrator,
                              public mfem::LinearFormIntegrator
{
public:
   /// Construct a curl curl nonlinear form integrator for Nedelec elements
   /// \param[in] m - model describing nonlinear material parameter
   /// \param[in] a - used to move to lhs or rhs
   CurlCurlNLFIntegrator(StateCoefficient *m,
                        double a = 1.0)
      : model(m), alpha(a) {}

   /// Construct a curl curl nonlinear form integrator for Nedelec elements
   /// \param[in] m - model describing nonlinear material parameter
   /// \param[in] state - the state to use when evaluating
   ///                    \frac{\partial psi^T R}{\partial X}
   /// \param[in] adjoint - the adjoint to use when evaluating
   ///                      \frac{\partial psi^T R}{\partial X}
   /// \param[in] a - used to move to lhs or rhs
   CurlCurlNLFIntegrator(StateCoefficient *m, mfem::GridFunction *_state,
                         mfem::GridFunction *_adjoint, double a = 1.0)
      : model(m), state(_state), adjoint(_adjoint), alpha(a) {}

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

   /// \brief calculate the functional \psi^T R(u)
   /// \param[in] el - the finite element
   /// \param[in] trans - defines the reference to physical element mapping
   /// \param[in] elfun - state vector of the element
   double GetElementEnergy(const mfem::FiniteElement &el,
                           mfem::ElementTransformation &trans,
                           const mfem::Vector &elfun) override;

   /// \brief - assemble an element's contribution to
   ///          \frac{\partial psi^T R}{\partial X}, needed for finding the total
   ///          derivative of a functional with respect to the mesh nodes
   /// \param[in] el - the finite element that describes the mesh element
   /// \param[in] trans - the transformation between reference and physical space
   /// \param[out] elvect - \frac{\partial J}{\partial X} for the element
   /// \note this is the `LinearFormIntegrator` component, the LinearForm that
   ///       assembles this integrator's FiniteElementSpace MUST be the mesh's
   ///       nodal finite element space
   void AssembleRHSElementVect(const mfem::FiniteElement &el,
                               mfem::ElementTransformation &trans,
                               mfem::Vector &elvect) override;

private:
	/// material (thus mesh) dependent model describing electromagnetic behavior
	StateCoefficient *model;
   /// the state to use when evaluating \frac{\partial psi^T R}{\partial X}
   mfem::GridFunction *state;
   /// the adjoint to use when evaluating \frac{\partial psi^T R}{\partial X}
   mfem::GridFunction *adjoint;
   /// scales the terms; can be used to move to rhs/lhs
	double alpha;

#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix curlshape, curlshape_dFt;
   mfem::Vector b_vec, b_hat, curl_psi, curl_psi_hat, temp_vec;
#endif

};

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
		: nu(nu), mag(M), alpha(a) {}

   /// Construct the element local residual
   /// \param[in] el - the finite element whose residual we want
   /// \param[in] trans - defines the reference to physical element mapping
   /// \param[in] elfun - element local state vector
   /// \param[out] elvect - element local residual
   virtual void AssembleElementVector(const mfem::FiniteElement &el,
                                      mfem::ElementTransformation &trans,
                                      const mfem::Vector &elfun,
                                      mfem::Vector &elvect);

   /// Construct the element local Jacobian
   /// \param[in] el - the finite element whose Jacobian we want
   /// \param[in] trans - defines the reference to physical element mapping
   /// \param[in] elfun - element local state vector
   /// \param[out] elmat - element local Jacobian
   virtual void AssembleElementGrad(const mfem::FiniteElement &el,
                                    mfem::ElementTransformation &trans,
                                    const mfem::Vector &elfun,
                                    mfem::DenseMatrix &elmat);

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

class VectorFECurldJdXIntegerator : public mfem::LinearFormIntegrator
{
   /// Construct a curl curl nonlinear form integrator for Nedelec elements
   /// \param[in] nu - coefficient describing reluctivity
   /// \param[in] state - the state to use when evaluating
   ///                    \frac{\partial psi^T R}{\partial X}
   /// \param[in] adjoint - the adjoint to use when evaluating
   ///                      \frac{\partial psi^T R}{\partial X}
   /// \note it is assumed that `state` is an a H(div) finite element space and
   ///       that adjoint is in a H(curl) finite element space
   VectorFECurldJdXIntegerator(mfem::Coefficient *_nu,
                               mfem::GridFunction *_state,
                               mfem::GridFunction *_adjoint)
      : nu(_nu), state(_state), adjoint(_adjoint) {};

   /// \brief - assemble an element's contribution to
   ///          \frac{\partial psi^T R}{\partial X}, needed for finding the total
   ///          derivative of a functional with respect to the mesh nodes
   /// \param[in] el - the finite element that describes the mesh element
   /// \param[in] trans - the transformation between reference and physical space
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
   mfem::GridFunction *state;
   /// the adjoint to use when evaluating \frac{\partial psi^T R}{\partial X}
   mfem::GridFunction *adjoint;

#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix curlshape, curlshape_dFt;
   mfem::DenseMatrix vshape, vshape_dFt;
   mfem::Vector m_vec, m_hat, curl_psi, curl_psi_hat, temp_vec;
#endif

};

/// Integrator to compute the magnetic energy
class MagneticEnergyIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   /// \param[in] nu - model describing reluctivity
   MagneticEnergyIntegrator(StateCoefficient *_nu) : nu(_nu) {};

   /// \param[in] el - the finite element
   /// \param[in] trans - defines the reference to physical element mapping
   /// \param[in] elfun - state vector of the element
   double GetElementEnergy(const mfem::FiniteElement &el,
                           mfem::ElementTransformation &trans,
                           const mfem::Vector &elfun) override;

private:
   /// material (thus mesh) dependent model describing reluctivity
   StateCoefficient *nu;
#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix curlshape, curlshape_dFt, M;
   mfem::Vector b_vec;
#endif
};

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
   /// \param[in] trans - the transformation between reference and physical space
   /// \param[out] elvect - \frac{\partial J}{\partial X} for the element
   /// \note this is the `LinearFormIntegrator` component, the LinearForm that
   ///       assembles this integrator's FiniteElementSpace MUST be the mesh's
   ///       nodal finite element space
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

/// Integrator to compute the magnetic co-energy
class BNormIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   /// \param[in] nu - model describing reluctivity
   BNormIntegrator() {};

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
#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix curlshape, curlshape_dFt, M;
   mfem::Vector b_vec, temp_vec;
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
   BNormdJdx(mfem::GridFunction &_state)
      : LinearFormIntegrator(), state(_state) {}


   /// \brief - assemble an element's contribution to \frac{\partial J}{\partial X}
   /// \param[in] el - the finite element that describes the mesh element
   /// \param[in] trans - the transformation between reference and physical space
   /// \param[out] elvect - \frac{\partial J}{\partial X} for the element
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
   nuBNormIntegrator(StateCoefficient *_nu)
      : nu(_nu) {};

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
      : LinearFormIntegrator(), state(_state), nu(_nu) {}


   /// \brief - assemble an element's contribution to \frac{\partial J}{\partial X}
   /// \param[in] el - the finite element that describes the mesh element
   /// \param[in] trans - the transformation between reference and physical space
   /// \param[out] elvect - \frac{\partial J}{\partial X} for the element
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
   nuFuncIntegrator(StateCoefficient *_nu)
      : state(NULL), nu(_nu) {};

   /// \brief - linear form integrator to assemble the vector
   ///          \frac{\partial J}{\partial X}
   /// \param[in] state - the current state (A)
   /// \note the finite element space used to by the linear form that assembles
   ///       this integrator will use the mesh's nodal finite element space
   nuFuncIntegrator(mfem::GridFunction *_state, StateCoefficient *_nu)
      : state(_state), nu(_nu) {};

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

   /// \brief - assemble an element's contribution to \frac{\partial J}{\partial X}
   /// \param[in] el - the finite element that describes the mesh element
   /// \param[in] trans - the transformation between reference and physical space
   /// \param[out] elvect - \frac{\partial J}{\partial X} for the element
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

// /// Integrator for forces due to electromagnetic fields
// /// \note - Requires PUMI
// class ForceIntegrator : public mfem::NonlinearFormIntegrator
// {
// public:
//    /// \param[in] solver - pointer to solver, used to get PUMI mesh
//    /// \param[in] regions - list of regions to find the resultant force on
//    /// \param[in] free_regions - list of regions of free space that surround
//    ///                           `regions`
//    /// \param[in] nu - model describing reluctivity
//    /// \param[in] dir - direction to find the force in
//    ForceIntegrator(AbstractSolver *solver,
//                    std::unordered_set<int> regions,
//                    std::unordered_set<int> free_regions,
//                    StateCoefficient *nu,
//                    mfem::Vector dir);

//    /// \param[in] el - the finite element
//    /// \param[in] Tr - defines the reference to physical element mapping
//    /// \param[in] elfun - state vector of the element
//    /// \note this function will call PUMI APIs to figure out which nodes are
//    ///       in free space/on the rotor (fixed/free)
//    double GetElementEnergy(const mfem::FiniteElement &el,
//                            mfem::ElementTransformation &trans,
//                            const mfem::Vector &elfun) override;

// private:
//    /// pointer to abstract solver (used to get PUMI mesh)
//    AbstractSolver * const solver;
//    /// list of regions to find the resultant force on
//    const std::unordered_set<int> regions, free_regions;
//    /// material (thus mesh) dependent model describing reluctivity
//    StateCoefficient * const nu;
//    /// direction to calculate the force
//    const mfem::Vector dir;
//    /// model faces that define the interface between moving and fixed regions
//    std::unordered_set<int> face_list;
//    /// set of element indices to be used to integrate over
//    std::unordered_set<int> el_ids;

// #ifndef MFEM_THREAD_SAFE
//    mfem::DenseMatrix curlshape, curlshape_dFt, M;
//    mfem::Vector b_vec;
// #endif

// };

// /// Integrator for torques due to electromagnetic fields
// class VWTorqueIntegrator : public mfem::NonlinearFormIntegrator
// {
// public:
//    /// \param[in] el - the finite element
//    /// \param[in] Tr - defines the reference to physical element mapping
//    /// \param[in] elfun - state vector of the element
//    /// \note this function will call PUMI API's to figure out which nodes are
//    ///       in free space/on the rotor (fixed/free)
//    VWTorqueIntegrator(StateCoefficient *m,
//                      double a = 1.0)
//    : model(m), alpha(a) {}

//    /// \param[in] el - the finite element
//    /// \param[in] Tr - defines the reference to physical element mapping
//    /// \param[in] elfun - state vector of the element
//    /// \note this function will call PUMI API's to figure out which nodes are
//    ///       in free space/on the rotor (fixed/free)
//    double GetElementEnergy(const mfem::FiniteElement &el,
//                            mfem::ElementTransformation &Tr,
//                            const mfem::Vector &elfun) override;

// private:
//    /// material (thus mesh) dependent model describing electromagnetic behavior
//    StateCoefficient *model;
//    /// scales the terms; can be used to move to rhs/lhs
//    double alpha;

// #ifndef MFEM_THREAD_SAFE
//    mfem::DenseMatrix curlshape, curlshape_dFt;
//    mfem::Vector b_vec, temp_vec;
// #endif

// };

} // namespace mach

#endif
