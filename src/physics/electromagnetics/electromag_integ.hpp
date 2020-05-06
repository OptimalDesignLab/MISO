#ifndef MACH_ELECTROMAG_INTEG
#define MACH_ELECTROMAG_INTEG

#include <unordered_set>

#include "mfem.hpp"

namespace mach
{

class AbstractSolver;
class StateCoefficient;

/// Integrator for (\nu(u)*curl u, curl v) for Nedelec elements
class CurlCurlNLFIntegrator : public mfem::NonlinearFormIntegrator
{
public:
	/// Construct a curl curl nonlinear form integrator for Nedelec elements
	/// \param[in] m - model describing nonlinear material parameter
   /// \param[in] a - used to move to lhs or rhs
   CurlCurlNLFIntegrator(StateCoefficient *m,
								 double a = 1.0)
		: model(m), alpha(a) {}

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
	/// material (thus mesh) dependent model describing electromagnetic behavior
	StateCoefficient *model;
   /// scales the terms; can be used to move to rhs/lhs
	double alpha;

#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix curlshape, curlshape_dFt;
   mfem::Vector b_vec, temp_vec;
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
class MagneticCoenergyIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   /// \param[in] nu - model describing reluctivity
   MagneticCoenergyIntegrator(StateCoefficient *_nu)
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
