#ifndef MACH_ELECTROMAG_INTEG
#define MACH_ELECTROMAG_INTEG

#include "mfem.hpp"

#include "coefficient.hpp"
#include "solver.hpp"

namespace mach
{

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

/// Integrator for to compute the magnetic energy
class MagneticEnergyIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   /// \param[in] nu - model describing reluctivity
   MagneticEnergyIntegrator(StateCoefficient *_nu) : nu(_nu) {};

   /// \param[in] el - the finite element
   /// \param[in] Tr - defines the reference to physical element mapping
   /// \param[in] elfun - state vector of the element
   double GetElementEnergy(const mfem::FiniteElement &el,
                           mfem::ElementTransformation &Tr,
                           const mfem::Vector &elfun) override;

private:
   /// material (thus mesh) dependent model describing reluctivity
   StateCoefficient *nu;
#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix curlshape, curlshape_dFt, M;
   mfem::Vector b_vec;
#endif
};

/// Integrator for to compute the magnetic co-energy
class MagneticCoenergyIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   /// \param[in] nu - model describing reluctivity
   MagneticCoenergyIntegrator(StateCoefficient *_nu) : nu(_nu) {};

   /// \param[in] el - the finite element
   /// \param[in] Tr - defines the reference to physical element mapping
   /// \param[in] elfun - state vector of the element
   double GetElementEnergy(const mfem::FiniteElement &el,
                           mfem::ElementTransformation &Tr,
                           const mfem::Vector &elfun) override;

private:
   /// material (thus mesh) dependent model describing reluctivity
   StateCoefficient *nu;
#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix curlshape, curlshape_dFt, M;
   mfem::Vector b_vec;
#endif
};

// /// Integrator for forces due to electromagnetic fields
// class ForceIntegrator : public mfem::NonlinearFormIntegrator
// {
// public:
//    /// \param[in] dir - direction to calculate the force (x, y, z) -> (0, 1, 2)
//    /// \param[in] regions - list of regions to find the resultant force on
//    /// \param[in] nu - model describing reluctivity
//    /// \param[in] M - model describing permanent magnetization sources
//    /// \param[in] J - model describing current sources
//    ForceIntegrator(const int dir, std::vector<int> _regions,
//                    StateCoefficient *_nu, mfem::Coefficient *M,
//                    mfem::Coefficient *J);

//    /// \param[in] el1 - the element on one side of the interface
//    /// \param[in] el2 - the element on the other side
//    /// \param[in] Tr - holds geometry and mapping information about the face
//    /// \param[in] elfun - state vector on the element
//    /// \note - this function will call PUMI APIs to determine which faces bound
//    ///         the regions of interest
//    double GetFaceEnergy(const mfem::FiniteElement &el1,
//                         const mfem::FiniteElement &el2,
//                         mfem::FaceElementTransformations &Tr,
//                         const mfem::Vector &elfun) override;

// private:
//    /// direction to calculate the force (x, y, z) -> (0, 1, 2)
//    const int dir;
//    /// list of regions to find the resultant force on
//    std::vector<int> regions;
//    /// material (thus mesh) dependent model describing reluctivity
//    StateCoefficient *nu;
//    /// material dependent model describing permanent magnetization
//    mfem::Coefficient *M;
//    /// material (thus mesh) dependent model describing current sources
//    mfem::Coefficient *J;
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