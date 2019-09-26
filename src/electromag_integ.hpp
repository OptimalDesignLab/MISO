#ifndef MACH_ELECTROMAG_INTEG
#define MACH_ELECTROMAG_INTEG

#include "mfem.hpp"
#include "solver.hpp"

namespace mach
{

/// Abstract class for electromagnetic material models (permeability)
class ElectromageticMaterialModel
{
public:
	/// Construct an electromagnetic material model
   ElectromageticMaterialModel() { }

   virtual ~ElectromageticMaterialModel() { }

   /// \brief Evaluate the material model 
   /// \param[in] trans - element transformation holds point in space
   ///                    as well as mesh attributes
   /// \param[in] stateVec - element state vector
   /// \param[out] outVec - evaluated material model at every degree of freedom
   /// \note It is assumed that trans.SetIntPoint() has already called for the
	/// point of interest.
   void Eval(mfem::ElementTransformation &trans, mfem::Vector stateVec,  
             mfem::Vector outVec);


   // /// A reference-element to target-element transformation that can be used to
   // /// evaluate Coefficient
   // /// @note It is assumed that _Ttr.SetIntPoint() is already called for the
	// /// point of interest.
   // void SetTransformation(mfem::ElementTransformation &_Ttr) { Ttr = &_Ttr; }

   // /// \brief Evaluate the strain energy density function, W = W(Jpt).
	// /// \param[in] Jpt  Represents the target->physical transformation
   // ///                 Jacobian matrix.
   // virtual double EvalW(const mfem::DenseMatrix &Jpt) const = 0;

   // /// \brief Evaluate the 1st Piola-Kirchhoff stress tensor, P = P(Jpt).
   // /// \param[in] Jpt  Represents the target->physical transformation
   // ///                 Jacobian matrix.
   // /// \param[out]  P  The evaluated 1st Piola-Kirchhoff stress tensor. */
   // virtual void EvalP(const mfem::DenseMatrix &Jpt, mfem::DenseMatrix &P) const = 0;

   // /// \brief Evaluate the derivative of the 1st Piola-Kirchhoff stress tensor
   // ///        and assemble its contribution to the local gradient matrix 'A'.
   // /// \param[in] Jpt     Represents the target->physical transformation
   // ///                    Jacobian matrix.
   // /// \param[in] DS      Gradient of the basis matrix (dof x dim).
   // /// \param[in] weight  Quadrature weight coefficient for the point.
   // /// \param[in,out]  A  Local gradient matrix where the contribution from this
   // ///                    point will be added.
	// ///
   // /// Computes weight * d(dW_dxi)_d(xj) at the current point, for all i and j,
   // /// where x1 ... xn are the FE dofs. This function is usually defined using
   // /// the matrix invariants and their derivatives.
   // virtual void AssembleH(const mfem::DenseMatrix &Jpt, const mfem::DenseMatrix &DS,
   //                        const double weight, mfem::DenseMatrix &A) const = 0;

protected:
	/// Reference-element to target-element transformation.
   // mfem::ElementTransformation *Ttr;
};

/// Integrator for (\nu(u)*curl u, curl v) for Nedelec elements
class CurlCurlNLFIntegrator : public mfem::NonlinearFormIntegrator
{
public:
	/// Construct a curl curl nonlinear form integrator for Nedelec elements
   /// \param[in] diff_stack - for algorithmic differentiation
	/// \param[in] m - model describing nonlinear material parameter
   CurlCurlNLFIntegrator(ElectromageticMaterialModel *m,
								 double a = 1.0) 
		: model(m), alpha(a) {}

   /// Construct the element local residual
   /// \param[in] el - the finite element whose residual we want
   /// \param[in] Trans - defines the reference to physical element mapping
   /// \param[in] elfun - element local state vector
   /// \param[out] elvect - element local residual
   virtual void AssembleElementVector(const mfem::FiniteElement &el,
                                      mfem::ElementTransformation &Trans,
                                      const mfem::Vector &elfun,
                                      mfem::Vector &elvect);

   // virtual void AssembleElementGrad(const mfem::FiniteElement &el,
   //                                  mfem::ElementTransformation &Ttr,
   //                                  const mfem::Vector &elfun,
   //                                  mfem::DenseMatrix &elmat);

private:
	/// material model describing electromagnetic behavior (ex. permeability)
	ElectromageticMaterialModel *model;
   /// scales the terms; can be used to move to rhs/lhs
	double alpha;

#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix curlshape, curlshape_dFt, M;
   mfem::DenseMatrix vshape, projcurl;
   mfem::Vector tempVec;
#endif

};

} // namespace mach

#endif