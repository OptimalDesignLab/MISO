#ifndef MACH_LAPLACE
#define MACH_LAPLACE

#include "viscous_integ.hpp"

namespace mach
{

/// Volume integrator for scalar, second-order operator
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \note This derived class uses the CRTP
template <int dim>
class LaplaceIntegrator : public SymmetricViscousIntegrator<LaplaceIntegrator<dim>>
{
public:
   /// Construct a Laplace integrator
   /// \param[in] diff_stack - for algorithmic differentiation (not used)
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   LaplaceIntegrator(adept::Stack &diff_stack, double a = 1.0)
       : SymmetricViscousIntegrator<LaplaceIntegrator<dim>>(
             diff_stack, 1, a) {}

   /// A do-nothing conversion
   /// \param[in] u - states that are to be converted
   /// \param[out] w - copy of state out
   void convertVars(const mfem::Vector &u, mfem::Vector &w) { w = u; }

   void convertVarsJacState(const mfem::Vector &q, mfem::DenseMatrix &dwdu) {}

   /// Applies a scaling to the derivatives
   /// \param[in] d - index `d` in \f$ C_{d,:} \f$ scalars
   /// \param[in] x - coordinate location at which scaling evaluated
   /// \param[in] u - state at which the symmetric matrices `C` are evaluated
   /// \param[in] Du - `Du[:,d2]` stores derivative of `u` in direction `d2`. 
   /// \param[out] CDu - product of the multiplication between the `C` and `Du`.
   void applyScaling(int d, const mfem::Vector &x, const mfem::Vector &u,
                     const mfem::DenseMatrix &Du, mfem::Vector &CDu)
   {
      double mu = x(0); 
      CDu(0) = mu*Du(0,d); // no cross derivatives
   }

   void applyScalingJacState(int d, const mfem::Vector &x, 
                             const mfem::Vector &u,
                             const mfem::DenseMatrix &Du,
                             mfem::DenseMatrix &CDu_jac) {}

   void applyScalingJacDw(
      int d, const mfem::Vector &x, const mfem::Vector &u,
      const mfem::DenseMatrix &Du,
      std::vector<mfem::DenseMatrix> &CDu_jac) {}

   /// This allows the base class to access the number of dimensions
   static const int ndim = dim;
};

/// Integrator for Laplace natural boundaries
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \note This derived class uses the CRTP
template <int dim>
class LaplaceNaturalBC : public ViscousBoundaryIntegrator<LaplaceNaturalBC<dim>>
{
public:
   /// Constructs an integrator for a Laplace natural boundary flux
   /// \param[in] diff_stack - for algorithmic differentiation (not used)
   /// \param[in] fe_coll - used to determine the face elements
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   LaplaceNaturalBC(adept::Stack &diff_stack,
                    const mfem::FiniteElementCollection *fe_coll,
                    double a = 1.0)
       : ViscousBoundaryIntegrator<LaplaceNaturalBC<dim>>(
             diff_stack, fe_coll, 1, a) {}

   /// A do-nothing conversion
   /// \param[in] u -  state in that are to be converted
   /// \param[out] w - copy of state out
   void convertVars(const mfem::Vector &u, mfem::Vector &w) { w = u; }

   void convertVarsJacState(const mfem::Vector &u, mfem::DenseMatrix &dwdu) {}

   double calcBndryFun(const mfem::Vector &x, const mfem::Vector &dir,
                       double jac, const mfem::Vector &u,
                       const mfem::DenseMatrix &Dw) {}

   /// Compute Laplace natural boundary flux
   /// \param[in] x - coordinate location at which flux is evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] jac - mapping Jacobian (needed by no-slip penalty)
   /// \param[in] u - state variables at which to evaluate the flux
   /// \param[in] Du - space derivatives of the state
   /// \param[out] flux_vec - value of the flux
   void calcFlux(const mfem::Vector &x, const mfem::Vector &dir, double jac,
                 const mfem::Vector &q, const mfem::DenseMatrix &Dw,
                 mfem::Vector &flux_vec)
   {
      double mu = x(0);
      flux_vec = 0.0;
      for (int d = 0; d < dim; ++d)
      {
         flux_vec(0) -= mu*Dw(0,d)*dir(d);
      }
   }

   /// Compute boundary fluxes that are scaled by test function derivative
   /// \param[in] x - coordinate location at which fluxes are evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] u - state at which to evaluate the flux
   /// \param[out] flux_mat - `flux_mat[:,di]` to be scaled by `D_[di] v` 
   void calcFluxDv(const mfem::Vector &x, const mfem::Vector &dir,
               const mfem::Vector &u, mfem::DenseMatrix &flux_mat)
   {
      flux_mat = 0.0;
   }

   void calcFluxJacState(const mfem::Vector &x, const mfem::Vector &dir,
                         double jac, const mfem::Vector &q,
                         const mfem::DenseMatrix &Dw,
                         mfem::DenseMatrix &flux_jac) {}

   void calcFluxJacDw(const mfem::Vector &x, const mfem::Vector &dir,
                      double jac, const mfem::Vector &q,
                      const mfem::DenseMatrix &Dw,
                      std::vector<mfem::DenseMatrix> &flux_jac) {}

   void calcFluxDvJacState(const mfem::Vector &x, const mfem::Vector dir, 
                           const mfem::Vector &u,
                           std::vector<mfem::DenseMatrix> &flux_jac) {}
};

/// Implements a source for verifying the viscous terms
/// \note The source is based on d/dx( x * d/dx (x^deg))
class SourceIntegrator : public mfem::NonlinearFormIntegrator
{
public :
   SourceIntegrator(int degree) { deg = degree; }

   /// Construct the element local residual
   /// \param[in] el - the finite element whose residual we want
   /// \param[in] Trans - defines the reference to physical element mapping
   /// \param[in] elfun - element local state function
   /// \param[out] elvect - element local residual
   virtual void AssembleElementVector(const mfem::FiniteElement &el,
                                      mfem::ElementTransformation &Trans,
                                      const mfem::Vector &elfun,
                                      mfem::Vector &elvect)
   {
      using namespace mfem;
      const SBPFiniteElement &sbp = dynamic_cast<const SBPFiniteElement &>(el);
      int num_nodes = sbp.GetDof();
      int dim = sbp.GetDim();
      Vector xi(dim);
      elvect.SetSize(num_nodes);
      elvect = 0.0;
      for (int i = 0; i < num_nodes; ++i)
      {
         // get the Jacobian (Trans.Weight) and cubature weight (node.weight)
         const IntegrationPoint &node = el.GetNodes().IntPoint(i);
         Trans.SetIntPoint(&node);
         double H = (sbp.getDiagNormEntry(i) * Trans.Weight());
         Trans.Transform(node, xi);
         // d/dx( x d x^deg/dx) = d/dx( x deg*x^(deg-1)) = deg^2 x^(deg-1)
         double source = pow(xi(0), deg-1)*deg*deg;
         elvect(i) += H*source;         
      }
   }

private:
   /// Degree of the target operator being tested
   int deg; 
};

} // namespace mach

#endif