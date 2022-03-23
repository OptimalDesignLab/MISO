#ifndef MACH_DIAG_MASS_INTEG
#define MACH_DIAG_MASS_INTEG

#include "mfem.hpp"

namespace mach
{

/// Integrator for diagonal mass matrices that arise in SBP discretizations
class DiagMassIntegrator : public mfem::BilinearFormIntegrator
{
public:
   /// Constructs a diagonal-mass matrix integrator.
   /// \param[in] nvar - number of state variables
   explicit DiagMassIntegrator(int nvar = 1, bool space_var = false)
      : num_state(nvar), space_vary_dt(space_var) { }

   /// Finds the diagonal mass matrix for the given element.
   /// \param[in] el - the element for which the mass matrix is desired
   /// \param[in,out] trans - curvilinear transformation
   /// \param[out] elmat - the element mass matrix
   void AssembleElementMatrix(const mfem::FiniteElement &el,
                              mfem::ElementTransformation &Trans,
                              mfem::DenseMatrix &elmat);

private:
   /// number of state variables; helps determine the dimensions of mass matrix
   int num_state;

   /// if true, the diagonal entries are scaled by det(Jac)^(1/dim)
   bool space_vary_dt;
};

} // namespace mach

#endif