#ifndef MACH_FINITE_ELEMENT_DUAL
#define MACH_FINITE_ELEMENT_DUAL

#include <memory>

#include "mfem.hpp"

#include "finite_element_vector.hpp"

namespace mach
{
/**
 * @brief Class for encapsulating the dual vector space of a finite element
 * space (i.e. the space of linear forms as applied to a specific basis set)
 */
class FiniteElementDual : public FiniteElementVector
{
public:
   /**
    * @brief Use the finite element vector constructors
    */
   using FiniteElementVector::FiniteElementVector;

   /**
    * @brief Returns a non-owning reference to the local degrees of freedom
    *
    * @return mfem::Vector& The local dof vector
    * @note While this is a grid function for plotting and parallelization, we
    * only return a vector type as the user should not use the interpolation
    * capabilities of a grid function on the dual space
    * @note Shared degrees of freedom live on multiple MPI ranks
    */
   mfem::Vector &localVec() { return *gf_; }
   /// @overload
   const mfem::Vector &localVec() const { return *gf_; }

   /**
    * @brief Set a finite element dual to a constant value
    *
    * @param value The constant to set the finite element dual to
    * @return The modified finite element dual
    * @note This sets the true degrees of freedom and then broadcasts to the
    * shared grid function entries. This means that if a different value is
    * given on different processors, a shared DOF will be set to the owning
    * processor value.
    */
   FiniteElementDual &operator=(const double value)
   {
      FiniteElementVector::operator=(value);
      return *this;
   }
};

}  // namespace mach

#endif
