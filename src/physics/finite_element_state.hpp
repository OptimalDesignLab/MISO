#ifndef MACH_FINITE_ELEMENT_STATE
#define MACH_FINITE_ELEMENT_STATE

#include <functional>
#include <memory>

#include "mfem.hpp"

#include "finite_element_vector.hpp"

namespace mach
{
/**
 * @brief Class for encapsulating the critical MFEM components of a primal
 * finite element field
 *
 * Namely: Mesh, FiniteElementCollection, FiniteElementState,
 * GridFunction, and a Vector of the solution
 */
class FiniteElementState : public FiniteElementVector
{
public:
   /**
    * @brief Use the finite element vector constructors
    */
   using FiniteElementVector::FiniteElementVector;
   using FiniteElementVector::operator=;

   /**
    * Returns a non-owning reference to the internal grid function
    */
   mfem::ParGridFunction &gridFunc() { return *gf; }
   /// \overload
   const mfem::ParGridFunction &gridFunc() const { return *gf; }

   /**
    * Returns a GridFunctionCoefficient referencing the internal grid function
    */
   mfem::GridFunctionCoefficient gridFuncCoef() const
   {
      return mfem::GridFunctionCoefficient{gf.get(), gf->VectorDim()};
   }

   /**
    * Returns a VectorGridFunctionCoefficient referencing the internal grid
    * function
    */
   mfem::VectorGridFunctionCoefficient vectorGridFuncCoef() const
   {
      return mfem::VectorGridFunctionCoefficient{gf.get()};
   }

   /**
    * Projects a coefficient (vector or scalar) onto the field
    * @param[in] coef The coefficient to project
    */
   void project(mfem::Coefficient &coef)
   {
      gf->ProjectCoefficient(coef);
      initializeTrueVec();
   }
   /// \overload
   void project(mfem::VectorCoefficient &coef)
   {
      gf->ProjectCoefficient(coef);
      initializeTrueVec();
   }

   /**
    * @brief Set a finite element state to a constant value
    *
    * @param value The constant to set the finite element state to
    * @return The modified finite element state
    * @note This sets the true degrees of freedom and then broadcasts to the
    * shared grid function entries. This means that if a different value is
    * given on different processors, a shared DOF will be set to the owning
    * processor value.
    */
   FiniteElementState &operator=(const double value)
   {
      FiniteElementVector::operator=(value);
      return *this;
   }
};

/**
 * @brief Calculate the Lp norm of a finite element state
 *
 * @param state The state variable to compute a norm of
 * @param p Order of the norm
 * @return The norm value
 */
double norm(const FiniteElementState &state, double p = 2);

}  // namespace mach

#endif
