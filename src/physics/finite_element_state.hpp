#ifndef MACH_FINITE_ELEMENT_STATE
#define MACH_FINITE_ELEMENT_STATE

#include <functional>
#include <memory>

#include "mfem.hpp"

#include "finite_element_vector.hpp"

namespace mach
{
/// \brief Class for encapsulating the critical MFEM components of a primal
/// finite element field
class FiniteElementState : public FiniteElementVector
{
public:
   /// \brief Use the finite element vector constructors
   using FiniteElementVector::FiniteElementVector;
   using FiniteElementVector::operator=;

   /// Returns a non-owning reference to the internal grid function
   mfem::ParGridFunction &gridFunc() { return *gf; }
   /// \overload
   const mfem::ParGridFunction &gridFunc() const { return *gf; }

   /// \brief Set the internal grid function using the true DOF values
   /// \param[in] true_vec - the true dof vector containing the values to
   /// \note This sets the finite element dofs by multiplying the true dofs
   /// by the prolongation operator.
   /// \see <a href="https://mfem.org/pri-dual-vec/">MFEM documentation</a> for
   /// details
   void distributeSharedDofs(const mfem::Vector &true_vec) override
   {
      gridFunc().SetFromTrueDofs(true_vec);
   }

   /// \brief Set the true vector from the grid function values
   /// \param[out] true_vec - the true dof vector to set from the local field
   /// \note This sets the true vector dofs by multiplying the finite element
   /// dofs by the restriction operator.
   /// \see <a href="https://mfem.org/pri-dual-vec/">MFEM documentation</a> for
   /// details
   void setTrueVec(mfem::Vector &true_vec) override
   {
      gridFunc().GetTrueDofs(true_vec);
   }

   /// Returns a GridFunctionCoefficient referencing the internal grid function
   mfem::GridFunctionCoefficient gridFuncCoef() const
   {
      return mfem::GridFunctionCoefficient{gf.get(), gf->VectorDim()};
   }

   /// Returns a VectorGridFunctionCoefficient referencing the internal grid
   /// function
   mfem::VectorGridFunctionCoefficient vectorGridFuncCoef() const
   {
      return mfem::VectorGridFunctionCoefficient{gf.get()};
   }

   /// Projects a coefficient (vector or scalar) onto the field
   /// \param[in] coef The coefficient to project
   /// \param[out] true_vec The true degree of freedom vector with the
   /// projected dofs
   void project(mfem::Coefficient &coef, mfem::Vector &true_vec)
   {
      gf->ProjectCoefficient(coef);
      setTrueVec(true_vec);
   }
   /// \overload
   void project(mfem::VectorCoefficient &coef, mfem::Vector &true_vec)
   {
      gf->ProjectCoefficient(coef);
      setTrueVec(true_vec);
   }

   void project(std::function<double(const mfem::Vector &)> fun,
                mfem::Vector &true_vec)
   {
      mfem::FunctionCoefficient coeff(std::move(fun));
      project(coeff, true_vec);
   }

   void project(std::function<void(const mfem::Vector &, mfem::Vector &)> fun,
                mfem::Vector &true_vec)
   {
      int vdim = gridFunc().VectorDim();
      mfem::VectorFunctionCoefficient coeff(vdim, std::move(fun));
      project(coeff, true_vec);
   }

   // /// \brief Set a finite element state to a constant value
   // /// \param value The constant to set the finite element state to
   // /// \return The modified finite element state
   // /// \note This sets the true degrees of freedom and then broadcasts to the
   // /// shared grid function entries. This means that if a different value is
   // /// given on different processors, a shared DOF will be set to the owning
   // /// processor value.
   // FiniteElementState &operator=(const double value)
   // {
   //    FiniteElementVector::operator=(value);
   //    return///this;
   // }
};

/// \brief Calculate the Lp norm of a finite element state
/// \param[in] state - The state variable to compute a norm of
/// \param[in] p - Order of the norm
/// \return The norm value
double calcLpNorm(const FiniteElementState &state, double p = 2);

/// \brief Calculate the L2 error of a finite element state
/// \param[in] state - The state variable to compute a error of
/// \param[in] ex_sol - Coefficient describing the exact solution
/// \return The error value
double calcL2Error(const FiniteElementState &state, mfem::Coefficient &ex_sol);

/// \brief Calculate the L2 error of a finite element state
/// \param[in] state - The state variable to compute a error of
/// \param[in] ex_sol - Coefficient describing the exact solution
/// \param[in] entry - if >= 0, the Lp error of state `entry` is returned
/// \return The error value
double calcL2Error(const FiniteElementState &state,
                   mfem::VectorCoefficient &ex_sol,
                   const int entry);

/// \brief Calculate the Lp error of a finite element state
/// \param[in] state - The state variable to compute a error of
/// \param[in] ex_sol - Coefficient describing the exact solution
/// \param[in] p - Order of the norm
/// \return The error value
double calcLpError(const FiniteElementState &state,
                   mfem::Coefficient &exsol,
                   const double p = 2);
///\overload
inline double calcLpError(const FiniteElementState &state,
                          std::function<double(const mfem::Vector &)> ex_sol,
                          double p = 2)
{
   mfem::FunctionCoefficient coeff(std::move(ex_sol));
   return calcLpError(state, coeff, p);
}

/// \brief Calculate the Lp error of a finite element state
/// \param[in] state - The state variable to compute a error of
/// \param[in] ex_sol - Coefficient describing the exact solution
/// \param[in] p - Order of the norm
/// \param[in] entry - if >= 0, the Lp error of state `entry` is returned
/// \return The error value
double calcLpError(const FiniteElementState &state,
                   mfem::VectorCoefficient &ex_sol,
                   const double p = 2,
                   const int entry = -1);
/// \overload
inline double calcLpError(
    const FiniteElementState &state,
    std::function<void(const mfem::Vector &, mfem::Vector &)> ex_sol,
    double p = 2,
    int entry = -1)
{
   int vdim = state.gridFunc().VectorDim();
   mfem::VectorFunctionCoefficient coeff(vdim, std::move(ex_sol));
   return calcLpError(state, coeff, p, entry);
}

}  // namespace mach

#endif
