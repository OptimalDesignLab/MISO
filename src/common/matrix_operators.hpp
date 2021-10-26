#ifndef MATRIX_OPERATORS
#define MATRIX_OPERATORS

#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "mach_residual.hpp"

namespace mach
{

/// Defines Jacobian-vector products using finite-differences
/// \note Presently the object computes the residual at the given baseline 
/// state; this is likely unnecessary, since the residual at that state will be 
/// needed elsewhere.  Therefore, passing in the evaluated residual should be 
/// considered in the future.
/// \tparam T - a residual class
template <typename T>
class JacobianFree : public mfem::Operator
{
public:
   /// Construct a Jacobian-free matrix-vector product operator 
   /// \param[in] residual - the equation/residual that defines the Jacobian
   /// \param[in] comm - MPI communicator needed for global inner products 
   JacobianFree(T &residual, MPI_Comm comm=MPI_COMM_WORLD);

   /// Sets the state at which the Jacobian is evaluated
   /// \param[in] baseline - state where Jacobian is to be evaluated
   /// \note This function also evaluates the residual at `baseline`
   void setState(const mfem::Vector &baseline);
      
   /// Approximates `y = J*x` using a forward difference approximation
   /// \param[in] x - the vector being multiplied by the Jacobian 
   /// \param[out] y - the result of the product
   void Mult(const mfem::Vector &x, mfem::Vector &y) const override;

private:
   /// MPI communicator for vector dot products 
   MPI_Comm comm;
   /// residual that defines the Jacobian
   T &res;
   /// pointer to the baseline state 
   const mfem::Vector *state;
   /// residual evaluated at `state`
   mfem::Vector res_at_state;
   /// work vector needed to compute the Jacobian-free product 
   mutable mfem::Vector state_pert;

   /// Returns a (hopefully) appropriate forward-difference step size
   /// \param[in] baseline - the state at which the Jacobian is computed
   /// \param[in] pert - the perturbed state
   /// \returns the forward difference step size
   double getStepSize(const mfem::Vector &baseline,
                      const mfem::Vector &pert) const;
};

template <typename T>
JacobianFree<T>::JacobianFree(T &residual, MPI_Comm comm)
                              : res(residual),
                              res_at_state(getSize(res)),
                              state_pert(getSize(res)) {}

template <typename T>
void JacobianFree<T>::setState(const mfem::Vector &baseline)
{
   state = &baseline;
   // initialize the res_at_state vector for later use
   auto inputs = MachInputs({{"state", state->GetData()}});
   evaluate(res, inputs, res_at_state);
}

template <typename T>
void JacobianFree<T>::Mult(const mfem::Vector &x, mfem::Vector &y) const
{
   double eps_fd = getStepSize(*state, x);
   // create the perturbed vector, and evaluate the residual
   add(*state, eps_fd, x, state_pert);
   auto inputs = MachInputs({{"state", state_pert.GetData()}});
   evaluate(res, inputs, y);
   // subtract the baseline residual and divide by eps_fd to get product
   subtract(1/eps_fd, y, res_at_state, y);
}

template <typename T>
double JacobianFree<T>::getStepSize(const mfem::Vector &baseline,
                                    const mfem::Vector &pert) const
{
   // This currently uses the step size suggested by Chisholm and Zingg in 
   // "A Jacobian-Free Newton-Krylov Algorithm for Compressible Turbulent Fluid 
   // Flows", https://doi.org/10.1016/j.jcp.2009.02.004
   const double delta = 1e-10;
   double prod = InnerProduct(comm, pert, pert);
   return sqrt(delta/prod);
}

} // namespace mach 

#endif // MATRIX_OPERATORS