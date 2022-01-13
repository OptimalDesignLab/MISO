#ifndef MATRIX_OPERATORS
#define MATRIX_OPERATORS

#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "mach_residual.hpp"

namespace mach
{
/// Linear combination of two `Operators`
class SumOfOperators : public mfem::Operator
{
public:
   /// Construct an operator out of the given linear combination
   /// \param[in] alpha - scalar multiplying `oper1`
   /// \param[in] oper1 - first operator in the sum
   /// \param[in] beta - scalar multiplying `oper2`
   /// \param[in] oper2 - second operator in the sum
   /// \note Produces an operator whose `Mult(x,y)` performs `y = a*v + b*w`
   /// where `oper1.Mult(x,v)` and `oper2.Mult(x,w)`
   SumOfOperators(double alpha,
                  mfem::Operator &oper1,
                  double beta,
                  mfem::Operator &oper2);

   /// Performs the linear combination of the operators on `x` to produce `y`
   /// \param[in] x - the vector being acted/operated on
   /// \param[out] y - the result of the action
   void Mult(const mfem::Vector &x, mfem::Vector &y) const override;

private:
   /// scalars in the linear combination
   double a, b;
   /// first operator making up the sum
   mfem::Operator &oper_a;
   /// second operator making up the sum
   mfem::Operator &oper_b;
   /// work vector
   mutable mfem::Vector work_vec;
};

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
   JacobianFree(T &residual, MPI_Comm incomm = MPI_COMM_WORLD);

   /// Sets the state at which the Jacobian is evaluated
   /// \param[in] baseline - state where Jacobian is to be evaluated
   /// \note This function also evaluates the residual at `baseline`
   void setState(const mfem::Vector &baseline);

   /// Sets the state at which the Jacobian is evaluated
   /// \param[in] inputs - contains key "state" where Jacobian is evaluated
   /// \note This function also evaluates the residual at given state
   void setState(const MachInputs &inputs);

   /// Approximates `y = J*x` using a forward difference approximation
   /// \param[in] x - the vector being multiplied by the Jacobian
   /// \param[out] y - the result of the product
   void Mult(const mfem::Vector &x, mfem::Vector &y) const override;

private:
   /// MPI communicator for vector dot products
   MPI_Comm comm;
   /// residual that defines the Jacobian
   T &res;
   /// baseline state about which we perturb
   mfem::Vector state;
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
JacobianFree<T>::JacobianFree(T &residual, MPI_Comm incomm)
 : Operator(getSize(residual)),
   comm(incomm),
   res(residual),
   state(getSize(res)),
   res_at_state(getSize(res)),
   state_pert(getSize(res))
{ }

template <typename T>
void JacobianFree<T>::setState(const mfem::Vector &baseline)
{
   // store baseline state, because we need to perturb it later
   state = baseline;
   // initialize the res_at_state vector for later use
   auto inputs = MachInputs({{"state", baseline}});
   evaluate(res, inputs, res_at_state);
}

template <typename T>
void JacobianFree<T>::setState(const MachInputs &inputs)
{
   // store baseline state, because we need to perturb it later
   setVectorFromInputs(inputs, "state", state, true, true);
   evaluate(res, inputs, res_at_state);
}

template <typename T>
void JacobianFree<T>::Mult(const mfem::Vector &x, mfem::Vector &y) const
{
   double eps_fd = getStepSize(state, x);
   // create the perturbed vector, and evaluate the residual
   add(state, eps_fd, x, state_pert);
   auto inputs = MachInputs({{"state", state_pert}});
   evaluate(res, inputs, y);
   // subtract the baseline residual and divide by eps_fd to get product
   subtract(1 / eps_fd, y, res_at_state, y);
}

template <typename T>
double JacobianFree<T>::getStepSize(const mfem::Vector &baseline,
                                    const mfem::Vector &pert) const
{
   // This is based on the step size suggested by Chisholm and Zingg in
   // "A Jacobian-Free Newton-Krylov Algorithm for Compressible Turbulent Fluid
   // Flows", https://doi.org/10.1016/j.jcp.2009.02.004
   const double delta = 1e-10;
   double prod = InnerProduct(comm, pert, pert);
   if (prod > 1e-6)
   {
      return sqrt(delta / prod);
   }
   else
   {
      return 1e-7;
   }
}

}  // namespace mach

#endif  // MATRIX_OPERATORS