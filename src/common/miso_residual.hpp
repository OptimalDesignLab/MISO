#ifndef MISO_RESIDUAL
#define MISO_RESIDUAL

#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "miso_input.hpp"
#include "utils.hpp"

namespace miso
{
template <typename T>
void linearize(T & /*unused*/, const MISOInputs & /*unused*/)
{
   throw NotImplementedException(
       "not specialized for concrete residual type!\n");
}

template <typename T>
MPI_Comm getMPIComm(const T & /*unused*/)
{
   throw MISOException("getComm not specialized for concrete residual type!\n");
}

template <typename T>
mfem::Operator &getJacobianTranspose(T & /*unused*/,
                                     const MISOInputs & /*unused*/,
                                     const std::string & /*unused*/)
{
   throw NotImplementedException(
       "not specialized for concrete residual type!\n");
}

template <typename T>
void setUpAdjointSystem(T & /*unused*/,
                        mfem::Solver & /*unused*/,
                        const MISOInputs & /*unused*/,
                        mfem::Vector & /*unused*/,
                        mfem::Vector & /*unused*/)
{
   throw NotImplementedException(
       "not specialized for concrete residual type!\n");
}

template <typename T>
void finalizeAdjointSystem(T & /*unused*/,
                           mfem::Solver & /*unused*/,
                           const MISOInputs & /*unused*/,
                           mfem::Vector & /*unused*/,
                           mfem::Vector & /*unused*/)
{
   throw NotImplementedException(
       "not specialized for concrete residual type!\n");
}

template <typename T>
double jacobianVectorProduct(T & /*unused*/,
                             const mfem::Vector & /*unused*/,
                             const std::string & /*unused*/)
{
   throw NotImplementedException("not specialized for concrete type!\n");
}

template <typename T>
void jacobianVectorProduct(T & /*unused*/,
                           const mfem::Vector & /*unused*/,
                           const std::string & /*unused*/,
                           mfem::Vector & /*unused*/)
{
   throw NotImplementedException("not specialized for concrete type!\n");
}
template <typename T>
double vectorJacobianProduct(T & /*unused*/,
                             const mfem::Vector & /*unused*/,
                             const std::string & /*unused*/)
{
   throw NotImplementedException("not specialized for concrete type!\n");
}

template <typename T>
void vectorJacobianProduct(T &,
                           const mfem::Vector & /*unused*/,
                           const std::string & /*unused*/,
                           mfem::Vector & /*unused*/)
{
   throw NotImplementedException("not specialized for concrete type!\n");
}

template <typename T>
double calcEntropy(T & /*unused*/, const MISOInputs & /*unused*/)
{
   throw NotImplementedException(
       "not specialized for concrete residual type!\n");
}

template <typename T>
double calcEntropyChange(T & /*unused*/, const MISOInputs & /*unused*/)
{
   throw NotImplementedException(
       "not specialized for concrete residual type!\n");
}

template <typename T>
double calcSupplyRate(T & /*unused*/, const MISOInputs & /*unused*/)
{
   throw MISOException(
       "calcSupplyRate not specialized for concrete residual type!\n");
}

template <typename T>
mfem::Operator *getMassMatrix(T & /*unused*/, const nlohmann::json & /*unused*/)
{
   return nullptr;
}

template <typename T>
mfem::Solver *getPreconditioner(T & /*unused*/)
{
   return nullptr;
}

template <typename T>
mfem::Operator &getJacobianBlock(T & /*unused*/,
                                 const MISOInputs & /*unused*/,
                                 int)
{
   throw MISOException(
       "getJacobianBlock not specialized for concrete residual type!\n");
}

/// Defines a common interface for residual functions used by miso.
/// A MISOResidual can wrap any type `T` that has the interface of a residual
/// function.  For example, one instance of `T` is given by `MISONonlinearForm`,
/// so that every nonlinear form can be wrapped by a MISOResidual.
/// \note We use this approach to achieve polymorphism without inheritance.
/// This is important because we need to derive from `mfem` classes frequently,
/// but at the same time we need to build on their classes' functionality.
/// Without this approach, we would need to use frequent dynamic casts because
/// we have pointers to base classes.
/// \note This approach is based on the example in Sean Parent's talk:
/// ``Inheritance is the base class of evil''
class MISOResidual final : public mfem::Operator
{
public:
   /// Returns a reference to the underlying concrete type
   /// \tparam T - a residual class
   /// \note This is useful when the underlying concrete type is known at the
   /// client side, and methods specific to the concrete type need to be called.
   template <typename T>
   friend T &getConcrete(MISOResidual &residual);

   template <typename T>
   friend const T &getConcrete(const MISOResidual &residual);

   /// Get the MPI Communicator associated with the residual
   /// \param[in] residual - the residual whose comm is desired
   /// \returns the MPI Communicator for the residual
   /// \note Optional.  Needed for some global operations (inner products)
   friend MPI_Comm getMPIComm(const MISOResidual &residual);

   /// Gets the number of equations/unknowns of the underlying residual type
   /// \param[inout] residual - the residual whose size is being queried
   /// \returns the number of equations/unknowns
   /// \note Needed, e.g., by the ODESystemOperator constructor (see evolver.*)
   friend int getSize(const MISOResidual &residual);

   /// Set inputs in the underlying residual type
   /// \param[inout] residual - the residual being assigned the input
   /// \param[in] inputs - the inputs that are being assigned
   /// \note Ends up calling `setInputs` on either the `MISONonlinearForm` or
   /// a specialized version for each particular residual.
   friend void setInputs(MISOResidual &residual, const MISOInputs &inputs);

   /// Set options in the underlying residual type
   /// \param[inout] residual - the residual whose options are being set
   /// \param[in] options - the options that are being assigned
   friend void setOptions(MISOResidual &residual,
                          const nlohmann::json &options);

   /// Evaluate the residual function at given inputs and return as `res_vec`
   /// \param[inout] residual - the residual being evaluated
   /// \param[in] inputs - the independent variables at which to evaluate `res`
   /// \param[out] res_vec - the dependent variable, the output from `residual`
   friend void evaluate(MISOResidual &residual,
                        const MISOInputs &inputs,
                        mfem::Vector &res_vec);

   /// Cache inputs for the residual and internally store Jacobians
   /// \param[inout] residual - the residual being evaluated
   /// \param[in] inputs - the independent variables at which to evaluate `res`
   friend void linearize(MISOResidual &residual, const MISOInputs &inputs);

   /// Compute the Jacobian of the given residual and return a reference to it
   /// \param[inout] residual - function whose Jacobian we want
   /// \param[in] inputs - the variables needed to evaluate the Jacobian
   /// \param[in] wrt - the input we are differentiating with respect to
   /// \returns a reference to the residual's Jacobian with respect to `wrt`
   /// \note the underlying `Operator` is owned by `residual`
   friend mfem::Operator &getJacobian(MISOResidual &residual,
                                      const MISOInputs &inputs,
                                      const std::string &wrt);

   /// Compute the transpose of the Jacobian of the given residual and return
   /// a reference to it
   /// \param[inout] residual - function whose Jacobian we want
   /// \param[in] inputs - the variables needed to evaluate the Jacobian
   /// \param[in] wrt - the input we are differentiating with respect to
   /// \returns a reference to the residual's Jacobian with respect to `wrt`
   /// transposed
   /// \note the underlying `Operator` is owned by `residual`
   friend mfem::Operator &getJacobianTranspose(MISOResidual &residual,
                                               const MISOInputs &inputs,
                                               const std::string &wrt);

   /// Get a reference to `iblock`th block of the Jacobian for a block system
   /// \param[inout] residual - function whose Jacobian we want
   /// \param[in] inputs - the variables needed to evaluate the Jacobian
   /// \param[in] iblock - the block whose Jacobian is sought
   /// \note this is for the state Jacobian only
   friend mfem::Operator &getJacobianBlock(MISOResidual &residual,
                                           const MISOInputs &inputs,
                                           int iblock);

   friend void setUpAdjointSystem(MISOResidual &residual,
                                  mfem::Solver &adj_solver,
                                  const MISOInputs &inputs,
                                  mfem::Vector &state_bar,
                                  mfem::Vector &adjoint);

   friend void finalizeAdjointSystem(MISOResidual &residual,
                                     mfem::Solver &adj_solver,
                                     const MISOInputs &inputs,
                                     mfem::Vector &state_bar,
                                     mfem::Vector &adjoint);

   /// Compute the residual's sensitivity to a scalar and contract it with
   /// wrt_dot
   /// \param[inout] residual - the residual whose sensitivity we want
   /// \param[in] wrt_dot - the "wrt"-sized vector to contract with the
   /// sensitivity
   /// \param[in] wrt - string denoting what variable to take the derivative
   /// with respect to
   /// \return the assembled/contracted sensitivity
   friend double jacobianVectorProduct(MISOResidual &residual,
                                       const mfem::Vector &wrt_dot,
                                       const std::string &wrt);

   /// Compute the residual's sensitivity to a vector and contract it with
   /// wrt_dot
   /// \param[inout] residual - the residual whose sensitivity we want
   /// \param[in] wrt_dot - the "wrt"-sized vector to contract with the
   /// sensitivity
   /// \param[in] wrt - string denoting what variable to take the derivative
   /// with respect to
   /// \param[inout] res_dot - the assembled/contracted sensitivity is
   /// accumulated into res_dot
   friend void jacobianVectorProduct(MISOResidual &residual,
                                     const mfem::Vector &wrt_dot,
                                     const std::string &wrt,
                                     mfem::Vector &res_dot);

   /// Compute the residual's sensitivity to a scalar and contract it with
   /// res_bar
   /// \param[inout] residual - the residual whose sensitivity we want
   /// \param[in] res_bar - the residual-sized vector to contract with the
   /// sensitivity
   /// \param[in] wrt - string denoting what variable to take the derivative
   /// with respect to
   /// \return the assembled/contracted sensitivity
   friend double vectorJacobianProduct(MISOResidual &residual,
                                       const mfem::Vector &res_bar,
                                       const std::string &wrt);

   /// Compute the residual's sensitivity to a vector and contract it with
   /// res_bar
   /// \param[inout] residual - the residual whose sensitivity we want
   /// \param[in] res_bar - the residual-sized vector to contract with the
   /// sensitivity
   /// \param[in] wrt - string denoting what variable to take the derivative
   /// with respect to
   /// \param[inout] wrt_bar - the assembled/contracted sensitivity is
   /// accumulated into wrt_bar
   friend void vectorJacobianProduct(MISOResidual &residual,
                                     const mfem::Vector &res_bar,
                                     const std::string &wrt,
                                     mfem::Vector &wrt_bar);

   /// Evaluate the entropy functional at the given state
   /// \param[inout] residual - function with an associated entropy
   /// \param[in] inputs - the variables needed to evaluate the entropy
   /// \return the entropy functional
   /// \note optional, but must be implemented for relaxation RK
   friend double calcEntropy(MISOResidual &residual, const MISOInputs &inputs);

   /// Evaluate the residual weighted by the entropy variables
   /// \param[inout] residual - function with an associated entropy
   /// \param[in] inputs - the variables needed to evaluate the entropy
   /// \return the product `w^T res`
   /// \note The entropy variables, `w`, are evaluated at `state`, and `res` is
   /// equal to `-state_dot`.
   friend double calcEntropyChange(MISOResidual &residual,
                                   const MISOInputs &inputs);

   friend double calcSupplyRate(MISOResidual &residual,
                                const MISOInputs &inputs);

   /// Return the mass matrix corresponding to the residual
   /// \param[inout] residual - the object owning the mass matrix
   /// \param[in] options - options specific to the mass matrix (if needed)
   /// \return pointer to the mass matrix
   /// \note if a concrete residual type does not define a getMassMatrix
   /// function a `nullptr` will be returned.
   /// \note pointer owned by the residual.
   friend mfem::Operator *getMassMatrix(MISOResidual &residual,
                                        const nlohmann::json &options);

   /// Return a preconditioner for the residual's state Jacobian
   /// \param[inout] residual - the object owning the preconditioner
   /// \return pointer to preconditioner for the state Jacobian
   /// \note if a concrete residual type does not define a getPreconditioner
   /// function a `nullptr` will be returned.
   /// \note pointer owned by the residual.
   friend mfem::Solver *getPreconditioner(MISOResidual &residual);

   /// We need to support these overrides so that the MISOResidual type can be
   /// directly set as the operator for an MFEM NonlinearSolver
   void Mult(const mfem::Vector &state, mfem::Vector &res_vec) const override
   {
      MISOInputs inputs{{"state", state}};
      self_->eval_(inputs, res_vec);
   }

   /// We need to support these overrides so that the MISOResidual type can be
   /// directly set as the operator for an MFEM NonlinearSolver
   mfem::Operator &GetGradient(const mfem::Vector &state) const override
   {
      MISOInputs inputs{{"state", state}};
      return self_->getJac_(inputs, "state");
   }

   // TODO: we will eventual want to add functions for Jacobian products

   // The following constructors, assignment operators, and destructors allow
   // the `MISOResidual` to wrap the generic type `T`.
   template <typename T>
   MISOResidual(T x) : Operator(getSize(x)), self_(new model<T>(std::move(x)))
   { }

private:
   /// Abstract base class with common functions needed by all residuals
   class concept_t
   {
   public:
      virtual ~concept_t() = default;
      virtual MPI_Comm getComm_() const = 0;
      virtual int getSize_() const = 0;
      virtual void setInputs_(const MISOInputs &inputs) = 0;
      virtual void setOptions_(const nlohmann::json &options) = 0;
      virtual void eval_(const MISOInputs &inputs, mfem::Vector &res_vec) = 0;
      virtual void linearize_(const MISOInputs &inputs) = 0;
      virtual mfem::Operator &getJac_(const MISOInputs &inputs,
                                      const std::string &wrt) = 0;
      virtual mfem::Operator &getJacT_(const MISOInputs &inputs,
                                       const std::string &wrt) = 0;
      virtual mfem::Operator &getJacBlock_(const MISOInputs &inputs,
                                           int iblock) = 0;
      virtual void setUpAdjointSystem_(mfem::Solver &adj_solver,
                                       const MISOInputs &inputs,
                                       mfem::Vector &state_bar,
                                       mfem::Vector &adjoint) = 0;
      virtual void finalizeAdjointSystem_(mfem::Solver &adj_solver,
                                          const MISOInputs &inputs,
                                          mfem::Vector &state_bar,
                                          mfem::Vector &adjoint) = 0;
      virtual double jacobianVectorProduct_(const mfem::Vector &wrt_dot,
                                            const std::string &wrt) = 0;
      virtual void jacobianVectorProduct_(const mfem::Vector &wrt_dot,
                                          const std::string &wrt,
                                          mfem::Vector &res_dot) = 0;
      virtual double vectorJacobianProduct_(const mfem::Vector &res_bar,
                                            const std::string &wrt) = 0;
      virtual void vectorJacobianProduct_(const mfem::Vector &res_bar,
                                          const std::string &wrt,
                                          mfem::Vector &wrt_bar) = 0;
      virtual double calcEntropy_(const MISOInputs &inputs) = 0;
      virtual double calcEntropyChange_(const MISOInputs &inputs) = 0;
      virtual double calcSupplyRate_(const MISOInputs &inputs) = 0;
      virtual mfem::Operator *getMass_(const nlohmann::json &options) = 0;
      virtual mfem::Solver *getPrec_() = 0;
   };

   /// Concrete (templated) class for residuals
   /// \tparam T - a residual class
   template <typename T>
   class model final : public concept_t
   {
   public:
      model(T x) : data_(std::move(x)) { }
      MPI_Comm getComm_() const override { return getMPIComm(data_); }
      int getSize_() const override { return getSize(data_); }
      void setInputs_(const MISOInputs &inputs) override
      {
         setInputs(data_, inputs);
      }
      void setOptions_(const nlohmann::json &options) override
      {
         setOptions(data_, options);
      }
      void eval_(const MISOInputs &inputs, mfem::Vector &res_vec) override
      {
         evaluate(data_, inputs, res_vec);
      }
      void linearize_(const MISOInputs &inputs) override
      {
         linearize(data_, inputs);
      }
      mfem::Operator &getJac_(const MISOInputs &inputs,
                              const std::string &wrt) override
      {
         return getJacobian(data_, inputs, wrt);
      }
      mfem::Operator &getJacT_(const MISOInputs &inputs,
                               const std::string &wrt) override
      {
         return getJacobianTranspose(data_, inputs, wrt);
      }
      mfem::Operator &getJacBlock_(const MISOInputs &inputs,
                                   int iblock) override
      {
         return getJacobianBlock(data_, inputs, iblock);
      }
      void setUpAdjointSystem_(mfem::Solver &adj_solver,
                               const MISOInputs &inputs,
                               mfem::Vector &state_bar,
                               mfem::Vector &adjoint) override
      {
         setUpAdjointSystem(data_, adj_solver, inputs, state_bar, adjoint);
      }
      void finalizeAdjointSystem_(mfem::Solver &adj_solver,
                                  const MISOInputs &inputs,
                                  mfem::Vector &state_bar,
                                  mfem::Vector &adjoint) override
      {
         finalizeAdjointSystem(data_, adj_solver, inputs, state_bar, adjoint);
      }
      double jacobianVectorProduct_(const mfem::Vector &wrt_dot,
                                    const std::string &wrt) override
      {
         return jacobianVectorProduct(data_, wrt_dot, wrt);
      }
      void jacobianVectorProduct_(const mfem::Vector &wrt_dot,
                                  const std::string &wrt,
                                  mfem::Vector &res_dot) override
      {
         jacobianVectorProduct(data_, wrt_dot, wrt, res_dot);
      }
      double vectorJacobianProduct_(const mfem::Vector &res_bar,
                                    const std::string &wrt) override
      {
         return vectorJacobianProduct(data_, res_bar, wrt);
      }
      void vectorJacobianProduct_(const mfem::Vector &res_bar,
                                  const std::string &wrt,
                                  mfem::Vector &wrt_bar) override
      {
         vectorJacobianProduct(data_, res_bar, wrt, wrt_bar);
      }
      double calcEntropy_(const MISOInputs &inputs) override
      {
         return calcEntropy(data_, inputs);
      }
      double calcEntropyChange_(const MISOInputs &inputs) override
      {
         return calcEntropyChange(data_, inputs);
      }
      double calcSupplyRate_(const MISOInputs &inputs) override
      {
         return calcSupplyRate(data_, inputs);
      }
      mfem::Operator *getMass_(const nlohmann::json &options) override
      {
         return getMassMatrix(data_, options);
      }
      mfem::Solver *getPrec_() override { return getPreconditioner(data_); }

      T data_;
   };

   /// Pointer to `model` via its abstract base class `concept_t`
   std::unique_ptr<concept_t> self_;
};

template <typename T>
inline T &getConcrete(MISOResidual &residual)
{
   auto *model = dynamic_cast<MISOResidual::model<T> *>(residual.self_.get());
   if (model == nullptr)
   {
      throw MISOException("getConcrete() called with inconsistent template!");
   }
   else
   {
      return model->data_;
   }
}

template <typename T>
inline const T &getConcrete(const MISOResidual &residual)
{
   auto *model = dynamic_cast<MISOResidual::model<T> *>(residual.self_.get());
   if (model == nullptr)
   {
      throw MISOException("getConcrete() called with inconsistent template!");
   }
   else
   {
      return model->data_;
   }
}

inline MPI_Comm getMPIComm(const MISOResidual &residual)
{
   return residual.self_->getComm_();
}

inline int getSize(const MISOResidual &residual)
{
   return residual.self_->getSize_();
}

inline void setInputs(MISOResidual &residual, const MISOInputs &inputs)
{
   // passes `inputs` on to the `setInputs` function for the concrete
   // residual type
   residual.self_->setInputs_(inputs);
}

inline void setOptions(MISOResidual &residual, const nlohmann::json &options)
{
   // passes `options` on to the `setOptions` function for the concrete
   // residual type
   residual.self_->setOptions_(options);
}

inline void evaluate(MISOResidual &residual,
                     const MISOInputs &inputs,
                     mfem::Vector &res_vec)
{
   // passes `inputs` and `res_vec` on to the `evaluate` function for the
   // concrete residual type
   residual.self_->eval_(inputs, res_vec);
}

inline void linearize(MISOResidual &residual, const MISOInputs &inputs)
{
   residual.self_->linearize_(inputs);
}

inline mfem::Operator &getJacobian(MISOResidual &residual,
                                   const MISOInputs &inputs,
                                   const std::string &wrt)
{
   // passes `inputs` and `res_vec` on to the `getJacobian` function for the
   // concrete residual type
   return residual.self_->getJac_(inputs, wrt);
}

inline mfem::Operator &getJacobianTranspose(MISOResidual &residual,
                                            const MISOInputs &inputs,
                                            const std::string &wrt)
{
   // passes `inputs` and `res_vec` on to the `getJacobianTranspose` function
   // for the concrete residual type
   return residual.self_->getJacT_(inputs, wrt);
}

inline mfem::Operator &getJacobianBlock(MISOResidual &residual,
                                        const MISOInputs &inputs,
                                        int iblock)
{
   // passes `inputs` and `res_vec` on to the `getJacobianBlock` function for
   // the concrete residual type
   return residual.self_->getJacBlock_(inputs, iblock);
}

inline void setUpAdjointSystem(MISOResidual &residual,
                               mfem::Solver &adj_solver,
                               const MISOInputs &inputs,
                               mfem::Vector &state_bar,
                               mfem::Vector &adjoint)
{
   residual.self_->setUpAdjointSystem_(adj_solver, inputs, state_bar, adjoint);
}

inline void finalizeAdjointSystem(MISOResidual &residual,
                                  mfem::Solver &adj_solver,
                                  const MISOInputs &inputs,
                                  mfem::Vector &state_bar,
                                  mfem::Vector &adjoint)
{
   residual.self_->finalizeAdjointSystem_(adj_solver, inputs, state_bar,
                                          adjoint);
}

inline double jacobianVectorProduct(MISOResidual &residual,
                                    const mfem::Vector &wrt_dot,
                                    const std::string &wrt)
{
   return residual.self_->jacobianVectorProduct_(wrt_dot, wrt);
}

inline void jacobianVectorProduct(MISOResidual &residual,
                                  const mfem::Vector &wrt_dot,
                                  const std::string &wrt,
                                  mfem::Vector &res_dot)
{
   residual.self_->jacobianVectorProduct_(wrt_dot, wrt, res_dot);
}

inline double vectorJacobianProduct(MISOResidual &residual,
                                    const mfem::Vector &res_bar,
                                    const std::string &wrt)
{
   return residual.self_->vectorJacobianProduct_(res_bar, wrt);
}

inline void vectorJacobianProduct(MISOResidual &residual,
                                  const mfem::Vector &res_bar,
                                  const std::string &wrt,
                                  mfem::Vector &wrt_bar)
{
   residual.self_->vectorJacobianProduct_(res_bar, wrt, wrt_bar);
}

inline double calcEntropy(MISOResidual &residual, const MISOInputs &inputs)
{
   return residual.self_->calcEntropy_(inputs);
}

inline double calcEntropyChange(MISOResidual &residual,
                                const MISOInputs &inputs)
{
   return residual.self_->calcEntropyChange_(inputs);
}

inline double calcSupplyRate(MISOResidual &residual,
                             const MISOInputs &inputs)
{
   return residual.self_->calcSupplyRate_(inputs);
}

inline mfem::Operator *getMassMatrix(MISOResidual &residual,
                                     const nlohmann::json &options)
{
   return residual.self_->getMass_(options);
}

inline mfem::Solver *getPreconditioner(MISOResidual &residual)
{
   return residual.self_->getPrec_();
}

}  // namespace miso

#endif  // MISO_RESIDUAL
