#ifndef MACH_RESIDUAL
#define MACH_RESIDUAL

#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "mach_input.hpp"
#include "utils.hpp"

namespace mach
{

template <typename T>
MPI_Comm getMPIComm(const T & /*unused*/)
{
   throw MachException(
       "getComm not specialized for concrete residual type!\n");
}

template <typename T>
void setInputs(T & /*unused*/, const MachInputs & /*unused*/)
{ }

template <typename T>
void setOptions(T & /*unused*/, const nlohmann::json & /*unused*/)
{ }

template <typename T>
double calcEntropy(T & /*unused*/, const MachInputs & /*unused*/)
{
   throw MachException(
       "calcEntropy not specialized for concrete residual type!\n");
}

template <typename T>
double calcEntropyChange(T & /*unused*/, const MachInputs & /*unused*/)
{
   throw MachException(
       "calcEntropyChange not specialized for concrete residual type!\n");
}

template <typename T>
mfem::Operator *getMassMatrix(T & /*unused*/, const nlohmann::json & /*unused*/)
{
   return nullptr;
}

template <typename T>
mfem::Solver *getPreconditioner(T & /*unused*/,
                                const nlohmann::json & /*unused*/)
{
   return nullptr;
}

template <typename T>
mfem::Operator &getJacobianBlock(T & /*unused*/,
                                 const MachInputs & /*unused*/,
                                 int)
{
   throw MachException(
       "getJacobianBlock not specialized for concrete residual type!\n");
}

/// Defines a common interface for residual functions used by mach.
/// A MachResidual can wrap any type `T` that has the interface of a residual
/// function.  For example, one instance of `T` is given by `MachNonlinearForm`,
/// so that every nonlinear form can be wrapped by a MachResidual.
/// \note We use this approach to achieve polymorphism without inheritance.
/// This is important because we need to derive from `mfem` classes frequently,
/// but at the same time we need to build on their classes' functionality.
/// Without this approach, we would need to use frequent dynamic casts because
/// we have pointers to base classes.
/// \note This approach is based on the example in Sean Parent's talk:
/// ``Inheritance is the base class of evil''
class MachResidual final : public mfem::Operator
{
public:
   /// Returns a reference to the underlying concrete type
   /// \tparam T - a residual class
   /// \note This is useful when the underlying concrete type is known at the
   /// client side, and methods specific to the concrete type need to be called.
   template <typename T>
   friend T &getConcrete(MachResidual &residual);

   template <typename T>
   friend const T &getConcrete(const MachResidual &residual);

   /// Get the MPI Communicator associated with the residual
   /// \param[in] residual - the residual whose comm is desired
   /// \returns the MPI Communicator for the residual
   /// \note Optional.  Needed for some global operations (inner products)
   friend int getMPIComm(const MachResidual &residual);

   /// Gets the number of equations/unknowns of the underlying residual type
   /// \param[inout] residual - the residual whose size is being queried
   /// \returns the number of equations/unknowns
   /// \note Needed, e.g., by the ODESystemOperator constructor (see evolver.*)
   friend int getSize(const MachResidual &residual);

   /// Set inputs in the underlying residual type
   /// \param[inout] residual - the residual being assigned the input
   /// \param[in] inputs - the inputs that are being assigned
   /// \note Ends up calling `setInputs` on either the `MachNonlinearForm` or
   /// a specialized version for each particular residual.
   friend void setInputs(MachResidual &residual, const MachInputs &inputs);

   /// Set options in the underlying residual type
   /// \param[inout] residual - the residual whose options are being set
   /// \param[in] options - the options that are being assigned
   friend void setOptions(MachResidual &residual,
                          const nlohmann::json &options);

   /// Evaluate the residual function at given inputs and return as `res_vec`
   /// \param[inout] residual - the residual being evaluated
   /// \param[in] inputs - the independent variables at which to evaluate `res`
   /// \param[out] res_vec - the dependent variable, the output from `residual`
   friend void evaluate(MachResidual &residual,
                        const MachInputs &inputs,
                        mfem::Vector &res_vec);

   /// Compute the Jacobian of the given residual and return a reference to it
   /// \param[inout] residual - function whose Jacobian we want
   /// \param[in] inputs - the variables needed to evaluate the Jacobian
   /// \param[in] wrt - the input we are differentiating with respect to
   /// \returns a reference to the residuals Jacobian with respect to `wrt`
   /// \note the underlying `Operator` is owned by `residual`
   friend mfem::Operator &getJacobian(MachResidual &residual,
                                      const MachInputs &inputs,
                                      const std::string &wrt);

   /// Get a reference to `iblock`th block of the Jacobian for a block system
   /// \param[inout] residual - function whose Jacobian we want
   /// \param[in] inputs - the variables needed to evaluate the Jacobian
   /// \param[in] iblock - the block whose Jacobian is sought
   /// \note this is for the state Jacobian only
   friend mfem::Operator &getJacobianBlock(MachResidual &residual, 
                                           const MachInputs &inputs,
                                           int iblock);

   /// Evaluate the entropy functional at the given state
   /// \param[inout] residual - function with an associated entropy
   /// \param[in] inputs - the variables needed to evaluate the entropy
   /// \return the entropy functional
   /// \note optional, but must be implemented for relaxation RK
   friend double calcEntropy(MachResidual &residual, const MachInputs &inputs);

   /// Evaluate the residual weighted by the entropy variables
   /// \param[inout] residual - function with an associated entropy
   /// \param[in] inputs - the variables needed to evaluate the entropy
   /// \return the product `w^T res`
   /// \note `w` and `res` are evaluated at `state + dt*state_dot` and time
   /// `t+dt` \note optional, but must be implemented for relaxation RK
   friend double calcEntropyChange(MachResidual &residual,
                                   const MachInputs &inputs);

   /// Return the mass matrix corresponding to the residual
   /// \param[inout] residual - the object owning the mass matrix
   /// \param[in] options - options specific to the mass matrix (if needed)
   /// \return pointer to the mass matrix
   /// \note if a concrete residual type does not define a getMassMatrix
   /// function a `nullptr` will be returned.
   /// \note pointer owned by the residual.
   friend mfem::Operator *getMassMatrix(MachResidual &residual,
                                        const nlohmann::json &options);

   /// Return a preconditioner for the residual's state Jacobian
   /// \param[inout] residual - the object owning the preconditioner
   /// \param[in] options - options specific to the preconditioner (if needed)
   /// \return pointer to preconditioner for the state Jacobian
   /// \note if a concrete residual type does not define a getPreconditioner
   /// function a `nullptr` will be returned.
   /// \note pointer owned by the residual.
   friend mfem::Solver *getPreconditioner(MachResidual &residual,
                                          const nlohmann::json &options);

   /// We need to support these overrides so that the MachResidual type can be
   /// directly set as the operator for an MFEM NonlinearSolver
   void Mult(const mfem::Vector &state, mfem::Vector &res_vec) const override
   {
      MachInputs inputs{{"state", state}};
      self_->eval_(inputs, res_vec);
   }

   /// We need to support these overrides so that the MachResidual type can be
   /// directly set as the operator for an MFEM NonlinearSolver
   mfem::Operator &GetGradient(const mfem::Vector &state) const override
   {
      MachInputs inputs{{"state", state}};
      return self_->getJac_(inputs, "state");
   }

   // TODO: we will eventual want to add functions for Jacobian products

   // The following constructors, assignment operators, and destructors allow
   // the `MachResidual` to wrap the generic type `T`.
   template <typename T>
   MachResidual(T x) : Operator(getSize(x)), self_(new model<T>(std::move(x)))
   { }

private:
   /// Abstract base class with common functions needed by all residuals
   class concept_t
   {
   public:
      virtual ~concept_t() = default;
      virtual MPI_Comm getComm_() const = 0;
      virtual int getSize_() const = 0;
      virtual void setInputs_(const MachInputs &inputs) = 0;
      virtual void setOptions_(const nlohmann::json &options) = 0;
      virtual void eval_(const MachInputs &inputs, mfem::Vector &res_vec) = 0;
      virtual mfem::Operator &getJac_(const MachInputs &inputs,
                                      const std::string &wrt) = 0;
      virtual mfem::Operator &getJacBlock_(const MachInputs &inputs,
                                           int iblock) = 0;
      virtual double calcEntropy_(const MachInputs &inputs) = 0;
      virtual double calcEntropyChange_(const MachInputs &inputs) = 0;
      virtual mfem::Operator *getMass_(const nlohmann::json &options) = 0;
      virtual mfem::Solver *getPrec_(const nlohmann::json &options) = 0;
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
      void setInputs_(const MachInputs &inputs) override
      {
         setInputs(data_, inputs);
      }
      void setOptions_(const nlohmann::json &options) override
      {
         setOptions(data_, options);
      }
      void eval_(const MachInputs &inputs, mfem::Vector &res_vec) override
      {
         evaluate(data_, inputs, res_vec);
      }
      mfem::Operator &getJac_(const MachInputs &inputs,
                              const std::string &wrt) override
      {
         return getJacobian(data_, inputs, wrt);
      }
      mfem::Operator &getJacBlock_(const MachInputs &inputs,
                                   int iblock) override 
      {
         return getJacobianBlock(data_, inputs, iblock);
      }
      double calcEntropy_(const MachInputs &inputs) override
      {
         return calcEntropy(data_, inputs);
      }
      double calcEntropyChange_(const MachInputs &inputs) override
      {
         return calcEntropyChange(data_, inputs);
      }
      mfem::Operator *getMass_(const nlohmann::json &options) override 
      {
         return getMassMatrix(data_, options);
      }
      mfem::Solver *getPrec_(const nlohmann::json &options) override
      { 
         return getPreconditioner(data_, options);
      }

      T data_;
   };

   /// Pointer to `model` via its abstract base class `concept_t`
   std::unique_ptr<concept_t> self_;
};

template <typename T>
inline T &getConcrete(MachResidual &residual)
{
   auto *model = dynamic_cast<MachResidual::model<T> *>(residual.self_.get());
   if (model == nullptr)
   {
      throw MachException("getConcrete() called with inconsistent template!");
   }
   else
   {
      return model->data_;
   }
}

template <typename T>
inline const T &getConcrete(const MachResidual &residual)
{
   auto *model = dynamic_cast<MachResidual::model<T> *>(residual.self_.get());
   if (model == nullptr)
   {
      throw MachException("getConcrete() called with inconsistent template!");
   }
   else
   {
      return model->data_;
   }
}

inline MPI_Comm getMPIComm(const MachResidual &residual)
{
   return residual.self_->getComm_();
}

inline int getSize(const MachResidual &residual)
{
   return residual.self_->getSize_();
}

inline void setInputs(MachResidual &residual, const MachInputs &inputs)
{
   // passes `inputs` on to the `setInputs` function for the concrete
   // residual type
   residual.self_->setInputs_(inputs);
}

inline void setOptions(MachResidual &residual, const nlohmann::json &options)
{
   // passes `options` on to the `setOptions` function for the concrete
   // residual type
   residual.self_->setOptions_(options);
}

inline void evaluate(MachResidual &residual,
                     const MachInputs &inputs,
                     mfem::Vector &res_vec)
{
   // passes `inputs` and `res_vec` on to the `evaluate` function for the
   // concrete residual type
   residual.self_->eval_(inputs, res_vec);
}

inline mfem::Operator &getJacobian(MachResidual &residual,
                                   const MachInputs &inputs,
                                   const std::string &wrt)
{
   // passes `inputs` and `res_vec` on to the `getJacobian` function for the
   // concrete residual type
   return residual.self_->getJac_(inputs, wrt);
}

inline mfem::Operator &getJacobianBlock(MachResidual &residual,
                                        const MachInputs &inputs,
                                        int iblock)
{
   // passes `inputs` and `res_vec` on to the `getJacobianBlock` function for 
   // the concrete residual type
   return residual.self_->getJacBlock_(inputs, iblock);
}

inline double calcEntropy(MachResidual &residual, const MachInputs &inputs)
{
   return residual.self_->calcEntropy_(inputs);
}

inline double calcEntropyChange(MachResidual &residual,
                                const MachInputs &inputs)
{
   return residual.self_->calcEntropyChange_(inputs);
}

inline mfem::Operator *getMassMatrix(MachResidual &residual,
                                     const nlohmann::json &options)
{
   return residual.self_->getMass_(options);
}

inline mfem::Solver *getPreconditioner(MachResidual &residual,
                                       const nlohmann::json &options)
{
   return residual.self_->getPrec_(options);
}

}  // namespace mach

#endif  // MACH_RESIDUAL
