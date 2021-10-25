#ifndef MACH_FINITE_ELEMENT_VECTOR
#define MACH_FINITE_ELEMENT_VECTOR

#include <memory>
#include <string>

#include "mfem.hpp"

#include "utils.hpp"

namespace mach
{
/**
   @brief Class for encapsulating the data associated with a vector derived
   from a MFEM finite element space. Specifically, it contains the information
   needed for both primal finite element state fields and dual finite element
   vectors.

   Namely: the Mesh, FiniteElementCollection, GridFunction, and a distributed
   vector of the solution
*/
class FiniteElementVector
{
public:
   /// \brief Structure for optionally configuring a FiniteElementVector
   struct Options
   {
      /// \brief The polynomial order that should be used for the problem
      int order = 1;

      /// \brief The number of state variables of the finite element collections
      int num_states = 1;

      /// \brief The FECollection to use - defaults to an H1_FECollection
      std::unique_ptr<mfem::FiniteElementCollection> coll = {};

      /// \brief The DOF ordering that should be used interally by MFEM
      mfem::Ordering::Type ordering = mfem::Ordering::byVDIM;

      /// \brief The name of the field encapsulated by the state object
      std::string name = "";
   };

   /**
    * @brief Main constructor for building a new finite element vector
    * @param[in] mesh The problem mesh (object does not take ownership)
    * @param[in] options The options specified, namely those relating to the
    * order of the problem, the dimension of the FESpace, the type of FEColl,
    * the DOF ordering that should be used, and the name of the field
    */
   FiniteElementVector(mfem::ParMesh &mesh,
                       Options &&options = {.order = 1,
                                            .num_states = 1,
                                            .coll = {},
                                            .ordering = mfem::Ordering::byVDIM,
                                            .name = ""});

   /**
    * @brief Minimal constructor for a FiniteElementVector given a finite
    * element space
    * @param[in] mesh The problem mesh (object does not take ownership)
    * @param[in] space The space to use for the finite element state. This space
    * is deep copied into the new FE state
    * @param[in] name The name of the field
    */
   FiniteElementVector(mfem::ParMesh &mesh,
                       mfem::ParFiniteElementSpace &space,
                       std::string name = "");

   /**
    * @brief Constructor for a FiniteElementVector given a finite
    * element space and collection
    * @param[in] mesh The problem mesh (object does not take ownership)
    * @param[in] coll The collection to use for the finite element state.
    * @param[in] space The space to use for the finite element state.
    * @param[in] name The name of the field
    */
   FiniteElementVector(mfem::ParMesh &mesh,
                       mfem::FiniteElementCollection *coll,
                       mfem::ParFiniteElementSpace *space,
                       std::string name = "");

   FiniteElementVector(const FiniteElementVector &other) = delete;
   FiniteElementVector &operator=(const FiniteElementVector &other) = delete;
   FiniteElementVector(FiniteElementVector &&other) noexcept;
   FiniteElementVector &operator=(FiniteElementVector &&other) noexcept;

   /**
    * @brief Returns the MPI communicator for the state
    * @return The underlying MPI communicator
    */
   MPI_Comm comm() const { return retrieve(space_).GetComm(); }

   /**
    * @brief Returns a non-owning reference to the internal mesh object
    * @return The underlying mesh
    */
   mfem::ParMesh &mesh() { return *mesh_; }

   /**
    * @brief Returns a non-owning reference to the internal FESpace
    * @return The underlying finite element space
    */
   mfem::ParFiniteElementSpace &space() { return retrieve(space_); }
   /// \overload
   const mfem::ParFiniteElementSpace &space() const { return retrieve(space_); }

   /**
    * @brief Returns a non-owning reference to the vector of true DOFs
    * @return The underlying true degree of freedom vector
    * @note This is a "true dof" vector in the standard MFEM sense. Each degree
    * of freedom is on fully independent (e.g. not constrained by non-conforming
    * meshes) and exists on exactly one MPI rank. Please see the <a
    * href="https://mfem.org/pri-dual-vec/">MFEM</a> and <a
    * href="https://libceed.readthedocs.io/en/latest/libCEEDapi/#terminology-and-notation">CEED</a>
    * documentation for more details.
    */
   mfem::HypreParVector &trueVec() { return *true_vec; }
   /// \overload
   const mfem::HypreParVector &trueVec() const { return *true_vec; }

   /**
    * @brief Returns the name of the FEState (field)
    * @return The name of the finite element vector
    */
   std::string name() const { return name_; }

   /**
    * @brief Set the internal grid function using the true DOF values
    */
   void distributeSharedDofs() { gf->SetFromTrueDofs(*true_vec); }

   /**
    * @brief Initialize the true DOF vector by extracting true DOFs from the
    * internal grid function/local into the internal true DOF vector
    */
   void initializeTrueVec() { gf->GetTrueDofs(*true_vec); }

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
   FiniteElementVector &operator=(const double value);

protected:
   /// \brief A non-owning pointer to the mesh on which the field is defined
   mfem::ParMesh *mesh_ = nullptr;

   /// \brief Finite element or SBP operators
   MaybeOwningPointer<const mfem::FiniteElementCollection> coll;

   /// \brief Discrete finite element space
   MaybeOwningPointer<mfem::ParFiniteElementSpace> space_;

   /// \brief GridFunction containing the process local degrees of freedom
   std::unique_ptr<mfem::ParGridFunction> gf;

   /// \brief The hypre vector containing the true degrees of freedom
   /// \note Each entry in this vector is owned by exactly one MPI rank
   std::unique_ptr<mfem::HypreParVector> true_vec;

   /// \brief The name of the finite element vector
   std::string name_ = "";
};

/**
 * @brief Find the average value of a finite element vector across all dofs
 *
 * @param fe_vector The state variable to compute the average of
 * @return The average value
 * @note This acts on the actual scalar degree of freedom values, not the
 * interpolated shape function values. This implies these may or may not be
 * nodal averages depending on the choice of finite element basis.
 */
double avg(const FiniteElementVector &fe_vector);

/**
 * @brief Find the max value of a finite element vector across all dofs
 *
 * @param fe_vector The state variable to compute a max of
 * @return The max value
 * @note This acts on the actual scalar degree of freedom values, not the
 * interpolated shape function values. This implies these may or may not be
 * nodal averages depending on the choice of finite element basis.
 */
double max(const FiniteElementVector &fe_vector);

/**
 * @brief Find the min value of a finite element vector across all dofs
 *
 * @param fe_vector The state variable to compute a min of
 * @return The min value
 * @note This acts on the actual scalar degree of freedom values, not the
 * interpolated shape function values. This implies these may or may not be
 * nodal averages depending on the choice of finite element basis.
 */
double min(const FiniteElementVector &fe_vector);

}  // namespace mach

#endif
