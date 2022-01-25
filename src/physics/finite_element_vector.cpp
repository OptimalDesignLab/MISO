#include <cmath>

#include "sbp_fe.hpp"
#include "utils.hpp"

#include "finite_element_vector.hpp"

namespace
{
std::unique_ptr<const mfem::FiniteElementCollection> copyFEColl(
    const mfem::FiniteElementCollection &coll)
{
   const auto *name = coll.Name();

   auto const *csbp = dynamic_cast<mfem::SBPCollection const *>(&coll);
   if (csbp != nullptr)
   {
      int p = atoi(name + 8);
      int dim = atoi(name + 4);
      return std::make_unique<mfem::SBPCollection>(p, dim);
   }
   auto const *dsbp = dynamic_cast<mfem::DSBPCollection const *>(&coll);
   if (dsbp != nullptr)
   {
      int p = atoi(name + 9);
      int dim = atoi(name + 5);
      return std::make_unique<mfem::DSBPCollection>(p, dim);
   }
   return std::unique_ptr<mfem::FiniteElementCollection>(
       mfem::FiniteElementCollection::New(name));
}

}  // namespace

namespace mach
{
FiniteElementVector::FiniteElementVector(mfem::ParMesh &mesh,
                                         FiniteElementVector::Options &&options)
 : mesh_(&mesh),
   coll_(options.coll
             ? std::move(options.coll)
             : std::make_unique<mfem::H1_FECollection>(options.order,
                                                       mesh.Dimension())),
   space_(std::make_unique<mfem::ParFiniteElementSpace>(&mesh,
                                                        &retrieve(coll_),
                                                        options.num_states,
                                                        options.ordering)),
   gf(std::make_unique<mfem::ParGridFunction>(&retrieve(space_))),
   // true_vec(std::make_unique<mfem::HypreParVector>(&retrieve(space_))),
   name_(options.name)
{
   // *true_vec = 0.0;
}

FiniteElementVector::FiniteElementVector(mfem::ParMesh &mesh,
                                         const nlohmann::json &space_options,
                                         const int num_states,
                                         std::string name)
 : FiniteElementVector(
       mesh,
       [&]() -> FiniteElementVector::Options
       {
          const int dim = mesh.Dimension();
          const auto order = space_options["degree"].get<int>();
          const auto basis_type =
              space_options["basis-type"].get<std::string>();
          const bool galerkin_diff = space_options.value("GD", false);
          // Define the SBP elements and finite-element space; eventually, we
          // will want to have a case or if statement here for both CSBP and
          // DSBP, and (?) standard FEM. and here it is for first two
          std::unique_ptr<mfem::FiniteElementCollection> fec;
          if (basis_type == "CSBP" || basis_type == "csbp")
          {
             fec = std::make_unique<mfem::SBPCollection>(order, dim);
          }
          else if (basis_type == "DSBP" || basis_type == "dsbp" ||
                   galerkin_diff)
          {
             fec = std::make_unique<mfem::DSBPCollection>(order, dim);
          }
          else if (basis_type == "ND" || basis_type == "nd" ||
                   basis_type == "nedelec")
          {
             fec = std::make_unique<mfem::ND_FECollection>(order, dim);
          }
          else if (basis_type == "RT" || basis_type == "rt")
          {
             fec = std::make_unique<mfem::RT_FECollection>(order, dim);
          }
          else if (basis_type == "H1" || basis_type == "h1" ||
                   basis_type == "CG" || basis_type == "cg")
          {
             fec = std::make_unique<mfem::H1_FECollection>(order, dim);
          }
          else if (basis_type == "L2" || basis_type == "l2" ||
                   basis_type == "DG" || basis_type == "dg")
          {
             fec = std::make_unique<mfem::L2_FECollection>(order, dim);
          }

          return {.order = order,
                  .num_states = num_states,
                  .coll = std::move(fec),
                  .ordering = mfem::Ordering::byVDIM,
                  .name = std::move(name)};
       }())
{ }

FiniteElementVector::FiniteElementVector(mfem::ParMesh &mesh,
                                         mfem::ParFiniteElementSpace &space,
                                         std::string name)
 : mesh_(&mesh),
   coll_(copyFEColl(*space.FEColl())),
   space_(std::make_unique<mfem::ParFiniteElementSpace>(space,
                                                        mesh_,
                                                        &retrieve(coll_))),
   gf(std::make_unique<mfem::ParGridFunction>(&retrieve(space_))),
   // true_vec(std::make_unique<mfem::HypreParVector>(&retrieve(space_))),
   name_(std::move(name))
{
   // *true_vec = 0.0;
}

// FiniteElementVector::FiniteElementVector(const FiniteElementVector &other)
//  : mesh_(other.mesh_),
//    coll_(copyFEColl(*other.space_->FEColl())),
//    space_(std::make_unique<mfem::ParFiniteElementSpace>(*other.space_,
//                                                         &mesh_.get(),
//                                                         coll_.get())),
//    gf_(std::make_unique<mfem::ParGridFunction>(*other.gf_)),
//    true_vec_(space_.get()),
//    name_(other.name_)
// {
//    true_vec_ = other.true_vec_;
// }

// FiniteElementVector &FiniteElementVector::operator=(
//     const FiniteElementVector &other)
// {
//    mesh_ = other.mesh_;
//    coll_ = copyFEColl(*other.space_->FEColl()),
//    space_ = std::make_unique<mfem::ParFiniteElementSpace>(
//        *other.space_, &mesh_.get(), coll_.get());
//    gf_ = std::make_unique<mfem::ParGridFunction>(*other.gf_);
//    true_vec_ = mfem::HypreParVector(space_.get());
//    true_vec_ = other.true_vec_;
//    name_ = other.name_;
//    return *this;
// }

FiniteElementVector::FiniteElementVector(FiniteElementVector &&other) noexcept
 : mesh_(other.mesh_),
   coll_(std::move(other.coll_)),
   space_(std::move(other.space_)),
   gf(std::move(other.gf)),
   // true_vec(std::move(other.true_vec)),
   name_(std::move(other.name_))
{
   // auto *par_vec = other.true_vec.StealParVector();
   // true_vec.WrapHypreParVector(par_vec);
}

FiniteElementVector &FiniteElementVector::operator=(
    FiniteElementVector &&other) noexcept
{
   mesh_ = other.mesh_;
   coll_ = std::move(other.coll_);
   space_ = std::move(other.space_);
   gf = std::move(other.gf);
   // auto *par_vec = other.true_vec.StealParVector();
   // true_vec.WrapHypreParVector(par_vec);
   // true_vec = std::move(other.true_vec);
   name_ = std::move(other.name_);
   return *this;
}

// FiniteElementVector &FiniteElementVector::operator=(const double value)
// {
//    // *true_vec = value;
//    distributeSharedDofs();
//    return *this;
// }

// double avg(const FiniteElementVector &fe_vector)
// {
//    double global_sum = NAN;
//    double local_sum = fe_vector.trueVec().Sum();
//    int global_size = 0;
//    int local_size = fe_vector.trueVec().Size();
//    MPI_Allreduce(
//        &local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, fe_vector.comm());
//    MPI_Allreduce(
//        &local_size, &global_size, 1, MPI_INT, MPI_SUM, fe_vector.comm());
//    return global_sum / global_size;
// }

// double max(const FiniteElementVector &fe_vector)
// {
//    double global_max = NAN;
//    double local_max = fe_vector.trueVec().Max();
//    MPI_Allreduce(
//        &local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, fe_vector.comm());
//    return global_max;
// }

// double min(const FiniteElementVector &fe_vector)
// {
//    double global_min = NAN;
//    double local_min = fe_vector.trueVec().Min();
//    MPI_Allreduce(
//        &local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, fe_vector.comm());
//    return global_min;
// }

}  // namespace mach
