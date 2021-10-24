#include "sbp_fe.hpp"

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
 : mesh_(mesh),
   coll_(options.coll
             ? std::move(options.coll)
             : std::make_unique<mfem::H1_FECollection>(options.order,
                                                       mesh.Dimension())),
   space_(std::make_unique<mfem::ParFiniteElementSpace>(&mesh,
                                                        coll_.get(),
                                                        options.num_states,
                                                        options.ordering)),
   gf_(std::make_unique<mfem::ParGridFunction>(space_.get())),
   true_vec_(space_.get()),
   name_(options.name)
{
   true_vec_ = 0.0;
}

FiniteElementVector::FiniteElementVector(mfem::ParMesh &mesh,
                                         mfem::ParFiniteElementSpace &space,
                                         std::string name)
 : mesh_(mesh),
   coll_(copyFEColl(*space.FEColl())),
   space_(std::make_unique<mfem::ParFiniteElementSpace>(space,
                                                        &mesh,
                                                        coll_.get())),
   gf_(std::make_unique<mfem::ParGridFunction>(space_.get())),
   true_vec_(space_.get()),
   name_(std::move(name))
{
   true_vec_ = 0.0;
}

FiniteElementVector::FiniteElementVector(const FiniteElementVector &other)
 : mesh_(other.mesh_),
   coll_(copyFEColl(*other.space_->FEColl())),
   space_(std::make_unique<mfem::ParFiniteElementSpace>(*other.space_,
                                                        &mesh_.get(),
                                                        coll_.get())),
   gf_(std::make_unique<mfem::ParGridFunction>(*other.gf_)),
   true_vec_(space_.get()),
   name_(other.name_)
{
   true_vec_ = other.true_vec_;
}

FiniteElementVector &FiniteElementVector::operator=(
    const FiniteElementVector &other)
{
   mesh_ = other.mesh_;
   coll_ = copyFEColl(*other.space_->FEColl()),
   space_ = std::make_unique<mfem::ParFiniteElementSpace>(
       *other.space_, &mesh_.get(), coll_.get());
   gf_ = std::make_unique<mfem::ParGridFunction>(*other.gf_);
   true_vec_ = mfem::HypreParVector(space_.get());
   true_vec_ = other.true_vec_;
   name_ = other.name_;
   return *this;
}

FiniteElementVector::FiniteElementVector(FiniteElementVector &&other)
 : mesh_(std::move(other.mesh_)),
   coll_(std::move(other.coll_)),
   gf_(std::move(other.gf_)),
   name_(std::move(other.name_))
{
   auto *par_vec = other.true_vec_.StealParVector();
   true_vec_.WrapHypreParVector(par_vec);
}

FiniteElementVector &FiniteElementVector::operator=(FiniteElementVector &&other)
{
   mesh_ = std::move(other.mesh_);
   coll_ = std::move(other.coll_);
   gf_ = std::move(other.gf_);
   auto *par_vec = other.true_vec_.StealParVector();
   true_vec_.WrapHypreParVector(par_vec);
   name_ = std::move(other.name_);
   return *this;
}

FiniteElementVector &FiniteElementVector::operator=(const double value)
{
   true_vec_ = value;
   distributeSharedDofs();
   return *this;
}

double avg(const FiniteElementVector &fe_vector)
{
   double global_sum;
   double local_sum = fe_vector.trueVec().Sum();
   int global_size;
   int local_size = fe_vector.trueVec().Size();
   MPI_Allreduce(
       &local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, fe_vector.comm());
   MPI_Allreduce(
       &local_size, &global_size, 1, MPI_INT, MPI_SUM, fe_vector.comm());
   return global_sum / global_size;
}

double max(const FiniteElementVector &fe_vector)
{
   double global_max;
   double local_max = fe_vector.trueVec().Max();
   MPI_Allreduce(
       &local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, fe_vector.comm());
   return global_max;
}

double min(const FiniteElementVector &fe_vector)
{
   double global_min;
   double local_min = fe_vector.trueVec().Min();
   MPI_Allreduce(
       &local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, fe_vector.comm());
   return global_min;
}

}  // namespace mach
