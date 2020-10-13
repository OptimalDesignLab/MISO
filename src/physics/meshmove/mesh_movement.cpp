#include <fstream>

#include "coefficient.hpp"
#include "mesh_movement.hpp"

using namespace mfem;

namespace mach
{

LEAnalogySolver::LEAnalogySolver(
   const nlohmann::json &options,
   std::unique_ptr<mfem::Mesh> smesh,
   MPI_Comm comm)
   : MeshMovementSolver(options, move(smesh), comm)
{
   if (options["space-dis"]["degree"] != mesh->GetNodes()->FESpace()->GetOrder(0))
   {
      throw MachException("Linear Elasticity mesh movement solver must use same "
                          "degree as mesh order!\n");
   }
   if (options["space-dis"]["basis-type"] != "H1")
   {
      throw MachException("Linear Elasticity mesh movement solver must use H1 "
                          "basis functions\n");
   }

   dim = mesh->SpaceDimension();
}

void LEAnalogySolver::constructCoefficients()
{
   /// assign stiffness
   if (options["uniform-stiff"]["on"].template get<bool>())
   {
      double lambda = options["uniform-stiff"]["lambda"].template get<double>();
      double mu = options["uniform-stiff"]["mu"].template get<double>();
      lambda_c.reset(new ConstantCoefficient(lambda));
      mu_c.reset(new ConstantCoefficient(mu));
   }
   else
   {
      lambda_c.reset(new LameFirstParameter());
      mu_c.reset(new LameSecondParameter());
   } 
}

void LEAnalogySolver::constructForms()
{
   mass.reset(new BilinearFormType(fes.get()));
   res.reset(new NonlinearFormType(fes.get()));
   // stiff.reset(new BilinearFormType(fes.get()));
   load.reset(new LinearFormType(fes.get()));
   *load = 0.0;
}

void LEAnalogySolver::addMassIntegrators(double alpha)
{
   mass->AddDomainIntegrator(new VectorMassIntegrator());
}

// void LEAnalogySolver::addStiffVolumeIntegrators(double alpha)
// {
//    stiff->AddDomainIntegrator(new ElasticityIntegrator(*lambda_c, *mu_c));
// }

void LEAnalogySolver::addResVolumeIntegrators(double alpha)
{
   res->AddDomainIntegrator(new ElasticityIntegrator(*lambda_c, *mu_c));
}

void LEAnalogySolver::setEssentialBoundaries()
{
   ess_bdr.SetSize(mesh->bdr_attributes.Max());
   ess_bdr = 1;
}

} //namespace mach
