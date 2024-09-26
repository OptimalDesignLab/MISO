#include <fstream>
#include <algorithm>
#include <memory>

#include "solver.hpp"
#include "coefficient.hpp"
#include "mesh_move_integ.hpp"
#include "utils.hpp"
#include "mesh_movement.hpp"

using namespace mfem;

namespace miso
{
LEAnalogySolver::LEAnalogySolver(const nlohmann::json &options,
                                 std::unique_ptr<Mesh> smesh,
                                 MPI_Comm comm)
 : MeshMovementSolver(options, move(smesh), comm)
{
   if (options["space-dis"]["degree"] !=
       mesh->GetNodes()->FESpace()->GetOrder(0))
   {
      throw MISOException(
          "Linear Elasticity mesh movement solver must use same "
          "degree as mesh order!\n");
   }
   if (options["space-dis"]["basis-type"] != "H1")
   {
      throw MISOException(
          "Linear Elasticity mesh movement solver must use H1 "
          "basis functions\n");
   }

   dim = mesh->SpaceDimension();
}

void LEAnalogySolver::setInitialCondition(
    ParGridFunction &state,
    const std::function<void(const Vector &, Vector &)> &u_init)
{
   // AbstractSolver::setInitialCondition(state, u_init);
   state = 0.0;

   VectorFunctionCoefficient u0(dim, u_init);
   state.ProjectBdrCoefficient(u0, ess_bdr);
   printField("uinit", state, "solution");
}

double LEAnalogySolver::calcStepSize(int iter,
                                     double t,
                                     double t_final,
                                     double dt_old,
                                     const ParGridFunction &state) const
{
   if (options["time-dis"]["steady"].get<bool>())
   {
      // ramp up time step for pseudo-transient continuation
      // TODO: the l2 norm of the weak residual is probably not ideal here
      // A better choice might be the l1 norm
      double res_norm = calcResidualNorm(state);
      if (std::abs(res_norm) <= 1e-14)
      {
         return 1e14;
      }
      double exponent = options["time-dis"]["res-exp"];
      double dt = options["time-dis"]["dt"].get<double>() *
                  pow(res_norm0 / res_norm, exponent);
      return std::max(dt, dt_old);
   }
   else
   {
      throw MISOException("LEAnalogySolver requires steady time-dis!\n");
   }
}

void LEAnalogySolver::initialHook(const ParGridFunction &state)
{
   if (options["time-dis"]["steady"].get<bool>())
   {
      // res_norm0 is used to compute the time step in PTC
      res_norm0 = calcResidualNorm(state);
   }
   else
   {
      throw MISOException("LEAnalogySolver requires steady time-dis!\n");
   }
}

bool LEAnalogySolver::iterationExit(int iter,
                                    double t,
                                    double t_final,
                                    double dt,
                                    const ParGridFunction &state) const
{
   if (options["time-dis"]["steady"].get<bool>())
   {
      // use tolerance options for Newton's method
      double norm = calcResidualNorm(state);
      return norm <= options["time-dis"]["steady-abstol"].get<double>() ||
             norm <=
                 res_norm0 * options["time-dis"]["steady-reltol"].get<double>();
   }
   else
   {
      throw MISOException("LEAnalogySolver requires steady time-dis!\n");
   }
}

void LEAnalogySolver::constructCoefficients()
{
   /// assign stiffness
   if (options["problem-opts"].contains("uniform-stiff"))
   {
      auto lambda =
          options["problem-opts"]["uniform-stiff"]["lambda"].get<double>();
      auto mu = options["problem-opts"]["uniform-stiff"]["mu"].get<double>();
      lambda_c = std::make_unique<ConstantCoefficient>(lambda);
      mu_c = std::make_unique<ConstantCoefficient>(mu);
   }
   else
   {
      lambda_c = std::make_unique<LameFirstParameter>();
      mu_c = std::make_unique<LameSecondParameter>();
   }
}

void LEAnalogySolver::constructForms()
{
   mass = std::make_unique<BilinearFormType>(fes.get());
   res = std::make_unique<NonlinearFormType>(fes.get());
}

void LEAnalogySolver::addMassIntegrators(double alpha)
{
   mass->AddDomainIntegrator(new VectorMassIntegrator());
}

void LEAnalogySolver::addResVolumeIntegrators(double alpha)
{
   res->AddDomainIntegrator(new ElasticityIntegrator(*lambda_c, *mu_c));
}

void LEAnalogySolver::setEssentialBoundaries()
{
   ess_bdr.SetSize(mesh->bdr_attributes.Max());
   ess_bdr = 1;

   Array<int> ess_tdof_list;
   fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   res->SetEssentialTrueDofs(ess_tdof_list);
}

}  // namespace miso
