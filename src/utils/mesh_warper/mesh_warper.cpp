#include "mfem.hpp"
#include "mpi.h"
#include "nlohmann/json.hpp"

#include "mach_nonlinearform.hpp"
#include "mesh_move_integ.hpp"
#include "mfem_extensions.hpp"

#include "mesh_warper.hpp"

namespace
{
class MeshWarperResidual final
{
public:
   friend int getSize(const MeshWarperResidual &residual);

   friend void setInputs(MeshWarperResidual &residual,
                         const mach::MachInputs &inputs);

   friend void setOptions(MeshWarperResidual &residual,
                          const nlohmann::json &options);

   friend void evaluate(MeshWarperResidual &residual,
                        const mach::MachInputs &inputs,
                        mfem::Vector &res_vec);

   friend void linearize(MeshWarperResidual &residual,
                         const mach::MachInputs &inputs);

   friend mfem::Operator &getJacobian(MeshWarperResidual &residual,
                                      const mach::MachInputs &inputs,
                                      const std::string &wrt);

   friend mfem::Operator &getJacobianTranspose(MeshWarperResidual &residual,
                                               const mach::MachInputs &inputs,
                                               const std::string &wrt);

   friend double vectorJacobianProduct(MeshWarperResidual &residual,
                                       const mfem::Vector &res_bar,
                                       const std::string &wrt);

   friend void vectorJacobianProduct(MeshWarperResidual &residual,
                                     const mfem::Vector &res_bar,
                                     const std::string &wrt,
                                     mfem::Vector &wrt_bar);

   friend mfem::Solver *getPreconditioner(MeshWarperResidual &residual);

   MeshWarperResidual(mfem::ParFiniteElementSpace &fes,
                      std::map<std::string, mach::FiniteElementState> &fields,
                      const nlohmann::json &options)
    : res(fes, fields),
      lambda_c(std::make_unique<mfem::ConstantCoefficient>(1.0)),
      mu_c(std::make_unique<mfem::ConstantCoefficient>(1.0)),
      prec(constructPreconditioner(fes, options["lin-prec"]))

   {
      res.addDomainIntegrator(
          new mach::ElasticityPositionIntegrator(*lambda_c, *mu_c));
   }

private:
   /// Nonlinear form that solves linear elasticity problem
   mach::MachNonlinearForm res;

   /// Stiffness coefficients
   std::unique_ptr<mfem::Coefficient> lambda_c;
   std::unique_ptr<mfem::Coefficient> mu_c;

   /// preconditioner for inverting residual's state Jacobian
   std::unique_ptr<mfem::Solver> prec;

   std::unique_ptr<mfem::Solver> constructPreconditioner(
       mfem::ParFiniteElementSpace &fes,
       const nlohmann::json &prec_options)
   {
      auto amg = std::make_unique<mfem::HypreBoomerAMG>();
      amg->SetPrintLevel(prec_options["printlevel"].get<int>());
      return amg;
   }
};

int getSize(const MeshWarperResidual &residual)
{
   return getSize(residual.res);
}

void setInputs(MeshWarperResidual &residual, const mach::MachInputs &inputs)
{
   setInputs(residual.res, inputs);
}

void setOptions(MeshWarperResidual &residual, const nlohmann::json &options)
{
   setOptions(residual.res, options);
}

void evaluate(MeshWarperResidual &residual,
              const mach::MachInputs &inputs,
              mfem::Vector &res_vec)
{
   evaluate(residual.res, inputs, res_vec);
}

void linearize(MeshWarperResidual &residual, const mach::MachInputs &inputs)
{
   linearize(residual.res, inputs);
}

mfem::Operator &getJacobian(MeshWarperResidual &residual,
                            const mach::MachInputs &inputs,
                            const std::string &wrt)
{
   return getJacobian(residual.res, inputs, wrt);
}

mfem::Operator &getJacobianTranspose(MeshWarperResidual &residual,
                                     const mach::MachInputs &inputs,
                                     const std::string &wrt)
{
   return getJacobianTranspose(residual.res, inputs, wrt);
}

double vectorJacobianProduct(MeshWarperResidual &residual,
                             const mfem::Vector &res_bar,
                             const std::string &wrt)
{
   return vectorJacobianProduct(residual.res, res_bar, wrt);
}

void vectorJacobianProduct(MeshWarperResidual &residual,
                           const mfem::Vector &res_bar,
                           const std::string &wrt,
                           mfem::Vector &wrt_bar)
{
   vectorJacobianProduct(residual.res, res_bar, wrt, wrt_bar);
}

mfem::Solver *getPreconditioner(MeshWarperResidual &residual)
{
   return residual.prec.get();
}

}  // anonymous namespace

namespace mach
{
int MeshWarper::getSurfaceCoordsSize() const { return surf_coords.Size(); }

void MeshWarper::getInitialSurfaceCoords(mfem::Vector &surface_coords) const
{
   surface_coords = surf_coords;
}

int MeshWarper::getVolumeCoordsSize() const { return vol_coords.Size(); }

void MeshWarper::getInitialVolumeCoords(mfem::Vector &volume_coords) const
{
   volume_coords = vol_coords;
}

void MeshWarper::getSurfCoordIndices(mfem::Array<int> &indices) const
{
   indices = surface_indices;
}

MeshWarper::MeshWarper(MPI_Comm incomm,
                       const nlohmann::json &solver_options,
                       std::unique_ptr<mfem::Mesh> smesh)
 : AbstractSolver2(incomm, solver_options),
   mesh_(constructMesh(comm, options["mesh"], std::move(smesh), true))
{
   auto num_states = mesh().SpaceDimension();

   FiniteElementState state(mesh(), options["space-dis"], num_states, "state");
   fields.emplace("state", std::move(state));

   FiniteElementState adjoint(
       mesh(), options["space-dis"], num_states, "adjoint");
   fields.emplace("adjoint", std::move(adjoint));

   FiniteElementDual residual(
       mesh(), options["space-dis"], num_states, "residual");
   duals.emplace("residual", std::move(residual));

   auto &mesh_gf = *dynamic_cast<mfem::ParGridFunction *>(mesh().GetNodes());
   auto *mesh_fespace = mesh_gf.ParFESpace();

   /// create new state vector copying the mesh's fe space
   fields.emplace(std::piecewise_construct,
                  std::forward_as_tuple("mesh_coords"),
                  std::forward_as_tuple(mesh(), *mesh_fespace, "mesh_coords"));
   FiniteElementState &mesh_coords = fields.at("mesh_coords");
   /// set the values of the new GF to those of the mesh's old nodes
   mesh_coords.gridFunc() = mesh_gf;
   // mesh_coords.setTrueVec();  // distribute coords
   /// tell the mesh to use this GF for its Nodes
   /// (and that it doesn't own it)
   mesh().NewNodes(mesh_coords.gridFunc(), false);

   options["time-dis"]["type"] = "steady";
   spatial_res = std::make_unique<MachResidual>(
       MeshWarperResidual(fes(), fields, options));
   setOptions(*spatial_res, options);

   auto *prec = getPreconditioner(*spatial_res);
   auto lin_solver_opts = options["lin-solver"];
   linear_solver = mach::constructLinearSolver(comm, lin_solver_opts, prec);
   auto nonlin_solver_opts = options["nonlin-solver"];
   nonlinear_solver =
       mach::constructNonlinearSolver(comm, nonlin_solver_opts, *linear_solver);
   nonlinear_solver->SetOperator(*spatial_res);

   /// Set initial volume coords true vec
   fields.at("mesh_coords").setTrueVec(vol_coords);

   /// Get the indices of the surface mesh dofs into the volume mesh
   mfem::Array<int> ess_bdr(mesh().bdr_attributes.Max());
   ess_bdr = 1;
   fes().GetEssentialTrueDofs(ess_bdr, surface_indices);

   /// Set the initial surface coords
   vol_coords.GetSubVector(surface_indices, surf_coords);
}

}  // namespace mach
