#include <memory>
#include <random>

#include "catch.hpp"
#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "abstract_solver.hpp"
#include "electromag_integ.hpp"
#include "coefficient.hpp"
#include "current_load.hpp"
#include "mach_input.hpp"
#include "mach_linearform.hpp"
#include "mach_load.hpp"
#include "mach_nonlinearform.hpp"
#include "mach_residual.hpp"
#include "mfem_extensions.hpp"
#include "pde_solver.hpp"
#include "utils.hpp"

/// Class for PDE that follows the MachResidual API
class MagnetostaticResidual final
{
public:
   MagnetostaticResidual(mfem::ParFiniteElementSpace &fes,
                         const nlohmann::json &options,
                         mfem::VectorCoefficient &current_coeff,
                         mach::StateCoefficient &nu)
    : fes_(fes),
      fields(std::make_unique<std::unordered_map<std::string, mfem::ParGridFunction>>()),
      res(fes_, *fields),
      load(std::make_unique<mach::CurrentLoad>(fes_, options, current_coeff)),
      prec(constructPreconditioner(fes_.GetComm(), options["lin-prec"]))
   {
      res.addDomainIntegrator(new mach::CurlCurlNLFIntegrator(nu));
   }

   friend int getSize(const MagnetostaticResidual &residual)
   {
      return getSize(residual.res);
   }

   friend void setInputs(MagnetostaticResidual &residual, const mach::MachInputs &inputs)
   {
      setInputs(residual.res, inputs);
      setInputs(*residual.load, inputs);
   }

   friend void setOptions(MagnetostaticResidual &residual, const nlohmann::json &options)
   {
      setOptions(residual.res, options);
      setOptions(*residual.load, options);
   }

   friend void evaluate(MagnetostaticResidual &residual,
                        const mach::MachInputs &inputs,
                        mfem::Vector &res_vec)
   {
      evaluate(residual.res, inputs, res_vec);
      setInputs(*residual.load, inputs);
      addLoad(*residual.load, res_vec);
   }

   friend mfem::Operator &getJacobian(MagnetostaticResidual &residual,
                                      const mach::MachInputs &inputs,
                                      std::string wrt)
   {
      return getJacobian(residual.res, inputs, std::move(wrt));
   }

   friend mfem::Solver *getPreconditioner(MagnetostaticResidual &residual,
                                          const nlohmann::json &options)
   {
      return residual.prec.get();
   }

private:
   mfem::ParFiniteElementSpace &fes_;
   std::unique_ptr<std::unordered_map<std::string, mfem::ParGridFunction>> fields;
   mach::MachNonlinearForm res;
   std::unique_ptr<mach::CurrentLoad> load;

   /// preconditioner for inverting residual's state Jacobian
   std::unique_ptr<mfem::Solver> prec;

   std::unique_ptr<mfem::Solver> constructPreconditioner(
       MPI_Comm comm,
       const nlohmann::json &prec_options)
   {
      auto ams = std::make_unique<mfem::HypreAMS>(&fes_);
      ams->SetPrintLevel(prec_options["printlevel"].get<int>());
      ams->SetSingularProblem();
      return ams;
   }
};

/// permeability of free space
constexpr double mu_0 = 4e-7 * M_PI;

class BoxCoefficient : public mach::StateCoefficient
{
   double Eval(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip) override
   {
      return Eval(trans, ip, 0);
   }

   double Eval(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               double state) override
   {
      return 1.0;
   }

   double EvalStateDeriv(mfem::ElementTransformation &trans,
                         const mfem::IntegrationPoint &ip,
                         double state) override
   {
      return 0.0;
   }
};

/// Solver that uses `MagnetostaticResidual` to define its dynamics
class MagnetostaticSolver : public mach::PDESolver
{
public:
   MagnetostaticSolver(MPI_Comm comm,
                       const nlohmann::json &solver_options,
                       std::unique_ptr<mfem::Mesh> smesh)
      : PDESolver(comm, solver_options, num_states, std::move(smesh)),
      current_coeff(3, [](const mfem::Vector &p, mfem::Vector& J)
      {
         J.SetSize(3);
         J = 0.0;
         J(2) = sin(p(0)) + cos(p(1));
         // J(2) = -6 * p(0);
      })
   {
      options["time-dis"]["type"] = "steady";
      spatial_res = std::make_unique<mach::MachResidual>(MagnetostaticResidual(fes(), options, current_coeff, nu));
      setOptions(*spatial_res, options);

      nlohmann::json prec_options;
      auto *prec = getPreconditioner(*spatial_res, prec_options);
      auto lin_solver_opts = options["lin-solver"];
      linear_solver = mach::constructLinearSolver(comm, lin_solver_opts, prec);
      auto nonlin_solver_opts = options["nonlin-solver"];
      nonlinear_solver = mach::constructNonlinearSolver(comm, nonlin_solver_opts, *linear_solver);
      nonlinear_solver->SetOperator(*spatial_res);
   }
private:
   static constexpr int num_states = 1;
   mfem::VectorFunctionCoefficient current_coeff;
   BoxCoefficient nu;
};

TEST_CASE("Testing PDESolver unsteady heat equation MMS")
{
   const bool verbose = true; // set to true for some output 
   std::ostream *out = verbose ? mach::getOutStream(0) : mach::getOutStream(1);

   // Provide the options explicitly for regression tests
   auto options = R"(
   {
      "print-options": true,
      "time-dis": {
         "type": "steady",
         "t-final": 1.0,
         "dt": 0.05,
         "max-iter": 1
      },
      "space-dis": {
         "basis-type": "nedelec",
         "degree": 2
      },
      "lin-solver": {
         "type": "minres",
         "reltol": 1e-12,
         "abstol": 0.0,
         "printlevel": 1,
         "maxiter": 500
      },
      "lin-prec": {
         "type": "hypreams",
         "printlevel": -1
      },
      "nonlin-solver": {
         "maxiter": 1,
         "printlevel": 3
      },
      "ess-bdr": "all"
   })"_json;

   constexpr int nxy = 4;
   auto mesh = std::make_unique<mfem::Mesh>(
      mfem::Mesh::MakeCartesian3D(nxy, nxy, 2, mfem::Element::TETRAHEDRON,
                                  1.0, 1.0, 2.0 / double(nxy)));

   // Create solver and solve for the state 
   MagnetostaticSolver solver(MPI_COMM_WORLD, options, std::move(mesh));
   auto &state = solver.getState();

   mfem::VectorFunctionCoefficient exact_sol(3, [](const mfem::Vector &p, mfem::Vector& A)
   {
      A.SetSize(3);
      A = 0.0;
      A(2) = sin(p(0)) + cos(p(1));
      // A(2) = pow(p(0), 3);
   });
   state.gridFunc() = 0.0;
   mfem::Array<int> bdr_attr(6);
   bdr_attr = 1;
   state.gridFunc().ProjectCoefficient(exact_sol);
   // state.gridFunc().ProjectBdrCoefficientTangent(exact_sol, bdr_attr);
   mfem::Vector state_tv(solver.getStateSize());
   state.setTrueVec(state_tv);

   mach::MachInputs inputs;
   solver.solveForState(inputs, state_tv);
   state.distributeSharedDofs(state_tv);

   auto error = state.gridFunc().ComputeLpError(2, exact_sol);

   if (verbose)
   {
      std::cout << "terminal solution error = " << error << std::endl;
   }
   REQUIRE(error == Approx(0.00131875).margin(1e-5));
   /// 2nd order:
   /// 1 rank error: 0.00131875
   /// 2 ranks error: 0.00131708
   /// 4 ranks error: 0.00131498
   /// 8 ranks error: 0.00131322
   /// 4th order:
   /// 1 rank error: 0.0000005963
   /// 2 ranks error: 0.00000062
   /// 4 ranks error: 0.0000006282
   /// 8 ranks error: 0.0000006518

}
