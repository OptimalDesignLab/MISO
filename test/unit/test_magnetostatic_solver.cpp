#include <random>

#include "catch.hpp"
#include "mfem.hpp"
#include "json.hpp"

#include "magnetostatic.hpp"
#include "electromag_test_data.hpp"

TEST_CASE("MagnetostaticSolver::getMeshSensitivities",
          "[MagnetostaticSolver]")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3;
   const double delta = 1e-5;
   const double fd_delta = 1e-7;

   int mesh_el = 4;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         // generate initial tet mesh
         auto init_mesh = getMesh(mesh_el,1);
         nlohmann::json options = getBoxOptions(p);

         mach::MagnetostaticSolver solver(options, move(init_mesh));
         solver.initDerived();
         solver.solveForState();
         double energy = solver.calcOutput("co-energy");
         GridFunction *dJdX = solver.getMeshSensitivities();

         GridFunction v(dJdX->FESpace());
         Vector zero;
         Vector rand(3);
         randState(zero, rand);
         VectorConstantCoefficient v_rand(rand);
         // VectorFunctionCoefficient v_rand(dim, randState);
         v.ProjectCoefficient(v_rand);

         double dJdX_v = (*dJdX) * v;

         /// now compute centered difference difference
         double dJdX_v_cd = 0.0;
         // back step
         {
            auto back_mesh = getMesh(mesh_el,1);

            mach::GridFunType back_pert(*static_cast<mach::GridFunType*>(back_mesh->GetNodes()));
            back_pert.Add(-delta, v);
            back_mesh->SetNodes(back_pert);
            mach::MagnetostaticSolver back_solver(options, move(back_mesh));
            back_solver.initDerived();
            back_solver.solveForState();
            dJdX_v_cd -= back_solver.calcOutput("co-energy");
         }

         // forward step
         {
            auto forward_mesh = getMesh(mesh_el,1);

            GridFunction forward_pert(*forward_mesh->GetNodes());
            forward_pert.Add(delta, v);
            forward_mesh->SetNodes(forward_pert);
            mach::MagnetostaticSolver forward_solver(options, move(forward_mesh));
            forward_solver.initDerived();
            forward_solver.solveForState();
            double forward_energy = forward_solver.calcOutput("co-energy");
            dJdX_v_cd += forward_energy;
         }
         dJdX_v_cd /= (2*delta);

         /// now compute forward difference
         double dJdX_v_fd = -energy;
         // forward step
         {
            auto forward_mesh = getMesh(mesh_el,1);

            GridFunction forward_pert(*forward_mesh->GetNodes());
            forward_pert.Add(fd_delta, v);
            forward_mesh->SetNodes(forward_pert);
            mach::MagnetostaticSolver forward_solver(options, move(forward_mesh));
            forward_solver.initDerived();
            forward_solver.solveForState();
            dJdX_v_fd += forward_solver.calcOutput("co-energy");
         }
         dJdX_v_fd /= fd_delta;

         std::cout << "Forward difference: " << dJdX_v_fd << "\n";
         std::cout << "Central difference: " << dJdX_v_cd << "\n";
         std::cout << "Analytic derivative: " << dJdX_v << "\n";

         // REQUIRE(dJdX_v == Approx(dJdX_v_cd).margin(1e-10));
         REQUIRE(dJdX_v == Approx(dJdX_v_fd).margin(1e-10));
      }
   }
}