#include <fstream>

#include "euler.hpp"
#include "evolver.hpp"

using namespace std;
using namespace mfem;

namespace mach
{

template<int dim, bool entvar>
void EulerSolver<dim, entvar>::verifyParamSens()
{
	std::cout << "Verifying Drag Sensitivity to Mach Number..." << std::endl;
	double delta = 1e-7;
	double delta_cd = 1e-5;
   string drags = "drag";
	double dJdX_fd_v = -calcOutput(drags)/delta;
	double dJdX_cd_v;
	double dJdX_a = getParamSens();
    // extract mesh nodes and get their finite-element space

   // compute finite difference approximation
   mach_fs += delta;
   std::cout << "Solving Forward Step..." << std::endl;
   constructMesh(nullptr);
	initDerived();
   constructLinearSolver(options["lin-solver"]);
	constructNewtonSolver();
	constructEvolver();
   HYPRE_ClearAllErrors();
   Vector qfar(4);
   getFreeStreamState(qfar);
	setInitialCondition(qfar);
   solveForState();
   std::cout << "Solver Done" << std::endl;
   dJdX_fd_v += calcOutput(drags)/delta;

   std::cout << "Mach Number Sensitivity (FD Only):  " << std::endl;
    std::cout << "Finite Difference:  " << dJdX_fd_v << std::endl;
    std::cout << "Analytic: 		  " << dJdX_a << std::endl;
	std::cout << "FD Relative: 		  " << (dJdX_a-dJdX_fd_v)/dJdX_a << std::endl;
    std::cout << "FD Absolute: 		  " << dJdX_a - dJdX_fd_v << std::endl;

	//central difference approximation
	std::cout << "Solving CD Backward Step..." << std::endl;
	mach_fs -= delta; mach_fs -= delta_cd;
	constructMesh(nullptr);
	initDerived();
   constructLinearSolver(options["lin-solver"]);
	constructNewtonSolver();
	constructEvolver();
   HYPRE_ClearAllErrors();
   getFreeStreamState(qfar);
	setInitialCondition(qfar);
   solveForState();
   std::cout << "Solver Done" << std::endl;
   dJdX_cd_v = -calcOutput(drags)/(2*delta_cd);

	std::cout << "Solving CD Forward Step..." << std::endl;
   mach_fs += 2*delta_cd;
	constructMesh(nullptr);
	initDerived();
   constructLinearSolver(options["lin-solver"]);
	constructNewtonSolver();
	constructEvolver();
   HYPRE_ClearAllErrors();
   getFreeStreamState(qfar);
	setInitialCondition(qfar);
   solveForState();
   std::cout << "Solver Done" << std::endl;
   dJdX_cd_v += calcOutput(drags)/(2*delta_cd);

	std::cout << "Mach Number Sensitivity:  " << std::endl;
    std::cout << "Finite Difference:  " << dJdX_fd_v << std::endl;
	std::cout << "Central Difference: " << dJdX_cd_v << std::endl;
    std::cout << "Analytic: 		  " << dJdX_a << std::endl;
	std::cout << "FD Relative: 		  " << (dJdX_a-dJdX_fd_v)/dJdX_a << std::endl;
    std::cout << "FD Absolute: 		  " << dJdX_a - dJdX_fd_v << std::endl;
	std::cout << "CD Relative: 		  " << (dJdX_a-dJdX_cd_v)/dJdX_a << std::endl;
    std::cout << "CD Absolute: 		  " << dJdX_a - dJdX_cd_v << std::endl;
}

}