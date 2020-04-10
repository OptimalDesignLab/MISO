#include "joule.hpp"

#include "magnetostatic.hpp"
#include "thermal.hpp"

using namespace std;
using namespace mfem;

namespace mach
{

JouleSolver::JouleSolver(
	const std::string &opt_file_name,
   std::unique_ptr<mfem::Mesh> smesh,
	int dim)
	: AbstractSolver(opt_file_name, move(smesh))
{
   std::string em_opt_file = "1";
   std::string thermal_opt_file = "1";

   em_solver.reset(new MagnetostaticSolver(em_opt_file, nullptr));
   /// TODO: this should be moved to an init derived when a factory is made
   em_solver->initDerived();
   em_fields = em_solver->getFields();

   /// TODO: get order?
	/// Create the H(Div) finite element collection
   h_div_coll.reset(new RT_FECollection(1, dim));
	/// Create the H(Div) finite element space
	h_div_space.reset(new SpaceType(mesh.get(), h_div_coll.get()));
   /// Create magnetic flux grid function
	mapped_mag_field.reset(new GridFunType(h_div_space.get()));

   thermal_solver.reset(new ThermalSolver(thermal_opt_file, nullptr, dim,
                                          mapped_mag_field.get()));
   thermal_solver->initDerived();
   thermal_fields = thermal_solver->getFields();
}

void JouleSolver::printSolution(const std::string &filename, int refine)
{
   std::string em_filename = filename + "_em";
   std::string thermal_filename = filename + "_thermal";
   em_solver->printSolution(thermal_filename, refine);
   thermal_solver->printSolution(thermal_filename, refine);
}

std::vector<GridFunType*> JouleSolver::getFields(void)
{
	return {thermal_fields[0], em_fields[0], em_fields[1]};
}

void JouleSolver::solveForState()
{
   em_solver->solveForState();

   transferSolution(*em_solver->getMesh(), *thermal_solver->getMesh(),
                    *em_fields[1], *mapped_mag_field);

   thermal_solver->solveForState();
}

} // namespace mach
