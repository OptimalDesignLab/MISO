#include "joule.hpp"

#include "magnetostatic.hpp"
#include "thermal.hpp"

using namespace std;
using namespace mfem;

namespace mach
{

/// TODO: read options file and construct options files for EM and 
JouleSolver::JouleSolver(
	const std::string &opt_file_name,
   std::unique_ptr<mfem::Mesh> smesh)
	// : AbstractSolver(opt_file_name, move(smesh))
   : AbstractSolver(opt_file_name)
{
   nlohmann::json em_opts = options["em-opts"];
   nlohmann::json thermal_opts = options["thermal-opts"];

   auto mesh_file = options["mesh"]["file"].get<std::string>();
   auto model_file = options["mesh"]["model-file"].get<std::string>();
   auto mesh_out_file = options["mesh"]["out-file"].get<std::string>();

   /// get mesh file name and extension 
   std::string mesh_name;
   std::string mesh_ext;
   {
      size_t i = mesh_file.rfind('.', mesh_file.length());
      if (i != string::npos) {
         mesh_name = (mesh_file.substr(0, i));
         mesh_ext = (mesh_file.substr(i+1, mesh_file.length() - i));
      }
      else
      {
         throw MachException("JouleSolver::JouleSolver()\n"
                           "\tMesh file has no extension!\n");
      }
   }

   em_opts["mesh"]["file"] = mesh_name + "_em." + mesh_ext;
   thermal_opts["mesh"]["file"] = mesh_name + "_thermal." + mesh_ext;

   /// get model file name and extension 
   std::string model_name;
   std::string model_ext;
   {
      size_t i = model_file.rfind('.', model_file.length());
      if (i != string::npos) {
         model_name = (model_file.substr(0, i));
         model_ext = (model_file.substr(i+1, model_file.length() - i));
      }
      else
      {
         throw MachException("JouleSolver::JouleSolver()\n"
                           "\tModel file has no extension!\n");
      }
   }

   em_opts["mesh"]["model-file"] = model_name + "_em." + model_ext;
   thermal_opts["mesh"]["model-file"] = model_name + "_thermal." + model_ext;

   em_opts["mesh"]["out-file"] = mesh_out_file + "_em";
   thermal_opts["mesh"]["out-file"] = mesh_out_file + "_thermal";

   em_opts["components"] = options["components"];
   thermal_opts["components"] = options["components"];

   em_opts["problem-opts"] = options["problem-opts"];
   thermal_opts["problem-opts"] = options["problem-opts"];

   /// TODO: need to do this until Magnetostatic solver is updated to support
   /// newer abstract solver construction model
   std::string em_opt_filename = "em_opt_file.json";
   {
      std::ofstream em_opt_file(em_opt_filename, std::ios::trunc);
      em_opt_file << em_opts;
      em_opt_file.close();
   }

   *out << "EM options:\n";
   *out << setw(3) << em_opts << endl;

   *out << "Thermal options:\n";
   *out << setw(3) << thermal_opts << endl;

   *out << "Thermal bcs options:\n";
   *out << setw(3) << options["thermal-opts"] << endl;

   std::unique_ptr<mfem::Mesh> em_smesh = nullptr;
   std::unique_ptr<mfem::Mesh> thermal_smesh = nullptr;
   if (smesh != nullptr)
   {
      std::ofstream smesh_out("joule_smesh_out.mesh");
      smesh->Print(smesh_out);
      smesh_out.close();

      std::ifstream em_smesh_in("joule_smesh_out.mesh");
      em_smesh.reset(new Mesh(em_smesh_in));
      em_smesh_in.close();

      std::ifstream thermal_smesh_in("joule_smesh_out.mesh");
      thermal_smesh.reset(new Mesh(thermal_smesh_in));
      thermal_smesh_in.close();
   }

   em_solver.reset(new MagnetostaticSolver(em_opt_filename, move(em_smesh)));
   /// TODO: this should be moved to an init derived when a factory is made
   // em_solver->initDerived();

   thermal_solver.reset(new ThermalSolver(thermal_opts, move(thermal_smesh)));
   // thermal_solver->initDerived();

}

void JouleSolver::initDerived()
{
   em_solver->initDerived();
   em_fields = em_solver->getFields();
   thermal_solver->setAField(em_fields[0]);
}

JouleSolver::~JouleSolver()
{
   *out << "Deleting Joule Solver..." << endl;
}

/// TODO: Change this in AbstractSolver to mark a flag so that unsteady solutions can be saved
void JouleSolver::printSolution(const std::string &filename, int refine)
{
   std::string em_filename = filename + "_em";
   std::string thermal_filename = filename + "_thermal";
   em_solver->printSolution(thermal_filename, refine);
   thermal_solver->printSolution(thermal_filename, refine);
}

void JouleSolver::setInitialCondition(double (*u_init)(const mfem::Vector &))
{
   thermal_init = u_init;
}

std::vector<GridFunType*> JouleSolver::getFields(void)
{
	return {thermal_fields[0], em_fields[0], em_fields[1]};
}

void JouleSolver::solveForState()
{
   em_solver->solveForState();
   thermal_solver->initDerived();
   thermal_fields = thermal_solver->getFields();
   thermal_solver->setInitialCondition(thermal_init);
   thermal_solver->solveForState();
}

void JouleSolver::solveForAdjoint(const std::string &fun)
{
   if (fun == "temp-agg")
   {
      thermal_solver->solveForAdjoint(fun);
      Vector psiTdRtdA;
      thermal_solver->getASensitivity(psiTdRtdA);
      em_solver->solveForAdjoint(psiTdRtdA);
   }
   else if (fun == "co-energy")
   {
      em_solver->solveForAdjoint(fun);
   }
}

// Vector* JouleSolver::getMeshSensitivities()
// {

// }

void JouleSolver::addOutputs()
{
   auto &fun = options["outputs"];
}

} // namespace mach
