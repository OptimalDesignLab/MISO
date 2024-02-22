#ifndef MISO_JOULE
#define MISO_JOULE

#include "solver.hpp"

namespace miso
{
class MagnetostaticSolver;
class ThermalSolver;

class JouleSolver : public AbstractSolver
{
public:
   /// Class constructor.
   /// \param[in] opt_file_name - file where options are stored
   /// \param[in] smesh - if provided, defines the mesh for the problem
   /// \param[in] comm - MPI communicator for parallel operations
   JouleSolver(const std::string &opt_file_name,
               std::unique_ptr<mfem::Mesh> smesh,
               MPI_Comm comm);

   /// Fully initialize the Joule Solver and its sub-solvers
   void initDerived() override;

   /// Write the solutions of both the EM and thermal problems to a vtk file
   /// \param[in] file_name - prefix file name **without** .vtk extension
   ///                        EM vtk file will be filename_em,
   ///                        Thermal vtk file will be filename_thermal
   /// \param[in] refine - if >=0, indicates the number of refinements to make
   /// \todo make this work for parallel!
   /// \note the `refine` argument is useful for high-order meshes and
   /// solutions; it divides the elements up so it is possible to visualize.
   void printSolution(const std::string &filename, int refine = -1) override;

   /// Initializes the state variable to a given function.
   /// \param[in] u_init - function that defines the initial condition
   void setInitialCondition(
       const std::function<double(const mfem::Vector &)> &u_init) override;

   /// \brief Returns a vector of pointers to grid functions that define fields
   /// returns {T, A, B}
   std::vector<GridFunType *> getFields() override;

   void solveForState() override;

   void solveForAdjoint(const std::string &fun) override;

   // mfem::Vector* getMeshSensitivities() override;

   int getNumState() override { return 0; }

private:
   std::unique_ptr<MagnetostaticSolver> em_solver;
   std::unique_ptr<ThermalSolver> thermal_solver;

   std::vector<GridFunType *> em_fields;
   std::vector<GridFunType *> thermal_fields;

   std::function<double(const mfem::Vector &)> thermal_init;

   friend SolverPtr createSolver<JouleSolver>(const std::string &opt_file_name,
                                              std::unique_ptr<mfem::Mesh> smesh,
                                              MPI_Comm comm);
};

}  // namespace miso

#endif
