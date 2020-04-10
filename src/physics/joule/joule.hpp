#ifndef MACH_JOULE
#define MACH_JOULE

#include "solver.hpp"

namespace mach
{

class MagnetostaticSolver;
class ThermalSolver;


class JouleSolver : public AbstractSolver
{
public:
	/// Class constructor.
   /// \param[in] opt_file_name - file where options are stored
   /// \param[in] smesh - if provided, defines the mesh for the problem
   /// \param[in] dim - number of dimensions
   JouleSolver(const std::string &opt_file_name,
               std::unique_ptr<mfem::Mesh> smesh = nullptr,
					int dim = 3);

   /// Write the solutions of both the EM and thermal problems to a vtk file
   /// \param[in] file_name - prefix file name **without** .vtk extension
   ///                        EM vtk file will be filename_em,
   ///                        Thermal vtk file will be filename_thermal
   /// \param[in] refine - if >=0, indicates the number of refinements to make
   /// \todo make this work for parallel!
   /// \note the `refine` argument is useful for high-order meshes and
   /// solutions; it divides the elements up so it is possible to visualize.
   void printSolution(const std::string &file_name, int refine = -1) override;

   /// \brief Returns a vector of pointers to grid functions that define fields
   /// returns {T, A, B}
   std::vector<GridFunType*> getFields() override;

   void solveForState() override;

private:
   std::unique_ptr<MagnetostaticSolver> em_solver;
   std::unique_ptr<ThermalSolver> thermal_solver;

   std::vector<GridFunType*> em_fields;
   std::vector<GridFunType*> thermal_fields;

   /// Raviart-Thomas finite element collection
   std::unique_ptr<mfem::FiniteElementCollection> h_div_coll;
   /// H(Div) finite element space
   std::unique_ptr<SpaceType> h_div_space;
   /// magnetic field mapped onto thermal solver's mesh
   std::unique_ptr<GridFunType> mapped_mag_field;
};

} // namespace mach

#endif