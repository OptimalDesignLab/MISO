// #include "gdsolver.hpp"

// using namespace std;
// using namespace mfem;

// namespace mach
// {

// template <int dim>
// GDSolver<dim>::GDSolver(const string &opt_file_name,
//                         unique_ptr<Mesh2> smesh)
//    : EulerSolver<dim>(opt_file_name, move(smesh))
// {
//    int degree = this->options["GD"]["degree"].template get<int>();
//    this->fes.reset(new GalerkinDifference(degree, smesh, dim+2, Ordering::byVDIM));
// }

// } // end of namespace mach