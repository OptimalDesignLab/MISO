#include "solver.hpp"

using namespace std;
using namespace mfem;

namespace mach
{
adept::Stack AbstractSolver::diff_stack;

// Note: the following constructor is protected
AbstractSolver2::AbstractSolver2(const string &opt_file_name, MPI_Comm incomm)
{
   // Set the options; the defaults are overwritten by the values in the file
   // using the merge_patch method
   nlohmann::json file_options;
   ifstream options_file(opt_file_name);
   options_file >> file_options;
   options = default_options;
   options.merge_patch(file_options);
   // display options, unless "silent" is true
   bool silent = options.value("silent", false);
   out = getOutStream(rank, silent);
   *out << setw(3) << options << endl;

   // comm = incomm;
   MPI_Comm_dup(incomm, &comm);
   MPI_Comm_rank(comm, &rank);
   out = getOutStream(rank);

   // construct various solvers and preconditioners
   prec = constructPreconditioner(options["lin-prec"], comm);
   solver = constructLinearSolver(options["lin-solver"], *prec, comm);
   newton_solver = constructNonlinearSolver(options["nonlin-solver"], *solver, 
                                            comm);
   ode_solver = constructODESolver(options["time-dis"]);
}

AbstractSolver2::~AbstractSolver2()
{
   *out << "Deleting Abstract Solver..." << endl;
   MPI_Comm_free(&comm);
}

void AbstractSolver2::solveForState(const MachInputs &inputs,
                                    mfem::Vector &state)
{
   double t = 0.0; // this should probably be based on an option
   SetTime(t);
   ode_solver->Init(*this);
   auto t_final = options["time-dis"]["t-final"].template get<double>();
   *out << "t_final is " << t_final << '\n';
   int ti = 0;
   double dt = 0.0;
   initialHook(state);
   for (ti = 0; ti < options["time-dis"]["max-iter"].get<int>(); ++ti)
   {
      dt = calcStepSize(ti, t, t_final, dt, state);
      *out << "iter " << ti << ": time = " << t << ": dt = " << dt;
      if (!options["time-dis"]["steady"].get<bool>())
      {
         *out << " (" << round(100 * t / t_final) << "% complete)";
      }
      *out << endl;
      iterationHook(ti, t, dt, state);
      ode_solver->Step(state, t, dt);

      if (iterationExit(ti, t, t_final, dt, state)) break;
   }
   terminalHook(ti, t, state);
}

double AbstractSolver2::calcStepSize(int iter,
                                     double t,
                                     double t_final,
                                     double dt_old,
                                     const Vector &state) const
{
   auto dt = options["time-dis"]["dt"].get<double>();
   dt = min(dt, t_final - t);
   return dt;
}

bool AbstractSolver2::iterationExit(int iter,
                                   double t,
                                   double t_final,
                                   double dt,
                                   const Vector &state) const
{
   return t >= t_final - 1e-14 * dt;
}

} // namespace mach