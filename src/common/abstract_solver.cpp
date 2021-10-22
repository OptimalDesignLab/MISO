#include "default_options.hpp"

#include "abstract_solver.hpp"

namespace mach
{
AbstractSolver2::AbstractSolver2(const nlohmann::json &solver_options,
                                 MPI_Comm incomm)
{
   /// Set the options; the defaults are overwritten by the values in the file
   /// using the merge_patch method
   options = default_options;
   options.merge_patch(solver_options);

   MPI_Comm_dup(incomm, &comm);
   MPI_Comm_rank(comm, &rank);
}

void AbstractSolver2::solveForState(const MachInputs &inputs,
                                    mfem::Vector &state)
{
   auto ode_opts = options["time-dis"];

   double t = 0.0;  // this should probably be based on an option
   auto t_final = ode_opts["t-final"].get<double>();
   std::cout << "t_final is " << t_final << '\n';
   int ti = 0;
   double dt = 0.0;
   initialHook(state);
   for (ti = 0; ti < ode_opts["max-iter"].get<int>(); ++ti)
   {
      dt = calcStepSize(ti, t, t_final, dt, state);
      std::cout << "iter " << ti << ": time = " << t << ": dt = " << dt;
      if (!ode_opts["steady"].get<bool>())
      {
         std::cout << " (" << round(100 * t / t_final) << "% complete)";
      }
      std::cout << std::endl;
      iterationHook(ti, t, dt, state);
      ode->step(state, t, dt);
      if (iterationExit(ti, t, t_final, dt, state)) break;
   }
   terminalHook(ti, t, state);
}

double AbstractSolver2::calcStepSize(int iter,
                                     double t,
                                     double t_final,
                                     double dt_old,
                                     const mfem::Vector &state) const
{
   auto dt = options["time-dis"]["dt"].get<double>();
   dt = std::min(dt, t_final - t);
   return dt;
}

bool AbstractSolver2::iterationExit(int iter,
                                    double t,
                                    double t_final,
                                    double dt,
                                    const mfem::Vector &state) const
{
   return t >= t_final - 1e-14 * dt;
}

}  // namespace mach
