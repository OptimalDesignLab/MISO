#include "default_options.hpp"
#include "mfem_extensions.hpp"
#include "utils.hpp"

#include "abstract_solver.hpp"

namespace
{
void logState(mach::DataLogger &logger,
              const mfem::Vector &state,
              std::string fieldname,
              int timestep,
              double time,
              int rank)
{
   std::visit([&](auto &&log)
              { log.saveState(state, fieldname, timestep, time, rank); },
              logger);
}

}  // namespace

namespace mach
{
adept::Stack AbstractSolver2::diff_stack;

AbstractSolver2::AbstractSolver2(MPI_Comm incomm,
                                 const nlohmann::json &solver_options)
{
   /// Set the options; the defaults are overwritten by the values in the file
   /// using the merge_patch method
   options = default_options;
   options.merge_patch(solver_options);

   MPI_Comm_dup(incomm, &comm);
   MPI_Comm_rank(comm, &rank);
}

void AbstractSolver2::setState_(std::any function,
                                const std::string &name,
                                mfem::Vector &state)
{
   useAny(function,
          [&](std::function<void(mfem::Vector &)> &fun) { fun(state); });
}

double AbstractSolver2::calcStateError_(std::any ex_sol,
                                        const std::string &name,
                                        const mfem::Vector &state)
{
   return useAny(
       ex_sol,
       [&](std::function<void(mfem::Vector &)> &fun)
       {
          work.SetSize(state.Size());
          fun(work);
          subtract(work, state, work);
          return work.Norml2();
       },
       [&](mfem::Vector &vec)
       {
          if (vec.Size() != state.Size())
          {
             throw MachException(
                 "Input vector for exact solution is not "
                 "the same size as the "
                 "state vector!");
          }
          work.SetSize(state.Size());
          subtract(vec, state, work);
          return work.Norml2();
       });
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
      if (iterationExit(ti, t, t_final, dt, state))
      {
         break;
      }
   }
   terminalHook(ti, t, state);
}

void AbstractSolver2::calcResidual(const mfem::Vector &state,
                                   mfem::Vector &residual) const
{
   MachInputs inputs{{"state", state}};
   calcResidual(inputs, residual);
}

void AbstractSolver2::calcResidual(const MachInputs &inputs,
                                   mfem::Vector &residual) const
{
   auto timestepper = options["time-dis"]["type"].get<std::string>();
   if (!(timestepper == "steady" || timestepper == "PTC"))
   {
      throw MachException(
          "calcResidual should only be called for steady problems!\n");
   }
   evaluate(*res, inputs, residual);
}

double AbstractSolver2::calcResidualNorm(const mfem::Vector &state) const
{
   MachInputs inputs{{"state", state}};
   return calcResidualNorm(inputs);
}

double AbstractSolver2::calcResidualNorm(const MachInputs &inputs) const
{
   work.SetSize(getSize(*res));
   calcResidual(inputs, work);
   return sqrt(InnerProduct(comm, work, work));
}

int AbstractSolver2::getFieldSize(const std::string &name) const
{
   if (name == "state" || name == "residual" || name == "adjoint")
   {
      return getSize(*res);
   }
   // may be better way to return something not found without throwing error
   return 0;
}

void AbstractSolver2::createOutput(const std::string &output)
{
   nlohmann::json options;
   createOutput(output, options);
}

void AbstractSolver2::createOutput(const std::string &output,
                                   const nlohmann::json &options)
{
   if (outputs.count(output) == 0)
   {
      addOutput(output, options);
   }
   else
   {
      throw MachException("Output with name " + output + " already created!\n");
   }
}

void AbstractSolver2::setOutputOptions(const std::string &output,
                                       const nlohmann::json &options)
{
   try
   {
      auto output_iter = outputs.find(output);
      if (output_iter == outputs.end())
      {
         throw MachException("Did not find " + output + " in output map?");
      }
      mach::setOptions(output_iter->second, options);
   }
   catch (const std::out_of_range &exception)
   {
      std::cerr << exception.what() << std::endl;
   }
}

double AbstractSolver2::calcOutput(const std::string &output,
                                   const MachInputs &inputs)
{
   try
   {
      auto output_iter = outputs.find(output);
      if (output_iter == outputs.end())
      {
         throw MachException("Did not find " + output + " in output map?");
      }
      return mach::calcOutput(output_iter->second, inputs);
   }
   catch (const std::out_of_range &exception)
   {
      std::cerr << exception.what() << std::endl;
      return std::nan("");
   }
}

void AbstractSolver2::calcOutputPartial(const std::string &of,
                                        const std::string &wrt,
                                        const MachInputs &inputs,
                                        double &partial)
{
   try
   {
      auto output_iter = outputs.find(of);
      if (output_iter == outputs.end())
      {
         throw MachException("Did not find " + of + " in output map?");
      }
      double part = mach::calcOutputPartial(output_iter->second, wrt, inputs);
      partial += part;
   }
   catch (const std::out_of_range &exception)
   {
      std::cerr << exception.what() << std::endl;
      partial = std::nan("");
   }
}

void AbstractSolver2::calcOutputPartial(const std::string &of,
                                        const std::string &wrt,
                                        const MachInputs &inputs,
                                        mfem::Vector &partial)
{
   try
   {
      auto output_iter = outputs.find(of);
      if (output_iter == outputs.end())
      {
         throw MachException("Did not find " + of + " in output map?");
      }
      mach::calcOutputPartial(output_iter->second, wrt, inputs, partial);
   }
   catch (const std::out_of_range &exception)
   {
      std::cerr << exception.what() << std::endl;
   }
}

void AbstractSolver2::initialHook(const mfem::Vector &state)
{
   for (auto &pair : loggers)
   {
      auto &logger = pair.first;
      auto &options = pair.second;
      if (options.initial_state)
      {
         logState(logger, state, "state", 0, 0.0, rank);
      }
   }
}

void AbstractSolver2::iterationHook(int iter,
                                    double t,
                                    double dt,
                                    const mfem::Vector &state)
{
   for (auto &pair : loggers)
   {
      auto &logger = pair.first;
      auto &options = pair.second;
      if (options.each_timestep)
      {
         logState(logger, state, "state", iter, t, rank);
      }
   }
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

void AbstractSolver2::terminalHook(int iter,
                                   double t_final,
                                   const mfem::Vector &state)
{
   for (auto &pair : loggers)
   {
      auto &logger = pair.first;
      auto &options = pair.second;
      if (options.final_state)
      {
         logState(logger, state, "state", iter, t_final, rank);
      }
   }
}

}  // namespace mach
