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
AbstractSolver2::AbstractSolver2(MPI_Comm incomm,
                                 const nlohmann::json &solver_options)
 : diff_stack(getDiffStack())
{
   /// Set the options; the defaults are overwritten by the values in the file
   /// using the merge_patch method
   options = default_options;
   options.merge_patch(solver_options);

   MPI_Comm_dup(incomm, &comm);
   MPI_Comm_rank(comm, &rank);

   bool silent = options.value("silent", false);
   out = getOutStream(rank, silent);
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
   if (spatial_res)
   {
      setInputs(*spatial_res, inputs);
   }

   /// if solving an unsteady problem
   if (ode)
   {
      auto ode_opts = options["time-dis"];

      double t = ode_opts["t-initial"].get<double>();
      auto t_final = ode_opts["t-final"].get<double>();
      *out << "t_final is " << t_final << '\n';
      int ti = 0;
      double dt = 0.0;
      initialHook(state);
      for (ti = 0; ti < ode_opts["max-iter"].get<int>(); ++ti)
      {
         dt = calcStepSize(ti, t, t_final, dt, state);
         *out << "iter " << ti << ": time = " << t << ": dt = " << dt;
         if (!ode_opts["steady"].get<bool>())
         {
            *out << " (" << round(100 * t / t_final) << "% complete)";
         }
         *out << std::endl;
         iterationHook(ti, t, dt, state);
         ode->step(state, t, dt);
         if (iterationExit(ti, t, t_final, dt, state))
         {
            break;
         }
      }
      terminalHook(ti, t, state);
   }
   else  /// steady problem, use Newton on spatial residual directly
   {
      /// use input state as initial guess
      nonlinear_solver->iterative_mode = true;

      mfem::Vector zero;
      nonlinear_solver->Mult(zero, state);

      /// log final state
      for (auto &pair : loggers)
      {
         auto &logger = pair.first;
         logState(logger, state, "state", 0, 0.0, rank);
      }
   }
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
   evaluate(*spatial_res, inputs, residual);
}

double AbstractSolver2::calcResidualNorm(const mfem::Vector &state) const
{
   // dt must be set to zero, so that the TimeDependentResidual knows to just
   // evaluate the spatial residual.
   MachInputs inputs{{"state", state}, {"dt", 0.0}};
   return calcResidualNorm(inputs);
}

double AbstractSolver2::calcResidualNorm(const MachInputs &inputs) const
{
   work.SetSize(getSize(*spatial_res));
   *out << "before calcResidual..." << std::endl;
   *out << "work.Size() = " << work.Size() << std::endl;
   calcResidual(inputs, work);
   *out << "after calcResidual..." << std::endl;
   return sqrt(InnerProduct(comm, work, work));
}

int AbstractSolver2::getStateSize() const
{
   if (spatial_res)
   {
      return getSize(*spatial_res);
   }
   else if (space_time_res)
   {
      return getSize(*space_time_res);
   }
   else
   {
      throw MachException(
          "getStateSize(): residual not defined! State size unknown.\n");
   }
}

int AbstractSolver2::getFieldSize(const std::string &name) const
{
   if (name == "state" || name == "residual" || name == "adjoint")
   {
      return getStateSize();
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

void AbstractSolver2::calcOutput(const std::string &output,
                                 const MachInputs &inputs,
                                 mfem::Vector &out_vec)
{
   try
   {
      auto output_iter = outputs.find(output);
      if (output_iter == outputs.end())
      {
         throw MachException("Did not find " + output + " in output map?");
      }
      mach::calcOutput(output_iter->second, inputs, out_vec);
   }
   catch (const std::out_of_range &exception)
   {
      std::cerr << exception.what() << std::endl;
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
