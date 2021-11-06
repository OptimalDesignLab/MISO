#ifndef MACH_DATA_LOGGING
#define MACH_DATA_LOGGING

#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>
#include <variant>

#include "mfem.hpp"

namespace mach
{

struct LoggingOptions
{
   bool initial_state = true;
   bool each_timestep = false;
   bool final_state = true;
};

class ASCIILogger
{
public:
   static void saveState(const mfem::Vector &state,
                         const std::string &fieldname,
                         int timestep,
                         double time,
                         int rank)
   {
      std::filesystem::create_directory(prefix);
      auto filename = prefix + "/" + fieldname + "_" +
                      std::to_string(timestep) + "_" + std::to_string(rank);
      const double *data = state.GetData();
      int size = state.Size();
      std::ofstream file(filename, std::ios::out);
      file << std::setprecision(18);
      file << time << "\n";
      file << size << "\n";
      for (int i = 0; i < size; ++i)
      {
         file << data[i] << "\n";
      }
   }

   static void readState(const std::string &fieldname,
                         int timestep,
                         int rank,
                         double &time,
                         mfem::Vector &state)
   {
      auto filename = prefix + "/" + fieldname + "_" +
                      std::to_string(timestep) + "_" + std::to_string(rank);
      std::ifstream infile(filename);
      infile >> time;

      int size = 0;
      infile >> size;

      state.SetSize(size);
      for (int i = 0; i < size; ++i)
      {
         infile >> state(i);
      }
   }

private:
   inline static const std::string prefix = "ASCIILogger";
};

class BinaryLogger
{
public:
   static void saveState(const mfem::Vector &state,
                         const std::string &fieldname,
                         int timestep,
                         double time,
                         int rank)
   {
      std::filesystem::create_directory(prefix);
      auto filename = prefix + "/" + fieldname + "_" +
                      std::to_string(timestep) + "_" + std::to_string(rank);
      const double *data = state.GetData();
      int size = state.Size();
      std::ofstream file(filename, std::ios::out | std::ios::binary);
      file.write(reinterpret_cast<const char *>(&time),                // NOLINT
                 sizeof(double));
      file.write(reinterpret_cast<const char *>(&size), sizeof(int));  // NOLINT
      file.write(reinterpret_cast<const char *>(data),                 // NOLINT
                 std::streamsize(size * sizeof(double)));
   }

   static void readState(const std::string &fieldname,
                         int timestep,
                         int rank,
                         double &time,
                         mfem::Vector &state)
   {
      auto filename = prefix + "/" + fieldname + "_" +
                      std::to_string(timestep) + "_" + std::to_string(rank);
      std::ifstream infile(filename, std::ios::binary);
      infile.read(reinterpret_cast<char *>(&time), sizeof(double));  // NOLINT
      int size = 0;
      infile.read(reinterpret_cast<char *>(&size), sizeof(int));  // NOLINT
      state.SetSize(size);
      auto *data = state.GetData();
      infile.read(reinterpret_cast<char *>(data),  // NOLINT
                  std::streamsize(size * sizeof(double)));
   }

private:
   inline static const std::string prefix = "BinaryLogger/";
};

class ParaViewLogger
{
public:
   void saveState(const mfem::Vector &state,
                  const std::string &fieldname,
                  int timestep,
                  double time,
                  int rank)
   {
      fields.at(fieldname)->SetFromTrueDofs(state);
      pv.SetCycle(timestep);
      pv.SetTime(time);
      pv.Save();
   }

   void registerField(const std::string &name, mfem::ParGridFunction &field)
   {
      pv.RegisterField(name, &field);
      fields.emplace(name, &field);
      auto field_order = field.ParFESpace()->GetMaxElementOrder();
      if (field_order > refine)
      {
         refine = field_order;
         pv.SetLevelsOfDetail(refine);
      }
   }

   ParaViewLogger(const std::string &name, mfem::ParMesh *mesh = nullptr)
    : pv(name, mesh)
   {
      pv.SetPrefixPath("ParaView");
      pv.SetLevelsOfDetail(refine);
      pv.SetDataFormat(mfem::VTKFormat::BINARY);
      pv.SetHighOrderOutput(true);
   }

private:
   /// ParaView object for saving fields
   mfem::ParaViewDataCollection pv;
   /// Map of all state vectors that may be saved by ParaView
   std::map<std::string, mfem::ParGridFunction *> fields;
   /// Paraview levels of refinement for field printing
   int refine = 1;
};

using DataLogger = std::variant<ASCIILogger, BinaryLogger, ParaViewLogger>;
using DataLoggerWithOpts = std::pair<DataLogger, LoggingOptions>;

}  // namespace mach

#endif
