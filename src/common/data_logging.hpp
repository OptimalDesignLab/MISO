#ifndef MACH_DATA_LOGGING
#define MACH_DATA_LOGGING

#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>

#include "mfem.hpp"

namespace mach
{

class ASCIILogger
{
public:
   void saveState(const mfem::Vector &state,
                  std::string fieldname,
                  int timestep,
                  double time,
                  int rank)
   {
      std::filesystem::create_directory(prefix);
      auto filename = prefix + "/" + fieldname + "_" + std::to_string(timestep) + "_" + std::to_string(rank);
      const double *data = state.GetData();
      int size = state.Size();
      std::ofstream file(filename, std::ios::out);
      file << std::setprecision(std::numeric_limits<int>::max());
      file << time << "\n";
      file << size << "\n";
      for (int i = 0; i < size; ++i)
      {
         file << data[i] << "\n";
      }
   }

   void readState(std::string fieldname,
                  int timestep,
                  int rank,
                  double &time,
                  mfem::Vector &state)
   {
      auto filename = prefix + "/" + fieldname + "_" + std::to_string(timestep) + "_" + std::to_string(rank);
      std::ifstream infile(filename);
      infile >> time;

      int size;
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
   void saveState(const mfem::Vector &state,
                  std::string fieldname,
                  int timestep,
                  double time,
                  int rank)
   {
      std::filesystem::create_directory(prefix);
      auto filename = prefix + "/" + fieldname + "_" + std::to_string(timestep) + "_" + std::to_string(rank);
      const double *data = state.GetData();
      int size = state.Size();
      std::ofstream file(filename, std::ios::out | std::ios::binary);
      file.write(reinterpret_cast<const char*>(&time), sizeof(double));
      file.write(reinterpret_cast<const char*>(&size), sizeof(int));
      file.write(reinterpret_cast<const char*>(data), std::streamsize(size*sizeof(double)));
   }

   void readState(std::string fieldname,
                  int timestep,
                  int rank,
                  double &time,
                  mfem::Vector &state)
   {
      auto filename = prefix + "/" + fieldname + "_" + std::to_string(timestep) + "_" + std::to_string(rank);
      std::ifstream infile(filename, std::ios::binary);
      infile.read(reinterpret_cast<char*>(&time), sizeof(double));
      int size;
      infile.read(reinterpret_cast<char*>(&size), sizeof(int));
      state.SetSize(size);
      auto *data = state.GetData();
      infile.read(reinterpret_cast<char*>(data), std::streamsize(size*sizeof(double)));
   }

private:
   inline static const std::string prefix = "BinaryLogger/";
};

class ParaViewLogger
{
public:
   void saveState(const mfem::Vector &state, 
                  std::string fieldname,
                  int timestep,
                  double time,
                  int rank)
   {
      pv.SetCycle(timestep);
      pv.SetTime(time);
      pv.Save();
   }

   void registerField(std::string name, mfem::ParGridFunction &field)
   {
      fields.emplace(std::move(name), &field);
      auto field_order = field.ParFESpace()->GetMaxElementOrder();
      refine = field_order > refine ? field_order : refine;
   }

   ParaViewLogger(const std::string &name,
                  mfem::ParMesh *mesh = nullptr)
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

} // namespace mach

#endif
