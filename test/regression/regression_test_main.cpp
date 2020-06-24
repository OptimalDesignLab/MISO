
// In a Catch project with multiple files, dedicate one file to compile the
// source code of Catch itself and reuse the resulting object file for linking.

// Let Catch provide main():
//#define CATCH_CONFIG_MAIN

// Define main for catch, since we need MPI
#define CATCH_CONFIG_RUNNER
#include "catch.hpp"
#include "mpi.h"

int main( int argc, char* argv[] ) {
  
  // global setup...
   // Initialize MPI if parallel
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  int result = Catch::Session().run( argc, argv );

  // global clean-up...
   MPI_Finalize();

  return result;
}