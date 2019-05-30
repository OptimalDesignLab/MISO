// Using this to test different ideas

#include "mfem.hpp"
#include "solver.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mach;

int main(int argc, char *argv[])
{
  // Parse command-line options and create solver object
  OptionsParser args(argc, argv);
  PDESolver solver(args);
}