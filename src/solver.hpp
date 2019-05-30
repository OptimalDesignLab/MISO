#ifndef MACH_SOLVER
#define MACH_SOLVER

#include "mfem.hpp"

using namespace std;  // TODO: needed?
using namespace mfem;

namespace mach
{

/*!
 * \class PDESolver
 * \brief 
 */
class PDESolver
{
protected:
    int degree;
    int temp;

public:
    PDESolver(OptionsParser &args);

};
    
} // namespace mach

#endif 