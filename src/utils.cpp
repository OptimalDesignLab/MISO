#include "utils.hpp"

using namespace mfem;
using namespace std;

namespace mach
{
	double quadInterp(double x0, double y0, double dydx0,
								double x1, double y1)
	{
		double c0, c1, c2;
		c0 = (dydx0*x0*x0*x1 + y1*x0*x0 - dydx0*x0*x1*x1
				-2*y0*x0*x1 + y0*x1*x1)/(x0*x0 - 2*x1*x0 + x1*x1);
		c1 = (2*x0*y0 - 2*x0*y1 - x0*x0*dydx0 + x1*x1*dydx0)
				/(x0*x0 - 2*x1*x0 + x1*x1);
		c2 = -(y0 - y1 - x0*dydx0 + x1*dydx0)/(x0*x0 - 2*x1*x0 + x1*x1);
		//std::cout << c2 <<  "  "<< c1 << "  " << c0 <<std::endl;
		return -c1/(2*c2);
	}
} // end of namespace
