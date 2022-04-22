/// Solve the wedge shock problem on a 2D mesh, exploiting symmetry
constexpr bool entvar = false;
#include "mfem.hpp"
#include "euler.hpp"
#include "euler_fluxes.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mach;

/// \brief Defines the exact solution for the wedge shock problem
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - conservative variables stored as a 4-vector
void uexact(const Vector &x, Vector& u);
void uinit(const Vector&x, Vector& u);

int main(int argc, char *argv[])
{
	int nx = 10;
	int ny = 10;

   OptionsParser args(argc, argv);
   const char *options_file = "wedgeshock_options.json";
   args.AddOption(&nx, "-nx", "--num-rad", "number of radial segments");
   args.AddOption(&ny, "-ny", "--num-theta", "number of angular segments");
   args.AddOption(&options_file, "-o", "--options",
                  "Options file to use.");               
   args.Parse();

   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   try
   {
      // construct the solver, set the initial condition, and solve
      string opt_file_name(options_file);
		unique_ptr<Mesh> bmesh (new Mesh(nx,ny,Element::TRIANGLE,true,2.0,1.0,true));
      int numBasis = bmesh->GetNE();
      Vector center(2*numBasis);
      Vector loc(2);
      for (int k = 0; k < numBasis; k++)
      {  
         bmesh->GetElementCenter(k,loc);
         center(k*2) = loc(0);
         center(k*2+1) = loc(1);
      }

      const int dim = 2;

      unique_ptr<Mesh> smesh(new Mesh(nx,ny,Element::TRIANGLE,true,2.0,1.0,true));
      std::cout << "Number of elements " << smesh->GetNE() <<'\n';
      unique_ptr<AbstractSolver> solver(
         new EulerSolver<2, entvar>(opt_file_name, move(smesh)));
      solver->initDerived(center);
      solver->setInitialCondition(uexact);
      mfem::out << "\n|| rho_h - rho ||_{L^2} = " 
                << solver->calcL2Error(uexact, 0) << '\n' << endl;
      mfem::out << "\ninitial residual norm = " << solver->calcResidualNorm()
                << endl;
      solver->solveForState();
      mfem::out << "\nfinal residual norm = " << solver->calcResidualNorm()
                << endl;
      mfem::out << "\n|| rho_h - rho ||_{L^2} = " 
                << solver->calcL2Error(uexact, 0) << '\n' << endl;

   }
   catch (MachException &exception)
   {
      exception.print_message();
   }
   catch (std::exception &exception)
   {
      cerr << exception.what() << endl;
   }
#ifdef MFEM_USE_MPI
   MPI_Finalize();
#endif
}

// Exact solution; 
void uinit(const Vector &x, Vector& u)
{
   u.SetSize(4);
   double Mai = 2.4; //Ma1
   double rhoi = 1.0; //rho1
   double prsi = 1.0/euler::gamma;
   //assuming theta = 25 degrees, Ma1 = 2.4
   double theta = 25*2*M_PI/360;
   double beta =  52.17187440*2*M_PI/360; 
   //taken from Figure 9.9, Anderson for theta = 25 degrees, Ma1 = 2.4
   
   //compute mach number downstream of shock
   double Ma1n = Mai*sin(beta);
   double Ma2n = sqrt((1+(.5*euler::gami)*Ma1n*Ma1n) /
                     (euler::gamma*Ma1n*Ma1n - .5*euler::gami));
   double Ma = Ma2n/sin(beta-theta);
   
   //compute other quantities using continuity, momentum, and energy equations
   double rho = rhoi*(euler::gamma+1)*Ma1n*Ma1n / 
                  (2+euler::gami*Ma1n*Ma1n);
   double press = prsi*(1 + (2*euler::gamma/(euler::gamma+1))*(Ma1n*Ma1n - 1)); 
   double a = sqrt(euler::gamma*press/rho);

   
   
   double thresh = x(1)/tan(beta); //assuming wedge tip is origin
   // if behind shock, set back to upstream state
   //if(x(0) <= thresh+.5)
   {
      theta = 0;
      Ma = Mai;
      rho = rhoi;
      press = prsi;
      a = sqrt(euler::gamma*press/rho);
   }

   u(0) = .99*rho;
   u(1) = .99*rho*a*Ma*cos(theta);
   u(2) = .99*rho*a*Ma*sin(theta);
   u(3) = .99*(press/euler::gami + 0.5*rho*a*a*Ma*Ma);
   
   // u(0) = 1;
   // u(1) = 0;
   // u(2) = 0;
   // u(3) = 1;
}

// Exact solution; 
void uexact(const Vector &x, Vector& u)
{
   u.SetSize(4);
   double Mai = 2.4; //Ma1
   double rhoi = 1.0; //rho1
   double prsi = 1.0/euler::gamma;
   //assuming theta = 25 degrees, Ma1 = 2.4
   double theta = 25*2*M_PI/360;
   double beta =  52.17187440*2*M_PI/360; 
   //taken from Figure 9.9, Anderson for theta = 25 degrees, Ma1 = 2.4
   
   //compute mach number downstream of shock
   double Ma1n = Mai*sin(beta);
   double Ma2n = sqrt((1+(.5*euler::gami)*Ma1n*Ma1n) /
                     (euler::gamma*Ma1n*Ma1n - .5*euler::gami));
   double Ma = Ma2n/sin(beta-theta);
   
   //compute other quantities using continuity, momentum, and energy equations
   double rho = rhoi*(euler::gamma+1)*Ma1n*Ma1n / 
                  (2+euler::gami*Ma1n*Ma1n);
   double press = prsi*(1 + (2*euler::gamma/(euler::gamma+1))*(Ma1n*Ma1n - 1)); 
   double a = sqrt(euler::gamma*press/rho);

   
   double thresh = x(1)/tan(beta); //assuming wedge tip is origin
   // if behind shock, set back to upstream state
   if(x(0) <= thresh+.5)
   {
      theta = 0;
      Ma = Mai;
      rho = rhoi;
      press = prsi;
      a = sqrt(euler::gamma*press/rho);
   }

   u(0) = rho;
   u(1) = rho*a*Ma*cos(theta);
   u(2) = rho*a*Ma*sin(theta);
   u(3) = press/euler::gami + 0.5*rho*a*a*Ma*Ma;
   
   // u(0) = 1;
   // u(1) = 0;
   // u(2) = 0;
   // u(3) = 1;
}

