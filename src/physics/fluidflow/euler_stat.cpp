#include <fstream>

#include "euler.hpp"
#include "evolver.hpp"
#include "gauss_hermite.hpp"

using namespace std;
using namespace mfem;

namespace mach
{

template<int dim, bool entvar>
void EulerSolver<dim, entvar>::calcStatistics()
{
   std::string type = options["statistics"]["type"].get<std::string>();
  
   //get uncertain parameters
   auto &params = options["statistics"]["param"];
   auto &outputs = options["outputs"];
   double pmean; double pstdv;
   double mean; double stdev; int order;
   if (params.find("mach") != params.end())
   {
      auto tmp = params["mach"].get<vector<double>>();
      pmean = tmp[0]; pstdv = tmp[1];
   }
   Vector qfar(dim+2);

   //get scheme
   if (type == "collocation")
   {
      order = options["statistics"]["order"].get<int>();
      Vector abs(order); Vector wt(order); Vector pts(order); Vector eval(order); Vector meansq(order);
      switch (order)
      {
         case 1:
            abs = gho1_x; wt = gho1_w;
            break;
         case 2:
            abs = gho2_x; wt = gho2_w;
            break;
         case 3:
            abs = gho3_x; wt = gho3_w;
            break;
         case 4:
            abs = gho4_x; wt = gho4_w;
            break;
         case 5:
            abs = gho5_x; wt = gho5_w;
            break;
         case 6:
            abs = gho6_x; wt = gho6_w;
            break;
         default:
            mfem_error("Gauss-Hermite collocation is currently only supported for 1 <= order <= 6");
            break;
      }

      //compute realizations (1D)
      for(int i = 0; i < order; i++)
      {
         if (params.find("mach") != params.end())
         {
            mach_fs = pmean + sqrt(2.0)*pstdv*abs(i);
         }

	      initDerived();
         constructLinearSolver(options["lin-solver"]);
	      constructNewtonSolver();
	      constructEvolver();
         HYPRE_ClearAllErrors();
         getFreeStreamState(qfar);
	      setInitialCondition(qfar);
         solveForState();
         std::cout << "Solver Done" << std::endl;
         if (outputs.find("drag") != outputs.end())
         {
            string drags = "drag";
            eval(i) = calcOutput(drags);
         }
         meansq(i) = eval(i)*eval(i);
      }

      //print realizations
      stringstream evalname;
      evalname << "realization_outs_"<<pmean<<"_"<<pstdv<<"_o"<<order<<".txt";
      std::ofstream evalfile(evalname.str());
      evalfile.precision(18);
      eval.Print(evalfile);

      mean = eval*wt;
      stdev = sqrt(meansq*wt - mean*mean);

      cout << "Stochastic Collocation Order "<<order<<endl;
   }
   else if (type == "MM1")
   {
      //compute realization at param mean
      if (params.find("mach") != params.end())
      {
         mach_fs = pmean;
      }
	   initDerived();
      constructLinearSolver(options["lin-solver"]);
	   constructNewtonSolver();
	   constructEvolver();
      HYPRE_ClearAllErrors();
      getFreeStreamState(qfar);
	   setInitialCondition(qfar);
      solveForState();
      std::cout << "Solver Done" << std::endl;
      double d1 = getParamSens();

      //compute mean
      if (outputs.find("drag") != outputs.end())
      {
         string drags = "drag";
         mean = calcOutput(drags);
      }

      //compute standard deviation
      stdev = pstdv*d1; //sqrt(pstdv*d1);?

      cout << "Moment Method Order 1"<<endl;
   }
   else if (type == "cubicpoly")
   {
      order = options["statistics"]["order"].get<int>();
      Vector abs(order); Vector wt(order); Vector pts(order); Vector eval(order); Vector meansq(order);
      double p[2]; double pd[2];
      switch (order)
      {
         case 6:
            abs = gho6_x; wt = gho6_w;
            break;
         default:
            mfem_error("Gauss-Hermite collocation is currently only supported for 1 <= order <= 6");
            break;
      }

      //compute realizations to approximate cubic polynomial(1D)
      for(int i = 0; i < 2; i++)
      {
         if (params.find("mach") != params.end())
         {
            mach_fs = pmean + sqrt(2.0)*pstdv*abs(1+3*i); //only for order = 6 for now
         }

	      initDerived();
         constructLinearSolver(options["lin-solver"]);
	      constructNewtonSolver();
	      constructEvolver();
         HYPRE_ClearAllErrors();
         getFreeStreamState(qfar);
	      setInitialCondition(qfar);
         solveForState();
         std::cout << "Solver Done" << std::endl;
         if (outputs.find("drag") != outputs.end())
         {
            string drags = "drag";
            p[i] = calcOutput(drags);
         }
         pd[i] = getParamSens();
      }
      //print points and derivatives
      stringstream pointname;
      pointname << "point_outs_"<<pmean<<"_"<<pstdv<<"_o"<<order<<".txt";
      std::ofstream pointfile(pointname.str());
      pointfile.precision(18);
      pointfile << p[0] << " " << pd[0] <<"\n"<<p[1] << " " << pd[1];

      //construct polynomial
      Vector crhs(4); crhs(0) = p[0]; crhs(1) = pd[0]; crhs(2) = p[1]; crhs(3) = pd[1];
      DenseMatrix coeffs(4); Vector c(4);
      for(int j = 0; j < 2; j++)
      {
         double point = pmean + sqrt(2.0)*pstdv*abs(1+3*j);
         coeffs(j*2, 0) = point*point*point;
         coeffs(j*2, 1) = point*point;
         coeffs(j*2, 2) = point;
         coeffs(j*2, 3) = 1;
         coeffs(j*2+1, 0) = 3*point*point;
         coeffs(j*2+1, 1) = 2*point;
         coeffs(j*2+1, 2) = 1;
         coeffs(j*2+1, 3) = 0;
      }
      coeffs.Invert();
      coeffs.Mult(crhs, c);

      //print coefficients
      stringstream cname;
      cname << "realization_poly_coeffs_"<<pmean<<"_"<<pstdv<<"_o"<<order<<".txt";
      std::ofstream cfile(cname.str());
      cfile.precision(18);
      c.Print(cfile);

      //compute approximate realizations
      for(int i = 0; i < order; i++)
      {
         double point = pmean + sqrt(2.0)*pstdv*abs(i);
         eval(i) = c(0)*point*point*point + c(1)*point*point + c(2)*point + c(3);
         meansq(i) = eval(i)*eval(i);
      }      

      //print realizations
      stringstream evalname;
      evalname << "realization_poly_outs_"<<pmean<<"_"<<pstdv<<"_o"<<order<<".txt";
      std::ofstream evalfile(evalname.str());
      evalfile.precision(18);
      eval.Print(evalfile);

      mean = eval*wt;
      stdev = sqrt(meansq*wt - mean*mean);

      cout << "Cubic Polynomial Collocation Order "<<order<<endl;
   }

   cout << "Mean: "<<mean<<endl;
   cout << "Standard Deviation: "<<stdev<<endl;

   //write to file
   stringstream statname;
   statname << "euler_stats_"<<type<<"_"<<pmean<<"_"<<pstdv;
   if(type == "collocation" || "cubicpoly")
      statname << "_o"<<order;
   statname <<".txt";
   std::ofstream statfile(statname.str());
   statfile.precision(18);
   statfile << mean << "\n" << stdev;
}

}