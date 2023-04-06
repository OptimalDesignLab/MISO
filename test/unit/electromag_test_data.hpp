#ifndef ELECTROMAG_TEST_DATA
#define ELECTROMAG_TEST_DATA

#include <limits>
#include <random>
#include <vector>

#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "coefficient.hpp"

namespace electromag_data
{
// define the random-number generator; uniform between -1 and 1
static std::default_random_engine gen;
static std::uniform_real_distribution<double> uniform_rand(-1.0,1.0);

double randNumber()
{
   return uniform_rand(gen);
}

void randBaselineVectorPert(const mfem::Vector &x, mfem::Vector &u)
{
   const double scale = 0.5;
   for (int i = 0; i < u.Size(); ++i)
   {
      u(i) = (2.0 + scale*uniform_rand(gen));
   }
}

double randState(const mfem::Vector &x)
{
   return 2.0 * uniform_rand(gen) - 1.0;
}

void randVectorState(const mfem::Vector &x, mfem::Vector &u)
{
   // std::cout << "u size: " << u.Size() << std::endl;
   for (int i = 0; i < u.Size(); ++i)
   {
      // std::cout << i << std::endl;
      u(i) = uniform_rand(gen);
      // u(i) = 1/sqrt(3);
      // u(i) = 1.0;
   }
}

void mag_func(const mfem::Vector &x, mfem::Vector &y)
{
   y = 1.0;
}

void vectorFunc(const mfem::Vector &x, mfem::Vector &y)
{
   y.SetSize(3);
   y(0) = x(0)*x(0) - x(1);
   y(1) = x(0) * exp(x(1));
   y(2) = x(2)*x(0) - x(1);
}

void vectorFuncRevDiff(const mfem::Vector &x, const mfem::Vector &v_bar, mfem::Vector &x_bar)
{
   x_bar(0) = v_bar(0) * 2*x(0) + v_bar(1) * exp(x(1)) + v_bar(2)*x(2);
   x_bar(1) = -v_bar(0) + v_bar(1) * x(0) * exp(x(1)) - v_bar(2); 
   x_bar(2) = v_bar(2) * x(0); 
}

void vectorFunc2(const mfem::Vector &x, mfem::Vector &y)
{
   y.SetSize(3);
   y(0) = sin(x(0))*x(2)*x(2);
   y(1) = x(1) - x(0)*x(2);
   y(2) = sin(x(1))*exp(x(2));
}

void vectorFunc2RevDiff(const mfem::Vector &x, const mfem::Vector &v_bar, mfem::Vector &x_bar)
{
   x_bar(0) = cos(x(0))*x(2)*x(2)*v_bar(0) - x(2)*v_bar(1);
   x_bar(1) = v_bar(1) + cos(x(1))*exp(x(2))*v_bar(2); 
   x_bar(2) = 2*sin(x(0))*x(2)*v_bar(0) - x(0)*v_bar(1) + sin(x(1))*exp(x(2))*v_bar(2);
}

/// Simple linear coefficient for testing CurlCurlNLFIntegrator. Also using for ACLFI, ACLFDI
class LinearCoefficient : public mach::StateCoefficient
{
public:
   LinearCoefficient(double val = 1.0) : value(val) {}

   double Eval(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               const double state) override
   {
      return value;
   }

   double EvalStateDeriv(mfem::ElementTransformation &trans,
                        const mfem::IntegrationPoint &ip,
                        const double state) override
   {
      return 0.0;
   }

private:
   double value;
};

/// Simple nonlinear coefficient for testing CurlCurlNLFIntegrator. Also using for DCLFI, DCLFDI
class NonLinearCoefficient : public mach::StateCoefficient
{
public:
   NonLinearCoefficient(double exponent = -0.5) : exponent(exponent) {}

   double Eval(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               const double state) override
   {
      // mfem::Vector state;
      // stateGF->GetVectorValue(trans.ElementNo, ip, state);
      // double state_mag = state.Norml2();
      // return pow(state, 2.0);
      // return state;
      // return 0.5*pow(state+1, -0.5);
      return 0.5*pow(state+1, exponent);
   }

   double EvalStateDeriv(mfem::ElementTransformation &trans,
                         const mfem::IntegrationPoint &ip,
                         const double state) override
   {
      // mfem::Vector state;
      // stateGF->GetVectorValue(trans.ElementNo, ip, state);
      // double state_mag = state.Norml2();
      // return 2.0*pow(state, 1.0);
      // return 1.0;
      // return -0.25*pow(state+1, -1.5);
      return 0.5*(exponent)*pow(state+1, exponent-1);
   }

   double EvalState2ndDeriv(mfem::ElementTransformation &trans,
                            const mfem::IntegrationPoint &ip,
                            const double state) override
   {
      // return 2.0;
      // return 0.0;
      // return 0.375*pow(state+1, -2.5);
      return 0.5*(exponent)*(exponent-1)*pow(state+1, exponent-2);
   }
private:
   double exponent;
};

/// Simple linear coefficient (vectorstatecoefficient analog) for testing IEAggDemag 
class LinearVectorCoefficient : public mach::VectorStateCoefficient
{
public:
   LinearVectorCoefficient(int dim, double val = 1.0) : mach::VectorStateCoefficient(dim), value(val) {}
   void Eval(mfem::Vector &vec, 
            mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               const double state) override
   {
      vec = value; // independent of state value
   }

   void EvalRevDiff(const mfem::Vector &V_bar,
                    mfem::ElementTransformation &trans,
                    const mfem::IntegrationPoint &ip,
                    mfem::DenseMatrix &PointMat_bar) override
   {
      // Not using, just need to "implement" because it is virtual in the base class
   }

private:
   int dim;
   double value;
};

/// Simple nonlinear coefficient (vectorstatecoefficient analog) for testing IEAggDemag 
class NonLinearVectorCoefficient : public mach::VectorStateCoefficient
{
public:
   NonLinearVectorCoefficient(int dim, double exponent = -0.5) : mach::VectorStateCoefficient(dim), exponent(exponent) {}   
   
   void Eval(mfem::Vector &vec, 
            mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               const double state) override
   {
      vec = 0.5*pow(state+1, exponent);
   }

   void EvalRevDiff(const mfem::Vector &V_bar,
                    mfem::ElementTransformation &trans,
                    const mfem::IntegrationPoint &ip,
                    mfem::DenseMatrix &PointMat_bar) override
   {
      // Not using, just need to "implement" because it is virtual in the base class
   }

private:
   int dim;
   double exponent;
};

/*** No longer using SigmaCoefficient (unnecessary)
/// Simple coefficient for conductivity for testing resistivity (and DCLFIMS)
class SigmaCoefficient : public mach::StateCoefficient
{
public:
   SigmaCoefficient(double alpha_resistivity = 3.8e-3, 
                    double T_ref = 293.15,
                    double sigma_T_ref = 5.6497e7) 
   : alpha_resistivity(alpha_resistivity), T_ref(T_ref), sigma_T_ref(sigma_T_ref) {}

   double Eval(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               const double state) override
   {
      // Logic from ConductivityCoefficient
      return sigma_T_ref/(1+alpha_resistivity*(state-T_ref));
   }

   double EvalStateDeriv(mfem::ElementTransformation &trans,
                         const mfem::IntegrationPoint &ip,
                         const double state) override
   {
      // Logic from ConductivityCoefficient
      return (-sigma_T_ref*alpha_resistivity)/std::pow(1+alpha_resistivity*(state-T_ref),2);

   }

   double EvalState2ndDeriv(mfem::ElementTransformation &trans,
                            const mfem::IntegrationPoint &ip,
                            const double state) override
   {
      // Logic from ConductivityCoefficient
      return (2*sigma_T_ref*std::pow(alpha_resistivity,2))/std::pow(1+alpha_resistivity*(state-T_ref),3);
   }

   // Implementing method for the purposes of DCLossFunctionalIntegratorMeshSens test
   void EvalRevDiff(const double Q_bar,
                    mfem::ElementTransformation &trans,
                    const mfem::IntegrationPoint &ip,
                    mfem::DenseMatrix &PointMat_bar) override
   {
      // Implementing method for the purposes of DCLossFunctionalIntegratorMeshSens test
      // Trivial implementation (MFEM just needs method to be present so it can call it)
      ///FormerTODO: Make an actual implementation (nonzero implementation). Currently, PointMat_bar is not being updated
      
      // Using the old SteinmetzCoefficient::EvalRevDiff as inspiration

      ///FormerTODO: Figure out how to get the element nodes so can pass them in to loop below (the below is not it)
      // double *p;
      // const int dim=PointMat_bar.Height();
      // ip.Get(p, dim);
      // std::cout << "p=[" << p[0];
      // std::cout << "," << p[1] << ","; 
      // std::cout << p[2] << "\n";

      // Evaluate the reverse mode derivative of sigma w/r/t the mesh nodes
      for (int d = 0; d < PointMat_bar.Height(); ++d)
      {
         for (int j = 0; j < PointMat_bar.Width(); ++j)
         {
            
            // std::cout << "nodes(" << d << "," << j << " = " << nodes(d,j) << "\n";

            ///FormerTODO: Assuming have the node locations, calculate the value for EvalRevDiff
            // double dT_nodedx = 0;
            // double T_node=37;
            // PointMat_bar(d,j)+=(-sigma_T_ref*alpha_resistivity*dT_nodedx)/std::pow(1+alpha_resistivity*(T_node-T_ref),2);
         }
      }

      // PointMat_bar+=0;
      // PointMat_bar+=(-sigma_T_ref*alpha_resistivity)/std::pow(1+alpha_resistivity*(state-T_ref),2);
      
   } 
private:
   double alpha_resistivity;
   double T_ref;
   double sigma_T_ref;
};
*/

/// Simple two state coefficient for testing PMDemagIntegrator
class SimpleTwoStateCoefficient : public mach::TwoStateCoefficient
{
public:
   SimpleTwoStateCoefficient() {}; 

   double Eval(mfem::ElementTransformation &trans,
                              const mfem::IntegrationPoint &ip,
                              const double state1,
                              const double state2) override
   {
      // Assuming state1=flux density and state2=temperature
      auto B = state1;
      auto T = state2;

      auto C_BT = pow(B,2)*T;
      return C_BT;
   }

   double EvalDerivS1(mfem::ElementTransformation &trans,
                              const mfem::IntegrationPoint &ip,
                              const double state1,
                              const double state2) override
   {
      // Assuming state1=flux density and state2=temperature
      // First derivative with respect to flux density

      auto B = state1;
      auto T = state2;

      // auto C_BT = pow(B,2)*T;
      auto dC_BTdB = 2.0 * B * T;
      return dC_BTdB;
   }

   double EvalDerivS2(mfem::ElementTransformation &trans,
                              const mfem::IntegrationPoint &ip,
                              const double state1,
                              const double state2) override
   {
      // Assuming state1=flux density and state2=temperature
      // First derivative with respect to temperature

      auto B = state1;
      // auto T = state2;

      // auto C_BT = pow(B,2)*T;
      auto dC_BTdT = pow(B,2);
      return dC_BTdT;
   }

   double Eval2ndDerivS1(mfem::ElementTransformation &trans,
                              const mfem::IntegrationPoint &ip,
                              const double state1,
                              const double state2) override
   {
      // Assuming state1=flux density and state2=temperature
      // Second derivative with respect to flux density

      // auto B = state1;
      auto T = state2;

      // auto C_BT = pow(B,2)*T;
      // auto dC_BTdB = 2 * B * T;
      auto d2C_BTdB2 = 2.0 * T; 
      return d2C_BTdB2;
   }

   double Eval2ndDerivS2(mfem::ElementTransformation &trans,
                              const mfem::IntegrationPoint &ip,
                              const double state1,
                              const double state2) override
   {
      // Assuming state1=flux density and state2=temperature
      // Second derivative with respect to temperature

      // auto B = state1;
      // auto T = state2;

      // auto C_BT = pow(B,2)*T;
      // auto dC_BTdT = pow(B,2);
      auto d2C_BTdT2 = 0.0;
      return d2C_BTdT2;
   }

   double Eval2ndDerivS1S2(mfem::ElementTransformation &trans,
                              const mfem::IntegrationPoint &ip,
                              const double state1,
                              const double state2) override
   {
      // Assuming state1=flux density and state2=temperature
      // Derivative with respect to flux density then temperature

      auto B = state1;
      // auto T = state2;

      // auto C_BT = pow(B,2)*T;
      // auto dC_BTdB = 2.0 * B * T;
      auto d2C_BTdBdT = 2.0 * B;
      return d2C_BTdBdT;
   }

   ///TODO: Likely not necessary because of Eval2ndDerivS2S1
   double Eval2ndDerivS2S1(mfem::ElementTransformation &trans,
                              const mfem::IntegrationPoint &ip,
                              const double state1,
                              const double state2) override
   {
      // Assuming state1=flux density and state2=temperature
      // Derivative with respect to temperature then flux density

      auto B = state1;
      // auto T = state2;

      // auto C_BT = pow(B,2)*T;
      // auto dC_BTdT = pow(B,2);      
      auto d2C_BTdTdB = 2.0 * B;
      return d2C_BTdTdB;
   }
};



/// Simple coefficient for CAL2 coefficients for testing CAL2CoreLossIntegrator
// Can represent either CAL2_kh coefficients or CAL2_ke coefficients
class CAL2Coefficient : public mach::ThreeStateCoefficient
{
public:
   // CAL2Coefficient(double &T0, 
   //                  std::vector<double> &k_T0,
   //                  double &T1,
   //                  std::vector<double> &k_T1) 
   // : T0(T0), k_T0(k_T0), T1(T1), k_T1(k_T1) {}
   CAL2Coefficient() {};

   double Eval(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
               double state1,
               double state2,
               double state3) override
   {      
      // Assuming state1=temperature, state2=frequency, state3=max alternating flux density
      auto T = state1;
      // auto f = state2;
      auto Bm = state3;

      return T*pow(Bm,-1.0);
   }

   double EvalDerivS1(mfem::ElementTransformation &trans,
                           const mfem::IntegrationPoint &ip,
                           const double state1,
                           const double state2,
                           const double state3) override
   {
      // Assuming state1=temperature, state2=frequency, state3=max alternating flux density
      // auto T = state1;
      // auto f = state2;
      auto Bm = state3;

      // return T*pow(Bm,-1.0);
      return pow(Bm,-1.0);
   }

   double EvalDerivS2(mfem::ElementTransformation &trans,
                              const mfem::IntegrationPoint &ip,
                              const double state1,
                              const double state2,
                              const double state3) override
   {
      // Assuming state1=temperature, state2=frequency, state3=max alternating flux density
      // auto T = state1;
      // auto f = state2;
      // auto Bm = state3;

      // return T*pow(Bm,-1.0);
      return 0;
   }

   double EvalDerivS3(mfem::ElementTransformation &trans,
                              const mfem::IntegrationPoint &ip,
                              const double state1,
                              const double state2,
                              const double state3) override
   {
      // Assuming state1=temperature, state2=frequency, state3=max alternating flux density
      auto T = state1;
      // auto f = state2;
      auto Bm = state3;

      // return T*pow(Bm,-1.0);
      return -T*pow(Bm,-2.0);
   }

   double Eval2ndDerivS1(mfem::ElementTransformation &trans,
                              const mfem::IntegrationPoint &ip,
                              const double state1,
                              const double state2,
                              const double state3) override
   {
      // Assuming state1=temperature, state2=frequency, state3=max alternating flux density
      // auto T = state1;
      // auto f = state2;
      // auto Bm = state3;

      // return T*pow(Bm,-1.0);
      return 0;
   }

   double Eval2ndDerivS2(mfem::ElementTransformation &trans,
                              const mfem::IntegrationPoint &ip,
                              const double state1,
                              const double state2,
                              const double state3) override
   {
      // Assuming state1=temperature, state2=frequency, state3=max alternating flux density
      // auto T = state1;
      // auto f = state2;
      // auto Bm = state3;

      // return T*pow(Bm,-1.0);
      return 0;
   }

   double Eval2ndDerivS3(mfem::ElementTransformation &trans,
                              const mfem::IntegrationPoint &ip,
                              const double state1,
                              const double state2,
                              const double state3) override
   {
      // Assuming state1=temperature, state2=frequency, state3=max alternating flux density
      auto T = state1;
      // auto f = state2;
      auto Bm = state3;

      // return T*pow(Bm,-1.0);
      return 2.0*T*pow(Bm,-3.0);
   }

   double Eval2ndDerivS1S2(mfem::ElementTransformation &trans,
                              const mfem::IntegrationPoint &ip,
                              const double state1,
                              const double state2,
                              const double state3) override
   {
      // Assuming state1=temperature, state2=frequency, state3=max alternating flux density
      // auto T = state1;
      // auto f = state2;
      // auto Bm = state3;

      // return T*pow(Bm,-1.0);
      return 0;
   }

   double Eval2ndDerivS1S3(mfem::ElementTransformation &trans,
                              const mfem::IntegrationPoint &ip,
                              const double state1,
                              const double state2,
                              const double state3) override
   {
      // Assuming state1=temperature, state2=frequency, state3=max alternating flux density
      // auto T = state1;
      // auto f = state2;
      auto Bm = state3;

      // return T*pow(Bm,-1.0);
      return -1.0*pow(Bm,-2.0);
   }

   double Eval2ndDerivS2S3(mfem::ElementTransformation &trans,
                              const mfem::IntegrationPoint &ip,
                              const double state1,
                              const double state2,
                              const double state3) override
   {
      // Assuming state1=temperature, state2=frequency, state3=max alternating flux density
      // auto T = state1;
      // auto f = state2;
      // auto Bm = state3;

      // return T*pow(Bm,-1.0);
      return 0;
   }

   ///TODO: Likely not necessary because of Eval2ndDerivS1S2
   double Eval2ndDerivS2S1(mfem::ElementTransformation &trans,
                              const mfem::IntegrationPoint &ip,
                              const double state1,
                              const double state2,
                              const double state3) override
   {
      // Assuming state1=temperature, state2=frequency, state3=max alternating flux density
      // auto T = state1;
      // auto f = state2;
      // auto Bm = state3;

      // return T*pow(Bm,-1.0);
      return 0;
   }

   ///TODO: Likely not necessary because of Eval2ndDerivS1S3
   double Eval2ndDerivS3S1(mfem::ElementTransformation &trans,
                              const mfem::IntegrationPoint &ip,
                              const double state1,
                              const double state2,
                              const double state3) override
   {
      // Assuming state1=temperature, state2=frequency, state3=max alternating flux density
      // auto T = state1;
      // auto f = state2;
      auto Bm = state3;

      // return T*pow(Bm,-1.0);
      return -1.0*pow(Bm,-2.0);
   }

   ///TODO: Likely not necessary because of Eval2ndDerivS2S3
   double Eval2ndDerivS3S2(mfem::ElementTransformation &trans,
                              const mfem::IntegrationPoint &ip,
                              const double state1,
                              const double state2,
                              const double state3) override
   {
      // Assuming state1=temperature, state2=frequency, state3=max alternating flux density
      // auto T = state1;
      // auto f = state2;
      // auto Bm = state3;

      // return T*pow(Bm,-1.0);
      return 0;
   }

private:
   // double &T0;
   // std::vector<double> &k_T0;
   // double &T1;
   // std::vector<double> &k_T1;
};

nlohmann::json getBoxOptions(int order)
{
   nlohmann::json box_options = {
      {"silent", true},
      {"space-dis", {
         {"basis-type", "nedelec"},
         {"degree", order}
      }},
      {"steady", true},
      {"lin-solver", {
         {"type", "hypregmres"},
         {"pctype", "hypreams"},
         {"printlevel", -1},
         {"maxiter", 100},
         {"abstol", 1e-10},
         {"reltol", 1e-14}
      }},
      {"adj-solver", {
         {"type", "hypregmres"},
         {"pctype", "hypreams"},
         {"printlevel", -1},
         {"maxiter", 100},
         {"abstol", 1e-10},
         {"reltol", 1e-14}
      }},
      {"newton", {
         {"printlevel", -1},
         {"reltol", 1e-10},
         {"abstol", 0.0}
      }},
      {"components", {
         {"attr1", {
            {"material", "box1"},
            {"attr", 1},
            {"linear", true}
         }},
         {"attr2", {
            {"material", "box2"},
            {"attr", 2},
            {"linear", true}
         }}
      }},
      {"problem-opts", {
         {"fill-factor", 1.0},
         {"current_density", 1.0},
         {"current", {
            {"box1", {1}},
            {"box2", {2}}
         }},
         {"box", true}
      }},
      {"outputs", {
         {"co-energy", {""}}
      }}
   };
   return box_options;
}

std::unique_ptr<mfem::Mesh> getMesh(int nxy = 2, int nz = 2)
{
   using namespace mfem;
   // generate a simple tet mesh
   std::unique_ptr<Mesh> mesh(
      new Mesh(Mesh::MakeCartesian3D(nxy, nxy, nz,
                                     Element::TETRAHEDRON,
                                     1.0, 1.0, (double)nz / (double)nxy, true)));
   mesh->EnsureNodes();

   // assign attributes to top and bottom sides
   for (int i = 0; i < mesh->GetNE(); ++i)
   {
      Element *elem = mesh->GetElement(i);

      Array<int> verts;
      elem->GetVertices(verts);

      bool below = true;
      for (int i = 0; i < 4; ++i)
      {
         auto vtx = mesh->GetVertex(verts[i]);
         if (vtx[1] <= 0.5)
         {
            below = below & true;
         }
         else
         {
            below = below & false;
         }
      }
      if (below)
      {
         elem->SetAttribute(1);
      }
      else
      {
         elem->SetAttribute(2);
      }
   }
   return mesh;
}

#ifdef MFEM_USE_PUMI
nlohmann::json getWireOptions(int order)
{
   nlohmann::json wire_options = {
      {"silent", false},
      {"mesh", {
         {"file", "cut_wire.smb"},
         {"model-file", "cut_wire.egads"}
      }},
      {"space-dis", {
         {"basis-type", "nedelec"},
         {"degree", order}
      }},
      {"steady", true},
      {"lin-solver", {
         {"type", "hypregmres"},
         {"pctype", "hypreams"},
         {"printlevel", -1},
         {"maxiter", 100},
         {"abstol", 1e-10},
         {"reltol", 1e-14}
      }},
      {"adj-solver", {
         {"type", "hypregmres"},
         {"pctype", "hypreams"},
         {"printlevel", -1},
         {"maxiter", 100},
         {"abstol", 1e-10},
         {"reltol", 1e-14}
      }},
      {"newton", {
         {"printlevel", -1},
         {"reltol", 1e-10},
         {"abstol", 0.0}
      }},
      {"components", {
         {"wire", {
            {"material", "copperwire"},
            {"attrs", {1, 3, 4, 5}},
            {"linear", true}
         }},
         {"farfields", {
            {"material", "air"},
            {"attrs", {2, 6, 7, 8}},
            {"linear", true}
         }}
      }},
      {"problem-opts", {
         {"fill-factor", 1.0},
         {"current_density", 10000.0},
         {"current", {
            {"z", {1, 3, 4, 5}},
         }},
         {"box", true}
      }},
      {"outputs", {
         {"co-energy", {""}}
      }}
   };
   return wire_options;
}
#endif

} // namespace electromag_data

#endif
