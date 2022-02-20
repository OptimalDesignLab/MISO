#include <random>

#include "catch.hpp"
#include "mfem.hpp"
#include "adept.h"

#include "flow_control_residual.hpp"
#include "flow_residual.hpp"
#include "sbp_fe.hpp"
#include "euler_fluxes.hpp"
#include "mach_input.hpp"

using namespace std;
using namespace mfem;
using namespace mach;

auto options = R"(
{
   "flow-param": {
      "viscous": false,
      "entropy-state": false,
      "mu": -1.0,
      "Re": 100,
      "Pr": 0.72,
      "mach": 0.5,
      "aoa": 0.0,
      "roll-axis": 0,
      "pitch-axis": 1
   },
   "space-dis": {
      "degree": 1,
      "lps-coeff": 1.0,
      "iface-coeff": 0.0,
      "basis-type": "csbp",
      "flux-fun": "Euler"
   },
   "bcs": {
      "far-field": [0, 1, 1, 1],
      "control": [1, 0, 0, 0]
   },
   "lin-prec": {
      "type": "hypreilu",
      "lev-fill": 1,
      "ilu-type": 0,
      "ilu-reorder": 1,
      "printlevel": 0
   },
   "outputs":
      { 
         "boundary-entropy": {
            "boundaries": [1, 0, 0, 0]
         }
      }
})"_json;

TEST_CASE("ControlResidual construction and evaluation", "[ControlResidual]")
{
   // construct the residual
   ControlResidual res(MPI_COMM_WORLD, options);
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   int num_var = getSize(res);
   if (rank == 0)
   {
      REQUIRE(num_var == 2);
   }
   else
   {
      REQUIRE(num_var == 0);
   }

   // Define parameters for a nominal control
   const double Kp = 0.4, Ti = 0.8, Td = 0.5, beta = 2.5, eta = 0.8;
   const double target_entropy = 0.0, boundary_entropy = 1.0;
   bool closed_loop = true;
   Vector P(4);
   double sigma = -0.5*(beta*Ti + eta*Td);
   double rho = beta*eta*Ti*Td - sigma*(beta*Ti + eta*Td);
   P(0) = rho;
   P(1) = sigma;
   P(2) = sigma;
   P(3) = 1.0;
   auto inputs = MachInputs({{"Kp", Kp},
                             {"Ti", Ti},
                             {"Td", Td},
                             {"beta", beta},
                             {"eta", eta},
                             {"target-entropy", target_entropy},
                             {"boundary-entropy", boundary_entropy},
                             {"closed-loop", float(closed_loop)}});
   setInputs(res, inputs);

   // evaluate the residual at an arbitrary state
   Vector x(num_var);
   std::default_random_engine gen(std::random_device{}());
   std::uniform_real_distribution<double> uniform_rand(0.0, 1.0);
   for (int i = 0; i < num_var; ++i)
   {
      x(i) = uniform_rand(gen);
   }
   double time = uniform_rand(gen);
   inputs = MachInputs({{"state", x}, {"time", time}});
   Vector res_vec(num_var);
   evaluate(res, inputs, res_vec);
   if (rank == 0)
   {
      Vector r(2);
      double fac = 1 / (beta * eta * Ti * Td);
      r(0) = fac * x(1);
      r(1) = -x(0) + fac * (beta * Ti + eta * Td) * x(1);
      double error = target_entropy - boundary_entropy;
      double scaled_error = error * Kp / (eta * Ti * Td);
      r(0) += (1.0 / (eta * beta) - 1.0) * scaled_error;
      r(1) +=
          ((beta * Ti + eta * Td) / (eta * beta) - (Ti + Td)) * scaled_error;

      REQUIRE(res_vec(0) == Approx(r(0)).margin(1e-14));
      REQUIRE(res_vec(1) == Approx(r(1)).margin(1e-14));
   }
   else
   {
      REQUIRE( res_vec.Size() == 0 );
   }

   // evaluate the Jacobian; perform matrix-vector product on random vector
   Operator &Jac = getJacobian(res, inputs, "state");
   Vector v(num_var), Jac_v(num_var);
   for (int i = 0; i < num_var; ++i)
   {
      v(i) = uniform_rand(gen);
   }
   Jac.Mult(v, Jac_v);
   if (rank == 0)
   {
      double fac = 1/(beta*eta*Ti*Td);
      Vector prod(2);
      prod(0) = fac*v(1);
      prod(1) = -1.0*v(0) + fac*(beta*Ti + eta*Td)*v(1);

      REQUIRE( Jac_v(0) == Approx(prod(0)).margin(1e-14) );
      REQUIRE( Jac_v(1) == Approx(prod(1)).margin(1e-14) );
   }

   // get the Preconditioner, which should be the exact inverse here    
   Vector w(num_var);
   Solver * prec = getPreconditioner(res, options["lin-prec"]);
   prec->SetOperator(Jac);
   prec->Mult(Jac_v, w);
   if (rank == 0)
   {
      REQUIRE( w(0) == Approx(v(0)).margin(1e-14) );
      REQUIRE( w(1) == Approx(v(1)).margin(1e-14) );
   }

   // check the entropy 
   double entropy = calcEntropy(res, inputs);
   if (rank == 0)
   {
      double ent = 0.5*(x(0)*P(0)*x(0) + 2.0*x(0)*P(1)*x(1) + x(1)*P(3)*x(1));
      REQUIRE(entropy == Approx(ent).margin(1e-14));
   }

   // check entropy change; `calcEntropyChange` uses `k = -state_dot` directly,
   // so since `k + res_vec = 0`, we need to scale res_vec by neg one.
   res_vec *= -1.0;
   inputs = MachInputs({{"state", x}, {"state_dot", res_vec}});
   double entropy_change = calcEntropyChange(res, inputs);
   if (rank == 0)
   {
      double dent = x(0) * P(0) * res_vec(0) + x(0) * P(1) * res_vec(1) +
                    x(1) * P(2) * res_vec(0) + x(1) * P(3) * res_vec(1);
      dent *= -1.0;
      REQUIRE(entropy_change == Approx(dent).margin(1e-14));
      REQUIRE(entropy_change > 0.0); // positive because res is on LHS
   }
}

TEST_CASE("FlowControlResidual construction and evaluation",
          "[FlowControlResidual]")
{
   const int dim = 2; // templating is hard here because mesh constructors
   int num_state = dim + 2;
   adept::Stack diff_stack;

   // generate a 8 element mesh and build the finite-element space
   int num_edge = 2;
   Mesh smesh(Mesh::MakeCartesian2D(num_edge, num_edge, Element::TRIANGLE,
                                    true /* gen. edges */, 1.0, 1.0, true));
   ParMesh mesh(MPI_COMM_WORLD, smesh);
   int p = options["space-dis"]["degree"].get<int>();
   SBPCollection fec(p, dim);
   ParFiniteElementSpace fespace(&mesh, &fec, num_state, Ordering::byVDIM);

   // construct the residuals
   MachResidual res(
       FlowControlResidual<dim, false>(options, fespace, diff_stack));
   FlowResidual<dim, false> flow_res(options, fespace, diff_stack);
   ControlResidual control_res(MPI_COMM_WORLD, options);
   int num_var = getSize(res);
   REQUIRE(num_var == getSize(flow_res) + getSize(control_res));

   // Define parameters for a nominal control
   const double Kp = 0.4, Ti = 0.8, Td = 0.5, beta = 2.5, eta = 0.8;
   const double target_entropy = 0.0;
   bool closed_loop = true;
   Vector P(4);
   double sigma = -0.5*(beta*Ti + eta*Td);
   double rho = beta*eta*Ti*Td - sigma*(beta*Ti + eta*Td);
   P(0) = rho;
   P(1) = sigma;
   P(2) = sigma;
   P(3) = 1.0;
   auto inputs = MachInputs({{"Kp", Kp},
                             {"Ti", Ti},
                             {"Td", Td},
                             {"beta", beta},
                             {"eta", eta},
                             {"target-entropy", target_entropy},
                             {"boundary-entropy", 0.0},
                             {"closed-loop", float(closed_loop)}});
   setInputs(res, inputs);
   setInputs(control_res, inputs);

   // We will evaluate the flow residual using a perturbed state, and the
   // control residual using a random state.
   Vector q(num_var);
   std::default_random_engine gen(std::random_device{}());
   std::uniform_real_distribution<double> uniform_rand(0.0, 1.0);
   double mach = options["flow-param"]["mach"].get<double>();
   double aoa = options["flow-param"]["aoa"].get<double>();
   int ptr = 0;
   for (int i = 0; i < getSize(control_res); ++i)
   {
      q(ptr) = uniform_rand(gen);
      ptr += 1;
   }
   for (int i = 0; i < getSize(flow_res) / num_state; ++i)
   {
      getFreeStreamQ<double, dim>(
          mach, aoa, 0, 1, q.GetData() + ptr + num_state * i);
      for (int j = 0; j < num_state; ++j)
      {
         q(ptr + i * num_state + j) += uniform_rand(gen) * 0.01;
      }
   }

   // Set up the inputs to the three residuals, and evaluate coupling variables
   Vector x_actuator({0.0, 0.5});
   MachOutput boundary_entropy(flow_res.constructOutput(
       "boundary-entropy", options["outputs"]["boundary-entropy"]));
   Vector flow_state(q.GetData() + ptr, getSize(flow_res));
   double time = 0.0;
   auto flow_inputs = MachInputs(
       {{"state", flow_state}, {"time", time}, {"x-actuator", x_actuator}});
   double bndry_ent = calcOutput(boundary_entropy, flow_inputs);
   auto control_inputs = MachInputs(
       {{"state", q}, {"time", time}, {"boundary-entropy", bndry_ent}});
   double vel_control = control_res.getControlVelocity(control_inputs);
   flow_inputs.emplace("control", vel_control);
   inputs =
       MachInputs({{"state", q}, {"time", time}, {"x-actuator", x_actuator}});

   Vector res_vec(num_var);
   Vector res_vec_control(getSize(control_res));
   Vector res_vec_flow(getSize(flow_res));
   evaluate(res, inputs, res_vec);
   evaluate(control_res, control_inputs, res_vec_control);
   evaluate(flow_res, flow_inputs, res_vec_flow);
   for (int i = 0; i < getSize(control_res); ++i)
   {
      REQUIRE( res_vec(i) == Approx(res_vec_control(i)).margin(1e-14) );
   }
   for (int i = 0; i < getSize(flow_res); ++i)
   {
      REQUIRE( res_vec(ptr+i) == Approx(res_vec_flow(i)).margin(1e-14) );
   }

   // check the entropy calculation for consistency between the compound and 
   // individual residuals
   double entropy = calcEntropy(res, inputs);
   double control_entropy = calcEntropy(control_res, control_inputs);
   double flow_entropy = calcEntropy(flow_res, flow_inputs);
   REQUIRE( entropy == Approx(control_entropy + flow_entropy).margin(1e-14) );

   // check for consistency between Jacobians 
   JacobianFree jac(res);
   jac.setState(inputs);
   //Operator &jac = getJacobian(res, inputs, "state");
   Operator &control_jac = getJacobian(control_res, control_inputs, "state");
   Operator &flow_jac = getJacobian(flow_res, flow_inputs, "state");

   // Check consistency with control Jacobian.
   Vector v(num_var), Jac_v(num_var);
   v = 0.0;
   for (int i = 0; i < getSize(control_res); ++i)
   {
      v(i) = uniform_rand(gen);
   }
   jac.Mult(v, Jac_v);
   Vector vc(v.GetData(), getSize(control_res)), Jac_vc(getSize(control_res));
   control_jac.Mult(vc, Jac_vc);
   for (int i = 0; i < getSize(control_res); ++i)
   {
      REQUIRE( Jac_v(i) == Approx(Jac_vc(i)).margin(1e-14) );
   }

   // Check that the extracted block Jacobian works 
   Operator &sub_jac = jac.getDiagonalBlock(0);
   sub_jac.Mult(vc, Jac_vc);
   for (int i = 0; i < getSize(control_res); ++i)
   {
      REQUIRE( Jac_v(i) == Approx(Jac_vc(i)).margin(1e-14) );
   }

   // Check consistency with the flow Jacobian 
   v = 0.0;
   for (int i = 0; i < getSize(flow_res); ++i)
   {
      v(ptr+i) = uniform_rand(gen);
   }
   jac.Mult(v, Jac_v);
   Vector vf(v.GetData()+ptr, getSize(flow_res)), Jac_vf(getSize(flow_res));
   flow_jac.Mult(vf, Jac_vf);
   for (int i = 0; i < getSize(flow_res); ++i)
   {
      REQUIRE( Jac_v(ptr+i) == Approx(Jac_vf(i)).margin(1e-5) );
   }

   // Check that the extracted block Jacobian works 
   Operator &sub_jac2 = jac.getDiagonalBlock(1);
   sub_jac2.Mult(vf, Jac_vf);
   for (int i = 0; i < getSize(flow_res); ++i)
   {
      REQUIRE( Jac_v(ptr+i) == Approx(Jac_vf(i)).margin(1e-5) );
   }
}