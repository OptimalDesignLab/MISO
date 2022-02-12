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
      "mach": 0.5,
      "aoa": 0.0
   },
   "space-dis": {
      "degree": 1,
      "lps-coeff": 1.0,
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

   // evaluate the residual at an arbitrary state
   Vector x(num_var);
   std::default_random_engine gen(std::random_device{}());
   std::uniform_real_distribution<double> uniform_rand(0.0, 1.0);
   for (int i = 0; i < num_var; ++i)
   {
      x(i) = uniform_rand(gen);
   }
   double time = uniform_rand(gen);
   auto inputs = MachInputs({{"state", x}, {"time", time}});
   Vector res_vec(num_var);
   evaluate(res, inputs, res_vec);
   if (rank == 0)
   {
      REQUIRE( res_vec(0) == Approx(-0.05*x(1)).margin(1e-14) );
      REQUIRE( res_vec(1) == Approx(0.05*x(0)).margin(1e-14) );
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
      REQUIRE( Jac_v(0) == Approx(-0.05*v(1)).margin(1e-14) );
      REQUIRE( Jac_v(1) == Approx(0.05*v(0)).margin(1e-14) );
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
      REQUIRE(entropy == Approx(x(0) * x(0) + x(1) * x(1)).margin(1e-14));
   }

   // check entropy change
   v = 0.0;
   inputs = MachInputs(
       {{"state", x}, {"time", time}, {"state_dot", v}, {"dt", 0.0}});
   double entropy_change = calcEntropyChange(res, inputs);
   if (rank == 0)
   {
      REQUIRE(entropy_change == Approx(0.0).margin(1e-14));
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

   // construct the residual
   MachResidual res(
       FlowControlResidual<dim, false>(options, fespace, diff_stack));
   FlowResidual<dim, false> flow_res(options, fespace, diff_stack);
   ControlResidual control_res(MPI_COMM_WORLD, options);
   int num_var = getSize(res);
   REQUIRE(num_var == getSize(flow_res) + getSize(control_res));

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

   // Set up the inputs to the three residuals
   Vector x_actuator({0.0, 0.5});
   double vel_control = control_res.getControlVelocity(q);
   MachOutput boundary_entropy(flow_res.constructOutput(
       "boundary-entropy", options["outputs"]["boundary-entropy"]));
   Vector flow_state(q.GetData() + ptr, getSize(flow_res));
   double time = 0.0;
   auto flow_inputs = MachInputs({{"state", flow_state},
                                  {"time", time},
                                  {"x-actuator", x_actuator},
                                  {"control", vel_control}});
   double bndry_ent = calcOutput(boundary_entropy, flow_inputs);
   auto inputs =
       MachInputs({{"state", q}, {"time", time}, {"x-actuator", x_actuator}});
   auto control_inputs = MachInputs(
       {{"state", q}, {"time", time}, {"boundary-entropy", bndry_ent}});

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