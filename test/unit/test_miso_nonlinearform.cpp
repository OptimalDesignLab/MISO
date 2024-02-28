#include <iostream>
#include <random>

#include "catch.hpp"
#include "nlohmann/json.hpp"
#include "mfem.hpp"

#include "finite_element_state.hpp"
#include "miso_integrator.hpp"
#include "miso_nonlinearform.hpp"

class TestIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   TestIntegrator()
   : time(0.0)
   { }

   // The form here is just the integral of v * cos(t) * u^2, where v is the 
   // test function, t is the time, and u is the state.  The test below sets 
   // v = 1 and u = x + y and integrates over [0,1]^2.  This should give a
   // value of 7*cos(t)/6 when `elvect` is summed.
   void AssembleElementVector(
      const mfem::FiniteElement &el,
      mfem::ElementTransformation &trans,
      const mfem::Vector &elfun,
      mfem::Vector &elvect) override
   {
      const mfem::IntegrationRule *ir = IntRule;
      if (ir == nullptr)
      {
         const int order = 2*el.GetOrder() - 2;
         ir = &mfem::IntRules.Get(el.GetGeomType(), order);
      }
      shape.SetSize(el.GetDof());
      elvect.SetSize(el.GetDof());
      elvect = 0.0;
      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const mfem::IntegrationPoint &ip = ir->IntPoint(i);
         trans.SetIntPoint(&ip);
         const double w = ip.weight * trans.Weight();
         el.CalcShape(ip, shape);
         auto u2 = pow(mfem::InnerProduct(shape, elfun), 2);
         for (int j = 0; j < el.GetDof(); ++j)
         {
            elvect(j) += shape(j) * w * cos(time) * u2;
         }
      }
   }

   // used to set the time variable for the form 
   friend void setInputs(TestIntegrator &integ, const miso::MISOInputs &inputs);

private:
   double time;
   mfem::Vector shape;
   friend class TestIntegratorMeshSens;
   friend class TestIntegratorTimeSens;
};

class TestIntegratorMeshSens : public mfem::LinearFormIntegrator
{
public:
   TestIntegratorMeshSens(mfem::GridFunction &state,
                          mfem::GridFunction &adjoint,
                          TestIntegrator &integ)
    : state(state), adjoint(adjoint), integ(integ)
   { }

   void AssembleRHSElementVect(const mfem::FiniteElement &mesh_el,
                               mfem::ElementTransformation &mesh_trans,
                               mfem::Vector &mesh_coords_bar) override
   {
      /// get the proper element, transformation, and state vector
      int element = mesh_trans.ElementNo;
      const auto &el = *state.FESpace()->GetFE(element);
      auto &trans = *state.FESpace()->GetElementTransformation(element);

      const int ndof = mesh_el.GetDof();
      const int el_ndof = el.GetDof();
      const int dim = el.GetDim();
      mesh_coords_bar.SetSize(ndof * dim);
      mesh_coords_bar = 0.0;

      auto *dof_tr = state.FESpace()->GetElementVDofs(element, vdofs);
      state.GetSubVector(vdofs, elfun);
      if (dof_tr != nullptr)
      {
         dof_tr->InvTransformPrimal(elfun);
      }

      dof_tr = adjoint.FESpace()->GetElementVDofs(element, vdofs);
      adjoint.GetSubVector(vdofs, psi);
      if (dof_tr != nullptr)
      {
         dof_tr->InvTransformPrimal(psi);
      }

      auto &shape = integ.shape;
      shape.SetSize(el.GetDof());
      PointMat_bar.SetSize(dim, ndof);

      // cast the ElementTransformation
      auto &isotrans = dynamic_cast<mfem::IsoparametricTransformation &>(trans);

      const mfem::IntegrationRule *ir = IntRule;
      if (ir == nullptr)
      {
         const int order = 2*el.GetOrder() - 2;
         ir = &mfem::IntRules.Get(el.GetGeomType(), order);
      }
      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const auto &ip = ir->IntPoint(i);
         trans.SetIntPoint(&ip);

         const double w = ip.weight * trans.Weight();

         el.CalcShape(ip, shape);
         auto u2 = pow(mfem::InnerProduct(shape, elfun), 2);

         auto v = psi * shape;

         /// dummy functional for adjoint-weighted residual
         // double fun += v * cos(time) * u2 * w;

         /// start reverse pass
         double fun_bar = 1.0;

         double w_bar = fun_bar * v * cos(integ.time) * u2;

         /// const double w = ip.weight * trans.Weight();
         double trans_weight_bar = w_bar * ip.weight;

         PointMat_bar = 0.0;
         isotrans.WeightRevDiff(trans_weight_bar, PointMat_bar);
         // insert PointMat_bar into mesh_coords_bar;
         for (int j = 0; j < ndof; ++j)
         {
            for (int d = 0; d < dim; ++d)
            {
               mesh_coords_bar(d * ndof + j) += PointMat_bar(d, j);
            }
         }
      }
   }

private:
   /// the state to use when evaluating d(psi^T R)/dX
   mfem::GridFunction &state;
   /// the adjoint to use when evaluating d(psi^T R)/dX
   mfem::GridFunction &adjoint;
   /// reference to primal integrator
   TestIntegrator &integ;

   mfem::DenseMatrix PointMat_bar;
   mfem::Array<int> vdofs;
   mfem::Vector elfun, psi;
};

class TestIntegratorTimeSens : public mfem::NonlinearFormIntegrator
{
public:
   TestIntegratorTimeSens(mfem::GridFunction &adjoint,
                          TestIntegrator &integ)
    : adjoint(adjoint), integ(integ)
   { }

   double GetElementEnergy(const mfem::FiniteElement &el,
                               mfem::ElementTransformation &trans,
                               const mfem::Vector &elfun) override
   {
      /// get the proper element, transformation, and state vector

      const int ndof = el.GetDof();
      const int dim = el.GetDim();

      int element = trans.ElementNo;
      auto *dof_tr = adjoint.FESpace()->GetElementVDofs(element, vdofs);
      adjoint.GetSubVector(vdofs, psi);
      if (dof_tr != nullptr)
      {
         dof_tr->InvTransformPrimal(psi);
      }

      auto &shape = integ.shape;
      shape.SetSize(el.GetDof());


      const mfem::IntegrationRule *ir = IntRule;
      if (ir == nullptr)
      {
         const int order = 2*el.GetOrder() - 2;
         ir = &mfem::IntRules.Get(el.GetGeomType(), order);
      }

      double time_bar = 0.0;
      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const auto &ip = ir->IntPoint(i);
         trans.SetIntPoint(&ip);

         const double w = ip.weight * trans.Weight();

         el.CalcShape(ip, shape);
         auto u2 = pow(mfem::InnerProduct(shape, elfun), 2);

         auto v = psi * shape;

         auto cos_time = cos(integ.time);

         /// dummy functional for adjoint-weighted residual
         // double fun += v * cos(time) * u2 * w;

         /// start reverse pass
         double fun_bar = 1.0;

         double cos_time_bar = fun_bar * v * u2 * w;

         /// auto cos_time = cos(integ.time);
         time_bar -= cos_time_bar * sin(integ.time);
      }
      return time_bar;
   }

private:
   /// the adjoint to use when evaluating d(psi^T R)/dX
   mfem::GridFunction &adjoint;
   /// reference to primal integrator
   TestIntegrator &integ;

   mfem::Array<int> vdofs;
   mfem::Vector psi;
};

void setInputs(TestIntegrator &integ, const miso::MISOInputs &inputs)
{
   miso::setValueFromInputs(inputs, "time", integ.time);
}

inline void addDomainSensitivityIntegrator(
    TestIntegrator &primal_integ,
    std::map<std::string, miso::FiniteElementState> &fields,
    std::map<std::string, mfem::ParLinearForm> &rev_sens,
    std::map<std::string, mfem::ParNonlinearForm> &rev_scalar_sens,
    std::map<std::string, mfem::ParLinearForm> &fwd_sens,
    std::map<std::string, mfem::ParNonlinearForm> &fwd_scalar_sens,
    mfem::Array<int> *attr_marker,
    std::string adjoint_name)
{
   auto &mesh_fes = fields.at("mesh_coords").space();
   rev_sens.emplace("mesh_coords", &mesh_fes);
   rev_sens.at("mesh_coords")
       .AddDomainIntegrator(
           new TestIntegratorMeshSens(fields.at("state").gridFunc(),
                                      fields.at(adjoint_name).gridFunc(),
                                      primal_integ));

   auto &state_fes = fields.at("state").space();
   rev_scalar_sens.emplace("time", &state_fes);
   rev_scalar_sens.at("time")
       .AddDomainIntegrator(
           new TestIntegratorTimeSens(fields.at(adjoint_name).gridFunc(),
                                      primal_integ));
}

using namespace miso;

TEST_CASE("MISONonlinearForm vectorJacobianProduct (vector) test")
{
   static std::default_random_engine gen;
   static std::uniform_real_distribution<double> uniform_rand(-1.0,1.0);

   // set up the mesh and finite-element space
   auto numx = 4;
   auto numy = 4;
   auto smesh = mfem::Mesh::MakeCartesian2D(numx, numy, mfem::Element::TRIANGLE);
   mfem::ParMesh mesh(MPI_COMM_WORLD, smesh);
   mesh.EnsureNodes();

   auto p = 2;

   // create a MISONonlinearForm and wrap it into a MISOResidual
   std::map<std::string, FiniteElementState> fields;

   fields.emplace(std::piecewise_construct,
                  std::forward_as_tuple("state"),
                  std::forward_as_tuple(mesh, FiniteElementState::Options{.order=p}));

   auto &mesh_gf = *dynamic_cast<mfem::ParGridFunction *>(mesh.GetNodes());
   auto *mesh_fespace = mesh_gf.ParFESpace();

   /// create new state vector copying the mesh's fe space
   fields.emplace(std::piecewise_construct,
                  std::forward_as_tuple("mesh_coords"),
                  std::forward_as_tuple(mesh, *mesh_fespace, "mesh_coords"));
   FiniteElementState &mesh_coords = fields.at("mesh_coords");
   /// set the values of the new GF to those of the mesh's old nodes
   mesh_coords.gridFunc() = mesh_gf;
   /// tell the mesh to use this GF for its Nodes
   /// (and that it doesn't own it)
   mesh.NewNodes(mesh_coords.gridFunc(), false);

   auto &state = fields.at("state");

   MISONonlinearForm form(state.space(), fields);
   form.addDomainIntegrator(new TestIntegrator);

   mfem::Vector state_tv(state.space().GetTrueVSize());
   state.project([](const mfem::Vector&xy) { return xy(0) + xy(1); }, state_tv);

   mfem::Vector mesh_coords_tv(mesh_coords.space().GetTrueVSize());
   mesh_coords.setTrueVec(mesh_coords_tv);

   auto inputs = MISOInputs({
      {"state", state_tv},
      {"mesh_coords", mesh_coords_tv}
   });
   setInputs(form, inputs);

   mfem::Vector res_bar(getSize(form));
   for (int i = 0; i < res_bar.Size(); ++i)
   {
      res_bar(i) = uniform_rand(gen);
   }

   mfem::Vector wrt_bar(mesh_coords.space().GetTrueVSize());
   wrt_bar = 0.0;
   vectorJacobianProduct(form, res_bar, "mesh_coords", wrt_bar);

   // initialize the vector that we use to perturb the mesh nodes
   mfem::Vector v_tv(mesh_coords.space().GetTrueVSize());
   for (int i = 0; i < v_tv.Size(); ++i)
   {
      v_tv(i) = uniform_rand(gen);
   }

   auto dJdx_v_local = wrt_bar * v_tv;
   double dJdx_v;
   MPI_Allreduce(&dJdx_v_local,
                 &dJdx_v,
                 1,
                 MPI_DOUBLE,
                 MPI_SUM,
                 MPI_COMM_WORLD);

   // now compute the finite-difference approximation...
   auto delta = 1e-5;
   double dJdx_v_fd_local = 0.0;
   mfem::Vector res_vec(getSize(form));

   add(mesh_coords_tv, delta, v_tv, mesh_coords_tv);
   mesh_coords.distributeSharedDofs(mesh_coords_tv); // update mesh nodes
   inputs.at("mesh_coords") = mesh_coords_tv;

   res_vec = 0.0;
   evaluate(form, inputs, res_vec);
   dJdx_v_fd_local += res_bar * res_vec;

   add(mesh_coords_tv, -2*delta, v_tv, mesh_coords_tv);
   mesh_coords.distributeSharedDofs(mesh_coords_tv); // update mesh nodes
   inputs.at("mesh_coords") = mesh_coords_tv;

   res_vec = 0.0;
   evaluate(form, inputs, res_vec);
   dJdx_v_fd_local -= res_bar * res_vec;

   dJdx_v_fd_local /= 2*delta;

   double dJdx_v_fd;
   MPI_Allreduce(&dJdx_v_fd_local,
                 &dJdx_v_fd,
                 1,
                 MPI_DOUBLE,
                 MPI_SUM,
                 MPI_COMM_WORLD);

   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   if (rank == 0)
   {
      std::cout << "dJdx_v: " << dJdx_v << "\n";
      std::cout << "dJdx_v_fd: " << dJdx_v_fd << "\n";
   }

   REQUIRE(dJdx_v == Approx(dJdx_v_fd).margin(1e-8));
}

TEST_CASE("MISONonlinearForm vectorJacobianProduct (scalar) test")
{
   static std::default_random_engine gen;
   static std::uniform_real_distribution<double> uniform_rand(-1.0,1.0);

   // set up the mesh and finite-element space
   auto numx = 4;
   auto numy = 4;
   auto smesh = mfem::Mesh::MakeCartesian2D(numx, numy, mfem::Element::TRIANGLE);
   mfem::ParMesh mesh(MPI_COMM_WORLD, smesh);
   mesh.EnsureNodes();

   auto p = 2;

   // create a MISONonlinearForm and wrap it into a MISOResidual
   std::map<std::string, FiniteElementState> fields;

   fields.emplace(std::piecewise_construct,
                  std::forward_as_tuple("state"),
                  std::forward_as_tuple(mesh, FiniteElementState::Options{.order=p}));

   auto &mesh_gf = *dynamic_cast<mfem::ParGridFunction *>(mesh.GetNodes());
   auto *mesh_fespace = mesh_gf.ParFESpace();

   /// create new state vector copying the mesh's fe space
   fields.emplace(std::piecewise_construct,
                  std::forward_as_tuple("mesh_coords"),
                  std::forward_as_tuple(mesh, *mesh_fespace, "mesh_coords"));
   FiniteElementState &mesh_coords = fields.at("mesh_coords");
   /// set the values of the new GF to those of the mesh's old nodes
   mesh_coords.gridFunc() = mesh_gf;
   /// tell the mesh to use this GF for its Nodes
   /// (and that it doesn't own it)
   mesh.NewNodes(mesh_coords.gridFunc(), false);

   auto &state = fields.at("state");

   MISONonlinearForm form(state.space(), fields);
   form.addDomainIntegrator(new TestIntegrator);

   mfem::Vector state_tv(state.space().GetTrueVSize());
   state.project([](const mfem::Vector&xy) { return xy(0) + xy(1); }, state_tv);

   mfem::Vector mesh_coords_tv(mesh_coords.space().GetTrueVSize());
   mesh_coords.setTrueVec(mesh_coords_tv);

   double time = M_PI/2;

   auto inputs = MISOInputs({
      {"state", state_tv},
      {"mesh_coords", mesh_coords_tv},
      {"time", time}
   });
   setInputs(form, inputs);

   mfem::Vector res_bar(getSize(form));
   for (int i = 0; i < res_bar.Size(); ++i)
   {
      res_bar(i) = uniform_rand(gen);
   }

   double wrt_bar_local = vectorJacobianProduct(form, res_bar, "time");
   double wrt_bar;
   MPI_Allreduce(&wrt_bar_local,
                 &wrt_bar,
                 1,
                 MPI_DOUBLE,
                 MPI_SUM,
                 MPI_COMM_WORLD);   

   auto delta = 1e-5;
   double wrt_bar_fd_local = 0.0;
   mfem::Vector res_vec(getSize(form));

   res_vec = 0.0;
   inputs.at("time") = time + delta;
   setInputs(form, inputs);
   evaluate(form, inputs, res_vec);
   wrt_bar_fd_local += res_bar * res_vec;

   res_vec = 0.0;
   inputs.at("time") = time - delta;
   setInputs(form, inputs);
   evaluate(form, inputs, res_vec);
   wrt_bar_fd_local -= res_bar * res_vec;

   wrt_bar_fd_local /= 2*delta;
   double wrt_bar_fd;
   MPI_Allreduce(&wrt_bar_fd_local,
                 &wrt_bar_fd,
                 1,
                 MPI_DOUBLE,
                 MPI_SUM,
                 MPI_COMM_WORLD); 

   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   if (rank == 0)
   {
      std::cout << "wrt_bar: " << wrt_bar << "\n";
      std::cout << "wrt_bar_fd: " << wrt_bar_fd << "\n";
   }
   REQUIRE(wrt_bar == Approx(wrt_bar_fd));
}
