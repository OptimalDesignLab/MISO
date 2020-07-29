#include <random>

#include "catch.hpp"
#include "mfem.hpp"
#include "euler_integ.hpp"
#include "rans_fluxes.hpp"
#include "rans_integ.hpp"
#include "euler_test_data.hpp"

TEMPLATE_TEST_CASE_SIG("SA Inviscid Flux Test", "[sa_inviscid_flux_test]",
                       ((int dim), dim), 1, 2, 3)
{
    using namespace euler_data;
    mfem::Vector qL(dim + 3);
    mfem::Vector qR(dim + 3);
    mfem::Vector flux1(dim + 2);
    mfem::Vector flux2(dim + 3);
    mfem::Vector nrm(dim);
    qL(0) = rho; qR(0) = rho*1.1;
    qL(dim + 1) = rhoe; qR(dim + 1) = rhoe*1.1;
    for (int di = 0; di < dim; ++di)
    {
       qL(di + 1) = rhou[di];
       qR(di + 1) = rhou[di]*1.1;
       nrm(di) = dir[di];
    }
    qL(dim + 2) = 5.0; qR(dim + 2) = 5.5;
    adept::Stack stack;
    mach::IsmailRoeIntegrator<dim> irinteg(stack);
    mach::SAInviscidIntegrator<dim> sainteg(stack);

    SECTION("Check if SA integrator flux function matches the output for the conservative variables")
    {
        // calculate the flux 
        irinteg.calcFlux(0, qL, qR, flux1);
        sainteg.calcFlux(0, qL, qR, flux2);

        // check if euler variables are the same as before
        for(int n = 0; n < dim+2; n++)
        {
            std::cout << "Ismail-Roe " << n << " Flux: " << flux1(n) << std::endl; 
            std::cout << "Spalart-Allmaras " << n << " Flux: " << flux2(n) << std::endl; 
            REQUIRE(flux1(n) - flux2(n) == Approx(0.0));
        }
    }
}

TEMPLATE_TEST_CASE_SIG("SAInviscid Jacobian", "[SAInviscid]",
                       ((int dim), dim), 1, 2, 3)
{
    using namespace euler_data;
    // copy the data into mfem vectors for convenience
    mfem::Vector qL(dim + 3);
    mfem::Vector qR(dim + 3);
    mfem::Vector flux(dim + 3);
    mfem::Vector flux_plus(dim + 3);
    mfem::Vector flux_minus(dim + 3);
    mfem::Vector v(dim + 3);
    mfem::Vector jac_v(dim + 3);
    mfem::DenseMatrix jacL(dim + 3, 2 * (dim + 3));
    mfem::DenseMatrix jacR(dim + 3, 2 * (dim + 3));
    double delta = 1e-5;
    qL(0) = rho;
    qL(dim + 1) = rhoe;
    qL(dim + 2) = 4;
    qR(0) = rho2;
    qR(dim + 1) = rhoe2;
    qR(dim + 2) = 4.5;
    for (int di = 0; di < dim; ++di)
    {
       qL(di + 1) = rhou[di];
       qR(di + 1) = rhou2[di];
    }
    // create perturbation vector
    for (int di = 0; di < dim + 3; ++di)
    {
       v(di) = vec_pert[di];
    }
    // perturbed vectors
    mfem::Vector qL_plus(qL), qL_minus(qL);
    mfem::Vector qR_plus(qR), qR_minus(qR);
    adept::Stack diff_stack;
    mach::SAInviscidIntegrator<dim> sainvinteg(diff_stack);
    // +ve perturbation
    qL_plus.Add(delta, v);
    qR_plus.Add(delta, v);
    // -ve perturbation
    qL_minus.Add(-delta, v);
    qR_minus.Add(-delta, v);
    for (int di = 0; di < dim; ++di)
    {
        DYNAMIC_SECTION("Ismail-Roe flux jacsainvintegian is correct w.r.t left state ")
        {
            // get perturbed states flux vector
            sainvinteg.calcFlux(di, qL_plus, qR, flux_plus);
            sainvinteg.calcFlux(di, qL_minus, qR, flux_minus);
            // compute the jacobian
            sainvinteg.calcFluxJacStates(di, qL, qR, jacL, jacR);
            jacL.Mult(v, jac_v);
            // finite difference jacobian
            mfem::Vector jac_v_fd(flux_plus);
            jac_v_fd -= flux_minus;
            jac_v_fd /= 2.0 * delta;
            // compare each component of the matrix-vector products
            for (int i = 0; i < dim + 2; ++i)
            {
                REQUIRE(jac_v[i] == Approx(jac_v_fd[i]).margin(1e-10));
            }
        }
    }
}

TEMPLATE_TEST_CASE_SIG("SAFarFieldBC Jacobian", "[SAFarFieldBC]",
                       ((int dim), dim), 1, 2, 3)
{
    using namespace euler_data;
    // copy the data into mfem vectors for convenience
    mfem::Vector nrm(dim); mfem::Vector x(dim);
    mfem::Vector q(dim + 3);
    mfem::Vector qfar(dim + 3);
    mfem::Vector flux(dim + 3);
    mfem::Vector v(dim + 3);
    mfem::Vector jac_v(dim + 3);
    mfem::DenseMatrix Dw;
    mfem::DenseMatrix jac(dim + 3, dim + 3);
    double delta = 1e-5;
    
    x = 0.0;
    q(0) = rho;
    q(dim + 1) = rhoe;
    q(dim + 2) = 4;
    for (int di = 0; di < dim; ++di)
    {
       q(di + 1) = rhou[di];
    }
    qfar.Set(1.1, q);
    // create direction vector
    for (int di = 0; di < dim; ++di)
    {
       nrm(di) = dir[di];
    }
    // create perturbation vector
    for (int di = 0; di < dim + 3; ++di)
    {
       v(di) = vec_pert[di];
    }
    // perturbed vectors
    mfem::Vector q_plus(q), q_minus(q);
    mfem::Vector flux_plus(q), flux_minus(q);
    adept::Stack diff_stack;
    mfem::H1_FECollection fe_coll(1); //dummy
    double Re = 1.0; double Pr = 1.0;
    mach::SAFarFieldBC<dim> safarfieldinteg(diff_stack, &fe_coll, Re, Pr, qfar, 1.0);
    // +ve perturbation
    q_plus.Add(delta, v);
    // -ve perturbation
    q_minus.Add(-delta, v);
    for (int di = 0; di < 2; di++) //reverse direction to check both inflow and outflow
    {
        if(di == 1)
            nrm *= -1.0;
        
        DYNAMIC_SECTION("SA Far-Field BC safarfieldinteg jacobian is correct w.r.t state when di = "<<di)
        {
            
            // get perturbed states flux vector
            safarfieldinteg.calcFlux(x, nrm, 1.0, q_plus, Dw, flux_plus);
            safarfieldinteg.calcFlux(x, nrm, 1.0, q_minus, Dw, flux_minus);
            // compute the jacobian
            safarfieldinteg.calcFluxJacState(x, nrm, 1.0, q, Dw, jac);
            jac.Mult(v, jac_v);
            // finite difference jacobian
            mfem::Vector jac_v_fd(flux_plus);
            jac_v_fd -= flux_minus;
            jac_v_fd /= 2.0 * delta;
            // compare each component of the matrix-vector products
            for (int i = 0; i < dim + 3; ++i)
            {
                std::cout << "FD " << i << " Deriv: " << jac_v_fd[i]  << std::endl; 
                std::cout << "AN " << i << " Deriv: " << jac_v[i] << std::endl; 
                REQUIRE(jac_v[i] == Approx(jac_v_fd[i]).margin(1e-10));
            }
        }
    }
}

TEMPLATE_TEST_CASE_SIG("SALPS Jacobian", "[SALPS]",
                       ((int dim), dim), 1, 2, 3)
{
    using namespace euler_data;
    // copy the data into mfem vectors for convenience
    mfem::Vector q(dim + 3);
    mfem::Vector conv(dim + 3);
    mfem::Vector conv_plus(dim + 3);
    mfem::Vector conv_minus(dim + 3);
    mfem::Vector scale(dim + 3);
    mfem::Vector scale_plus(dim + 3);
    mfem::Vector scale_minus(dim + 3);
    mfem::Vector scale_plus_2(dim + 3);
    mfem::Vector scale_minus_2(dim + 3);
    mfem::Vector vec(vec_pert, dim + 3);
    mfem::Vector v(dim + 3);
    mfem::Vector jac_v(dim + 3);
    mfem::DenseMatrix adjJ(dim, dim);
    mfem::DenseMatrix jac_conv(dim + 3, dim + 3);
    mfem::DenseMatrix jac_scale(dim + 3, dim + 3);
    mfem::DenseMatrix jac_scale_2(dim + 3, dim + 3);
    double delta = 1e-5;
    adjJ = 0.7;
    q(0) = rho;
    q(dim + 1) = rhoe;
    q(dim + 2) = 4;
    for (int di = 0; di < dim; ++di)
    {
       q(di + 1) = rhou[di];
    }
    // create perturbation vector
    for (int di = 0; di < dim + 3; ++di)
    {
       v(di) = vec_pert[di];
    }
    // perturbed vectors
    mfem::Vector q_plus(q), q_minus(q);
    mfem::Vector vec_plus(vec), vec_minus(vec);
    adept::Stack diff_stack;
    mach::SALPSIntegrator<dim> salpsinteg(diff_stack);
    // +ve perturbation
    q_plus.Add(delta, v);
    vec_plus.Add(delta, v);
    // -ve perturbation
    q_minus.Add(-delta, v);
    vec_minus.Add(-delta, v);
    DYNAMIC_SECTION("LPS salpsinteg convertVars jacobian is correct w.r.t state ")
    {
        // get perturbed states conv vector
        salpsinteg.convertVars(q_plus, conv_plus);
        salpsinteg.convertVars(q_minus, conv_minus);
        salpsinteg.applyScaling(adjJ, q_plus, vec, scale_plus);
        salpsinteg.applyScaling(adjJ, q_minus, vec, scale_minus);
        salpsinteg.applyScaling(adjJ, q, vec_plus, scale_plus_2);
        salpsinteg.applyScaling(adjJ, q, vec_minus, scale_minus_2);
        // compute the jacobian
        salpsinteg.convertVarsJacState(q, jac_conv);
        salpsinteg.applyScalingJacState(adjJ, q, vec, jac_scale);
        salpsinteg.applyScalingJacV(adjJ, q, jac_scale_2);
        jac_conv.Mult(v, jac_v);
        // finite difference jacobian
        mfem::Vector jac_v_fd(conv_plus);
        jac_v_fd -= conv_minus;
        jac_v_fd /= 2.0 * delta;
        // compare each component of the matrix-vector products
        std::cout << "convertVars jac:" << std::endl; 
        for (int i = 0; i < dim + 3; ++i)
        {
            std::cout << "FD " << i << " Deriv: " << jac_v_fd[i]  << std::endl; 
            std::cout << "AN " << i << " Deriv: " << jac_v[i] << std::endl; 
            REQUIRE(jac_v[i] == Approx(jac_v_fd[i]).margin(1e-10));
        }
        
        jac_scale.Mult(v, jac_v);
        // finite difference jacobian
        jac_v_fd = scale_plus;
        jac_v_fd -= scale_minus;
        jac_v_fd /= 2.0 * delta;
        std::cout << "applyScaling jac:" << std::endl; 
        for (int i = 0; i < dim + 3; ++i)
        {
            std::cout << "FD " << i << " Deriv: " << jac_v_fd[i]  << std::endl; 
            std::cout << "AN " << i << " Deriv: " << jac_v[i] << std::endl; 
            REQUIRE(jac_v[i] == Approx(jac_v_fd[i]).margin(1e-10));
        }

        jac_scale_2.Mult(v, jac_v);
        // finite difference jacobian
        jac_v_fd = scale_plus_2;
        jac_v_fd -= scale_minus_2;
        jac_v_fd /= 2.0 * delta;
        std::cout << "applyScaling jac vec:" << std::endl; 
        for (int i = 0; i < dim + 3; ++i)
        {
            std::cout << "FD " << i << " Deriv: " << jac_v_fd[i]  << std::endl; 
            std::cout << "AN " << i << " Deriv: " << jac_v[i] << std::endl; 
            REQUIRE(jac_v[i] == Approx(jac_v_fd[i]).margin(1e-10));
        }
    }
}

TEST_CASE("SALPSIntegrator::AssembleElementGrad", "[SALPSIntegrator]")
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 2; // templating is hard here because mesh constructors
   int num_state = dim + 3;
   adept::Stack diff_stack;
   double delta = 1e-5;

   // generate a 2 element mesh
   int num_edge = 2;
   std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, Element::TRIANGLE,
                                       true /* gen. edges */, 1.0, 1.0, true));
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
             new SBPCollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             mesh.get(), fec.get(), num_state, Ordering::byVDIM));

         NonlinearForm res(fes.get());
         res.AddDomainIntegrator(new mach::SALPSIntegrator<dim>(diff_stack));

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselinePert<2>);
         q.ProjectCoefficient(pert);

         // initialize the vector that the Jacobian multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(num_state, randState);
         v.ProjectCoefficient(v_rand);

         // evaluate the Jacobian and compute its product with v
         Operator &Jac = res.GetGradient(q);
         GridFunction jac_v(fes.get());
         Jac.Mult(v, jac_v);

         // now compute the finite-difference approximation...
         GridFunction q_pert(q), r(fes.get()), jac_v_fd(fes.get());
         q_pert.Add(-delta, v);
         res.Mult(q_pert, r);
         q_pert.Add(2 * delta, v);
         res.Mult(q_pert, jac_v_fd);
         jac_v_fd -= r;
         jac_v_fd /= (2 * delta);

         for (int i = 0; i < jac_v.Size(); ++i)
         {
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)).margin(1e-10));
         }
      }
   }
}


#if 0
TEMPLATE_TEST_CASE_SIG("SAInviscid Gradient",
                       "[SAInviscidGrad]",
                       ((bool entvar), entvar), false)
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 2; // templating is hard here because mesh constructors
   double delta = 1e-5;

   // generate a 8 element mesh
   int num_edge = 2;
   std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, Element::TRIANGLE,
                                       true /* gen. edges */, 1.0, 1.0, true));
   mesh->EnsureNodes();
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         // get the finite-element space for the state and adjoint
         std::unique_ptr<FiniteElementCollection> fec(
             new SBP_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             mesh.get(), fec.get()));

         // we use res for finite-difference approximation
         NonLinearForm res(fes.get());
         res.AddDomainIntegrator(
            new SAInviscidIntegrator<dim, entvar>( //Inviscid term
                this->diff_stack, alpha));

         // initialize state ; here we randomly perturb a constant state
         GridFunction state(fes.get())
         VectorFunctionCoefficient v_rand(dim+3, randState);
         state.ProjectCoefficient(pert);

         // initialize the vector that we use to perturb the mesh nodes
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(dim, randState);
         v.ProjectCoefficient(v_rand);

         // evaluate df/dx and contract with v
         GridFunction dfdx(*x_nodes);
         dfdx_form.Mult(*x_nodes, dfdx);
         double dfdx_v = dfdx * v;

         // now compute the finite-difference approximation...
         GridFunction x_pert(*x_nodes);
         GridFunction r(fes.get());
         x_pert.Add(delta, v);
         mesh->SetNodes(x_pert);
         res.Assemble();
         double dfdx_v_fd = adjoint * res;
         x_pert.Add(-2 * delta, v);
         mesh->SetNodes(x_pert);
         res.Assemble();
         dfdx_v_fd -= adjoint * res;
         dfdx_v_fd /= (2 * delta);
         mesh->SetNodes(*x_nodes); // remember to reset the mesh nodes

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-10));
      }
   }
}
#endif