#include <fstream>

#include "pfem_extras.hpp"

#include "magnetostatic.hpp"
#include "solver.hpp"
#include "electromag_integ.hpp"
#include "mesh_movement.hpp"

using namespace std;
using namespace mfem;

namespace
{
/// permeability of free space
const double mu_0 = 4e-7*M_PI;

std::unique_ptr<mfem::Coefficient>
constructReluctivityCoeff(nlohmann::json &component, nlohmann::json &materials)
{
   std::unique_ptr<mfem::Coefficient> temp_coeff;
   std::string material = component["material"].get<std::string>();
   if (!component["linear"].get<bool>())
   {
      auto b = materials[material]["B"].get<std::vector<double>>();
      auto h = materials[material]["H"].get<std::vector<double>>();
      temp_coeff.reset(new mach::ReluctivityCoefficient(b, h));
   }
   else
   {
      auto mu_r = materials[material]["mu_r"].get<double>();
      temp_coeff.reset(new mfem::ConstantCoefficient(1.0/(mu_r*mu_0)));
      std::cout << "new coeff with mu_r: " << mu_r << "\n";
   }
   return temp_coeff;
}
} // anonymous namespace

namespace mach
{

MagnetostaticSolver::MagnetostaticSolver(
   const std::string &opt_file_name,
   std::unique_ptr<mfem::Mesh> smesh)
   : AbstractSolver(opt_file_name, move(smesh))
{
   dim = getMesh()->Dimension();
   int order = options["space-dis"]["degree"].get<int>();
   // num_state = dim;

   mesh->ReorientTetMesh();
   mesh->RemoveInternalBoundaries();

   /// Create the H(Div) finite element collection
   h_div_coll.reset(new RT_FECollection(order, dim));
   /// Create the H1 finite element collection
   h1_coll.reset(new H1_FECollection(order, dim));
   /// Create the L2 finite element collection
   l2_coll.reset(new L2_FECollection(order, dim));

   /// Create the H(Div) finite element space
   h_div_space.reset(new SpaceType(mesh.get(), h_div_coll.get()));
   /// Create the H1 finite element space
   h1_space.reset(new SpaceType(mesh.get(), h1_coll.get()));
   /// Create the L2 finite element space
   l2_space.reset(new SpaceType(mesh.get(), l2_coll.get()));

   /// Create magnetic flux grid function
   B.reset(new GridFunType(h_div_space.get()));
}

void MagnetostaticSolver::printSolution(const std::string &file_name,
                                       int refine)
{
   printFields(file_name,
               {u.get(), B.get()},
               {"MVP", "Magnetic_Flux_Density"},
               refine);
}

void MagnetostaticSolver::setEssentialBoundaries()
{
   /// apply zero tangential boundary condition everywhere
   ess_bdr.SetSize(mesh->bdr_attributes.Max());
   ess_bdr = 1;

   Array<int> ess_tdof_list;
   fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   res->SetEssentialTrueDofs(ess_tdof_list);
   /// set current vector's ess_tdofs to zero
   current_vec->SetSubVector(ess_tdof_list, 0.0);
}

void MagnetostaticSolver::solveSteady()
{
   if (newton_solver == nullptr)
      constructNewtonSolver();

   setEssentialBoundaries();

   Vector Zero(3);
   Zero = 0.0;
   bc_coef.reset(new VectorConstantCoefficient(Zero)); // for motor 
   // bc_coef.reset(new VectorFunctionCoefficient(3, a_exact)); // for box problem

   *u = 0.0;
   u->ProjectBdrCoefficientTangent(*bc_coef, ess_bdr);

   HypreParVector *u_true = u->GetTrueDofs();
   HypreParVector *current_true = current_vec->GetTrueDofs();
   newton_solver->Mult(*current_true, *u_true);
   MFEM_VERIFY(newton_solver->GetConverged(), "Newton solver did not converge.");
   u->SetFromTrueDofs(*u_true);

   computeSecondaryFields();
}

void MagnetostaticSolver::addOutputs()
{
   auto &fun = options["outputs"];
   std::cout << fun;
   if (fun.find("energy") != fun.end())
   { 
      std::cout << "adding energy!\n";
      output.emplace("energy", fes.get());
      output.at("energy").AddDomainIntegrator(
         new MagneticEnergyIntegrator(nu.get()));
   }
   if (fun.find("co-energy") != fun.end())
   {
      std::cout << "adding co-energy!\n"; 
      output.emplace("co-energy", fes.get());
      output.at("co-energy").AddDomainIntegrator(
         new MagneticCoenergyIntegrator(*u, nu.get()));
   }
   std::cout << "done adding outputs!\n";
   /// TODO: implement torque
}

std::vector<GridFunType*> MagnetostaticSolver::getFields(void)
{
   return {u.get(), B.get()};
}

Vector* MagnetostaticSolver::getMeshSensitivities()
{	
   /// assign mesh node space to forms
   mesh->EnsureNodes();
   GridFunction *x_nodes_s = mesh->GetNodes();
   FiniteElementSpace *mesh_fes_s = x_nodes_s->FESpace();
   SpaceType mesh_fes(*mesh_fes_s, *mesh);
   GridFunType x_nodes(&mesh_fes, x_nodes_s);

   dLdX.reset(new GridFunType(x_nodes));
   *dLdX = 0.0;

   /// Add mesh sensitivities of functional 
   LinearFormType dJdX(&mesh_fes);
   dJdX.AddDomainIntegrator(
      new MagneticCoenergyIntegrator(*u, nu.get()));
   dJdX.Assemble();

   /// TODO I don't know if this works in parallel / when we need to use tdof vectors
   *dLdX += dJdX;

   res_mesh_sens_l.reset(new LinearFormType(&mesh_fes));

   /// compute \psi_A
   solveForAdjoint(options["outputs"]["co-energy"].get<std::string>());

   /// add integrators R = CurlCurl(A) + Cm + Mj = 0
   /// \psi^T CurlCurl(A)
   res_mesh_sens_l->AddDomainIntegrator(
      new CurlCurlNLFIntegrator(nu.get(), u.get(), adj.get()));
   /// \psi^T C m 
   res_mesh_sens_l->AddDomainIntegrator(
      new VectorFECurldJdXIntegerator(nu.get(), M.get(), adj.get()));
   /// \psi^T M j
   res_mesh_sens_l->AddDomainIntegrator(
      new VectorFEMassdJdXIntegerator(div_free_current_vec.get(), adj.get()));

   /// Compute the derivatives and accumulate the result
   res_mesh_sens_l->Assemble();

   /// dJdX = \partialJ / \partial X + \psi^T \partial R / \partial X
   dLdX->Add(-1, *res_mesh_sens_l);

   return dLdX.get();
}

void MagnetostaticSolver::constructCoefficients()
{
   current_vec.reset(new GridFunType(fes.get()));
   div_free_current_vec.reset(new GridFunType(fes.get()));
   
   /// read options file to set the proper values of static member variables
   setStaticMembers();
   /// Construct current source coefficient
   constructCurrent();
   /// Assemble current source vector
   assembleCurrentSource();
   /// Construct magnetization coefficient
   constructMagnetization();
   /// Assemble magnetization source vector and add it into current
   assembleMagnetizationSource();
   /// Construct reluctivity coefficient
   constructReluctivity();
}

void MagnetostaticSolver::addVolumeIntegrators(double alpha)
{
   /// TODO: Add a check in `CurlCurlNLFIntegrator` to check if |B| is close to
   ///       zero, and if so set the second term of the Jacobian to be zero.
   /// add curl curl integrator to residual
   res->AddDomainIntegrator(new CurlCurlNLFIntegrator(nu.get()));
}

void MagnetostaticSolver::setStaticMembers()
{
   if (options["components"].contains("magnets"))
   {
      auto &magnets = options["components"]["magnets"];
      std::string material = magnets["material"].get<std::string>();
      remnant_flux = materials[material]["B_r"].get<double>();
      mag_mu_r = materials[material]["mu_r"].get<double>();
   }
   fill_factor = options["problem-opts"].value("fill-factor", 1.0);
   current_density = options["problem-opts"].value("current-density", 1.0);
}

void MagnetostaticSolver::constructReluctivity()
{
   /// set up default reluctivity to be that of free space
   // const double mu_0 = 4e-7*M_PI;
   std::unique_ptr<Coefficient> nu_free_space(
      new ConstantCoefficient(1.0/mu_0));
   nu.reset(new MeshDependentCoefficient(move(nu_free_space)));

   /// loop over all components, construct either a linear or nonlinear
   ///    reluctivity coefficient for each
   for (auto& component : options["components"])
   {
      int attr = component.value("attr", -1);
      if (-1 != attr)
      {
         std::unique_ptr<mfem::Coefficient> temp_coeff;
         temp_coeff = constructReluctivityCoeff(component, materials);
         nu->addCoefficient(attr, move(temp_coeff));
      }
      else
      {
         auto attrs = component["attrs"].get<std::vector<int>>();
         for (auto& attribute : attrs)
         {
            std::unique_ptr<mfem::Coefficient> temp_coeff;
            temp_coeff = constructReluctivityCoeff(component, materials);
            nu->addCoefficient(attribute, move(temp_coeff));
         }
      }
   }
}

/// TODO - this approach cannot support general magnet topologies where the
///        magnetization cannot be described by a single vector function 
void MagnetostaticSolver::constructMagnetization()
{
   mag_coeff.reset(new VectorMeshDependentCoefficient(dim));

   if (options["problem-opts"].contains("magnets"))
   {
      auto &magnets = options["problem-opts"]["magnets"];
      if (magnets.contains("north"))
      {
         auto attrs = magnets["north"].get<std::vector<int>>();
         for (auto& attr : attrs)
         {
            std::unique_ptr<mfem::VectorCoefficient> temp_coeff(
                  new VectorFunctionCoefficient(dim,
                                                magnetization_source_north));
            mag_coeff->addCoefficient(attr, move(temp_coeff));
         }
      }
      if (magnets.contains("south"))
      {
         auto attrs = magnets["south"].get<std::vector<int>>();
         
         for (auto& attr : attrs)
         {
            std::unique_ptr<mfem::VectorCoefficient> temp_coeff(
                  new VectorFunctionCoefficient(dim,
                                                magnetization_source_south));
            mag_coeff->addCoefficient(attr, move(temp_coeff));
         }
      }
      if (magnets.contains("x"))
      {
         auto attrs = magnets["x"].get<std::vector<int>>();
         for (auto& attr : attrs)
         {
            std::unique_ptr<mfem::VectorCoefficient> temp_coeff(
                  new VectorFunctionCoefficient(dim,
                                                x_axis_magnetization_source));
            current_coeff->addCoefficient(attr, move(temp_coeff));
         }
      }
      if (magnets.contains("y"))
      {
         auto attrs = magnets["y"].get<std::vector<int>>();
         for (auto& attr : attrs)
         {
            std::unique_ptr<mfem::VectorCoefficient> temp_coeff(
                  new VectorFunctionCoefficient(dim,
                                                y_axis_magnetization_source));
            current_coeff->addCoefficient(attr, move(temp_coeff));
         }
      }
      if (magnets.contains("z"))
      {
         auto attrs = magnets["z"].get<std::vector<int>>();
         for (auto& attr : attrs)
         {
            std::unique_ptr<mfem::VectorCoefficient> temp_coeff(
                  new VectorFunctionCoefficient(dim,
                                                z_axis_magnetization_source));
            current_coeff->addCoefficient(attr, move(temp_coeff));
         }
      }
   }
}

/// TODO - use options to select which winding belongs to which phase, need to
///        finalize mesh reading before I finish this, as the mesh will decide
///        how to do this (each winding getting its own attribute or not)
void MagnetostaticSolver::constructCurrent()
{
   current_coeff.reset(new VectorMeshDependentCoefficient());

   if (options["problem-opts"].contains("current"))
   {
      auto &current = options["problem-opts"]["current"];
      if (current.contains("Phase-A"))
      {
         auto attrs = current["Phase-A"].get<std::vector<int>>();
         for (auto& attr : attrs)
         {
            std::unique_ptr<mfem::VectorCoefficient> temp_coeff(
                  new VectorFunctionCoefficient(dim, phase_a_source));
            current_coeff->addCoefficient(attr, move(temp_coeff));
         }
      }
      if (current.contains("Phase-B"))
      {
         auto attrs = current["Phase-B"].get<std::vector<int>>();
         for (auto& attr : attrs)
         {
            std::unique_ptr<mfem::VectorCoefficient> temp_coeff(
                  new VectorFunctionCoefficient(dim, phase_b_source));
            current_coeff->addCoefficient(attr, move(temp_coeff));
         }
      }
      if (current.contains("Phase-C"))
      {
         auto attrs = current["Phase-C"].get<std::vector<int>>();
         for (auto& attr : attrs)
         {
            std::unique_ptr<mfem::VectorCoefficient> temp_coeff(
                  new VectorFunctionCoefficient(dim, phase_c_source));
            current_coeff->addCoefficient(attr, move(temp_coeff));
         }
      }
      if (current.contains("x"))
      {
         auto attrs = current["x"].get<std::vector<int>>();
         for (auto& attr : attrs)
         {
            std::unique_ptr<mfem::VectorCoefficient> temp_coeff(
                  new VectorFunctionCoefficient(dim, x_axis_current_source));
            current_coeff->addCoefficient(attr, move(temp_coeff));
         }
      }
      if (current.contains("y"))
      {
         auto attrs = current["y"].get<std::vector<int>>();
         for (auto& attr : attrs)
         {
            std::unique_ptr<mfem::VectorCoefficient> temp_coeff(
                  new VectorFunctionCoefficient(dim, y_axis_current_source));
            current_coeff->addCoefficient(attr, move(temp_coeff));
         }
      }
      if (current.contains("z"))
      {
         auto attrs = current["z"].get<std::vector<int>>();
         for (auto& attr : attrs)
         {
            std::unique_ptr<mfem::VectorCoefficient> temp_coeff(
                  new VectorFunctionCoefficient(dim, z_axis_current_source));
            current_coeff->addCoefficient(attr, move(temp_coeff));
         }
      }
      if (current.contains("ring"))
      {
         auto attrs = current["ring"].get<std::vector<int>>();
         for (auto& attr : attrs)
         {
            std::unique_ptr<mfem::VectorCoefficient> temp_coeff(
                     new VectorFunctionCoefficient(dim, ring_current_source));
            current_coeff->addCoefficient(attr, move(temp_coeff));
         }
      }
   }
}

void MagnetostaticSolver::assembleCurrentSource()
{
   int fe_order = options["space-dis"]["degree"].get<int>();

   /// Create the H1 finite element collection and space, only used by the
   /// divergence free projectors so we define them here and then throw them
   /// away
   // auto h1_coll = H1_FECollection(fe_order, dim);
   // auto h1_space = SpaceType(mesh.get(), &h1_coll);

   /// get int rule (approach followed my MFEM Tesla Miniapp)
   int irOrder = h1_space->GetElementTransformation(0)->OrderW()
               + 2 * fe_order;
   int geom = h1_space->GetFE(0)->GetGeomType();
   const IntegrationRule *ir = &IntRules.Get(geom, irOrder);

   /// compute the divergence free current source
   auto *grad = new mfem::common::ParDiscreteGradOperator(h1_space.get(), fes.get());

   // assemble gradient form
   grad->Assemble();
   grad->Finalize();

   auto div_free_proj = mfem::common::DivergenceFreeProjector(*h1_space, *fes,
                                             irOrder, NULL, NULL, grad);

   GridFunType j = GridFunType(fes.get());
   j.ProjectCoefficient(*current_coeff);

   GridFunType j_div_free = GridFunType(fes.get());
   // Compute the discretely divergence-free portion of j
   // div_free_proj.Mult(j, j_div_free);
   *div_free_current_vec = 0.0;
   div_free_proj.Mult(j, *div_free_current_vec);
   /// create current linear form vector by multiplying mass matrix by
   /// divergence free current source grid function
   // ConstantCoefficient one(1.0);
   BilinearFormIntegrator *h_curl_mass_integ = new VectorFEMassIntegrator;
   h_curl_mass_integ->SetIntRule(ir);
   BilinearFormType *h_curl_mass = new BilinearFormType(fes.get());
   h_curl_mass->AddDomainIntegrator(h_curl_mass_integ);

   // assemble mass matrix
   h_curl_mass->Assemble();
   h_curl_mass->Finalize();

   *current_vec = 0.0;
   h_curl_mass->AddMult(*div_free_current_vec, *current_vec);
   std::cout << "below h_curl add mult\n";
   // I had strange errors when not using pointer versions of these
   delete h_curl_mass;
   delete grad;
}

void MagnetostaticSolver::assembleMagnetizationSource(void)
{
   M.reset(new GridFunType(h_div_space.get()));

   auto weakCurlMuInv_ = new ParMixedBilinearForm(h_div_space.get(), fes.get());
   weakCurlMuInv_->AddDomainIntegrator(new VectorFECurlIntegrator(*nu));

   weakCurlMuInv_->Assemble();
   weakCurlMuInv_->Finalize();

   M->ProjectCoefficient(*mag_coeff);
   weakCurlMuInv_->AddMult(*M, *current_vec, 1.0);

   delete weakCurlMuInv_;
}

void MagnetostaticSolver::computeSecondaryFields()
{
   std::cout << "before curl constructed\n";
   DiscreteCurlOperator curl(fes.get(), h_div_space.get());
   std::cout << "curl constructed\n";
   curl.Assemble();
   curl.Finalize();
   curl.Mult(*u, *B);
   std::cout << "secondary quantities computed\n";
}

/// TODO: Find a better way to handle solving the simple box problem
void MagnetostaticSolver::phase_a_source(const Vector &x,
                                       Vector &J)
{
   // example of needed geometric parameters, this should be all you need
   int n_s = 12; //number of slots
   double zb = 0.0; //bottom of stator
   double zt = 0.25; //top of stator


   // compute r and theta from x and y
   // double r = sqrt(x(0)*x(0) + x(1)*x(1)); (r not needed)
   double tha = atan2(x(1), x(0));
   double th;

   double thw = 2*M_PI/n_s; //total angle of slot
   int w; //current slot
   J = 0.0;

   // check which winding we're in
   th = remquo(tha, thw, &w);

   // check if we're in the stator body
   if(x(2) >= zb && x(2) <= zt)
   {
      // check if we're in left or right half
      if(th > 0)
      {
         J(2) = -1; // set to 1 for now, and direction depends on current direction
      }
      if(th < 0)
      {
         J(2) = 1;	
      }
   }
   else  // outside of the stator body, check if above or below
   {
      // 'subtract' z position to 0 depending on if above or below
      mfem::Vector rx(x);
      if(x(2) > zt) 
      {
         rx(2) -= zt; 
      }
      if(x(2) < zb) 
      {
         rx(2) -= zb; 
      }

      // draw top rotation axis
      mfem::Vector ax(3);
      mfem::Vector Jr(3);
      ax = 0.0;
      ax(0) = cos(w*thw);
      ax(1) = sin(w*thw);

      // take x cross ax, normalize
      Jr(0) = rx(1)*ax(2) - rx(2)*ax(1);
      Jr(1) = rx(2)*ax(0) - rx(0)*ax(2);
      Jr(2) = rx(0)*ax(1) - rx(1)*ax(0);
      Jr /= Jr.Norml2();
      J = Jr;
   }
   J *= current_density * fill_factor;
}

void MagnetostaticSolver::phase_b_source(const Vector &x,
                                       Vector &J)
{
   // example of needed geometric parameters, this should be all you need
   int n_s = 12; //number of slots
   double zb = 0.0; //bottom of stator
   double zt = 0.25; //top of stator


   // compute r and theta from x and y
   // double r = sqrt(x(0)*x(0) + x(1)*x(1)); (r not needed)
   double tha = atan2(x(1), x(0));
   double th;

   double thw = 2*M_PI/n_s; //total angle of slot
   int w; //current slot
   J = 0.0;

   // check which winding we're in
   th = remquo(tha, thw, &w);

   // check if we're in the stator body
   if(x(2) >= zb && x(2) <= zt)
   {
      // check if we're in left or right half
      if(th > 0)
      {
         J(2) = -1; // set to 1 for now, and direction depends on current direction
      }
      if(th < 0)
      {
         J(2) = 1;	
      }
   }
   else  // outside of the stator body, check if above or below
   {
      // 'subtract' z position to 0 depending on if above or below
      mfem::Vector rx(x);
      if(x(2) > zt) 
      {
         rx(2) -= zt; 
      }
      if(x(2) < zb) 
      {
         rx(2) -= zb; 
      }

      // draw top rotation axis
      mfem::Vector ax(3);
      mfem::Vector Jr(3);
      ax = 0.0;
      ax(0) = cos(w*thw);
      ax(1) = sin(w*thw);

      // take x cross ax, normalize
      Jr(0) = rx(1)*ax(2) - rx(2)*ax(1);
      Jr(1) = rx(2)*ax(0) - rx(0)*ax(2);
      Jr(2) = rx(0)*ax(1) - rx(1)*ax(0);
      Jr /= Jr.Norml2();
      J = Jr;
   }
   J *= -current_density * fill_factor;
}

void MagnetostaticSolver::phase_c_source(const Vector &x,
                                       Vector &J)
{
   J.SetSize(3);
   J = 0.0;
   // Vector r = x;
   // r(2) = 0.0;
   // r /= r.Norml2();
   // J(0) = -r(1);
   // J(1) = r(0);
   // J *= current_density;
}

/// TODO: Find a better way to handle solving the simple box problem
/// TODO: implement other kinds of sources
void MagnetostaticSolver::magnetization_source_north(const Vector &x,
                                                   Vector &M)
{
   Vector plane_vec = x;
   plane_vec(2) = 0;
   M = plane_vec;
   M /= M.Norml2();
   M *= remnant_flux;
}

void MagnetostaticSolver::magnetization_source_south(const Vector &x,
                                                   Vector &M)
{
   Vector plane_vec = x;
   plane_vec(2) = 0;
   M = plane_vec;
   M /= M.Norml2();
   M *= -remnant_flux;

   // M = 0.0;
   // M(2) = remnant_flux;
}

void MagnetostaticSolver::x_axis_current_source(const Vector &x,
                                                Vector &J)
{
   J.SetSize(3);
   J = 0.0;
   J(0) = current_density;
}

void MagnetostaticSolver::y_axis_current_source(const Vector &x,
                                                Vector &J)
{
   J.SetSize(3);
   J = 0.0;
   J(1) = current_density;   
}

void MagnetostaticSolver::z_axis_current_source(const Vector &x,
                                                Vector &J)
{
   J.SetSize(3);
   J = 0.0;
   J(2) = current_density;
}

void MagnetostaticSolver::ring_current_source(const Vector &x,
                                              Vector &J)
{
   J.SetSize(3);
   J = 0.0;
   Vector r = x;
   r(2) = 0.0;
   r /= r.Norml2();
   J(0) = -r(1);
   J(1) = r(0);
   J *= current_density;
}

void MagnetostaticSolver::x_axis_magnetization_source(const Vector &x,
                                                      Vector &M)
{
   M.SetSize(3);
   M = 0.0;
   M(0) = remnant_flux;
}

void MagnetostaticSolver::y_axis_magnetization_source(const Vector &x,
                                                      Vector &M)
{
   M.SetSize(3);
   M = 0.0;
   M(1) = remnant_flux;
}

void MagnetostaticSolver::z_axis_magnetization_source(const Vector &x,
                                                      Vector &M)
{
   M.SetSize(3);
   M = 0.0;
   M(2) = remnant_flux;
}

/// TODO: Find a better way to handle solving the simple box problem
void MagnetostaticSolver::a_exact(const Vector &x, Vector &A)
{
   A.SetSize(3);
   A = 0.0;
   double y = x(1) - .5;
   if ( x(1) <= .5)
   {
      A(2) = y*y*y; 
   }
   else 
   {
      A(2) = -y*y*y;
   }
}

void MagnetostaticSolver::b_exact(const Vector &x, Vector &B)
{
   B.SetSize(3);
   B = 0.0;
   double y = x(1) - .5;
   if ( x(1) <= .5)
   {
      B(0) = 3*y*y; 
   }
   else 
   {
      B(0) = -3*y*y;
   }	
}

double MagnetostaticSolver::remnant_flux = 0.0;
double MagnetostaticSolver::mag_mu_r = 0.0;
double MagnetostaticSolver::fill_factor = 0.0;
double MagnetostaticSolver::current_density = 0.0;

} // namespace mach
