#ifndef MACH_FORCE_INTEGRATOR
#define MACH_FORCE_INTEGRATOR

#include "mfem.hpp"
#include "solver.hpp"

namespace mach
{

/// Integrator that evaluate functionals 
class ForceIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   /// Constructs a boundary integrator based on a given boundary flux
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] bndFun - boundary function
   /// \param[in] fe_coll - used to determine the face elements
   /// \param[in] num_state_vars - the number of state variables
   ForceIntegrator(adept::Stack &diff_stack,
                              void (*fluxFun)(const double *x,
                                              const double *nrm,
                                              const double *u,
                                              double *flux_vec),
                              const mfem::Vector direction,
                              const mfem::FiniteElementCollection *fe_coll,
                              int num_state_vars = 1)
       : num_states(num_state_vars), stack(diff_stack),
         bnd_fun(fluxFun), dir(direction), fec(fe_coll) { }
   
	/// Construct the contribution from the boundary elements faces
	/// \param[in] el_bnd - boundary element that contribute to the functional
	/// \param[in] el_unused - summy element that is not used for boundaries
   /// \param[in] trans - hold geometry and mapping information about the face
   /// \param[in] elfun - element local state function
   /// \return a double indicate the local contribution to functional
	double GetFaceEnergy(const mfem::FiniteElement &el_bnd,
                                   const mfem::FiniteElement &el_unused,
                                   mfem::FaceElementTransformations &trans,
                                   const mfem::Vector &elfun);

private: 
   /// number of states
   int num_states;
   /// stack used for algorithmic differentiation
   adept::Stack &stack;

   /// Direction of the desired force
   mfem::Vector dir;
   
   /// boundary function used on the given boundary
   void (*bnd_fun)(const double *x, const double *nrm, const double *u, double *q);

   /// derivative of functional respect to the state variables
   // void (*diff_bnd_fun)(const double *x, const double *nrm,
   //          const double *u, double *dJdu);

   /// used to select the appropriate face element
   const mfem::FiniteElementCollection *fec;

#ifndef MFEM_THREAD_SAFE
   /// used to reference the state at face node
   mfem::Vector u_face; 
   /// store the physical location of a node
   mfem::Vector x;
   /// the outward pointing (scaled) normal to the boundary at a node
   mfem::Vector nrm;
   /// stores the flux evaluated by `bnd_flux`
   mfem::Vector flux_face;
#endif
};

} // namespace mach

#endif