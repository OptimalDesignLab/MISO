#ifndef MACH_COEFFICIENT
#define MACH_COEFFICIENT

#include "mfem.hpp"

#include "mach_types.hpp"


namespace mach
{

// should i make this associated with a grid function or should I pass the state to it to evaluate?
// need to check how Joule handles solving for temp, see if the state passed to an integrator is unique
// to a finite element space (A and T would be seperate spaces). Also need to look at how much reluctivity
// changes with temperature. (Ask prof Shah?) 

class ReluctivityCoefficient : public mfem::Coefficient
{
public:
	/// Define a temperature independent reluctivity model
	/// \param model - user defined function to evalaute relctivuty based on 
	///					 magnetic flux
	/// \param *B_ - pointer to existing magnetic flux grid function
	ReluctivityCoefficient(double (*model)(mfem::Vector), GridFunType *B_)
	 : Bmodel(model), BTmodel(NULL),
		magnetic_flux_GF(B_), temperature_GF(NULL) {}

	/// Define a temperature dependent reluctivity model
	/// \param model - user defined function to evalaute relctivuty based on 
	///					 magnetic flux density and temperature
	/// \param *B_ - pointer to existing magnetic flux grid function
	/// \param *T_ - pointer to existing temperature grid function
	ReluctivityCoefficient(double (*model)(mfem::Vector, double),
								  GridFunType *B_, GridFunType *T_)
	 : Bmodel(NULL), BTmodel(model), 
		magnetic_flux_GF(B_), temperature_GF(T_) {}

	/// \brief Evaluate the reluctivity in the element described by trans at the
	/// point ip. Checks which model was initialized, temperature-dependent or
	/// not, and evalutes the correct one.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   virtual double Eval(mfem::ElementTransformation &trans,
                       const mfem::IntegrationPoint &ip);

	/// \brief Evaluate the derivative of reluctivity with respsect to magnetic
	/// flux in the element described by trans at the point ip. Checks which
	/// model was initialized, temperature-dependent or not, and evalutes the
	/// correct one.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
	virtual double EvalStateDeriv(mfem::ElementTransformation &trans,
                       				const mfem::IntegrationPoint &ip);

	/// class destructor. Not sure if I need to delete anything?
	~ReluctivityCoefficient() {}

protected:
	/// Function to evalaute reluctivity model with no temperature dependence
	/// \param mfem::Vector B - magnetic flux density
	double (*Bmodel)(const mfem::Vector);

	/// Function to evalaute reluctivity model with temperature dependence
	/// \param mfem::Vector B - magnetic flux density
	/// \param double T - temperature
	double (*BTmodel)(const mfem::Vector, double);

	/// reference to magnetic flux grid function
	GridFunType *magnetic_flux_GF;
	/// reference to temperature grid function
	GridFunType *temperature_GF;
};

} // namespace mach

#endif