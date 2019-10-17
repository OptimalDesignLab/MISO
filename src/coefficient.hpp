#ifndef MACH_COEFFICIENT
#define MACH_COEFFICIENT

#include <map>
#include "mfem.hpp"

#include "mach_types.hpp"
#include "utils.hpp"


namespace mach
{

/// TODO - replace mfem_error with mach_error type
class StateCoefficient : public mfem::Coefficient
{
public:
	virtual double Eval(mfem::ElementTransformation &trans,
							  const mfem::IntegrationPoint &ip)
	{
		std::cerr << "Wrong Eval method for StateCoefficient!" << std::endl;
		return 0.0;
	}
	virtual double Eval(mfem::ElementTransformation &trans,
							  const mfem::IntegrationPoint &ip,
							  const double state) = 0;

	virtual double EvalStateDeriv(mfem::ElementTransformation &trans,
											const mfem::IntegrationPoint &ip,
											const double state) = 0;
};


class MeshDependentCoefficient : public StateCoefficient
{
public:
	MeshDependentCoefficient()
	{
		// initialize empty material_map
		// material_map = {};

		// // create `hasEvalStateDeriv` anonymous function
		// const mfem::ElementTransformation &trans = mfem::IsoparametricTransformation();
		// const mfem::IntegrationPoint &ip = mfem::IntegrationPoint();
		// hasEvalStateDeriv = mach::is_valid([](auto&& x) -> decltype(x.EvalStateDeriv(trans, ip)) { });
	}

	MeshDependentCoefficient(const std::map<const int, mfem::Coefficient*> 
									 &input_map)
	 : material_map(input_map)
	 {
		// // create `hasEvalStateDeriv` anonymous function
		// const mfem::ElementTransformation &trans = mfem::IsoparametricTransformation();
		// const mfem::IntegrationPoint &ip = mfem::IntegrationPoint();
		// hasEvalStateDeriv = mach::is_valid([](auto&& x) -> decltype(x.EvalStateDeriv(trans, ip)) { });
	 }
	
	/// Adds <int, mfem::Coefficient*> pair to material_map where the int
	/// corresponds to a material attribute in the mesh.
	/// \param coeff - attribute-Coefficient pair where the Coefficient is the
	///					 one to evaluate on elements identified by the attribute
	virtual void addCoefficient(std::pair<const int, mfem::Coefficient*> coeff)
	{
		std::pair <std::map<const int, mfem::Coefficient*>::iterator, bool> status;
		status = material_map.insert(coeff);
		// if the pair failed to insert
		if (!status.second)
		{
			mfem::mfem_error("Key already present in map!");
		}
	}

	/// \brief Search the map of coefficients and evaluate the one whose key is
	/// 		  the same as the element's `Attribute` at the point defined by
	///		  `ip`.
	/// \param[in] trans - element transformation relating real element to
	///					 	  reference element
	/// \param[in] ip - the integration point to evalaute the coefficient at
	/// \param[in] state - the state at which to evaluate the coefficient
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
	virtual double Eval(mfem::ElementTransformation &trans,
							  const mfem::IntegrationPoint &ip,
							  const double state);

	/// TODO - implement expression SFINAE when iterating over map
	/// TODO - Consider different model for coefficient's dependent upon multiple
	///		  GridFunctions
	/// \brief Search the map of coefficients and evaluate the derivative with
	/// 		  respect to the state of the one whose key is the same as the
	///		  element's `Attribute` at the point defined by `ip`.
	/// \param[in] trans - element transformation relating real element to
	///					 	  reference element
	/// \param[in] ip - the integration point to evalaute the coefficient at
	/// \param[in] state - the state at which to evaluate the coefficient
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
	virtual double EvalStateDeriv(mfem::ElementTransformation &trans,
											const mfem::IntegrationPoint &ip,
											const double state);

protected:

	/// \brief Method to be called if a coefficient matching the element's
	/// 		  attribute is a subclass of `StateCoefficient and
	///		  thus implements `Eval()` with state argument
	/// \param[in] *coeff - pointer to the coefficient in the map
	/// \param[in] trans - element transformation relating real element to
	///					 	  reference element
	/// \param[in] ip - the integration point to evalaute the coefficient at
	/// \param[in] state - the state at which to evaluate the coefficient
	/// \tparam T - templated type, must be a subclass of `mfem::Coefficient`
	/// \tparam typename - Uses template meta programming and SFINAE to check if
	///						  `T` is a subclass of `ExplictStateDependentCoefficient`
	///						  If it is not, typename is void and this function is
	///						  an invalid overload and not considered. This enables
	///						  compile-time introspection of object.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
	template <class T, typename
				 std::enable_if<std::is_base_of<StateCoefficient,
				 T>::value, int>::type= 0>
	inline double Eval(T *coeff,
							 mfem::ElementTransformation &trans,
							 const mfem::IntegrationPoint &ip,
							 const double state)
	{
		return coeff->Eval(trans, ip, state);
	}

	/// \brief Method to be called if a coefficient matching the element's
	/// 		  attribute is not a subclass of `StateCoefficient and thus
	///		  does not implement `Eval()` with state argument
	/// \param[in] *coeff - pointer to the coefficient in the map
	/// \param[in] trans - element transformation relating real element to
	///					 	  reference element
	/// \param[in] ip - the integration point to evalaute the coefficient at
	/// \param[in] state - the state at which to evaluate the coefficient
	/// \tparam T - templated type, must be a subclass of `mfem::Coefficient`
	/// \tparam typename - Uses template meta programming and SFINAE to check if
	///						  `T` is a subclass of `ExplictStateDependentCoefficient`
	///						  If it is not, typename is void and this function is
	///						  an invalid overload and not considered. This enables
	///						  compile-time introspection of object.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
	template <class T, typename
				 std::enable_if<!std::is_base_of<StateCoefficient,
				 T>::value, int>::type= 0>
	inline double Eval(T *coeff,
							 mfem::ElementTransformation &trans,
							 const mfem::IntegrationPoint &ip,
							 const double state)
	{
		return coeff->Eval(trans, ip);
	}

	/// \brief Method to be called if a coefficient matching the element's
	/// 		  attribute is a subclass of `StateCoefficient and
	///		  thus implements `EvalStateDeriv()`
	/// \param[in] *coeff - pointer to the coefficient in the map
	/// \param[in] trans - element transformation relating real element to
	///					 	  reference element
	/// \param[in] ip - the integration point to evalaute the coefficient at
	/// \param[in] state - the state at which to evaluate the coefficient
	/// \tparam T - templated type, must be a subclass of `mfem::Coefficient`
	/// \tparam typename - Uses template meta programming and SFINAE to check if
	///						  `T` is a subclass of `ExplictStateDependentCoefficient`
	///						  If it is not, typename is void and this function is
	///						  an invalid overload and not considered. This enables
	///						  compile-time introspection of object.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
	template <class T, typename
				 std::enable_if<std::is_base_of<StateCoefficient,
				 T>::value, int>::type= 0>
	inline double EvalStateDeriv(T *coeff,
										  mfem::ElementTransformation &trans,
										  const mfem::IntegrationPoint &ip,
										  const double state)
	{
		return coeff->EvalStateDeriv(trans, ip, state);
	}

	/// \brief Method to be called if a coefficient matching the element's
	/// 		  attribute is not a subclass of `StateCoefficient
	///		  and does not implement `EvalStateDeriv()`
	/// \param[in] *coeff - pointer to the coefficient in the map
	/// \param[in] trans - element transformation relating real element to
	///					 	  reference element
	/// \param[in] ip - the integration point to evalaute the coefficient at
	/// \param[in] state - the state at which to evaluate the coefficient
	/// \tparam T - templated type, must be a subclass of `mfem::Coefficient`
	/// \tparam typename - Uses template meta programming and SFINAE to check if
	///						  `T` is a subclass of `ExplictStateDependentCoefficient`
	///						  If it is not, typename is void and this function is
	///						  an invalid overload and not considered. This enables
	///						  compile-time introspection of object.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
	template <class T, typename
				 std::enable_if<!std::is_base_of<StateCoefficient,
				 T>::value, int>::type = 0>
	inline double EvalStateDeriv(T *coeff,
										  mfem::ElementTransformation &trans,
										  const mfem::IntegrationPoint &ip,
										  const double state)
	{
		return 0.0;
	}

private:
	std::map<const int, mfem::Coefficient*> material_map;
	// static const auto hasEvalStateDeriv;
};

// should i make this associated with a grid function or should I pass the state to it to evaluate?
// need to check how Joule handles solving for temp, see if the state passed to an integrator is unique
// to a finite element space (A and T would be seperate spaces). Also need to look at how much reluctivity
// changes with temperature. (Ask prof Shah?)

class ReluctivityCoefficient : public StateCoefficient
{
public:
	/// Define a temperature independent reluctivity model
	/// \param model - user defined function to evalaute relctivuty based on
	///					 magnetic flux
	ReluctivityCoefficient(double (*model)(const double))
	 : Bmodel(model), BTmodel(NULL),
		temperature_GF(NULL) {}

	/// Define a temperature dependent reluctivity model
	/// \param model - user defined function to evalaute relctivuty based on
	///					 magnetic flux density and temperature
	/// \param *T_ - pointer to existing temperature grid function
	ReluctivityCoefficient(double (*model)(const double, double),
								  GridFunType *T_)
	 : Bmodel(NULL), BTmodel(model),
		temperature_GF(T_) {}

	/// \brief Evaluate the reluctivity in the element described by trans at the
	/// point ip. Checks which model was initialized, temperature-dependent or
	/// not, and evalutes the correct one.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   virtual double Eval(mfem::ElementTransformation &trans,
                       const mfem::IntegrationPoint &ip,
							  const double state);

	/// \brief Evaluate the derivative of reluctivity with respsect to magnetic
	/// flux in the element described by trans at the point ip. Checks which
	/// model was initialized, temperature-dependent or not, and evalutes the
	/// correct one.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
	virtual double EvalStateDeriv(mfem::ElementTransformation &trans,
                       				const mfem::IntegrationPoint &ip,
											const double state);

	/// class destructor. Not sure if I need to delete anything?
	~ReluctivityCoefficient() {}

protected:
	/// Function to evalaute reluctivity model with no temperature dependence
	/// \param mfem::Vector B - magnetic flux density
	double (*Bmodel)(const double);

	/// Function to evalaute reluctivity model with temperature dependence
	/// \param mfem::Vector B - magnetic flux density
	/// \param double T - temperature
	double (*BTmodel)(const double, double);

	/// reference to temperature grid function
	GridFunType *temperature_GF;
};

} // namespace mach

#endif