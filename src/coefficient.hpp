#ifndef MACH_COEFFICIENT
#define MACH_COEFFICIENT

#include <map>
#include "mfem.hpp"

#include "mach_types.hpp"
#include "utils.hpp"


namespace mach
{

/// Abstract class StateCoefficient
/// Defines new signature for Eval() and new method EvalStateDeriv() that
/// subclasses must implement.
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


/// MeshDependentCoefficient
/// A class that contains a map of material attributes and coefficients to
/// evaluate on for each attribute.
class MeshDependentCoefficient : public StateCoefficient
{
public:
	/// Construct MeshDependentCoefficient
	/// \param [in] dflt - default coefficient to evaluate if element attribute
	///						  is not found in the map. If not set, will default
	///						  to zero
	MeshDependentCoefficient(std::unique_ptr<mfem::Coefficient> dflt = NULL)
		: default_coeff(move(dflt)) {}

	/// Adds <int, std::unique_ptr<mfem::Coefficient> pair to material_map
	/// \param[in] attr - attribute integer indicating which elements coeff
	///					    should be evaluated on
	/// \param[in] coeff - the coefficient the to evaluate on elements
	///						  identified by the attribute
	virtual void addCoefficient(const int attr,
										 std::unique_ptr<mfem::Coefficient> coeff)
	{
		auto status = material_map.insert(std::make_pair(attr, std::move(coeff)));
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
	std::unique_ptr<mfem::Coefficient> default_coeff;
	std::map<const int, std::unique_ptr<mfem::Coefficient>> material_map;
};

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


	/// TODO: implement
	/// Define a reluctivity model based on experimental B-H curve
	/// \param[in] B - magnetic flux density values from B-H curve 
	/// \param[in] H - magnetic field intensity valyes from B-H curve
	ReluctivityCoefficient(std::vector<double> B, std::vector<double> H);

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
	/// \param const double - magnitude of magnetic flux density
	double (*Bmodel)(const double);

	/// Function to evalaute reluctivity model with temperature dependence
	/// \param const double - magnitude of magnetic flux density
	/// \param double T - temperature
	double (*BTmodel)(const double, const double);

	/// reference to temperature grid function
	GridFunType *temperature_GF;
};

class VectorMeshDependentCoefficient : public mfem::VectorCoefficient
{
public:
	VectorMeshDependentCoefficient(const int dim = 3,
											 std::unique_ptr<mfem::VectorCoefficient>
												dflt = NULL)
		 : default_coeff(move(dflt)), VectorCoefficient(dim) {}

	/// Adds <int, std::unique_ptr<mfem::VectorCoefficient> pair to material_map
	/// \param[in] attr - attribute integer indicating which elements coeff
	///					    should be evaluated on
	/// \param[in] coeff - the coefficient the to evaluate on elements
	///						  identified by the attribute
	virtual void addCoefficient(const int attr,
										 std::unique_ptr<mfem::VectorCoefficient> coeff)
	{
		auto status = material_map.insert(std::make_pair(attr, std::move(coeff)));
		// if the pair failed to insert
		if (!status.second)
		{
			mfem::mfem_error("Key already present in map!");
		}
	}

	/// \brief Search the map of coefficients and evaluate the one whose key is
	/// 		  the same as the element's `Attribute` at the point defined by
	///		  `ip`.
	/// \param[out] vec - output vector storing the result of the evaluation
	/// \param[in] trans - element transformation relating real element to
	///					 	  reference element
	/// \param[in] ip - the integration point to evalaute the coefficient at
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
	void Eval(mfem::Vector &vec,
				 mfem::ElementTransformation &trans,
			    const mfem::IntegrationPoint &ip);

	// /// TODO - implement expression SFINAE when iterating over map
	// /// TODO - Consider different model for coefficient's dependent upon multiple
	// ///		  GridFunctions
	// /// \brief Search the map of coefficients and evaluate the derivative with
	// /// 		  respect to the state of the one whose key is the same as the
	// ///		  element's `Attribute` at the point defined by `ip`.
	// /// \param[in] trans - element transformation relating real element to
	// ///					 	  reference element
	// /// \param[in] ip - the integration point to evalaute the coefficient at
	// /// \param[in] state - the state at which to evaluate the coefficient
   // /// \note When this method is called, the caller must make sure that the
   // /// IntegrationPoint associated with trans is the same as ip. This can be
   // /// achieved by calling trans.SetIntPoint(&ip).
	// virtual double EvalStateDeriv(mfem::ElementTransformation &trans,
	// 										const mfem::IntegrationPoint &ip,
	// 										const double state);

protected:
	std::unique_ptr<mfem::VectorCoefficient> default_coeff;
	std::map<const int, std::unique_ptr<mfem::VectorCoefficient>> material_map;
};

} // namespace mach

#endif