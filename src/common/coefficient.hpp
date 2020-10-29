#ifndef MACH_COEFFICIENT
#define MACH_COEFFICIENT

#include <map>

#include "mfem.hpp"

#include "mach_types.hpp"
#include "utils.hpp"
#include "spline.hpp"


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
      return Eval(trans, ip, 0);
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
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
	virtual double Eval(mfem::ElementTransformation &trans,
							  const mfem::IntegrationPoint &ip);

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

	/// \brief Search the map of coefficients and evaluate the one whose key is
   ///        the same as the element's `Attribute` at the point defined by
   ///        `ip`.
   /// \param[in] Q_bar - derivative of functional with respect to `Q`
   /// \param[in] trans - element transformation relating real element to
   ///                    reference element
   /// \param[in] ip - defines location in reference space
   /// \param[out] PointMat_bar - derivative of function w.r.t. mesh nodes
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   virtual void EvalRevDiff(const double Q_bar,
                            mfem::ElementTransformation &trans,
                            const mfem::IntegrationPoint &ip,
                            mfem::DenseMatrix &PointMat_bar);

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
	///						  `T` is a subclass of `StateCoefficient`
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
	///						  `T` is a subclass of `StateCoefficient`
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

	// /// \brief Method to be called if a coefficient matching the element's
	// /// 		  attribute is a subclass of `StateCoefficient and
	// ///		  thus implements `Eval()` with state argument
	// /// \param[in] *coeff - pointer to the coefficient in the map
	// /// \param[in] trans - element transformation relating real element to
	// ///					 	  reference element
	// /// \param[in] ip - the integration point to evalaute the coefficient at
	// /// \tparam T - templated type, must be a subclass of `mfem::Coefficient`
	// /// \tparam typename - Uses template meta programming and SFINAE to check if
	// ///						  `T` is a subclass of `StateCoefficient`
	// ///						  If it is not, typename is void and this function is
	// ///						  an invalid overload and not considered. This enables
	// ///						  compile-time introspection of object.
   // /// \note When this method is called, the caller must make sure that the
   // /// IntegrationPoint associated with trans is the same as ip. This can be
   // /// achieved by calling trans.SetIntPoint(&ip).
	// template <class T, typename
	// 			 std::enable_if<std::is_base_of<StateCoefficient,
	// 			 T>::value, int>::type= 0>
	// inline double Eval(T *coeff,
	// 						 mfem::ElementTransformation &trans,
	// 						 const mfem::IntegrationPoint &ip)
	// {
	// 	return coeff->Eval(trans, ip, 0);
	// }

	// /// \brief Method to be called if a coefficient matching the element's
	// /// 		  attribute is not a subclass of `StateCoefficient and thus
	// ///		  does not implement `Eval()` with state argument
	// /// \param[in] *coeff - pointer to the coefficient in the map
	// /// \param[in] trans - element transformation relating real element to
	// ///					 	  reference element
	// /// \param[in] ip - the integration point to evalaute the coefficient at
	// /// \tparam T - templated type, must be a subclass of `mfem::Coefficient`
	// /// \tparam typename - Uses template meta programming and SFINAE to check if
	// ///						  `T` is a subclass of `StateCoefficient`
	// ///						  If it is not, typename is void and this function is
	// ///						  an invalid overload and not considered. This enables
	// ///						  compile-time introspection of object.
   // /// \note When this method is called, the caller must make sure that the
   // /// IntegrationPoint associated with trans is the same as ip. This can be
   // /// achieved by calling trans.SetIntPoint(&ip).
	// template <class T, typename
	// 			 std::enable_if<!std::is_base_of<StateCoefficient,
	// 			 T>::value, int>::type= 0>
	// inline double Eval(T *coeff,
	// 						 mfem::ElementTransformation &trans,
	// 						 const mfem::IntegrationPoint &ip)
   // {
	// 	return coeff->Eval(trans, ip);
	// }

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
	///						  `T` is a subclass of `StateCoefficient`
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
	///						  `T` is a subclass of `StateCoefficient`
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
	/// \param[in] B - magnetic flux density values from B-H curve 
	/// \param[in] H - magnetic field intensity valyes from B-H curve
	ReluctivityCoefficient(std::vector<double> B, std::vector<double> H);

	/// TODO - implement
	/// Define a temperature dependent reluctivity model
	/// \param[in] B - magnetic flux density values from B-H curve 
	/// \param[in] H - magnetic field intensity valyes from B-H curve
	/// \param *T_ - pointer to existing temperature grid function
	/// \note not currently implemented
	ReluctivityCoefficient(std::vector<double> B, std::vector<double> H,
								  GridFunType *T_);

	/// \brief Evaluate the reluctivity in the element described by trans at the
	/// point ip. Checks which model was initialized, temperature-dependent or
	/// not, and evalutes the correct one.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   double Eval(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
					const double state) override;

	/// \brief Evaluate the derivative of reluctivity with respsect to magnetic
	/// flux in the element described by trans at the point ip. Checks which
	/// model was initialized, temperature-dependent or not, and evalutes the
	/// correct one.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
	double EvalStateDeriv(mfem::ElementTransformation &trans,
                       	 const mfem::IntegrationPoint &ip,
								 const double state) override;

	/// class destructor. Not sure if I need to delete anything?
	~ReluctivityCoefficient() = default;

protected:
	/// reference to temperature grid function
	GridFunType *temperature_GF;

	/// spline representing B-H curve, 1st deriv is reluctivity
	Spline b_h_curve;
};

class MagneticFluxCoefficient : public StateCoefficient
{
public:
	/// Define a temperature independent magnetic flux model
	/// \param[in] B - magnetic flux density values from B-H curve 
	/// \param[in] H - magnetic field intensity valyes from B-H curve
	MagneticFluxCoefficient(std::vector<double> B, std::vector<double> H);

	/// TODO - implement
	/// Define a temperature dependent magnetic flux model
	/// \param[in] B - magnetic flux density values from B-H curve 
	/// \param[in] H - magnetic field intensity valyes from B-H curve
	/// \param *T_ - pointer to existing temperature grid function
	/// \note not currently implemented
	MagneticFluxCoefficient(std::vector<double> B, std::vector<double> H,
								   GridFunType *T_);

	/// \brief Evaluate the B-H curve in the element described by trans at
	/// the point ip. Checks which model was initialized, temperature-dependent
	/// or not, and evalutes the correct one.
	/// \param[in] state - H
	/// \returns B
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
	double Eval(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip,
					const double state) override;

	/// \brief Evaluate the permeability in the element described by trans at
	/// the point ip. Checks which model was initialized, temperature-dependent
	/// or not, and evalutes the correct one.
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
	double EvalStateDeriv(mfem::ElementTransformation &trans,
                       	 const mfem::IntegrationPoint &ip,
								 const double state) override;

	/// class destructor. Not sure if I need to delete anything?
	~MagneticFluxCoefficient() = default;

protected:
	/// reference to temperature grid function
	GridFunType *temperature_GF;

	/// spline representing B-H curve, 1st deriv is reluctivity
	Spline b_h_curve;
};

class VectorMeshDependentCoefficient : public mfem::VectorCoefficient
{
public:
	VectorMeshDependentCoefficient(const int dim = 3,
											 std::unique_ptr<mfem::VectorCoefficient>
												dflt = NULL)
		 : VectorCoefficient(dim), default_coeff(move(dflt)) {}

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

   /// \brief Search the map of coefficients and evaluate the one whose key is
   ///        the same as the element's `Attribute` at the point defined by
   ///        `ip`.
   /// \param[in] V_bar - derivative of functional with respect to `V`
   /// \param[in] trans - element transformation relating real element to
   ///                    reference element
   /// \param[in] ip - defines location in reference space
   /// \param[out] PointMat_bar - derivative of function w.r.t. mesh nodes
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
   virtual void EvalRevDiff(const mfem::Vector &V_bar,
                            mfem::ElementTransformation &trans,
                            const mfem::IntegrationPoint &ip,
                            mfem::DenseMatrix &PointMat_bar);

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

class SteinmetzCoefficient : public mfem::Coefficient
{
public:
	/// Define a coefficient to represent the Steinmetz core losses
	/// \param[in] rho - TODO: material density?
	/// \param[in] alpha - TODO
	/// \param[in] f - electrical frequency of excitation
	/// \param[in] kh - Steinmetz hysteresis coefficient
	/// \param[in] ke - Steinmetz eddy currnt coefficient
	/// \param[in] A - magnetic vector potential GridFunction 
	SteinmetzCoefficient(double rho, double alpha, double f, double kh,
								double ke, mfem::ParGridFunction &A)
		: rho(rho), alpha(alpha), freq(f), kh(kh), ke(ke), A(A) {}

	/// Evaluate the Steinmetz coefficient
	double Eval(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip) override;

	/// Evaluate the derivative of the Steinmetz coefficient with respect to x
	void EvalRevDiff(const double Q_bar,
    					  mfem::ElementTransformation &trans,
    					  const mfem::IntegrationPoint &ip,
    					  mfem::DenseMatrix &PointMat_bar) override;

private:
	double rho, alpha, freq, kh, ke;
	mfem::ParGridFunction &A;
};

class SteinmetzVectorDiffCoefficient : public mfem::VectorCoefficient
{
public:
	/// Define a coefficient to represent the Steinmetz core losses differentiated
	/// with respect to the magnetic vector potential
	/// \param[in] rho - TODO: material density?
	/// \param[in] alpha - TODO
	/// \param[in] f - electrical frequency of excitation
	/// \param[in] kh - Steinmetz hysteresis coefficient
	/// \param[in] ke - Steinmetz eddy currnt coefficient
	/// \param[in] A - magnetic vector potential GridFunction 
	/// \note this coefficient only works on meshes with only one element type
	SteinmetzVectorDiffCoefficient(double rho, double alpha, double f,
											 double kh, double ke, mfem::GridFunction &A)
		: VectorCoefficient(A.FESpace()->GetFE(0)->GetDof()), rho(rho),
		  alpha(alpha), freq(f), kh(kh), ke(ke), A(A) {}

	/// Evaluate the derivative of the Steinmetz coefficient with respect to A
	void Eval(mfem::Vector &V, mfem::ElementTransformation &T,
             const mfem::IntegrationPoint &ip) override;

private:
	double rho, alpha, freq, kh, ke;
	mfem::GridFunction &A;
};

/// ElementFunctionCoefficient
/// A class that maps coefficients as functions of the element they are in.
/// Used to set stiffness of elements for mesh movement.
class ElementFunctionCoefficient : public mfem::Coefficient
{
public:
	/// Construct ElementFunctionCoefficient
	/// \param [in] dflt - default coefficient to evaluate if element attribute
	///						  is not found in the map. If not set, will default
	///						  to zero
	ElementFunctionCoefficient(double (*f)(const mfem::Vector &, int)) 
	{
		Function = f;
		TDFunction = NULL;
	}

	// Time Dependent Version
	ElementFunctionCoefficient(double (*tdf)(const mfem::Vector &, int, double)) 
	{
		Function = NULL;
		TDFunction = tdf;
	}

	/// \brief Get element number from the transformation and accept as argument
	/// 		for the given function coefficient.
	/// \param[in] trans - element transformation relating real element to
	///					 	  reference element
	/// \param[in] ip - the integration point to evalaute the coefficient at
   /// \note When this method is called, the caller must make sure that the
   /// IntegrationPoint associated with trans is the same as ip. This can be
   /// achieved by calling trans.SetIntPoint(&ip).
	virtual double Eval(mfem::ElementTransformation &trans,
							  const mfem::IntegrationPoint &ip);

protected:
	double (*Function)(const mfem::Vector &, int);
	double (*TDFunction)(const mfem::Vector &, int, double);
private:

};

class LameFirstParameter: public mfem::Coefficient
{
public:
   double Eval(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip) override
   {
      return 1.0 / trans.Weight();
   }

};

class LameSecondParameter: public mfem::Coefficient
{
public:
   double Eval(mfem::ElementTransformation &trans,
               const mfem::IntegrationPoint &ip) override
   {
      return 1.0;
   }

};

} // namespace mach

#endif
