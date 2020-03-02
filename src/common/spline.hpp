#ifndef MACH_SPLINE_H
#define MACH_SPLINE_H

namespace mach
{

/// Class for cubic spline interpolation
class Spline
{
public:
   /// Enum for boundary type, if either the first or second derivative is
   /// specified at the boundary
   enum bnd_type
   {
      first_deriv = 1,
      second_deriv = 2
   };

   /// Construct Spline 
   /// by default constuct a natual spline (zero curvature at both ends)
   Spline()
      : left_bnd(second_deriv), right_bnd(second_deriv),
        left_bnd_value(0.0), right_bnd_value(0.0),
        linear_extrap(false) {}

   /// \brief Optional function to set first or second derivative values at
   ///        each boundary
   /// \note must be called before set_points()
   void set_boundary(bnd_type left_boundary, double left_boundary_value,
                     bnd_type right_boundary, double right_boundary_value,
                     bool linear_extrap = false);

   /// \breif Sets the points the spline will interpolate, and solves linear
   ///        system for the spline coeffients
   /// \param[in] x_data - standard vector containing x data points
   /// \param[in] y_data - standard vector containing y data points
   /// \param[in] cubic - boolean value indicating wether to create a cubic
   ///                    spline. If false will linearly interpolate instead
   void set_points(const std::vector<double>& x_data,
                  const std::vector<double>& y_data, bool cubic = true);
   
   /// \brief Evaluate the spline interpolant
   /// \param[in] x_eval - x coordinate to evaluate spline at
   double operator() (double x_eval) const;

   /// \brief Evaluate the derivative of the spline interpolant
   /// \param[in] order - derivative order
   /// \param[in] x_eval - x coordinate to evaluate derivative at
   double deriv(int order, double x_eval) const;

private:
   /// x and y coordinates of points for spline
   std::vector<double> x, y;
   /// Spline coefficients in interpolation function 
   /// f(x) = a(x-x_i)^3 + b(x-x_i)^2 + c(x-x_i) + y_i
   std::vector<double> a, b, c;
   /// spline coefficients used for left extrapolation
   double b0, c0;
   /// left and right boundary type indicators
   bnd_type left_bnd, right_bnd;
   /// left and right boundary values
   double left_bnd_value, right_bnd_value;
   /// flag to use linear extrapolation instead of quadratic
   bool linear_extrap;
};

} // namespace mach

#endif