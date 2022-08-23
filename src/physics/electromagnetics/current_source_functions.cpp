#include <memory>
#include <string>
#include <vector>

#include "adept.h"
#include "mfem.hpp"

#include "current_source_functions.hpp"

namespace
{
using adept::adouble;

template <typename xdouble = double>
void cw_stator_current(int n_slots,
                       double stack_length,
                       const xdouble *x,
                       xdouble *J)
{
   J[0] = 0.0;
   J[1] = 0.0;
   J[2] = 0.0;

   xdouble zb = -stack_length / 2;  // bottom of stator
   xdouble zt = stack_length / 2;   // top of stator

   // compute theta from x and y
   xdouble tha = atan2(x[1], x[0]);
   xdouble thw = 2 * M_PI / n_slots;  // total angle of slot

   // check which winding we're in
   xdouble w = round(tha / thw);  // current slot
   xdouble th = tha - w * thw;

   // check if we're in the stator body
   if (x[2] >= zb && x[2] <= zt)
   {
      // check if we're in left or right half
      if (th > 0)
      {
         J[2] = -1;  // set to 1 for now, and direction depends on current
                     // direction
      }
      if (th < 0)
      {
         J[2] = 1;
      }
   }
   else  // outside of the stator body, check if above or below
   {
      // 'subtract' z position to 0 depending on if above or below
      xdouble rx[] = {x[0], x[1], x[2]};
      if (x[2] > zt)
      {
         rx[2] -= zt;
      }
      if (x[2] < zb)
      {
         rx[2] -= zb;
      }

      // draw top rotation axis
      xdouble ax[] = {0.0, 0.0, 0.0};
      ax[0] = cos(w * thw);
      ax[1] = sin(w * thw);

      // take x cross ax, normalize
      J[0] = rx[1] * ax[2] - rx[2] * ax[1];
      J[1] = rx[2] * ax[0] - rx[0] * ax[2];
      J[2] = rx[0] * ax[1] - rx[1] * ax[0];
      xdouble norm_J = sqrt(J[0] * J[0] + J[1] * J[1] + J[2] * J[2]);
      J[0] /= norm_J;
      J[1] /= norm_J;
      J[2] /= norm_J;
   }
   J[0] *= -1.0;
   J[1] *= -1.0;
   J[2] *= -1.0;
}

template <typename xdouble = double>
void ccw_stator_current(int n_slots,
                        double stack_length,
                        const xdouble *x,
                        xdouble *J)
{
   J[0] = 0.0;
   J[1] = 0.0;
   J[2] = 0.0;

   xdouble zb = -stack_length / 2;  // bottom of stator
   xdouble zt = stack_length / 2;   // top of stator

   // compute theta from x and y
   xdouble tha = atan2(x[1], x[0]);
   xdouble thw = 2 * M_PI / n_slots;  // total angle of slot

   // check which winding we're in
   xdouble w = round(tha / thw);  // current slot
   xdouble th = tha - w * thw;

   // check if we're in the stator body
   if (x[2] >= zb && x[2] <= zt)
   {
      // check if we're in left or right half
      if (th > 0)
      {
         J[2] = -1;  // set to 1 for now, and direction depends on current
                     // direction
      }
      if (th < 0)
      {
         J[2] = 1;
      }
   }
   else  // outside of the stator body, check if above or below
   {
      // 'subtract' z position to 0 depending on if above or below
      xdouble rx[] = {x[0], x[1], x[2]};
      if (x[2] > zt)
      {
         rx[2] -= zt;
      }
      if (x[2] < zb)
      {
         rx[2] -= zb;
      }

      // draw top rotation axis
      xdouble ax[] = {0.0, 0.0, 0.0};
      ax[0] = cos(w * thw);
      ax[1] = sin(w * thw);

      // take x cross ax, normalize
      J[0] = rx[1] * ax[2] - rx[2] * ax[1];
      J[1] = rx[2] * ax[0] - rx[0] * ax[2];
      J[2] = rx[0] * ax[1] - rx[1] * ax[0];
      xdouble norm_J = sqrt(J[0] * J[0] + J[1] * J[1] + J[2] * J[2]);
      J[0] /= norm_J;
      J[1] /= norm_J;
      J[2] /= norm_J;
   }
}

template <typename xdouble = double, int sign = 1>
void x_axis_current(const xdouble *x, xdouble *J)
{
   J[0] = sign;
   J[1] = 0.0;
   J[2] = 0.0;
}

template <typename xdouble = double, int sign = 1>
void y_axis_current(const xdouble *x, xdouble *J)
{
   J[0] = 0.0;
   J[1] = sign;
   J[2] = 0.0;
}

template <typename xdouble = double, int sign = 1>
void z_axis_current(const xdouble *x, xdouble *J)
{
   J[0] = 0.0;
   J[1] = 0.0;
   J[2] = sign;
}

template <typename xdouble = double>
void ring_current(const xdouble *x, xdouble *J)
{
   for (int i = 0; i < 3; ++i)
   {
      J[i] = 0.0;
   }
   xdouble r[] = {0.0, 0.0, 0.0};
   r[0] = x[0];
   r[1] = x[1];
   xdouble norm_r = sqrt(r[0] * r[0] + r[1] * r[1]);
   r[0] /= norm_r;
   r[1] /= norm_r;
   J[0] = -r[1];
   J[1] = r[0];
}

template <typename xdouble = double>
void box1_current(const xdouble *x, xdouble *J)
{
   for (int i = 0; i < 3; ++i)
   {
      J[i] = 0.0;
   }

   xdouble y = x[1] - .5;

   // J[2] = -current_density*6*y*(1/(M_PI*4e-7)); // for real scaled problem
   J[2] = -6 * y;
}

template <typename xdouble = double>
void box2_current(const xdouble *x, xdouble *J)
{
   for (int i = 0; i < 3; ++i)
   {
      J[i] = 0.0;
   }

   xdouble y = x[1] - .5;

   // J[2] = current_density*6*y*(1/(M_PI*4e-7)); // for real scaled problem
   J[2] = 6 * y;
}

template <typename xdouble = double>
xdouble box1_current2D(const xdouble *x)
{
   auto y = x[1] - .5;

   // J[2] = -current_density*6*y*(1/(M_PI*4e-7)); // for real scaled problem
   return -6 * y;
}

template <typename xdouble = double>
xdouble box2_current2D(const xdouble *x)
{
   auto y = x[1] - .5;

   // J[2] = current_density*6*y*(1/(M_PI*4e-7)); // for real scaled problem
   return 6 * y;
}

/// function to get the sign of a number
template <typename T>
int sgn(T val)
{
   return (T(0) < val) - (val < T(0));
}

template <typename xdouble = double>
void team13_current(const xdouble *X, xdouble *J)
{
   for (int i = 0; i < 3; ++i)
   {
      J[i] = 0.0;
   }

   auto x = X[0];
   auto y = X[1];

   if (y >= -0.075 && y <= 0.075)
   {
      J[1] = sgn(x);
   }
   else if (x >= -0.075 && x <= 0.075)
   {
      J[0] = -sgn(y);
   }
   else if (x > 0.075 && y > 0.075)
   {
      J[0] = -(y - 0.075);
      J[1] = (x - 0.075);
   }
   else if (x < 0.075 && y > 0.075)
   {
      J[0] = -(y - 0.075);
      J[1] = (x + 0.075);
   }
   else if (x < 0.075 && y < 0.075)
   {
      J[0] = -(y + 0.075);
      J[1] = (x + 0.075);
   }
   else if (x > 0.075 && y < 0.075)
   {
      J[0] = -(y + 0.075);
      J[1] = (x - 0.075);
   }

   auto norm = sqrt(J[0] * J[0] + J[1] * J[1]);

   J[0] /= norm;
   J[1] /= norm;
}

/// function describing current density in clock-wise wound windings
/// \param[in] n_slots - number of slots in the stator
/// \param[in] stack_length - axial depth of the stator
/// \param[in] x - position x in space of evaluation
/// \param[out] J - current density at position x
void cwStatorCurrentSource(int n_slots,
                           double stack_length,
                           const mfem::Vector &x,
                           mfem::Vector &J)
{
   cw_stator_current(n_slots, stack_length, x.GetData(), J.GetData());
}

void cwStatorCurrentSourceRevDiff(adept::Stack &diff_stack,
                                  int n_slots,
                                  double stack_length,
                                  const mfem::Vector &x,
                                  const mfem::Vector &V_bar,
                                  mfem::Vector &x_bar)
{
   mfem::DenseMatrix source_jac(3);
   // declare vectors of active input variables
   std::vector<adouble> x_a(x.Size());
   // copy data from mfem::Vector
   adept::set_values(x_a.data(), x.Size(), x.GetData());
   // start recording
   diff_stack.new_recording();
   // the depedent variable must be declared after the recording
   std::vector<adouble> J_a(x.Size());
   cw_stator_current<adouble>(n_slots, stack_length, x_a.data(), J_a.data());
   // set the independent and dependent variable
   diff_stack.independent(x_a.data(), x.Size());
   diff_stack.dependent(J_a.data(), x.Size());
   // calculate the jacobian w.r.t position
   diff_stack.jacobian(source_jac.GetData());
   source_jac.MultTranspose(V_bar, x_bar);
}

/// function describing current density in counter-clock-wise wound windings
/// \param[in] n_slots - number of slots in the stator
/// \param[in] stack_length - axial depth of the stator
/// \param[in] x - position x in space of evaluation
/// \param[out] J - current density at position x
void ccwStatorCurrentSource(int n_slots,
                            double stack_length,
                            const mfem::Vector &x,
                            mfem::Vector &J)
{
   ccw_stator_current(n_slots, stack_length, x.GetData(), J.GetData());
}

void ccwStatorCurrentSourceRevDiff(adept::Stack &diff_stack,
                                   int n_slots,
                                   double stack_length,
                                   const mfem::Vector &x,
                                   const mfem::Vector &V_bar,
                                   mfem::Vector &x_bar)
{
   mfem::DenseMatrix source_jac(3);
   // declare vectors of active input variables
   std::vector<adouble> x_a(x.Size());
   // copy data from mfem::Vector
   adept::set_values(x_a.data(), x.Size(), x.GetData());
   // start recording
   diff_stack.new_recording();
   // the depedent variable must be declared after the recording
   std::vector<adouble> J_a(x.Size());
   ccw_stator_current<adouble>(n_slots, stack_length, x_a.data(), J_a.data());
   // set the independent and dependent variable
   diff_stack.independent(x_a.data(), x.Size());
   diff_stack.dependent(J_a.data(), x.Size());
   // calculate the jacobian w.r.t position
   diff_stack.jacobian(source_jac.GetData());
   source_jac.MultTranspose(V_bar, x_bar);
}

void xAxisCurrentSource(const mfem::Vector &x, mfem::Vector &J)
{
   x_axis_current(x.GetData(), J.GetData());
}

void xAxisCurrentSourceRevDiff(adept::Stack &diff_stack,
                               const mfem::Vector &x,
                               const mfem::Vector &V_bar,
                               mfem::Vector &x_bar)
{
   mfem::DenseMatrix source_jac(3);
   // declare vectors of active input variables
   std::vector<adouble> x_a(x.Size());
   // copy data from mfem::Vector
   adept::set_values(x_a.data(), x.Size(), x.GetData());
   // start recording
   diff_stack.new_recording();
   // the depedent variable must be declared after the recording
   std::vector<adouble> J_a(x.Size());
   x_axis_current<adouble>(x_a.data(), J_a.data());
   // set the independent and dependent variable
   diff_stack.independent(x_a.data(), x.Size());
   diff_stack.dependent(J_a.data(), x.Size());
   // calculate the jacobian w.r.t state vaiables
   diff_stack.jacobian(source_jac.GetData());
   source_jac.MultTranspose(V_bar, x_bar);
}

void yAxisCurrentSource(const mfem::Vector &x, mfem::Vector &J)
{
   y_axis_current(x.GetData(), J.GetData());
}

void yAxisCurrentSourceRevDiff(adept::Stack &diff_stack,
                               const mfem::Vector &x,
                               const mfem::Vector &V_bar,
                               mfem::Vector &x_bar)
{
   mfem::DenseMatrix source_jac(3);
   // declare vectors of active input variables
   std::vector<adouble> x_a(x.Size());
   // copy data from mfem::Vector
   adept::set_values(x_a.data(), x.Size(), x.GetData());
   // start recording
   diff_stack.new_recording();
   // the depedent variable must be declared after the recording
   std::vector<adouble> J_a(x.Size());
   y_axis_current<adouble>(x_a.data(), J_a.data());
   // set the independent and dependent variable
   diff_stack.independent(x_a.data(), x.Size());
   diff_stack.dependent(J_a.data(), x.Size());
   // calculate the jacobian w.r.t state vaiables
   diff_stack.jacobian(source_jac.GetData());
   source_jac.MultTranspose(V_bar, x_bar);
}

void zAxisCurrentSource(const mfem::Vector &x, mfem::Vector &J)
{
   z_axis_current(x.GetData(), J.GetData());
}

void zAxisCurrentSourceRevDiff(adept::Stack &diff_stack,
                               const mfem::Vector &x,
                               const mfem::Vector &V_bar,
                               mfem::Vector &x_bar)
{
   mfem::DenseMatrix source_jac(3);
   // declare vectors of active input variables
   std::vector<adouble> x_a(x.Size());
   // copy data from mfem::Vector
   adept::set_values(x_a.data(), x.Size(), x.GetData());
   // start recording
   diff_stack.new_recording();
   // the depedent variable must be declared after the recording
   std::vector<adouble> J_a(x.Size());
   z_axis_current<adouble>(x_a.data(), J_a.data());
   // set the independent and dependent variable
   diff_stack.independent(x_a.data(), x.Size());
   diff_stack.dependent(J_a.data(), x.Size());
   // calculate the jacobian w.r.t state vaiables
   diff_stack.jacobian(source_jac.GetData());
   source_jac.MultTranspose(V_bar, x_bar);
}

void nzAxisCurrentSource(const mfem::Vector &x, mfem::Vector &J)
{
   z_axis_current<double, -1>(x.GetData(), J.GetData());
}

void nzAxisCurrentSourceRevDiff(adept::Stack &diff_stack,
                                const mfem::Vector &x,
                                const mfem::Vector &V_bar,
                                mfem::Vector &x_bar)
{
   mfem::DenseMatrix source_jac(3);
   // declare vectors of active input variables
   std::vector<adouble> x_a(x.Size());
   // copy data from mfem::Vector
   adept::set_values(x_a.data(), x.Size(), x.GetData());
   // start recording
   diff_stack.new_recording();
   // the depedent variable must be declared after the recording
   std::vector<adouble> J_a(x.Size());
   z_axis_current<adouble, -1>(x_a.data(), J_a.data());
   // set the independent and dependent variable
   diff_stack.independent(x_a.data(), x.Size());
   diff_stack.dependent(J_a.data(), x.Size());
   // calculate the jacobian w.r.t state vaiables
   diff_stack.jacobian(source_jac.GetData());
   source_jac.MultTranspose(V_bar, x_bar);
}

void ringCurrentSource(const mfem::Vector &x, mfem::Vector &J)
{
   ring_current(x.GetData(), J.GetData());
}

void ringCurrentSourceRevDiff(adept::Stack &diff_stack,
                              const mfem::Vector &x,
                              const mfem::Vector &V_bar,
                              mfem::Vector &x_bar)
{
   mfem::DenseMatrix source_jac(3);
   // declare vectors of active input variables
   std::vector<adouble> x_a(x.Size());
   // copy data from mfem::Vector
   adept::set_values(x_a.data(), x.Size(), x.GetData());
   // start recording
   diff_stack.new_recording();
   // the depedent variable must be declared after the recording
   std::vector<adouble> J_a(x.Size());
   ring_current<adouble>(x_a.data(), J_a.data());
   // set the independent and dependent variable
   diff_stack.independent(x_a.data(), x.Size());
   diff_stack.dependent(J_a.data(), x.Size());
   // calculate the jacobian w.r.t state vaiables
   diff_stack.jacobian(source_jac.GetData());
   source_jac.MultTranspose(V_bar, x_bar);
}

void box1CurrentSource(const mfem::Vector &x, mfem::Vector &J)
{
   box1_current(x.GetData(), J.GetData());
}

void box1CurrentSourceRevDiff(adept::Stack &diff_stack,
                              const mfem::Vector &x,
                              const mfem::Vector &V_bar,
                              mfem::Vector &x_bar)
{
   mfem::DenseMatrix source_jac(3);
   // declare vectors of active input variables
   std::vector<adouble> x_a(x.Size());
   // copy data from mfem::Vector
   adept::set_values(x_a.data(), x.Size(), x.GetData());
   // start recording
   diff_stack.new_recording();
   // the depedent variable must be declared after the recording
   std::vector<adouble> J_a(x.Size());
   box1_current<adouble>(x_a.data(), J_a.data());
   // set the independent and dependent variable
   diff_stack.independent(x_a.data(), x.Size());
   diff_stack.dependent(J_a.data(), x.Size());
   // calculate the jacobian w.r.t state vaiables
   diff_stack.jacobian(source_jac.GetData());
   source_jac.MultTranspose(V_bar, x_bar);
}

void box2CurrentSource(const mfem::Vector &x, mfem::Vector &J)
{
   box2_current(x.GetData(), J.GetData());
}

void box2CurrentSourceRevDiff(adept::Stack &diff_stack,
                              const mfem::Vector &x,
                              const mfem::Vector &V_bar,
                              mfem::Vector &x_bar)
{
   mfem::DenseMatrix source_jac(3);
   // declare vectors of active input variables
   std::vector<adouble> x_a(x.Size());
   // copy data from mfem::Vector
   adept::set_values(x_a.data(), x.Size(), x.GetData());
   // start recording
   diff_stack.new_recording();
   // the depedent variable must be declared after the recording
   std::vector<adouble> J_a(x.Size());
   box2_current<adouble>(x_a.data(), J_a.data());
   // set the independent and dependent variable
   diff_stack.independent(x_a.data(), x.Size());
   diff_stack.dependent(J_a.data(), x.Size());
   // calculate the jacobian w.r.t state vaiables
   diff_stack.jacobian(source_jac.GetData());
   source_jac.MultTranspose(V_bar, x_bar);
}

double box1CurrentSource2D(const mfem::Vector &x)
{
   return box1_current2D(x.GetData());
}

void box1CurrentSource2DRevDiff(adept::Stack &diff_stack,
                                const mfem::Vector &x,
                                const double &J_bar,
                                mfem::Vector &x_bar)
{
   // mfem::DenseMatrix source_jac(3);
   // // declare vectors of active input variables
   // std::vector<adouble> x_a(x.Size());
   // // copy data from mfem::Vector
   // adept::set_values(x_a.data(), x.Size(), x.GetData());
   // // start recording
   // diff_stack.new_recording();
   // // the depedent variable must be declared after the recording
   // std::vector<adouble> J_a(x.Size());
   // box1_current<adouble>(x_a.data(), J_a.data());
   // // set the independent and dependent variable
   // diff_stack.independent(x_a.data(), x.Size());
   // diff_stack.dependent(J_a.data(), x.Size());
   // // calculate the jacobian w.r.t state vaiables
   // diff_stack.jacobian(source_jac.GetData());
   // source_jac.MultTranspose(V_bar, x_bar);
}

double box2CurrentSource2D(const mfem::Vector &x)
{
   return box2_current2D(x.GetData());
}

void box2CurrentSource2DRevDiff(adept::Stack &diff_stack,
                                const mfem::Vector &x,
                                const double &J_bar,
                                mfem::Vector &x_bar)
{
   // mfem::DenseMatrix source_jac(3);
   // // declare vectors of active input variables
   // std::vector<adouble> x_a(x.Size());
   // // copy data from mfem::Vector
   // adept::set_values(x_a.data(), x.Size(), x.GetData());
   // // start recording
   // diff_stack.new_recording();
   // // the depedent variable must be declared after the recording
   // adouble J_a = box2_current2D<adouble>(x_a.data());
   // // set the independent and dependent variable
   // diff_stack.independent(x_a.data(), x.Size());
   // diff_stack.dependent(J_a);
   // // calculate the jacobian w.r.t state vaiables
   // diff_stack.jacobian(source_jac.GetData());
   // source_jac.MultTranspose(V_bar, x_bar);
}

void team13CurrentSource(const mfem::Vector &x, mfem::Vector &J)
{
   team13_current(x.GetData(), J.GetData());
}

void team13CurrentSourceRevDiff(adept::Stack &diff_stack,
                                const mfem::Vector &x,
                                const mfem::Vector &V_bar,
                                mfem::Vector &x_bar)
{
   mfem::DenseMatrix source_jac(3);
   // declare vectors of active input variables
   std::vector<adouble> x_a(x.Size());
   // copy data from mfem::Vector
   adept::set_values(x_a.data(), x.Size(), x.GetData());
   // start recording
   diff_stack.new_recording();
   // the depedent variable must be declared after the recording
   std::vector<adouble> J_a(x.Size());
   team13_current<adouble>(x_a.data(), J_a.data());
   // set the independent and dependent variable
   diff_stack.independent(x_a.data(), x.Size());
   diff_stack.dependent(J_a.data(), x.Size());
   // calculate the jacobian w.r.t state vaiables
   diff_stack.jacobian(source_jac.GetData());
   source_jac.MultTranspose(V_bar, x_bar);
}

}  // anonymous namespace

namespace mach
{
void CurrentDensityCoefficient::cacheCurrentDensity()
{
   for (auto &[group, coeff] : group_map)
   {
      cached_inputs.at(group) = coeff.constant;
   }
}

void CurrentDensityCoefficient::zeroCurrentDensity()
{
   for (auto &[group, coeff] : group_map)
   {
      coeff.constant = 0.0;
   }
}

void CurrentDensityCoefficient::resetCurrentDensityFromCache()
{
   for (auto &[group, value] : cached_inputs)
   {
      group_map.at(group).constant = value;
   }
}

bool setInputs(CurrentDensityCoefficient &current, const MachInputs &inputs)
{
   bool updated = false;
   for (auto &[group, coeff] : current.group_map)
   {
      auto old_const = coeff.constant;
      std::string cd_group_id = "current_density:" + group;
      setValueFromInputs(inputs, cd_group_id, coeff.constant);
      if (coeff.constant != old_const)
      {
         updated = true;
      }
   }

   for (auto &[input, value] : current.cached_inputs)
   {
      auto old_value = value;
      setValueFromInputs(inputs, input, value);
      if (value != old_value)
      {
         updated = true;
      }
   }
   return updated;
}

void CurrentDensityCoefficient::Eval(mfem::Vector &V,
                                     mfem::ElementTransformation &trans,
                                     const mfem::IntegrationPoint &ip)
{
   current_coeff.Eval(V, trans, ip);
}

void CurrentDensityCoefficient::EvalRevDiff(const mfem::Vector &V_bar,
                                            mfem::ElementTransformation &trans,
                                            const mfem::IntegrationPoint &ip,
                                            mfem::DenseMatrix &PointMat_bar)
{
   current_coeff.EvalRevDiff(V_bar, trans, ip, PointMat_bar);
}

CurrentDensityCoefficient::CurrentDensityCoefficient(
    adept::Stack &diff_stack,
    const nlohmann::json &current_options,
    int vdim)
 : mfem::VectorCoefficient(vdim), current_coeff(vdim)
{
   for (auto &[group, group_details] : current_options.items())
   {
      group_map.emplace(group, mfem::ConstantCoefficient{1.0});
      auto &group_coeff = group_map.at(group);

      cached_inputs.emplace(group, 0.0);

      for (auto &[source, attrs] : group_details.items())
      {
         if (source == "cwStator")
         {
            cached_inputs.emplace("n_slots", 24.0);
            auto n_slots = cached_inputs.at("n_slots");
            cached_inputs.emplace("stack_length", 0.345);
            auto stack_length = cached_inputs.at("stack_length");
            for (const auto &attr : attrs)
            {
               source_coeffs.emplace(
                   attr,
                   mfem::VectorFunctionCoefficient(
                       vdim,
                       [&n_slots, &stack_length](const mfem::Vector &x,
                                                 mfem::Vector &J)
                       { cwStatorCurrentSource(n_slots, stack_length, x, J); },
                       [&diff_stack, &n_slots, &stack_length](
                           const mfem::Vector &x,
                           const mfem::Vector &J_bar,
                           mfem::Vector &x_bar)
                       {
                          cwStatorCurrentSourceRevDiff(diff_stack,
                                                       n_slots,
                                                       stack_length,
                                                       x,
                                                       J_bar,
                                                       x_bar);
                       }));
               auto &source_coeff = source_coeffs.at(attr);

               current_coeff.addCoefficient(
                   attr,
                   std::make_unique<mfem::ScalarVectorProductCoefficient>(
                       group_coeff, source_coeff));
            }
         }
         else if (source == "ccwStator")
         {
            cached_inputs.emplace("n_slots", 24.0);
            auto n_slots = cached_inputs.at("n_slots");
            cached_inputs.emplace("stack_length", 0.345);
            auto stack_length = cached_inputs.at("stack_length");
            for (const auto &attr : attrs)
            {
               source_coeffs.emplace(
                   attr,
                   mfem::VectorFunctionCoefficient(
                       vdim,
                       [&n_slots, &stack_length](const mfem::Vector &x,
                                                 mfem::Vector &J)
                       { ccwStatorCurrentSource(n_slots, stack_length, x, J); },
                       [&diff_stack, &n_slots, &stack_length](
                           const mfem::Vector &x,
                           const mfem::Vector &J_bar,
                           mfem::Vector &x_bar)
                       {
                          ccwStatorCurrentSourceRevDiff(diff_stack,
                                                        n_slots,
                                                        stack_length,
                                                        x,
                                                        J_bar,
                                                        x_bar);
                       }));
               auto &source_coeff = source_coeffs.at(attr);

               current_coeff.addCoefficient(
                   attr,
                   std::make_unique<mfem::ScalarVectorProductCoefficient>(
                       group_coeff, source_coeff));
            }
         }
         else if (source == "x")
         {
            for (const auto &attr : attrs)
            {
               source_coeffs.emplace(
                   attr,
                   mfem::VectorFunctionCoefficient(
                       vdim,
                       xAxisCurrentSource,
                       [&diff_stack](const mfem::Vector &x,
                                     const mfem::Vector &J_bar,
                                     mfem::Vector &x_bar) {
                          xAxisCurrentSourceRevDiff(
                              diff_stack, x, J_bar, x_bar);
                       }));
               auto &source_coeff = source_coeffs.at(attr);

               current_coeff.addCoefficient(
                   attr,
                   std::make_unique<mfem::ScalarVectorProductCoefficient>(
                       group_coeff, source_coeff));
            }
         }
         else if (source == "y")
         {
            for (const auto &attr : attrs)
            {
               source_coeffs.emplace(
                   attr,
                   mfem::VectorFunctionCoefficient(
                       vdim,
                       yAxisCurrentSource,
                       [&diff_stack](const mfem::Vector &x,
                                     const mfem::Vector &J_bar,
                                     mfem::Vector &x_bar) {
                          yAxisCurrentSourceRevDiff(
                              diff_stack, x, J_bar, x_bar);
                       }));
               auto &source_coeff = source_coeffs.at(attr);

               current_coeff.addCoefficient(
                   attr,
                   std::make_unique<mfem::ScalarVectorProductCoefficient>(
                       group_coeff, source_coeff));
            }
         }
         else if (source == "z")
         {
            for (const auto &attr : attrs)
            {
               source_coeffs.emplace(
                   attr,
                   mfem::VectorFunctionCoefficient(
                       vdim,
                       zAxisCurrentSource,
                       [&diff_stack](const mfem::Vector &x,
                                     const mfem::Vector &J_bar,
                                     mfem::Vector &x_bar) {
                          zAxisCurrentSourceRevDiff(
                              diff_stack, x, J_bar, x_bar);
                       }));
               auto &source_coeff = source_coeffs.at(attr);

               current_coeff.addCoefficient(
                   attr,
                   std::make_unique<mfem::ScalarVectorProductCoefficient>(
                       group_coeff, source_coeff));
            }
         }
         else if (source == "-z")
         {
            for (const auto &attr : attrs)
            {
               source_coeffs.emplace(
                   attr,
                   mfem::VectorFunctionCoefficient(
                       vdim,
                       nzAxisCurrentSource,
                       [&diff_stack](const mfem::Vector &x,
                                     const mfem::Vector &J_bar,
                                     mfem::Vector &x_bar) {
                          nzAxisCurrentSourceRevDiff(
                              diff_stack, x, J_bar, x_bar);
                       }));
               auto &source_coeff = source_coeffs.at(attr);

               current_coeff.addCoefficient(
                   attr,
                   std::make_unique<mfem::ScalarVectorProductCoefficient>(
                       group_coeff, source_coeff));
            }
         }
         else if (source == "ring")
         {
            for (const auto &attr : attrs)
            {
               source_coeffs.emplace(
                   attr,
                   mfem::VectorFunctionCoefficient(
                       vdim,
                       ringCurrentSource,
                       [&diff_stack](const mfem::Vector &x,
                                     const mfem::Vector &J_bar,
                                     mfem::Vector &x_bar) {
                          ringCurrentSourceRevDiff(diff_stack, x, J_bar, x_bar);
                       }));
               auto &source_coeff = source_coeffs.at(attr);

               current_coeff.addCoefficient(
                   attr,
                   std::make_unique<mfem::ScalarVectorProductCoefficient>(
                       group_coeff, source_coeff));
            }
         }
         else if (source == "box1")
         {
            for (const auto &attr : attrs)
            {
               source_coeffs.emplace(
                   attr,
                   mfem::VectorFunctionCoefficient(
                       vdim,
                       box1CurrentSource,
                       [&diff_stack](const mfem::Vector &x,
                                     const mfem::Vector &J_bar,
                                     mfem::Vector &x_bar) {
                          box1CurrentSourceRevDiff(diff_stack, x, J_bar, x_bar);
                       }));
               auto &source_coeff = source_coeffs.at(attr);

               current_coeff.addCoefficient(
                   attr,
                   std::make_unique<mfem::ScalarVectorProductCoefficient>(
                       group_coeff, source_coeff));
            }
         }
         else if (source == "box2")
         {
            for (const auto &attr : attrs)
            {
               source_coeffs.emplace(
                   attr,
                   mfem::VectorFunctionCoefficient(
                       vdim,
                       box2CurrentSource,
                       [&diff_stack](const mfem::Vector &x,
                                     const mfem::Vector &J_bar,
                                     mfem::Vector &x_bar) {
                          box2CurrentSourceRevDiff(diff_stack, x, J_bar, x_bar);
                       }));
               auto &source_coeff = source_coeffs.at(attr);

               current_coeff.addCoefficient(
                   attr,
                   std::make_unique<mfem::ScalarVectorProductCoefficient>(
                       group_coeff, source_coeff));
            }
         }
         else if (source == "team13")
         {
            for (const auto &attr : attrs)
            {
               source_coeffs.emplace(
                   attr,
                   mfem::VectorFunctionCoefficient(
                       vdim,
                       team13CurrentSource,
                       [&diff_stack](const mfem::Vector &x,
                                     const mfem::Vector &J_bar,
                                     mfem::Vector &x_bar) {
                          team13CurrentSourceRevDiff(
                              diff_stack, x, J_bar, x_bar);
                       }));
               auto &source_coeff = source_coeffs.at(attr);

               current_coeff.addCoefficient(
                   attr,
                   std::make_unique<mfem::ScalarVectorProductCoefficient>(
                       group_coeff, source_coeff));
            }
         }
      }
   }
}

void CurrentDensityCoefficient2D::cacheCurrentDensity()
{
   for (auto &[group, coeff] : group_map)
   {
      cached_inputs.at(group) = coeff.constant;
   }
}

void CurrentDensityCoefficient2D::zeroCurrentDensity()
{
   for (auto &[group, coeff] : group_map)
   {
      coeff.constant = 0.0;
   }
}

void CurrentDensityCoefficient2D::resetCurrentDensityFromCache()
{
   for (auto &[group, value] : cached_inputs)
   {
      group_map.at(group).constant = value;
   }
}

bool setInputs(CurrentDensityCoefficient2D &current, const MachInputs &inputs)
{
   bool updated = false;
   for (auto &[group, coeff] : current.group_map)
   {
      auto old_const = coeff.constant;
      std::string cd_group_id = "current_density:" + group;
      setValueFromInputs(inputs, cd_group_id, coeff.constant);
      if (coeff.constant != old_const)
      {
         updated = true;
      }
   }

   for (auto &[input, value] : current.cached_inputs)
   {
      auto old_value = value;
      setValueFromInputs(inputs, input, value);
      if (value != old_value)
      {
         updated = true;
      }
   }
   return updated;
}

double CurrentDensityCoefficient2D::Eval(mfem::ElementTransformation &trans,
                                         const mfem::IntegrationPoint &ip)
{
   return -1.0 * current_coeff.Eval(trans, ip);
}

void CurrentDensityCoefficient2D::EvalRevDiff(
    double Q_bar,
    mfem::ElementTransformation &trans,
    const mfem::IntegrationPoint &ip,
    mfem::DenseMatrix &PointMat_bar)
{
   Q_bar *= 1.0;
   current_coeff.EvalRevDiff(Q_bar, trans, ip, PointMat_bar);
}

CurrentDensityCoefficient2D::CurrentDensityCoefficient2D(
    adept::Stack &diff_stack,
    const nlohmann::json &current_options)
{
   for (auto &[group, group_details] : current_options.items())
   {
      group_map.emplace(group, mfem::ConstantCoefficient{1.0});
      auto &group_coeff = group_map.at(group);

      cached_inputs.emplace(group, 0.0);

      for (auto &[source, attrs] : group_details.items())
      {
         if (source == "z")
         {
            for (const auto &attr : attrs)
            {
               source_coeffs.emplace(
                   attr,
                   mfem::FunctionCoefficient([](const mfem::Vector &x)
                                             { return 1.0; },
                                             [](const mfem::Vector &x,
                                                const double Q_bar,
                                                mfem::Vector &x_bar) {}));
               auto &source_coeff = source_coeffs.at(attr);

               current_coeff.addCoefficient(
                   attr,
                   std::make_unique<mfem::ProductCoefficient>(group_coeff,
                                                              source_coeff));
            }
         }
         else if (source == "-z")
         {
            for (const auto &attr : attrs)
            {
               // source_coeffs.emplace(attr, mfem::FunctionCoefficient(-1.0));
               source_coeffs.emplace(
                   attr,
                   mfem::FunctionCoefficient([](const mfem::Vector &)
                                             { return -1.0; },
                                             [](const mfem::Vector &x,
                                                const double Q_bar,
                                                mfem::Vector &x_bar) {}));
               auto &source_coeff = source_coeffs.at(attr);

               current_coeff.addCoefficient(
                   attr,
                   std::make_unique<mfem::ProductCoefficient>(group_coeff,
                                                              source_coeff));
            }
         }
         else if (source == "box1")
         {
            for (const auto &attr : attrs)
            {
               source_coeffs.emplace(attr,
                                     mfem::FunctionCoefficient(
                                         box1CurrentSource2D,
                                         [&diff_stack](const mfem::Vector &x,
                                                       const double &J_bar,
                                                       mfem::Vector &x_bar) {
                                            box1CurrentSource2DRevDiff(
                                                diff_stack, x, J_bar, x_bar);
                                         }));
               auto &source_coeff = source_coeffs.at(attr);

               current_coeff.addCoefficient(
                   attr,
                   std::make_unique<mfem::ProductCoefficient>(group_coeff,
                                                              source_coeff));
            }
         }
         else if (source == "box2")
         {
            for (const auto &attr : attrs)
            {
               source_coeffs.emplace(attr,
                                     mfem::FunctionCoefficient(
                                         box2CurrentSource2D,
                                         [&diff_stack](const mfem::Vector &x,
                                                       const double &J_bar,
                                                       mfem::Vector &x_bar) {
                                            box2CurrentSource2DRevDiff(
                                                diff_stack, x, J_bar, x_bar);
                                         }));
               auto &source_coeff = source_coeffs.at(attr);

               current_coeff.addCoefficient(
                   attr,
                   std::make_unique<mfem::ProductCoefficient>(group_coeff,
                                                              source_coeff));
            }
         }
      }
   }
}

}  // namespace mach
