#include <vector>
#include <algorithm>

#include "utils.hpp"

#include "spline.hpp"

namespace mach
{

/// TODO: Use mfem::DenseMatrix?
/// Banded matrix solver
class BandedMatrix
{
public:
   /// Default BandedMatrix constructor
   BandedMatrix() {};
   /// \brief Construct BandedMatrix
   /// \param[in] dim - (square) matrix dimension
   /// \param[in] n_u - number of bands above diagonal
   /// \param[in] n_l - number of bands below diagonal
   BandedMatrix(int dim, int n_u, int n_l);
   /// BandedMatrix destructor
   ~BandedMatrix() {};

   /// \brief Resize BandedMatrix
   /// \param[in] dim - (square) matrix dimension
   /// \param[in] n_u - number of bands above diagonal
   /// \param[in] n_l - number of bands below diagonal
   void resize(int dim, int n_u, int n_l);
   /// \brief return matrix dimension
   int dim() const;
   /// \brief return number of bands above diagonal
   int num_upper() const
   {
      return upper.size()-1;
   }
   /// \brief return number of bands below diagonal
   int num_lower() const
   {
      return lower.size()-1;
   }
   /// \brief overload () operator to access matrix element with write access
   /// \param[in] i - row index
   /// \param[in] j - column index
   double& operator () (int i, int j);
   /// \brief overload () operator to access matrix element with only read
   ///        access
   /// \param[in] i - row index
   /// \param[in] j - column index
   double operator () (int i, int j) const;

   double& saved_diag(int i);
   double saved_diag(int i) const;

   /// \brief factor matrix
   void decompose();

   /// \brief solve linear system
   /// \param[in] b - right hand side of linear system
   /// \param[in] is_decomposed - flag indicating if the matrix has been
   ///                            factored
   std::vector<double> solve(const std::vector<double>& b,
                              bool is_decomposed = false);
private:
   /// vector of vectors containing data in upper bands
   std::vector<std::vector<double>> upper;
   /// vector of vectors containing data in lower bands
   std::vector<std::vector<double>> lower;

   /// intermediate functions used to solve the linear system
   std::vector<double> r_solve(const std::vector<double>& b) const;
   std::vector<double> l_solve(const std::vector<double>& b) const;

};

void Spline::set_boundary(bnd_type left_boundary, double left_boundary_value,
                          bnd_type right_boundary, double right_boundary_value,
                          bool linear_extrapolation)
{
   if (x.size() != 0)
   {
      throw mach::MachException(
         "set_points() must not have been called yet");
   }
   left_bnd = left_boundary;
   right_bnd = right_boundary;
   left_bnd_value = left_boundary_value;
   right_bnd_value = right_boundary_value;
   linear_extrap = linear_extrapolation;
}


void Spline::set_points(const std::vector<double> &x_data,
                        const std::vector<double> &y_data, bool cubic)
{
   if (x_data.size() != y_data.size())
   {
      throw mach::MachException(
         "x_data and y_data data arrays must be the same size");
   }
   if (x_data.size() <= 2)
   {
      throw mach::MachException(
         "data arrays must be at least 3 elements long");  
   }
   x = x_data;
   y = y_data;
   int n = x.size();

   /// TODO: sort x and y rather than returning an error
   for(int i = 0; i < n-1; i++)
   {
      if (x[i] >= x[i+1])
      {
         throw mach::MachException(
            "x_data array must be sorted and contain unique entries!");
      }
   }

   if (cubic) // cubic spline interpolation
   {
      // setting up the matrix and right hand side of the equation system
      // for the parameters b
      BandedMatrix A(n, 1, 1);
      std::vector<double> rhs(n);
      for(int i=1; i<n-1; i++)
      {
         A(i,i-1) = 1.0/3.0*(x[i]-x[i-1]);
         A(i,i) = 2.0/3.0*(x[i+1]-x[i-1]);
         A(i,i+1) = 1.0/3.0*(x[i+1]-x[i]);
         rhs[i] = (y[i+1]-y[i])/(x[i+1]-x[i]) - (y[i]-y[i-1])/(x[i]-x[i-1]);
      }
      
      // boundary conditions
      if(left_bnd == Spline::second_deriv)
      {
         // 2*b[0] = f''
         A(0,0) = 2.0;
         A(0,1) = 0.0;
         rhs[0] = left_bnd_value;
      }
      else if (left_bnd == Spline::first_deriv)
      {
         // c[0] = f', needs to be re-expressed in terms of b:
         // (2b[0]+b[1])(x[1]-x[0]) = 3 ((y[1]-y[0])/(x[1]-x[0]) - f')
         A(0,0) = 2.0*(x[1]-x[0]);
         A(0,1) = 1.0*(x[1]-x[0]);
         rhs[0] = 3.0*((y[1]-y[0])/(x[1]-x[0]) - left_bnd_value);
      }
      else
      {
         throw mach::MachException(
            "bad boundary conditions for left side!");
      }

      if (right_bnd == Spline::second_deriv)
      {
         // 2*b[n-1] = f''
         A(n-1,n-1) = 2.0;
         A(n-1,n-2) = 0.0;
         rhs[n-1] = right_bnd_value;
      }
      else if (right_bnd == Spline::first_deriv)
      {
         // c[n-1] = f', needs to be re-expressed in terms of b:
         // (b[n-2]+2b[n-1])(x[n-1]-x[n-2])
         // = 3 (f' - (y[n-1]-y[n-2])/(x[n-1]-x[n-2]))
         A(n-1,n-1) = 2.0*(x[n-1]-x[n-2]);
         A(n-1,n-2) = 1.0*(x[n-1]-x[n-2]);
         rhs[n-1] = 3.0*(right_bnd_value - (y[n-1]-y[n-2])/(x[n-1]-x[n-2]));
      }
      else
      {
         throw mach::MachException(
            "bad boundary conditions for right side!");
      }

      // solve the equation system to obtain the parameters b
      b=A.solve(rhs);

      // calculate parameters a and c based on b
      a.resize(n);
      c.resize(n);
      for(int i = 0; i < n-1; i++)
      {
         a[i] = 1.0/3.0 * (b[i+1]-b[i]) / (x[i+1]-x[i]);
         c[i] = (y[i+1]-y[i]) / (x[i+1]-x[i])
                  - 1.0/3.0 * (2.0*b[i]+b[i+1]) * (x[i+1]-x[i]);
      }
   }
   else // linear interpolation
   {
      a.resize(n);
      b.resize(n);
      c.resize(n);
      for(int i = 0; i < n-1; i++)
      {
         a[i] = 0.0;
         b[i] = 0.0;
         c[i] = (y[i+1]-y[i]) / (x[i+1]-x[i]);
      }
   }

   // for left extrapolation coefficients
   b0 = (linear_extrap) ? 0.0 : b[0];
   c0 = c[0];

   // for the right extrapolation coefficients
   // f_{n-1}(x) = b(x-x_{n-1})^2 + c(x-x_{n-1}) + y_{n-1}
   double h = x[n-1] - x[n-2];
   // b[n-1] is determined by the boundary condition
   a[n-1] = 0.0;
   c[n-1] = 3.0*a[n-2]*h*h+2.0*b[n-2]*h+c[n-2]; // = f'_{n-2}(x_{n-1})
   if (linear_extrap)
   {
      b[n-1] = 0.0;
   }
}

double Spline::operator() (double x_eval) const
{
   size_t n = x.size();

   // find the closest point x[idx] < x, idx = 0 even if x < x[0]
   std::vector<double>::const_iterator it;
   it = std::lower_bound(x.begin(), x.end(), x_eval);
   int idx = std::max(int(it - x.begin()) - 1, 0);

   double h = x_eval - x[idx];
   double val;
   if (x_eval < x[0])
   {
      // extrapolation to the left
      val = (b0*h + c0)*h + y[0];
   }
   else if (x_eval > x[n-1])
   {
      // extrapolation to the right
      val = (b[n-1]*h + c[n-1])*h + y[n-1];
   }
   else
   {
      // interpolation
      val = ((a[idx]*h + b[idx])*h + c[idx])*h + y[idx];
   }
   return val;
}

double Spline::deriv(int order, double x_eval) const
{
   if (order <= 0)
   {
      throw mach::MachException(
         "derivative order must be positive!");
   }

   size_t n = x.size();
   // find the closest point m_x[idx] < x, idx=0 even if x<m_x[0]
   std::vector<double>::const_iterator it;
   it=std::lower_bound(x.begin(), x.end(), x_eval);
   int idx=std::max(int(it - x.begin()) - 1, 0);

   double h = x_eval - x[idx];
   double val;
   if (x_eval < x[0]) // extrapolation on the left
   {
      switch (order)
      {
      case 1:
         val = 2.0*b0*h + c0;
         break;
      case 2:
         val = 2.0*b0;
         break;
      default:
         val = 0.0;
         break;
      }
   }
   else if (x_eval > x[n-1]) // extrapolation on the right
   {
      switch (order)
      {
      case 1:
         val = 2.0*b[n-1]*h + c[n-1];
         break;
      case 2:
         val = 2.0*b[n-1];
         break;
      default:
         val = 0.0;
         break;
      }
   }
   else // interpolation
   {
      switch (order)
      {
      case 1:
         val = (3.0*a[idx]*h + 2.0*b[idx])*h + c[idx];
         break;
      case 2:
         val = 6.0*a[idx]*h + 2.0*b[idx];
         break;
      case 3:
         val = 6.0*a[idx];
         break;
      default:
         val = 0.0;
         break;
      }
   }
   return val;
}

BandedMatrix::BandedMatrix(int dim, int n_u, int n_l)
{
   resize(dim, n_u, n_l);
}

void BandedMatrix::resize(int dim, int n_u, int n_l)
{
   if (dim <= 0)
   {
      throw mach::MachException(
         "dim must be positive!");
   }
   if (n_u < 0)
   {
      throw mach::MachException(
         "n_u must not be negative!");
   }
   if (n_l < 0)
   {
      throw mach::MachException(
         "n_l must not be negative!");
   }
   upper.resize(n_u+1);
   lower.resize(n_l+1);
   for (size_t i = 0; i < upper.size(); i++)
   {
      upper[i].resize(dim);
   }
   for (size_t i = 0; i < lower.size(); i++)
   {
      lower[i].resize(dim);
   }
}
int BandedMatrix::dim() const
{
   if (upper.size() > 0)
   {
      return upper[0].size();
   }
   else
   {
      return 0;
   }
}


// defines the new operator (), so that we can access the elements
// by A(i,j), index going from i=0,...,dim()-1
double& BandedMatrix::operator()(int i, int j)
{
   int k = j - i;
   if (!((i >= 0) && (i < dim()) && (j >= 0) && (j < dim())))
   {
      throw mach::MachException(
            "bad index in operator()!");
   }
   if (!((-num_lower() <= k) && (k <= num_upper())))
   {
      throw mach::MachException(
            "bad index in operator()!");
   }
   if (k >= 0)
   {
      return upper[k][i];
   }
   else
   {
      return lower[-k][i];
   }
}

double BandedMatrix::operator()(int i, int j) const
{
   int k = j - i;
      if (!((i >= 0) && (i < dim()) && (j >= 0) && (j < dim())))
   {
      throw mach::MachException(
            "bad index in operator()!");
   }
   if (!((-num_lower() <= k) && (k <= num_upper())))
   {
      throw mach::MachException(
            "bad index in operator()!");
   }
   if (k >= 0)
   {
      return upper[k][i];
   }
   else
   {
      return lower[-k][i];
   }
}

// second diag (used in LU decomposition), saved in m_lower
double BandedMatrix::saved_diag(int i) const
{
   if (!((i >= 0) && (i < dim())))
   {
      throw mach::MachException(
            "bad index in saved_diag!");
   }
   return lower[0][i];
}
double& BandedMatrix::saved_diag(int i)
{
   if (!((i >= 0) && (i < dim())))
   {
      throw mach::MachException(
            "bad index in saved_diag!");
   }
   return lower[0][i];
}

// LR-Decomposition of a band matrix
void BandedMatrix::decompose()
{
   int i_max, j_max;
   int j_min;
   double x;

   // preconditioning
   // normalize column i so that a_ii=1
   for (int i = 0; i < this->dim(); i++)
   {
      if (this->operator()(i,i)==0.0)
      {
         throw mach::MachException(
            "encountered zero pivot!");
      }

      this->saved_diag(i)=1.0/this->operator()(i,i);
      j_min=std::max(0,i-this->num_lower());
      j_max=std::min(this->dim()-1,i+this->num_upper());
      for (int j = j_min; j <= j_max; j++)
      {
         this->operator()(i,j) *= this->saved_diag(i);
      }
      this->operator()(i,i)=1.0;          // prevents rounding errors
   }

   // Gauss LR-Decomposition
   for (int k = 0; k < this->dim(); k++)
   {
      i_max=std::min(this->dim()-1,k+this->num_lower());  // num_lower not a mistake!
      for (int i = k+1; i <= i_max; i++)
      {
         if (this->operator()(k,k)==0.0)
         {
            throw mach::MachException(
               "encountered zero pivot!");
         }
         x = -this->operator()(i,k) / this->operator()(k,k);
         this->operator()(i,k) = -x;                         // assembly part of L
         j_max = std::min(this->dim()-1,k+this->num_upper());
         for (int j = k+1; j <= j_max; j++)
         {
            // assembly part of R
            this->operator()(i,j)=this->operator()(i,j)+x*this->operator()(k,j);
         }
      }
   }
}

// solves Ly=b
std::vector<double> BandedMatrix::l_solve(const std::vector<double>& b) const
{
   if (this->dim() != (int)b.size())
   {
      throw mach::MachException(
            "bad dimensions in l_solve!");
   }
   std::vector<double> x(this->dim());
   int j_start;
   double sum;
   for (int i = 0; i<this->dim(); i++)
   {
      sum = 0;
      j_start = std::max(0,i-this->num_lower());
      for (int j = j_start; j < i; j++)
      {
         sum += this->operator()(i,j)*x[j];
      }
      x[i] = (b[i]*this->saved_diag(i)) - sum;
   }
   return x;
}
// solves Rx=y
std::vector<double> BandedMatrix::r_solve(const std::vector<double>& b) const
{
   if (this->dim() != (int)b.size())
   {
      throw mach::MachException(
            "bad dimensions in r_solve!");
   }
   std::vector<double> x(this->dim());
   int j_stop;
   double sum;
   for (int i = this->dim()-1; i >= 0; i--) {
      sum = 0;
      j_stop = std::min(this->dim() - 1, i + this->num_upper());
      for (int j = i+1; j <= j_stop; j++)
      {
         sum += this->operator()(i,j) * x[j];
      }
      x[i] = (b[i] - sum) / this->operator()(i,i);
   }
   return x;
}

std::vector<double> BandedMatrix::solve(const std::vector<double>& b,
        bool is_decomposed)
{
   if (this->dim() != (int)b.size())
   {
      throw mach::MachException(
            "bad dimensions in solve!");
   }
   std::vector<double> x,y;
   if (is_decomposed == false)
   {
      this->decompose();
   }
   y = this->l_solve(b);
   x = this->r_solve(y);
   return x;
}

} // namespace mach