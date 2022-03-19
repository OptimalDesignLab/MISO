double StrongWolfe::FindStepLength() {
  // get data at intial point
  phi_init_ = merit_->EvalFunc(0.0);
  dphi_init_ = merit_->EvalGrad(0.0);
  double alpha_old = 0;
  double phi_old = phi_init_;
  double dphi_old = dphi_init_;

  // check search direction
  if (dphi_init_ > 0) {
    cerr << "StrongWolfe(FindStepLength): "
         << "search direction is not a descent direction" << endl;
    throw(-1);
  }

  double alpha_new;
  double phi_new;
  double dphi_new;
  double quad_coeff;
  bool deriv_hi = false;
  for (int i = 0; i < max_iter_; i++) {
    // get new step, function value, and gradient
    //alpha_new = std::min(pow(1.5, i)*alpha_init_, alpha_max_);
    if (i == 0)
      alpha_new = alpha_init_;
    else {
      if (quad_coeff > 0.0) {
        alpha_new = alpha_old - 0.5*dphi_old/quad_coeff;
        if ( (alpha_new < alpha_old) ||
             (alpha_new > alpha_max_) ) {
          alpha_new = std::min(2.0*alpha_old, alpha_max_);
        }
      } else
        alpha_new = std::min(2.0*alpha_old, alpha_max_);
      *out_ << "StrongWolfe::FindStepLength(): alpha_new = " << alpha_new
            << endl;
    }     
    
    phi_new = merit_->EvalFunc(alpha_new);
#ifdef VERBOSE_DEBUG
    *out_ << "StrongWolfe::FindStepLength(): " << "iter " << i << endl;
    *out_ << "alpha_old, alpha_new = " << alpha_old << ", " << alpha_new << endl;
    *out_ << "phi_old, phi_new     = " << phi_old << ", " << phi_new << endl;
    *out_ << "dphi_old             = " << dphi_old << endl;
#endif

    // check if new step violates the sufficient decrease condition, or
    // (when i > 0) if new phi is greater than old phi; if so, call zoom
    if ( (phi_new > phi_init_ + suff_*alpha_new*dphi_init_) ||
         ( (i > 0) && (phi_new >= phi_old) ) ) {
#ifdef VERBOSE_DEBUG
      *out_ << "StrongWolfe::FindStepLength(): "
            << "switching to zoom at iteration " << i << endl;
      *out_ << "[alpha_low, alpha_hi] = [" << alpha_old << "," 
            << alpha_new << "]" << endl;
      *out_ << "phi_low, phi_hi = " << phi_old << ", " << phi_new << endl;
      *out_ << "dphi_low = " << dphi_old << endl;
#endif
      dphi_new = 0.0;
      deriv_hi = false;
      return Zoom(alpha_old, alpha_new, phi_old, phi_new, dphi_old,
                  dphi_new, deriv_hi);
    }
    
    // get new gradient
    dphi_new = merit_->EvalGrad(alpha_new);
    
    // check curvature condition
    if ( fabs(dphi_new) <= -curv_*dphi_init_ ) {
      // curvature condition is satisfied; 
      if (curv_ > 1e-6) {
#ifdef VERBOSE_DEBUG
        *out_ << "StrongWolfe::FindStepLength(): "
              << "curvature condition satisfied at iteration " << i << endl;
#endif
        return alpha_new;
      }
        
      // if curv_ is very small, i.e., we want a true critical point, 
      // this is suspicious; check for local maximum or inflection
      double perturb = merit_->EvalFunc(alpha_new - alpha_max_*1e-6);
      if ( perturb < phi_new) {
        phi_new = perturb;
        dphi_new = merit_->EvalGrad(alpha_new - alpha_max_*1e-6);
      } else {
        perturb = merit_->EvalFunc(alpha_new + alpha_max_*1e-6);
        if ( perturb < phi_new) {
          phi_new = perturb;
          dphi_new = merit_->EvalGrad(alpha_new + alpha_max_*1e-6);
        } else {
          // seems to be a true minimum
          return alpha_new;
        }
      }
    }     
    
    if (dphi_new >= 0) {
      // if we get here, the curvature condition is not satisfied, and
      // phi_new < phi_old
#ifdef VERBOSE_DEBUG
      *out_ << "StrongWolfe::FindStepLength(): "
            << "switching to zoom at iteration " << i << endl;
      *out_ << "[alpha_low, alpha_hi] = [" << alpha_new << "," 
            << alpha_old << "]" << endl;
      *out_ << "phi_low, phi_hi = " << phi_new << ", " << phi_old << endl;
      *out_ << "dphi_low, dphi_hi = " << dphi_new << ", " 
            << dphi_old << endl;
#endif
      deriv_hi = true;
      return Zoom(alpha_new, alpha_old, phi_new, phi_old, dphi_new,
                  dphi_old, deriv_hi);
    }
    
    // update the old variables
    quad_coeff = alpha_new - alpha_old;
    quad_coeff = ((phi_new - phi_old) - dphi_new*quad_coeff)
        /(quad_coeff*quad_coeff);
    alpha_old = alpha_new;
    phi_old = phi_new;
    dphi_old = dphi_new;
  
  } // int i < max_iter_ loop

  // if we get here, the maximum number of iterations was exceeded
  cerr << "StrongWolfe(FindStepLengthZoom): "
       << "maximum number of iterations exceeded" << endl;
  throw(-1);
}


// ==============================================================================
double StrongWolfe::Zoom(double & alpha_low, double & alpha_hi, 
                         double & phi_low, double & phi_hi,
                         double & dphi_low, double & dphi_hi,
                         bool & deriv_hi) {      
  // limit the number of zooms
  double phi_new, dphi_new, alpha_new;
  for (int j = 0; j < max_iter_; j++) {
    
#ifdef VERBOSE_DEBUG
    *out_ << "Zoom(): interval = [" << alpha_low << "," << alpha_hi << "]"
          << ": [phi(alpha_low), phi(alpha_hi)] = ["
          << phi_low << "," << phi_hi << "]" << endl;
#endif
        
    // use interpolation to get new step, then find the new function value    
    double alpha_new = InterpStep(alpha_low, alpha_hi, phi_low, phi_hi, 
                                  dphi_low, dphi_hi, deriv_hi, *out_);
    phi_new = merit_->EvalFunc(alpha_new);
        
    // check if phi_new violates the sufficient decrease condition, or if
    // phi_new > phi_low; if so, this step gives the new alpha_hi value
    if ( (phi_new > phi_init_ + suff_*alpha_new*dphi_init_) || 
         (phi_new > phi_low) ) {
      alpha_hi = alpha_new;
      phi_hi = phi_new;
      dphi_hi = 0;
      deriv_hi = false; // we no longer know the derivative at alpha_hi

    } else {
      // the sufficent decrease is satisfied, and phi_new < phi_low
    
      // evaluate dphi at the new step
      dphi_new = merit_->EvalGrad(alpha_new);

      if (fabs(dphi_new) <= -curv_*dphi_init_) {
        // curvature condition has been satisfied, so stop
        return alpha_new;
        
      } else if ( dphi_new*(alpha_hi - alpha_low) >= 0 ) {
        // in this case, alpha_low and alpha_new bracket a minimum
        alpha_hi = alpha_low;
        phi_hi = phi_low;
        dphi_hi = dphi_low;
        deriv_hi = true; // we now know the derivative at alpha_hi
      }

      // the new low step is alpha_new
      alpha_low = alpha_new;
      phi_low = phi_new;
      dphi_low = dphi_new;
    }
  } // j < max_iter_
    
  if (phi_new < phi_init_ + suff_*alpha_new*dphi_init_) {
    *out_ << "WARNING in Zoom(): "
          << "step found, but curvature condition not met" << endl;
    return alpha_new;
    //f_new = phi_new;
  }

  cerr << "Zoom(): maximum number of iterations exceeded" << endl;
  throw(-1);
}


double InterpStep(const double & alpha_low, const double & alpha_hi,
                  const double & f_low, const double & f_hi,
                  const double & df_low, const double & df_hi,
                  const bool & deriv_hi, ostream& out) {

  //return 0.5*(alpha_low + alpha_hi);
  return QuadraticStep(alpha_low, alpha_hi, f_low, f_hi, df_low, out);

  if (!deriv_hi) {
    // use quadratic interpolation
#ifdef VERBOSE_DEBUG
    out << "InterpStep: no derivative at alpha_hi." << endl;
#endif
    return QuadraticStep(alpha_low, alpha_hi, f_low, f_hi, df_low);
  }

  // the derivative is available at alpha_hi, so try the cubic
  // interpolation
  double dalpha = alpha_hi - alpha_low;
  double a = 6.0*(f_low - f_hi) + 3.0*(df_low + df_hi)*dalpha;
  double b = 6.0*(f_hi - f_low) - 2.0*(2.0*df_low + df_hi)*dalpha;
  double c = df_low*dalpha;
  
  // check the discriminant; if negative resort to quadratic fit
  double disc = pow(b, 2.0) - 4.0*a*c;
  if (disc < 0) {
    cerr << "InterpStep: discriminant is negative?" << endl;
    throw(-1);
    return QuadraticStep(alpha_low, alpha_hi, f_low, f_hi, df_low);
  }

  // if the denominator is too small resort to quadratic fitting
  if (fabs(a) < 1.e-10) 
    return QuadraticStep(alpha_low, alpha_hi, f_low, f_hi, df_low);

  // calculate the two extrenum 
  double x1 = (-b + sqrt(disc))/(2.0*a);
  double x2 = (-b - sqrt(disc))/(2.0*a);
  x1 = alpha_low + x1*(alpha_hi - alpha_low);
  x2 = alpha_low + x2*(alpha_hi - alpha_low);
  
  // get range of steps
  double min_alpha = std::min(alpha_hi, alpha_low);
  double max_alpha = std::max(alpha_hi, alpha_low);
  double step;

  if ( (x1 >= min_alpha) && (x1 <= max_alpha) ) {
    // x1 is in the alpha range
    
    if ( (x2 >= min_alpha) && (x2 <= max_alpha) ) {
      cerr << "InterpStep: found x2 in range?" << endl;
      throw(-1);
      // if both x1 and x2 are in the appropriate range, take the one
      // closer to alpha_low
      if (fabs(x1 - alpha_low) < fabs(x2 - alpha_low))
         step = x1;
      else {
        //step = x2;
      }
      
    } else {
      // x1 is in the range, but x2 is not
      step = x1;
    }
  } else if ( (x2 >= min_alpha) && (x2 <= max_alpha) ) {
    // x2 is in the range, but x1 is not
    cerr << "InterpStep: found x2 in range?" << endl;
    throw(-1);
    //return x2;
  } else {
    // neither x1 nor x2 are in the range, so resort to quadratic fitting
    cerr << "InterpStep: x1 is not in range?" << endl;
    throw(-1);
    //return QuadraticStep(alpha_low, alpha_hi, f_low, f_hi, df_low);
  }
#if 0
  if (fabs(step - min_alpha) < 1.e-4*fabs(max_alpha - min_alpha)) {
    // safe-guard against very small steps
    step 0.5*(alpha_low + alpha_hi);
  }
#endif
  return step;
}


double QuadraticStep(const double & alpha_low, const double & alpha_hi,
                     const double & f_low, const double & f_hi,
                     const double & df_low, ostream& out) {
  double dalpha = alpha_hi - alpha_low;
  double step = alpha_low - 0.5*df_low*dalpha*dalpha /
      (f_hi - f_low - df_low*dalpha);

  double min_alpha = std::min(alpha_hi, alpha_low);
  double max_alpha = std::max(alpha_hi, alpha_low);
  if ( (step < min_alpha) || (step > max_alpha) ) {
    cerr << "QuadraticStep(): step = " << step 
         << " out of interval = [" 
         << min_alpha << "," << max_alpha << "]" << endl;
    cerr << "alpha_low = " << alpha_low << endl;
    cerr << "alpha_hi  = " << alpha_hi << endl;
    cerr << "f_low     = " << f_low << endl;
    cerr << "f_hi      = " << f_hi << endl;
    cerr << "df_low    = " << df_low << endl;
    cerr << "check Zoom for bugs"
         << endl;
    throw(-1);
  }
  //return step; // turn off safe-guard for INK?
  // safe-guard against small steps
  if ( (step - min_alpha) < 1.e-2*(max_alpha - min_alpha) ) {
    step = 0.5*(alpha_low + alpha_hi);
#ifdef VERBOSE_DEBUG
    out << "QuadraticStep: invoking safeguard" << endl;
#endif
  }
  return step;
}