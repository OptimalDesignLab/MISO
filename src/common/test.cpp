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