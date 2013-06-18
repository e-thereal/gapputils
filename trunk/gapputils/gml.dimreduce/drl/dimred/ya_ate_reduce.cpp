/***************************************************************************
                              ya_ate_reduce.cpp
                             -------------------
                         Shawn Martin, W. Michael Brown

  Dimensionality reduction with the AutoEncoder

 __________________________________________________________________________
    This file is part of the Dr.L. Library
 __________________________________________________________________________

    begin      : Mon Jun 25 2007
    authors    : Shawn Martin, W. Michael Brown
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

#ifndef YA_ATE_REDUCE_CPP
#define YA_ATE_REDUCE_CPP

using namespace yala;

std::ostream & yala::operator<<(std::ostream &out,
                           const YAATEReduce<double>::conv_parms &c) {
  out << c.conv << std::endl << std::endl << c.momentum << std::endl;
  return out;
}

template <class eltype>
YAATEReduce<eltype>::YAATEReduce() {
   rng=NULL;
   _train_split=0.8;
  _rbm_iters=10; //40;
  _bp_iters=30; //200;
  _des_rbm_size=10;
  _des_bp_size=100;
}

// This function allocates space for the weights and biases
// and initializes them to zero.  It assumes that the layers, weights
// and biases do not already exist.
template <class eltype>
template <class vm_type>
void YAATEReduce<eltype>::init_Wb ( const vm_type &init_layers ) {
  // initialize private layers variable
  layers = init_layers;
  
  YA_DEBUG_ERROR(layers.numel()>0,
                 "Cannot initialize autoencoder with 0 layers.");
  this->_high_dim=layers(0);
  this->_low_dim=(layers(layers.numel()-1));

  // initialize all forward weights and biases to zero
  weights.setup( 2*(layers.numel()-1) );
  biases.setup( 2*(layers.numel()-1) );

  ya_sizet lN=layers.numel()-1;
  for ( ya_sizet i = 0; i < lN; i++ ) {
    weights(i)=vmzeros(layers(i+1),layers(i));
    biases(i)=vmzeros(layers(i+1),1);
  }
  
  // initialize all inverse weights and biases to zero
  for ( ya_sizet i = lN; i > 0; i-- ) {
    ya_sizet j = 2*(lN)-i;    
    weights(j)=vmzeros(layers(i-1),layers(i));
    biases(j)=vmzeros(layers(i-1),1);
  }
}

// This function splits a dataset of size N into batches of minimum size
// rbm_size() and bp_size().  rbm_size() and bp_size() are reset accordingly
template <class eltype>
void YAATEReduce<eltype>::compute_batch_size (const ya_sizet N)
{
  // use brute force to search for factors of N greater than DES_BATCH_SIZE
  ya_sizet batch_size = 0;
  for ( ya_sizet i = rbm_size(); i <= N; i++ )
    if ( N%i == 0 ) {
      batch_size = i;
      break;
    }
  if ( batch_size == 0 )
    batch_size = N;
  if ( batch_size!=rbm_size() ) {
    if (YA_MAX_NOTE>6)
      std::cout << "RBM batch size of " << rbm_size() << " does not evenly "
           << "divide the training set. Changing to: " << batch_size
           << std::endl;
    rbm_size(batch_size);
  }
        
  batch_size = 0;
  for ( ya_sizet i = bp_size(); i <= N; i++ )
    if ( N%i == 0 ) {
      batch_size = i;
      break;
    }
  if ( batch_size == 0 )
    batch_size = N;
  if ( batch_size!=bp_size() ) {
    if (YA_MAX_NOTE>6)
      std::cout << "BP batch size of " << bp_size() << " does not evenly "
           << "divide the training set. Changing to: " << batch_size
           << std::endl;
    bp_size(batch_size);
  }
}

// This function fine tunes the weights using optimmization via
// back propagation of errors.
// INPUTS: train_data -- matrix with points as columns
//         test_data -- another matrix with points as columns (same # of rows)
// OUTPUT: BP_error -- a matrix of training and test set error, the
//                     first row contains training errors and the second
//                     test set errors
template <class eltype>
template <class vm_type, class ya_type2, class ya_type3>
void YAATEReduce<eltype>::back_prop ( const vm_type &train_data,
                                 const ya_type2 &test_data,
                                 ya_type3 &BP_error ) {
  // get ready to record training and testing error
  BP_error.setup(2,bp_iters());

  // remember number of layers, size of training/test sets
  ya_sizet num_maps, num_train, num_test;
  num_maps = layers.numel()-1;
  num_train = size(train_data,1);
  num_test = size(test_data,1);

  // compute number of batches for given batch size
  ya_sizet num_batches;
  num_batches = size(train_data,1)/bp_size();
  YA_MatT curr_data(size(train_data,0), bp_size());
  
  // for computing reconstruct error on each train/test example separately
  YA_VecT Xtilde;

  #ifdef YA_PROGMETER
  YA_ProgMeter pm;
  if (this->_verbose) pm.start("Back Propagation:",70,bp_iters(),false);
  #endif
  ya_sizet bpN=bp_iters();
  for ( ya_sizet curr_iter = 0; curr_iter < bpN; curr_iter++ ) {
    // compute training reconstruction error
    eltype AE_err;
    eltype norm_sq; 
    AE_err = 0.0;
    for ( ya_sizet j = 0; j < num_train; j++ ) {
      // do forward maps
      Xtilde = train_data(":",j);
      for ( ya_sizet i = 0; i < (num_maps-1); i++ ) {// logistic units
        // --- Xtilde = 1.0/(1.0+exp(-weights(i)*Xtilde - biases(i));
        YA_VecT tempv(biases(i));
        cblas_op5(-1.0,weights(i),Xtilde,-1.0,tempv);
        Xtilde = dotinv(1.0+exp(tempv));
      }
      // linear unit
      // --- Xtilde = weights(num_maps-1)*Xtilde + biases(num_maps-1);
      YA_VecT tempv(biases(num_maps-1));
      cblas_op5(1.0,weights(num_maps-1),Xtilde,1.0,tempv);
      Xtilde=tempv;
      
      // do inverse maps
      for ( ya_sizet i = num_maps; i < 2*num_maps; i++ ) { // all logistic
        // --- Xtilde = 1.0/(1.0+exp(-weights(i)*Xtilde - biases(i)));
        YA_VecT tempv(biases(i));
        cblas_op5(-1.0,weights(i),Xtilde,-1.0,tempv);
        Xtilde=dotinv(1.0+exp(tempv));
      }
      
      // accumulate error
      norm_sq = norm(YA_VecT(train_data(":",j)-Xtilde));
      AE_err = AE_err + norm_sq*norm_sq/(static_cast<eltype>(num_train));
    }
    BP_error(0,curr_iter) = sqrt(AE_err);

    // compute test reconstruction error
    AE_err = 0.0;
    for ( ya_sizet j = 0; j < num_test; j++ ) {
      // do forward maps
      Xtilde = test_data(":",j);
      for ( ya_sizet i = 0; i < (num_maps-1); i++ ) { // logistic units
        // --- Xtilde = 1.0/(1.0+exp(-weights(i)*Xtilde - biases(i)));
        YA_VecT tempv(biases(i));
        cblas_op5(-1.0,weights(i),Xtilde,-1.0,tempv);
        Xtilde=dotinv(1.0+exp(tempv));
      }
      // linear unit
      // --- Xtilde = weights(num_maps-1)*Xtilde + biases(num_maps-1);
      YA_VecT tempv(biases(num_maps-1));
      cblas_op5(1.0,weights(num_maps-1),Xtilde,1.0,tempv);
      Xtilde=tempv;

      // do inverse maps
      for ( ya_sizet i = num_maps; i < 2*num_maps; i++ )  { // all logistic
        // --- Xtilde = 1.0/(1.0+exp(-weights(i)*Xtilde - biases(i)));
        YA_VecT tempv(biases(i));
        cblas_op5(-1.0,weights(i),Xtilde,-1.0,tempv);
        Xtilde=dotinv(1.0+exp(tempv));
      }

      // accumulate error
      norm_sq = norm(YA_VecT(test_data(":",j)-Xtilde));
      AE_err = AE_err + norm_sq*norm_sq/num_test;
    }
    BP_error(1,curr_iter) = sqrt(AE_err);
    
    // do conjugate gradient line searches in batches
    for ( ya_sizet curr_batch = 0; curr_batch < num_batches; curr_batch++ )
      minimize ( train_data(":",vmcount(curr_batch*bp_size(),":",
                                        (curr_batch+1)*bp_size()-1)) );
    #ifdef YA_PROGMETER
    if (this->_verbose) pm.iterate();
    #endif
  }
  #ifdef YA_PROGMETER
  if (this->_verbose) pm.finish();
  #endif
}

// This function minimizes a differentiable multivariate function.  It is based
// on a Matlab script (c) by Carl Edward Rasmussen (2006), re-written in C++
// and specialized for the computation of the AutoEncoder.
//
// Here is C. Rasmussen's original description of the algorithm:
// The code falls naturally into 3 parts, after the initial line search is
// started in the direction of steepest descent. 1) we first enter a while loop
// which uses point 1 (p1) and (p2) to compute an extrapolation (p3), until we
// have extrapolated far enough (Wolfe-Powell conditions). 2) if necessary, we
// enter the second loop which takes p2, p3 and p4 chooses the subinterval
// containing a (local) minimum, and interpolates it, unil an acceptable point
// is found (Wolfe-Powell conditions). Note, that points are always maintained
// in order p0 <= p1 <= p2 < p3 < p4. 3) compute a new search direction using
// conjugate gradients (Polack-Ribiere flavour), or revert to steepest if there
// was a problem in the previous line-search. Return the best value so far, if
// two consecutive line-searches fail, or whenever we run out of function
// evaluations or line-searches.
template <class eltype>
template <class vm_type>
void YAATEReduce<eltype>::minimize ( const vm_type &curr_data ) {
  // NOTE: f4, x4, and d4 potential used before initialization
  //       Same code as matlab distribution
  eltype f0, F0, f1, f2, f3, f4=0;          // cross entropy values
  eltype x1, x2, x3, x4=0;                  // step sizes for conjugate gradient
  eltype d0, d1, d2, d3, d4=0;              // slope value for gradient
  AE_Wb s, dF0, df0, df1, df2, df3, df4;    // cross entropy derivatives
  eltype A, B, Coeff;                     // cubic extrapolation coefficients
  eltype Disc, Denom;                // discriminant or denominator that might
                                       // lead to numerical problems
  
  // create local copies of Wb for use in optimization
  AE_Wb X, X0, X1;

  // initialize X to current Wb
  X.weights = weights;
  X.biases = biases;

  // initialize gradient
  cross_entropy ( f0, df0, X, curr_data );    // get function value and gradient
  s.weights=-df0.weights;
  s.biases=-df0.biases;
  d0 = -dot_Wb(s,s);
  x3 = 1.0/(1.0-d0);

  ya_sizet i = 0;                 // zero run length counter
  ya_sizet M;                     // function evaluation counter
  bool ls_failed = false;         // no previous line search has failed
  while ( i < num_ls() ) {
    i = i + 1;
    X0 = X; F0 = f0; dF0 = df0; M = cg_max();
    while ( true ) { // extrapolation
      x2 = 0.0; f2 = f0; d2 = d0; df2 = df0; f3 = f0; df3 = df0;
      
      // compute X1 = X + x3*s;
      X1 = X;
      ya_sizet xwN=X.weights.numel();
      for ( ya_sizet i = 0; i < xwN; i++ ) {
        // --- X1.weights(i)+=x3*s.weights(i)    X1.biases(i) += x3*s.biases(i);
        cblas_op3(x3,s.weights(i),X1.weights(i));
        cblas_op3(x3,s.biases(i),X1.biases(i));
      }
      M = M - 1;
      cross_entropy ( f3, df3, X1, curr_data );

      if ( f3 < F0 ) { // keep best values
        X0 = X1; F0 = f3; dF0 = df3;
      }

      // compute d3 = df3'*s as new slope
      d3 = dot_Wb(df3, s);
    
      // are we done extrapolating?
      if ( (d3 > cg_sig()*d0) || (f3 > (f0 + x3*cg_rho()*d0)) || M == 0)
        break;

      x1 = x2; f1 = f2; d1 = d2; df1 = df2;   // move point 2 to point 1
      x2 = x3; f2 = f3; d2 = d3; df2 = df3;   // move point 3 to point 2
      A = 6.0*(f1-f2)+3.0*(d2+d1)*(x2-x1);    // cubic extrapolation
      B = 3.0*(f2-f1)-(2.0*d1+d2)*(x2-x1);
      Disc = B*B - A*d1*(x2-x1);
      if ( (Disc < 0.0) || (fabs(B-sqrt(Disc))<CG_EPS()) ) // numerical problems
        x3 = x2*cg_ext();                               // maximum extrapolation
      else {
        x3 = x1 - d1*(x2-x1)*(x2-x1)/(B+sqrt(Disc)); // standard extrapolation
        if ( (x3 < 0.0) || (x3 > x2*cg_ext()) )      // bad sign or beyond limit
          x3 = x2*cg_ext();                        // use maximum extrapolation
        else if ( x3 < (x2* + cg_int()*(x2-x1)) )  // new point too close to old
          x3 = x2+cg_int()*(x2-x1);
      }
    } // end extrapolation

    // start interpolation
    while ( (fabs(d3) > (-cg_sig()*d0)) || (f3 > (f0+x3*cg_rho()*d0)) ) {
      if ( (d3 > 0.0) || (f3 > (f0 + x3*cg_rho()*d0)) ) {  // choose subinterval
        x4 = x3; f4 = f3; d4 = d3; df4 = df3;         // move point 3 to point 4
      } else {
        x2 = x3; f2 = f3; d2 = d3; df2 = df3;         // move point 3 to point 2
      }

      if ( f4 > f0 ) {     // should we use quadratic or cubic interpolation?
        Denom = f4-f2-d2*(x4-x2);     // denominator for quadratic interpolation
        if ( fabs(Denom) < CG_EPS() ) // numerical problem
          x3 = (x2+x4)/2.0;           // use bisection
        else
          // otherwise use quadratic interpolation        
          x3 = x2-(0.5*d2*(x4-x2)*(x4-x2))/Denom;  
      } else {
        Denom = x4 - x2;
        if ( fabs(Denom) < CG_EPS() )     // numerical problem
          x3 = (x2+x4)/2.0;             // use bisection
        else {
          A = 6.0*(f2-f4)/Denom + 3.0*(d4+d2);  // cubic interpolation
          B = 3.0*(f4-f2) - (2.0*d2+d4)*(x4-x2);
          Disc = B*B - A*d2*(x4-x2)*(x4-x2);
          if ( (Disc < 0.0) || (fabs(A) < CG_EPS()) ) // numerical problem
            x3 = (x2+x4)/2.0;                       // use bisection
          else
            x3 = x2 + (sqrt(Disc) - B)/A;
        }
      }

      // don't accept too close
      x3 = std::max(min(x3, x4-cg_int()*(x4-x2)),x2+cg_int()*(x4-x2));

      // compute X1 = X + x3*s;
      X1 = X;
      ya_sizet xwN=X.weights.numel();
      for ( ya_sizet i = 0; i < xwN; i++ ) {
        cblas_op3(x3,s.weights(i),X1.weights(i));
        cblas_op3(x3,s.biases(i),X1.biases(i));
      }

      // compute cross entropy
      cross_entropy ( f3, df3, X1, curr_data );

      // keep best values
      if (f3 < F0) {
        X0 = X1; F0 = f3; dF0 = df3;
      }

      // update slope direction
      d3 = dot_Wb(df3,s);
    } // end interpolation

    // update local copy of weights and biases (X)
    if ( (fabs(d3) < (-cg_sig()*d0)) && (f3 < (f0+x3*cg_rho()*d0)) ) {
      // line search suceeded
      // update variables and show progress
      X = X1; f0 = f3;

      // compute Polack-Ribiere CG direction
      Coeff = (dot_Wb(df3,df3) - dot_Wb(df0,df3))/dot_Wb(df0,df0);
      ya_sizet swN=s.weights.numel();
      for ( ya_sizet i = 0; i < swN; i++ ) {
        s.weights(i) = Coeff*s.weights(i) - df3.weights(i);
        s.biases(i) = Coeff*s.biases(i) - df3.biases(i);
      }

      // swap derivatives
      df0 = df3;
      d3 = d0; d0 = dot_Wb(df0,s);

      // new slope must be negative
      if ( d0 > 0 ) {
        s.weights=-df0.weights;
        s.biases=-df0.biases;
        d0 = -dot_Wb(s,s);
      }

      // slope ratio or max CG_RATIO
      x3 = x3*min(cg_ratio(), d3/(d0-CG_REALMIN()));
      
      ls_failed = false;    // line search did not fail
    } else {
      X = X0; f0 = F0; df0 = dF0;   // restore best point so far
      if ( ls_failed || (i > num_ls()) )
        break;    // line search failed twice in a row or we ran out of time

      s.weights=-df0.weights;
      s.biases=-df0.biases;

      // update slope info
      d0 = -dot_Wb(s,s);
      x3 = 1.0/(1.0-d0);    // retry steepest
      ls_failed = true;
    } // end update X
  }

  // update actual weights and biases
  weights = X.weights;
  biases = X.biases;
}

// This function computes the dot product of the weights and biases
// of two AutoEncoders as if they were vectors.
// INPUTS: X -- weights and biases of first AutoEncoder
//         Y -- weights and biases of second AutoEncoder
// OUTPUT: returns the scalar product of X and Y as vectors
// NOTE: we assume that X and Y represent the same size AutoEncoders
template <class eltype>
eltype YAATEReduce<eltype>::dot_Wb ( const AE_Wb &X, const AE_Wb &Y ) {
  eltype vec_acc = 0.0;
  ya_sizet xwN=X.weights.numel();
  for ( ya_sizet i = 0; i < xwN; i++ ) {
    vec_acc+=dot(X.weights(i),Y.weights(i));
    vec_acc+=dot(X.biases(i),Y.biases(i));
  }  
  return vec_acc;
}

// This function computes the cross entropy error and derivatives of an auto
// encoder.
// INPUTS: Wb -- weights and biases of proposed AutoEncoder
//         curr_data -- a batch of data for use in computing error/derivatives
// OUTPUTS: CE_err -- cross entropy error using inputs
//          dWb -- derivatives of weights and biases using inputs
template <class eltype>
template <class vm_type>
void YAATEReduce<eltype>::cross_entropy (eltype &CE_err, AE_Wb &dWb, 
                                           const AE_Wb &Wb,
                                           const vm_type &curr_data ) {
  // get number of layers in AutoEncoder and number of data points
  ya_sizet num_maps,N;
  num_maps = layers.numel()-1;
  N = size(curr_data,1);
  
  // compute function at all layer in AutoEncoder and store in AE_vals
  YA_Mat <YA_MatT> AE_vals( 2*num_maps+1 );
  AE_vals(0) = curr_data;
  
  // do logistic unit levels
  for ( ya_sizet level = 0; level < (num_maps-1); level++ ) {
    // --- AE_vals(level+1) = 1.0/(1.0+exp(-Wb.weights(level)*AE_vals(level) -
    //                        repmat(Wb.biases(level),1,N)));
    AE_vals(level+1)=colrep(Wb.biases(level),N);
    cblas_op5(-1.0,Wb.weights(level),AE_vals(level),-1.0,AE_vals(level+1));
    AE_vals(level+1)=dotinv(1.0+exp(AE_vals(level+1)));
  }
  
  // do linear unit
  AE_vals(num_maps) = Wb.weights(num_maps-1)*AE_vals(num_maps-1) +
                      repmat(Wb.biases(num_maps-1),1,N);
  AE_vals(num_maps)=colrep(Wb.biases(num_maps-1),N);
  cblas_op5(1.0,Wb.weights(num_maps-1),AE_vals(num_maps-1),1.0,
            AE_vals(num_maps));

  // do inverse map (all logisitic units)
  for ( ya_sizet level = num_maps; level < 2*num_maps; level++ ) {
    //--- AE_vals(level+1) = 1.0/(1.0+exp(-Wb.weights(level)*AE_vals(level) -
    //                       repmat(Wb.biases(level),1,N)));
    AE_vals(level+1)=colrep(Wb.biases(level),N);
    cblas_op5(-1.0,Wb.weights(level),AE_vals(level),-1.0,AE_vals(level+1));
    AE_vals(level+1) = dotinv(1.0+exp(AE_vals(level+1)));
  }
  
  // compute cross entropy error
  eltype CE=dot(AE_vals(0),log(AE_vals(2*num_maps)));
  CE+=dot(1.0-AE_vals(0),log(1.0-AE_vals(2*num_maps)));
  CE_err = -CE/static_cast<eltype>(N);
  
  // compute backpropagation derivatives
  YA_MatT delta;
  delta = (dot_star(1.0-AE_vals(0),AE_vals(2*num_maps)) -
           dot_star(AE_vals(0),1.0 - AE_vals(2*num_maps)))/static_cast<eltype>(N);

  // initialize
  dWb.weights.setup ( 2*num_maps );
  dWb.biases.setup ( 2*num_maps );
  dWb.weights ( 2*num_maps-1 ) = delta*transpose(AE_vals(2*num_maps-1));
  dWb.biases ( 2*num_maps-1 ) = row_sum(delta);
  
  // go up to linear map
  for ( ya_sizet level = 2*num_maps-2; level > num_maps-1; level-- ) {
    delta = dot_star(dot_star(YA_MatT(transpose(Wb.weights(level+1))*delta),
                                AE_vals(level+1)),1.0-AE_vals(level+1));
    dWb.weights(level) = delta*transpose(AE_vals(level));
    dWb.biases(level) = row_sum(delta);
  }

  // do linear map
  delta = YA_MatT(transpose(Wb.weights(num_maps))*delta);
  dWb.weights(num_maps-1) = delta*transpose(AE_vals(num_maps-1));
  dWb.biases(num_maps-1) = row_sum(delta);

  // finish all maps
  for ( ya_sizet level = num_maps-1; level >0; level-- ) {
    delta = dot_star (dot_star(YA_MatT(transpose(Wb.weights(level))*delta),
                       AE_vals(level)),1.0-AE_vals(level));
    dWb.weights(level-1) = delta*transpose(AE_vals(level-1));
    dWb.biases(level-1) = row_sum(delta);
  }
}

// This function trains each RBM layer using 1-step contrastive divergence
// INPUTS: train_data -- matrix with points as columns
// OUTPUT: private variables weights & biases are updated according to results
//         RBM_error -- a matrix of layers by iterations containing training
//                      reconstruction error
template <class eltype>
template <class vm_type, class ya_type2>
void YAATEReduce<eltype>::train_RBMs (const vm_type &train_data, 
                                        ya_type2 &RBM_error) {
  // change from initial to final momentum is a "compile time" parameter
  eltype momentum_threshold;
  momentum_threshold = (static_cast<eltype>(rbm_iters()))/momentum_denom();

  // compute number of batches for given batch size
  ya_sizet num_batches;
  num_batches = size(train_data,1)/rbm_size();

  // get ready to record training error
  RBM_error.setup(layers.numel()-1,rbm_iters());
  
  // train an RBM for each layer
  #ifdef YA_PROGMETER
  YA_ProgMeter pm;
  if (this->_verbose) 
    pm.start("RBM Training:",70,rbm_iters()*num_batches*(layers.numel()-1),
             false);
  #endif
  ya_sizet lN=layers.numel()-1;
  for ( ya_sizet RBM_layer = 0; RBM_layer < lN; RBM_layer++ ) {
    // what are the dimensions for this layer?
    ya_sizet D, num_hid, num_cases;
    D = layers(RBM_layer);
    num_hid = layers(RBM_layer+1);
    num_cases = size(train_data,1);
  
    // initialize convergence values
    eltype epsilonW, epsilonb, epsilonc, weightcost,
             momentum, initial_momentum, final_momentum;
    epsilonW = _epsilon(RBM_layer).conv(0);  // learning rate for weights
    epsilonb = _epsilon(RBM_layer).conv(1);  // learning rate for forward biases
    epsilonc = _epsilon(RBM_layer).conv(2);  // learning rate for inverse biases
    weightcost = _epsilon(RBM_layer).conv(3);
    initial_momentum = _epsilon(RBM_layer).momentum(0);
    final_momentum = _epsilon(RBM_layer).momentum(1);

    // initialize weights (biases are already initialized to zero)
    ip_randn(weights(RBM_layer),*rng);
    weights(RBM_layer)*=0.1;

    // the forward biases are given as biases(RBM_layer), corresponding to c
    // the inverse biases (b) are recorded in the biases(inverse_layer)
    ya_sizet inverse_layer;
    inverse_layer = 2*lN-RBM_layer-1;
    
    // initialize variables for 1-step contrastive divergence
    YA_MatT W_inc(vmzeros(num_hid,D));
    YA_VecT b_inc(vmzeros(D,1)), c_inc(vmzeros(num_hid,1));

    // compute data values for current layer
    YA_MatT X(size(train_data,0),size(train_data,1)); // data matrix
    X = train_data;
    for ( ya_sizet i = 0; i < RBM_layer; i++ ) { 
      // only logistic layers are needed
      //X=1.0/(1.0+exp(-weights(i)*X - repmat(biases(i),1,size(train_data,1))));
      YA_MatT tempm(colrep(biases(i),size(train_data,1)));
      cblas_op5(-1.0,weights(i),X,-1.0,tempm);
      X = dotinv(1.0+exp(tempm));
    }
    
    // update weights and biases iteratively for current layer
    YA_MatT X0(D,rbm_size()), Y0(num_hid,rbm_size());
    YA_MatT Y0_bool(num_hid,rbm_size()), X1(D,rbm_size());
    YA_MatT Y1(num_hid,rbm_size());
    
    // temporaries for matrix computations
    eltype X0_X1;
    YA_MatT Y0X0_Y1X1(num_hid,D);
    YA_VecT X0_X1_sum(D), Y0_Y1_sum(num_hid);

    for ( ya_sizet curr_iter = 0; curr_iter < rbm_iters(); curr_iter++ ) {
      // do contrastive divergence for each batch
      eltype tot_err = 0.0;
      for ( ya_sizet batch = 0; batch < num_batches; batch++ ) {
        // get batch (this is a sample)
        X0 = X(":",vmcount(batch*rbm_size(),":",(batch+1)*rbm_size()-1));
    
        // X0 -> Y0
        if ( RBM_layer < (layers.numel()-2) ) { // logistic units
          // --- Y0 = 1.0/(1.0 + exp(-weights(RBM_layer)*X0 -
          //               repmat(biases(RBM_layer),1,rbm_size()))); 
          Y0=colrep(biases(RBM_layer),rbm_size());
          cblas_op5(-1.0,weights(RBM_layer),X0,-1.0,Y0);
          Y0 = dotinv(1.0 + exp(Y0)); 
        } else { // linear units
          //Y0 = weights(RBM_layer)*X0 + repmat(biases(RBM_layer),1,rbm_size());
          Y0 = colrep(biases(RBM_layer),rbm_size());
          cblas_op5(1.0,weights(RBM_layer),X0,1.0,Y0);
        }
        
        // sample from forward mapped values (treated as probabilities)
        // sampling here prevents overtraining in this stage
        if ( RBM_layer < (layers.numel()-2) )
            Y0_bool = vm_cast<eltype>::sc(Y0 > vm_random_exc(num_hid, 
                                                               rbm_size(),*rng));
        else
            Y0_bool = Y0 + vm_random_normal(num_hid, rbm_size(),*rng);

        // for Y0 -> X1 we need to update the inverse level weights and biases
        // --- X1 = 1.0/(1.0 + exp(-transpose(weights(RBM_layer))*Y0_bool -
        //                         repmat(biases(inverse_layer),1,rbm_size())));
        X1=colrep(biases(inverse_layer),rbm_size());
        cblas_op5(-1.0,transpose(weights(RBM_layer)),Y0_bool,-1.0,X1);
        X1=dotinv(1.0+exp(X1));

        // X1 -> Y1
        if ( RBM_layer < (layers.numel()-2) ) {
          // --- Y1 = 1.0/(1.0 + exp(-weights(RBM_layer)*X1 -
          //               repmat(biases(RBM_layer),1,rbm_size())));
          Y1=colrep(biases(RBM_layer),rbm_size());
          cblas_op5(-1.0,weights(RBM_layer),X1,-1.0,Y1);
          Y1 = dotinv(1.0 + exp(Y1));
        } else {
          //Y1 = weights(RBM_layer)*X1 + repmat(biases(RBM_layer),1,rbm_size());
          Y1=colrep(biases(RBM_layer),rbm_size());
          cblas_op5(1.0,weights(RBM_layer),X1,1.0,Y1);
        }
        
        // compute reconstruction error
        YA_VecT tempv(reshape(X0-X1,length(X0),1));
        X0_X1 = norm(tempv);
        tot_err = tot_err + X0_X1*X0_X1;

        // update momentum
        if ( (curr_iter+1) > momentum_threshold )
            momentum=final_momentum;
        else
            momentum=initial_momentum;

        // update weights and biases
        // --- Y0X0_Y1X1 = Y0*transpose(X0)-Y1*transpose(X1);
        Y0X0_Y1X1 = Y1*transpose(X1);
        cblas_op5(1.0,Y0,transpose(X0),-1.0,Y0X0_Y1X1);

        W_inc*=momentum;
        W_inc+=epsilonW*(Y0X0_Y1X1/(static_cast<eltype>(rbm_size())) -
               weightcost*weights(RBM_layer));
        X0_X1_sum = row_sum(X0)-row_sum(X1);
        b_inc*=momentum;
        cblas_op3(epsilonb/static_cast<eltype>(rbm_size()),X0_X1_sum,b_inc);
        Y0_Y1_sum=row_sum(Y0)-row_sum(Y1);
        c_inc*=momentum;
        cblas_op3(epsilonc/static_cast<eltype>(rbm_size()),Y0_Y1_sum,c_inc);

        // update weights and biases
        weights(RBM_layer) += W_inc;
        biases(RBM_layer) += c_inc;
        biases(inverse_layer) += b_inc;
        #ifdef YA_PROGMETER
        if (this->_verbose) pm.iterate();
        #endif
      }
      RBM_error(RBM_layer, curr_iter) = tot_err;
    }
      
    // fill in remainder of weights with transposes
    weights(inverse_layer) = transpose(weights(RBM_layer));
  }
  #ifdef YA_PROGMETER
  if (this->_verbose) pm.finish();
  #endif
}

// This function sets the training/test set split.
template <class eltype>
template <class vm_type, class ya_type2, class ya_type3, class ya_type4>
void YAATEReduce<eltype>::train(const vm_type &all_data, 
                                  const ya_type2 &layers,
                                   ya_type3 &RBM_error, ya_type4 &BP_error) {
  YA_DEBUG_ERROR(rng!=NULL,"Must call set_rng() before training autoencoder");

  // first default is training/test set split
  ya_sizet N, N_train, N_test;
  N = size(all_data,0);
  N_train = static_cast<ya_sizet>(floor(N*train_split()+0.5));
  N_test = N - N_train;
  if (YA_MAX_NOTE>9)
    std::cout << "Using a training set size of " << N_train
         << " and a test set size of " << N_test << std::endl;
  
  // shuffle columns of data matrix to get random training/test set
  YA_RowI rand_perm(randperm(N,*rng));
  YA_RowI rand_train(rand_perm(vmcount(N_train)));
  YA_RowI rand_test(rand_perm(vmcount(N_train,":",N-1)));

  // now we call next version of train to set the rest of the defaults
  train (all_data(rand_train,":"), all_data(rand_test,":"), layers, RBM_error,
         BP_error);
}

// set epsilon parameters for RBM
// (shown below for logistic and linear nodes)
template <class eltype>
template <class vm_type, class ya_type2, class ya_type3,
          class ya_type4, class vm_type5>
void YAATEReduce<eltype>::train(const vm_type &train_data, 
                                  const ya_type2 &test_data,
                                  const ya_type3 &layers, ya_type4 &RBM_error,
                                  vm_type5 &BP_error ) {
  conv_parms logistic, linear;
  logistic.conv = logistic_conv();
  logistic.momentum = logistic_momentum();
  linear.conv = linear_conv();
  linear.momentum = linear_momentum();

  // fill in parameters based on layer height
  YA_Mat<conv_parms> epsilon;
  ya_sizet lN=layers.numel()-1;
  epsilon.setup(lN,1);
  for ( ya_sizet i = 1; i < lN; i++ ) 
    epsilon(i-1)=logistic;
  
  // last layer is linear
  epsilon(layers.numel()-2)=linear;

  // finally we call next version
  train ( train_data, test_data, layers, epsilon, RBM_error, BP_error );
}

// This function actually trains the AutoEncoder.
// INPUTS: train_data, test_data -- training/test data, with points as columns.
//         epsilon -- vector of conv_parms containing convergence criteria
//                    for training each RBM.
// OUTPUTS: RBM_error -- matrix containing RBM reconstruction error, each row
//                       corresponds to a layer of the AutoEncoder
//          BP_error -- matrix containing back-propagation error, first row
//                      is training error, second row is test set error.
template <class eltype>
template <class vm_type, class ya_type2, class ya_type3,
          class ya_type4, class vm_type5>
void YAATEReduce<eltype>::train(const vm_type &train_data, 
                                  const ya_type2 &test_data,
                                  const ya_type3 &layers, 
                                  const YA_Mat<conv_parms> &epsilon,
                                  ya_type4 &RBM_error, vm_type5 &BP_error) {
  YA_DEBUG_ERROR(rng!=NULL,"Must call set_rng() before training autoencoder");

  init_Wb(layers);
  YA_DEBUG_ERROR(layers(0)==train_data.cols(),
                 "First layer must have the dimensionality of training set.");

  compute_batch_size(size(train_data,0));
  _epsilon=epsilon;
       
  // error check for range of data (must be between 0 and 1)
  eltype min_train, max_train;
  min_train = minimum(train_data);
  max_train = maximum(train_data);
  YA_MatT scaled_train, scaled_test;
  if ( (min_train < 0) || (max_train > 1) ) {
    scale_times=1.0/(max_train-min_train);
    scale_add=min_train/(min_train-max_train);
    if (YA_MAX_NOTE>9)
      std::cout << "Scaling training data between 0 and 1.\n";
    scaled_train=train_data.T()*scale_times+scale_add;
    scaled_test=test_data.T()*scale_times+scale_add;
  } else {
    scale_times=1.0;
    scale_add=0.0;
    scaled_train=train_data.T();
    scaled_test=test_data.T();
  }
  // do pre-training with RBMs
  train_RBMs ( scaled_train, RBM_error );
  // finish with back-propagation optimization
  back_prop ( scaled_train, scaled_test, BP_error );
}

template <class eltype>
template <class vm_type, class ya_type2>
void YAATEReduce<eltype>::forward_map(const vm_type &X, ya_type2 &Y) const {
  ya_sizet num_maps, num_cases;
  num_maps = layers.numel()-1;
  num_cases = size(X,0);
    
  // do logistic unit levels
  if (scale_times==1.0 && scale_add==0.0)
    Y = X.T();
  else
    Y = X.T()*scale_times + scale_add;
  for ( ya_sizet i = 0; i < (num_maps-1); i++ ) {
    //Y=1.0/(1.0+exp(YA_MatT(-weights(i)*Y) - repmat(biases(i),1,num_cases)));
    YA_MatT tempm(colrep(biases(i),num_cases));
    cblas_op5(-1.0,weights(i),Y,-1.0,tempm);
    Y = dotinv(1.0+exp(tempm));
  }
  
  // do linear unit
  // =transpose(weights(num_maps-1)*Y + repmat(biases(num_maps-1),1,num_cases));
  YA_MatT tempm(colrep(biases(num_maps-1),num_cases));
  cblas_op5(1.0,weights(num_maps-1),Y,1.0,tempm);
  Y = transpose(tempm);
}

template <class eltype>
template <class vm_type, class ya_type2>
void YAATEReduce<eltype>::reverse_map(const vm_type &Y, 
                    ya_type2 &X_tilde) const {
  ya_sizet num_maps, num_cases;
  num_maps = layers.numel()-1;
  num_cases = size(Y,0);

  // do all logistic unit levels
  X_tilde=Y.T();
  for ( ya_sizet i = num_maps; i < 2*num_maps; i++ ) {
    // X~ = 1.0/(1.0+exp(-weights(i)*X_tilde-repmat(biases(i),1,num_cases)));
    YA_MatT tempm(colrep(biases(i),num_cases));
    cblas_op5(-1.0,weights(i),X_tilde,-1.0,tempm);
    X_tilde = dotinv(1.0+exp(tempm));
  }
  
  if (scale_times!=1.0 || scale_add!=0.0) {
    X_tilde-=scale_add;
    X_tilde/=scale_times;
  }

  ip_transpose(X_tilde);
}

template <class eltype>
void YAATEReduce<eltype>::save_map(std::ostream &out) const {
  ya_sizet precision=out.precision();
  out.precision(16);
  this->write_header(out);
  out << scale_times << " " << scale_add << std::endl;
  write(out,layers);
  for (ya_sizet i=0; i<weights.numel(); i++)
    write(out,weights(i));
  for (ya_sizet i=0; i<biases.numel(); i++)
    write(out,biases(i));
  out.precision(precision);
}

template <class eltype>
int YAATEReduce<eltype>::load_map(std::istream &in) {
  int header_err = this->read_header(in);
  if (header_err !=0 && header_err != 312)
    return header_err;
  in >> scale_times >> scale_add;
  YA_RowI tlayers;
  int err_flag = read(in,tlayers);
  if (err_flag != 0)
    return err_flag;
  init_Wb(tlayers);
  for (ya_sizet i=0; i<2*(layers.numel()-1); i++) {
    err_flag = read(in,weights(i));
    if (err_flag != 0)
      return err_flag;
  }
  for (ya_sizet i=0; i<2*(layers.numel()-1); i++) {
    err_flag = read(in,biases(i));
    if (err_flag != 0)
      return err_flag;
  }
  if (!in || in.eof() || layers.numel()==0 || 
      weights.numel()==0 || biases.numel()==0) {
    ya_addwarn(310,10,method_id(),"Improperly formatted map file.");
    return 310;
  }
  return header_err;
}

#endif
