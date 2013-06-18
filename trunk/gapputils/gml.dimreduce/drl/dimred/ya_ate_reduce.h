/***************************************************************************
                               ya_ate_reduce.h
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

#ifndef YA_ATE_REDUCE_H
#define YA_ATE_REDUCE_H

YALANAMESPACE

/// Dimensionality Reduction via the AutoEncoder
/** Autoencoder Formulation in C++
  * ------------------------------
  * 
  * This implements an autoencoder for dimensionality reduction as described in:
  *
  *  - Hinton, G.E., and R.R. Salakhutdinov (2006). "Reducing the dimensionality
  *    of data with neural networks." Science 313(5786): 504-507.
  *
  * An AutoEncoder is given by a sequence of weight matrices, biases and unit
  * functions.  These parameters specify the encoder and the decoder which together
  * make up the AutoEncoder.  An AutoEncoder is specified by a sequence of integers
  * n1-n2-n3-...-nd, where n1 is the input dimension, and n2-n3-...-nd are the
  * dimensions of the intermediate maps, with nd representing the final (low
  * dimensional) representation.
  * 
  * Corresponding to the layers of the AutoEncoder we have weight matrices
  * W1,...,W(2*d-2) and biases b1,...,b(2*d-2).  The 1 to d-1 the weight matrices
  * and biases specify the encoder and have dimensions as follows: W1 is n2xn1,
  * b1 is n2x1, W2 is n3xn2, b2 is n3x1, etc.
  * 
  * The encoder is assumed to have all logistic unit functions of the form
  * 1/(1+exp(-x)) except for one linear unit function mapping n(d-1) -> nd.
  * 
  * If X is the n1 by N data matrix with points as columns then the first step
  * of the forward map is given by (using matlab notation)
  * 
  * 1./(1+exp(-W1*X - repmat(b1,1,N))
  * 
  * The full forward map is given by compositions of these functions, the last
  * being linear of the form W(d-1)*X(d-1) + repmat(b(d-1),1,N), where W(d-1)
  * and b(d-1) are the d-1 weight matrix and bias vector, respectively, and
  * X(d-1) is the d-1 image of X under the previous logistic maps.
  * 
  * Once we reach the ndxn(d-1) weight matrix W(d-1) and ndx1 bias b(d-1), we
  * switch to the decoder, which is specified as follows: Wd is ndxn(d-1), bd is
  * n(d-1), etc.  The decoder doesn't have any linear maps.  All maps are of the
  * form
  * 
  * 1./(1+exp(-W(d)*Y - repmat(b(d),1,N)).
  * 
  * The full AutoEncoder is the composition of all of the forward and inverse maps.
  * 
  **/
template <class eltype>
class YAATEReduce : public YADimReduce<eltype> {
 public:
  YAATEReduce();

  /// Return the name of the dimensionality reduction method
  const std::string method_name() const {
  return "AutoEncoder"; }
  
  /// Return the id of the dimensionality reduction method
  const std::string method_id() const {
  return "AE"; }

  /// Return the version of the dimensionality reduction method
  const std::string version() const {
  return "0.1"; }

  /// Set the random number generator to be used during training
  void set_rng(MathRandom<MathMersenneTwister> *_rng) { rng=_rng; }

  /// Find the map for reduction based on initial samples
  /** \param input Matrix storing initial samples (columns are dimensions)
    * \param output Matrix storing low-dimensioal coords of samples 
    * \param dim New dimensionality for output matrix **/
  int find(const YA_MatT &input, YA_MatT &output, ya_sizet dim,
           EigenOptions &eigopts) {
    YA_DEBUG_ERROR1(this->ate_layers.numel()!=0 && 
                    dim==this->ate_layers.back(),
                    "Error specifying layers for autoencoder",
                    this->ate_layers);
    train(input,this->ate_layers,this->rbm_error,this->bp_error);
    forward_map(input,output);
    return 0;
  }
  /// Find the map for reduction based on initial samples with specific test
  /** \param input Train Matrix of initial samples (columns are dimensions)
    * \param test Test Matrix
    * \param output Matrix storing low-dimensioal coords of samples 
    * \param dim New dimensionality for output matrix **/
  int find(const YA_MatT &input, YA_MatT &test, YA_MatT &output, 
           ya_sizet dim, EigenOptions &eigopts) {
    YA_DEBUG_ERROR1(this->ate_layers.numel()!=0 && 
                    dim==this->ate_layers.back(),
                    "Error specifying layers for autoencoder",
                    this->ate_layers);
    train(input,test,this->ate_layers,this->rbm_error,this->bp_error);
    forward_map(input,output);
    return 0;
  }
  /// Map in the forward direction to reduce the dimensionality of a matrix
  void forward(const YA_MatT &input, YA_MatT &output) const 
    { forward_map(input,output); }
  /// Map in the reverse direction to increase the dimensionality of a matrix
  void reverse(const YA_MatT &input, YA_MatT &output) const 
    { reverse_map(input,output); }
  
  using YADimReduce<eltype>::load_map;
  using YADimReduce<eltype>::save_map;
  /// Save a map to the given stream
  void save_map(std::ostream &out) const;
  /// Load a map from the given stream
  int load_map(std::istream &in);

  /// This structure contains convergence parameters for the 1-step constrastive
  /** divergence algorithm used to train the Restricted Boltzmann Machines
    * (see train_RBM). **/
  struct conv_parms {
    YA_RowT conv;
    YA_RowT momentum;
  };

  /// Train the AutoEncoder using a random split of the data into train/test
  /** The data is split based on train_split().
    * \param all_data data to be used for training and testing
    * \param layers dimensionality of each layer to be used for mapping
    * \param RBM_error RBM Training Error. Rows are layers, columns iterations
    * \param BP_error BP Training Error. First row is train, second row test
    **/
  template <class vm_type, class ya_type2, class ya_type3, class ya_type4>
  void train ( const vm_type &all_data, const ya_type2 &layers,
               ya_type3 &RBM_error, ya_type4 &BP_error );
  /// Train the AutoEncoder using specified training and testing data
  /** \sa train() **/
  template <class vm_type, class ya_type2, class ya_type3,
            class ya_type4, class vm_type5>
  void train ( const vm_type &train_data, const ya_type2 &test_data,
               const ya_type3 &layers,
               ya_type4 &RBM_error, vm_type5 &BP_error );
  /// Train the AutoEncoder using specified training/testing data and epsilon
  /** \sa train() **/
  template <class vm_type, class ya_type2, class ya_type3,
            class ya_type4, class vm_type5>
  void train ( const vm_type &train_data, const ya_type2 &test_data,
               const ya_type3 &layers, const YA_Mat<conv_parms> &epsilon,
               ya_type4 &RBM_error, vm_type5 &BP_error );
              
  /// Forward map
  /** The forward map has all logistic units except for the final map,
      which is linear. **/
  template <class vm_type, class ya_type2>
  void forward_map ( const vm_type &input, ya_type2 &output ) const;
  /// Inverse map
  /** The inverse map has all logistic units. **/
  template <class vm_type, class ya_type2>
  void reverse_map ( const vm_type &input, ya_type2 &output ) const;
  
  // --------------- User Controlled Options
  
  /// percentage of data set that will become training set
  inline double train_split() const { return _train_split; }
  /// set percentage of data set that will become training set
  inline void train_split(double i) { _train_split=i; }
  /// number of iterations for RBM training
  inline ya_sizet rbm_iters() const { return _rbm_iters; }
  /// set number of iterations for RBM training
  inline void rbm_iters(ya_sizet i) { _rbm_iters=i; }
  /// number of iterations for back propagation training
  inline ya_sizet bp_iters() const { return _bp_iters; }
  /// set number of iterations for back propagation training
  inline void bp_iters(ya_sizet i) { _bp_iters=i; }
  /// desired batch sizes for RBM training
  inline ya_sizet rbm_size() const { return _des_rbm_size; }
  /// set desired batch sizes for RBM training
  inline void rbm_size(ya_sizet i) { _des_rbm_size=i; }
  /// desired batch sizes for BP training
  inline ya_sizet bp_size() const { return _des_bp_size; }
  /// set desired batch sizes for BP training
  inline void bp_size(ya_sizet i) { _des_bp_size=i; }
 
  // --------------- Options that are currently constants
  /// RBM optimization parameters for logistic and linear units
  inline const std::string logistic_conv() const { return "[.1 .1 .1 .0002]"; }
  /// RBM optimization parameters for logistic and linear units
  inline const std::string logistic_momentum() const { return "[.5 .9]"; }
  /// RBM optimization parameters for logistic and linear units
  inline const std::string linear_conv() const { return "[.001 .001 .001 .0002]"; }
  /// RBM optimization parameters for logistic and linear units
  inline const std::string linear_momentum() const { return "[.5 .9]"; }
 
  /// optimization default for RBM (when to switch from initial to final momentum)
  /** computed as number of iterations/MOMENTUM_DENOM **/
  inline double momentum_denom() const { return 4; }
  /// number of line searches for conjugate gradient optimization (see minimize)
  /** \note must be > 0. **/
  inline ya_sizet num_ls() const { return 3; }

  /// numerical constants for minimizer
  inline eltype CG_EPS() const { return 2.2204E-16; }
  /// numerical constants for minimizer
  inline eltype CG_REALMIN() const { return 2.2251E-308; }

  // ----------------- parameters for optimization
  
  /// don't reevaluate within 0.1 of the limit of the current bracket
  inline eltype cg_int() const { return 0.1; }
  /// extrapolate maximum 3 times the current step-size
  inline eltype cg_ext() const { return 3.0; }
  /// maximum number of function evaluations per iteration for CG routine
  inline ya_sizet cg_max() const { return 20; }
  /// maximum allowed slope ratio
  inline eltype cg_ratio() const { return 10.0; }
  
  /// SIG and RHO are the constants controlling the Wolfe-Powell conditions.
  /** SIG is the maximum allowed absolute ratio between
    * previous and new slopes (derivatives in the search direction), thus setting
    * SIG to low (positive) values forces higher precision in the line-searches.
    * RHO is the minimum allowed fraction of the expected (from the slope at the
    * initial point in the linesearch). Constants must satisfy 0 < RHO < SIG < 1.
    * Tuning of SIG (depending on the nature of the function to be optimized) may
    * speed up the minimization; it is probably not worth playing much with RHO. **/
  inline eltype cg_sig() const { return 0.1; }
  /// SIG and RHO are the constants controlling the Wolfe-Powell conditions.
  /** \sa cg_sig() **/  
  inline eltype cg_rho() const { return cg_sig()/2.0; }
  
 private:
  MathRandom<MathMersenneTwister> *rng;
    
  
  // This structure contains all the pertinent information for the AutoEncoder,
  // including weights, and biases.  This is needed for use by the back
  // propagation optimization functions, including minimize, update, and
  // cross_entropy.
  struct AE_Wb {
    YA_Mat<YA_MatT> weights; 
    YA_Mat<YA_MatT> biases;  
  };

  // ------------------------ AutoEncoder Options
  double _train_split;
  ya_sizet _rbm_iters;
  ya_sizet _bp_iters;
  ya_sizet _des_rbm_size;
  ya_sizet _des_bp_size;
  YA_Mat<conv_parms> _epsilon;

  // ------------------------ Forward and Reverse Maps
  YA_RowI layers;
  YA_Mat<YA_MatT> weights; 
  YA_Mat<YA_MatT> biases;
  // Data is scaled as: scaled_data = data*scale_times + scale_add;
  eltype scale_times;
  eltype scale_add; 
  
  // initialize AutoEncoder
  template <class vm_type>
  void init_Wb ( const vm_type &init_layers );
  
  // compute batch size nearest desired batch size:
  void compute_batch_size ( const ya_sizet N );

  // trains all RBMs
  template <class vm_type, class ya_type2>
  void train_RBMs ( const vm_type &train_data, ya_type2 &RBM_error );
  
  // back propagation
  template <class vm_type, class ya_type2, class ya_type3>
  void back_prop ( const vm_type &train_data, const ya_type2 &test_data,
                   ya_type3 &BP_error);

  // conjugate gradient line search
  template <class vm_type>
  void minimize ( const vm_type &curr_data );

  // compute vectorized dot product of X and Y
  eltype dot_Wb ( const AE_Wb &X, const AE_Wb &Y );
  
  // compute cross entropy and derivatives using back propagation
  template <class vm_type>
  void cross_entropy ( eltype &CE_err, AE_Wb &dWb, const AE_Wb &Wb,
                       const vm_type &curr_data );
};

std::ostream & operator<<(std::ostream &out,
                     const YAATEReduce<double>::conv_parms &c);

}
#endif
