/***************************************************************************
                               ya_dim_reduce.h
                             -------------------
                               W. Michael Brown

  Pure Base Class for Dimensionality Reduction

 __________________________________________________________________________
    This file is part of the Dr.L. Library
 __________________________________________________________________________

    begin      : Wed Jun 7 2006
    authors    : W. Michael Brown
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

/*! \file */

#ifndef YA_DIM_REDUCE_H
#define YA_DIM_REDUCE_H

YALANAMESPACE

/// General Base Class for Dimensionality Reduction
template <class eltype>
class YADimReduce {
 public:
  YADimReduce() { _high_dim=0; _low_dim=0; num_neighbors=10; 
                  _neighbor_weight_mode=0; _neighbor_mode = 0; _verbose=2; }
  virtual ~YADimReduce() {}
  
  /// Return the name of the dimensionality reduction method
  virtual const std::string method_name() const = 0;
  /// Return the string id of the dimensionality reduction method
  virtual const std::string method_id() const = 0;
  /// Return the version of the dimensionality reduction method
  virtual const std::string version() const = 0;

  /// Find the map for reduction based on initial samples
  /** \param input Matrix storing initial samples (columns are dimensions)
    * \param output Matrix storing low-dimensioal mapping of samples 
    * \param dim New dimensionality for output matrix **/
  template <class ya_type1, class ya_type2>
  int find_t(const ya_type1 &input, ya_type2 &output, ya_sizet dim,
             EigenOptions &eigopts)
    { YA_MatT ot; int errf=find(input,ot,dim,eigopts); output=ot; return errf;}
  /// Map in the forward direction to reduce the dimensionality of a matrix
  template <class ya_type1, class ya_type2>
  void forward_t(const ya_type1 &input, ya_type2 &output) const
    { YA_MatT ot; forward(input,ot); output=ot; }
  /// Map in the reverse direction to increase the dimensionality of a matrix
  template <class ya_type1, class ya_type2>
  void reverse_t(const ya_type1 &input, ya_type2 &output) const
    { YA_MatT ot; reverse(input,ot); output=ot; }

  /// Find the map for reduction based on initial samples
  /** \param input Matrix storing initial samples (columns are dimensions)
    * \param output Matrix storing low-dimensioal mapping of samples 
    * \param dim New dimensionality for output matrix **/
  virtual int find(const YA_MatT &input, YA_MatT &output, ya_sizet dim,
                   EigenOptions &eigopts) = 0;
  /// Map in the forward direction to reduce the dimensionality of a matrix
  virtual void forward(const YA_MatT &input, YA_MatT &output) const = 0;
  /// Map in the reverse direction to increase the dimensionality of a matrix
  virtual void reverse(const YA_MatT &input, YA_MatT &output) const = 0;
  
  /// RMSD representing the reconstruction error from the input matrix
  /** This is calculating by applying the forward map follwed by the reverse
    * map. **/
  template <class ya_type1>
  eltype reconstruct_error(const ya_type1 &input) const;
    
  /// Save dimensionality reduction map to the given file name
  int save_map(const std::string filename) const;
  /// Save dimensionality reduction map to the given stream
  virtual void save_map(std::ostream &out) const = 0;
  
  /// Load dimensionality reduction maps from the given file name
  /** Can generate the following errors:
    * - 310/10 Improperly formatted reduction file
  * - 311/10 Reduction method does not match object type
  * - 312/4  Version does not match current version **/
  int load_map(const std::string filename);
  /// Save a map from the given stream
  virtual int load_map(std::istream &in) = 0;

  /// Return the degree of output to console by routines
  /** 0 is no output, 1 outputs a progress meter, 2 also output nbor stats **/
  inline int verbose() { return _verbose; }
  /// Set the degree of output to console by routines
  /** 0 is no output, 1 outputs a progress meter, 2 also output nbor stats **/
  inline void verbose(const int v) { _verbose=v; }

  /// Return the dimensionality for the high dimensional space
  inline ya_sizet high_dim() const { return _high_dim; }
  /// Return the dimensionality for the low dimensional space
  inline ya_sizet low_dim() const { return _low_dim; }
  /// Return the vector of eigen values obtained for map
  const YA_VecT & eigenvalues() const { return eigen_values; } 

  /// Get the number of neighbors for LLE and IsoMap
  ya_sizet neighbors() const { return num_neighbors; }
  /// Set the number of neighbors for LLE and IsoMap and use nearest neighbors
  void neighbors(ya_sizet n) { num_neighbors=n; _neighbor_mode=0; }

  /// Get the number of neighbors for LLE and IsoMap
  eltype epsilon() const { return _epsilon; }
  /// Set the number of neighbors for LLE and IsoMap and use epsilon
  void epsilon(eltype n) { 
    YA_DEBUG_ERROR(n>0,"epsilon must be positive for LLE and IsoMap");
    _epsilon=n; _neighbor_mode=1; }

  /// Get the method for obtaining neighbors for LLE and IsoMap
  /** 0 is nearest neighbors and 1 is epsilon **/
  ya_sizet neighbor_mode() const { return _neighbor_mode; }
  /// Set the method for obtaining neighbors for LLE and IsoMap
  /** 0 is nearest neighbors and 1 is epsilon 
    * The neighbor_mode is automatically set accordingly when
    * using neighbors() or epsilon() **/
  void neighbor_mode(ya_sizet n) { 
    YA_DEBUG_ERROR(n==0 || n==1,"Invalid neighbor_mode for LLE/IsoMap");
    _neighbor_mode=n; }
  /// Get the method for obtaining weights in nbors_forward/nbors_reverse
  /** \sa neighbor_weight_mode(), nbors_forward(), nbors_reverse() **/
  inline ya_sizet neighbor_weight_mode() const { return _neighbor_weight_mode; }
  /// Set the method for obtaining weights in nbors_forward/nbors_reverse
  /** - 0 : The weights are set as 1/distance to each neighbor and scaled to
    *     sum to 1
    * - 1 : A least-squares problem is solved to optimize the weights as is
    *     performed in LLE
    * \sa neighbor_weight_mode(), nbors_forward(), nbors_reverse() **/
  void neighbor_weight_mode(const ya_sizet t) { _neighbor_weight_mode=t; }

  /// Set the dimensionalities of the layers for the AutoEncoder
  template <class vm_type>
  void ae_layers(const vm_type &in)
    { ate_layers=reshape(in,1,in.numel()); _low_dim=in(in.numel()-1); }
  /// Get the RBM errors from AutoEncoder training
  const YA_MatT & rbm_errors() const { return rbm_error; }
  /// Get the BP errors from AutoEncoder training
  const YA_MatT & bp_errors() const { return bp_error; }

  /// Perform forward map as a weighted average of k-neighbors
  /** The method used to determine the weights is set by neighbor_weight_mode()
    * \note the number of neighbors used is determined by neighbors() **/
  template <class vm_type, class ya_type2>
  int nbors_forward(const vm_type &in, ya_type2 &out) const;
  /// Perform reverse map as a weighted average of k-neighbors
  /** The method used to determine the weights is set by neighbor_weight_mode()
    * \note the number of neighbors used is determined by neighbors() **/
  template <class vm_type, class ya_type2>
  int nbors_reverse(const vm_type &in, ya_type2 &out) const;
  
  /// Perform forward map using weighted average to dim smaller than lowdim
  /** \sa nbors_forward **/
  template <class vm_type, class ya_type2>
  int nbors_forward(const vm_type &in, const ya_sizet dim,ya_type2 &out) const;
  /// Perform reverse map using weighted average from smaller than lowdim
  /** \sa nbors_reverse() **/
  template <class vm_type, class ya_type2>
  int nbors_reverse(const vm_type &in, const ya_sizet dim,ya_type2 &out) const;

  /// Double center a matrix
  template <class mat_type>
  void double_center(YA_BaseT &mat);
 protected:
  template <class vm_type, class ya_type2, class ya_type3, class ya_type4>
  void lle_weights(const vm_type &input,const ya_type2 &kinput,const ya_sizet k,
                   ya_type3 &neighbors, ya_type4 &weights, 
                   const bool samemat) const;
  template <class vm_type, class ya_type2>
  void lle_weightse(const vm_type &input,const ya_type2 &kinput,
                    const eltype epsilon, std::vector<YA_DynI> &neighbors,
                    std::vector<YA_DynT> &weights, const bool samemat) const;
  template <class vm_type, class ya_type2, class ya_type3, class ya_type4>
  void dist_weights(const vm_type &input,const ya_type2 &kinput,
                    const ya_sizet k, ya_type3 &neighbors, 
                    ya_type4 &weights) const;
  template <class vm_type, class ya_type2>
  void dist_weightse(const vm_type &input,const ya_type2 &kinput,
                     const eltype epsilon, std::vector<YA_DynI> &nbors,
                     std::vector<YA_DynT> &weights) const;
  void write_header(std::ostream &out) const;
  int read_header(std::istream &in);
  ya_sizet _high_dim;
  ya_sizet _low_dim;
  YA_VecT eigen_values;
  // LLE and IsoMap
  ya_sizet num_neighbors;
  eltype _epsilon;
  ya_sizet _neighbor_mode; // 0 for nearest neighbors, 1 for epsilon
  ya_sizet _neighbor_weight_mode;   // true if LLE weights are used, false if distances
  // AutoEncoder
  YA_RowI ate_layers;
  YA_MatT rbm_error,bp_error;
  YA_MatT input_coords,output_coords;
  // Degree of routine output to console (0 off, 1 progmeter, 2 neighbor stats)
  int _verbose;
};

}
#endif
