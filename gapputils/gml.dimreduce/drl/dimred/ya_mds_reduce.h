/***************************************************************************
                               ya_mds_reduce.h
                             -------------------
                               W. Michael Brown

  Multi-dimensional scaling

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

#ifndef YA_MDS_REDUCE_H
#define YA_MDS_REDUCE_H

YALANAMESPACE

/// Dimensionality reduction using multidimensional scaling
template<class eltype>
class YAMDSReduce : public YADimReduce<eltype> {
 public:
  YAMDSReduce() {}
  ~YAMDSReduce() {};

  /// Return the name of the dimensionality reduction method
  const std::string method_name() const {
  return "Multidimensional Scaling"; }
  
  /// Return the id of the dimensionality reduction method
  const std::string method_id() const {
  return "MDS"; }

  /// Return the version of the dimensionality reduction method
  const std::string version() const {
  return "0.1"; }

  /// Find the reduction based on initial samples
  /** \param input Symmetric matrix storing initial distances
    * \param output Matrix storing low-dimensioal coords of samples 
    * \param dim New dimensionality for output matrix **/
  template <class ya_type1, class ya_type2>
  int find_map(const ya_type1 &input, ya_type2 &output, ya_sizet dim,
               EigenOptions &eigopts);

  /// Map in the forward direction to reduce the dimensionality of a matrix
  template <class ya_type1, class ya_type2>
  void forward_map(const ya_type1 &input, ya_type2 &output) const {
    std::cerr << "Forward map is not implemented for MDS.\n"; exit(1);
  }
  
  /// Map in the reverse direction to increase the dimensionality of a matrix
  template <class ya_type1, class ya_type2>
  void reverse_map(const ya_type1 &input, ya_type2 &output) const {
    std::cerr << "Forward map is not implemented for MDS.\n"; exit(1);
  }

  /// Find the reduction based on initial samples
  /** \param input Matrix storing initial samples (columns are dimensions)
    * \param output Matrix storing low-dimensioal coords of samples 
    * \param dim New dimensionality for output matrix **/
  int find(const YA_MatT &input, YA_MatT &output, ya_sizet dim,
           EigenOptions &eigopts) {
    return find_map(input,output,dim,eigopts);
  }
  /// Map in the forward direction to reduce the dimensionality of a matrix
  void forward(const YA_MatT &input, 
             YA_MatT &output) const {
  forward_map(input,output); }
  /// Map in the reverse direction to increase the dimensionality of a matrix
  void reverse(const YA_MatT &input, 
             YA_MatT &output) const {
  reverse_map(input,output); }
  
  using YADimReduce<eltype>::load_map;
  using YADimReduce<eltype>::save_map;
  /// Save a map to the given stream
  void save_map(std::ostream &out) const {}
  /// Load a map from the given stream
  int load_map(std::istream &in) { return 0; }

 private:
  YA_MatT eigen_vectors;
};

}
#endif
