/***************************************************************************
                               ya_pca_reduce.h
                             -------------------
                               W. Michael Brown

  Principal Component Analysis

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

#ifndef YA_PCA_REDUCE_H
#define YA_PCA_REDUCE_H

YALANAMESPACE

/// Dimensionality reduction using principal component analysis
template<class eltype>
class YAPCAReduce : public YADimReduce<eltype> {
 public:
  YAPCAReduce() {}
  ~YAPCAReduce() {};

  /// Return the name of the dimensionality reduction method
  const std::string method_name() const {
  return "Principal Component Analysis"; }
  
  /// Return the id of the dimensionality reduction method
  const std::string method_id() const {
  return "PCA"; }

  /// Return the version of the dimensionality reduction method
  const std::string version() const {
  return "0.1"; }

  /// Find the map for reduction based on initial samples
  /** \param input Matrix storing initial samples (columns are dimensions)
    * \param output Matrix storing low-dimensioal coords of samples 
    * \param dim New dimensionality for output matrix **/
  template <class ya_type1, class ya_type2>
  int find_map(const ya_type1 &input, ya_type2 &output, ya_sizet dim,
               EigenOptions &eigopts);

  /// Map in the forward direction to reduce the dimensionality of a matrix
  template <class ya_type1, class ya_type2>
  void forward_map(const ya_type1 &input, ya_type2 &output) const;
  
  /// Map in the reverse direction to increase the dimensionality of a matrix
  template <class ya_type1, class ya_type2>
  void reverse_map(const ya_type1 &input, ya_type2 &output) const;

  /// Find the map for reduction based on initial samples
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
  
  using YADimReduce<eltype>::reconstruct_error;
  /// Get the reconstruction errors for the given low dimensionalities
  /** \param input High dimensional coordinates
    * \param output Low dimensional coordinates
    * \param dims SORTED Dimensionalities to calculate the reconstruction errors
    * \param rmsd Root mean squared deviations for reconstruction errors **/
  template <class ya_type1, class ya_type2, class ya_type3, class ya_type4>
  void reconstruct_error(const ya_type1 &input, const ya_type2 &output,
                         const ya_type3 &dims, ya_type4 &rmsd);

  using YADimReduce<eltype>::load_map;
  using YADimReduce<eltype>::save_map;
  /// Save a map to the given stream
  void save_map(std::ostream &out) const;
  /// Load a map from the given stream
  int load_map(std::istream &in);

 private:
  YA_MatT eigen_vectors;
  YA_Rows<eltype,1> column_means;
};

}
#endif
