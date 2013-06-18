/***************************************************************************
                               ya_iso_reduce.h
                             -------------------
                         W. Michael Brown, Haixia Jia

  IsoMap

 __________________________________________________________________________
    This file is part of the Dr.L. Library
 __________________________________________________________________________

    begin      : Wed Jun 7 2006
    authors    : W. Michael Brown, Haixia Jia
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

/*! \file */

#ifndef YA_ISO_REDUCE_H
#define YA_ISO_REDUCE_H

YALANAMESPACE

/// Dimensionality reduction using IsoMap
/** This implements the IsoMap dimensionality reduction routine in:
  *
  * J. B. Tenenbaum, V. de Silva, and J. C. Langford, Science, 290, 2319 (2000).
  *
  **/
template<class eltype>
class YAIsoReduce : public YADimReduce<eltype> {
 public:
  YAIsoReduce() {}
  ~YAIsoReduce() {};

  /// Return the name of the dimensionality reduction method
  const std::string method_name() const {
  return "IsoMap"; }
  
  /// Return the id of the dimensionality reduction method
  const std::string method_id() const {
  return "ISOMAP"; }

  /// Return the version of the dimensionality reduction method
  const std::string version() const {
  return "0.1"; }

  /// Find the reduction based on initial samples
  /** \param input Matrix storing initial coordinates
    * \param output Matrix storing low-dimensioal coords of samples 
    * \param dim New dimensionality for output matrix **/
  template <class ya_type1, class ya_type2>
  int find_map(const ya_type1 &input, ya_type2 &output, ya_sizet dim,
               EigenOptions &eigopts);

  /// Find the reduction based on initial samples using landmarks
  /** \param input Matrix storing initial coordinates
    * \param output Matrix storing low-dimensioal coords of samples 
    * \param dim New dimensionality for output matrix 
    * \param landmarks Vector storing indices of landmark points in input 
    * \note if more than 1 connected manifold is found, landmarks
    *       will store the subset of landmarks in the connected
    *       manifold **/
  template <class ya_type1, class ya_type2, class ya_type3>
  int find_map(const ya_type1 &input, ya_type2 &output, ya_sizet dim,
               EigenOptions &eigopts, ya_type3 &landmarks);

  /// Map in the forward direction to reduce the dimensionality of a matrix
  template <class ya_type1, class ya_type2>
  void forward_map(const ya_type1 &input, ya_type2 &output) const;
  
  /// Map in the reverse direction to increase the dimensionality of a matrix
  template <class ya_type1, class ya_type2>
  void reverse_map(const ya_type1 &input, ya_type2 &output) const;

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
  void save_map(std::ostream &out) const;
  /// Load a map from the given stream
  int load_map(std::istream &in);

  /// Returns the graph theoretical distance matrix
  const YA_MatT & graph_dists() { return _graph_dists; }

  /// Find how many connected components and their sizes result from neighboring
  /** \param n x m input matrix for n datapoints with a dimensionality = m
    * \param num_neighbors number of neighbors to use for neighboring routine
    * \param stat length C vector containing the sizes of the C connected
    *             components found during neighboring. **/
  template <class ya_type1>
  void component_stat(ya_type1 &m, const ya_sizet num_neighbors,
                      YA_DynI &stats);

  /// Find how many connected components and their sizes result from neighboring
  /** Same as component_stat() but landmarks can be specified. **/
  template <class ya_type1, class ya_type2>
  void component_stat(ya_type1 &m,const ya_sizet num_neighbors,
                      YA_DynI &stats, const ya_type2 &landmarks);
 private:
  YA_MatT eigen_vectors;
  YA_MatT _graph_dists;
};

}
#endif
