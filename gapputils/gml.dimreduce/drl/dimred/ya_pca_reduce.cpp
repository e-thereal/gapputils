/***************************************************************************
                               ya_pca_reduce.cpp
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

#ifndef YA_PCA_REDUCE_CPP
#define YA_PCA_REDUCE_CPP

YALANAMESPACE

template <class eltype>
template <class ya_type1, class ya_type2>
int YAPCAReduce<eltype>::find_map(const ya_type1 &input, 
                                  ya_type2 &output, ya_sizet dim, 
                                  EigenOptions &eigopts) {
  // Target dimensionality
  YA_DEBUG_ERROR(dim<=input.cols(),
                 "Not enough dimensions in input matrix");
  eigopts.dim(dim);
  this->_high_dim=input.cols();
  this->_low_dim=dim;

  // Column center the matrix
  column_means=sum(input)/static_cast<eltype>(input.rows());
  YA_MatT input_cen(input.rows(),input.cols());
  ya_sizet iC=input.cols();
  for (ya_sizet i=0; i<iC; i++)
    input_cen(":",i)=input(":",i)-column_means(i);

  // Calculate covarience matrix
  #ifdef YA_PROGMETER
  YA_TimeKeeper tk;
  YA_ProgMeter pm;
  if (this->_verbose) {
    pm.start("Computing Forward and Reverse Maps:", 70, 0, false);
    tk.start();
  }
  #endif
  
  VM_SymMat covmat=input_cen.T()*input_cen;

  eigs(covmat,eigen_vectors,this->eigen_values,eigopts);

  #ifdef YA_PROGMETER
  if (this->_verbose) {
    pm.finish();
    tk.end();
    cerr << "Done. " << tk << ".\n";
  }
  #endif

  // Calculate the reduced output
  output=input_cen*eigen_vectors;
  return 0;
}

template <class eltype>
template <class ya_type1, class ya_type2>
void YAPCAReduce<eltype>::forward_map(const ya_type1 &input, 
                    ya_type2 &output) const {
  YA_DEBUG_ERROR(input.cols()==this->high_dim(),
                 "Dimensions in input matrix do not match map");
  YA_DEBUG_ERROR(eigen_vectors.rows()==this->high_dim() &&
                 eigen_vectors.cols()==this->low_dim(),
                 "Map does not match high and low dimensionalities");
  output=YA_MatT(input-rowrep(column_means,input.rows()))*eigen_vectors;
}

template <class eltype>
template <class ya_type1, class ya_type2>
void YAPCAReduce<eltype>::reverse_map(const ya_type1 &input, 
                    ya_type2 &output) const {
  YA_DEBUG_ERROR(input.cols()==this->low_dim(),
                 "Dimensions in input matrix do not match map");
  YA_DEBUG_ERROR(eigen_vectors.rows()==this->high_dim() &&
                 eigen_vectors.cols()==this->low_dim(),
                 "Map does not match high and low dimensionalities");
  output=input*transpose(eigen_vectors)+rowrep(column_means,input.rows());
}

template <class eltype>
template <class ya_type1, class ya_type2, class ya_type3, class ya_type4>
void YAPCAReduce<eltype>::reconstruct_error(const ya_type1 &input, 
                                              const ya_type2 &output,
                                              const ya_type3 &dims, 
                                              ya_type4 &rmsd_vec) {
  YA_DEBUG_ERROR(input.cols()==this->high_dim() && 
                 output.cols()==this->low_dim(),
                 "Dimensions in input matrix do not match map");
  YA_DEBUG_ERROR(maximum(dims)<=this->low_dim() && minimum(dims)>0,
                 "A reconstruction dimensionality is greater than map");
  YA_DEBUG_ERROR(eigen_vectors.rows()==this->high_dim() &&
                 eigen_vectors.cols()>=this->low_dim(),
                 "Map does not match high and low dimensions");

  YA_MatT new_coords=rowrep(column_means,input.rows());
  rmsd_vec.setup(dims.numel());
  ya_sizet dim=0;
  ya_sizet dN=dims.numel();
  for (ya_sizet i=0; i<dN; i++) {
    while (dim<dims(i)) {
      new_coords+=output(":",dim)*transpose(eigen_vectors(":",dim));
      dim++;
    }
    rmsd_vec(i)=rmsd(input,new_coords);
  }
}

template <class eltype>
void YAPCAReduce<eltype>::save_map(std::ostream &out) const {
  ya_sizet precision=out.precision();
  out.precision(16);
  this->write_header(out);
  write(out,eigen_vectors);
  write(out,this->eigen_values);
  write(out,column_means);
  out.precision(precision);
}
  
template <class eltype>
int YAPCAReduce<eltype>::load_map(std::istream &in) {
  int header_err = this->read_header(in);
  if (header_err !=0 && header_err != 312)
    return header_err;
  int err_flag = read(in,eigen_vectors);
  if (err_flag != 0)
    return err_flag;
  err_flag = read(in,this->eigen_values);
  if (err_flag != 0)
    return err_flag;
  err_flag = read(in,column_means);
  if (err_flag != 0)
    return err_flag;

  if (!in || in.eof() || eigen_vectors.numel()==0 || 
      column_means.numel()==0) {
    ya_addwarn(310,10,method_id(),"Improperly formatted map file.");
    return 310;
  }
  return header_err;
}

}
#endif
