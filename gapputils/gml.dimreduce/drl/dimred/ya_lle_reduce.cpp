/***************************************************************************
                               ya_lle_reduce.cpp
                             -------------------
                         W. Michael Brown, Haixia Jia

  Locally Linear Embedding

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

#ifndef YA_LLE_REDUCE_CPP
#define YA_LLE_REDUCE_CPP

using namespace yala;

template <class eltype>
template <class mat_type, class ya_type2>
int YALLEReduce<eltype>::find_map(const YA_BaseT &input, 
                                  ya_type2 &output, ya_sizet dim, 
                                  EigenOptions &eigopts) {
  YA_DEBUG_ERROR(input.cols()>=dim,
                 "Not enough dimensions in input matrix");
  YA_DEBUG_ERROR(this->neighbors()<input.rows(),"More neighbors than data");

  this->_high_dim=input.cols();
  this->_low_dim=dim;
  VM_SymMat m(vmeye(input.rows()));

  if (this->_neighbor_mode==0){
    // Find k-nearest neighbors
    YA_MatI neighbors;
    YA_MatT w;
    this->lle_weights(input,input,this->neighbors(),neighbors,w,true);
    // Compute the embedding
    ya_sizet iR=input.rows();
    for (ya_sizet i=0; i<iR; i++) {
      VM_Column lw(&w,i);
      VM_RowSliceU jj(&neighbors,i);
      m(jj,i)-=lw;
      m(jj,jj)=VM_SymMat(m(jj,jj)+lw*lw.T());
    }
  } else {
    std::vector<YA_DynI> neighbors;
    std::vector<YA_DynT> dists;
    this->lle_weightse(input,input,this->_epsilon,neighbors,dists,true);
    // Compute the embedding
    ya_sizet iR=input.rows();
    for (ya_sizet i=0; i<iR; i++) {
      int nk = neighbors[i].numel(); 
      if (nk==0)
        continue;
      m(neighbors[i],i)-=dists[i];
      m(neighbors[i],neighbors[i])=VM_SymMat(m(neighbors[i],neighbors[i])+
                                             dists[i]*dists[i].T());
    }
  }

  #ifdef YA_PROGMETER
  YA_TimeKeeper tk;
  tk.start();
  YA_ProgMeter pm;
  pm.start("Computing Embedding:", 70, 0, false);
  #endif
  
  eigopts.dim(dim+1);
  eigs_small(m,eigen_vectors,this->eigen_values,eigopts);

  #ifdef YA_PROGMETER
  pm.finish();
  tk.end();
  cerr << "Done. " << tk << ".\n";
  #endif

  output.setup(input.rows(),dim);
  // Need smallest eigen_values
  for (ya_sizet i=0; i<dim; i++)
    output(":",i)=eigen_vectors(":",i+1);

  output*=sqrt(static_cast<eltype>(input.rows()));
  this->input_coords=input;
  this->output_coords=output;
  return 0;
}

template <class eltype>
template <class ya_type1, class ya_type2>
void YALLEReduce<eltype>::forward_map(const ya_type1 &input, 
                                        ya_type2 &output) const {
  this->nbors_forward(input,output);
}

template <class eltype>
template <class ya_type1, class ya_type2>
void YALLEReduce<eltype>::reverse_map(const ya_type1 &input, 
                                        ya_type2 &output) const {
  this->nbors_reverse(input,output);
}

template <class eltype>
void YALLEReduce<eltype>::save_map(std::ostream &out) const {
  ya_sizet precision=out.precision();
  out.precision(16);
  this->write_header(out);
  write(out,this->input_coords);
  write(out,this->output_coords);
  out.precision(precision);
}
  
template <class eltype>
int YALLEReduce<eltype>::load_map(std::istream &in) {
  int header_err = this->read_header(in);
  if (header_err !=0 && header_err != 312)
    return header_err;
  int err_flag = read(in,this->input_coords);
  if (err_flag != 0)
    return err_flag;
  err_flag = read(in,this->output_coords);
  if (err_flag != 0)
    return err_flag;

  if (!in || in.eof() || this->input_coords.numel()==0 || 
      this->output_coords.numel()==0) {
    ya_addwarn(310,10,method_id(),"Improperly formatted map file.");
    return 310;
  }
  return header_err;
}

#endif
