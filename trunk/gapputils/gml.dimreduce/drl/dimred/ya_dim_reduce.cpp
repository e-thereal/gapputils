/***************************************************************************
                              ya_dim_reduce.cpp
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

#ifndef YA_DIM_REDUCE_CPP
#define YA_DIM_REDUCE_CPP

using namespace yala;

// RMSD representing the reconstruction error from the input matrix
template <class eltype>
template <class ya_type1>
eltype YADimReduce<eltype>::reconstruct_error(const ya_type1 &input) const {
  YA_MatT output,newin;
  forward(input,output);
  reverse(output,newin);
  return rmsd(input,newin);
}

template <class eltype>
int YADimReduce<eltype>::save_map(const std::string filename) const {
  std::ofstream out;
  int err_flag=ya_fileopen(out,filename);
  if (err_flag!=0)
    return err_flag;
  save_map(out);
  return ya_fileclose(out);
}

template <class eltype>
int YADimReduce<eltype>::load_map(const std::string filename) {
  std::ifstream in;
  int err_flag=ya_fileopen(in,filename);
  if (err_flag!=0)
    return err_flag;
  err_flag = load_map(in);
  if (err_flag != 0) {
    ya_fileclose(in);
    return err_flag;
  }
  return ya_fileclose(in);
}

template <class eltype>
void YADimReduce<eltype>::write_header(std::ostream &out) const {
  out << method_id() << " " << version() << " " << high_dim() << " " 
      << low_dim() << std::endl;
}

template <class eltype>
int YADimReduce<eltype>::read_header(std::istream &in) {
  std::string name, version_in;
  in >> name >> version_in >> _high_dim >> _low_dim;
  if (!in) {
    ya_addwarn(310,10,method_id(),"Improperly formatted map file");
    return 310;
  }
  if (name!=method_id()) {
    ya_addwarn(311,10,method_id(),"Map file specifies "+name+", not "+
               method_id());
    return 311;
  }
  if (high_dim()<=0 || low_dim()<=0) {
    ya_addwarn(310,10,method_id(),
      "Error reading map file. The high and low dimensionalities must be positive.");
    return 310;
  }
  if (high_dim()<low_dim()) {
    ya_addwarn(310,10,method_id(),
      "Error reading map file. The high dimensionality must not be less than the low dimensionality.");
    return 310;
  }
  if (version_in!=version()) {
    ya_addwarn(312,4,method_id(),"Version in input file, "+
                  version_in+" does not match current version: "+
                  version());
    return 312;
  }
  return 0;
}

template <class eltype>
template <class vm_type, class ya_type2, class ya_type3, class ya_type4>
void YADimReduce<eltype>::lle_weights(const vm_type &input,
                                      const ya_type2 &kinput,const ya_sizet k,
                                      ya_type3 &neighbors, ya_type4 &w, 
                                      const bool samemat) const {
  YA_DEBUG_ERROR(input.cols()==kinput.cols(),
                 "Dimenions differ in source matrix and neighbor matrix");
  YA_DEBUG_ERROR((kinput.rows()>k && samemat) || (kinput.rows()>=k && !samemat),
                 "More neighbors than datapoints");
                                          
  YA_MatT dists;
  if (samemat)
    kneighbors(input,k,neighbors,dists,_verbose);
  else
    kneighbors(input,kinput,k,neighbors,dists,_verbose);

  // Regularization parameter
  eltype tol=1e-3;

  // Solve for reconstruction weights
  ya_sizet iR=input.rows();
  w=vmzeros(k,iR);
  for (ya_sizet i=0; i<iR; i++) {
    if (dists(i,0)==0) {
      w(0,i)=1;
      continue;
    }

    // Create a matrix with neighbors for i such that i is at the origin
    YA_MatT z(input.cols(),k);
    for (ya_sizet j=0; j<k; j++)
      z(":",j)=transpose(kinput(neighbors(i,j),":")-input(i,":"));
    
    // Compute the local covarience
    VM_SymMat c=z.T()*z;
    // Regularization?
    if (k>input.cols())
      c+=vmeye(k)*tol*trace(c);
    VM_OnesV ones(k);
    int err_flag=chol_solve(c,w(":",i),ones,false);
    if (err_flag!=0) {
      c+=vmeye(k)*tol*trace(c);
      err_flag=chol_solve(c,w(":",i),ones,true);
    }
    w(":",i)/=total(w(":",i));
  }
}

template <class eltype>
template <class vm_type, class ya_type2>
void YADimReduce<eltype>::lle_weightse(const vm_type &input,
                                       const ya_type2 &kinput,
                                       const eltype epsilon, 
                                       std::vector<YA_DynI> &neighbors,
                                       std::vector<YA_DynT> &weights,
                                       const bool samemat) const {
  YA_DEBUG_ERROR(input.cols()==kinput.cols(),
                 "Dimenions differ in source matrix and neighbor matrix");
                                          
  weights.resize(input.rows());
  std::vector<YA_DynT> dists;
  if (samemat)
    eneighbors(input,epsilon,neighbors,dists,_verbose);
  else
    eneighbors(input,kinput,epsilon,neighbors,dists,_verbose);

  // Regularization parameter
  eltype tol=1e-3;

  // Solve for reconstruction weights
  ya_sizet iR=input.rows();
  for (ya_sizet i=0; i<iR; i++) {
    int k=neighbors[i].numel();
    if (k==0)
      continue;
    if (dists[i][0]==0) {
      weights[i].push_back(1.0);
      for (int j=0; j<k; j++)
        weights[i].push_back(0.0);
      continue;
    }

    YA_ColT w=vmzeros(k);
    // Create a matrix with neighbors for i such that i is at the origin
    YA_MatT z(input.cols(),k);
    for (ya_sizet j=0; j<k; j++)
      z(":",j)=transpose(kinput(neighbors[i][j],":")-input(i,":"));
    
    // Compute the local covarience
    VM_SymMat c=z.T()*z;
    // Regularization?
    if (k>input.cols())
      c+=vmeye(k)*tol*trace(c);
    VM_OnesV ones(k);
    int err_flag = chol_solve(c,w,ones,false);
    if (err_flag!=0) {
      c+=vmeye(k)*tol*trace(c);
      chol_solve(c,w,ones,true);
    }
    eltype dtot=total(w);
    for (int j=0; j<k; j++)
      weights[i].push_back(w(j)/dtot);
  }
}

template <class eltype>
template <class vm_type, class ya_type2, class ya_type3, class ya_type4>
void YADimReduce<eltype>::dist_weights(const vm_type &input,
                                       const ya_type2 &kinput,const ya_sizet k,
                                       ya_type3 &nborsj, 
                                       ya_type4 &idists) const {
  YA_DEBUG_ERROR(input.cols()==kinput.cols(),
                 "Dimensions differ in source and neighbor matrices");
  YA_DEBUG_ERROR(kinput.rows()>=k,"More neighbors than datapoints");

  YA_MatT dists;
  ya_sizet iR=input.rows();
  idists=vmzeros(k,iR);
  kneighbors(input,kinput,k,nborsj,dists,_verbose);
  for (ya_sizet i=0; i<iR; i++) {
    if (dists(i,0)==0) {
      idists(0,i)=1;
      continue;
    }
    idists(":",i)=dotinv(dists(i,":"));
    idists(":",i)/=total(idists(":",i));
  }
}

template <class eltype>
template <class vm_type, class ya_type2>
void YADimReduce<eltype>::dist_weightse(const vm_type &input,
                                        const ya_type2 &kinput,
                                        const eltype epsilon, 
                                        std::vector<YA_DynI> &nbors,
                                        std::vector<YA_DynT> &weights) const{
  YA_DEBUG_ERROR(input.cols()==kinput.cols(),
                 "Dimensions differ in source and neighbor matrices");

  std::vector<YA_DynT> dists;
  eneighbors(input,kinput,epsilon,nbors,dists,_verbose);
  weights.resize(dists.size());
  ya_sizet iR=input.rows();
  for (ya_sizet i=0; i<iR; i++) {
    int k = nbors[i].numel();
    if (k==0)
      continue;
    if (dists[i][0]==0) {
      weights[i]=vmzeros(k);
      weights[i][0]=1;
      continue;
    }
    weights[i]=dotinv(dists[i]);
    weights[i]/=total(weights[i]);
  }
}

template <class eltype>
template <class vm_type, class ya_type2>
int YADimReduce<eltype>::nbors_forward(const vm_type &in, 
                                       ya_type2 &out) const {
  YA_DEBUG_ERROR(input_coords.numel()!=0 && output_coords.numel()!=0 &&
                 input_coords.cols()==high_dim() && 
                 output_coords.cols()==low_dim() && 
                 input_coords.rows()==output_coords.rows() && 
                 in.cols()==high_dim(),
                 "Loaded map does not match the dimensions for forward map.");
  int err_flag=0;

  if (_neighbor_mode==0) { 
    YA_DEBUG_ERROR(neighbors()<=input_coords.rows(),
                 "More neighbors than data points for forward map.");
    YA_DEBUG_ERROR(neighbors()>1,
                   "Must use more than 1 neighbor for mapping.");
    YA_MatI nborsj;
    YA_MatT weights;
    if (neighbor_weight_mode()==0)
      dist_weights(in,input_coords,neighbors(),nborsj,weights);
    else
      lle_weights(in,input_coords,neighbors(),nborsj,weights,false);
    ya_sizet iR=in.rows();
    out.setup(iR,low_dim());
    for (ya_sizet i=0; i<iR; i++)
      out(i,":")=sum(diag(weights(":",i))*
                   output_coords(nborsj(i,":"),":"));
  } else {
    std::vector<YA_DynI> nborsj;
    std::vector<YA_DynT> weights;
    if (neighbor_weight_mode()==0)
      dist_weightse(in,input_coords,epsilon(),nborsj,weights);
    else
      lle_weightse(in,input_coords,epsilon(),nborsj,weights,false);
    ya_sizet iR=in.rows();
    out.setup(iR,low_dim());
    for (ya_sizet i=0; i<iR; i++) {
      if (nborsj[i].numel()<2) {
        ya_addwarn(322,9,"YADimReduce",
                   "Mapping point with <2 neighbors. Increase epsilon.");
        if (nborsj[i].numel()==0)
          out(i,":")=vmzeros(1,low_dim());
        else
          out(i,":")=weights[0]*output_coords(nborsj[i],":");
        err_flag=322;
      } else
        out(i,":")=sum(diag(weights[i])*output_coords(nborsj[i],":"));
    }
  }
  return err_flag;
}

template <class eltype>
template <class vm_type, class ya_type2>
int YADimReduce<eltype>::nbors_reverse(const vm_type &in, 
                                              ya_type2 &out) const {
  YA_DEBUG_ERROR(input_coords.numel()!=0 && output_coords.numel()!=0 &&
                 input_coords.cols()==high_dim() && 
                 output_coords.cols()==low_dim() && 
                 input_coords.rows()==output_coords.rows() && 
                 in.cols()==low_dim(),
                 "Loaded map does not match the dimensions for reverse map.");
  int err_flag=0;

  if (_neighbor_mode==0){ 
    YA_DEBUG_ERROR(neighbors()<=output_coords.rows(),
                 "More neighbors than data points for reverse map.");
    YA_DEBUG_ERROR(neighbors()>1,
                   "Must use more than 1 neighbor for mapping.");
    YA_MatI nborsj;
    YA_MatT weights;
    if (neighbor_weight_mode()==0)
      dist_weights(in,output_coords,neighbors(),nborsj,weights);
    else
      lle_weights(in,output_coords,neighbors(),nborsj,weights,false);

    ya_sizet iR=in.rows();
    out.setup(iR,high_dim());
    for (ya_sizet i=0; i<iR; i++)
      out(i,":")=sum(diag(weights(":",i))*
                     input_coords(nborsj(i,":"),":"));
  } else {
    std::vector<YA_DynI> nborsj;
    std::vector<YA_DynT> weights;
    if (neighbor_weight_mode()==0)
      dist_weightse(in,output_coords,epsilon(),nborsj,weights);
    else
      lle_weightse(in,output_coords,epsilon(),nborsj,weights,false);
    ya_sizet iR=in.rows();
    out.setup(iR,high_dim());
    for (ya_sizet i=0; i<iR; i++) {
      if (nborsj[i].numel()==0) {
        ya_addwarn(322,9,"YADimReduce",
                   "Mapping point with <2 neighbors. Increase epsilon.");
        if (nborsj[i].numel()==0)
          out(i,":")=vmzeros(1,high_dim());
        else
          out(i,":")=weights[0]*input_coords(nborsj[i],":");
        err_flag=322;
      } else
        out(i,":")=sum(diag(weights[i])*input_coords(nborsj[i],":"));
    }
  }
  return err_flag;
}

template <class eltype>
template <class vm_type, class ya_type2>
int YADimReduce<eltype>::nbors_forward(const vm_type &in, const ya_sizet dim,
                                       ya_type2 &out) const {
  YA_DEBUG_ERROR(input_coords.numel()!=0 && output_coords.numel()!=0 &&
                 input_coords.cols()==high_dim() && 
                 output_coords.cols()>=dim && 
                 input_coords.rows()==output_coords.rows() && 
                 in.cols()==high_dim(),
                 "Loaded map does not match the dimensions for forward map.");
  int err_flag=0;

  if (_neighbor_mode==0){ 
    YA_DEBUG_ERROR(neighbors()<=input_coords.rows(),
                   "More neighbors than data points for forward map.");
    YA_DEBUG_ERROR(neighbors()>1,
                   "Must use more than 1 neighbor for mapping.");
    YA_MatI nborsj;
    YA_MatT weights;
    if (neighbor_weight_mode()==0)
      dist_weights(in,input_coords,neighbors(),nborsj,weights);
    else
      lle_weights(in,input_coords,neighbors(),nborsj,weights,false);

    ya_sizet iR=in.rows();
    out.setup(iR,dim);
    for (ya_sizet i=0; i<iR; i++)
      out(i,":")=sum(diag(weights(":",i))*
                     output_coords(nborsj(i,":"),vmcount(dim)));
  } else {
    std::vector<YA_DynI> nborsj;
    std::vector<YA_DynT> weights;
    if (neighbor_weight_mode()==0)
      dist_weightse(in,input_coords,epsilon(),nborsj,weights);
    else
      lle_weightse(in,input_coords,epsilon(),nborsj,weights,false);
    ya_sizet iR=in.rows();
    out.setup(iR,dim);
    for (ya_sizet i=0; i<iR; i++) {
      if (nborsj[i].numel()==0) {
        ya_addwarn(322,9,"YADimReduce",
                   "Mapping point with <2 neighbors. Increase epsilon.");
        if (nborsj[i].numel()==0)
          out(i,":")=vmzeros(1,dim);
        else
          out(i,":")=weights[0]*output_coords(nborsj[i],vmcount(dim));
        err_flag=322;
      } else
        out(i,":")=sum(diag(weights[i])*output_coords(nborsj[i],vmcount(dim)));
    }
  }
  return err_flag;
}

template <class eltype>
template <class vm_type, class ya_type2>
int YADimReduce<eltype>::nbors_reverse(const vm_type &in, const ya_sizet dim, 
                                       ya_type2 &out) const {
  YA_DEBUG_ERROR(input_coords.numel()!=0 && output_coords.numel()!=0 &&
                 input_coords.cols()==high_dim() && 
                 output_coords.cols()>=dim && 
                 input_coords.rows()==output_coords.rows() && 
                 in.cols()==dim,
                 "Loaded map does not match the dimensions for reverse map.");
  int err_flag=0;
  if (_neighbor_mode==0) { 
    YA_DEBUG_ERROR(neighbors()<=output_coords.rows(),
                   "More neighbors than data points for reverse map.");
    YA_DEBUG_ERROR(neighbors()>1,
                   "Must use more than 1 neighbor for mapping.");
    YA_MatI nborsj;
    YA_MatT weights;
    if (neighbor_weight_mode()==0)
      dist_weights(in,output_coords(":",vmcount(dim)),neighbors(),nborsj,weights);
    else
      lle_weights(in,output_coords(":",vmcount(dim)),neighbors(),nborsj,weights,
                  false);
    ya_sizet iR=in.rows();
    out.setup(iR,high_dim());
    for (ya_sizet i=0; i<iR; i++)
      out(i,":")=sum(diag(weights(":",i))*
                     input_coords(nborsj(i,":"),":"));
  } else {
    std::vector<YA_DynI> nborsj;
    std::vector<YA_DynT> weights;
    if (neighbor_weight_mode()==0)
      dist_weightse(in,output_coords(":",vmcount(dim)),epsilon(),nborsj,
                    weights);
    else
      lle_weightse(in,output_coords(":",vmcount(dim)),epsilon(),nborsj,weights,
                   false);
    ya_sizet iR=in.rows();
    out.setup(iR,high_dim());
    for (ya_sizet i=0; i<iR; i++) {
      if (nborsj[i].numel()==0) {
        ya_addwarn(322,9,"YADimReduce",
                   "Mapping point with <2 neighbors. Increase epsilon.");
        if (nborsj[i].numel()==0)
          out(i,":")=vmzeros(1,high_dim());
        else
          out(i,":")=weights[0]*input_coords(nborsj[i],":");
        err_flag=322;
      } else
        out(i,":")=sum(diag(weights[i])*input_coords(nborsj[i],":"));
    }
  }
  return err_flag;
}

/// Double center a matrix
/** Center the matrix by substracting the mean of each column, row, 
  * adding the overall mean, and multiplying by -0.5 **/
template <class eltype>
template <class mat_type>
void YADimReduce<eltype>::double_center(YA_BaseT &mat) {
  YA_VecT means=sum(mat)/static_cast<eltype>(mat.rows());
  YA_RowT row_means=row_sum(mat)/static_cast<eltype>(mat.cols());
  eltype allmean=total(means)/static_cast<eltype>(mat.cols());
  
  typename YA_COPY_TYPE addf=allmean-rowrep(means,mat.rows())-
                             colrep(row_means,mat.cols());
  mat+=addf;
  mat*=-0.5;
}  

#endif
