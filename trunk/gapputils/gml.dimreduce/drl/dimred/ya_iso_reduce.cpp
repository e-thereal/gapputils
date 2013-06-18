/***************************************************************************
                               ya_iso_reduce.cpp
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

#ifndef YA_ISO_REDUCE_CPP
#define YA_ISO_REDUCE_CPP

#include <iostream>
#include <iomanip>

using namespace yala;

template <class eltype>
template <class ya_type1, class ya_type2>
int YAIsoReduce<eltype>::find_map(const ya_type1 &m, 
                                  ya_type2 &output, ya_sizet dim, 
                                  EigenOptions &eigopts) {
  YA_VecI landmarks;
  return find_map(m,output,dim,eigopts,landmarks);
}


template <class eltype>
template <class ya_type1, class ya_type2, class ya_type3>
int YAIsoReduce<eltype>::find_map(const ya_type1 &m, ya_type2 &output, 
                                  ya_sizet dim, EigenOptions &eigopts,
                                  ya_type3 &landmarks) {
  YA_DEBUG_ERROR(landmarks.numel()>dim || landmarks.numel()==0,
                 "Can't use isomap with more dimensions than landmarks.");
  int err_flag=0;
  this->_high_dim=m.cols();
  this->_low_dim=dim;
  
  ListGraph graph(m.rows());
  if (this->_neighbor_mode == 0){
    // Find k-nearest neighbors
    YA_MatI neighbors;
    YA_MatD dists;
    kneighbors(m,this->neighbors(),neighbors,dists,this->_verbose);
    // Get the graph
    graph.addneighborhood(neighbors,dists);
  } else {
    // Find E-nearest neighbors
    std::vector<YA_DynI> neighbors;
    std::vector<YA_DynT> dists;
    eneighbors(m,this->_epsilon,neighbors,dists,this->_verbose);
    // Get the graph
    graph.addneighborhood(neighbors, dists);
  } 

  // Similarity matrix for MDS
  // Compute pairwise distance matrix from Dijkstra's
  if (landmarks.numel()==0) {
    VM_SymMat distt;
    graph.all_shortest_paths(distt,this->_verbose);
    _graph_dists=distt;
  } else
    graph.shortest_paths(landmarks,_graph_dists,this->_verbose);
                                       
  // Check for connected graph
  #ifdef YA_PROGMETER
  YA_TimeKeeper tk;
  YA_ProgMeter pm;
  if (this->_verbose) {
    tk.start();
    pm.start("Checking for isolated manifold components:", 70, 0, false);
  }
  #endif

  YA_RowI tmp,firsts;
  tmp=min(_graph_dists==BIG_DOUBLE,firsts);
  YA_DynI comps=unique(firsts);
  ya_sizet num_comp=comps.numel();
  YA_RowI isizes=sum(transpose(rowrep(firsts,num_comp)==
                               (comps*
                                vm_cast<ya_sizet>::sc(vmones(1,m.rows())))));
  #ifdef YA_PROGMETER
  if (this->_verbose) {
    pm.finish();
    tk.end();
    cerr << "Done. " << tk << ".\n";
  }
  #endif

  if (YA_MAX_NOTE>9)
    std::cout << "\nFound " << num_comp << " connected components.\n";

  YA_RowD itercounts=pow(vm_cast<double>::sc(isizes),3.0);
  itercounts/=maximum(itercounts)*70.0;
  unsigned eig_meter_total=total(vm_cast<unsigned>::sc(itercounts));

  // Use classical MDS
  #ifdef YA_PROGMETER
  if (this->_verbose) {
    tk.start();
    pm.start("Computing Embedding:", 70, eig_meter_total, false);
  } 
  #endif

  eltype old_center=0;
  eltype old_range=0;

  YA_DynI not_embedded;
  for (ya_sizet i=0; i<num_comp; i++) {
    YA_DynI manifold;
    YA_MatT graph_dists;
    if (num_comp>1) {
      output.setup(m.rows(),dim);
      manifold=::find(firsts==comps(i));
      if (landmarks.numel()!=0) {
        YA_DynI no_embed=intersect(manifold,::find(min(is_inf(_graph_dists))));
        ya_sizet nN=no_embed.numel();
        if (nN!=0)
          for (ya_sizet j=0; j<nN; j++)
            not_embedded.push_back(no_embed(j));
        manifold=setdiff(manifold,::find(min(is_inf(_graph_dists))));
        if (manifold.numel()==0)
          continue;
        YA_DynI newlandmarks;
        YA_DynI land_ind;
        intersect(landmarks,manifold,newlandmarks,land_ind);
        if (newlandmarks.numel()==1) {
          ya_sizet mN=manifold.numel();
          for (ya_sizet i=0; i<mN; i++)
            not_embedded.push_back(manifold(i));
          continue;
        }
        graph_dists=YA_MatT(_graph_dists(newlandmarks,manifold));
      } else
        graph_dists=YA_MatT(_graph_dists(manifold,manifold));
      //if (i>0)
      //  std::cout << manifold.T() << std::endl;
        //std::cout << "GRAPH: " << i << " " << manifold.T() << std::endl;
    } else {
      graph_dists=_graph_dists;
    }
    
    VM_SymMat MDSmat;
    YA_MatT subB;
    if (landmarks.numel()!=0) {
      subB=transpose(square(graph_dists));
      this->double_center(subB);
      MDSmat=subB.T()*subB;
    } else {
      MDSmat=square(graph_dists);
      this->double_center(MDSmat);
    }
  
    YA_VecT eigen_temp;
    if (MDSmat.rows()<dim) {
      std::ostringstream owarn;
      owarn << "Number of samples on connected manifold ("
            << MDSmat.rows() << ") is less than the "
            << "dimensionality (" << dim << "). Zeroing extra dimensions.";
      ya_addwarn(306,5,"IsoMap",owarn.str());
      err_flag = 306;
    
      eigopts.dim(MDSmat.rows());
      eigs(MDSmat,eigen_vectors,eigen_temp,eigopts);
      eigen_vectors=YA_MatT(concat(eigen_vectors,
                            YA_MatT(vmzeros(MDSmat.rows(),dim-MDSmat.rows()))));
      eigen_temp=YA_VecT(concat(eigen_temp,";",
                                YA_VecT(vmzeros(dim-MDSmat.rows(),1))));
    } else {
      eigopts.dim(dim);
      eigs(MDSmat,eigen_vectors,eigen_temp,eigopts);
    }
  
    if (landmarks.numel()!=0) {
      eigen_temp(::find(eigen_temp<=0.0))=1.0;
      eigen_temp=sqrt(eigen_temp);
      eigen_vectors=YA_MatT(subB*eigen_vectors*diag(1.0/eigen_temp));
    } else
      eigen_temp(::find(eigen_temp<0.0))=0.0;

    // Calculate the projection
    if (manifold.numel()!=0) {
      output(manifold,":")=eigen_vectors*diag(sqrt(eigen_temp));
      if (i==0)
        old_range=maximum(output(manifold,":"))-minimum(output(manifold,":"));
      else {
        old_center+=old_range;
        old_range=maximum(output(manifold,":"))-minimum(output(manifold,":"));
        old_center+=old_range;
        output(manifold,0)+=old_center;
        old_center+=old_range/2.0;
      }
      if (old_range==0.0)
        old_range=5.0;
    } else
      output=eigen_vectors*diag(sqrt(eigen_temp));
    if (i==0)
      this->eigen_values=eigen_temp*static_cast<eltype>(isizes(i));
    else
      this->eigen_values+=eigen_temp*static_cast<eltype>(isizes(i));
  
    #ifdef YA_PROGMETER
    if (this->_verbose) pm.iterate(static_cast<unsigned>(itercounts(i)));
    #endif
  }
  
  if (not_embedded.numel()!=0) {
    std::ostringstream owarn;
    owarn << "Found " << not_embedded.numel() << " samples connected to 1 or "
          << "fewer landmarks. Embedding as non-overlapping single points.";
    ya_addwarn(306,5,"IsoMap",owarn.str());
    err_flag = 306;
    
    if (this->eigen_values.numel()!=dim) {
      this->eigen_values=vmzeros(this->eigen_values.numel());
      old_range=5.0;
    }
    
    output(not_embedded,":")=vmzeros(not_embedded.numel(),dim);
    output(not_embedded,0)=vmlinspace(old_center,
                                      old_range*(not_embedded.numel()-1)+
                                      old_center,
                                      not_embedded.numel()).T();
  }

  this->eigen_values/=static_cast<eltype>(m.rows());
  
  #ifdef YA_PROGMETER
  if (this->_verbose) {
    pm.finish();
    tk.end();
    cerr << "Done. " << tk << ".\n";
  }
  #endif

  this->input_coords=m;
  this->output_coords=output;
  return err_flag;
}

template <class eltype>
template <class ya_type1>
void YAIsoReduce<eltype>::component_stat(ya_type1 &m, 
                                         const ya_sizet num_neighbors,
                                         YA_DynI &stats) {
  YA_VecI landmarks;
  component_stat(m,num_neighbors,stats,landmarks);
}                                             

template <class eltype>
template <class ya_type1, class ya_type2>
void YAIsoReduce<eltype>::component_stat(ya_type1 &m, 
                                           const ya_sizet num_neighbors,
                                           YA_DynI &stats,
                                           const ya_type2 &landmarks) {
  ListGraph graph(m.rows());
  if (this->_neighbor_mode == 0){
    // Find k-nearest neighbors
    YA_MatI neighbors;
    YA_MatD dists;
    kneighbors(m,num_neighbors,neighbors,dists,this->_verbose);
    // Get the graph
    graph.addneighborhood(neighbors,dists);
  } else {
    // Find E-nearest neighbors
    std::vector<YA_DynI> neighbors;
    std::vector<YA_DynT> dists;
    eneighbors(m,this->_epsilon,neighbors,dists,this->_verbose);
    // Get the graph
    graph.addneighborhood(neighbors, dists);
  } 
  
  // Compute pairwise distance matrix from Dijkstra's
  YA_VecI templand;                                             
  if (landmarks.numel()==0) {
    VM_SymMat distt;
    graph.all_shortest_paths(distt,this->_verbose);
    _graph_dists=distt;
  } else {
    graph.shortest_paths(landmarks,_graph_dists,this->_verbose);
    templand=landmarks;
  }
  
  YA_RowI tmp,firsts;
  tmp=vm_cast<ya_sizet>::sc(min(_graph_dists==BIG_DOUBLE,firsts));
  YA_DynI comps=unique(firsts);
  ya_sizet num_comp=comps.numel();
  YA_RowI isizes=sum(transpose(rowrep(firsts,num_comp)==
                               (comps*
                                vm_cast<ya_sizet>::sc(vmones(1,m.rows())))));
  if (landmarks.numel()==0) {
    stats=transpose(sort(isizes));
    ip_flipcm(stats);
    return;
  }
    
  stats.clear();
  ya_sizet cN=comps.numel();
  for (ya_sizet i=0; i<cN; i++) {
    YA_DynI manifold=::find(firsts==comps(i));
    manifold=setdiff(manifold,::find(min(is_inf(_graph_dists))));
    if (manifold.numel()>0)
      stats.push_back(manifold.numel());
  }
  stats=sort(stats);
  ip_flipcm(stats);
}

template <class eltype>
template <class ya_type1, class ya_type2>
void YAIsoReduce<eltype>::forward_map(const ya_type1 &input, 
                                        ya_type2 &output) const {
  this->nbors_forward(input,output);
}

template <class eltype>
template <class ya_type1, class ya_type2>
void YAIsoReduce<eltype>::reverse_map(const ya_type1 &input, 
                                        ya_type2 &output) const {
  this->nbors_reverse(input,output);
}

template <class eltype>
void YAIsoReduce<eltype>::save_map(std::ostream &out) const {
  ya_sizet precision=out.precision();
  out.precision(16);
  this->write_header(out);
  write(out,this->input_coords);
  write(out,this->output_coords);
  out.precision(precision);
}
  
template <class eltype>
int YAIsoReduce<eltype>::load_map(std::istream &in) {
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
