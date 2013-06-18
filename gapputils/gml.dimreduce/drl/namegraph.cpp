/***************************************************************************
                                namegraph.cpp
                             -------------------
                               W. Michael Brown

  Base Graph Container

 __________________________________________________________________________
    This file is part of the YALA++ Library
 __________________________________________________________________________

    begin      : Tue Oct 24 2006
    authors    : W. Michael Brown
    email      : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.
   ----------------------------------------------------------------------- */

#ifndef NAMEGRAPH_CPP
#define NAMEGRAPH_CPP

//template class NameGraph<ListNode>;
//template class NameGraph<MapNode>;

using namespace yala;

// Initialize
template<class ntype>
NameGraph<ntype>::NameGraph() {
}

// Graph with a specified number of nodes
template<class ntype>
NameGraph<ntype>::NameGraph(const ya_sizet numnodes) {
  graph.assign(numnodes,ntype());
}

// Graph with a specified number of nodes, each with a reserve for num_edges
template<class ntype>
NameGraph<ntype>::NameGraph(const ya_sizet numnodes, const ya_sizet numedges) {
  ntype empty;
  graph.assign(numnodes,empty);
}

template<class ntype>
NameGraph<ntype>::~NameGraph() {
}

// Access a node
template<class ntype>
ntype & NameGraph<ntype>::operator[](const ya_sizet nodenum) {
  YA_DEBUG_ERROR(nodenum>=0 && nodenum<size(),"Node is bigger than graph");
  return graph[nodenum];
}

// Add an edge
template<class ntype>
void NameGraph<ntype>::add_edge(const ya_sizet node1, 
                                const ya_sizet node2, const double weight) {
  YA_DEBUG_ERROR(node1>=0 && node2>=0 && node1<size() && node2<size(),
                 "Node is bigger than graph");
  graph[node1].add_edge(node2,weight);
  graph[node2].add_edge(node1,weight);
}

// Add a k-nearest neighbors matrix of edges and weights equal to distances
template<class ntype>
void NameGraph<ntype>::addneighborhood(const YA_MatI &neighbors,
                                       const YA_MatD &dists) {
  YA_DEBUG_ERROR(neighbors.rows()==dists.rows() &&
                 neighbors.cols()==dists.cols(),
                 "Size of distance matrix is different than neighbor matrix");
  ya_sizet nR=neighbors.rows();
  ya_sizet nC=neighbors.cols();
  for (ya_sizet i=0; i<nR; i++)
    for (ya_sizet j=0; j<nC; j++)
      add_edge(i,neighbors(i,j),dists(i,j));
}      

// Add a epsilon-nearest neighbors matrix of edges and weights (distances)
template<class ntype>
void NameGraph<ntype>::addneighborhood(const std::vector<YA_DynI> &neighbors,
                                       const std::vector<YA_DynD> &dists) {
  size_t nS=neighbors.size();
  for (size_t i=0; i<nS; i++) {
    size_t nN=neighbors[i].numel();
    for (size_t j=0; j<nN; j++)
      add_edge(i,neighbors[i][j],dists[i][j]);
  }
}      

// Calculate shortest path from start to all other nodes
template<class ntype>
void NameGraph<ntype>::dijkstra(const ya_sizet start, std::vector<double> &dist) {
  YA_DEBUG_ERROR(start>=0 && start<size(),
                 "Start node is bigger than graph.");

  std::multimap<double,ya_sizet> Q;
  Q.insert(std::pair<double,ya_sizet>(0,start));
  dist.assign(size(),BIG_DOUBLE);
  dist[start]=0;

  while (!Q.empty()) {
    ya_sizet u=Q.begin()->second;
    Q.erase(Q.begin());
    for (typename ntype::edge_pointer i=graph[u].begin();
         i!=graph[u].end(); i++) {
      ya_sizet v=graph[u].node(i);
      if (dist[v]>dist[u]+graph[u].weight(i)) {
        dist[v]=dist[u]+graph[u].weight(i);
        Q.insert(std::pair<double,ya_sizet>(dist[v],v));
      }
    }
  }
}

// Calculate a pairwise distance matrix based on shortest paths
template<class ntype>
template<class vm_type,class ya_type2>
void NameGraph<ntype>::shortest_paths(const vm_type &source_nodes, 
                                      ya_type2 &pair_dists, const int verbose) {
  #ifdef YA_PROGMETER
  YA_TimeKeeper tk;
  YA_ProgMeter pm;
  if (verbose) {
    tk.start();
    pm.start("Calculating shortest paths:",70,source_nodes.numel(),false);
  }
  #endif

  pair_dists.setup(source_nodes.numel(),size());
  ya_sizet sN=source_nodes.numel();
  for (ya_sizet k=0; k<sN; k++) {
    YA_DEBUG_ERROR(source_nodes[k]>=0 && source_nodes[k]<size(),
                   "Source node bigger than graph.");
    std::multimap<double,ya_sizet> Q;
    Q.insert(std::pair<double,ya_sizet>(0,source_nodes[k]));
    setall(pair_dists(k,":"),BIG_DOUBLE);
    pair_dists(k,source_nodes[k])=0;

    while (!Q.empty()) {
      ya_sizet u=Q.begin()->second;
      Q.erase(Q.begin());
      for (typename ntype::edge_pointer i=graph[u].begin();
           i!=graph[u].end(); i++) {
        ya_sizet v=graph[u].node(i);
        double newdist=pair_dists(k,u)+graph[u].weight(i);
        if (pair_dists(k,v)>newdist) {
          pair_dists(k,v)=newdist;
          Q.insert(std::pair<double,ya_sizet>(newdist,v));
        }
      }
    }
    #ifdef YA_PROGMETER
    if (verbose) pm.iterate();
    #endif
  }
  #ifdef YA_PROGMETER
  if (verbose) {
    pm.finish();
    tk.end();
    cerr << "Done. " << tk << ".\n";
  }
  #endif
}

// Calculate a pairwise distance matrix based on shortest paths
template<class ntype>
void NameGraph<ntype>::all_shortest_paths(VM_SymMatD &pair_dists_sym,
                                          const int verbose) {
  #ifdef YA_PROGMETER
  YA_TimeKeeper tk;
  YA_ProgMeter pm;
  if (verbose) {
    tk.start();
    pm.start("Calculating shortest paths:",70,size(),false);
  }
  #endif

  YA_MatD pair_dists(size(),size());

  ya_sizet gsize=size();
  #pragma omp parallel
  {
  #pragma omp for
  for (ya_sizet k=0; k<gsize; k++) {
    std::multimap<double,ya_sizet> Q;
    Q.insert(std::pair<double,ya_sizet>(0,k));
    setall(pair_dists(k,":"),BIG_DOUBLE);
    pair_dists(k,k)=0;

    while (!Q.empty()) {
      ya_sizet u=Q.begin()->second;
      Q.erase(Q.begin());
      for (typename ntype::edge_pointer i=graph[u].begin();
           i!=graph[u].end(); i++) {
        ya_sizet v=graph[u].node(i);
        double newdist=pair_dists(k,u)+graph[u].weight(i);
        if (pair_dists(k,v)>newdist) {
          pair_dists(k,v)=newdist;
          Q.insert(std::pair<double,ya_sizet>(newdist,v));
        }
      }
    }
    #ifdef YA_PROGMETER
    if (verbose) pm.iterate();
    #endif
  }
  }
  #ifdef YA_PROGMETER
  if (verbose) {
    pm.finish();
    tk.end();
    cerr << "Done. " << tk << ".\n";
  }
  #endif
  pair_dists_sym=pair_dists;
}

// Write out the edges to a file
template<class ntype>
void NameGraph<ntype>::writeedges(const std::string &filename) {
  std::ofstream out;
  ya_fileopen(out,filename);
  for (ya_sizet i=0; i<size(); i++)
    for (typename ntype::edge_pointer j=graph[i].begin(); j!=graph[i].end();
         j++)
      if (i<graph[i].node(j))
        out << i << "\t" << graph[i].node(j) << "\t" << graph[i].weight(j) << std::endl;
  ya_fileclose(out);
}

#endif
